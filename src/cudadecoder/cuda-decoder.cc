// cudadecoder/cuda-decoder.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if HAVE_CUDA == 1

#include "cuda-decoder.h"
#include "cuda-decoder-kernels.h"

#include <algorithm>
#include <cfloat>
#include <map>
#include <tuple>
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>

namespace kaldi {
namespace cuda_decoder {
CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config,
                         int32 nlanes, int32 nchannels)
    : fst_(fst),
      nlanes_(nlanes),
      nchannels_(nchannels),
      default_beam_(config.default_beam),
      lattice_beam_(config.lattice_beam),
      ntokens_pre_allocated_(config.ntokens_pre_allocated),
      max_active_(config.max_active),
      aux_q_capacity_(config.aux_q_capacity),
      main_q_capacity_(config.main_q_capacity),
      extra_cost_min_delta_(0.0f) {
  // Static asserts on constants
  CheckStaticAsserts();
  // Runtime asserts
  KALDI_ASSERT(nlanes > 0);
  KALDI_ASSERT(nchannels > 0);
  KALDI_ASSERT(nlanes_ <= KALDI_CUDA_DECODER_MAX_N_LANES);
  KALDI_ASSERT(nlanes_ <= nchannels_);
  // All GPU work in decoder will be sent to compute_st_
  cudaStreamCreate(&compute_st_);
  // For all the allocating/initializing process
  // We create a special channel
  // containing the exact state a channel should have when starting a new decode
  // It contains fst.Start(), the non-emitting tokens created by fst.Start(),
  // and all the data used by the decoder.
  // When calling InitDecoding() on a new channel, we simply clone this special
  // channel into that new channel
  ++nchannels_;                       // adding the special initial channel
  init_channel_id_ = nchannels_ - 1;  // Using last one as init_channel_params
  AllocateHostData();
  AllocateDeviceData();
  AllocateDeviceKernelParams();

  InitDeviceParams();
  InitHostData();
  InitDeviceData();

  ComputeInitialChannel();
  --nchannels_;  // removing the special initial channel from the count

  // Making sure that everything is ready to use
  cudaStreamSynchronize(compute_st_);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void CudaDecoder::AllocateDeviceData() {
  hashmap_capacity_ =
      KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR * main_q_capacity_;
  d_channels_counters_.Resize(nchannels_, 1);
  d_lanes_counters_.Resize(nlanes_, 1);
  d_main_q_state_and_cost_.Resize(nchannels_, main_q_capacity_);
  d_main_q_info_.Resize(nlanes_, main_q_capacity_);
  d_aux_q_state_and_cost_.Resize(nlanes_, aux_q_capacity_);
  d_aux_q_info_.Resize(nlanes_, aux_q_capacity_);
  d_main_q_degrees_prefix_sum_.Resize(nchannels_, main_q_capacity_);
  d_histograms_.Resize(nlanes_, KALDI_CUDA_DECODER_HISTO_NBINS);
  d_main_q_extra_prev_tokens_prefix_sum_.Resize(nlanes_, main_q_capacity_);
  d_main_q_n_extra_prev_tokens_local_idx_.Resize(nlanes_, main_q_capacity_);

  d_main_q_state_hash_idx_.Resize(nlanes_, main_q_capacity_);
  d_main_q_extra_prev_tokens_.Resize(nlanes_, main_q_capacity_);
  d_main_q_extra_and_acoustic_cost_.Resize(nlanes_, main_q_capacity_);
  d_main_q_block_sums_prefix_sum_.Resize(
      nlanes_, KALDI_CUDA_DECODER_DIV_ROUND_UP(main_q_capacity_,
                                               KALDI_CUDA_DECODER_1D_BLOCK) +
                   1);
  d_main_q_arc_offsets_.Resize(nchannels_, main_q_capacity_);
  d_hashmap_values_.Resize(nlanes_, hashmap_capacity_);
  d_main_q_acoustic_cost_.Resize(nlanes_, main_q_capacity_);
  d_aux_q_acoustic_cost_.Resize(nlanes_, aux_q_capacity_);
  d_extra_and_acoustic_cost_concat_matrix.Resize(nlanes_, main_q_capacity_);
  // Reusing data from aux_q. Those two are never used at the same time
  // d_list_final_tokens_in_main_q_ is used in GetBestPath.
  // the aux_q is used in AdvanceDecoding
  d_list_final_tokens_in_main_q_ = d_aux_q_state_and_cost_.GetView();
  d_extra_and_acoustic_cost_concat__ =
      d_extra_and_acoustic_cost_concat_matrix.lane(0);
  d_acoustic_cost_concat_ = d_aux_q_acoustic_cost_.lane(0);
  d_infotoken_concat_ = d_aux_q_info_.lane(0);
}

void CudaDecoder::AllocateHostData() {
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaMallocHost(&h_extra_and_acoustic_cost_concat__,
                     nlanes_ * main_q_capacity_ *
                         sizeof(*h_extra_and_acoustic_cost_concat__)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_acoustic_cost_concat_,
      nlanes_ * main_q_capacity_ * sizeof(*h_acoustic_cost_concat_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_extra_prev_tokens_concat_,
      nlanes_ * main_q_capacity_ * sizeof(*h_extra_prev_tokens_concat_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_infotoken_concat_,
      nlanes_ * main_q_capacity_ * sizeof(*h_infotoken_concat_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaMallocHost(&h_lanes_counters_, nlanes_ * sizeof(*h_lanes_counters_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_channels_counters_, nchannels_ * sizeof(*h_channels_counters_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_list_final_tokens_in_main_q_,
      main_q_capacity_ * sizeof(*h_list_final_tokens_in_main_q_)));

  h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_.resize(nchannels_);
  h_all_tokens_acoustic_cost_.resize(nchannels_);
  h_all_tokens_extra_prev_tokens_.resize(nchannels_);
  h_all_tokens_info_.resize(nchannels_);
  for (int32 ichannel = 0; ichannel < nchannels_; ++ichannel) {
    h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel].reserve(
        ntokens_pre_allocated_);
    h_all_tokens_acoustic_cost_[ichannel].reserve(ntokens_pre_allocated_);
    h_all_tokens_info_[ichannel].reserve(ntokens_pre_allocated_);
  }
  h_main_q_end_lane_offsets_.resize(nlanes_ + 1);
  h_emitting_main_q_end_lane_offsets_.resize(nlanes_ + 1);
  h_n_extra_prev_tokens_lane_offsets_.resize(nlanes_ + 1);
  frame_offsets_.resize(nchannels_);
  num_frames_decoded_.resize(nchannels_, -1);
  main_q_emitting_end_.resize(nlanes_);
}

void CudaDecoder::InitDeviceData() {
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemsetAsync(
      d_histograms_.lane(0), 0,
      sizeof(int32) * KALDI_CUDA_DECODER_HISTO_NBINS * nlanes_, compute_st_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemsetAsync(
      d_channels_counters_.MutableData(), 0,
      nchannels_ * sizeof(*d_channels_counters_.MutableData()), compute_st_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemsetAsync(
      d_lanes_counters_.MutableData(), 0,
      nlanes_ * sizeof(*d_lanes_counters_.MutableData()), compute_st_));
  InitHashmapKernel(KaldiCudaDecoderNumBlocks(hashmap_capacity_, nlanes_),
                    KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                    *h_device_params_);
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void CudaDecoder::InitHostData() {
  // Adding a tolerance on max_active_
  // This is because we will usually not be able to limit the number of tokens
  // to exactly max_active
  // We will set it as close as possible to max_active, and we don't want to
  // keep calling the histograms kernels for a few tokens above the limit
  int32 tolerance = max_active_ * KALDI_CUDA_DECODER_MAX_ACTIVE_TOLERANCE;
  // Checking for overflow
  int32 overflow_limit = INT_MAX - tolerance;
  max_active_thresh_ =
      (max_active_ < overflow_limit) ? (max_active_ + tolerance) : INT_MAX;
}

void CudaDecoder::AllocateDeviceKernelParams() {
  h_device_params_ = new DeviceParams();
  h_kernel_params_ = new KernelParams();
}

void CudaDecoder::InitDeviceParams() {
  // Setting Kernel Params
  // Sent to cuda kernels by copy
  // Making sure we'll be able to send it to the kernels
  KALDI_ASSERT((sizeof(KernelParams) + sizeof(DeviceParams)) <
               KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE);

  h_device_params_->d_channels_counters = d_channels_counters_.GetView();
  h_device_params_->d_lanes_counters = d_lanes_counters_.GetView();
  h_device_params_->d_main_q_state_and_cost =
      d_main_q_state_and_cost_.GetView();
  h_device_params_->d_main_q_info = d_main_q_info_.GetView();
  h_device_params_->d_aux_q_state_and_cost = d_aux_q_state_and_cost_.GetView();
  h_device_params_->d_main_q_extra_and_acoustic_cost =
      d_main_q_extra_and_acoustic_cost_.GetView();
  h_device_params_->d_main_q_acoustic_cost = d_main_q_acoustic_cost_.GetView();
  h_device_params_->d_aux_q_acoustic_cost = d_aux_q_acoustic_cost_.GetView();
  h_device_params_->d_aux_q_info = d_aux_q_info_.GetView();
  h_device_params_->d_main_q_degrees_prefix_sum =
      d_main_q_degrees_prefix_sum_.GetView();
  h_device_params_->d_main_q_block_sums_prefix_sum =
      d_main_q_block_sums_prefix_sum_.GetView();
  h_device_params_->d_main_q_state_hash_idx =
      d_main_q_state_hash_idx_.GetView();
  h_device_params_->d_main_q_extra_prev_tokens_prefix_sum =
      d_main_q_extra_prev_tokens_prefix_sum_.GetView();
  h_device_params_->d_main_q_n_extra_prev_tokens_local_idx =
      d_main_q_n_extra_prev_tokens_local_idx_.GetView();
  h_device_params_->d_main_q_extra_prev_tokens =
      d_main_q_extra_prev_tokens_.GetView();
  h_device_params_->d_main_q_arc_offsets = d_main_q_arc_offsets_.GetView();
  h_device_params_->d_hashmap_values = d_hashmap_values_.GetView();
  h_device_params_->d_histograms = d_histograms_.GetView();
  h_device_params_->d_arc_e_offsets = fst_.d_e_offsets_;
  h_device_params_->d_arc_ne_offsets = fst_.d_ne_offsets_;
  h_device_params_->d_arc_pdf_ilabels = fst_.d_arc_pdf_ilabels_;
  h_device_params_->d_arc_weights = fst_.d_arc_weights_;
  h_device_params_->d_arc_nextstates = fst_.d_arc_nextstates_;
  h_device_params_->d_fst_final_costs = fst_.d_final_;
  h_device_params_->default_beam = default_beam_;
  h_device_params_->lattice_beam = lattice_beam_;
  h_device_params_->main_q_capacity = main_q_capacity_;
  h_device_params_->aux_q_capacity = aux_q_capacity_;
  h_device_params_->init_channel_id = init_channel_id_;
  h_device_params_->max_nlanes = nlanes_;
  h_device_params_->nstates = fst_.num_states_;
  h_device_params_->init_state = fst_.Start();
  KALDI_ASSERT(h_device_params_->init_state != fst::kNoStateId);
  h_device_params_->init_cost = StdWeight::One().Value();
  h_device_params_->hashmap_capacity = hashmap_capacity_;
  h_device_params_->max_active = max_active_;
  // For the first static_beam_q_length elements of the queue, we will keep the
  // beam static
  int32 static_beam_q_length =
      aux_q_capacity_ / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT;
  // For the last adaptive_beam_q_length elements of the queue, we will decrease
  // the beam, segment by segment
  // For more information, please refer to the definition of GetAdaptiveBeam in
  // cuda-decoder-kernels.cu
  int32 adaptive_beam_q_length = (aux_q_capacity_ - static_beam_q_length);
  int32 adaptive_beam_bin_width =
      adaptive_beam_q_length / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NSTEPS;
  h_device_params_->adaptive_beam_static_segment = static_beam_q_length;
  h_device_params_->adaptive_beam_bin_width = adaptive_beam_bin_width;

  // Reusing aux_q memory to list final states in GetLattice
  // Those cannot be used at the same time
  h_device_params_->d_list_final_tokens_in_main_q =
      d_list_final_tokens_in_main_q_;
}

CudaDecoder::~CudaDecoder() {
  cudaStreamDestroy(compute_st_);

  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_lanes_counters_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_channels_counters_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaFreeHost(h_extra_and_acoustic_cost_concat__));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_acoustic_cost_concat_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_extra_prev_tokens_concat_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_infotoken_concat_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaFreeHost(h_list_final_tokens_in_main_q_));
  // Will call the cudaFrees inside destructors
  delete h_kernel_params_;
  delete h_device_params_;
}

void CudaDecoder::ComputeInitialChannel() {
  KALDI_ASSERT(nlanes_ > 0);
  const int32 ilane = 0;
  KALDI_ASSERT(ilane == 0);
  // Following kernels working channel_id
  std::vector<ChannelId> channels = {init_channel_id_};
  SetChannelsInKernelParams(channels);  // not calling LoadChannelsStateToLanes,
                                        // init_channel_id_ is a special case

  // Adding the start state to the initial token queue
  InitializeInitialLaneKernel(KaldiCudaDecoderNumBlocks(1, 1),
                              KALDI_CUDA_DECODER_ONE_THREAD_BLOCK, compute_st_,
                              *h_device_params_);

  h_lanes_counters_[ilane].post_expand_aux_q_end = 1;

  PruneAndPreprocess();
  FinalizeProcessNonEmittingKernel(
      KaldiCudaDecoderNumBlocks(1, 1), KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);

  CopyLaneCountersToHostSync();
  PostProcessingMainQueue();
  CopyLaneCountersToHostSync();

  const int32 main_q_end = h_lanes_counters_[0].main_q_narcs_and_end.y;
  KALDI_ASSERT(main_q_end > 0);
  // All arcs traversed until now are non-emitting
  h_all_tokens_acoustic_cost_[init_channel_id_].resize(main_q_end, 0.0f);

  // Moving all data linked to init_channel_id_ to host
  // that data will be cloned to other channels when calling InitDecoding
  CopyMainQueueDataToHost();
  SaveChannelsStateFromLanes();

  KALDI_ASSERT(
      h_channels_counters_[init_channel_id_].prev_main_q_narcs_and_end.x > 0);
  KALDI_ASSERT(
      h_channels_counters_[init_channel_id_].prev_main_q_narcs_and_end.y > 0);
}

void CudaDecoder::InitDecoding(const std::vector<ChannelId> &channels) {
  // Cloning the init_channel_id_ channel into all channels in the channels vec
  const int nlanes_used = channels.size();
  // Getting *h_kernel_params ready to use
  SetChannelsInKernelParams(channels);

  // Size of the initial main_q
  const int32 init_main_q_size =
      h_channels_counters_[init_channel_id_].prev_main_q_narcs_and_end.y;

  KALDI_ASSERT(init_main_q_size > 0);
  // Getting the channels ready to compute new utterances
  InitDecodingOnDeviceKernel(
      KaldiCudaDecoderNumBlocks(init_main_q_size, nlanes_used),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  cudaStreamSynchronize(compute_st_);
  KALDI_DECODER_CUDA_CHECK_ERROR();
  for (ChannelId ichannel : channels) {
    // Tokens from initial main_q needed on host
    // Deep copy
    h_all_tokens_info_[ichannel] = h_all_tokens_info_[init_channel_id_];
    h_all_tokens_acoustic_cost_[ichannel] =
        h_all_tokens_acoustic_cost_[init_channel_id_];
    h_all_tokens_extra_prev_tokens_[ichannel] =
        h_all_tokens_extra_prev_tokens_[init_channel_id_];
    h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel] =
        h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_
            [init_channel_id_];

    int32 n_initial_tokens = h_all_tokens_info_[init_channel_id_].size();

    h_channels_counters_[ichannel] = h_channels_counters_[init_channel_id_];
    num_frames_decoded_[ichannel] = 0;
    frame_offsets_[ichannel].clear();
    frame_offsets_[ichannel].push_back(n_initial_tokens);
  }
}

void CudaDecoder::LoadChannelsStateToLanes(
    const std::vector<ChannelId> &channels) {
  // Setting that channels configuration in kernel_params
  SetChannelsInKernelParams(channels);
  KALDI_ASSERT(nlanes_used_ > 0);
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
    h_lanes_counters_[ilane].main_q_narcs_and_end =
        h_channels_counters_[ichannel].prev_main_q_narcs_and_end;
  }
  LoadChannelsStateInLanesKernel(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                                 KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
                                 compute_st_, *h_device_params_,
                                 *h_kernel_params_);
}

void CudaDecoder::SaveChannelsStateFromLanes() {
  KALDI_ASSERT(nlanes_used_ > 0);
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
    h_channels_counters_[ichannel].prev_main_q_narcs_and_end =
        h_lanes_counters_[ilane].main_q_narcs_and_end;
    h_channels_counters_[ichannel].prev_main_q_global_offset =
        h_lanes_counters_[ilane].main_q_global_offset;
  }
  SaveChannelsStateFromLanesKernel(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                                   KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
                                   compute_st_, *h_device_params_,
                                   *h_kernel_params_);
  ResetChannelsInKernelParams();
}

int32 CudaDecoder::GetMaxForAllLanes(
    std::function<int32(const LaneCounters &)> func) {
  int32 max_val = 0;
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    const int32 val = func(h_lanes_counters_[ilane]);
    max_val = std::max(max_val, val);
  }
  return max_val;
}

void CudaDecoder::CopyLaneCountersToHostAsync() {
  cudaMemcpyAsync(h_lanes_counters_, d_lanes_counters_.MutableData(),
                  nlanes_used_ * sizeof(*h_lanes_counters_),
                  cudaMemcpyDeviceToHost, compute_st_);
}

void CudaDecoder::CopyLaneCountersToHostSync() {
  CopyLaneCountersToHostAsync();
  cudaStreamSynchronize(compute_st_);
}

template <typename T>
void CudaDecoder::PerformConcatenatedCopy(
    std::function<int32(const LaneCounters &)> func, LaneMatrixView<T> src,
    T *d_concat, T *h_concat, cudaStream_t st,
    std::vector<int32> *lanes_offsets_ptr) {
  // Computing the lane offsets
  // Saving them into *lanes_offsets_ptr and
  // h_kernel_params_->main_q_end_lane_offsets
  int32 lane_offset = 0;
  int32 max_val = 0;
  std::vector<int32> &lanes_offsets = *lanes_offsets_ptr;
  KALDI_ASSERT(lanes_offsets.size() >= (nlanes_used_ + 1));
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    const int32 val = func(h_lanes_counters_[ilane]);
    max_val = std::max(max_val, val);
    lanes_offsets[ilane] = lane_offset;
    h_kernel_params_->main_q_end_lane_offsets[ilane] = lane_offset;
    lane_offset += val;
  }
  lanes_offsets[nlanes_used_] = lane_offset;
  h_kernel_params_->main_q_end_lane_offsets[nlanes_used_] = lane_offset;
  int32 sum_val = lane_offset;
  if (sum_val == 0) return;  // nothing to do

  // Concatenating lanes data into a single continuous array,
  // stored into d_concat
  ConcatenateLanesDataKernel<T>(
      KaldiCudaDecoderNumBlocks(max_val, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, st, *h_device_params_, *h_kernel_params_,
      src, d_concat);

  // Moving the d_concat to h_concat (host), async
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(
      h_concat, d_concat, sum_val * sizeof(T), cudaMemcpyDeviceToHost, st));
}

// One sync has to happen between PerformConcatenatedCopy and
// MoveConcatenatedCopyToVector
template <typename T>
void CudaDecoder::MoveConcatenatedCopyToVector(
    const std::vector<int32> &lanes_offsets, T *h_concat,
    std::vector<std::vector<T>> *vecvec) {
  // Unpacking the concatenated vector into individual channel storage
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    int32 beg = lanes_offsets[ilane];
    int32 end = lanes_offsets[ilane + 1];
    ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
    auto &vec = (*vecvec)[ichannel];
    vec.insert(vec.end(), h_concat + beg, h_concat + end);
  }
}

void CudaDecoder::ApplyMaxActiveAndReduceBeam(enum QUEUE_ID queue_id) {
  // If at least one lane queue is bigger than max_active,
  // we'll apply a topk on that queue (k=max_active_)
  auto func_aux_q_end = [](const LaneCounters &c) {
    return c.post_expand_aux_q_end;
  };
  auto func_main_q_end = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.y;
  };
  int32 max_q_end = (queue_id == AUX_Q) ? GetMaxForAllLanes(func_aux_q_end)
                                        : GetMaxForAllLanes(func_main_q_end);

  if (max_q_end <= max_active_thresh_) {
    // The queues are already smaller than max_active_thresh_
    // nothing to do
    return;
  }

  bool use_aux_q = (queue_id == AUX_Q);
  ComputeCostsHistogramKernel(
      KaldiCudaDecoderNumBlocks(max_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_, use_aux_q);

  UpdateBeamUsingHistogramKernel(
      KaldiCudaDecoderNumBlocks(1, nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_, use_aux_q);
}

int32 CudaDecoder::NumFramesToDecode(
    const std::vector<ChannelId> &channels,
    std::vector<CudaDecodableInterface *> &decodables, int32 max_num_frames) {
  int32 nframes_to_decode = INT_MAX;
  // std::vector<int> debug_ntokens;
  // std::vector<int> debug_narcs;
  for (int32 ilane = 0; ilane < nlanes_used_; ++ilane) {
    const ChannelId ichannel = channels[ilane];
    const int32 num_frames_decoded = num_frames_decoded_[ichannel];
    KALDI_ASSERT(num_frames_decoded >= 0 &&
                 "You must call InitDecoding() before AdvanceDecoding()");
    int32 num_frames_ready = decodables[ilane]->NumFramesReady();
    // num_frames_ready must be >= num_frames_decoded, or else
    // the number of frames ready must have decreased (which doesn't
    // make sense) or the decodable object changed between calls
    // (which isn't allowed).
    KALDI_ASSERT(num_frames_ready >= num_frames_decoded);
    int32 channel_nframes_to_decode = num_frames_ready - num_frames_decoded;
    nframes_to_decode = std::min(nframes_to_decode, channel_nframes_to_decode);
  }
  if (max_num_frames >= 0)
    nframes_to_decode = std::min(nframes_to_decode, max_num_frames);

  return nframes_to_decode;
}

void CudaDecoder::ExpandArcsEmitting() {
  auto func_narcs = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.x;
  };
  int32 max_main_q_narcs = GetMaxForAllLanes(func_narcs);

  KALDI_ASSERT(max_main_q_narcs > 0);
  ExpandArcsKernel<true>(
      KaldiCudaDecoderNumBlocks(max_main_q_narcs, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  // Updating a few counters, like resetting aux_q_end to 0...
  // true is for IS_EMITTING
  PostExpandKernel<true>(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                         KALDI_CUDA_DECODER_ONE_THREAD_BLOCK, compute_st_,
                         *h_device_params_, *h_kernel_params_);
}

void CudaDecoder::ExpandArcsNonEmitting(bool *should_iterate) {
  auto func_main_q_narcs = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.x;
  };
  int32 max_main_q_narcs = GetMaxForAllLanes(func_main_q_narcs);

  // If we have only a few arcs, jumping to the one-CTA per lane
  // persistent version
  bool launch_persistent_version =
      (max_main_q_narcs < KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS);
  // If we cannot launch the persistent version, we will have to iterate on the
  // heavy load kernels
  *should_iterate = !launch_persistent_version;
  if (launch_persistent_version) {
    // Finalizing process non emitting. Takes care of the long tail,
    // the final iterations with a small numbers of arcs. Do the work inside a
    // single CTA (per lane),
    FinalizeProcessNonEmittingKernel(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                                     KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
                                     compute_st_, *h_device_params_,
                                     *h_kernel_params_);

    return;
  }

  // false is for non emitting
  ExpandArcsKernel<false>(
      KaldiCudaDecoderNumBlocks(max_main_q_narcs, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  // false is for non emitting
  PostExpandKernel<false>(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                          KALDI_CUDA_DECODER_ONE_THREAD_BLOCK, compute_st_,
                          *h_device_params_, *h_kernel_params_);
}

void CudaDecoder::PruneAndPreprocess() {
  auto func_aux_q_end = [](const LaneCounters &c) {
    return c.post_expand_aux_q_end;
  };
  int32 max_aux_q_end = GetMaxForAllLanes(func_aux_q_end);

  // having all aux_q_end == 0 is not likely, but possible
  // in a valid workflow
  if (max_aux_q_end > 0) {
    NonEmittingPreprocessAndContractKernel(
        KaldiCudaDecoderNumBlocks(max_aux_q_end, nlanes_used_),
        KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
        *h_kernel_params_);
  }
}

void CudaDecoder::StartCopyAcousticCostsToHostAsync() {
  auto func_main_q_end = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.y;
  };
  PerformConcatenatedCopy(func_main_q_end,
                          h_device_params_->d_main_q_acoustic_cost,
                          d_acoustic_cost_concat_, h_acoustic_cost_concat_,
                          compute_st_, &h_emitting_main_q_end_lane_offsets_);
  for (int32 ilane = 0; ilane < nlanes_used_; ++ilane)
    main_q_emitting_end_[ilane] = func_main_q_end(h_lanes_counters_[ilane]);
}

void CudaDecoder::FinalizeCopyAcousticCostsToHost() {
  MoveConcatenatedCopyToVector(h_emitting_main_q_end_lane_offsets_,
                               h_acoustic_cost_concat_,
                               &h_all_tokens_acoustic_cost_);
}

void CudaDecoder::PostProcessingMainQueue() {
  auto func_main_q_end = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.y;
  };
  int32 max_main_q_end = GetMaxForAllLanes(func_main_q_end);
  KALDI_ASSERT(max_main_q_end > 0);

  ApplyMaxActiveAndReduceBeam(MAIN_Q);

  FillHashmapWithMainQKernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  EmittingPreprocessAndListExtraPrevTokensStep1Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  EmittingPreprocessAndListExtraPrevTokensStep2Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  EmittingPreprocessAndListExtraPrevTokensStep3Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  EmittingPreprocessAndListExtraPrevTokensStep4Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  // We need the infos about number of emitting arcs (for next frame),
  // the number of extra_prev_tokens, etc.
  CopyLaneCountersToHostAsync();

  ClearHashmapKernel(KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
                     KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                     *h_device_params_, *h_kernel_params_);
}

void CudaDecoder::CopyMainQueueDataToHost() {
  auto func_main_q_end = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.y;
  };
  PerformConcatenatedCopy(func_main_q_end, h_device_params_->d_main_q_info,
                          d_infotoken_concat_, h_infotoken_concat_, compute_st_,
                          &h_main_q_end_lane_offsets_);

  // Sync for :
  // - h_infotoken_concat_ copy done
  // - using lane_counters.main_q_n_extra_prev_tokens
  cudaStreamSynchronize(compute_st_);
  CheckOverflow();

  // Starting the extra_prev_tokens copies
  {
    auto func_main_q_n_extra_prev_tokens = [](const LaneCounters &c) {
      return c.main_q_n_extra_prev_tokens;
    };
    PerformConcatenatedCopy(func_main_q_n_extra_prev_tokens,
                            h_device_params_->d_main_q_extra_prev_tokens,
                            d_infotoken_concat_, h_extra_prev_tokens_concat_,
                            compute_st_, &h_n_extra_prev_tokens_lane_offsets_);
    PerformConcatenatedCopy(func_main_q_n_extra_prev_tokens,
                            h_device_params_->d_main_q_extra_and_acoustic_cost,
                            d_extra_and_acoustic_cost_concat__,
                            h_extra_and_acoustic_cost_concat__, compute_st_,
                            &h_n_extra_prev_tokens_lane_offsets_);
  }

  // Moving infotokens to vecs
  MoveConcatenatedCopyToVector(h_main_q_end_lane_offsets_, h_infotoken_concat_,
                               &h_all_tokens_info_);

  // Waiting for the copies
  cudaStreamSynchronize(compute_st_);

  // Moving the extra_prev_tokens to vecs
  MoveConcatenatedCopyToVector(h_n_extra_prev_tokens_lane_offsets_,
                               h_extra_prev_tokens_concat_,
                               &h_all_tokens_extra_prev_tokens_);
  MoveConcatenatedCopyToVector(
      h_n_extra_prev_tokens_lane_offsets_, h_extra_and_acoustic_cost_concat__,
      &h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_);
}

void CudaDecoder::AdvanceDecoding(
    const std::vector<ChannelId> &channels,
    std::vector<CudaDecodableInterface *> &decodables, int32 max_num_frames) {
  nvtxRangePushA("AdvanceDecoding");
  if (channels.size() == 0) return;  // nothing to do
  // Context switch : Loading the channels state in lanes
  LoadChannelsStateToLanes(channels);
  KALDI_ASSERT(nlanes_used_ > 0);

  // We'll decode nframes_to_decode, such as all channels have at least that
  // number
  // of frames available
  int32 nframes_to_decode =
      NumFramesToDecode(channels, decodables, max_num_frames);

  // Looping over the frames that we will compute
  nvtxRangePushA("Decoding");
  for (int32 iframe = 0; iframe < nframes_to_decode; ++iframe) {
    // Loglikelihoods from the acoustic model
    nvtxRangePop();  // Decoding
    // Setting the loglikelihoods pointers for that frame
    for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
      ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
      int32 frame = num_frames_decoded_[ichannel];
      h_kernel_params_->loglikelihoods_ptrs[ilane] =
          decodables[ilane]->GetLogLikelihoodsCudaPointer(frame);
    }
    cudaStreamSynchronize(cudaStreamPerThread);  // Nnet3 sync
    nvtxRangePushA("Decoding");

    // Processing emitting arcs. We've done the preprocess stage at the end of
    // the previous frame
    ExpandArcsEmitting();
    bool first_nonemitting = true;

    // We'll loop until we have a small enough number of non-emitting arcs
    // in the token queue. We'll then break the loop
    while (true) {
      // Moving the lanes_params to host,
      // we want to know the size of the aux queues after ExpandArcsEmitting
      CopyLaneCountersToHostSync();

      // If one of the aux_q contains more than max_active_ tokens,
      // we'll reduce the beam to only keep max_active_ tokens
      ApplyMaxActiveAndReduceBeam(AUX_Q);
      // Prune the aux_q. Apply the latest beam (using the one from
      // ApplyMaxActiveAndReduceBeam if triggered)
      // move the survival tokens to the main queue
      // and do the preprocessing necessary for the next ExpandArcs
      PruneAndPreprocess();

      // We want to know how many tokens were not pruned, and ended up in the
      // main queue
      // Because we've already done the preprocess stage on those tokens, we
      // also know
      // the number of non-emitting arcs out of those tokens
      // Copy for main_q_narcs and main_q_end
      CopyLaneCountersToHostSync();

      // If we're the first iteration after the emitting stage,
      // we need to copy the acoustic costs back to host.
      // We'll concatenate the costs from the different lanes into in a single
      // continuous array.
      if (first_nonemitting) {
        // Async: We'll need a sync before calling
        // FinalizeCopyLaneCountersToHost
        StartCopyAcousticCostsToHostAsync();
        first_nonemitting = false;
      }

      bool should_iterate;
      ExpandArcsNonEmitting(&should_iterate);
      if (!should_iterate) break;
    }
    // We now have our final token main queues for that frame

    // Moving back to host the final (for this frame) values of :
    // - main_q_end
    // - main_q_narcs
    CopyLaneCountersToHostAsync();

    // Sync for :
    // - CopyLaneCountersToHostAsync
    // - StartCopyAcousticCostsToHostAsync
    cudaStreamSynchronize(compute_st_);

    FinalizeCopyAcousticCostsToHost();

    // Post processing the tokens for that frame
    // - do the preprocess necessary for the next emitting expand (will happen
    // with next frame)
    // - if a state S has more than one token associated to it, generate the
    // list of those tokens
    // It allows to backtrack efficiently in GetRawLattice
    // - compute the extra costs
    PostProcessingMainQueue();

    // Moving the data necessary for GetRawLattice/GetBestPath back to host for
    // storage
    CopyMainQueueDataToHost();

    // Few sanity checks + adding acoustic costs for non emitting arcs
    for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
      const ChannelId ichannel = h_kernel_params_->channel_to_compute[ilane];
      KALDI_ASSERT(frame_offsets_[ichannel].back() ==
                   h_lanes_counters_[ilane].main_q_global_offset);
      KALDI_ASSERT(
          h_all_tokens_extra_prev_tokens_[ichannel].size() ==
          (h_lanes_counters_[ilane].main_q_extra_prev_tokens_global_offset +
           h_lanes_counters_[ilane].main_q_n_extra_prev_tokens));
      KALDI_ASSERT(h_all_tokens_extra_prev_tokens_[ichannel].size() ==
                   h_all_tokens_extra_prev_tokens_[ichannel].size());
      // We're done processing that frame
      ++num_frames_decoded_[ichannel];
      const int32 main_q_end = h_lanes_counters_[ilane].main_q_narcs_and_end.y;
      // Saving frame offsets for GetRawLattice
      frame_offsets_[ichannel].push_back(frame_offsets_[ichannel].back() +
                                         main_q_end);

      // Adding 0.0f acoustic_costs for non-emittings
      int32 ntokens_nonemitting = main_q_end - main_q_emitting_end_[ilane];
      auto &vec = h_all_tokens_acoustic_cost_[ichannel];
      vec.insert(vec.end(), ntokens_nonemitting, 0.0f);
      KALDI_ASSERT(vec.size() == h_all_tokens_info_[ichannel].size());
    }
  }
  SaveChannelsStateFromLanes();
  nvtxRangePop();  // Decoding
  nvtxRangePop();  // End AdvanceDecoding
}

void CudaDecoder::CheckOverflow() {
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    LaneCounters *lane_counters = &h_lanes_counters_[ilane];
    bool q_overflow = lane_counters->q_overflow;
    if (q_overflow != OVERFLOW_NONE) {
      // An overflow was prevented in a kernel
      // The algorithm can still go on but quality of the result can be reduced
      // (less tokens were generated)

      if ((q_overflow & OVERFLOW_MAIN_Q) == OVERFLOW_MAIN_Q) {
        // overflowed main_q
        KALDI_WARN
            << "Preventing overflow of main_q. Continuing "
            << "execution but the quality of the output may be decreased. "
            << "To prevent this from happening, please increase the parameter "
               "--main-q-capacity"
            << " and/or decrease --max-active";
      }
      if ((q_overflow & OVERFLOW_AUX_Q) == OVERFLOW_AUX_Q) {
        // overflowed aux_q
        KALDI_WARN
            << "Preventing overflow of aux_q. Continuing "
            << "execution but the quality of the output may be decreased. "
            << "To prevent this from happening, please increase the parameter "
               "--aux-q-capacity"
            << " and/or decrease --beam";
      }

      KALDI_ASSERT(lane_counters->main_q_narcs_and_end.y < main_q_capacity_);
      KALDI_ASSERT(lane_counters->main_q_narcs_and_end.x >= 0);
      KALDI_ASSERT(lane_counters->main_q_narcs_and_end.y >= 0);
      KALDI_ASSERT(lane_counters->post_expand_aux_q_end < aux_q_capacity_);
      KALDI_ASSERT(lane_counters->post_expand_aux_q_end >= 0);
      KALDI_ASSERT(lane_counters->aux_q_end < aux_q_capacity_);
      KALDI_ASSERT(lane_counters->aux_q_end >= 0);
    }
  }
}

// GetBestCost
// returns the minimum cost among all tokens cost in the current frame
// also returns the index of one token with that min cost
//
// Only called at the end of the computation of one audio file
// not optimized
void CudaDecoder::GetBestCost(const std::vector<ChannelId> &channels,
                              bool use_final_costs,
                              std::vector<std::pair<int32, CostType>> *argmins,
                              std::vector<std::vector<std::pair<int, float>>>
                                  *list_finals_token_idx_and_cost,
                              std::vector<bool> *has_reached_final) {
  if (channels.size() == 0) return;
  // Getting the lanes ready to be used with those channels
  LoadChannelsStateToLanes(channels);

  auto func_main_q_end = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.y;
  };
  int32 max_main_q_end = GetMaxForAllLanes(func_main_q_end);

  // Step1 : Finding the best cost in the last token queue, with and without
  // final costs.
  // Also saving the indexes of those min.
  GetBestCostStep1Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_, use_final_costs, StdWeight::Zero().Value());

  // Step2: Now that we now what the minimum cost is, we list all tokens within
  // [min_cost; min_cost+lattice_beam]
  // min_cost takes into account the final costs if use_final_costs is true,
  // AND if a final state is is present in the last token queue
  GetBestCostStep2Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_, use_final_costs, StdWeight::Zero().Value());
  // Moving the min_costs and their arguments to cost.
  // get_best_cost_kernel_step2 also set the number of tokens in [min_cost;
  // min_cost_lattice_beam]
  // moving that number as well
  CopyLaneCountersToHostSync();  // sync copy

  // Resetting the datastructures
  argmins->clear();
  has_reached_final->clear();
  list_finals_token_idx_and_cost->clear();
  // list_finals_token_idx_and_cost is a vector<vector<>>
  // Each channel will have its own list of tokens within [best;
  // best+lattice_beam]
  list_finals_token_idx_and_cost->resize(nlanes_used_);
  for (int32 ilane = 0; ilane < nlanes_used_; ++ilane) {
    int2 minarg = h_lanes_counters_[ilane].min_int_cost_and_arg;
    // Min cost in that channel last token queue
    CostType min_cost = orderedIntToFloatHost(minarg.x);
    // index of that min cost
    int32 arg = minarg.y;
    // Saving both in output
    argmins->push_back({arg, min_cost});
    // Whether or not the last token queue contains at least one token
    // associated with a final FST state
    has_reached_final->push_back(h_lanes_counters_[ilane].has_reached_final);
    // Number of tokens within [min_cost; min_cost+lattice_beam]
    int n_within_lattice_beam = h_lanes_counters_[ilane].n_within_lattice_beam;
    // Loading those tokens
    (*list_finals_token_idx_and_cost)[ilane].resize(n_within_lattice_beam);
    // Copying that list
    cudaMemcpyAsync(
        h_list_final_tokens_in_main_q_,
        d_list_final_tokens_in_main_q_.lane(ilane),
        n_within_lattice_beam * sizeof(*h_list_final_tokens_in_main_q_),
        cudaMemcpyDeviceToHost, compute_st_);
    // Waiting for the copy
    cudaStreamSynchronize(compute_st_);
    // Moving to output + int2float conversion
    for (int i = 0; i < n_within_lattice_beam; ++i) {
      int global_idx = h_list_final_tokens_in_main_q_[i].x;
      float cost_with_final =
          orderedIntToFloatHost(h_list_final_tokens_in_main_q_[i].y);
      (*list_finals_token_idx_and_cost)[ilane][i].first = global_idx;
      (*list_finals_token_idx_and_cost)[ilane][i].second = cost_with_final;
    }
  }
}

void CudaDecoder::GetBestPath(const std::vector<ChannelId> &channels,
                              std::vector<Lattice *> &fst_out_vec,
                              bool use_final_probs) {
  KALDI_ASSERT(channels.size() == fst_out_vec.size());
  nvtxRangePushA("GetBestPath");
  GetBestCost(channels, use_final_probs, &argmins_,
              &list_finals_token_idx_and_cost_, &has_reached_final_);

  std::vector<int32> reversed_path;
  for (int32 ilane = 0; ilane < channels.size(); ++ilane) {
    const ChannelId ichannel = channels[ilane];
    const int32 token_with_best_cost = argmins_[ilane].first;
    const bool isfinal = has_reached_final_[ilane];
    TokenId token_idx = token_with_best_cost;

    // Backtracking
    // Going all the way from the token with best cost
    // to the beginning (StartState)
    reversed_path.clear();

    // The first token was inserted at the beginning of the queue
    // it always has index 0
    // We backtrack until that first token
    while (token_idx != 0) {
      InfoToken token = h_all_tokens_info_[ichannel][token_idx];
      // We want an arc with extra_cost == 0
      int32 arc_idx;
      TokenId prev_token_idx;
      if (token.IsUniqueTokenForStateAndFrame()) {
        // If we have only one, it is an arc with extra_cost == 0
        arc_idx = token.arc_idx;
        prev_token_idx = token.prev_token;
      } else {
        // Using the first arc with extra_cost == 0
        int32 offset, size;
        std::tie(offset, size) = token.GetSameFSTStateTokensList();
        bool found_best = false;
        for (auto i = 0; i < size; ++i) {
          CostType arc_extra_cost =
              h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel]
                                                                     [offset +
                                                                      i].x;
          // Picking one arc on the best path (extra_cost == 0)
          if (arc_extra_cost == 0.0f) {
            InfoToken list_token =
                h_all_tokens_extra_prev_tokens_[ichannel][offset + i];
            arc_idx = list_token.arc_idx;
            prev_token_idx = list_token.prev_token;
            found_best = true;
            break;
          }
        }
        KALDI_ASSERT(found_best);
      }
      reversed_path.push_back(arc_idx);
      token_idx = prev_token_idx;
    }

    Lattice *fst_out = fst_out_vec[ilane];
    fst_out->DeleteStates();
    // Building the output Lattice
    OutputLatticeState curr_state = fst_out->AddState();
    fst_out->SetStart(curr_state);

    for (int32 i = reversed_path.size() - 1; i >= 1; i--) {
      int32 arc_idx = reversed_path[i];

      LatticeArc arc(fst_.h_arc_id_ilabels_[arc_idx],
                     fst_.h_arc_olabels_[arc_idx],
                     LatticeWeight(fst_.h_arc_weights_[arc_idx], 0),
                     fst_.h_arc_nextstate_[arc_idx]);

      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(curr_state, arc);
      curr_state = arc.nextstate;
    }

    // Adding final cost to final state
    if (isfinal && use_final_probs)
      fst_out->SetFinal(
          curr_state,
          LatticeWeight(fst_.h_final_[fst_.h_arc_nextstate_[reversed_path[0]]],
                        0.0));
    else
      fst_out->SetFinal(curr_state, LatticeWeight::One());

    fst::RemoveEpsLocal(fst_out);
  }
  nvtxRangePop();
}

void CudaDecoder::DebugValidateLattice() {
#if 0
	//validate lattice consistency
	for(int frame=0;frame<nframes;frame++) {
		int token_start=frame_offsets_[ichannel][frame];
		int token_end=(frame+1<nframes) ? frame_offsets_[ichannel][frame+1] : total_ntokens;
		int prev_frame_offset=(frame>0) ? frame_offsets_[ichannel][frame-1] : 0;
		int cur_frame_offset=token_start;
		int next_frame_offset=token_end;

		bool found_zero = false;
		//for each token in frame
		for(int i=token_start;i<token_end;i++) {
			if(i==0) continue;  //initial token skip this...
			InfoToken token=h_all_tokens_info_[ichannel][i];
			KALDI_ASSERT(token.prev_token>=0);

			if(token.IsUniqueTokenForStateAndFrame()) {
				//previous token must be lower than the next frame start
				KALDI_ASSERT(token.prev_token<next_frame_offset);
				//previous token must be larger then previous frame start
				KALDI_ASSERT(token.prev_token>=prev_frame_offset);
			} else {
				int32 offset, size;
				std::tie(offset,size) = token.GetNextStateTokensList();
				KALDI_ASSERT(size>0);
				KALDI_ASSERT(offset>=0 && offset<h_all_tokens_extra_prev_tokens_[ichannel].size());
				for(auto j=0; j<size; ++j) {
					KALDI_ASSERT(offset+j<h_all_tokens_extra_prev_tokens_[ichannel].size());
					InfoToken extra_token=h_all_tokens_extra_prev_tokens_[ichannel][offset+j];
					//previous token must be lower than the next frame start
					KALDI_ASSERT(extra_token.prev_token<next_frame_offset);
					//previous token must be larger then previous frame start
					KALDI_ASSERT(extra_token.prev_token>=prev_frame_offset);
				}
			}
		}
	}
#endif
}

CudaDecoder::LatticeStateInternalId CudaDecoder::GetLatticeStateInternalId(
    int32 total_ntokens, TokenId token_idx, InfoToken token) {
  // If we have a unique token for this (frame,fst_state)
  // Then its ID is a unique ID for (frame,fst_state)
  if (token.IsUniqueTokenForStateAndFrame()) return token_idx;

  // If we have multiple tokens for this (frame,fst_state),
  // let's use the "extra_prev_tokens" offset, which is unique for
  // (frame,fst_state) in that case

  // Adding the total_ntokens offset to avoid collisions with the previous
  // case
  return (total_ntokens + token.prev_token);
}

void CudaDecoder::AddFinalTokensToLattice(LaneId ilane, ChannelId ichannel,
                                          Lattice *fst_out) {
  // Total number of tokens for that utterance. Used in
  // GetLatticeStateInternalId
  const int32 total_ntokens = h_all_tokens_info_[ichannel].size();
  // Reading the overall best_cost for that utterance's last frame. Was set by
  // GetBestCost
  const CostType best_cost = argmins_[ilane].second;
  // Iterating through tokens associated with a final state in the last frame
  for (auto &p : list_finals_token_idx_and_cost_[ilane]) {
    // This final token has a final cost of final_token_cost
    CostType final_token_cost = p.second;
    // This token has possibly an extra cost compared to the best
    CostType extra_cost = final_token_cost - best_cost;
    // We only want to keep paths that have a cost within [best;
    // best+lattice_beam]
    if (extra_cost > lattice_beam_) {
      continue;
    }

    const TokenId final_token_idx = p.first;
    InfoToken final_token = h_all_tokens_info_[ichannel][final_token_idx];

    // Internal ID for our lattice_state=(iframe, fst_state)
    LatticeStateInternalId state_internal_id =
        GetLatticeStateInternalId(total_ntokens, final_token_idx, final_token);
    decltype(curr_f_raw_lattice_state_.end()) map_it;
    bool inserted;

    // We need to create the fst_lattice_state linked to our internal id in the
    // lattice if it doesn't already exists
    // Inserts only if the key doesn't exist in the map
    std::tie(map_it, inserted) = curr_f_raw_lattice_state_.insert(
        {state_internal_id, {FLT_MAX, -1, false}});

    // If we've inserted the element, it means that that state didn't exist in
    // the map
    // Because this is a final state, we need to do a bit of extra work to add
    // the final_cost to it
    if (inserted) {
      // We want to figure out which FST state this token is associated to
      // We don't have that info anymore, it wasn't transfered from the GPU
      // We still need it for final tokens, because we need to know which
      // final cost to add in the lattice.
      // To find that original FST state, we need the id of an arc going to
      // that state,
      // then we'll look in the graph and figure out next_state[arc_idx]
      // we just need a valid arc_idx
      int32 arc_idx;
      if (final_token.IsUniqueTokenForStateAndFrame()) {
        // If unique, we can directly use this arc_idx
        arc_idx = final_token.arc_idx;
      } else {
        // If we have multiple tokens associated to that fst state, just pick
        // the first one
        // from the list
        int32 offset, size;
        std::tie(offset, size) = final_token.GetSameFSTStateTokensList();
        InfoToken prev_token =
            h_all_tokens_extra_prev_tokens_[ichannel][offset];
        arc_idx = prev_token.arc_idx;
      }
      // Creating the state associated with our internal id in the lattice
      OutputLatticeState fst_lattice_final_state = fst_out->AddState();
      map_it->second.fst_lattice_state = fst_lattice_final_state;
      q_curr_frame_todo_.push_back({final_token_idx, final_token});

      if (has_reached_final_[ilane]) {
        // If we have reached final states, adding the final cost
        // We now have a valid arc_idx. We can read the FST state
        StateId fst_next_state = fst_.h_arc_nextstate_[arc_idx];

        fst_out->SetFinal(fst_lattice_final_state,
                          LatticeWeight(fst_.h_final_[fst_next_state], 0.0));
      } else {
        fst_out->SetFinal(fst_lattice_final_state, LatticeWeight::One());
      }
    }

    map_it->second.token_extra_cost =
        std::min(map_it->second.token_extra_cost, extra_cost);
  }
}

void CudaDecoder::AddArcToLattice(int32 list_arc_idx,
                                  TokenId list_prev_token_idx,
                                  InfoToken list_prev_token,
                                  int32 curr_frame_offset,
                                  CostType acoustic_cost,
                                  CostType this_arc_prev_token_extra_cost,
                                  LatticeStateInternalId src_state_internal_id,
                                  OutputLatticeState fst_lattice_start,
                                  OutputLatticeState to_fst_lattice_state,
                                  Lattice *fst_out, bool *must_replay_frame) {
  // We will now add this arc to the output lattice
  // We know the destination state of the arc (to_fst_lattice_state)
  // We need to figure out its source
  // And propagate the extra cost from the destination to the source of that arc
  // (we go backward)
  OutputLatticeState from_fst_lattice_state;
  // Having the predecessor in the previous frame
  // <=> that token is associated to an emiting arc
  bool emitting = (list_prev_token_idx < curr_frame_offset);
  // Checking if the source of that arc is the start state (original state at
  // the beginning of the decode)
  if (list_prev_token_idx != 0) {
    // Selecting the right map
    // - emitting arc -> previous frame map
    // - non emitting arc -> same frame map
    auto *extra_cost_map =
        emitting ? &prev_f_raw_lattice_state_ : &curr_f_raw_lattice_state_;
    decltype(extra_cost_map->end()) from_map_it;
    bool inserted;
    // Attempting to insert the state in the map
    std::tie(from_map_it, inserted) =
        extra_cost_map->insert({src_state_internal_id, {FLT_MAX, -1, false}});
    // If it was inserted, its the first time we insert that key in
    // the map
    // we need to put that state in the todo list to be considered
    // next
    if (inserted) {
      auto *todo_list = emitting ? &q_prev_frame_todo_ : &q_curr_frame_todo_;
      todo_list->push_back({list_prev_token_idx, list_prev_token});
      from_map_it->second.fst_lattice_state = fst_out->AddState();
    }

    // Updating the source extra cost using that arc
    // for an arc a->b
    // extra_cost(a) = min(extra_cost(a),
    //		extra_cost(b) + arc_extra_cost(a->b))
    CostType prev_token_extra_cost = from_map_it->second.token_extra_cost;
    if (this_arc_prev_token_extra_cost < prev_token_extra_cost) {
      // We found a new min
      CostType diff = (prev_token_extra_cost - this_arc_prev_token_extra_cost);
      // If the change is large enough,
      // and if the state that we're writing to was already closed,
      // then we need to replay that frame.
      // if the source state is already closed it means we've
      // read its extra_cost value. Now we're writing again to it.
      // We have to do the first read again, to get the updated
      // value
      // that's why we're replaying that frame
      // (between frames everything is in topological order)
      if (diff > extra_cost_min_delta_ && from_map_it->second.is_state_closed) {
        *must_replay_frame = true;
      }
      prev_token_extra_cost = this_arc_prev_token_extra_cost;
      from_map_it->second.token_extra_cost = prev_token_extra_cost;
    }

    // Reading the OutputLatticeState of the source state in the output lattice
    from_fst_lattice_state = from_map_it->second.fst_lattice_state;
  } else {
    from_fst_lattice_state =
        fst_lattice_start;  // we simply link it to the source
  }

  // Checking if it's the first time we insert an arc with that
  // arc_idx for that frame.
  // If we're replaying that frame, we don't want duplicates
  bool is_this_arc_new = f_arc_idx_added_.insert(list_arc_idx).second;
  if (is_this_arc_new) {
    // The following reads will most likely end up in cache misses
    // we could load everything sooner
    LatticeArc arc(
        fst_.h_arc_id_ilabels_[list_arc_idx], fst_.h_arc_olabels_[list_arc_idx],
        LatticeWeight(fst_.h_arc_weights_[list_arc_idx], acoustic_cost),
        to_fst_lattice_state);
    fst_out->AddArc(from_fst_lattice_state, arc);
  }
}

void CudaDecoder::ResetDataForGetRawLattice() {
  // Using one map per frame. We always know to which frame a token belongs.
  // Using one big map slows everything down
  prev_f_raw_lattice_state_.clear();
  curr_f_raw_lattice_state_.clear();
  // We want the unicity of each arc_idx for one frame. Important because we
  // can replay a frame (and possibly add multiple time the same arc)
  f_arc_idx_added_.clear();

  // Keeping track of which tokens need to be computed. Think of those as FIFO
  // queues, except that we don't want to pop the front right away, because we
  // may replay a frame
  // (and we need to remember what's in that frame)
  // We are also not using an iterator through the
  // [prev|curr]_f_raw_lattice_state because we are
  // sometimes adding stuff in q_curr_frame_todo_ while reading it.
  // We can possibly add the new element before the current map iterator
  // (and we wouldn't read it)
  q_curr_frame_todo_.clear();
  q_prev_frame_todo_.clear();
}

void CudaDecoder::GetTokenRawLatticeData(
    TokenId token_idx, InfoToken token, int32 total_ntokens,
    CostType *token_extra_cost, OutputLatticeState *to_fst_lattice_state) {
  LatticeStateInternalId next_state_internal_id =
      GetLatticeStateInternalId(total_ntokens, token_idx, token);
  auto to_map_it = curr_f_raw_lattice_state_.find(next_state_internal_id);
  // We know this token exists in the output lattice (because it's in
  // q_curr_frame_todo_)
  KALDI_ASSERT(to_map_it != curr_f_raw_lattice_state_.end());

  *token_extra_cost = to_map_it->second.token_extra_cost;
  *to_fst_lattice_state = to_map_it->second.fst_lattice_state;

  // We read the extra cost from lattice_next_state
  // We are now closing the state. If we write to it again, we will have
  // to replay that frame
  // (so that the latest extra_cost value is read)
  to_map_it->second.is_state_closed = true;
}

void CudaDecoder::GetSameFSTStateTokenList(
    ChannelId ichannel, InfoToken token, InfoToken **tok_beg,
    float2 **extra_extra_and_acoustic_cost_beg, int32 *nsame) {
  // We now need to consider all tokens related to that (iframe,
  // fst_state)
  // with fst_state being the state this current token is linked to
  // There's two possibilies:
  // a) only one token is associated with that fst_state in that frame.
  // The necessary information
  // is then stored directly in the token (arc_idx, prev_token)
  // b) multiple tokens are associated with that fst_state in that
  // frame. The token that we have right now
  // only contains information on where to find the list of those
  // tokens. It contains (offset, size)
  //
  // In any cases we consider the list of tokens to process as an array
  // of InfoToken, which will
  // be of size 1 in case a), of size > 1 in case b)
  if (token.IsUniqueTokenForStateAndFrame()) {
    *tok_beg = &token;
    // if we've got only one, extra_cost == 0.0
    *extra_extra_and_acoustic_cost_beg = NULL;
    *nsame = 1;
  } else {
    int32 offset, size;
    std::tie(offset, size) = token.GetSameFSTStateTokensList();
    *tok_beg = &h_all_tokens_extra_prev_tokens_[ichannel][offset];
    *extra_extra_and_acoustic_cost_beg =
        &h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel]
                                                                [offset];
    *nsame = size;
  }
}

void CudaDecoder::ConsiderTokenForLattice(
    ChannelId ichannel, int32 iprev, int32 total_ntokens, TokenId token_idx,
    OutputLatticeState fst_lattice_start, InfoToken *tok_beg,
    float2 *extra_extra_and_acoustic_cost_beg, CostType token_extra_cost,
    TokenId list_prev_token_idx, int32 list_arc_idx, InfoToken *list_prev_token,
    CostType *this_arc_prev_token_extra_cost, CostType *acoustic_cost,
    OutputLatticeState *lattice_src_state, bool *keep_arc,
    bool *dbg_found_zero) {
  CostType arc_extra_cost;
  if (extra_extra_and_acoustic_cost_beg) {
    float2 both = extra_extra_and_acoustic_cost_beg[iprev];
    arc_extra_cost = both.x;
    *acoustic_cost = both.y;
  } else {
    // If we have only one token for that (iframe,fst_state),
    // Its arc has an extra_cost of zero (it's the only way to
    // get to that state, so it's the best)
    arc_extra_cost = 0.0f;
    *acoustic_cost = h_all_tokens_acoustic_cost_[ichannel][token_idx];
  }
  // If we use that arc to go to prev_token, prev_token will have the
  // following extra cost
  *this_arc_prev_token_extra_cost = token_extra_cost + arc_extra_cost;
  // We need at least one arc_extra_cost of zero for each (iframe,
  // fst_state)
  // The only use for that boolean is in a KALDI_ASSERT,
  // because if something went wrong in the kernels it's not likely
  // that this property will be verified out of luck
  *dbg_found_zero |= (arc_extra_cost == 0.0f);
  *list_prev_token = h_all_tokens_info_[ichannel][list_prev_token_idx];
  // Source of the arc currently considered
  *lattice_src_state =
      (list_prev_token_idx != 0)
          ? GetLatticeStateInternalId(total_ntokens, list_prev_token_idx,
                                      *list_prev_token)
          : fst_lattice_start;

  // We only keep the arc if, when using that arc, we can end up
  // at the last frame with a cost not worse than (best+lattice_beam)
  // this_arc_prev_token_extra_cost contains the accumulated sums
  // of extra costs (through the cheapest possible way) to the last
  // frame
  *keep_arc = (*this_arc_prev_token_extra_cost < lattice_beam_);
}

void CudaDecoder::SwapPrevAndCurrLatticeMap(int32 iframe,
                                            bool dbg_found_best_path) {
  q_prev_frame_todo_.swap(q_curr_frame_todo_);
  q_prev_frame_todo_.clear();
  prev_f_raw_lattice_state_.swap(curr_f_raw_lattice_state_);
  prev_f_raw_lattice_state_.clear();
  f_arc_idx_added_.clear();

  KALDI_ASSERT(q_prev_frame_todo_.empty());
  if (iframe > 0) {
    KALDI_ASSERT(!q_curr_frame_todo_.empty());
    if (!dbg_found_best_path) {
      KALDI_WARN << "Warning didn't find exact best path in GetRawLattice";
    }
  }
}

void CudaDecoder::GetRawLattice(const std::vector<ChannelId> &channels,
                                std::vector<Lattice *> &fst_out_vec,
                                bool use_final_probs) {
  KALDI_ASSERT(channels.size() == fst_out_vec.size());
  // Getting the list of the best costs in the lastest token queue.
  // all costs within [best;best+lattice_beam]
  GetBestCost(channels, use_final_probs, &argmins_,
              &list_finals_token_idx_and_cost_, &has_reached_final_);

  for (int32 ilane = 0; ilane < channels.size(); ++ilane) {
    nvtxRangePushA("GetRawLatticeOneChannel");
    const ChannelId ichannel = channels[ilane];
    const int32 nframes = NumFramesDecoded(ichannel);

    // Total number of tokens generated by the utterance on channel ichannel
    const int32 total_ntokens = h_all_tokens_info_[ichannel].size();

    // Preparing output lattice
    // The start state has to be 0 (cf some asserts somewhere else in Kaldi)
    // Adding it now
    Lattice *fst_out = fst_out_vec[ilane];
    fst_out->DeleteStates();
    OutputLatticeState fst_lattice_start = fst_out->AddState();
    fst_out->SetStart(fst_lattice_start);

    ResetDataForGetRawLattice();
    // Adding the best tokens returned by GetBestCost to the lattice
    // We also add them to q_curr_frame_todo, and we'll backtrack from there
    AddFinalTokensToLattice(ilane, ichannel, fst_out);

    // We're now going to backtrack frame by frame
    // For each frame we're going to process tokens that need to be inserted
    // into the output lattice
    // and add their predecessors to the todo list
    // iframe == -1 contains the start state and the first non emitting tokens.
    // It is not linked to a real frame
    for (int32 iframe = nframes - 1; iframe >= -1; --iframe) {
      // Tokens for the current frame were inserted after this offset in the
      // token list
      const int32 curr_frame_offset =
          (iframe >= 0) ? frame_offsets_[ichannel][iframe] : 0;

      // bool must_replay_frame
      // In some cases we can update an extra_cost that has already been used
      // For instance we process arcs in that order :
      // 1) a -> b, which updates extra_cost[b] using extra_cost[a]
      // 2) c -> a, which updates extra-cost[a] (using extra_cost[c])
      // because the arcs were not considered in topological order, we need to
      // run
      // again the step 1,
      // to get the correct extra_cost[b] (using the latest extra_cost[a])
      // However, we only re-run the step 1 if the value extra_cost[a] has
      // changed more than extra_cost_min_delta_
      bool must_replay_frame;

      // dbg_found_best_path is used in an useful assert, making sure the best
      // path is still there for each frame
      // if something went wrong in the kernels, it's not likely we respect that
      // property out of luck
      bool dbg_found_best_path = false;
      do {
        must_replay_frame = false;
        // Reading something to do. We are pushing stuff back in
        // q_curr_frame_todo_ while reading it,
        // so it's important to always read q_curr_frame_todo_.size() directly
        // not using a queue, because we may need to recompute the frame (if
        // must_replay_frame is true)
        for (int32 u = 0; u < q_curr_frame_todo_.size(); ++u) {
          TokenId token_idx;
          InfoToken token;
          std::tie(token_idx, token) = q_curr_frame_todo_[u];
          KALDI_ASSERT(token_idx >= curr_frame_offset);
          CostType token_extra_cost;
          StateId to_fst_lattice_state;
          // Loading the current extra_cost of that token
          // + its associated state in the lattice
          GetTokenRawLatticeData(token_idx, token, total_ntokens,
                                 &token_extra_cost, &to_fst_lattice_state);
          dbg_found_best_path |= (token_extra_cost == 0.0f);

          InfoToken *tok_beg;
          float2 *extra_extra_and_acoustic_cost_beg;
          int32 nsamestate;
          // Getting the list of the tokens linked to the same FST state, in the
          // same frame
          // In the GPU decoder a token is linked to a single arc, but we can
          // generate
          // multiple token for a same fst_nextstate in the same frame.
          // In the CPU decoder we would use the forward_links list to store
          // everything in the same metatoken
          // GetSameFSTStateTokenList returns the list of tokens linked to the
          // same FST state than token
          // (in the current frame)
          GetSameFSTStateTokenList(ichannel, token, &tok_beg,
                                   &extra_extra_and_acoustic_cost_beg,
                                   &nsamestate);

          // Used for debugging. For each FST state, we have a token with the
          // best cost for that FST state
          // that token has an extra_cost of 0.0f. This is a sanity check
          bool dbg_found_zero = false;
          for (int32 iprev = 0; iprev < nsamestate; ++iprev) {
            int32 list_prev_token_idx, list_arc_idx;
            InfoToken list_prev_token;
            CostType acoustic_cost, this_arc_prev_token_extra_cost;
            bool keep_arc;
            LatticeStateInternalId src_state_internal_id;
            InfoToken list_token = tok_beg[iprev];
            list_prev_token_idx = list_token.prev_token;
            list_arc_idx = list_token.arc_idx;

            ConsiderTokenForLattice(
                ichannel, iprev, total_ntokens, token_idx, fst_lattice_start,
                tok_beg, extra_extra_and_acoustic_cost_beg, token_extra_cost,
                list_prev_token_idx, list_arc_idx, &list_prev_token,
                &this_arc_prev_token_extra_cost, &acoustic_cost,
                &src_state_internal_id, &keep_arc, &dbg_found_zero);

            if (keep_arc)
              AddArcToLattice(list_arc_idx, list_prev_token_idx,
                              list_prev_token, curr_frame_offset, acoustic_cost,
                              this_arc_prev_token_extra_cost,
                              src_state_internal_id, fst_lattice_start,
                              to_fst_lattice_state, fst_out,
                              &must_replay_frame);
          }
          KALDI_ASSERT(dbg_found_zero);
        }

        if (must_replay_frame) {
          // We need to replay the frame. Because all states will be read again,
          // we can reopen them (and they will be closed again when being read
          // from again)
          for (auto it = curr_f_raw_lattice_state_.begin();
               it != curr_f_raw_lattice_state_.end(); ++it) {
            it->second.is_state_closed = false;
          }
        }
      } while (must_replay_frame);

      // Done processing this frame. Swap the datastructures, move on to
      // previous frame (we go --iframe)
      SwapPrevAndCurrLatticeMap(iframe, dbg_found_best_path);
    }

    nvtxRangePop();
  }
}

void CudaDecoder::SetChannelsInKernelParams(
    const std::vector<ChannelId> &channels) {
  KALDI_ASSERT(channels.size() <= nchannels_);
  KALDI_ASSERT(channels.size() <= nlanes_);
  for (LaneId lane_id = 0; lane_id < channels.size(); ++lane_id)
    h_kernel_params_->channel_to_compute[lane_id] = channels[lane_id];
  h_kernel_params_->nlanes_used = channels.size();
  nlanes_used_ = channels.size();
}

void CudaDecoder::ResetChannelsInKernelParams() {
  h_kernel_params_->nlanes_used = 0;
  nlanes_used_ = 0;
}

int32 CudaDecoder::NumFramesDecoded(ChannelId ichannel) const {
  KALDI_ASSERT(ichannel < nchannels_);
  return num_frames_decoded_[ichannel];
}

void CudaDecoder::CheckStaticAsserts() {
  // Checking if all constants look ok

  // We need that because we need to be able to do the scan in one pass in the
  // kernel
  // update_beam_using_histogram_kernel
  KALDI_ASSERT(KALDI_CUDA_DECODER_HISTO_NBINS < KALDI_CUDA_DECODER_1D_BLOCK);
  KALDI_ASSERT(KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS > 0);
}
/*
        int32 CudaDecoder::NumFramesDecoded() const {
                return NumFramesDecoded(0);
        }
*/
}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // HAVE_CUDA == 1

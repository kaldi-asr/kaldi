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

#if !HAVE_CUDA
#error CUDA support must be configured to compile this library.
#endif

#include "cudadecoder/cuda-decoder.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <nvToolsExt.h>

#include "base/kaldi-utils.h"
#include "cudadecoder/cuda-decoder-kernels.h"
#include "cudamatrix/cu-common.h"
#include "online2/online-endpoint.h"
#include "util/text-utils.h"


namespace kaldi {
namespace cuda_decoder {

CudaDecoder::CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config,
                         int32 nlanes, int32 nchannels)
    : word_syms_(nullptr),
      generate_partial_hypotheses_(false),
      endpointing_(false),
      partial_traceback_(false),
      frame_shift_seconds_(FLT_MAX),
      fst_(fst),
      nlanes_(nlanes),
      nchannels_(nchannels),
      channel_lock_(nchannels + 1),
      extra_cost_min_delta_(0.0f),
      thread_pool_(nullptr),
      n_threads_used_(0),
      n_h2h_task_not_done_(0),
      n_init_decoding_h2h_task_not_done_(0),
      h2h_threads_running_(true) {
  ReadConfig(config);
  // Static asserts on constants
  CheckStaticAsserts();
  // Runtime asserts
  KALDI_ASSERT(nlanes_ > 0);
  KALDI_ASSERT(nchannels_ > 0);
  KALDI_ASSERT(nlanes_ <= nchannels_);
  // All GPU work in decoder will be sent to compute_st_
  CU_SAFE_CALL(cudaStreamCreate(&compute_st_));
  // Copies D2H of tokens for storage on host are done on
  // copy_st_, in parallel with compute_st_
  CU_SAFE_CALL(cudaStreamCreate(&copy_st_));
  // For all the allocating/initializing process
  // We create a special channel
  // containing the exact state a channel should have when starting a new
  // decode It contains fst.Start(), the non-emitting tokens created by
  // fst.Start(), and all the data used by the decoder. When calling
  // InitDecoding() on a new channel, we simply clone this special channel
  // into that new channel
  ++nchannels_;                       // adding the special initial channel
  init_channel_id_ = nchannels_ - 1;  // Using last one as init_channel_params
  AllocateHostData();
  AllocateDeviceData();
  AllocateDeviceKernelParams();

  InitDeviceParams();
  InitHostData();
  InitDeviceData();

  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventCreate(&nnet3_done_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventCreate(&d2h_copy_acoustic_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventCreate(&d2h_copy_infotoken_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaEventCreate(&d2h_copy_extra_prev_tokens_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaEventCreate(&concatenated_data_ready_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventCreate(&lane_offsets_ready_evt_));

  ComputeInitialChannel();
  --nchannels_;  // removing the special initial channel from the count

  // Making sure that everything is ready to use
  CU_SAFE_CALL(cudaStreamSynchronize(compute_st_));
  KALDI_DECODER_CUDA_CHECK_ERROR();
}

void CudaDecoder::ReadConfig(const CudaDecoderConfig &cst_config) {
  config_ = cst_config;  // deep copy
  // Sets the missing values using other values
  config_.ComputeConfig();
  default_beam_ = config_.default_beam;
  lattice_beam_ = config_.lattice_beam;
  ntokens_pre_allocated_ = config_.ntokens_pre_allocated;
  max_active_ = config_.max_active;
  aux_q_capacity_ = config_.aux_q_capacity;
  main_q_capacity_ = config_.main_q_capacity;

  KALDI_ASSERT(default_beam_ >= 0.0f);
  KALDI_ASSERT(lattice_beam_ >= 0.0f);
  KALDI_ASSERT(ntokens_pre_allocated_ >= 0);
  KALDI_ASSERT(max_active_ > 0);
  KALDI_ASSERT(main_q_capacity_ > 0);
  KALDI_ASSERT(aux_q_capacity_ >= main_q_capacity_);

  // Filling silence phones set with input config
  std::vector<int32> silence_phones_vec;
  if (!SplitStringToIntegers(config_.endpointing_config.silence_phones, ":",
                             false, &silence_phones_vec))
    KALDI_ERR << "Bad --endpoint.silence-phones option in endpointing config: "
              << config_.endpointing_config.silence_phones;
  for (int32 phone : silence_phones_vec) silence_phones_.insert(phone);
}

void CudaDecoder::AllocateDeviceData() {
  hashmap_capacity_ =
      KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR * main_q_capacity_;
  d_channels_counters_.Resize(nchannels_, 1);
  d_lanes_counters_.Resize(
      nlanes_ + 1,
      1);  // +1 because we sometimes need last+1 value (for offsets)
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
  d_extra_and_acoustic_cost_concat_matrix_.Resize(nlanes_, main_q_capacity_);
  d_acoustic_cost_concat_matrix_.Resize(nlanes_, main_q_capacity_);
  d_infotoken_concat_matrix_.Resize(nlanes_, main_q_capacity_);
  d_extra_prev_tokens_concat_matrix_.Resize(nlanes_, main_q_capacity_);
  // Reusing data from aux_q. Those two are never used at the same time
  // d_list_final_tokens_in_main_q_ is used in GetBestPath.
  // the aux_q is used in AdvanceDecoding
  h_list_final_tokens_in_main_q_.Resize(nlanes_, main_q_capacity_);
  d_extra_prev_tokens_concat_ = d_extra_prev_tokens_concat_matrix_.lane(0);
  d_extra_and_acoustic_cost_concat_ =
      d_extra_and_acoustic_cost_concat_matrix_.lane(0);
  d_acoustic_cost_concat_ = d_acoustic_cost_concat_matrix_.lane(0);
  d_infotoken_concat_ = d_infotoken_concat_matrix_.lane(0);
}

void CudaDecoder::AllocateHostData() {
  channel_to_compute_.resize(nlanes_);
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_extra_and_acoustic_cost_concat_,
      nlanes_ * main_q_capacity_ * sizeof(*h_extra_and_acoustic_cost_concat_)));
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
      cudaMallocHost(&h_extra_and_acoustic_cost_concat_tmp_,
                     nlanes_ * main_q_capacity_ *
                         sizeof(*h_extra_and_acoustic_cost_concat_tmp_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_acoustic_cost_concat_tmp_,
      nlanes_ * main_q_capacity_ * sizeof(*h_acoustic_cost_concat_tmp_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_extra_prev_tokens_concat_tmp_,
      nlanes_ * main_q_capacity_ * sizeof(*h_extra_prev_tokens_concat_tmp_)));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_infotoken_concat_tmp_,
      nlanes_ * main_q_capacity_ * sizeof(*h_infotoken_concat_tmp_)));
  h_lanes_counters_.Resize(
      nlanes_ + 1,
      1);  // +1 because we sometimes need last+1 value (for offsets)
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMallocHost(
      &h_channels_counters_, nchannels_ * sizeof(*h_channels_counters_)));

  h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_.resize(nchannels_);
  h_all_tokens_acoustic_cost_.resize(nchannels_);
  h_all_tokens_extra_prev_tokens_.resize(nchannels_);
  h_all_tokens_info_.resize(nchannels_);
  h_all_channels_partial_hypotheses_.resize(nchannels_);
  h_all_channels_partial_hypotheses_out_.resize(nchannels_);
  h_all_channels_endpoint_detected_.resize(nchannels_);
  for (int32 ichannel = 0; ichannel < nchannels_; ++ichannel) {
    h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel].reserve(
        ntokens_pre_allocated_);
    h_all_tokens_acoustic_cost_[ichannel].reserve(ntokens_pre_allocated_);
    h_all_tokens_info_[ichannel].reserve(ntokens_pre_allocated_);
  }
  h_main_q_end_lane_offsets_.resize(nlanes_ + 1);
  h_emitting_main_q_end_lane_offsets_.resize(nlanes_ + 1);
  h_n_extra_prev_tokens_lane_offsets_.resize(nlanes_ + 1);
  h_best_path_traceback_head_.resize(nlanes_);
  h_all_channels_prev_best_path_traceback_head_.resize(nchannels_);
  frame_offsets_.resize(nchannels_);
  num_frames_decoded_.resize(nchannels_, -1);
  lanes2channels_todo_.reserve(nlanes_);

  h_all_argmin_cost_.resize(nchannels_, {-1, 0.0f});
  h_all_final_tokens_list_.resize(nchannels_);
  h_all_has_reached_final_.resize(nchannels_);
}

void CudaDecoder::InitDeviceData() {
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

void CudaDecoder::InitHostData() {}

void CudaDecoder::AllocateDeviceKernelParams() {
  h_device_params_ = std::make_unique<DeviceParams>();
  h_kernel_params_ = std::make_unique<KernelParams>();
}

void CudaDecoder::InitDeviceParams() {
  // Setting Kernel Params
  // Sent to cuda kernels by copy
  // Making sure we'll be able to send it to the kernels
  KALDI_COMPILE_TIME_ASSERT((sizeof(KernelParams) + sizeof(DeviceParams)) <
                            KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE);

  h_device_params_->d_channels_counters = d_channels_counters_.GetView();
  h_device_params_->d_lanes_counters = d_lanes_counters_.GetView();
  h_device_params_->h_lanes_counters = h_lanes_counters_.GetView();
  h_device_params_->d_main_q_state_and_cost =
      d_main_q_state_and_cost_.GetView();
  h_device_params_->d_main_q_info = d_main_q_info_.GetView();
  h_device_params_->d_aux_q_state_and_cost = d_aux_q_state_and_cost_.GetView();
  h_device_params_->d_main_q_extra_and_acoustic_cost =
      d_main_q_extra_and_acoustic_cost_.GetView();
  h_device_params_->d_main_q_acoustic_cost = d_main_q_acoustic_cost_.GetView();
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
  h_device_params_->d_arc_e_offsets = fst_.d_e_offsets_.get();
  h_device_params_->d_arc_ne_offsets = fst_.d_ne_offsets_.get();
  h_device_params_->d_arc_pdf_ilabels = fst_.d_arc_pdf_ilabels_.get();
  h_device_params_->d_arc_weights = fst_.d_arc_weights_.get();
  h_device_params_->d_arc_nextstates = fst_.d_arc_nextstates_.get();
  h_device_params_->d_fst_final_costs = fst_.d_final_.get();
  h_device_params_->default_beam = default_beam_;
  h_device_params_->lattice_beam = lattice_beam_;
  h_device_params_->main_q_capacity = main_q_capacity_;
  h_device_params_->aux_q_capacity = aux_q_capacity_;
  h_device_params_->init_channel_id = init_channel_id_;
  h_device_params_->max_nlanes = nlanes_;
  h_device_params_->nstates = fst_.num_states_;
  h_device_params_->init_state = fst_.Start();
  KALDI_ASSERT(h_device_params_->init_state != fst::kNoStateId);
  h_device_params_->init_cost = CudaFst::Weight::One().Value();
  h_device_params_->hashmap_capacity = hashmap_capacity_;
  h_device_params_->max_active = max_active_;
  // For the first static_beam_q_length elements of the queue, we will
  // keep the beam static
  adaptive_beam_static_segment_ =
      aux_q_capacity_ / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT;
  // For the last adaptive_beam_q_length elements of the queue, we will
  // decrease the beam, segment by segment For more information, please
  // refer to the definition of GetAdaptiveBeam in cuda-decoder-kernels.cu
  int32 adaptive_beam_q_length =
      (aux_q_capacity_ - adaptive_beam_static_segment_);
  int32 adaptive_beam_bin_width =
      adaptive_beam_q_length / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NSTEPS;
  h_device_params_->adaptive_beam_static_segment =
      adaptive_beam_static_segment_;
  h_device_params_->adaptive_beam_bin_width = adaptive_beam_bin_width;

  // Reusing aux_q memory to list final states in GetLattice
  // Those cannot be used at the same time
  h_device_params_->h_list_final_tokens_in_main_q =
      h_list_final_tokens_in_main_q_.GetView();
  h_device_params_->fst_zero = CudaFst::Weight::Zero().Value();
}

CudaDecoder::~CudaDecoder() noexcept(false) {
  // Wait for D2H copies before stopping H2H tasks.
  CU_SAFE_CALL(cudaStreamSynchronize(compute_st_));
  CU_SAFE_CALL(cudaStreamSynchronize(copy_st_));
  // Stop h2h tasks.
  WaitForInitDecodingH2HCopies();
  WaitForH2HCopies();
  h2h_threads_running_ = false;
  n_h2h_main_task_todo_cv_.notify_all();
  for (std::thread &thread : cpu_dedicated_threads_) thread.join();
  KALDI_ASSERT(n_h2h_main_task_todo_ <= 0);
  KALDI_ASSERT(n_h2h_task_not_done_ == 0);

  CU_SAFE_CALL(cudaStreamDestroy(compute_st_));
  CU_SAFE_CALL(cudaStreamDestroy(copy_st_));

  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_channels_counters_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaFreeHost(h_extra_and_acoustic_cost_concat_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_acoustic_cost_concat_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_extra_prev_tokens_concat_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_infotoken_concat_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaFreeHost(h_extra_and_acoustic_cost_concat_tmp_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_acoustic_cost_concat_tmp_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaFreeHost(h_extra_prev_tokens_concat_tmp_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaFreeHost(h_infotoken_concat_tmp_));
  // Will call the cudaFrees inside destructors
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventDestroy(nnet3_done_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventDestroy(d2h_copy_acoustic_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventDestroy(d2h_copy_infotoken_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaEventDestroy(d2h_copy_extra_prev_tokens_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaEventDestroy(concatenated_data_ready_evt_));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaEventDestroy(lane_offsets_ready_evt_));
}

void CudaDecoder::ComputeInitialChannel() {
  KALDI_ASSERT(nlanes_ > 0);
  const LaneId kLane0 = 0;

  // Following kernels working channel_id
  std::vector<ChannelId> channels = {init_channel_id_};
  num_frames_decoded_[init_channel_id_] = 0;
  SetChannelsInKernelParams(channels);  // not calling LoadChannelsStateToLanes,
                                        // init_channel_id_ is a special case
  h_lanes_counters_.lane(kLane0)->channel_to_compute = init_channel_id_;

  CU_SAFE_CALL(cudaMemcpyAsync(d_lanes_counters_.MutableData(),
                               h_lanes_counters_.lane(0),
                               1 * sizeof(*h_lanes_counters_.lane(0)),
                               cudaMemcpyHostToDevice, compute_st_));
  h_lanes_counters_.lane(kLane0)->main_q_narcs_and_end.y = 0;

  // Adding the start state to the initial token queue
  InitializeInitialLaneKernel(KaldiCudaDecoderNumBlocks(1, 1),
                              KALDI_CUDA_DECODER_ONE_THREAD_BLOCK, compute_st_,
                              *h_device_params_);

  h_lanes_counters_.lane(kLane0)->post_expand_aux_q_end = 1;

  PruneAndPreprocess();
  FinalizeProcessNonEmittingKernel(
      KaldiCudaDecoderNumBlocks(1, 1), KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);

  CopyLaneCountersToHostSync();
  PostProcessingMainQueue();
  ConcatenateData();
  CopyLaneCountersToHostSync();

  const int32 main_q_end =
      h_lanes_counters_.lane(kLane0)->main_q_narcs_and_end.y;
  KALDI_ASSERT(main_q_end > 0);

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
  // Cloning the init_channel_id_ channel into all channels in the
  // channels vec
  const int nlanes_used = channels.size();
  // Getting *h_kernel_params ready to use
  LoadChannelsStateToLanes(channels);
  CU_SAFE_CALL(cudaMemcpyAsync(d_lanes_counters_.MutableData(),
                               h_lanes_counters_.lane(0),
                               nlanes_used_ * sizeof(*h_lanes_counters_.lane(0)),
                               cudaMemcpyHostToDevice, compute_st_));

  // Size of the initial main_q
  ChannelCounters &init_channel_counters =
      h_channels_counters_[init_channel_id_];
  const int32 init_main_q_size =
      init_channel_counters.prev_main_q_narcs_and_end.y;

  KALDI_ASSERT(init_main_q_size > 0);
  // Getting the channels ready to compute new utterances
  InitDecodingOnDeviceKernel(
      KaldiCudaDecoderNumBlocks(init_main_q_size, nlanes_used),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  {
    std::lock_guard<std::mutex> lk(n_init_decoding_h2h_task_not_done_mutex_);
    n_init_decoding_h2h_task_not_done_ += channels.size();
  }
  for (ChannelId ichannel : channels) {
    ChannelCounters &channel_counters = h_channels_counters_[ichannel];
    channel_counters.prev_main_q_narcs_and_end =
        init_channel_counters.prev_main_q_narcs_and_end;
    channel_counters.prev_main_q_n_extra_prev_tokens =
        init_channel_counters.prev_main_q_n_extra_prev_tokens;
    channel_counters.prev_main_q_global_offset = 0;
    channel_counters.prev_main_q_extra_prev_tokens_global_offset = 0;
    channel_counters.prev_beam = default_beam_;

    int32 n_initial_tokens = h_all_tokens_info_[init_channel_id_].size();
    num_frames_decoded_[ichannel] = 0;
    h_channels_counters_[ichannel] = h_channels_counters_[init_channel_id_];
    h_all_argmin_cost_[ichannel] = {-1, 0.0f};
    frame_offsets_[ichannel].clear();
    frame_offsets_[ichannel].push_back(n_initial_tokens);
    h_all_channels_partial_hypotheses_[ichannel].clear();
    h_all_channels_partial_hypotheses_out_[ichannel].clear();
    h_all_channels_endpoint_detected_[ichannel] = false;
    h_all_channels_prev_best_path_traceback_head_[ichannel].Reset();
    // TODO put it back
    // if (thread_pool_) {
    //  thread_pool_->post([ichannel, this] {
    //  InitDecodingH2HCopies(ichannel);
    //  });
    //} else
    InitDecodingH2HCopies(ichannel);
  }
}

void CudaDecoder::InitDecodingH2HCopies(ChannelId ichannel) {
  // Tokens from initial main_q needed on host
  std::unique_lock<std::mutex> channel_lk(channel_lock_[ichannel]);
  // Deep copy
  h_all_tokens_info_[ichannel] = h_all_tokens_info_[init_channel_id_];
  h_all_tokens_acoustic_cost_[ichannel] =
      h_all_tokens_acoustic_cost_[init_channel_id_];
  h_all_tokens_extra_prev_tokens_[ichannel] =
      h_all_tokens_extra_prev_tokens_[init_channel_id_];
  h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel] =
      h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[init_channel_id_];

  bool all_done;
  {
    std::lock_guard<std::mutex> lk_not_done(
        n_init_decoding_h2h_task_not_done_mutex_);
    all_done = (--n_init_decoding_h2h_task_not_done_ == 0);
  }
  if (all_done) {
    init_decoding_h2h_done_.notify_all();
  }
}

void CudaDecoder::LoadChannelsStateToLanes(
    const std::vector<ChannelId> &channels) {
  // Setting that channels configuration in kernel_params
  SetChannelsInKernelParams(channels);
  KALDI_ASSERT(nlanes_used_ > 0);
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    const ChannelId ichannel = channel_to_compute_[ilane];
    ChannelCounters &channel_counters = h_channels_counters_[ichannel];
    LaneCounters &lane_counters = *h_lanes_counters_.lane(ilane);
    lane_counters.channel_to_compute = ichannel;
    lane_counters.main_q_narcs_and_end =
        channel_counters.prev_main_q_narcs_and_end;
    lane_counters.main_q_n_extra_prev_tokens =
        channel_counters.prev_main_q_n_extra_prev_tokens;
    int32 int_beam = floatToOrderedIntHost(channel_counters.prev_beam);
    lane_counters.int_beam = int_beam;
    lane_counters.adaptive_int_beam_with_validity_index.x = int_beam;
    lane_counters.adaptive_int_beam_with_validity_index.y =
        adaptive_beam_static_segment_;
    lane_counters.main_q_global_offset =
        channel_counters.prev_main_q_global_offset;
    lane_counters.main_q_extra_prev_tokens_global_offset =
        channel_counters.prev_main_q_extra_prev_tokens_global_offset;

    lane_counters.min_int_cost =
        channel_counters.min_int_cost_and_arg_without_final.x;
    lane_counters.prev_arg_min_int_cost =
        channel_counters.min_int_cost_and_arg_without_final.y;
  }
}

void CudaDecoder::SaveChannelsStateFromLanes() {
  KALDI_ASSERT(nlanes_used_ > 0);
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    const ChannelId ichannel = channel_to_compute_[ilane];
    ChannelCounters &channel_counters = h_channels_counters_[ichannel];
    LaneCounters &lane_counters = *h_lanes_counters_.lane(ilane);
    channel_counters.prev_main_q_narcs_and_end =
        lane_counters.main_q_narcs_and_end;
    channel_counters.prev_main_q_extra_prev_tokens_global_offset =
        lane_counters.main_q_extra_prev_tokens_global_offset;
    channel_counters.prev_main_q_global_offset =
        lane_counters.main_q_global_offset;
    channel_counters.prev_main_q_n_extra_prev_tokens =
        lane_counters.main_q_n_extra_prev_tokens;
    channel_counters.prev_beam = orderedIntToFloatHost(lane_counters.int_beam);
    channel_counters.min_int_cost_and_arg_without_final = {
        lane_counters.min_int_cost, lane_counters.prev_arg_min_int_cost};
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
    const int32 val = func(*h_lanes_counters_.lane(ilane));
    max_val = std::max(max_val, val);
  }
  return max_val;
}

void CudaDecoder::CopyLaneCountersToHostAsync() {
  CU_SAFE_CALL(cudaMemcpyAsync(h_lanes_counters_.lane(0),
                               d_lanes_counters_.MutableData(),
                               nlanes_used_ * sizeof(*h_lanes_counters_.lane(0)),
                               cudaMemcpyDeviceToHost, compute_st_));
}

void CudaDecoder::CopyLaneCountersToHostSync() {
  CopyLaneCountersToHostAsync();
  CU_SAFE_CALL(cudaStreamSynchronize(compute_st_));
}

// One sync has to happen between PerformConcatenatedCopy and
// MoveConcatenatedCopyToVector
template <typename T>
void CudaDecoder::MoveConcatenatedCopyToVector(
    const int32 ilane, const int32 ichannel,
    const std::vector<int32> &lanes_offsets, T *h_concat,
    std::vector<std::vector<T>> *vecvec) {
  // Unpacking the concatenated vector into individual channel storage
  int32 beg = lanes_offsets[ilane];
  int32 end = lanes_offsets[ilane + 1];
  auto &vec = (*vecvec)[ichannel];
  vec.insert(vec.end(), h_concat + beg, h_concat + end);
}

void CudaDecoder::ApplyMaxActiveAndReduceBeam(enum QUEUE_ID queue_id) {
  // Checking if we should activate max active for the current frame
  // once it is active, it is active for the whole frame (for all non
  // emitting iterations) If at least one lane queue is bigger than
  // max_active, we'll apply a topk on that queue (k=max_active_)
  bool use_aux_q = (queue_id == AUX_Q);
  ComputeCostsHistogramKernel(KaldiCudaDecoderNumBlocks(nlanes_used_),
                              KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                              *h_device_params_, *h_kernel_params_, use_aux_q);

  UpdateBeamUsingHistogramKernel(
      KaldiCudaDecoderNumBlocks(1, nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_, use_aux_q);
}

void CudaDecoder::ExpandArcsEmitting() {
  ExpandArcsKernel<true>(KaldiCudaDecoderNumBlocks(nlanes_used_),
                         KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                         *h_device_params_, *h_kernel_params_);

  // Updating a few counters, like resetting aux_q_end to 0...
  // true is for IS_EMITTING
  PostExpandKernel<true>(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                         KALDI_CUDA_DECODER_ONE_THREAD_BLOCK, compute_st_,
                         *h_device_params_, *h_kernel_params_);
}

void CudaDecoder::ExpandArcsNonEmitting() {
  // false is for non emitting
  ExpandArcsKernel<false>(KaldiCudaDecoderNumBlocks(nlanes_used_),
                          KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                          *h_device_params_, *h_kernel_params_);

  // false is for non emitting
  PostExpandKernel<false>(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                          KALDI_CUDA_DECODER_ONE_THREAD_BLOCK, compute_st_,
                          *h_device_params_, *h_kernel_params_);
}

void CudaDecoder::PruneAndPreprocess() {
  NonEmittingPreprocessAndContractKernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);
  PostContractAndPreprocessKernel(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                                  KALDI_CUDA_DECODER_ONE_THREAD_BLOCK,
                                  compute_st_, *h_device_params_,
                                  *h_kernel_params_);
}

void CudaDecoder::PostProcessingMainQueue() {
  ApplyMaxActiveAndReduceBeam(MAIN_Q);

  FillHashmapWithMainQKernel(KaldiCudaDecoderNumBlocks(nlanes_used_),
                             KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                             *h_device_params_, *h_kernel_params_);

  EmittingPreprocessAndListExtraPrevTokensStep1Kernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);

  EmittingPreprocessAndListExtraPrevTokensStep2Kernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);

  // Step2 wrote main_q_n_extra_prev_tokens
  // it was the last value missing to compute the lanes offsets
  // doing it now
  ComputeLaneOffsetsKernel(KaldiCudaDecoderNumBlocks(1, 1),  // One CTA
                           KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                           *h_device_params_, *h_kernel_params_);
  CU_SAFE_CALL(cudaEventRecord(lane_offsets_ready_evt_, compute_st_));

  EmittingPreprocessAndListExtraPrevTokensStep3Kernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);

  EmittingPreprocessAndListExtraPrevTokensStep4Kernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);

  ClearHashmapKernel(KaldiCudaDecoderNumBlocks(nlanes_used_),
                     KALDI_CUDA_DECODER_1D_BLOCK, compute_st_,
                     *h_device_params_, *h_kernel_params_);
}

void CudaDecoder::CopyMainQueueDataToHost() {
  CU_SAFE_CALL(cudaEventRecord(concatenated_data_ready_evt_, compute_st_));
  // The copies on copy_st will wait on compute_st_.
  CU_SAFE_CALL(cudaStreamWaitEvent(copy_st_, concatenated_data_ready_evt_, 0));
  // We need the total size of each segment on the host.
  CU_SAFE_CALL(cudaEventSynchronize(lane_offsets_ready_evt_));
  LaunchD2HCopies();

  // Making sure the previous H2H copies are done
  WaitForInitDecodingH2HCopies();
  WaitForH2HCopies();

  std::swap(h_extra_and_acoustic_cost_concat_tmp_,
            h_extra_and_acoustic_cost_concat_);
  std::swap(h_infotoken_concat_tmp_, h_infotoken_concat_);
  std::swap(h_acoustic_cost_concat_tmp_, h_acoustic_cost_concat_);
  std::swap(h_extra_prev_tokens_concat_tmp_, h_extra_prev_tokens_concat_);
  // Saving the offsets computed previously
  lanes2channels_todo_.clear();
  for (int32 ilane = 0; ilane < (nlanes_used_ + 1); ++ilane) {
    h_emitting_main_q_end_lane_offsets_[ilane] =
        h_lanes_counters_.lane(ilane)->main_q_n_emitting_tokens_lane_offset;
    h_main_q_end_lane_offsets_[ilane] =
        h_lanes_counters_.lane(ilane)->main_q_end_lane_offset;
    h_n_extra_prev_tokens_lane_offsets_[ilane] =
        h_lanes_counters_.lane(ilane)->main_q_n_extra_prev_tokens_lane_offset;
    if (ilane < nlanes_used_) {
      lanes2channels_todo_.push_back(channel_to_compute_[ilane]);
      int32 global_offset = h_lanes_counters_.lane(ilane)->main_q_global_offset;
      h_best_path_traceback_head_[ilane].index =
          global_offset + h_lanes_counters_.lane(ilane)->prev_arg_min_int_cost;
      float relative_cost = orderedIntToFloatHost(
          h_lanes_counters_.lane(ilane)->int_relative_cost);
      h_best_path_traceback_head_[ilane].relative_cost = relative_cost;
      const ChannelId ichannel = channel_to_compute_[ilane];
      ++num_frames_decoded_[ichannel];
    }
  }

  LaunchH2HCopies();
}

void CudaDecoder::LaunchD2HCopies() {
  // Last offset = total
  int32 nelements_acoustic_costs = h_lanes_counters_.lane(nlanes_used_)
                                       ->main_q_n_emitting_tokens_lane_offset;
  // Moving the d_concat to h_concat (host), async
  if (nelements_acoustic_costs > 0) {
    KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(
        h_acoustic_cost_concat_tmp_, d_acoustic_cost_concat_,
        nelements_acoustic_costs * sizeof(*d_acoustic_cost_concat_),
        cudaMemcpyDeviceToHost, copy_st_));
  }
  CU_SAFE_CALL(cudaEventRecord(d2h_copy_acoustic_evt_, copy_st_));

  int32 nelements_infotoken =
      h_lanes_counters_.lane(nlanes_used_)->main_q_end_lane_offset;
  if (nelements_infotoken > 0) {
    KALDI_DECODER_CUDA_API_CHECK_ERROR(
        cudaMemcpyAsync(h_infotoken_concat_tmp_, d_infotoken_concat_,
                        nelements_infotoken * sizeof(*d_infotoken_concat_),
                        cudaMemcpyDeviceToHost, copy_st_));
  }
  CU_SAFE_CALL(cudaEventRecord(d2h_copy_infotoken_evt_, copy_st_));
  int32 nelements_extra_prev_tokens =
      h_lanes_counters_.lane(nlanes_used_)
          ->main_q_n_extra_prev_tokens_lane_offset;
  if (nelements_extra_prev_tokens > 0) {
    KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpyAsync(
        h_extra_prev_tokens_concat_tmp_, d_extra_prev_tokens_concat_,
        nelements_extra_prev_tokens * sizeof(*d_extra_prev_tokens_concat_),
        cudaMemcpyDeviceToHost, copy_st_));
    KALDI_DECODER_CUDA_API_CHECK_ERROR(
        cudaMemcpyAsync(h_extra_and_acoustic_cost_concat_tmp_,
                        d_extra_and_acoustic_cost_concat_,
                        nelements_extra_prev_tokens *
                            sizeof(*d_extra_and_acoustic_cost_concat_),
                        cudaMemcpyDeviceToHost, copy_st_));
  }
  CU_SAFE_CALL(cudaEventRecord(d2h_copy_extra_prev_tokens_evt_, copy_st_));
}

void CudaDecoder::ConcatenateData() {
  ConcatenateLanesDataKernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_,
      h_device_params_->d_main_q_acoustic_cost, d_acoustic_cost_concat_,
      &d_lanes_counters_.lane(0)->main_q_n_emitting_tokens_lane_offset);
  ConcatenateLanesDataKernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_,
      h_device_params_->d_main_q_info, d_infotoken_concat_,
      &d_lanes_counters_.lane(0)->main_q_end_lane_offset);
  ConcatenateLanesDataKernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_,
      h_device_params_->d_main_q_extra_prev_tokens, d_extra_prev_tokens_concat_,
      &d_lanes_counters_.lane(0)->main_q_n_extra_prev_tokens_lane_offset);
  ConcatenateLanesDataKernel(
      KaldiCudaDecoderNumBlocks(nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_,
      h_device_params_->d_main_q_extra_and_acoustic_cost,
      d_extra_and_acoustic_cost_concat_,
      &d_lanes_counters_.lane(0)->main_q_n_extra_prev_tokens_lane_offset);
}

void CudaDecoder::AdvanceDecoding(
    const std::vector<ChannelId> &channels,
    std::vector<CudaDecodableInterface *> &decodables, int32 max_num_frames) {
  int nframes_to_decode = INT_MAX;
  for (int32 ilane = 0; ilane < channels.size(); ++ilane) {
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

  std::vector<std::pair<ChannelId, const BaseFloat *>> lanes_assignments;
  for (int f = 0; f < nframes_to_decode; ++f) {
    lanes_assignments.clear();
    for (int32 ilane = 0; ilane < channels.size(); ++ilane) {
      const ChannelId ichannel = channels[ilane];
      int32 iframe = num_frames_decoded_[ichannel];
      const BaseFloat *ptr = decodables[ilane]->GetLogLikelihoodsCudaPointer(iframe);
      lanes_assignments.push_back({ichannel, ptr});
    }
    AdvanceDecoding(lanes_assignments);
  }
}

void CudaDecoder::AdvanceDecoding(
    const std::vector<std::pair<ChannelId, const BaseFloat *>> &lanes_assignements) {
  if (lanes_assignements.size() == 0) return;  // nothing to do
  // Context switch : Loading the channels state in lanes

  // Looping over the frames that we will compute
  // Loglikelihoods from the acoustic model
  // Setting the loglikelihoods pointers for that frame
  std::vector<ChannelId> channels;  // TODO
  channels.reserve(lanes_assignements.size());
  for (LaneId ilane = 0; ilane < lanes_assignements.size(); ++ilane) {
    ChannelId ichannel = lanes_assignements[ilane].first;
    channels.push_back(ichannel);
    channel_to_compute_[ilane] = ichannel;
    h_lanes_counters_.lane(ilane)->loglikelihoods =
        lanes_assignements[ilane].second;
  }
  LoadChannelsStateToLanes(channels);
  KALDI_ASSERT(nlanes_used_ > 0);
  CU_SAFE_CALL(cudaMemcpyAsync(d_lanes_counters_.MutableData(),
                               h_lanes_counters_.lane(0),
                               nlanes_used_ * sizeof(*h_lanes_counters_.lane(0)),
                               cudaMemcpyHostToDevice, compute_st_));
  // compute_st_ will wait for nnet3 to complete
  CU_SAFE_CALL(cudaEventRecord(nnet3_done_evt_, cudaStreamPerThread));
  CU_SAFE_CALL(cudaStreamWaitEvent(compute_st_, nnet3_done_evt_, 0));

  // Estimating cutoff using argmin from last frame
  ResetForFrameAndEstimateCutoffKernel(
      KaldiCudaDecoderNumBlocks(1, nlanes_used_), KALDI_CUDA_DECODER_1D_BLOCK,
      compute_st_, *h_device_params_, *h_kernel_params_);
  // Reset max active status. If necessary, ApplyMaxActiveAndReduceBeam
  // will switch it back on
  compute_max_active_ = false;

  // Processing emitting arcs. We've done the preprocess stage at the end
  // of the previous frame
  ExpandArcsEmitting();
  // We'll loop until we have a small enough number of non-emitting arcs
  // in the token queue. We'll then break the loop
  for (int i = 0; i < KALDI_CUDA_DECODER_N_NON_EMITTING_MAIN_ITERATIONS; ++i) {
    // If one of the aux_q contains more than max_active_ tokens,
    // we'll reduce the beam to only keep max_active_ tokens
    ApplyMaxActiveAndReduceBeam(AUX_Q);
    // Prune the aux_q. Apply the latest beam (using the one from
    // ApplyMaxActiveAndReduceBeam if triggered)
    // move the survival tokens to the main queue
    // and do the preprocessing necessary for the next ExpandArcs
    PruneAndPreprocess();

    // "heavy duty" kernel for non-emitting. The long tail of small
    // non-emitting iterations will be done in
    // FinalizeProcessNonEmittingKernel
    ExpandArcsNonEmitting();
  }
  ApplyMaxActiveAndReduceBeam(AUX_Q);
  PruneAndPreprocess();
  // Finalizing process non emitting. Takes care of the long tail,
  // the final iterations with a small numbers of arcs. Do the work inside
  // a single CTA (per lane),
  FinalizeProcessNonEmittingKernel(KaldiCudaDecoderNumBlocks(1, nlanes_used_),
                                   KALDI_CUDA_DECODER_LARGEST_1D_BLOCK,
                                   compute_st_, *h_device_params_,
                                   *h_kernel_params_);

  // We now have our final token main queues for that frame

  // Post processing the tokens for that frame
  // - do the preprocess necessary for the next emitting expand (will
  // happen with next frame)
  // - if a state S has more than one token associated to it, generate the
  // list of those tokens
  // It allows to backtrack efficiently in GetRawLattice
  // - compute the extra costs
  PostProcessingMainQueue();

  // Waiting on previous d2h before writing on same device memory
  CU_SAFE_CALL(cudaStreamWaitEvent(compute_st_,
                                   d2h_copy_extra_prev_tokens_evt_, 0));
  // Concatenating the data that will be moved to host into large arrays
  ConcatenateData();
  // Copying the final lane counters for that frame
  CopyLaneCountersToHostSync();
  CheckOverflow();

  // Moving the data necessary for GetRawLattice/GetBestPath back to host
  // for storage
  CopyMainQueueDataToHost();

  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    const ChannelId ichannel = channel_to_compute_[ilane];
    const int32 main_q_end =
        h_lanes_counters_.lane(ilane)->main_q_narcs_and_end.y;
    // Saving frame offsets for GetRawLattice
    frame_offsets_[ichannel].push_back(frame_offsets_[ichannel].back() +
                                       main_q_end);
  }
  SaveChannelsStateFromLanes();

  // Waiting for partial path to be ready (if set)
  // They are computed async
  WaitForPartialHypotheses();
}

void CudaDecoder::WaitForPartialHypotheses() {
  if (!generate_partial_hypotheses_) return;
  while (n_partial_traceback_threads_not_done_
             .load(std::memory_order_acquire) > 0) {
    Sleep(200e-6);
  }
}

void CudaDecoder::CheckOverflow() {
  for (LaneId ilane = 0; ilane < nlanes_used_; ++ilane) {
    LaneCounters *lane_counters = h_lanes_counters_.lane(ilane);
    int32_t q_overflow = lane_counters->q_overflow;
    if (q_overflow != OVERFLOW_NONE) {
      // An overflow was prevented in a kernel
      // The algorithm can still go on but quality of the
      // result can be reduced (less tokens were generated)

      if ((q_overflow & OVERFLOW_MAIN_Q) == OVERFLOW_MAIN_Q) {
        // overflowed main_q
        KALDI_WARN << ("Preventing overflow of main_q. The quality of the"
                       " output may be reduced. Increase --main-q-capacity"
                       " and/or decrease --max-active");
      }
      if ((q_overflow & OVERFLOW_AUX_Q) == OVERFLOW_AUX_Q) {
        // overflowed aux_q
        KALDI_WARN << ("Preventing overflow of aux_q. The quality of the output"
                       " may be reduced. Increase --aux-q-capacity and/or"
                       " decrease --beam");
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
  CU_SAFE_CALL(cudaMemcpyAsync(d_lanes_counters_.MutableData(),
                               h_lanes_counters_.lane(0),
                               nlanes_used_ * sizeof(*h_lanes_counters_.lane(0)),
                               cudaMemcpyHostToDevice, compute_st_));

  auto func_main_q_end = [](const LaneCounters &c) {
    return c.main_q_narcs_and_end.y;
  };
  int32 max_main_q_end = GetMaxForAllLanes(func_main_q_end);

  // Step1 : Finding the best cost in the last token queue, with and
  // without final costs. Also saving the indexes of those min.
  GetBestCostStep1Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_, use_final_costs);

  // Step2: Now that we now what the minimum cost is, we list all tokens
  // within
  // [min_cost; min_cost+lattice_beam]
  // min_cost takes into account the final costs if use_final_costs is
  // true, AND if a final state is is present in the last token queue
  GetBestCostStep2Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_, use_final_costs);

  // Step3 : Moves some data to host. We are moving the data that couldn't
  // be moved directly in step 2, e.g. results of atomics (we don't know
  // which one is last)
  GetBestCostStep3Kernel(
      KaldiCudaDecoderNumBlocks(max_main_q_end, nlanes_used_),
      KALDI_CUDA_DECODER_1D_BLOCK, compute_st_, *h_device_params_,
      *h_kernel_params_);

  // Resetting the datastructures
  argmins->clear();
  has_reached_final->clear();
  list_finals_token_idx_and_cost->clear();
  // list_finals_token_idx_and_cost is a vector<vector<>>
  // Each channel will have its own list of tokens within [best;
  // best+lattice_beam]
  list_finals_token_idx_and_cost->resize(nlanes_used_);
  // Waiting for the copy
  CU_SAFE_CALL(cudaStreamSynchronize(compute_st_));
  for (int32 ilane = 0; ilane < nlanes_used_; ++ilane) {
    int ichannel = channels[ilane];
    KALDI_ASSERT(NumFramesDecoded(ichannel) > 0);
    int2 minarg = h_lanes_counters_.lane(ilane)->min_int_cost_and_arg;
    // Min cost in that channel last token queue
    CostType min_cost = orderedIntToFloatHost(minarg.x);
    // index of that min cost
    int32 arg = minarg.y;
    // Saving both in output
    argmins->push_back({arg, min_cost});
    // Whether or not the last token queue contains at least one
    // token associated with a final FST state
    has_reached_final->push_back(
        h_lanes_counters_.lane(ilane)->has_reached_final);
    // Number of tokens within [min_cost; min_cost+lattice_beam]
    int n_within_lattice_beam =
        h_lanes_counters_.lane(ilane)->n_within_lattice_beam;
    // Loading those tokens
    (*list_finals_token_idx_and_cost)[ilane].resize(n_within_lattice_beam);
    // Moving to output + int2float conversion
    for (int i = 0; i < n_within_lattice_beam; ++i) {
      int global_idx = h_list_final_tokens_in_main_q_.lane(ilane)[i].x;
      float cost_with_final = orderedIntToFloatHost(
          h_list_final_tokens_in_main_q_.lane(ilane)[i].y);
      (*list_finals_token_idx_and_cost)[ilane][i].first = global_idx;
      (*list_finals_token_idx_and_cost)[ilane][i].second = cost_with_final;
    }
  }

  for (LaneId ilane = 0; ilane < channels.size(); ++ilane) {
    ChannelId ichannel = channels[ilane];
    std::lock_guard<std::mutex> channel_lk(channel_lock_[ichannel]);
    h_all_argmin_cost_[ichannel] = (*argmins)[ilane];
    h_all_final_tokens_list_[ichannel].swap(
        (*list_finals_token_idx_and_cost)[ilane]);
    h_all_has_reached_final_[ichannel] = (*has_reached_final)[ilane];
  }
}

void CudaDecoder::GetBestPredecessor(int32 ichannel, int32 curr_token_idx,
                                     int32 *prev_token_idx_out,
                                     int32 *arc_idx_out) {
  KALDI_ASSERT(curr_token_idx > 0);
  KALDI_ASSERT(curr_token_idx < h_all_tokens_info_[ichannel].size());
  InfoToken token = h_all_tokens_info_[ichannel][curr_token_idx];
  // We want an arc with extra_cost == 0
  int32 arc_idx;
  TokenId prev_token_idx;
  if (token.IsUniqueTokenForStateAndFrame()) {
    // If we have only one, it is an arc with
    // extra_cost == 0
    arc_idx = token.arc_idx;
    prev_token_idx = token.prev_token;
  } else {
    // Using the first arc with extra_cost == 0
    int32 offset, size;
    std::tie(offset, size) = token.GetSameFSTStateTokensList();
    bool found_best = false;
    for (int32 i = 0; i < size; ++i) {
      KALDI_ASSERT(
          (offset + i) <
          h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel]
              .size());
      CostType arc_extra_cost =
          h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel]
                                                                 [offset + i].x;
      // Picking one arc on the best path
      // (extra_cost == 0)
      if (arc_extra_cost == 0.0f) {
        KALDI_ASSERT(
            h_all_tokens_extra_prev_tokens_[ichannel].size() ==
            h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel]
                .size());
        KALDI_ASSERT((offset + i) <
                     h_all_tokens_extra_prev_tokens_[ichannel].size());
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
  *prev_token_idx_out = prev_token_idx;
  *arc_idx_out = arc_idx;
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
    Lattice *fst_out = fst_out_vec[ilane];
    fst_out->DeleteStates();
    const ChannelId ichannel = channels[ilane];
    if (NumFramesDecoded(ichannel) == 0) continue;  // nothing to do

    const int32 token_with_best_cost = argmins_[ilane].first;
    std::unique_lock<std::mutex> channel_lk(channel_lock_[ichannel]);
    // If that token in that frame f is available, then all tokens
    // in that frame f are available
    WaitForH2HCopies();
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
      int32 prev_token_idx, arc_idx;
      GetBestPredecessor(ichannel, token_idx, &prev_token_idx, &arc_idx);
      reversed_path.push_back(arc_idx);
      token_idx = prev_token_idx;
    }

    // Building the output Lattice
    OutputLatticeState curr_state = fst_out->AddState();
    fst_out->SetStart(curr_state);

    for (int32 i = reversed_path.size() - 1; i >= 0; i--) {
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

void CudaDecoder::AddFinalTokensToLattice(
    ChannelId ichannel,
    std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
    std::unordered_map<LatticeStateInternalId, RawLatticeState>
        *curr_f_raw_lattice_state,
    Lattice *fst_out) {
  // Total number of tokens for that utterance. Used in
  // GetLatticeStateInternalId
  const int32 total_ntokens = h_all_tokens_info_[ichannel].size();
  // Reading the overall best_cost for that utterance's last frame. Was
  // set by GetBestCost
  const CostType best_cost = h_all_argmin_cost_[ichannel].second;
  // Iterating through tokens associated with a final state in the last
  // frame
  for (const auto &p : h_all_final_tokens_list_[ichannel]) {
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
    decltype(curr_f_raw_lattice_state->end()) map_it;
    bool inserted;

    // We need to create the fst_lattice_state linked to our
    // internal id in the lattice if it doesn't already exists
    // Inserts only if the key doesn't exist in the map
    std::tie(map_it, inserted) = curr_f_raw_lattice_state->insert(
        {state_internal_id, {FLT_MAX, -1, false}});

    // If we've inserted the element, it means that that state
    // didn't exist in the map Because this is a final state, we
    // need to do a bit of extra work to add the final_cost to it
    if (inserted) {
      // We want to figure out which FST state this token is
      // associated to We don't have that info anymore, it
      // wasn't transfered from the GPU We still need it for
      // final tokens, because we need to know which final
      // cost to add in the lattice. To find that original FST
      // state, we need the id of an arc going to that state,
      // then we'll look in the graph and figure out
      // next_state[arc_idx] we just need a valid arc_idx
      int32 arc_idx;
      if (final_token.IsUniqueTokenForStateAndFrame()) {
        // If unique, we can directly use this arc_idx.
        arc_idx = final_token.arc_idx;
      } else {
        // If we have multiple tokens associated to that FST state, just pick
        // the first one from the list.
        int32 offset, size;
        std::tie(offset, size) = final_token.GetSameFSTStateTokensList();
        KALDI_ASSERT(!h_all_tokens_extra_prev_tokens_.empty());
        KALDI_ASSERT(ichannel < h_all_tokens_extra_prev_tokens_.size());
        KALDI_ASSERT(!h_all_tokens_extra_prev_tokens_[ichannel].empty());
        KALDI_ASSERT(offset < h_all_tokens_extra_prev_tokens_[ichannel].size());
        InfoToken prev_token =
            h_all_tokens_extra_prev_tokens_[ichannel][offset];
        arc_idx = prev_token.arc_idx;
      }
      // Creating the state associated with our internal id in
      // the lattice
      OutputLatticeState fst_lattice_final_state = fst_out->AddState();
      map_it->second.fst_lattice_state = fst_lattice_final_state;
      q_curr_frame_todo->push_back({final_token_idx, final_token});

      if (h_all_has_reached_final_[ichannel]) {
        // If we have reached final states, adding the
        // final cost We now have a valid arc_idx. We
        // can read the FST state
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

void CudaDecoder::AddArcToLattice(
    int32 list_arc_idx, TokenId list_prev_token_idx, InfoToken list_prev_token,
    int32 curr_frame_offset, CostType acoustic_cost,
    CostType this_arc_prev_token_extra_cost,
    LatticeStateInternalId src_state_internal_id,
    OutputLatticeState fst_lattice_start,
    OutputLatticeState to_fst_lattice_state,
    std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
    std::vector<std::pair<TokenId, InfoToken>> *q_prev_frame_todo,
    std::unordered_map<LatticeStateInternalId, RawLatticeState>
        *curr_f_raw_lattice_state,
    std::unordered_map<LatticeStateInternalId, RawLatticeState>
        *prev_f_raw_lattice_state,
    std::unordered_set<int32> *f_arc_idx_added, Lattice *fst_out,
    bool *must_replay_frame) {
  // We will now add this arc to the output lattice
  // We know the destination state of the arc (to_fst_lattice_state)
  // We need to figure out its source
  // And propagate the extra cost from the destination to the source of
  // that arc (we go backward)
  OutputLatticeState from_fst_lattice_state;
  // Having the predecessor in the previous frame
  // <=> that token is associated to an emiting arc
  bool emitting = (list_prev_token_idx < curr_frame_offset);
  // Checking if the source of that arc is the start state (original state
  // at the beginning of the decode)
  if (list_prev_token_idx != 0) {
    // Selecting the right map
    // - emitting arc -> previous frame map
    // - non emitting arc -> same frame map
    auto *extra_cost_map =
        emitting ? prev_f_raw_lattice_state : curr_f_raw_lattice_state;
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
      auto *todo_list = emitting ? q_prev_frame_todo : q_curr_frame_todo;
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
      // and if the state that we're writing to was already
      // closed, then we need to replay that frame. if the
      // source state is already closed it means we've read
      // its extra_cost value. Now we're writing again to it.
      // We have to do the first read again, to get the
      // updated value that's why we're replaying that frame
      // (between frames everything is in topological order)
      if (diff > extra_cost_min_delta_ && from_map_it->second.is_state_closed) {
        *must_replay_frame = true;
      }
      prev_token_extra_cost = this_arc_prev_token_extra_cost;
      from_map_it->second.token_extra_cost = prev_token_extra_cost;
    }

    // Reading the OutputLatticeState of the source state in the
    // output lattice
    from_fst_lattice_state = from_map_it->second.fst_lattice_state;
  } else {
    from_fst_lattice_state =
        fst_lattice_start;  // we simply link it to the source
  }

  // Checking if it's the first time we insert an arc with that
  // arc_idx for that frame.
  // If we're replaying that frame, we don't want duplicates
  bool is_this_arc_new = f_arc_idx_added->insert(list_arc_idx).second;
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

void CudaDecoder::GetTokenRawLatticeData(
    TokenId token_idx, InfoToken token, int32 total_ntokens,
    std::unordered_map<LatticeStateInternalId, RawLatticeState>
        *curr_f_raw_lattice_state,
    CostType *token_extra_cost, OutputLatticeState *to_fst_lattice_state) {
  LatticeStateInternalId next_state_internal_id =
      GetLatticeStateInternalId(total_ntokens, token_idx, token);
  auto to_map_it = curr_f_raw_lattice_state->find(next_state_internal_id);
  // We know this token exists in the output lattice (because it's in
  // q_curr_frame_todo_)
  KALDI_ASSERT(to_map_it != curr_f_raw_lattice_state->end());

  *token_extra_cost = to_map_it->second.token_extra_cost;
  *to_fst_lattice_state = to_map_it->second.fst_lattice_state;

  // We read the extra cost from lattice_next_state
  // We are now closing the state. If we write to it again, we will have
  // to replay that frame
  // (so that the latest extra_cost value is read)
  to_map_it->second.is_state_closed = true;
}

void CudaDecoder::GetSameFSTStateTokenList(
    ChannelId ichannel, InfoToken &token, InfoToken **tok_beg,
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

void CudaDecoder::SwapPrevAndCurrLatticeMap(
    int32 iframe, bool dbg_found_best_path,
    std::vector<std::pair<TokenId, InfoToken>> *q_curr_frame_todo,
    std::vector<std::pair<TokenId, InfoToken>> *q_prev_frame_todo,
    std::unordered_map<LatticeStateInternalId, RawLatticeState>
        *curr_f_raw_lattice_state,
    std::unordered_map<LatticeStateInternalId, RawLatticeState>
        *prev_f_raw_lattice_state,
    std::unordered_set<int32> *f_arc_idx_added) {
  q_prev_frame_todo->swap(*q_curr_frame_todo);
  q_prev_frame_todo->clear();
  prev_f_raw_lattice_state->swap(*curr_f_raw_lattice_state);
  prev_f_raw_lattice_state->clear();
  f_arc_idx_added->clear();

  KALDI_ASSERT(q_prev_frame_todo->empty());
  if (iframe > 0) {
    KALDI_ASSERT(!q_curr_frame_todo->empty());
    if (!dbg_found_best_path) {
      KALDI_WARN << "Warning didn't find exact best path in "
                    "GetRawLattice";
    }
  }
}

void CudaDecoder::WaitForH2HCopies() {
  Timer timer;
  std::unique_lock<std::mutex> lk(n_h2h_task_not_done_mutex_);
  h2h_done_.wait(lk, [this] { return (n_h2h_task_not_done_ == 0); });
}

void CudaDecoder::WaitForInitDecodingH2HCopies() {
  std::unique_lock<std::mutex> lk(n_init_decoding_h2h_task_not_done_mutex_);
  init_decoding_h2h_done_.wait(
      lk, [this] { return (n_init_decoding_h2h_task_not_done_ == 0); });
}

void CudaDecoder::FillWithNonEmptyChannels(
    const std::vector<ChannelId> &channels,
    std::vector<ChannelId> *out_nonempty_channels) {
  out_nonempty_channels->clear();
  for (ChannelId ichannel : channels) {
    if (NumFramesDecoded(ichannel) > 0)
      out_nonempty_channels->push_back(ichannel);
  }
}

void CudaDecoder::PrepareForGetRawLattice(
    const std::vector<ChannelId> &raw_channels, bool use_final_probs) {
  FillWithNonEmptyChannels(raw_channels, &nonempty_channels_);
  GetBestCost(nonempty_channels_, use_final_probs, &argmins_,
              &list_finals_token_idx_and_cost_, &has_reached_final_);
}

void CudaDecoder::ConcurrentGetRawLatticeSingleChannel(const ChannelId ichannel,
                                                       Lattice *fst_out) {
  nvtxRangePushA("GetRawLatticeOneChannel");
  // Allocating the datastructures that we need

  // [prev|curr]_f_raw_lattice_state
  // Used to get information about a lattice state (i.e. a (iframe,
  // fst_state) pair) using its LatticeStateInternalId (its ID inside of
  // the decoder) It gives us the OutputLatticeState (its ID in the output
  // lattice) alongside with the extra_cost of that state in the lattice
  // Those maps are used to build the external lattice using what we know
  // internally
  // Using one map per frame. We always know to which frame a token
  // belongs. Using one big map slows everything down
  std::unordered_map<LatticeStateInternalId, RawLatticeState>
      prev_f_raw_lattice_state, curr_f_raw_lattice_state;
  // We want the unicity of each arc_idx for one frame. Important because
  // we can replay a frame (and possibly add multiple time the same arc)
  std::unordered_set<int32> f_arc_idx_added;
  // When backtracking, we read tokens in the current frame (in
  // q_curr_frame_todo_),
  // we backtrack the associated arc, and we add the predecessor either to
  // q_curr_frame_todo_ (non-emitting arc, same frame)
  // or q_prev_frame_todo_ (emitting arc, source in previous frame)
  std::vector<std::pair<TokenId, InfoToken>> q_curr_frame_todo;
  std::vector<std::pair<TokenId, InfoToken>> q_prev_frame_todo;

  // We need to lock the channel to read argmin
  TokenId best_cost_idx;
  {
    std::lock_guard<std::mutex> channel_lk(channel_lock_[ichannel]);
    h_all_tokens_info_[ichannel].shrink_to_fit();
    h_all_tokens_acoustic_cost_[ichannel].shrink_to_fit();
    h_all_tokens_extra_prev_tokens_[ichannel].shrink_to_fit();
    h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_[ichannel]
        .shrink_to_fit();
    best_cost_idx = h_all_argmin_cost_[ichannel].first;
  }
  fst_out->DeleteStates();  // reset out lattice
  const int32 nframes = NumFramesDecoded(ichannel);
  // If no frames were decoded, leaving the lattice empty and return
  if (nframes == 0) return;

  KALDI_ASSERT(
      "You need to call PrepareForGetRawLattice before "
      "ConcurrentGetRawLatticeSingleChannel" &&
      best_cost_idx >= 0);

  // Making sure that this token is available for this channel.
  // We're going to read storage data from this channel. Locking it
  // If that token in that frame f is available, then all tokens in that
  // frame f are available
  WaitForH2HCopies();
  std::unique_lock<std::mutex> channel_lk(channel_lock_[ichannel]);

  // Total number of tokens generated by the utterance on channel ichannel
  const int32 total_ntokens = h_all_tokens_info_[ichannel].size();

  // Preparing output lattice
  // The start state has to be 0 (cf some asserts somewhere else in Kaldi)
  // Adding it now
  OutputLatticeState fst_lattice_start = fst_out->AddState();
  fst_out->SetStart(fst_lattice_start);

  // Adding the best tokens returned by GetBestCost to the lattice
  // We also add them to q_curr_frame_todo, and we'll backtrack from there
  AddFinalTokensToLattice(ichannel, &q_curr_frame_todo,
                          &curr_f_raw_lattice_state, fst_out);

  // We're now going to backtrack frame by frame
  // For each frame we're going to process tokens that need to be inserted
  // into the output lattice
  // and add their predecessors to the todo list
  // iframe == -1 contains the start state and the first non emitting
  // tokens. It is not linked to a real frame
  for (int32 iframe = nframes - 1; iframe >= -1; --iframe) {
    // Tokens for the current frame were inserted after this offset
    // in the token list
    const int32 curr_frame_offset =
        (iframe >= 0) ? frame_offsets_[ichannel][iframe] : 0;

    // bool must_replay_frame
    // In some cases we can update an extra_cost that has already
    // been used For instance we process arcs in that order : 1) a
    // -> b, which updates extra_cost[b] using extra_cost[a] 2) c ->
    // a, which updates extra-cost[a] (using extra_cost[c]) because
    // the arcs were not considered in topological order, we need to
    // run
    // again the step 1,
    // to get the correct extra_cost[b] (using the latest
    // extra_cost[a]) However, we only re-run the step 1 if the
    // value extra_cost[a] has changed more than
    // extra_cost_min_delta_
    bool must_replay_frame;

    // dbg_found_best_path is used in an useful assert, making sure
    // the best path is still there for each frame if something went
    // wrong in the kernels, it's not likely we respect that
    // property out of luck
    bool dbg_found_best_path = false;
    do {
      must_replay_frame = false;
      // Reading something to do. We are pushing stuff back in
      // q_curr_frame_todo while reading it,
      // so it's important to always read
      // q_curr_frame_todo_.size() directly not using a queue,
      // because we may need to recompute the frame (if
      // must_replay_frame is true)
      for (int32 u = 0; u < q_curr_frame_todo.size(); ++u) {
        TokenId token_idx;
        InfoToken token;
        std::tie(token_idx, token) = q_curr_frame_todo[u];
        KALDI_ASSERT(token_idx >= curr_frame_offset);
        CostType token_extra_cost;
        StateId to_fst_lattice_state;
        // Loading the current extra_cost of that token
        // + its associated state in the lattice
        GetTokenRawLatticeData(token_idx, token, total_ntokens,
                               &curr_f_raw_lattice_state, &token_extra_cost,
                               &to_fst_lattice_state);
        dbg_found_best_path |= (token_extra_cost == 0.0f);

        InfoToken *tok_beg;
        float2 *extra_extra_and_acoustic_cost_beg;
        int32 nsamestate;
        // Getting the list of the tokens linked to the
        // same FST state, in the same frame In the GPU
        // decoder a token is linked to a single arc,
        // but we can generate multiple token for a same
        // fst_nextstate in the same frame. In the CPU
        // decoder we would use the forward_links list
        // to store everything in the same metatoken
        // GetSameFSTStateTokenList returns the list of
        // tokens linked to the same FST state than
        // token (in the current frame)
        GetSameFSTStateTokenList(ichannel, token, &tok_beg,
                                 &extra_extra_and_acoustic_cost_beg,
                                 &nsamestate);

        // dbg_found_zero used for debugging. For each
        // FST state, we have a token with the best cost
        // for that FST state that token has an
        // extra_cost of 0.0f. This is a sanity check
        bool dbg_found_zero = false;
        for (int32 iprev = 0; iprev < nsamestate; ++iprev) {
          InfoToken list_prev_token;
          CostType acoustic_cost, this_arc_prev_token_extra_cost;
          bool keep_arc;
          LatticeStateInternalId src_state_internal_id;
          InfoToken list_token = tok_beg[iprev];
          int32 list_prev_token_idx = list_token.prev_token;
          int32 list_arc_idx = list_token.arc_idx;

          ConsiderTokenForLattice(
              ichannel, iprev, total_ntokens, token_idx, fst_lattice_start,
              tok_beg, extra_extra_and_acoustic_cost_beg, token_extra_cost,
              list_prev_token_idx, list_arc_idx, &list_prev_token,
              &this_arc_prev_token_extra_cost, &acoustic_cost,
              &src_state_internal_id, &keep_arc, &dbg_found_zero);

          if (keep_arc)
            AddArcToLattice(list_arc_idx, list_prev_token_idx, list_prev_token,
                            curr_frame_offset, acoustic_cost,
                            this_arc_prev_token_extra_cost,
                            src_state_internal_id, fst_lattice_start,
                            to_fst_lattice_state, &q_curr_frame_todo,
                            &q_prev_frame_todo, &curr_f_raw_lattice_state,
                            &prev_f_raw_lattice_state, &f_arc_idx_added,
                            fst_out, &must_replay_frame);
        }
        KALDI_ASSERT(dbg_found_zero);
      }

      if (must_replay_frame) {
        // We need to replay the frame. Because all
        // states will be read again, we can reopen them
        // (and they will be closed again when being
        // read from again)
        for (auto it = curr_f_raw_lattice_state.begin();
             it != curr_f_raw_lattice_state.end(); ++it) {
          it->second.is_state_closed = false;
        }
      }
    } while (must_replay_frame);

    // Done processing this frame. Swap the datastructures, move on
    // to previous frame (we go --iframe)
    SwapPrevAndCurrLatticeMap(iframe, dbg_found_best_path, &q_curr_frame_todo,
                              &q_prev_frame_todo, &curr_f_raw_lattice_state,
                              &prev_f_raw_lattice_state, &f_arc_idx_added);
  }
  nvtxRangePop();
}

void CudaDecoder::GetRawLattice(const std::vector<ChannelId> &channels,
                                std::vector<Lattice *> &fst_out_vec,
                                bool use_final_probs) {
  KALDI_ASSERT(channels.size() == fst_out_vec.size());
  // Getting the list of the best costs in the lastest token queue.
  // all costs within [best;best+lattice_beam]
  PrepareForGetRawLattice(channels, use_final_probs);
  for (int32 ilane = 0; ilane < channels.size(); ++ilane) {
    const ChannelId ichannel = channels[ilane];
    Lattice *fst_out = fst_out_vec[ilane];
    ConcurrentGetRawLatticeSingleChannel(ichannel, fst_out);
  }
}

void CudaDecoder::SetChannelsInKernelParams(
    const std::vector<ChannelId> &channels) {
  KALDI_ASSERT(channels.size() <= nchannels_);
  KALDI_ASSERT(channels.size() <= nlanes_);
  for (LaneId lane_id = 0; lane_id < channels.size(); ++lane_id)
    channel_to_compute_[lane_id] = channels[lane_id];
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

  // We need that because we need to be able to do the scan in one pass in
  // the kernel update_beam_using_histogram_kernel
  KALDI_COMPILE_TIME_ASSERT(
    KALDI_CUDA_DECODER_HISTO_NBINS < KALDI_CUDA_DECODER_1D_BLOCK);
  KALDI_COMPILE_TIME_ASSERT(KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS > 0);
}

void CudaDecoder::LaunchH2HCopies() {
  // Each H2H copy counter
  n_acoustic_h2h_copies_todo_.store(nlanes_used_ - 1);
  n_infotoken_h2h_copies_todo_.store(nlanes_used_ - 1);
  n_extra_prev_tokens_h2h_copies_todo_.store(nlanes_used_ - 1);
  if (partial_traceback_) {
    n_partial_traceback_threads_todo_.store(nlanes_used_ - 1);
    n_partial_traceback_threads_not_done_.store(thread_pool_ ? n_threads_used_
                                                             : 1);
  }
  {
    std::lock_guard<std::mutex> n_h2h_not_done_lk(n_h2h_task_not_done_mutex_);
    n_h2h_task_not_done_ += thread_pool_ ? n_threads_used_ : 1;
  }
  {
    std::lock_guard<std::mutex> n_h2h_todo_lk(n_h2h_main_task_todo_mutex_);
    n_h2h_main_task_todo_ = thread_pool_ ? n_threads_used_ : 1;
  }

  // Either do the copy locally or send it to the threadpool
  if (thread_pool_) {
    n_h2h_main_task_todo_cv_.notify_all();
  } else {
    ComputeH2HCopies();
  }
}

void CudaDecoder::ComputeH2HCopiesCPUWorker() {
  // Run by a dedicated CPU thread
  while (h2h_threads_running_) {
    ComputeH2HCopies();
  }
}

void CudaDecoder::GeneratePartialPath(LaneId ilane, ChannelId ichannel) {
  // Partial hypothesis
  // We use the (n-1) frame head
  auto prev_best_path_traceback_head =
      h_all_channels_prev_best_path_traceback_head_[ichannel];
  if (!prev_best_path_traceback_head.IsSet()) return;  // nothing to do
  int curr_token_idx = prev_best_path_traceback_head.index;
  if (curr_token_idx > 0) {
    // Locking, we need the host channel data to use GetBestPredecessor
    std::lock_guard<std::mutex> channel_lk(channel_lock_[ichannel]);

    int prev_token_idx;
    int arc_idx;
    GetBestPredecessor(ichannel, curr_token_idx, &prev_token_idx, &arc_idx);

    std::list<PartialPathArc> &partial_hypotheses =
        h_all_channels_partial_hypotheses_[ichannel];

    // Adding that link at the end of the partial path
    partial_hypotheses.emplace_back(curr_token_idx, arc_idx);
    // If this is the first link, we don't have to check that we're still on the
    // same best path than before
    if (partial_hypotheses.size() == 1) return;

    // Backtracking until we reconnect with our stored partial path
    // using prev(2):
    // 1. last valid element, the one we've just added
    // 2. the one before that, the one we need to check
    auto it = std::prev(partial_hypotheses.end(), 2);

    // The new partial best path is not directly to the previous partial
    // best path We need to backtrack until we reconnect with the previous
    // partial best path (or until we reach the root node)

    while (true) {
      int32 stored_prev_token_idx = it->token_idx;
      if (stored_prev_token_idx == prev_token_idx)
        break;  // no need to rewrite existing partial path
      curr_token_idx = prev_token_idx;
      GetBestPredecessor(ichannel, curr_token_idx, &prev_token_idx, &arc_idx);
      it->token_idx = curr_token_idx;
      it->arc_idx = arc_idx;
      it->olabel = -1;
      it->substring_end = -1;

      if (prev_token_idx == 0) break;
      if (it == partial_hypotheses.begin()) {
        // Our new path is longer than the previous one
        // Adding some elts
        partial_hypotheses.emplace_front();  // default for now, it will be set
                                             // on next iteration
      }
      --it;
    }

    if (prev_token_idx == 0) {
      // We've reached the beginning, we need to purge any elts
      partial_hypotheses.erase(partial_hypotheses.begin(), it);
    }
  }
}

void CudaDecoder::EndpointDetected(LaneId ilane, ChannelId ichannel) {
  // We use the (n-1) frame head
  auto prev_best_path_traceback_head =
      h_all_channels_prev_best_path_traceback_head_[ichannel];
  if (!prev_best_path_traceback_head.IsSet()) return;  // nothing to do

  CostType relative_cost = prev_best_path_traceback_head.relative_cost;

  int num_frames_decoded =
      num_frames_decoded_[ichannel] - 1;  // we use the n-1 frame head

  // Count silence frames
  std::list<PartialPathArc> &partial_hypotheses_internal =
      h_all_channels_partial_hypotheses_[ichannel];
  int num_silence_frames = 0;
  for (auto it = partial_hypotheses_internal.rbegin();
       it != partial_hypotheses_internal.rend(); ++it) {
    int arc_idx = it->arc_idx;
    int ilabel = fst_.h_arc_id_ilabels_[arc_idx];
    // If not a silence phone, exit
    // we could cache this boolean
    if (silence_phones_.find(ilabel) == silence_phones_.end()) break;
    ++num_silence_frames;
  }
  bool end_point = kaldi::EndpointDetected(
      config_.endpointing_config, num_frames_decoded, num_silence_frames,
      frame_shift_seconds_, relative_cost);

  h_all_channels_endpoint_detected_[ichannel] = end_point;
}

void CudaDecoder::BuildPartialHypothesisOutput(
    ChannelId ichannel,
    std::stack<std::pair<int, PartialPathArc *>> *traceback_buffer_) {
  // We assume that we own the channel lock
  std::list<PartialPathArc> &partial_hypotheses_internal =
      h_all_channels_partial_hypotheses_[ichannel];
  PartialHypothesis &out = h_all_channels_partial_hypotheses_out_[ichannel];
  // We should only append one word when not backtrack was done in
  // GeneratePartialPath

  KALDI_ASSERT(traceback_buffer_->empty());
  int cut_str_at = 0;
  for (auto link = partial_hypotheses_internal.rbegin();
       link != partial_hypotheses_internal.rend(); ++link) {
    int arc_idx = link->arc_idx;
    // First time we use this kink, find out the olabel
    if (link->olabel == -1) link->olabel = fst_.h_arc_olabels_[arc_idx];
    int olabel = link->olabel;
    if (olabel == 0) continue;

    int substring_end = link->substring_end;
    if (substring_end != -1) {
      // We've reconnected to a valid previous substring
      cut_str_at = substring_end;
      break;
    }
    traceback_buffer_->push({olabel, &(*link)});
  }

  std::string &str = out.out_str;
  // Cutting the previous string if necessary
  str.resize(cut_str_at);

  if (word_syms_) {
    while (!traceback_buffer_->empty()) {
      int olabel;
      PartialPathArc *link;
      std::tie(olabel, link) = traceback_buffer_->top();
      traceback_buffer_->pop();

      if (!str.empty()) str.append(" ");
      str.append(word_syms_->Find(olabel));
      link->substring_end = str.size();
    }
  }
}

void CudaDecoder::ComputeH2HCopies() {
  // Waiting for either something to do or the instruction to stop the
  // threads
  {
    std::unique_lock<std::mutex> n_h2h_lk(n_h2h_main_task_todo_mutex_);
    n_h2h_main_task_todo_cv_.wait(n_h2h_lk, [this] {
      return !h2h_threads_running_ || (n_h2h_main_task_todo_ > 0);
    });
    --n_h2h_main_task_todo_;
  }
  // If we are done, stop the wait and return now.
  // ComputeH2HCopiesCPUWorker will also return, stopping the thread
  if (!h2h_threads_running_) return;

  int32 ilane;
  if (partial_traceback_) {
    std::stack<std::pair<int, PartialPathArc *>> traceback_buffer_;
    while ((ilane = n_partial_traceback_threads_todo_.fetch_sub(1)) >= 0) {
      int32 ichannel = lanes2channels_todo_[ilane];
      GeneratePartialPath(ilane, ichannel);
      if (generate_partial_hypotheses_)
        BuildPartialHypothesisOutput(ichannel, &traceback_buffer_);
      if (endpointing_) EndpointDetected(ilane, ichannel);
      h_all_channels_prev_best_path_traceback_head_[ichannel] =
          h_best_path_traceback_head_[ilane];
    }
    n_partial_traceback_threads_not_done_.fetch_sub(1,
                                                    std::memory_order_release);
  }
  // Waiting for the D2H copies. This is threadsafe
  // Step 1: acoustic costs
  CU_SAFE_CALL(cudaEventSynchronize(d2h_copy_acoustic_evt_));
  while ((ilane = n_acoustic_h2h_copies_todo_.fetch_sub(1)) >= 0) {
    int32 ichannel = lanes2channels_todo_[ilane];
    // Lock Channel
    std::lock_guard<std::mutex> channel_lk(channel_lock_[ichannel]);
    MoveConcatenatedCopyToVector(
        ilane, ichannel, h_emitting_main_q_end_lane_offsets_,
        h_acoustic_cost_concat_, &h_all_tokens_acoustic_cost_);
    // Adding 0.0f acoustic_costs for non-emittings
    int32 main_q_end = h_main_q_end_lane_offsets_[ilane + 1] -
                       h_main_q_end_lane_offsets_[ilane];
    int32 ntokens_emitting = h_emitting_main_q_end_lane_offsets_[ilane + 1] -
                             h_emitting_main_q_end_lane_offsets_[ilane];
    int32 ntokens_nonemitting = main_q_end - ntokens_emitting;
    auto &vec = h_all_tokens_acoustic_cost_[ichannel];
    vec.insert(vec.end(), ntokens_nonemitting, 0.0f);
  }

  // Step 2: infotoken
  CU_SAFE_CALL(cudaEventSynchronize(d2h_copy_infotoken_evt_));
  while ((ilane = n_infotoken_h2h_copies_todo_.fetch_sub(1)) >= 0) {
    int32 ichannel = lanes2channels_todo_[ilane];
    // Lock Channel
    std::lock_guard<std::mutex> channel_lk(channel_lock_[ichannel]);
    MoveConcatenatedCopyToVector(ilane, ichannel, h_main_q_end_lane_offsets_,
                                 h_infotoken_concat_, &h_all_tokens_info_);
  }

  // Step 3:
  // - extra prev tokens
  // - partial path and endpointing
  CU_SAFE_CALL(cudaEventSynchronize(d2h_copy_extra_prev_tokens_evt_));
  while ((ilane = n_extra_prev_tokens_h2h_copies_todo_.fetch_sub(1)) >= 0) {
    int32 ichannel = lanes2channels_todo_[ilane];
    // Lock Channel
    std::lock_guard<std::mutex> channel_lk(channel_lock_[ichannel]);
    MoveConcatenatedCopyToVector(
        ilane, ichannel, h_n_extra_prev_tokens_lane_offsets_,
        h_extra_prev_tokens_concat_, &h_all_tokens_extra_prev_tokens_);
    MoveConcatenatedCopyToVector(
        ilane, ichannel, h_n_extra_prev_tokens_lane_offsets_,
        h_extra_and_acoustic_cost_concat_,
        &h_all_tokens_extra_prev_tokens_extra_and_acoustic_cost_);
  }

  // If we're the last cpu thread to complete the current tasks, notify
  // the main thread
  bool all_done;
  {
    std::lock_guard<std::mutex> lk_not_done(n_h2h_task_not_done_mutex_);
    all_done = (--n_h2h_task_not_done_ == 0);
  }
  if (all_done) {
    h2h_done_.notify_all();
  }
}

void CudaDecoder::SetThreadPoolAndStartCPUWorkers(ThreadPoolLight *thread_pool,
                                                  int32 nworkers) {
  KALDI_ASSERT(nworkers > 0);
  n_threads_used_ = nworkers;
  thread_pool_ = thread_pool;
  for (int32 i = 0; i < nworkers; ++i)
    cpu_dedicated_threads_.emplace_back(&CudaDecoder::ComputeH2HCopiesCPUWorker,
                                        this);
}

}  // namespace cuda_decoder
}  // namespace kaldi

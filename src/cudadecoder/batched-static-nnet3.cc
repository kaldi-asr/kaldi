// cudadecoder/batched-static-nnet3.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
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

#include "cudadecoder/batched-static-nnet3.h"

#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace cuda_decoder {

void BatchedStaticNnet3::ReadParametersFromModelAndConfig() {
  input_frames_per_chunk_ = config_.compute_opts.frames_per_chunk;
  int32 nnet_left_context, nnet_right_context;
  nnet3::ComputeSimpleNnetContext(am_nnet_.GetNnet(), &nnet_left_context,
                                  &nnet_right_context);
  total_nnet_left_context_ =
      nnet_left_context + config_.compute_opts.extra_left_context;
  total_nnet_right_context_ =
      nnet_right_context + config_.compute_opts.extra_right_context;
  total_nnet_context_ = total_nnet_left_context_ + total_nnet_right_context_;
  subsampling_factor_ = config_.compute_opts.frame_subsampling_factor,
  input_frames_per_chunk_ = config_.compute_opts.frames_per_chunk;
  input_frames_per_chunk_with_context_ = input_frames_per_chunk_ +
                                         total_nnet_left_context_ +
                                         total_nnet_right_context_;
  output_frames_per_chunk_ =
      (subsampling_factor_ - 1 + input_frames_per_chunk_) / subsampling_factor_;
  KALDI_ASSERT(output_frames_per_chunk_ > 0);

  input_dim_ = am_nnet_.InputDim();
  if (has_ivector_) ivector_dim_ = am_nnet_.IvectorDim();
}

void BatchedStaticNnet3::PresetKernelParams() {
  //   context_switch_kernel_params_.d_all_new_features; <- To be set when
  //   called
  context_switch_kernel_params_.d_batch_slot_assignement =
      d_batch_slot_assignement_;
  context_switch_kernel_params_.d_all_context_frames =
      d_all_context_frames_.Data();
  context_switch_kernel_params_.d_all_context_frames_frame_stride =
      d_all_context_frames_.Stride();
  context_switch_kernel_params_.d_all_context_frames_channel_stride =
      d_all_context_frames_.Stride() * total_nnet_context_;
  // context_switch_kernel_params_.d_batch_with_context = <- To be set
  // when called
  //      d_batch_with_context_.Data();
  //  context_switch_kernel_params_.batch_size; <- To be set when called
  // context_switch_kernel_params_.d_all_new_features_stride = <- To be
  // set when called

  context_switch_kernel_params_.input_dim = input_dim_;
  context_switch_kernel_params_.ivector_dim = ivector_dim_;
  context_switch_kernel_params_.total_nnet_context = total_nnet_context_;
  context_switch_kernel_params_.total_nnet_left_context =
      total_nnet_left_context_;
  context_switch_kernel_params_.total_nnet_right_context =
      total_nnet_right_context_;
  context_switch_kernel_params_.input_frames_per_chunk_with_context =
      input_frames_per_chunk_with_context_;
}

void BatchedStaticNnet3::Allocate() {
  cudaEventCreate(&batch_slot_assignement_copy_evt_);
  d_all_context_frames_.Resize(nchannels_ * total_nnet_context_, input_dim_);
  d_batch_with_context_.Resize(
      max_batch_size_ * input_frames_per_chunk_with_context_, input_dim_);
  if (has_ivector_) d_batch_ivectors_.Resize(max_batch_size_, ivector_dim_);
  cudaMalloc(&d_batch_slot_assignement_,
             max_batch_size_ * sizeof(*d_batch_slot_assignement_));
  cudaMallocHost(&h_batch_slot_assignement_,
                 max_batch_size_ * sizeof(*h_batch_slot_assignement_));
  channel_n_frames_in_context_.resize(nchannels_, -1);
  st_ = cudaStreamPerThread;
  PresetKernelParams();
}

void BatchedStaticNnet3::Deallocate() {
  CU_SAFE_CALL(cudaFreeHost(h_batch_slot_assignement_));
  CU_SAFE_CALL(cudaFree(d_batch_slot_assignement_));
  CU_SAFE_CALL(cudaEventDestroy(batch_slot_assignement_copy_evt_));
}

void BatchedStaticNnet3::CompileNnet3() {
  SetComputationRequest();
  config_.compute_opts.compiler_config.cache_capacity +=
      max_batch_size_ * input_frames_per_chunk_;
  compiler_.reset(new nnet3::CachingOptimizingCompiler(
      am_nnet_.GetNnet(), config_.compute_opts.compiler_config));
  computation_ = compiler_->Compile(request_);
}

void BatchedStaticNnet3::SetComputationRequest() {
  request_.need_model_derivative = false;
  request_.store_component_stats = false;
  request_.inputs.reserve(2);

  int32 num_input_frames = input_frames_per_chunk_ + total_nnet_left_context_ +
                           total_nnet_right_context_;
  int32 first_input_t = 0 - total_nnet_left_context_;
  int32 num_output_frames = output_frames_per_chunk_;
  int32 output_t_stride = subsampling_factor_;

  std::vector<nnet3::Index> input_indexes, ivector_indexes, output_indexes;
  input_indexes.reserve(nnet3_batch_size_ * num_input_frames);
  output_indexes.reserve(nnet3_batch_size_ * num_output_frames);
  if (has_ivector_) ivector_indexes.reserve(nnet3_batch_size_);
  for (int32 n = 0; n < nnet3_batch_size_; n++) {
    for (int32 t = first_input_t; t < first_input_t + num_input_frames; t++) {
      input_indexes.push_back(nnet3::Index(n, t, 0));
    }
    if (config_.has_ivector) ivector_indexes.push_back(nnet3::Index(n, 0, 0));
    for (int32 t = 0; t < num_output_frames; t++)
      output_indexes.push_back(nnet3::Index(n, t * output_t_stride, 0));
  }
  request_.inputs.push_back(nnet3::IoSpecification("input", input_indexes));
  if (has_ivector_)
    request_.inputs.push_back(
        nnet3::IoSpecification("ivector", ivector_indexes));
  request_.outputs.push_back(nnet3::IoSpecification("output", output_indexes));
}

void BatchedStaticNnet3::BatchContextSwitch(
    const std::vector<int> &channels,
    const std::vector<BaseFloat *> &d_features, const int features_frame_stride,
    const std::vector<BaseFloat *> &d_ivectors,
    const std::vector<int> &n_input_frames_valid, bool flush_eos_context,
    std::vector<int> *n_output_frames_valid) {
  int batch_size = channels.size();

  // AcceptInput destroys input, resizing
  d_batch_with_context_.Resize(
      max_batch_size_ * input_frames_per_chunk_with_context_, input_dim_);
  if (has_ivector_) d_batch_ivectors_.Resize(max_batch_size_, ivector_dim_);

  n_output_frames_valid->resize(batch_size);

  cudaEventSynchronize(
      batch_slot_assignement_copy_evt_);  // reusing same pinned memory
  for (int i = 0; i < channels.size(); ++i) {
    int channel = channels[i];
    int nframes_in_context = channel_n_frames_in_context_[channel];
    int ninput_frames = n_input_frames_valid[i];

    KALDI_ASSERT(ninput_frames <= input_frames_per_chunk_);
    h_batch_slot_assignement_[i].d_features = d_features[i];
    h_batch_slot_assignement_[i].d_ivectors =
        has_ivector_ ? d_ivectors[i] : NULL;
    h_batch_slot_assignement_[i].ichannel = channel;
    h_batch_slot_assignement_[i].n_frames_already_in_context =
        nframes_in_context;
    h_batch_slot_assignement_[i].n_new_frames = ninput_frames;

    // Left context will be generated as necessary (copying first
    // frame) However we must have a full right context to start
    // decoding frames
    KALDI_ASSERT(!flush_eos_context || ninput_frames == 0);
    int nframes_in_batch = ninput_frames;
    if (nframes_in_context == 0)
      nframes_in_batch += total_nnet_left_context_;  // using frame0 as left
                                                     // context
    else
      nframes_in_batch += nframes_in_context;
    if (flush_eos_context)
      nframes_in_batch += total_nnet_right_context_;  // using last frame as
                                                      // right context
    KALDI_ASSERT(
        "Please set --frames-per-chunk at least as large as the neural net "
        "right context" &&
        input_frames_per_chunk_ >= total_nnet_right_context_);

    channel_n_frames_in_context_[channel] =
        std::min(nframes_in_batch, total_nnet_context_);

    // Computing number of output frames
    int total_nframes_minus_context =
        std::max(0, nframes_in_batch - total_nnet_context_);
    int total_output_nframes =
        (total_nframes_minus_context + subsampling_factor_ - 1) /
        subsampling_factor_;
    (*n_output_frames_valid)[i] = total_output_nframes;
  }
  context_switch_kernel_params_.batch_size = batch_size;
  context_switch_kernel_params_.d_features_frame_stride = features_frame_stride;
  context_switch_kernel_params_.d_batch_with_context =
      d_batch_with_context_.Data();
  context_switch_kernel_params_.d_batch_with_context_frame_stride =
      d_batch_with_context_.Stride();
  context_switch_kernel_params_.d_batch_ivectors =
      has_ivector_ ? d_batch_ivectors_.Data() : NULL;
  context_switch_kernel_params_.d_batch_ivectors_stride =
      has_ivector_ ? d_batch_ivectors_.Stride() : 0;
  context_switch_kernel_params_.d_batch_with_context_batch_stride =
      d_batch_with_context_.Stride() * input_frames_per_chunk_with_context_;

  cudaMemcpyAsync(d_batch_slot_assignement_, h_batch_slot_assignement_,
                  batch_size * sizeof(*d_batch_slot_assignement_),
                  cudaMemcpyHostToDevice, st_);
  cudaEventRecord(batch_slot_assignement_copy_evt_, st_);

  dim3 grid = {1,
               static_cast<unsigned int>(input_frames_per_chunk_with_context_),
               static_cast<unsigned int>(batch_size)};
  dim3 block = {
      64, 1,
      1};  // Expecting chunks in the order of magnitude of 64 frames. It will
           // still work with any numbers of frames per chunk, this only impacts
           // performance. This kernel is not a bottleneck anyway
  if (flush_eos_context) {
    BuildBatchWithContextKernelContextFlush(grid, block, st_,
                                            context_switch_kernel_params_);
  } else {
    BuildBatchWithContextKernel(grid, block, st_,
                                context_switch_kernel_params_);
    SaveContextFromBatchKernel(grid, block, st_, context_switch_kernel_params_);
  }
}

void BatchedStaticNnet3::RunNnet3(CuMatrix<BaseFloat> *d_all_log_posteriors,
                                  int batch_size) {
  for (int off = 0; off < batch_size; off += nnet3_batch_size_) {
    // Nnet3 destroys input, resizing
    d_nnet3_input_.Resize(
        nnet3_batch_size_ * input_frames_per_chunk_with_context_, input_dim_);
    if (has_ivector_) d_nnet3_ivectors_.Resize(nnet3_batch_size_, ivector_dim_);

    int minibatch_size = std::min(nnet3_batch_size_, batch_size - off);
    {
      // Copy minibatch from batch : mfcc
      int frames_per_minibatch =
          minibatch_size * input_frames_per_chunk_with_context_;
      CuSubMatrix<BaseFloat> dst =
          d_nnet3_input_.RowRange(0, frames_per_minibatch);
      CuSubMatrix<BaseFloat> src = d_batch_with_context_.RowRange(
          off * input_frames_per_chunk_with_context_, frames_per_minibatch);
      dst.CopyFromMat(src);
    }

    if (has_ivector_) {
      // Copy minibatch from batch : ivectors
      CuSubMatrix<BaseFloat> dst =
          d_nnet3_ivectors_.RowRange(0, minibatch_size);
      CuSubMatrix<BaseFloat> src =
          d_batch_ivectors_.RowRange(off, minibatch_size);
      dst.CopyFromMat(src);
    }

    // Using pre-compiled computation_
    nnet3::NnetComputer computer(config_.compute_opts.compute_config,
                                 *computation_, am_nnet_.GetNnet(), NULL);

    computer.AcceptInput("input", &d_nnet3_input_);
    if (has_ivector_) computer.AcceptInput("ivector", &d_nnet3_ivectors_);
    computer.Run();

    d_nnet3_output_ = computer.GetOutput("output");

    {
      int output_rows_per_minibatch = minibatch_size * output_frames_per_chunk_;

      // Copy nnet3 minibatch output to batch
      CuSubMatrix<BaseFloat> src =
          d_nnet3_output_.RowRange(0, output_rows_per_minibatch);
      CuSubMatrix<BaseFloat> dst = d_all_log_posteriors->RowRange(
          off * output_frames_per_chunk_, output_rows_per_minibatch);
      dst.CopyFromMat(src);
    }
  }

  // Postprocessing of the loglikehoods
  if (log_priors_.Dim() != 0)
    d_all_log_posteriors->AddVecToRows(-1.0, log_priors_);
  if (config_.compute_opts.acoustic_scale != 1.0f)
    d_all_log_posteriors->Scale(config_.compute_opts.acoustic_scale);
}

void BatchedStaticNnet3::RunBatch(
    const std::vector<int> &channels,
    const std::vector<BaseFloat *> &d_features, const int features_stride,
    const std::vector<BaseFloat *> &d_ivectors,
    const std::vector<int> &n_input_frames_valid,
    const std::vector<bool> &is_first_chunk,
    const std::vector<bool> &is_last_chunk,
    CuMatrix<BaseFloat> *d_all_log_posteriors,
    std::vector<std::vector<std::pair<int, const BaseFloat *>>>
        *all_frames_log_posteriors_ptrs) {
  // Using >= to avoid having to recompute d_features
  // In some cases the ptrs in d_features and d_ivectors are always the same,
  // but the number of active channels vary
  KALDI_ASSERT(d_features.size() >= channels.size());
  KALDI_ASSERT(is_last_chunk.size() >= channels.size());
  KALDI_ASSERT(is_first_chunk.size() >= channels.size());
  if (has_ivector_) {
    KALDI_ASSERT(d_ivectors.size() >= channels.size());
  }
  // Initializing the new channels
  for (size_t i = 0; i < is_first_chunk.size(); ++i) {
    if (is_first_chunk[i]) InitChannel(channels[i]);
  }

  all_frames_log_posteriors_ptrs
      ->clear();  // will start setting output frames now

  //
  // Step1: Processing chunks in d_features
  //

  // Building a continuous execution batch made of the current assignements,
  // while adding left and right context to the chunks
  BatchContextSwitch(channels, d_features, features_stride, d_ivectors,
                     n_input_frames_valid, false, &n_output_frames_valid_);
  // Running this batch
  RunNnet3(d_all_log_posteriors, channels.size());
  // Building the list of pointers to output frames. Will be used by the decoder
  FormatOutputPtrs(channels, d_all_log_posteriors,
                   all_frames_log_posteriors_ptrs, n_output_frames_valid_);

  //
  // Step2: Flushing context for chunks with is_last_chunk set
  //

  eos_channels_.clear();
  d_eos_features_.clear();
  d_eos_ivectors_.clear();
  eos_n_input_frames_valid_.clear();
  eos_n_output_frames_offset_.clear();
  for (int i = 0; i < channels.size(); ++i) {
    if (!is_last_chunk[i]) continue;
    eos_channels_.push_back(channels[i]);
    d_eos_features_.push_back(NULL);  // the context will serve as features
    if (has_ivector_) d_eos_ivectors_.push_back(d_ivectors[i]);
    eos_n_input_frames_valid_.push_back(0);
    eos_n_output_frames_offset_.push_back(
        n_output_frames_valid_[i]);  // append to previously generated frames
                                     // (in step1)
  }

  if (!eos_channels_.empty()) {
    BatchContextSwitch(eos_channels_, d_eos_features_, 0, d_eos_ivectors_,
                       eos_n_input_frames_valid_, /* flush context */ true,
                       &eos_n_output_frames_valid_);
    d_all_eos_log_posteriors_.Resize(d_all_log_posteriors->NumRows(),
                                     d_all_log_posteriors->NumCols());
    RunNnet3(&d_all_eos_log_posteriors_, eos_channels_.size());
    FormatOutputPtrs(eos_channels_, &d_all_eos_log_posteriors_,
                     all_frames_log_posteriors_ptrs, eos_n_output_frames_valid_,
                     &eos_n_output_frames_offset_);
  }
}

void BatchedStaticNnet3::FormatOutputPtrs(
    const std::vector<int> &channels, CuMatrix<BaseFloat> *d_all_log_posteriors,
    std::vector<std::vector<std::pair<int, const BaseFloat *>>>
        *all_frames_log_posteriors_ptrs,
    const std::vector<int> &n_output_frames_valid,
    const std::vector<int> *n_output_frames_valid_offset) {
  // Build the list of pointers to output frames. Will be used by the decoder
  KALDI_ASSERT(channels.size() == n_output_frames_valid.size());
  for (int i = 0; i < channels.size(); ++i) {
    int ichannel = channels[i];
    int offset =
        (n_output_frames_valid_offset) ? (*n_output_frames_valid_offset)[i] : 0;
    int total_output_nframes = offset + n_output_frames_valid[i];
    if (all_frames_log_posteriors_ptrs->size() < total_output_nframes)
      all_frames_log_posteriors_ptrs->resize(total_output_nframes);
    for (int iframe = offset; iframe < total_output_nframes; ++iframe) {
      std::vector<std::pair<int, const BaseFloat *>> &this_frame =
          (*all_frames_log_posteriors_ptrs)[iframe];
      int local_iframe = iframe - offset;
      CuSubVector<BaseFloat> out = d_all_log_posteriors->Row(
          i * output_frames_per_chunk_ + local_iframe);
      BaseFloat *frame = out.Data();
      this_frame.push_back({ichannel, frame});
    }
  }
}

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // HAVE_CUDA

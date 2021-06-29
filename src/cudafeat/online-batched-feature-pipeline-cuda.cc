// cudafeat/online-batched-feature-pipeline-cuda.cc

//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
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

#if HAVE_CUDA

#include "cudafeat/online-batched-feature-pipeline-cuda.h"

#include <nvToolsExt.h>

namespace kaldi {

OnlineBatchedFeaturePipelineCuda::OnlineBatchedFeaturePipelineCuda(
    const OnlineNnet2FeaturePipelineConfig &config,
    int32_t max_chunk_size_samples, int32_t max_lanes, int32_t num_channels)
    : info_(config),
      cmvn_(NULL),
      max_chunk_size_samples_(max_chunk_size_samples),
      max_lanes_(max_lanes),
      num_channels_(num_channels) {
  spectral_feat_ = NULL;
  cmvn_ = NULL;
  ivector_ = NULL;

  // Temporary to get frame extraction options
  if (info_.feature_type == "mfcc") {
    MfccComputer computer(info_.mfcc_opts);
    frame_opts_ = computer.GetFrameOptions();
  } else if (info_.feature_type == "fbank") {
    FbankComputer computer(info_.fbank_opts);
    frame_opts_ = computer.GetFrameOptions();
  } else {
    // Which ever base feature was requested is not currently supported
    KALDI_ASSERT(false);
  }

  // compute maximum chunk size for a given number of samples
  // round up because there may be additional context provided
  int32_t shift = frame_opts_.WindowShift();
  max_chunk_size_frames_ = (max_chunk_size_samples_ + shift - 1) / shift;

  if (info_.feature_type == "mfcc") {
    spectral_feat_ = new CudaOnlineBatchedSpectralFeatures(
        info_.mfcc_opts, max_chunk_size_frames_, num_channels_, max_lanes_);
  } else if (info_.feature_type == "fbank") {
    spectral_feat_ = new CudaOnlineBatchedSpectralFeatures(
        info_.fbank_opts, max_chunk_size_frames_, num_channels_, max_lanes_);
  } else {
    // Which ever base feature was requested is not currently supported
    KALDI_ASSERT(false);
  }

  if (info_.use_cmvn) {
    if (info_.global_cmvn_stats.NumCols() == 0) {
      KALDI_ERR << "global_cmvn_stats for OnlineCmvn must be non-empty.";
    }
    OnlineCmvnState cmvn_state(info_.global_cmvn_stats);
    CudaOnlineCmvnState cu_cmvn_state(cmvn_state);

    // TODO do we want to parameterize stats coarsening factor?
    // Setting this likely won't impact performance or accuracy
    // but will improve memory usage.  It's unclear where we
    // would want to register this parameter though.
    cmvn_ =
        new CudaOnlineBatchedCmvn(info_.cmvn_opts, cu_cmvn_state, FeatureDim(),
                                  max_chunk_size_frames_, num_channels_, 1);
  }

  if (info_.use_ivectors) {
    OnlineIvectorExtractionConfig ivector_extraction_opts;
    ReadConfigFromFile(config.ivector_extraction_config,
                       &ivector_extraction_opts);
    info_.ivector_extractor_info.Init(ivector_extraction_opts);

    ivector_ = new BatchedIvectorExtractorCuda(ivector_extraction_opts,
                                               max_chunk_size_frames_,
                                               max_lanes_, num_channels_);
  }

  current_samples_stash_ = new int32_t[num_channels_];

  // allocated pinned memory for storing channel desc
  CU_SAFE_CALL(cudaMallocHost(&h_lanes_, sizeof(LaneDesc) * max_lanes_));

  // allocate device memory
  lanes_ =
      (LaneDesc *)CuDevice::Instantiate().Malloc(sizeof(LaneDesc) * max_lanes_);

  cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
}

OnlineBatchedFeaturePipelineCuda::~OnlineBatchedFeaturePipelineCuda() {
  if (spectral_feat_ != NULL) delete spectral_feat_;
  if (cmvn_ != NULL) delete cmvn_;
  if (ivector_ != NULL) delete ivector_;

  CU_SAFE_CALL(cudaFreeHost(h_lanes_));

  delete[] current_samples_stash_;

  CuDevice::Instantiate().Free(lanes_);

  CU_SAFE_CALL(cudaEventDestroy(event_));
}

void OnlineBatchedFeaturePipelineCuda::ComputeFeaturesBatched(
    int32_t num_lanes, const std::vector<ChannelId> &channels,
    const std::vector<int32_t> &num_chunk_samples,
    const std::vector<bool> &first, const std::vector<bool> &last,
    BaseFloat sample_freq, const CuMatrixBase<BaseFloat> &cu_waves,
    CuMatrix<BaseFloat> *input_features, CuVector<BaseFloat> *ivector_features,
    std::vector<int32_t> *num_frames_computed) {
  nvtxRangePushA("OnlineBatchedFeaturePipelineCuda::ComputeFeaturesBatched");
  KALDI_ASSERT(num_lanes <= max_lanes_);
  KALDI_ASSERT(num_lanes <= num_frames_computed->size());

  // Ensure that h_lanes_ is consumed before overwriting.
  cudaEventSynchronize(event_);

  /// for each lane copy input into pinned memory
  for (int32_t lane = 0; lane < num_lanes; lane++) {
    ChannelId channel = channels[lane];
    KALDI_ASSERT(channel < num_channels_);
    KALDI_ASSERT(num_chunk_samples[lane] <= max_chunk_size_samples_);

    LaneDesc desc;
    desc.channel = channel;
    desc.last = last[lane];
    desc.first = first[lane];

    desc.current_sample = 0;
    if (!desc.first)
      // If not the first chunk then grab sample count from stash
      desc.current_sample = current_samples_stash_[channel];

    desc.num_chunk_samples = num_chunk_samples[lane];
    desc.current_frame = NumFrames(desc.current_sample, frame_opts_, false);

    // Compute total number of samples and frames
    int32_t num_samples = desc.current_sample + desc.num_chunk_samples;
    int32_t num_frames = NumFrames(num_samples, frame_opts_, desc.last);

    desc.num_chunk_frames = num_frames - desc.current_frame;

    // store desc in lane array
    h_lanes_[lane] = desc;

    // update current_sames stash
    current_samples_stash_[channel] =
        desc.current_sample + desc.num_chunk_samples;

    // write how many frames will be computed to output array
    (*num_frames_computed)[lane] = desc.num_chunk_frames;
  }

  cudaMemcpyAsync(lanes_, h_lanes_, sizeof(LaneDesc) * num_lanes,
                  cudaMemcpyHostToDevice, cudaStreamPerThread);

  // record event to know when copy is finished so that we don't overwrite
  // pinned array
  cudaEventRecord(event_, cudaStreamPerThread);

  if (info_.feature_type == "mfcc" || info_.feature_type == "fbank") {
    // Fbank called via the MFCC codepath
    // MFCC
    float vtln_warp = 1.0;
    spectral_feat_->ComputeFeaturesBatched(
        lanes_, num_lanes, cu_waves, sample_freq, vtln_warp, input_features);
  } else {
    KALDI_ASSERT(false);
  }

  if (info_.use_cmvn) {
    cmvn_->ComputeFeaturesBatched(num_lanes, lanes_, *input_features,
                                  input_features);
  }
  // Ivector
  if (info_.use_ivectors && ivector_features != NULL) {
    ivector_->GetIvectors(*input_features, ivector_features, lanes_, num_lanes);
  }

  nvtxRangePop();
}

}  // namespace kaldi

#endif  // HAVE_CUDA

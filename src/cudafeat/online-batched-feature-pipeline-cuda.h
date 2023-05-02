// cudafeat/online-batched-feature-pipeline-cuda.h

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

#ifndef KALDI_CUDAFEAT_ONLINE_BATCHED_FEATURE_PIPELINE_CUDA_H_
#define KALDI_CUDAFEAT_ONLINE_BATCHED_FEATURE_PIPELINE_CUDA_H_

#include <deque>
#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "feat/feature-window.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"

#include "online2/online-nnet2-feature-pipeline.h"

#include "cudafeat/feature-online-batched-cmvn-cuda.h"
#include "cudafeat/feature-online-batched-ivector-cuda.h"
#include "cudafeat/feature-online-batched-spectral-cuda.h"
#include "cudafeat/lane-desc.h"

namespace kaldi {

class OnlineBatchedFeaturePipelineCuda {
 public:
  explicit OnlineBatchedFeaturePipelineCuda(
      const OnlineNnet2FeaturePipelineConfig &config, int32_t max_chunk_size,
      int32_t max_lanes, int32_t num_channels);

  // Computes features and ivectors for a batched chunk of audio data.
  // Upon exit this call guarentees that all input data is read
  // allowing the input arrays to be overwritten.
  //
  // All work is submitted to the per-thread-default stream and no
  // manual synchronization occurs outside of the per-thread-default-stream.
  //
  // if (num_lanes < max_lanes) only the valid lanes will be read and only
  // the valid lanes will be output.  Data in other lanes is undefined.
  //
  // if (num_chunk_samples[i] < max_chunk_size) only the valid samples
  // will be read by that lane.
  //
  // inputs:
  //   num_lanes:  number of lanes to compute featurs for
  //   channels:  lane vector specifying the channel for each lane
  //   num_chunk_samples:  lane vector specifying number of samples in each lane
  //     note: this cannot exceed max_chunk_size_samples
  //   first:  lane vector specifying if this is the first chunk of data
  //   last:  lane vector specifying if this is the last chunk of data
  //   sample_freq:  model sample frequency
  //   cu_waves:  lane matrix of input wave data
  //     with num rows equal to max_lanes
  //     and num cols equal to max_chunk_size
  // outputs:
  //   input_features:  lane matrix of output of base features
  //     with num rows equal to max_lanes * max_chunk_size
  //     and num_cols equal to feat_dim
  //   ivector_features:  lane vector of output of ivectors for chunk
  //     with Dim = max_lanes * ivector_dim.
  //     ivectors are output in order ivector_lane0, ivector_lane1, etc
  //     if ivector_features is null they will not be computed
  //   num_frames_computed:  output vector containing the number of
  //     frames to be computed for each lane.

  void ComputeFeaturesBatched(int32_t num_lanes,
                              const std::vector<ChannelId> &channels,
                              const std::vector<int32_t> &num_chunk_samples,
                              const std::vector<bool> &first,
                              const std::vector<bool> &last,
                              BaseFloat sample_freq,
                              const CuMatrixBase<BaseFloat> &cu_waves,
                              CuMatrix<BaseFloat> *input_features,
                              CuVector<BaseFloat> *ivector_features,
                              std::vector<int32_t> *num_frames_computed);

  ~OnlineBatchedFeaturePipelineCuda();

  // Returns the maximum number of frames in a single chunk.
  // This should be used to size the input_features array that is
  // passed into ComputeFeaturesBatched
  int32_t GetMaxChunkFrames() { return max_chunk_size_frames_; }

  int32_t FeatureDim() { return spectral_feat_->Dim(); }
  int32_t IvectorDim() {
    if (ivector_)
      return ivector_->IvectorDim();
    else
      return 0;
  }

  const FrameExtractionOptions &GetFrameOptions() { return frame_opts_; }

 private:
  OnlineNnet2FeaturePipelineInfo info_;

  CudaOnlineBatchedSpectralFeatures *spectral_feat_;
  CudaOnlineBatchedCmvn *cmvn_;
  BatchedIvectorExtractorCuda *ivector_;
  FrameExtractionOptions frame_opts_;

  int32_t max_chunk_size_samples_;  // The maximum size of a chunk in samples
  int32_t max_chunk_size_frames_;   // The maximum size of a chunk in frames
  int32_t max_lanes_;               // The maximum number of lanes
  int32_t num_channels_;            // The maximum number of channels

  // channel array for stashing sample count
  int32_t *current_samples_stash_;

  // Host and Device array of lane descriptions
  LaneDesc *h_lanes_, *lanes_;

  cudaEvent_t event_;
};
}  // namespace kaldi

#endif  // KALDI_CUDAFEAT_ONLINE_CUDA_BATCHED_FEATURE_EXTRACTOR_H_

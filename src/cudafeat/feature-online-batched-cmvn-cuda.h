// cudafeat/feature-online-batched-cmvn-cuda.h
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
#ifndef KALDI_CUDAFEAT_FEATURE_ONLINE_BATCHED_CMVN_CUDA_H_
#define KALDI_CUDAFEAT_FEATURE_ONLINE_BATCHED_CMVN_CUDA_H_

#include "cudafeat/feature-online-cmvn-cuda.h"
#include "cudafeat/lane-desc.h"
#include "cudamatrix/cu-matrix.h"
#include "feat/online-feature.h"

namespace kaldi {

class CudaOnlineBatchedCmvn {
 public:
  CudaOnlineBatchedCmvn(const OnlineCmvnOptions &opts,
                        const CudaOnlineCmvnState &cmvn_state,
                        int32_t feat_dim,
                        int32_t chunk_size, int32_t num_channels,
                        int32_t stats_coarsening_factor);

  // Computes a chunk of features for each channel included in lanes
  void ComputeFeaturesBatched(int32_t num_lanes, const LaneDesc *lanes,
                              const CuMatrixBase<BaseFloat> &feats_in,
                              CuMatrix<BaseFloat> *feats_out);

 private:
  const OnlineCmvnOptions &opts_;
  const CudaOnlineCmvnState cmvn_state_;

  int32_t chunk_size_;
  int32_t num_channels_;

  // The number of frames for each fragment of stats.
  // Larger = faster, less memory, but less accurate
  // Smaller = slower, more memory, but more accurate,
  // 1 is equivalent to the non-batched version
  int32_t stats_coarsening_factor_;
  int32_t num_fragments_;  // window_size / stats_coarsening_factor_

  // This matrix stores prefix sum audio statistics in a rolling
  // buffer.  The stats are coarsened by stats_coarsening_factor_.
  // Coarsening reduces memory usage at a potential cost in
  // accuracy.  Matrix stores both sum and sum2 as float2 but
  // the matrix type is float as CuMatrix does not support float2.
  // val.x = sum and val.y = sum^2
  // Rows = channels, Cols = feat_dim * chunk_size/coarsening factor * 2
  CuMatrix<float> stats_fragments_;
};

}  // namespace kaldi

#endif

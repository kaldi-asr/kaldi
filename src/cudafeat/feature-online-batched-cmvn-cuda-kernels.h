// cudafeat/feature-online-batched-cmvn-cuda-kernels.h
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

#ifndef KALDI_CUDAFEAT_FEATURE_ONLINE_BATCHED_CMVN_CUDA_KERNELS_H_
#define KALDI_CUDAFEAT_FEATURE_ONLINE_BATCHED_CMVN_CUDA_KERNELS_H_

#include "cudafeat/lane-desc.h"

namespace kaldi {

// ld{i,o} = size of inner dimension of matrix alloction
// stride{i,o} = stride between consecutive batch matrices

void compute_cmvn_stats(int32_t feat_dim, int32_t chunk_size,
                        int32_t stats_coarsening_factor, int32_t cmn_window,
                        const float *in_data, int32_t ldi, int32_t stridei,
                        float *stats_data, int32_t lds, const LaneDesc *lanes,
                        int32_t num_lanes);

void apply_cmvn(int32_t cmvn_window, bool var_norm, bool mean_norm,
                int32_t feat_dim, int32_t chunk_size,
                int32_t stats_coarsening_factor, const float *in_data,
                int32_t ldi, int32_t stridei, const float *stats_data,
                int32_t lds, const float *global_stats, int32_t ldg,
                int32_t global_frames, const float *speaker_stats, int32_t ldss,
                int32_t speaker_frames, float *out_data, int32_t ldo,
                int32_t strideo, const LaneDesc *lanes, int32_t num_lanes);
}  // namespace kaldi

#endif

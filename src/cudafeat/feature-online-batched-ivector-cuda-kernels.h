// cudafeat/feature-online-batched-ivector-cuda-kernels.h
//
// Copyright (c) 202020, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef CUDAFEAT_FEATURE_ONLINE_BATCHED_IVECTOR_CUDA_KERNELS_H
#define CUDAFEAT_FEATURE_ONLINE_BATCHED_IVECTOR_CUDA_KERNELS_H

#if HAVE_CUDA == 1
#include <cudafeat/lane-desc.h>
namespace kaldi {

// Kernel naming conventions:
//   ld*:  leading dimension allocation size for a matrix
//      Thus to compute the address of a matrix you use this formula:
//         matrix_pointer + row * ld + col
//   stride*:  offset to the next batch matrix or vector
//    Thus to compute the matrix pointer of a matrix you use this formula:
//         matrix_pointer = base_pointer + batch_number * stride

void zero_invalid_posteriors(int32_t num_chunk_frames, int32_t num_gauss,
                             float *posteriors, int32_t ldp, int32_t stridep,
                             int32_t right, const LaneDesc *lanes,
                             int32_t num_lanes);

void splice_features_batched(int32_t num_chunk_frames, int32_t feat_dim,
                             int32_t left, int32_t right, const float *feats,
                             int32_t ldf, int32_t stridef,
                             const float *stashed_feats, int32_t ldst,
                             int32_t stridest, float *spliced_feats,
                             int32_t lds, int32_t strides,
                             const LaneDesc *lanes, int32_t num_lanes);

void stash_feats(int32_t chunk_size, const float *feats, int32_t feat_dim,
                 int32_t ldf, int32_t stridef, float *stash, int32_t stash_size,
                 int32_t lds, int32_t strides, const LaneDesc *lanes,
                 int32_t num_lanes);

void batched_update_linear_and_quadratic_terms(
    int32_t n, float prior_offset, float posterior_scale, int32_t max_count,
    float *quadratic, int32_t ldq, int32_t strideq, float *linear,
    int32_t stridel, const LaneDesc *lanes, int32_t num_lanes);

void square_batched_matrix(int32_t chunk_frames, int32_t num_cols,
                           const float *feats, int32_t ldf, int32_t stridef,
                           float *feats_sq, int32_t lds, int32_t stides,
                           const LaneDesc *lanes, int32_t num_lanes);

void batched_compute_linear_term(int32_t num_gauss_, int32_t feat_dim_,
                                 int32_t ivector_dim_, float *sigma,
                                 int32_t lds, float *X, int32_t ldx,
                                 int32_t stridex, float *linear,
                                 int32_t stridel, const LaneDesc *lanes,
                                 int32_t num_lanes);

void batched_convert_sp_to_dense(int n, float *A_sp, int32_t strides, float *A,
                                 int32_t lda, int32_t stridea,
                                 const LaneDesc *lanes, int32_t num_lanes);

void batched_sum_posteriors(int32_t chunk_size, int32_t num_gauss,
                            float *posteriors, int32_t ldp, int32_t stridep,
                            float *gamma, int32_t strideg, float post_scale,
                            const LaneDesc *lanes, int32_t num_lanes);

void initialize_channels(int32_t num_gauss, int32_t feat_dim, float *gamma,
                         int32_t strideg, float *X, int32_t ldx, int32_t stridx,
                         const LaneDesc *lanes, int32_t num_lanes);

void apply_and_update_stash(int32_t num_gauss, int32_t feat_dim, float *gamma,
                            float *gamma_stash, int32_t strideg, float *X,
                            int32_t ldx, int32_t stridex, float *X_stash,
                            int32_t lds, int32_t strides, const LaneDesc *lanes,
                            int32_t num_lanes);

}  // namespace kaldi

#endif
#endif

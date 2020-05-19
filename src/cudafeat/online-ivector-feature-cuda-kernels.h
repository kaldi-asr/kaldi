// cudafeat/online-ivector-feature-cuda-kernels.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef CUDAFEAT_ONLINE_IVECTOR_FEATURE_CUDA_KERNELS
#define CUDAFEAT_ONLINE_IVECTOR_FEATURE_CUDA_KERNELS

namespace kaldi {
void batched_gemv_reduce(int batch_size, int rows, int cols, int A_stride,
                         const float *AT, int B_stride, const float *B,
                         float *C);

void splice_features(int32_t num_frames, int32_t feat_dim, int32_t left,
                     int32_t size, const float *feats, int32_t ldf,
                     float *sfeats, int32_t lds);

void update_linear_and_quadratic_terms(int32_t n, float old_num_frames,
                                       float prior_offset_,
                                       float *cur_tot_weight, int32_t max_count,
                                       float *quadratic, float *linear);

void get_matrix_sum_double_buffer(int32_t b, int32_t num_rows, int32_t num_cols,
                                  float *A, int32_t lda, float scale,
                                  float *sum);

void square_matrix(int32_t num_rows, int32_t num_cols, const float *feats,
                   int32_t ldf, float *feats_sq, int32_t lds);
}
#endif

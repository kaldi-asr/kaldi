// cudafeat/online-ivector-feature-cuda-kernels.cu
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

#include <cub/cub.cuh>
#include "cudafeat/online-ivector-feature-cuda-kernels.h"
#include "cudamatrix/cu-common.h"
namespace kaldi {

// Meant to be called with blockDim= 32x32
__global__ void batched_gemv_reduce_kernel(int rows, int cols,
                                           const float* __restrict__ A, int lda,
                                           const float* __restrict__ X, int ldx,
                                           float* C) {
  // Specialize WarpReduce for type float
  typedef cub::WarpReduce<float> WarpReduce;
  // Allocate WarpReduce shared memory for 32 warps
  __shared__ typename WarpReduce::TempStorage temp_storage[32];

  __shared__ float s_A[32][32 + 1];  //+1 to avoid bank conflicts on transpose

  int bid = blockIdx.x;   // batch id
  int tid = threadIdx.x;  // thread id
  int wid = threadIdx.y;  // warp id

  // Offset to input matrix to starting row for batch
  const float* __restrict__ A_in = A + bid * rows * lda;
  // Offset to input vector to starting column for batch
  const float* __restrict__ X_in = X + bid * ldx;

  for (int i = 0; i < cols; i += 32) {  // threadIdx.x, keep all threads present
    int c = i + tid;

    float sum = 0.0f;
    // Perform dot product
    for (int j = 0; j < rows;
         j += 32) {  // threadIdx.y, keep all threads present
      int r = j + wid;

      float val = 0.0f;
      if (c < cols && r < rows) {
        // coalesced reads
        val = A_in[r * lda + c] * X_in[r];
      }

      // write to shared memory
      __syncthreads();  // wait for shared memory to be written
      s_A[wid][tid] = val;
      __syncthreads();  // wait for shared memory to be consumed

      // transpose read from shared memory and collect sum
      sum += s_A[tid][wid];
    }
    // reduce sum in cub
    sum = WarpReduce(temp_storage[wid]).Sum(sum);

    // update c now that we are trasnposed
    c = i + wid;
    if (tid == 0 && c < cols) {
      // Add contribution to final sum.
      // Atomic necessary due to different batches updating this
      atomicAdd(&C[c], sum);
    }
  }
}

// computes feats^2.  This works in place and out of place.
__global__ void square_matrix_kernel(int32_t num_rows, int32_t num_cols,
                                     const float* feats, int32_t ldf,
                                     float* feats_sq, int32_t lds) {
  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < num_rows;
       i += blockDim.y * gridDim.y) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_cols;
         j += blockDim.x * gridDim.x) {
      float f = feats[i * ldf + j];
      feats_sq[i * lds + j] = f * f;
    }
  }
}

// takes features in feat and writes them into sfeats while applying
// the splicing algorithm for the left and right context.
// input features that are out of range are clamped.
__global__ void splice_features_kernel(int32_t num_frames, int32_t feat_dim,
                                       int32_t left, int32_t size,
                                       const float* __restrict__ feats,
                                       int32_t ldf, float* __restrict__ sfeats,
                                       int32_t lds) {
  int32_t frame = blockIdx.x;
  int32_t tid = threadIdx.x;

  // offset feature output to process frame
  float* feat_out = sfeats + lds * frame;

  // for each splice of input
  for (int i = 0; i < size; i++) {
    int r = frame + i + left;
    // clamp input row
    if (r < 0) r = 0;
    if (r > num_frames - 1) r = num_frames - 1;

    // for each column of input in parallel
    for (int c = tid; c < feat_dim; c += blockDim.x) {
      // read feature from input row offset by column
      float val = feats[r * ldf + c];

      // write feature to output offset by splice index and column
      feat_out[i * feat_dim + c] = val;
    }
  }
}

// Computes the sum of all terms in a matrix.
// The kernel double buffers the output such that the
// output is written to retval[b] where b is 0 or 1.
// The output element of retval is written as zero.
// Double buffering eliminates a call to cudaMemset
__global__ void get_matrix_sum_double_buffer_kernel(int32_t b, int32_t num_rows,
                                                    int32_t num_cols, float* A,
                                                    int32_t lda, float scale,
                                                    float* retval) {
  // Specialize WarpReduce for type float
  typedef cub::BlockReduce<float, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 32>
      BlockReduce;
  // Allocate WarpReduce shared memory for 32 warps
  __shared__ typename BlockReduce::TempStorage temp_storage;

  float sum = 0.0f;

  // compute local sums in parallel
  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < num_rows;
       i += blockDim.y * gridDim.y) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_cols;
         j += blockDim.x * gridDim.x) {
      sum += A[i * lda + j];
    }
  }

  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(&retval[b], sum * scale);
    int next_b = (b + 1) % 2;
    retval[next_b] = 0.0f;
  }
}

// This kernel updates the diagonal of the quadratic terms and
// element zero of the linear term. Code is meant to match
// ivector_extractor.cc:
//   double old_num_frames = num_frames_,
//          new_num_frames = num_frames_ + tot_weight;
//   double old_prior_scale = std::max(old_num_frames, max_count_) / max_count_,
//          new_prior_scale = std::max(new_num_frames, max_count_) / max_count_;
//   double prior_scale_change = new_prior_scale - old_prior_scale;
//   if (prior_scale_change != 0.0) {
//     linear_term_(0) += prior_offset_ * prior_scale_change;
//     quadratic_term_.AddToDiag(prior_scale_change);
//   }
//Extra 1.0f on prior_scale_change is to match ivector_extractor.cc:
//  linear_term_(0) += prior_offset;
//  quadratic_term_.AddToDiag(1.0);
__global__ void update_linear_and_quadratic_terms_kernel(
    int32_t n, float old_num_frames, float prior_offset, float* cur_tot_weight, 
    int32_t max_count, float* quadratic, float* linear) {
  float cur_weight = *cur_tot_weight;

  float new_num_frames = old_num_frames + cur_weight;
  float prior_scale_change = 1.0f;

  if(max_count!=0.0f) {
    float old_prior_scale = max(old_num_frames, (float)max_count) / max_count;
    float new_prior_scale = max(new_num_frames, (float)max_count) / max_count;
    prior_scale_change += new_prior_scale - old_prior_scale;
  }

  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
     i += blockDim.x * gridDim.x) {
    int32_t diag_idx = ((i + 1) * (i + 2) / 2) - 1;
    quadratic[diag_idx] += prior_scale_change;
  }

  if (threadIdx.x == 0 && blockIdx.x==0) {
    linear[0] += prior_offset * prior_scale_change;  
  }
}

void batched_gemv_reduce(int batch_size, int rows, int cols, int A_stride,
                         const float* AT, int B_stride, const float* B,
                         float* C) {
  batched_gemv_reduce_kernel<<<batch_size, dim3(32, 32)>>>(
      rows, cols, AT, A_stride, B, B_stride, C);
  CU_SAFE_CALL(cudaGetLastError());
}

void splice_features(int32_t num_frames, int32_t feat_dim, int32_t left,
                     int32_t size, const float* feats, int32_t ldf,
                     float* sfeats, int32_t lds) {
  int threads = (feat_dim + 31) / 32 * 32;  // round up to the nearest warp size
  if (threads > 1024) threads = 1024;       // Max block size is 1024 threads

  splice_features_kernel<<<num_frames, threads>>>(
      num_frames, feat_dim, left, size, feats, ldf, sfeats, lds);
  CU_SAFE_CALL(cudaGetLastError());
}

void update_linear_and_quadratic_terms(int32_t n, float old_num_frames,
                                       float prior_offset,
                                       float* cur_tot_weight, int32_t max_count,
                                       float* quadratic, float* linear) {
  // Only using 1 CTA here  for now as the updates are tiny and this lets us
  // use syncthreads as a global barrier.
  update_linear_and_quadratic_terms_kernel<<<1, 1024>>>(
      n, old_num_frames, prior_offset, cur_tot_weight, max_count, quadratic, 
      linear);
  CU_SAFE_CALL(cudaGetLastError());
}

void get_matrix_sum_double_buffer(int32_t b, int32_t num_rows, int32_t num_cols,
                                  float* A, int32_t lda, float scale,
                                  float* sum) {
  dim3 threads(32, 32);
  dim3 blocks((num_cols + threads.x - 1) / threads.x,
              (num_rows + threads.y - 1) / threads.y);

  get_matrix_sum_double_buffer_kernel<<<blocks, threads>>>(
      b, num_rows, num_cols, A, lda, scale, sum);
  CU_SAFE_CALL(cudaGetLastError());
}

void square_matrix(int32_t num_rows, int32_t num_cols, const float* feats,
                   int32_t ldf, float* feats_sq, int32_t lds) {
  dim3 threads(32, 32);
  dim3 blocks((num_cols + threads.x - 1) / threads.x,
              (num_rows + threads.y - 1) / threads.y);

  square_matrix_kernel<<<blocks, threads>>>(num_rows, num_cols, feats, ldf,
                                            feats_sq, lds);
  CU_SAFE_CALL(cudaGetLastError());
}
}

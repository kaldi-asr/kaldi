// cudafeat/feature-online-batched-ivector-cuda-kernels.cu
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

#if HAVE_CUDA == 1
#include <cub/cub.cuh>
#include "cudafeat/feature-online-batched-ivector-cuda-kernels.h"
#include "cudamatrix/cu-common.h"
namespace kaldi {

// computes pointwise square of each matrix
__global__ void square_batched_matrix_kernel(
    int32_t chunk_frames, int32_t num_cols, const float *feats, int32_t ldf,
    int32_t stridef, float *feats_sq, int32_t lds, int32_t strides,
    const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.z;

  feats = feats + lane * stridef;
  feats_sq = feats_sq + lane * strides;

  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < chunk_frames;
       i += blockDim.y * gridDim.y) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_cols;
         j += blockDim.x * gridDim.x) {
      float f = feats[i * ldf + j];
      feats_sq[i * lds + j] = f * f;
    }
  }
}

void square_batched_matrix(int32_t chunk_frames, int32_t num_cols,
                           const float *feats, int32_t ldf, int32_t stridef,
                           float *feats_sq, int32_t lds, int32_t strides,
                           const LaneDesc *lanes, int32_t num_lanes) {
  dim3 threads(32, 32);
  dim3 blocks((num_cols + threads.x - 1) / threads.x,
              (chunk_frames + threads.y - 1) / threads.y, num_lanes);

  square_batched_matrix_kernel<<<blocks, threads>>>(
      chunk_frames, num_cols, feats, ldf, stridef, feats_sq, lds, strides,
      lanes, num_lanes);
  CU_SAFE_CALL(cudaGetLastError());
}

// after computing posteriors some rows are invalid because they were created
// with rows with undefined data.  This kernel zeros those rows out so that
// they will not contribue to stats.
__global__ void zero_invalid_posteriors_kernel(
    int32_t chunk_size, int32_t num_gauss, float *posteriors, int32_t ldp,
    int32_t stridep, int32_t right, const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.z;

  LaneDesc desc = lanes[lane];
  int32_t num_chunk_frames = desc.num_chunk_frames;
  int32_t current_frame = desc.current_frame;
  bool last = desc.last;

  // last valid frame for reading
  int32_t num_computed_rows = current_frame + num_chunk_frames;

  // if not the last frame remove right context
  if (!last) {
    num_computed_rows -= right;
  }

  // offset by lane
  posteriors = posteriors + lane * stridep;

  for (int r = blockIdx.y * blockDim.y + threadIdx.y; r < chunk_size;
       r += blockDim.y * gridDim.y) {
    int global_row = current_frame + r - right;
    if (global_row < 0 || global_row >= num_computed_rows) {
      // zero this row out
      for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < num_gauss;
           c += blockDim.x * gridDim.x) {
        posteriors[r * ldp + c] = 0.0f;
      }
    }
  }
}

void zero_invalid_posteriors(int32_t num_chunk_frames, int32_t num_gauss,
                             float *posteriors, int32_t ldp, int32_t stridep,
                             int32_t right, const LaneDesc *lanes,
                             int32_t num_lanes) {
  dim3 threads(32, 32);
  dim3 blocks((num_gauss + 31) / 32, (num_chunk_frames + 31) / 32, num_lanes);

  zero_invalid_posteriors_kernel<<<blocks, threads>>>(
      num_chunk_frames, num_gauss, posteriors, ldp, stridep, right, lanes,
      num_lanes);
}

// Meant to be called with blockDim= 32x32
// takes features in feat and writes them into sfeats while applying
// the splicing algorithm for the left and right context.
// input features that are out of range are clamped.
__global__ void splice_features_batched_kernel(
    int32_t chunk_size, int32_t feat_dim, int32_t left, int32_t right,
    const float *__restrict__ feats_in, int32_t ldi, int32_t stridei,
    const float *__restrict__ feats_stash, int32_t ldst, int32_t stridest,
    float *__restrict__ feats_out, int32_t ldo, int32_t strideo,
    const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.y;
  // output frame index
  int32_t oframe = blockIdx.x;
  int32_t tid = threadIdx.x;

  LaneDesc desc = lanes[lane];
  int32_t num_chunk_frames = desc.num_chunk_frames;
  int32_t channel = desc.channel;
  int32_t current_frame = desc.current_frame;
  bool last = desc.last;

  // offset by lane
  feats_in = feats_in + lane * stridei;
  feats_out = feats_out + lane * strideo;

  // offset by channel
  feats_stash = feats_stash + channel * stridest;

  // offset feature output to process oframe
  feats_out = feats_out + ldo * oframe;

  // the size of the stash
  int32_t ssize = left + right;
  // the size of the window
  int32_t size = ssize + 1;

  // number of valid frame for reading
  int32_t num_valid_frames = current_frame + num_chunk_frames;

  // number of valid frames for writing
  int32_t num_computed_frames = num_valid_frames;

  // if not the last frame remove right context
  if (!last) {
    num_computed_frames -= right;
  }

  // subtract right context from logical frame to delay output
  int32_t local_frame = oframe - right;
  int32_t global_frame = current_frame + local_frame;

  // these frames are set to zeros
  if (global_frame < 0 || global_frame >= num_computed_frames) {
    for (int i = 0; i < size; i++) {
      for (int c = tid; c < feat_dim; c += blockDim.x) {
        feats_out[i * feat_dim + c] = 0.0f;
      }
    }
    return;
  }

  for (int i = -left; i <= right; i++) {
    int32_t g_in = global_frame + i;  // global frame index
    int32_t l_in = local_frame + i;   // local frame index

    // if global row is below zero clamp local to zero
    if (g_in < 0) l_in = 0;

    // if global row is larger than the number of valid frames
    if (g_in >= num_valid_frames) {
      // should only happen on last chunk
      assert(last);
      // clamp input
      l_in = num_chunk_frames - 1;
    }

    // set default input location
    const float *feats = feats_in;
    int32_t ld = ldi;

    // if l < 0 then feats come from the stash
    if (l_in < 0) {
      // input is from stash
      feats = feats_stash;
      ld = ldst;
      l_in += ssize;  // offset by stash size
    }

    // for each column of input in parallel
    for (int c = tid; c < feat_dim; c += blockDim.x) {
      // read feature from input row offset by column
      float val = feats[l_in * ld + c];

      // write feature to output offset by splice index and column
      feats_out[(i + left) * feat_dim + c] = val;
    }
  }
}

void splice_features_batched(int32_t num_chunk_frames, int32_t feat_dim,
                             int32_t left, int32_t right, const float *feats,
                             int32_t ldf, int32_t stridef,
                             const float *stashed_feats, int32_t ldst,
                             int32_t stridest, float *spliced_feats,
                             int32_t lds, int32_t strides,
                             const LaneDesc *lanes, int32_t num_lanes) {
  int threads = (feat_dim + 31) / 32 * 32;  // round up to the nearest warp size
  if (threads > 1024) threads = 1024;       // Max block size is 1024 threads

  dim3 blocks(num_chunk_frames, num_lanes);

  splice_features_batched_kernel<<<blocks, threads>>>(
      num_chunk_frames, feat_dim, left, right, feats, ldf, stridef,
      stashed_feats, ldst, stridest, spliced_feats, lds, strides, lanes,
      num_lanes);

  CU_SAFE_CALL(cudaGetLastError());
}

__global__ void shift_feats_kernel(int32_t chunk_size, const float *feats,
                                   int32_t feat_dim, int32_t ldf,
                                   int32_t stridef, float *stash, int32_t ssize,
                                   int32_t lds, int32_t strides,
                                   const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.x;
  int32_t frame = threadIdx.y;
  int32_t tid = threadIdx.x;

  LaneDesc desc = lanes[lane];
  int32_t num_chunk_frames = desc.num_chunk_frames;
  int32_t channel = desc.channel;

  // offset inputs/outputs
  feats = feats + lane * stridef;
  stash = stash + channel * strides;

  // shift stash by nun_chunk_frames
  if (num_chunk_frames < ssize) {
    // shift stash by num_chunk_frames
    int32_t dst_frame = frame;
    int32_t src_frame = frame + num_chunk_frames;

    // loop over columns of output in parallel but keep
    // CTA converged for syncthreads
    for (int i = 0; i < feat_dim; i += blockDim.x) {
      int c = i + tid;

      float val;

      if (src_frame < ssize) {
        // read stash values
        val = stash[src_frame * lds + c];
      }

      // wait for all reads to complete
      __syncthreads();

      if (src_frame < ssize) {
        // write stash values
        stash[dst_frame * lds + c] = val;
      }
    }
  }
}

__global__ void stash_feats_kernel(int32_t chunk_size, const float *feats,
                                   int32_t feat_dim, int32_t ldf,
                                   int32_t stridef, float *stash, int32_t ssize,
                                   int32_t lds, int32_t strides,
                                   const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.y;
  int32_t frame = blockIdx.x;
  int32_t tid = threadIdx.x;

  LaneDesc desc = lanes[lane];
  int32_t num_chunk_frames = desc.num_chunk_frames;
  int32_t channel = desc.channel;

  if (frame >= num_chunk_frames) return;

  // offset inputs/outputs
  feats = feats + lane * stridef;
  stash = stash + channel * strides;

  // r is the input frame to store
  int32_t r = num_chunk_frames - ssize + frame;
  if (r >= 0 && r < num_chunk_frames) {
    // copy feats to stash
    // for each column of input in parallel
    for (int c = tid; c < feat_dim; c += blockDim.x) {
      stash[frame * lds + c] = feats[r * ldf + c];
    }
  }
}
void stash_feats(int32_t chunk_size, const float *feats, int32_t feat_dim,
                 int32_t ldf, int32_t stridef, float *stash, int32_t stash_size,
                 int32_t lds, int32_t strides, const LaneDesc *lanes,
                 int32_t num_lanes) {
  {
    // First we need to shift feats to handle the case where num_chunk_frames
    // is less than stash size

    KALDI_ASSERT(stash_size <= 32);
    // This only works if stash size is <= 32 as we rely on __syncthreads()
    // to avoid read/write hazards when reading/writing in-place
    dim3 threads(32, 32);
    dim3 blocks(num_lanes);

    shift_feats_kernel<<<blocks, threads>>>(chunk_size, feats, feat_dim, ldf,
                                            stridef, stash, stash_size, lds,
                                            strides, lanes, num_lanes);
  }

  {
    int threads =
        (feat_dim + 31) / 32 * 32;       // round up to the nearest warp size
    if (threads > 1024) threads = 1024;  // Max block size is 1024 threads
    dim3 blocks(stash_size, num_lanes);

    // Then we need to copy feats from source into stash
    stash_feats_kernel<<<blocks, threads>>>(chunk_size, feats, feat_dim, ldf,
                                            stridef, stash, stash_size, lds,
                                            strides, lanes, num_lanes);
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
// Extra 1.0f on prior_scale_change is to match ivector_extractor.cc:
//  linear_term_(0) += prior_offset;
//  quadratic_term_.AddToDiag(1.0);
__global__ void batched_update_linear_and_quadratic_terms_kernel(
    int32_t ivector_dim, float prior_offset, float posterior_scale,
    int32_t max_count, float *quadratic, int32_t ldq, int32_t strideq,
    float *linear, int32_t stridel, const LaneDesc *lanes, int32_t num_lanes) {
  int lane = blockIdx.x;
  LaneDesc desc = lanes[lane];

  // offset arrays
  linear = linear + lane * stridel;
  quadratic = quadratic + lane * strideq;

  // This is always zero because linear and quadratic terms are not
  // being carried forward.  Thus we don't need to remove old prior
  // scale.  Keeping the code below so that it logically matches
  // the CPU code in case someone is looking at this in the future.
  float old_num_frames = 0;
  // float old_num_frames = desc.current_frame;
  float new_num_frames = desc.current_frame + desc.num_chunk_frames;

  // in CPU code the frame counts are scaled by posterior scale
  new_num_frames *= posterior_scale;
  old_num_frames *= posterior_scale;

  float prior_scale_change = 1.0f;

  if (max_count != 0.0f) {
    float old_prior_scale = max(old_num_frames, (float)max_count) / max_count;
    float new_prior_scale = max(new_num_frames, (float)max_count) / max_count;
    prior_scale_change += new_prior_scale - old_prior_scale;
  }

  for (int32_t i = threadIdx.x; i < ivector_dim; i += blockDim.x) {
    int32_t diag_idx = i * ldq + i;
    quadratic[diag_idx] += prior_scale_change;
  }

  if (threadIdx.x == 0) {
    linear[0] += prior_offset * prior_scale_change;
  }
}

void batched_update_linear_and_quadratic_terms(
    int32_t ivector_dim, float prior_offset, float posterior_scale,
    int32_t max_count, float *quadratic, int32_t ldq, int32_t strideq,
    float *linear, int32_t stridel, const LaneDesc *lanes, int32_t num_lanes) {
  // Only using 1 CTA per lane here  for now as the updates are tiny and this
  // lets us use syncthreads as a global barrier across the lane
  batched_update_linear_and_quadratic_terms_kernel<<<num_lanes, 1024>>>(
      ivector_dim, prior_offset, posterior_scale, max_count, quadratic, ldq,
      strideq, linear, stridel, lanes, num_lanes);

  CU_SAFE_CALL(cudaGetLastError());
}

// each CTA performs the multiplications for a specific gauss point
// sigma is cached in shared memory and used across all lanes this
// avoids repeated loads of this term saving memory bandwidth
__global__ void batched_compute_linear_term_kernel(
    int32_t num_gauss, int32_t feat_dim, int32_t ivector_dim,
    const float *__restrict__ sigma, int32_t lds, const float *__restrict__ X,
    int32_t ldx, int32_t stridex, float *linear, int32_t stridel,
    const LaneDesc *lanes, int32_t num_lanes) {
  int rows = feat_dim;
  int cols = ivector_dim;
  int gid = blockIdx.x;  // gauss point

  // dnyamic shared memory to cache A
  extern __shared__ float s_A[];

  // Offset sigma to gauss point matrix
  const float *__restrict__ A_in = sigma + gid * rows * lds;

  // cache A into shared memory
  for (int r = threadIdx.y; r < rows; r += blockDim.y) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      s_A[r * ivector_dim + c] = A_in[r * lds + c];
    }
  }
  // wait for s_A to be loaded
  __syncthreads();

  // for each lane in parallel across y CTA dimension
  for (int lane = threadIdx.y; lane < num_lanes; lane += blockDim.y) {
    // Offset to input vector to starting column for lane
    const float *__restrict__ X_in = X + lane * stridex + gid * ldx;
    // Offset output by lane
    float *C = linear + lane * stridel;
    // for each column in parallel across x cta dimension
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      float sum = 0.0f;
      // operate on rows in serial
      for (int r = 0; r < rows; r++) {
        // Read A from shared memory, X is a broadcast and should hit in cache
        float val = s_A[r * ivector_dim + c] * X_in[r];
        sum += val;
      }
      atomicAdd(&C[c], sum);
    }
  }
}

void batched_compute_linear_term(int32_t num_gauss, int32_t feat_dim,
                                 int32_t ivector_dim, float *sigma, int32_t lds,
                                 float *X, int32_t ldx, int32_t stridex,
                                 float *linear, int32_t stridel,
                                 const LaneDesc *lanes, int32_t num_lanes) {
  // 1 CTA per gauss point
  dim3 blocks(num_gauss);

  // 128 threads in ivector dimension, 8 threads in num_lanes dimension
  dim3 threads(128, 8);

  // dynamic shared memory size for caching A
  size_t shared_size = (ivector_dim * feat_dim) * sizeof(BaseFloat);

  batched_compute_linear_term_kernel<<<blocks, threads, shared_size>>>(
      num_gauss, feat_dim, ivector_dim, sigma, lds, X, ldx, stridex, linear,
      stridel, lanes, num_lanes);

  CU_SAFE_CALL(cudaGetLastError());
}

__global__ void batched_convert_sp_to_dense_kernel(int32_t n, float *A_sp,
                                                   int32_t strides, float *A,
                                                   int32_t lda, int32_t stridea,
                                                   const LaneDesc *lanes,
                                                   int32_t num_lanes) {
  int32_t lane = blockIdx.z;
  // Offset input and output array by lane
  A_sp = A_sp + lane * strides;
  A = A + lane * stridea;

  // For each output
  for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n;
       i += blockDim.y * gridDim.y) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += blockDim.x * gridDim.x) {
      int row, col;
      if (i <= j) {
        col = i;
        row = j;
      } else {
        row = i;
        col = j;
      }

      int32_t dst_idx = i * lda + j;
      int32_t src_idx = (row * (row + 1) / 2) + col;

      A[dst_idx] = A_sp[src_idx];
    }
  }
}

void batched_convert_sp_to_dense(int n, float *A_sp, int32_t strides, float *A,
                                 int32_t lda, int32_t stridea,
                                 const LaneDesc *lanes, int32_t num_lanes) {
  dim3 threads(32, 32);
  int block = (n + 31) / 32;  // blocks in x and y dimensions
  dim3 blocks(block, block, num_lanes);

  batched_convert_sp_to_dense_kernel<<<blocks, threads>>>(
      n, A_sp, strides, A, lda, stridea, lanes, num_lanes);
}

__global__ void batched_sum_posteriors_kernel(
    int32_t chunk_size, int32_t num_gauss, float *posteriors, int32_t ldp,
    int32_t stridep, float *gamma, int32_t strideg, float post_scale,
    const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.y;

  // offset input and output by lane
  posteriors = posteriors + lane * stridep;
  gamma = gamma + lane * strideg;

  // for each column in parallel
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < num_gauss;
       col += blockDim.x * gridDim.x) {
    // compute sum across rows for this column
    float sum = 0.0f;
    for (int row = 0; row < chunk_size; row++) {
      sum += posteriors[row * ldp + col];
    }

    // add to output vector
    gamma[col] = post_scale * sum;
  }
}

void batched_sum_posteriors(int32_t chunk_size, int32_t num_gauss,
                            float *posteriors, int32_t ldp, int32_t stridep,
                            float *gamma, int32_t strideg, float post_scale,
                            const LaneDesc *lanes, int32_t num_lanes) {
  int32_t threads = 128;
  dim3 blocks((num_gauss + threads - 1) / threads, num_lanes);

  batched_sum_posteriors_kernel<<<blocks, threads>>>(
      chunk_size, num_gauss, posteriors, ldp, stridep, gamma, strideg,
      post_scale, lanes, num_lanes);
}

__global__ void initialize_channels_kernel(int32_t num_gauss, int32_t feat_dim,
                                           float *gamma, int32_t strideg,
                                           float *X, int32_t ldx,
                                           int32_t stridex,
                                           const LaneDesc *lanes,
                                           int32_t num_lanes) {
  int32_t lane = blockIdx.x;
  LaneDesc desc = lanes[lane];
  int32_t channel = desc.channel;

  if (desc.current_frame == 0) {
    // offset to channel
    gamma = gamma + channel * strideg;
    X = X + channel * stridex;

    // initialize stashes to zero
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < num_gauss;
         i += blockDim.y * blockDim.x) {
      gamma[i] = 0.0f;
    }

    for (int i = threadIdx.y; i < num_gauss; i += blockDim.y) {
      for (int j = threadIdx.x; j < feat_dim; j += blockDim.x) {
        X[i * ldx + j] = 0.0f;
      }
    }
  }
}

void initialize_channels(int32_t num_gauss, int32_t feat_dim, float *gamma,
                         int32_t strideg, float *X, int32_t ldx,
                         int32_t stridex, const LaneDesc *lanes,
                         int32_t num_lanes) {
  dim3 threads(32, 32);
  int32_t blocks = num_lanes;

  initialize_channels_kernel<<<blocks, threads>>>(
      num_gauss, feat_dim, gamma, strideg, X, ldx, stridex, lanes, num_lanes);
}

__global__ void apply_and_update_stash_kernel(
    int32_t num_gauss, int32_t feat_dim, float *gamma, float *gamma_stash,
    int32_t strideg, float *X, int32_t ldx, int32_t stridex, float *X_stash,
    int32_t lds, int32_t strides, const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.x;
  LaneDesc desc = lanes[lane];
  int32_t channel = desc.channel;

  // offset to lane
  gamma = gamma + lane * strideg;
  X = X + lane * stridex;

  // offset to channel
  gamma_stash = gamma_stash + channel * strideg;
  X_stash = X_stash + channel * strides;

  // add gamma and stash together then store in both
  // use both x and y threads in the block for this
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < num_gauss;
       i += blockDim.y * blockDim.x) {
    float val = gamma_stash[i] + gamma[i];
    gamma_stash[i] = gamma[i] = val;
  }

  // add x and stash together then store in both
  for (int i = threadIdx.y; i < num_gauss; i += blockDim.y) {
    for (int j = threadIdx.x; j < feat_dim; j += blockDim.x) {
      float val = X[i * ldx + j] + X_stash[i * lds + j];
      X[i * ldx + j] = X_stash[i * lds + j] = val;
    }
  }
}

void apply_and_update_stash(int32_t num_gauss, int32_t feat_dim, float *gamma,
                            float *gamma_stash, int32_t strideg, float *X,
                            int32_t ldx, int32_t stridex, float *X_stash,
                            int32_t lds, int32_t strides, const LaneDesc *lanes,
                            int32_t num_lanes) {
  dim3 threads(32, 32);
  int32_t blocks = num_lanes;

  apply_and_update_stash_kernel<<<blocks, threads>>>(
      num_gauss, feat_dim, gamma, gamma_stash, strideg, X, ldx, stridex,
      X_stash, lds, strides, lanes, num_lanes);
}

}  // end namespace kaldi
#endif

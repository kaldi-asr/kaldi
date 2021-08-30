// cudafeat/feature-online-cmvn-cuda.cu
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
#include "cudafeat/feature-online-cmvn-cuda.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

__host__ __device__ inline float2 operator-(const float2 &a, const float2 &b) {
  float2 retval;
  retval.x = a.x - b.x;
  retval.y = a.y - b.y;
  return retval;
}
__host__ __device__ inline float2 operator+(const float2 &a, const float2 &b) {
  float2 retval;
  retval.x = a.x + b.x;
  retval.y = a.y + b.y;
  return retval;
}

#if __CUDA_ARCH__ == 750
__launch_bounds__ (1024, 1)
#else
__launch_bounds__ (1024, 2)
#endif
__global__ void compute_cmvn_stats_kernel(const float *data, int32_t ldd,
                                          int32_t num_frames, int32_t feat_dim,
                                          float *stats, int32_t lds) {
  typedef cub::BlockScan<float2, 1024> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int32_t feat = blockIdx.x;

  float2 running_sum = {0.0f, 0.0f};
  // for each frame, keep threads alive for cub
  for (int32_t r = 0; r < num_frames; r += blockDim.x) {
    int32_t rid = r + threadIdx.x;

    float val = 0.0f;

    if (rid < num_frames) {
      // uncoalesced, could transpose data or do some shared memory swizzling...
      val = data[rid * ldd + feat];
    }

    float2 sum = {val, val * val};  // this elements value and value squared

    float2 psum;   // row prefix sum
    float2 total;  // total count
    BlockScan(temp_storage).InclusiveSum(sum, psum, total);

    // offset by running sum
    psum = psum + running_sum;
    // increase running sum by new total
    running_sum = running_sum + total;

    // un-coalesced
    if (rid < num_frames) {
      reinterpret_cast<float2 *>(&stats[rid * lds])[feat] = psum;
    }
  }
}

__global__ void apply_cmvn_kernel(
    int32_t cmvn_window, bool var_norm, bool mean_norm, const float *feat_in,
    int32_t ldi, int32_t num_rows, int32_t num_cols,
    const float *__restrict__ stats, int32_t lds,
    const float *__restrict__ global_stats, int32_t ldg, int32_t global_frames,
    const float *__restrict__ speaker_stats, int32_t ldss,
    int32_t speaker_frames, float *feat_out, int32_t ldo) {
  int32_t r = blockIdx.x;

  for (int c = threadIdx.x; c < num_cols; c += blockDim.x) {
    float2 frame_stats =
        reinterpret_cast<const float2 __restrict__*>(&stats[r * lds])[c];

    float val = feat_in[r * ldi + c];

    float window_length = min(r + 1, cmvn_window);

    // we have to subtract row r-cmvn_window stats
    if (r >= cmvn_window) {
      // window starting row
      int32_t o = r - cmvn_window;

      // stats at the start row of the window that must be removed
      float2 ostats =
          reinterpret_cast<const float2 __restrict__*>(&stats[o * lds])[c];

      // remove start of the window stats
      frame_stats = frame_stats - ostats;
    }

    // Smooth stats by speaker frames if necessary
    float smooth_frames = cmvn_window - window_length;
    if (smooth_frames > 0 && speaker_frames > 0) {
      float count_from_speaker = min(smooth_frames, (float)speaker_frames);
      float speaker_count = speaker_stats[num_cols];

      if (count_from_speaker > 0.0) {
        float alpha = count_from_speaker / speaker_count;

        frame_stats.x += alpha * speaker_stats[c];         // update mean
        frame_stats.y += alpha * speaker_stats[ldss + c];  // update variance
        window_length += alpha * speaker_count;  // update window length

        // recompute smooth frames now that we have speaker stats
        smooth_frames = cmvn_window - window_length;
      }
    }

    // Smooth stats by global frames if necessary
    if (smooth_frames > 0 && global_frames > 0) {
      float count_from_global = min(smooth_frames, (float)global_frames);
      float global_count = global_stats[num_cols];

      if (count_from_global > 0.0) {
        float alpha = count_from_global / global_count;

        frame_stats.x += alpha * global_stats[c];        // update mean
        frame_stats.y += alpha * global_stats[ldg + c];  // update variance
        window_length += alpha * global_count;           // update window length
      }
    }

    float mean = frame_stats.x / window_length;
    float var = frame_stats.y / window_length - mean * mean;

    float floor = 1e-20;
    if (var < floor)  // avoid dividing by zero
      var = floor;

    if (!var_norm) {
      // skip variance normalization
      var = 1.0f;
    }
    if (!mean_norm) {
      assert(false);
      // skip mean normalization
      mean = 0.0f;
    }

    // shift by mean and scale by variance
    feat_out[r * ldo + c] = (val - mean) / sqrtf(var);
  }
}

namespace kaldi {

void CudaOnlineCmvn::ComputeFeatures(const CuMatrixBase<BaseFloat> &feats_in,
                               CuMatrix<BaseFloat> *feats_out) {
  int32_t num_frames = feats_in.NumRows();
  int32_t feat_dim = feats_in.NumCols();
  feats_out->Resize(num_frames, feat_dim, kUndefined);

  CuMatrix<float> stats(num_frames, feat_dim * 2, kUndefined);

  int threads = 1024;
  int blocks = feat_dim;

  // compute windowed sum/sum2 prefix sum along column of feats
  compute_cmvn_stats_kernel<<<blocks, threads>>>(
      feats_in.Data(), feats_in.Stride(), num_frames, feat_dim, stats.Data(),
      stats.Stride());
  CU_SAFE_CALL(cudaGetLastError());

  threads = (feat_dim + 31) / 32 * 32;  // round up to 32 threads
  if (threads > 1024) threads = 1024;

  const CuMatrix<float> &gstats = cmvn_state_.global_cmvn_stats;
  const CuMatrix<float> &sstats = cmvn_state_.speaker_cmvn_stats;

  int global_frames = opts_.global_frames;
  int speaker_frames = opts_.speaker_frames;

  if (gstats.NumRows() == 0) global_frames = 0;
  if (sstats.NumRows() == 0) speaker_frames = 0;

  // apply cmvn
  apply_cmvn_kernel<<<num_frames, threads>>>(
      opts_.cmn_window, opts_.normalize_variance, opts_.normalize_mean,
      feats_in.Data(), feats_in.Stride(), num_frames, feat_dim, stats.Data(),
      stats.Stride(), gstats.Data(), gstats.Stride(), global_frames,
      sstats.Data(), sstats.Stride(), speaker_frames, feats_out->Data(),
      feats_out->Stride());
  CU_SAFE_CALL(cudaGetLastError());
}
}

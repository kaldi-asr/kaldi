// cudafeat/feature-online-batched-cmvn-cuda-kernels.cu
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
//
#include <cub/cub.cuh>
#include "cudafeat/feature-online-batched-cmvn-cuda-kernels.h"

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

__device__ inline void atomicAdd(float2 *addr, float2 val) {
  atomicAdd(reinterpret_cast<float *>(addr), val.x);
  atomicAdd(reinterpret_cast<float *>(addr) + 1, val.y);
}

__device__ inline void operator+=(float2 &a, float2 &b) {
  // overloading +=
  a.x += b.x;
  a.y += b.y;
}

namespace kaldi {
// threadIdx.x = frame  (up to 1024?)
// blockIdx.x = feature
// blockIdx.y = batch id
__global__ void compute_cmvn_stats_kernel(
    int32_t feat_dim, int32_t chunk_size, int32_t stats_coarsening_factor,
    int32_t cmn_window, const float *in_data, int32_t ldi, int32_t stridei,
    float *stats_data, int32_t lds, const LaneDesc *lanes, int32_t num_lanes) {
  typedef cub::BlockScan<float2, 1024> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int32_t lane = blockIdx.y;
  int32_t feat = blockIdx.x;  // feature for this block
  int32_t tidx = threadIdx.x;

  // width of a window of stats data
  int32_t num_fragments = (chunk_size + cmn_window) / stats_coarsening_factor;

  // function to compute window location based on frame
  auto SIDX = [&](int frame, int feat) {
    int row = feat;
    int col = (frame / stats_coarsening_factor) % num_fragments;
    return row * num_fragments + col;
  };

  LaneDesc desc = lanes[lane];
  ChannelId channel = desc.channel;
  int32_t num_chunk_frames = desc.num_chunk_frames;

  // compute memory offsets for batch
  float2 *sdata = reinterpret_cast<float2 *>(stats_data + channel * lds);

  // batch is rows, cols is chunk_size x feat_dim, where feat_dim is
  // padded to ldi
  const float *idata = in_data + lane * stridei;

  // starting frame of audio
  int32_t start_frame = desc.current_frame;

  float2 running_sum = {0.0f, 0.0f};

  // load previous running sum if this is not the first frame
  if (start_frame > 0) running_sum = sdata[SIDX(start_frame - 1, feat)];

  // for each frame compute prefix sum
  for (int32_t f = 0; f < num_chunk_frames; f += blockDim.x) {
    int frame = f + tidx;

    float val = 0.0f;
    if (frame < num_chunk_frames) {
      // uncoalesced
      val = idata[frame * ldi + feat];
    }

    float2 sum = {val, val * val};
    float2 psum;   // row prefix sum
    float2 total;  // total count

    BlockScan(temp_storage).InclusiveSum(sum, psum, total);

    // offset by running sum
    psum = psum + running_sum;

    // increase running sum by new total
    running_sum = running_sum + total;

    // The last thread of each fragement will write their value to stats
    bool write = (frame < num_chunk_frames && frame % stats_coarsening_factor ==
                                                  stats_coarsening_factor - 1);

    // last frame will always write
    // this fagment may not have full stats
    // use our frame to fill in those stats
    if (f == num_chunk_frames - 1) {
      // This thread will write
      write = true;

      // number of frames in my fragement with stats
      int32_t in_frame = f % stats_coarsening_factor + 1;
      // number of frames int my fragement without stats
      int32_t not_in_frame = stats_coarsening_factor - in_frame;

      // multiply this frame stats by the number of frames not counted
      float2 add = make_float2(sum.x * not_in_frame, sum.y * not_in_frame);

      // if the fragment is full add will be (0,0)
      // Add in stats
      psum += add;
    }

    if (write) {
      // un-coalesced
      sdata[SIDX(start_frame + frame, feat)] = psum;
    }
  }
}

// For each channel in batch size, compute coarsened stats in rolling
// window
void compute_cmvn_stats(int32_t feat_dim, int32_t chunk_size,
                        int32_t stats_coarsening_factor, int32_t cmn_window,
                        const float *in_data, int32_t ldi, int32_t stridei,
                        float *stats_data, int32_t lds, const LaneDesc *lanes,
                        int32_t num_lanes) {
  int threads = 1024;
  dim3 blocks(feat_dim, num_lanes);

  compute_cmvn_stats_kernel<<<blocks, threads>>>(
      feat_dim, chunk_size, stats_coarsening_factor, cmn_window, in_data, ldi,
      stridei, stats_data, lds, lanes, num_lanes);
};

// threadIdx.x = feature (32?)
// threadIdx.y, blockIdx.x = frame
// blockIdx.y = batch id
__global__ void apply_cmvn_kernel(
    int32_t cmvn_window, bool var_norm, bool mean_norm, int32_t feat_dim,
    int32_t chunk_size, int32_t stats_coarsening_factor,
    const float *__restrict__ in_data, int32_t ldi, int32_t stridei,
    const float *__restrict__ stats_data, int32_t lds,
    const float *__restrict__ global_stats, int32_t ldg, int32_t global_frames,
    const float *__restrict__ speaker_stats, int32_t ldss,
    int32_t speaker_frames, float *out_data, int32_t ldo, int32_t strideo,
    const LaneDesc *lanes, int32_t num_lanes) {
  int32_t lane = blockIdx.y;
  LaneDesc desc = lanes[lane];
  ChannelId channel = desc.channel;

  // compute memory offsets for batch
  const float2 *sdata =
      reinterpret_cast<const float2 *>(stats_data + channel * lds);
  // batch is rows, cols is chunk_size x feat_dim, where feat_dim is
  // padded to ldi
  const float *idata = in_data + lane * stridei;
  float *odata = out_data + lane * strideo;

  // width of a window of stats data
  int32_t num_fragments = (chunk_size + cmvn_window) / stats_coarsening_factor;

  // function to compute window location based on frame
  auto SIDX = [&](int frame, int feat) {
    int row = feat;
    int col = (frame / stats_coarsening_factor) % num_fragments;
    return row * num_fragments + col;
  };

  int32_t current_frame = desc.current_frame;
  int32_t num_chunk_frames = desc.num_chunk_frames;
  for (int f = blockIdx.x * blockDim.y + threadIdx.y; f < num_chunk_frames;
       f += blockDim.y * gridDim.x) {
    int frame = current_frame + f;

    for (int feat = threadIdx.x; feat < feat_dim; feat += blockDim.x) {
      // Compute stats for frame
      float2 frame_stats = sdata[SIDX(frame, feat)];
      // load value
      float val = idata[f * ldi + feat];

      // compute window length
      float window_length = min(frame + 1, cmvn_window);

      // possibly remove stats -cmvn window away
      if (frame >= cmvn_window) {
        float2 old_frame_stats = sdata[SIDX(frame - cmvn_window, feat)];
        frame_stats = frame_stats - old_frame_stats;
      }

      // Smooth stats by speaker frames if necessary
      float smooth_frames = cmvn_window - window_length;
      if (smooth_frames > 0 && speaker_frames > 0) {
        float count_from_speaker = min(smooth_frames, (float)speaker_frames);
        float speaker_count = speaker_stats[feat_dim];

        if (count_from_speaker > 0.0) {
          float alpha = count_from_speaker / speaker_count;

          frame_stats.x += alpha * speaker_stats[feat];  // update mean
          frame_stats.y +=
              alpha * speaker_stats[ldss + feat];  // update variance
          window_length += alpha * speaker_count;  // update window length

          // recompute smooth frames now that we have speaker stats
          smooth_frames = cmvn_window - window_length;
        }
      }  // end speaker smooth

      // Smooth stats by global frames if necessary
      if (smooth_frames > 0 && global_frames > 0) {
        float count_from_global = min(smooth_frames, (float)global_frames);
        float global_count = global_stats[feat_dim];

        if (count_from_global > 0.0) {
          float alpha = count_from_global / global_count;

          frame_stats.x += alpha * global_stats[feat];        // update mean
          frame_stats.y += alpha * global_stats[ldg + feat];  // update variance
          window_length += alpha * global_count;  // update window length
        }
      }  // end global smooth

      float mean = frame_stats.x / window_length;
      float var = frame_stats.y / window_length - mean * mean;

      float floor = 1e-20;
      if (var < floor) {
        // avoid dividing by zero
        var = floor;
      }
      if (!var_norm) {
        // skip variance normalization
        var = 1.0f;
      }
      if (!mean_norm) {
        // skip mean normalization
        mean = 0.0f;
      }

      // shift by mean and scale by variance
      float oval = (val - mean) / sqrtf(var);

      odata[f * ldo + feat] = oval;
    }  // end feat loop
  }    // end frame loop
}

void apply_cmvn(int32_t cmvn_window, bool var_norm, bool mean_norm,
                int32_t feat_dim, int32_t chunk_size,
                int32_t stats_coarsening_factor, const float *in_data,
                int32_t ldi, int32_t stridei, const float *stats_data,
                int32_t lds, const float *global_stats, int32_t ldg,
                int32_t global_frames, const float *speaker_stats, int32_t ldss,
                int32_t speaker_frames, float *out_data, int32_t ldo,
                int32_t strideo, const LaneDesc *lanes, int32_t num_lanes) {
  // round threads to neared warp
  int threadsx = 64;
  int threadsy = 512 / threadsx;
  dim3 threads(threadsx, threadsy);

  int blocksx = (chunk_size + threadsy - 1) / threadsy;
  int blocksy = num_lanes;
  dim3 blocks(blocksx, blocksy);

  apply_cmvn_kernel<<<blocks, threads>>>(
      cmvn_window, var_norm, mean_norm, feat_dim, chunk_size,
      stats_coarsening_factor, in_data, ldi, stridei, stats_data, lds,
      global_stats, ldg, global_frames, speaker_stats, ldss, speaker_frames,
      out_data, ldo, strideo, lanes, num_lanes);
}

}  // namespace kaldi

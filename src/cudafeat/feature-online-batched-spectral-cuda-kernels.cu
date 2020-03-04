// cudafeature/feature-online-batched-spectral-cuda-kernels.cu
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens, Levi Barnes
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
#include <nvToolsExt.h>
#include <cub/cub.cuh>
#endif

#include "cudafeat/feature-online-batched-spectral-cuda-kernels.h"
#include "cudafeat/lane-desc.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {

// Mimics the functionality of mel_banks_compute_kernel
//  (found in feature-spectral-cuda.cu). The 3rd
//  dimension (z) of the block grid gives the hardware
//  "lane". lanes tells us which channel is in this lane,
//  what current frame and sample are processed in this
//  batch, etc.
__global__ void batched_mel_banks_compute_kernel(
    const LaneDesc *lanes, int32_t n_lanes, int32_t max_chunk_frames,
    float energy_floor, int32 *offsets, int32 *sizes, float **vecs,
    const float *feats, int32_t ldf, float *mels, int32_t ldm, bool use_log) {
  // Specialize WarpReduce for type float
  typedef cub::WarpReduce<float> WarpReduce;
  // Allocate WarpReduce shared memory for 8 warps
  __shared__ typename WarpReduce::TempStorage temp_storage[8];

  // warp will work together to compute sum
  int tid = threadIdx.x;
  int wid = threadIdx.y;
  // blocks in the x dimension take different bins
  int bin = blockIdx.x;
  // frame is a combination of blocks in the y dimension and threads in the y
  // dimension
  int frame = blockIdx.y * blockDim.y + threadIdx.y;
  int lane = blockIdx.z;

  LaneDesc desc = lanes[lane];
  int num_frames = desc.num_chunk_frames;

  // TODO get offsets, sizes, and vecs from laneInfo?
  int offset = offsets[bin];
  int size = sizes[bin];
  const float *v = vecs[bin];
  const float *w = feats + frame * ldf + lane * max_chunk_frames * ldf + offset;

  // perfom local sum
  float sum = 0;
  if (frame < num_frames) {  // exclude frames beyond the end
    for (int idx = tid; idx < size; idx += 32) {
      sum += v[idx] * w[idx];
    }
  }

  // Sum in cub
  sum = WarpReduce(temp_storage[wid]).Sum(sum);
  if (tid == 0 && frame < num_frames) {
    if (use_log) {
      // avoid log of zero
      if (sum < energy_floor) sum = energy_floor;
      float val = logf(sum);
      mels[lane * max_chunk_frames * ldm + frame * ldm + bin] = val;
    } else {
      mels[lane * max_chunk_frames * ldm + frame * ldm + bin] = sum;
    }
  }
}
// Mimics the functionality of apply_lifter_and_floor_energy
//  (found in feature-spectral-cuda.cu) for a chunk of data
//  from several audio channels. The 2nd dimension
//  (y) of the block grid gives the hardware "lane".
//  The lanes array tells us which channel is in this lane,
//  what current frame and sample are processed in this
//  batch, etc.
__global__ void batched_apply_lifter_and_floor_energy_kernel(
    const LaneDesc *lanes, int32_t n_lanes, int32_t max_chunk_frames,
    int num_cols, float cepstral_lifter, bool use_energy, float energy_floor,
    float *log_energy, int32_t ldl, float *lifter_coeffs, float *features,
    int32_t ldf) {
  int thread_id = threadIdx.x;
  int frame = blockIdx.x;
  int lane = blockIdx.y;

  LaneDesc desc = lanes[lane];
  if (frame > desc.num_chunk_frames) return;

  float *feats = features + frame * ldf + lane * max_chunk_frames * ldf;

  // apply lifter coefficients
  if (cepstral_lifter != 0.0f) {
    for (int c = thread_id; c < num_cols; c += CU1DBLOCK) {
      float lift = lifter_coeffs[c];
      float f = feats[c];
      feats[c] = f * lift;
    }
  }

  // Thread 0 for each frame will apply energy
  if (use_energy && thread_id == 0) {
    float energy = log_energy[frame + ldl * lane];
    float log_energy_floor = log(energy_floor);

    if (energy_floor > 0.0f && energy < log_energy_floor) {
      energy = log_energy_floor;
    }
    feats[0] = energy;
  }
}
// Mimics the functionality of process_window_kernel
//  (found in feature-spectral-cuda.cu) for a chunk of data
//  from several audio channels. The 2nd dimension
//  (y) of the block grid gives the hardware "lane".
//  The lanes array tells us which channel is in this lane,
//  what current frame and sample are processed in this
//  batch, etc.
__global__ void batched_process_window_kernel(
    const LaneDesc *lanes, int32_t n_lanes, int32_t max_chunk_frames,
    int frame_length, float dither, float energy_floor, bool remove_dc_offset,
    float preemph_coeff, bool need_raw_log_energy, float *log_energy_pre_window,
    int32_t lde, const float *windowing, float *tmp_windows, int32_t ldt,
    float *windows, int32_t ldw) {
  // Specialize WarpReduce for type float
  typedef cub::BlockReduce<float, CU1DBLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_id = threadIdx.x;
  int row = blockIdx.x;
  int lane = blockIdx.y;

  LaneDesc desc = lanes[lane];
  if (row >= desc.num_chunk_frames) return;

  float *tmp_window = tmp_windows + row * ldt + lane * max_chunk_frames * ldt;
  float *window = windows + row * ldw + lane * max_chunk_frames * ldw;

  __shared__ float ssum;

  float sum = 0;
  float wdot = 0;

  for (int idx = thread_id; idx < frame_length; idx += CU1DBLOCK) {
    // tmp_window contains optional dither.  Apply that on read.
    float wval = window[idx];
    if (dither != 0.0f) {
      wval += tmp_window[idx] * dither;
    }
    // compute local sum for removing dc offset
    sum += wval;
    // compute dot product for log energy
    wdot += wval * wval;

    float windowing_mul = 1;
    if (remove_dc_offset == false && preemph_coeff == 0.0f) {
      // we are done here so set windowing multiplication on write.
      windowing_mul = windowing[idx];
    }

    // write dithered output
    window[idx] = wval * windowing_mul;
  }
  __syncthreads();
  if (remove_dc_offset) {
    // we will recompute this below
    wdot = 0.0f;
    // use cub to reduce
    sum = BlockReduce(temp_storage).Sum(sum);

    // broadcast sum to entire block
    if (thread_id == 0) ssum = sum;
    __syncthreads();

    sum = -ssum / frame_length;
    for (int idx = thread_id; idx < frame_length; idx += CU1DBLOCK) {
      float windowing_mul = 1;
      float *out = window;
      if (preemph_coeff == 0.0f) {
        // we are done here so apply windowing
        windowing_mul = windowing[idx];
      } else {
        // write to temp window as we will copy back into window
        // when doing pre-emphasis
        out = tmp_window;
      }
      // updated window value
      float wval = window[idx] + sum;

      // compute new dot product with dc offset removed
      wdot += wval * wval;

      // write output
      out[idx] = wval * windowing_mul;
    }
  }
  __syncthreads();

  // if pointer is not NULL we will set energy to either
  // the computed energy or 0 depending on need_raw_log_energy
  if (log_energy_pre_window != NULL) {
    float energy = 0.0f;

    if (need_raw_log_energy) {
      // must sync to use retemp_storage
      if (remove_dc_offset) __syncthreads();
      // use cub to reduce
      wdot = BlockReduce(temp_storage).Sum(wdot);

      energy = max(wdot, energy_floor);
    }

    if (thread_id == 0) {
      log_energy_pre_window[row + lane * lde] = log(energy);
    }
  }

  // TODO this could be more efficient using shared memory instead of
  // tmp_window.
  if (preemph_coeff != 0.0f) {
    // wait for tmp_window to be computed
    __threadfence();
    __syncthreads();
    // starting thread idx at 0 to keep writes aligned.
    // unaligned reads are less painful then unaligned writes
    for (int idx = thread_id; idx < frame_length; idx += CU1DBLOCK) {
      float wval = tmp_window[idx];
      float prev_window = wval;
      if (idx > 0) {
        prev_window = tmp_window[idx - 1];
      }
      // use __fmul_rn to match CPU
      // window[idx] = (wval - preemph_coeff*prev_window) * windowing[idx];
      window[idx] =
          (wval - __fmul_rn(preemph_coeff, prev_window)) * windowing[idx];
    }
  }
}

__host__ __device__ inline int32 FirstSampleOfFrame(int32 frame,
                                                    int32 frame_shift,
                                                    int32 window_size,
                                                    bool snip_edges) {
  if (snip_edges) {
    return frame * frame_shift;
  } else {
    int32 midpoint_of_frame = frame_shift * frame + frame_shift / 2,
          beginning_of_frame = midpoint_of_frame - window_size / 2;
    return beginning_of_frame;
  }
}

// Mimics the functionality of extract_window_kernel
//  (found in feature-spectral-cuda.cu) for a chunk of data
//  from several audio channels. The 2nd dimension
//  (y) of the block grid gives the hardware "lane".
//  The lanes array tells us which channel is in this lane,
//  what current frame and sample are processed in this
//  batch, etc.
//  Extra samples not processed in this chunk are moved to
//  "stash" where they'll be pre-pended to the next chunk
//  from this channel
__global__ void batched_extract_window_kernel(
    const LaneDesc *lanes, int32_t num_lanes, int32 frame_shift,
    int32 frame_length, int32 frame_length_padded, bool snip_edges,
    const BaseFloat *__restrict__ wave, int32_t ldw,
    BaseFloat *__restrict__ windows, int32_t window_size, int32_t wlda,
    BaseFloat *stash, int32_t ssize, int32_t lds) {
  // local frame number
  int32_t fidx = blockIdx.x;
  int32_t tidx = threadIdx.x;
  int32_t lane = blockIdx.y;

  const LaneDesc desc = lanes[lane];
  ChannelId channel = desc.channel;
  // This is the current sample that is pointed to by wave
  int32_t current_sample = desc.current_sample;
  // current frame we are computing in global space
  int32_t current_frame = desc.current_frame;

  // global frame number computed by this block
  int32_t global_frame = current_frame + fidx;

  int32_t num_chunk_samples = desc.num_chunk_samples;

  if (fidx > desc.num_chunk_frames) return;

  // offset input/output by channels or lanes
  stash = stash + channel * lds;
  wave = wave + lane * ldw;
  BaseFloat *window = windows + fidx * wlda + gridDim.x * lane * wlda;

  // This is the first sample needed to compute this frame
  int32_t start_sample =
      FirstSampleOfFrame(global_frame, frame_shift, window_size, snip_edges);

  // Sample offset is how much we have to offset our index
  // into the input wave.
  int32_t wave_start = start_sample - current_sample;

  // wave_start and wave_end are start and end indexes into 'wave', for the
  // piece of wave that we're trying to extract.
  int32_t wave_end = wave_start + frame_length;

  // wave_start will be negative on successive chunks as we need
  // to grab context from stash.
  if ((current_frame > 0 || wave_start >= 0) && wave_end <= num_chunk_samples) {
    // the normal case-- no edge effects to consider.
    for (int i = tidx; i < frame_length; i += blockDim.x) {
      int32_t widx = wave_start + i;
      BaseFloat val;
      if (widx >= 0) {
        val = wave[widx];
      } else {
        // widx is negative. Add it to the stash size
        // to get the correct index into the stash
        int32_t sidx = ssize + widx;
        val = stash[sidx];
      }
      window[i] = val;
    }
  } else {
    // Deal with any end effects by reflection, if needed.  This code will only
    // be reached for about two frames per utterance, so we don't concern
    // ourselves excessively with efficiency.
    for (int s = tidx; s < frame_length; s += blockDim.x) {
      int32 s_in_wave = wave_start + s;
      while (s_in_wave < 0 || s_in_wave >= num_chunk_samples) {
        // reflect around the beginning or end of the wave.
        // e.g. -1 -> 0, -2 -> 1.
        // dim -> dim - 1, dim + 1 -> dim - 2.
        // the code supports repeated reflections, although this
        // would only be needed in pathological cases.
        if (s_in_wave < 0)
          s_in_wave = -s_in_wave - 1;
        else
          s_in_wave = 2 * num_chunk_samples - 1 - s_in_wave;
      }
      window[s] = wave[s_in_wave];
    }
  }

  if (frame_length_padded > frame_length) {
    for (int i = frame_length + tidx; i < frame_length_padded;
         i += blockDim.x) {
      window[i] = 0.0f;
    }
  }
}
// For each frame
//   compute logf(dot(signal_frame, signal_frame))
// This is the batched version. The y-dimension of the grid
// give the lane number
__global__ void batched_dot_log_kernel(int32_t max_chunk_frames,
                                       int32_t frame_length,
                                       float *signal_frame, int32_t lds,
                                       float *signal_log_energy, int32_t lde) {
  // Specialize WarpReduce for type float
  typedef cub::BlockReduce<float, CU1DBLOCK> BlockReduce;
  // Allocate WarpReduce shared memory for 8 warps
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int32_t frame = blockIdx.x;
  int32_t tid = threadIdx.x;
  int32_t lane = blockIdx.y;

  float *in = signal_frame + frame * lds + max_chunk_frames * lane * lds;
  float sum = 0;

  // preform local dot product
  for (int32_t i = tid; i < frame_length; i += blockDim.x) {
    float val = in[i];
    sum += val * val;
  }

  // reduce using cub
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    signal_log_energy[frame + lane * lde] = logf(sum);
  }
}

__global__ void batched_update_stash_kernel(const LaneDesc *lanes,
                                            int32_t num_lanes,
                                            const BaseFloat *wave, int32_t ldw,
                                            BaseFloat *stash, int32_t num_stash,
                                            int32_t lds) {
  int32_t lane = blockIdx.x;
  LaneDesc desc = lanes[lane];
  int32_t channel = desc.channel;
  int32_t num_chunk_samples = desc.num_chunk_samples;

  // offset memory by lane or channel
  wave = wave + lane * ldw;
  stash = stash + channel * lds;

  int32_t sample_offset = num_chunk_samples - num_stash;
  for (int i = threadIdx.x; i < num_stash; i += blockDim.x) {
    int32_t idx = sample_offset + i;

    float val;
    if (idx < 0) {
      // data must come from old stash
      val = stash[idx + num_stash];
    } else {
      // data comes from new wave
      val = wave[idx];
    }

    __syncthreads();

    stash[i] = val;
  }
}
// Each threadblock computes a different row of the matrix.
// Threads in the same block compute the row collaboratively.
// This kernel must be called out of place (A_in!=A_out).
__global__ void power_spectrum_kernel(int row_length, const float *A_in, int32_t ldi,
                                      float *A_out, int32_t ldo,
                                      bool use_power) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  const float *Ar = A_in + block_id * ldi;
  float *Aw = A_out + block_id * ldo;

  int half_length = row_length / 2;
  for (int idx = thread_id; idx < half_length; idx += CU1DBLOCK) {
    // ignore special case
    if (idx == 0) continue;

    float2 val = reinterpret_cast<const float2 *>(Ar)[idx];
    float ret = val.x * val.x + val.y * val.y;
    if (use_power) {
      Aw[idx] = ret;
    } else {
      Aw[idx] = sqrtf(ret);
    }
  }

  // handle special case
  if (threadIdx.x == 0) {
    float real = Ar[0];
    // cufft puts this at the end, this is different than kaldi does with its
    // own
    // internal implementation
    float im = Ar[row_length];

    if (use_power) {
      Aw[0] = real * real;
      Aw[half_length] = im * im;
    } else {
      Aw[0] = fabs(real);
      Aw[half_length] = fabs(im);
    }
  }
}


void cuda_power_spectrum(int32_t max_chunk_frames, int32_t num_lanes,
                         int row_length, const float *A_in, int32_t ldi,
                         float *A_out, int32_t ldo, bool use_power) {
  power_spectrum_kernel<<<max_chunk_frames * num_lanes, CU1DBLOCK>>>(
      row_length, A_in, ldi, A_out, ldo, use_power);
}

void cuda_mel_banks_compute(const LaneDesc *lanes, int32_t num_lanes,
                            int32_t max_chunk_frames, int32_t num_bins,
                            float energy_floor, int32 *offsets, int32 *sizes,
                            float **vecs, const float *feats, int32_t ldf,
                            float *mels, int32_t ldm, bool use_log) {
  dim3 Bl(32, 8);
  dim3 Gr(num_bins, (max_chunk_frames + Bl.y - 1) / Bl.y, num_lanes);
  batched_mel_banks_compute_kernel<<<Gr, Bl>>>(
      lanes, num_lanes, max_chunk_frames, energy_floor, offsets, sizes, vecs,
      feats, ldf, mels, ldm, use_log);
}

void cuda_apply_lifter_and_floor_energy(const LaneDesc *lanes,
                                        int32_t num_lanes,
                                        int32_t max_chunk_frames, int num_cols,
                                        float cepstral_lifter, bool use_energy,
                                        float energy_floor, float *log_energy,
                                        int32_t ldl, float *lifter_coeffs,
                                        float *features, int32_t ldf) {
  dim3 Gr(max_chunk_frames, num_lanes);
  batched_apply_lifter_and_floor_energy_kernel<<<Gr, CU1DBLOCK>>>(
      lanes, num_lanes, max_chunk_frames, num_cols, cepstral_lifter, use_energy,
      energy_floor, log_energy, ldl, lifter_coeffs, features, ldf);
}

void cuda_process_window(const LaneDesc *lanes, int32_t num_lanes,
                         int32_t max_chunk_frames, int frame_length,
                         float dither, float energy_floor,
                         bool remove_dc_offset, float preemph_coeff,
                         bool need_raw_log_energy, float *log_energy_pre_window,
                         int32_t lde, const float *windowing,
                         float *tmp_windows, int32_t ldt, float *windows,
                         int32_t ldw) {
  dim3 Gr(max_chunk_frames, num_lanes);
  int Bl = CU1DBLOCK;
  batched_process_window_kernel<<<Gr, Bl>>>(
      lanes, num_lanes, max_chunk_frames, frame_length, dither, energy_floor,
      remove_dc_offset, preemph_coeff, need_raw_log_energy,
      log_energy_pre_window, lde, windowing, tmp_windows, ldt, windows, ldw);
}

void cuda_extract_window(const LaneDesc *lanes, int32_t num_lanes,
                         int32_t max_chunk_frames, int32 frame_shift,
                         int32 frame_length, int32 frame_length_padded,
                         bool snip_edges, const float *wave, int32_t ldw,
                         float *windows, int32_t window_size, int32_t wlda,
                         BaseFloat *stash, int32_t ssize, int32_t lds) {
  dim3 Gr(max_chunk_frames, num_lanes);
  int Bl = CU1DBLOCK;
  batched_extract_window_kernel<<<Gr, Bl>>>(
      lanes, num_lanes, frame_shift, frame_length, frame_length_padded,
      snip_edges, wave, ldw, windows, window_size, wlda, stash, ssize, lds);
}

void cuda_dot_log(int32_t max_chunk_frames, int32_t num_lanes,
                  int32_t frame_length, float *signal_frame, int32_t lds,
                  float *signal_log_energy, int32_t lde) {
  dim3 Gr(max_chunk_frames, num_lanes);
  batched_dot_log_kernel<<<Gr, CU1DBLOCK>>>(max_chunk_frames, frame_length,
                                            signal_frame, lds,

                                            signal_log_energy, lde);
}

void cuda_update_stash(const LaneDesc *lanes, int32_t num_lanes,
                       const BaseFloat *wave, int32_t ldw, BaseFloat *stash,
                       int32_t num_stash, int32_t lds) {
  int Gr = num_lanes;
  int Bl = 1024;
  batched_update_stash_kernel<<<Gr, Bl>>>(lanes, num_lanes, wave, ldw, stash,
                                          num_stash, lds);
}
}  // namespace kaldi

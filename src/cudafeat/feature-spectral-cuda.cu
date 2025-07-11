// cudafeature/feature-spectral-cuda.cu
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

#include "cudafeat/feature-spectral-cuda.h"

#include <nvToolsExt.h>
#include <cub/cub.cuh>

#include "cudamatrix/cu-rand.h"

// Each thread block processes a unique frame
// threads in the same threadblock collaborate to
// compute the frame together.
__global__ void apply_lifter_and_floor_energy(
    int num_frames, int num_cols, float cepstral_lifter, bool use_energy,
    float energy_floor, float *log_energy, float *lifter_coeffs,
    float *features, int32_t ldf) {
  int thread_id = threadIdx.x;
  int frame = blockIdx.x;

  float *feats = features + frame * ldf;

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
    float energy = log_energy[frame];
    float log_energy_floor = log(energy_floor);

    if (energy_floor > 0.0f && energy < log_energy_floor) {
      energy = log_energy_floor;
    }
    feats[0] = energy;
  }
}

// Each threadblock computes a different row of the matrix.
// Threads in the same block compute the row collaboratively.
// This kernel must be called out of place (A_in!=A_out).
__global__ void power_spectrum_kernel(int row_length, float *A_in, int32_t ldi,
                                      float *A_out, int32_t ldo,
                                      bool use_power) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  float *Ar = A_in + block_id * ldi;
  float *Aw = A_out + block_id * ldo;

  int half_length = row_length / 2;
  for (int idx = thread_id; idx < half_length; idx += CU1DBLOCK) {
    // ignore special case
    if (idx == 0) continue;

    float2 val = reinterpret_cast<float2 *>(Ar)[idx];
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

// Expects to be called with 32x8 sized thread block.
// LDB: Adding use_log flag
__global__ void mel_banks_compute_kernel(int32_t num_frames, float energy_floor,
                                         int32 *offsets, int32 *sizes,
                                         float **vecs, const float *feats,
                                         int32_t ldf, float *mels, int32_t ldm,
                                         bool use_log) {
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

  if (frame >= num_frames) return;

  int offset = offsets[bin];
  int size = sizes[bin];
  const float *v = vecs[bin];
  const float *w = feats + frame * ldf + offset;

  // perfom local sum
  float sum = 0;
  for (int idx = tid; idx < size; idx += 32) {
    sum += v[idx] * w[idx];
  }

  // Sum in cub
  sum = WarpReduce(temp_storage[wid]).Sum(sum);
  if (tid == 0) {
    if (use_log) {
      // avoid log of zero
      if (sum < energy_floor) sum = energy_floor;
      float val = logf(sum);
      mels[frame * ldm + bin] = val;
    } else {
      mels[frame * ldm + bin] = sum;
    }
  }
}

__global__ void process_window_kernel(
    int frame_length, float dither, float energy_floor, bool remove_dc_offset,
    float preemph_coeff, bool need_raw_log_energy, float *log_energy_pre_window,
    const float *windowing, float *tmp_windows, int32_t ldt, float *windows,
    int32_t ldw) {
  // Specialize WarpReduce for type float
  typedef cub::BlockReduce<float, CU1DBLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_id = threadIdx.x;
  int row = blockIdx.x;
  float *tmp_window = tmp_windows + row * ldt;
  float *window = windows + row * ldw;

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
      log_energy_pre_window[row] = log(energy);
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

__device__ inline int32 FirstSampleOfFrame(int32 frame, int32 frame_shift,
                                           int32 window_size, bool snip_edges) {
  if (snip_edges) {
    return frame * frame_shift;
  } else {
    int32 midpoint_of_frame = frame_shift * frame + frame_shift / 2,
          beginning_of_frame = midpoint_of_frame - window_size / 2;
    return beginning_of_frame;
  }
}

__global__ void extract_window_kernel(
    int32 frame_shift, int32 frame_length, int32 frame_length_padded,
    int32 window_size, bool snip_edges, int32_t sample_offset,
    const BaseFloat * __restrict__ wave, int32 wave_dim,
    BaseFloat *__restrict__ windows, int32_t wlda) {
  int frame = blockIdx.x;
  int tidx = threadIdx.x;

  int32 start_sample =
      FirstSampleOfFrame(frame, frame_shift, window_size, snip_edges);

  // wave_start and wave_end are start and end indexes into 'wave', for the
  // piece of wave that we're trying to extract.
  int32 wave_start = int32(start_sample - sample_offset),
        wave_end = wave_start + frame_length;

  BaseFloat *window = windows + frame * wlda;
  if (wave_start >= 0 && wave_end <= wave_dim) {
    // the normal case-- no edge effects to consider.
    for (int i = tidx; i < frame_length; i += blockDim.x) {
      window[i] = wave[wave_start + i];
    }
  } else {
    // Deal with any end effects by reflection, if needed.  This code will only
    // be reached for about two frames per utterance, so we don't concern
    // ourselves excessively with efficiency.
    for (int s = tidx; s < frame_length; s += blockDim.x) {
      int32 s_in_wave = s + wave_start;
      while (s_in_wave < 0 || s_in_wave >= wave_dim) {
        // reflect around the beginning or end of the wave.
        // e.g. -1 -> 0, -2 -> 1.
        // dim -> dim - 1, dim + 1 -> dim - 2.
        // the code supports repeated reflections, although this
        // would only be needed in pathological cases.
        if (s_in_wave < 0)
          s_in_wave = -s_in_wave - 1;
        else
          s_in_wave = 2 * wave_dim - 1 - s_in_wave;
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
__global__ void dot_log_kernel(int32_t num_frames, int32_t frame_length,
                               float *signal_frame, int32_t lds,
                               float *signal_log_energy) {
  // Specialize WarpReduce for type float
  typedef cub::BlockReduce<float, CU1DBLOCK> BlockReduce;
  // Allocate WarpReduce shared memory for 8 warps
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int32_t frame = blockIdx.x;
  int32_t tid = threadIdx.x;

  float *in = signal_frame + frame * lds;
  float sum = 0;

  // preform local dot product
  for (int32_t i = tid; i < frame_length; i += blockDim.x) {
    float val = in[i];
    sum += val * val;
  }

  // reduce using cub
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    signal_log_energy[frame] = logf(sum);
  }
}

namespace kaldi {

CudaSpectralFeatures::CudaSpectralFeatures(const CudaSpectralFeatureOptions &opts)
    : MfccComputer(opts.mfcc_opts),
      cu_lifter_coeffs_(lifter_coeffs_),
      cu_dct_matrix_(dct_matrix_),
      window_function_(opts.mfcc_opts.frame_opts) {
  const MelBanks *mel_banks = GetMelBanks(1.0);
  const std::vector<std::pair<int32, Vector<BaseFloat>>> &bins =
      mel_banks->GetBins();
  int size = bins.size();
  bin_size_ = size;
  std::vector<int32> offsets(size), sizes(size);
  std::vector<float *> vecs(size);
  cu_vecs_ = new CuVector<float>[size];
  for (int i = 0; i < bins.size(); i++) {
    cu_vecs_[i].Resize(bins[i].second.Dim(), kUndefined);
    cu_vecs_[i].CopyFromVec(bins[i].second);
    vecs[i] = cu_vecs_[i].Data();
    sizes[i] = cu_vecs_[i].Dim();
    offsets[i] = bins[i].first;
  }
  offsets_ = static_cast<int32 *>(
      CuDevice::Instantiate().Malloc(size * sizeof(int32)));
  sizes_ = static_cast<int32 *>(
      CuDevice::Instantiate().Malloc(size * sizeof(int32)));
  vecs_ = static_cast<float **>(
      CuDevice::Instantiate().Malloc(size * sizeof(float *)));

  CU_SAFE_CALL(cudaMemcpyAsync(vecs_, &vecs[0], size * sizeof(float *),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaMemcpyAsync(offsets_, &offsets[0], size * sizeof(int32),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaMemcpyAsync(sizes_, &sizes[0], size * sizeof(int32),
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CU_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));

  frame_length_ = opts.mfcc_opts.frame_opts.WindowSize();
  padded_length_ = opts.mfcc_opts.frame_opts.PaddedWindowSize();
  fft_length_ = padded_length_ / 2;  // + 1;
  fft_size_ = 800;

  // place holders to get strides for cufft.  these will be resized correctly
  // later.  The +2 for cufft/fftw requirements of an extra element at the end.
  // turning off stride because cufft seems buggy with a stride
  cu_windows_.Resize(fft_size_, padded_length_, kUndefined,
                     kStrideEqualNumCols);
  tmp_window_.Resize(fft_size_, padded_length_ + 2, kUndefined,
                     kStrideEqualNumCols);

  stride_ = cu_windows_.Stride();
  tmp_stride_ = tmp_window_.Stride();

  CUFFT_SAFE_CALL(cufftPlanMany(&plan_, 1, &padded_length_, NULL, 1, stride_, NULL, 1,
                                tmp_stride_ / 2, CUFFT_R2C, fft_size_));
  CUFFT_SAFE_CALL(cufftSetStream(plan_, cudaStreamPerThread));
  cumfcc_opts_ = opts;
}

// ExtractWindow extracts a windowed frame of waveform with a power-of-two,
// padded size.  It does mean subtraction, pre-emphasis and dithering as
// requested.
void CudaSpectralFeatures::ExtractWindows(int32_t num_frames, int64 sample_offset,
                              const CuVectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts) {
  KALDI_ASSERT(sample_offset >= 0 && wave.Dim() != 0);
  int32 frame_length = opts.WindowSize(),
        frame_length_padded = opts.PaddedWindowSize();
  int64 num_samples = sample_offset + wave.Dim();

  extract_window_kernel<<<num_frames, CU1DBLOCK>>>(
      opts.WindowShift(), frame_length, frame_length_padded, opts.WindowSize(),
      opts.snip_edges, sample_offset, wave.Data(), wave.Dim(),
      cu_windows_.Data(), cu_windows_.Stride());
  CU_SAFE_CALL(cudaGetLastError());
}

void CudaSpectralFeatures::ProcessWindows(int num_frames,
                              const FrameExtractionOptions &opts,
                              CuVectorBase<BaseFloat> *log_energy_pre_window) {
  if (num_frames == 0) return;

  int fft_num_frames = cu_windows_.NumRows();
  KALDI_ASSERT(fft_num_frames % fft_size_ == 0);

  process_window_kernel<<<num_frames, CU1DBLOCK>>>(
      frame_length_, opts.dither, std::numeric_limits<float>::epsilon(),
      opts.remove_dc_offset, opts.preemph_coeff, NeedRawLogEnergy(),
      log_energy_pre_window->Data(), window_function_.cu_window.Data(),
      tmp_window_.Data(), tmp_window_.Stride(), cu_windows_.Data(),
      cu_windows_.Stride());

  CU_SAFE_CALL(cudaGetLastError());
}

void CudaSpectralFeatures::ComputeFinalFeatures(int num_frames, BaseFloat vtln_wrap,
                                    CuVector<BaseFloat> *cu_signal_log_energy,
                                    CuMatrix<BaseFloat> *cu_features) {
  MfccOptions mfcc_opts = cumfcc_opts_.mfcc_opts;
  Vector<float> tmp;
  assert(mfcc_opts.htk_compat == false);

  if (num_frames == 0) return;

  if (mfcc_opts.use_energy && !mfcc_opts.raw_energy) {
    dot_log_kernel<<<num_frames, CU1DBLOCK>>>(
        num_frames, cu_windows_.NumCols(), cu_windows_.Data(),
        cu_windows_.Stride(), cu_signal_log_energy->Data());
    CU_SAFE_CALL(cudaGetLastError());
  }

  // make sure a reallocation hasn't changed these
  KALDI_ASSERT(cu_windows_.Stride() == stride_);
  KALDI_ASSERT(tmp_window_.Stride() == tmp_stride_);

  // Perform FFTs in batches of fft_size.  This reduces memory requirements
  for (int idx = 0; idx < num_frames; idx += fft_size_) {
    CUFFT_SAFE_CALL(cufftExecR2C(
        plan_, cu_windows_.Data() + cu_windows_.Stride() * idx,
        (cufftComplex *)(tmp_window_.Data() + tmp_window_.Stride() * idx)));
  }

  // Compute Power spectrum
  CuMatrix<BaseFloat> power_spectrum(tmp_window_.NumRows(),
                                     padded_length_ / 2 + 1, kUndefined);

  power_spectrum_kernel<<<num_frames, CU1DBLOCK>>>(
      padded_length_, tmp_window_.Data(), tmp_window_.Stride(),
      power_spectrum.Data(), power_spectrum.Stride(), cumfcc_opts_.use_power);
  CU_SAFE_CALL(cudaGetLastError());

  // mel banks
  int num_bins = bin_size_;
  cu_mel_energies_.Resize(num_frames, num_bins, kUndefined);
  dim3 mel_threads(32, 8);
  dim3 mel_blocks(num_bins, (num_frames + mel_threads.y - 1) / mel_threads.y);
  mel_banks_compute_kernel<<<mel_blocks, mel_threads>>>(
      num_frames, std::numeric_limits<float>::epsilon(), offsets_, sizes_,
      vecs_, power_spectrum.Data(), power_spectrum.Stride(),
      cu_mel_energies_.Data(), cu_mel_energies_.Stride(),
      cumfcc_opts_.use_log_fbank);
  CU_SAFE_CALL(cudaGetLastError());

  // dct transform
  if (cumfcc_opts_.use_dct) {
     cu_features->AddMatMat(1.0, cu_mel_energies_, kNoTrans, cu_dct_matrix_,
                            kTrans, 0.0);

     apply_lifter_and_floor_energy<<<num_frames, CU1DBLOCK>>>(
         cu_features->NumRows(), cu_features->NumCols(),
	 mfcc_opts.cepstral_lifter, mfcc_opts.use_energy,
	 mfcc_opts.energy_floor, cu_signal_log_energy->Data(),
         cu_lifter_coeffs_.Data(), cu_features->Data(), cu_features->Stride());
  } else {
    cudaMemcpyAsync(cu_features->Data(), cu_mel_energies_.Data(),
               sizeof(BaseFloat) * num_frames * cu_features->Stride(),
               cudaMemcpyDeviceToDevice, cudaStreamPerThread);
  }
  CU_SAFE_CALL(cudaGetLastError());
}

void CudaSpectralFeatures::ComputeFeatures(const CuVectorBase<BaseFloat> &cu_wave,
                               BaseFloat sample_freq, BaseFloat vtln_warp,
                               CuMatrix<BaseFloat> *cu_features) {
  nvtxRangePushA("CudaSpectralFeatures::ComputeFeatures");
  const FrameExtractionOptions &frame_opts = GetFrameOptions();
  int num_frames = NumFrames(cu_wave.Dim(), frame_opts, true);
  // compute fft frames by rounding up to a multiple of fft_size_
  int fft_num_frames = num_frames + (fft_size_ - num_frames % fft_size_);
  int feature_dim = Dim();
  bool use_raw_log_energy = NeedRawLogEnergy();

  CuVector<BaseFloat> raw_log_energies;
  raw_log_energies.Resize(num_frames, kUndefined);

  cu_windows_.Resize(fft_num_frames, padded_length_, kUndefined,
                     kStrideEqualNumCols);
  cu_features->Resize(num_frames, feature_dim, kUndefined);
  //+1 matches cufft/fftw requirements
  tmp_window_.Resize(fft_num_frames, padded_length_ + 2, kUndefined,
                     kStrideEqualNumCols);

  if (frame_opts.dither != 0.0f) {
    // Calling cu-rand directly
    // CuRand class works on CuMatrixBase which must
    // assume that the matrix is part of a larger matrix
    // Doing this directly avoids unecessary memory copies
    CURAND_SAFE_CALL(
        curandGenerateNormal(GetCurandHandle(), tmp_window_.Data(),
                             tmp_window_.NumRows() * tmp_window_.Stride(),
                             0.0 /*mean*/, 1.0 /*stddev*/));
  }

  // Extract Windows
  ExtractWindows(num_frames, 0, cu_wave, frame_opts);

  // Process Windows
  ProcessWindows(num_frames, frame_opts, &raw_log_energies);

  // Compute Features
  ComputeFinalFeatures(num_frames, 1.0, &raw_log_energies, cu_features);

  nvtxRangePop();
}
CudaSpectralFeatures::~CudaSpectralFeatures() {
  delete[] cu_vecs_;
  CuDevice::Instantiate().Free(vecs_);
  CuDevice::Instantiate().Free(offsets_);
  CuDevice::Instantiate().Free(sizes_);
  CUFFT_SAFE_CALL(cufftDestroy(plan_));
}
}  // namespace kaldi

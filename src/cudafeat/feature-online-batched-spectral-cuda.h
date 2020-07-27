// cudafeat/feature-batched-spectral-cuda.h
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

#ifndef KALDI_CUDAFEAT_FEATURE_BATCHED_SPECTRAL_CUDA_H_
#define KALDI_CUDAFEAT_FEATURE_BATCHED_SPECTRAL_CUDA_H_

#if HAVE_CUDA == 1
#include <cufft.h>
#endif

#include "cudafeat/feature-spectral-cuda.h"
#include "cudafeat/feature-window-cuda.h"
#include "cudafeat/lane-desc.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/feature-fbank.h"
#include "feat/feature-mfcc.h"

namespace kaldi {
// This class implements MFCC and Fbank computation in CUDA.
// It handles batched input.
// It takes input from device memory and outputs to
// device memory.  It also does no synchronization.
class CudaOnlineBatchedSpectralFeatures : public MfccComputer {
 public:
  void ComputeFeatures(const CuVectorBase<BaseFloat> &cu_wave,
                       BaseFloat sample_freq, BaseFloat vtln_warp,
                       CuMatrix<BaseFloat> *cu_features) {
     // Non-batched processing not allowed from 
     //    CudaOnlineBatchedSpectralFeatures
     KALDI_ASSERT(false);
  }

  void ComputeFeaturesBatched(const LaneDesc *lanes, int32_t n_lanes,
                              const CuMatrixBase<BaseFloat> &cu_wave_in,
                              BaseFloat sample_freq, BaseFloat vtln_warp,
                              CuMatrix<BaseFloat> *cu_feats_out);

  CudaOnlineBatchedSpectralFeatures(const CudaSpectralFeatureOptions &opts,
                                    int32_t max_chunk_frames,
                                    int32_t num_channels, int32_t max_lanes);
  ~CudaOnlineBatchedSpectralFeatures();
  CudaSpectralFeatureOptions cumfcc_opts_;
  int32 Dim()
  // The dimension of the output is different for MFCC and Fbank.
  // This returns the appropriate value depending on the feature
  // extraction algorithm
  {
    if (cumfcc_opts_.feature_type == MFCC) return MfccComputer::Dim();
    // If we're running fbank, we need to set the dimension right
    else
      return cumfcc_opts_.mfcc_opts.mel_opts.num_bins +
             (cumfcc_opts_.mfcc_opts.use_energy ? 1 : 0);
  }

 private:

  void ExtractWindowsBatched(const LaneDesc *lanes, int32_t num_lanes,
                             const CuMatrixBase<BaseFloat> &wave);

  void UpdateStashBatched(const LaneDesc *lanes, int32_t num_lanes,
                          const CuMatrixBase<BaseFloat> &wave);

  void ProcessWindowsBatched(const LaneDesc *lanes, int32_t num_lanes,
                             const FrameExtractionOptions &opts,
                             CuMatrixBase<BaseFloat> *log_energy_pre_window);

  void ComputeFinalFeaturesBatched(const LaneDesc *lanes, int32_t num_lanes,
                                   BaseFloat vtln_wrap,
                                   CuMatrix<BaseFloat> *cu_signal_log_energy,
                                   CuMatrix<BaseFloat> *cu_features);

  CuVector<float> cu_lifter_coeffs_;
  CuMatrix<BaseFloat> cu_windows_;
  CuMatrix<float> tmp_window_, cu_mel_energies_;
  CuMatrix<float> cu_dct_matrix_;
  CuMatrix<BaseFloat> stash_;
  CuMatrix<BaseFloat> power_spectrum_;
  CuMatrix<BaseFloat> raw_log_energies_;

  int frame_length_, padded_length_, fft_length_, fft_batch_size_;
  cufftHandle plan_;
  CudaFeatureWindowFunction window_function_;

  int bin_size_;
  int32 *offsets_, *sizes_;
  CuVector<float> *cu_vecs_;
  float **vecs_;

  // for sanity checking cufft
  int32_t stride_, tmp_stride_;

  int32_t max_chunk_frames_;
  int32_t num_channels_;
  int32_t max_lanes_;
};
}  // namespace kaldi

#endif

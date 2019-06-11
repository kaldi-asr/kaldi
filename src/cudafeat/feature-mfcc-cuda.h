// cudafeat/feature-mfcc-cuda.h
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

#ifndef KALDI_CUDAFEAT_FEATURE_MFCC_CUDA_H_
#define KALDI_CUDAFEAT_FEATURE_MFCC_CUDA_H_

#if HAVE_CUDA == 1
#include <cufft.h>
#endif

#include "cudafeat/feature-window-cuda.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/feature-mfcc.h"

namespace kaldi {
// This class implements MFCC computation in CUDA.
// It takes input from device memory and outputs to
// device memory.  It also does no synchronization.
class CudaMfcc : public MfccComputer {
 public:
  void ComputeFeatures(const CuVectorBase<BaseFloat> &cu_wave,
                       BaseFloat sample_freq, BaseFloat vtln_warp,
                       CuMatrix<BaseFloat> *cu_features);

  CudaMfcc(const MfccOptions &opts);
  ~CudaMfcc();

 private:
  void ExtractWindows(int32 num_frames, int64 sample_offset,
                      const CuVectorBase<BaseFloat> &wave,
                      const FrameExtractionOptions &opts);

  void ProcessWindows(int num_frames, const FrameExtractionOptions &opts,
                      CuVectorBase<BaseFloat> *log_energy_pre_window);

  void ComputeFinalFeatures(int num_frames, BaseFloat vtln_wrap,
                            CuVector<BaseFloat> *cu_signal_log_energy,
                            CuMatrix<BaseFloat> *cu_features);

  CuMatrix<BaseFloat> cu_windows_;
  CuMatrix<float> tmp_window_, cu_mel_energies_;
  CuMatrix<float> cu_dct_matrix_;
  CuVector<float> cu_lifter_coeffs_;

  int frame_length_, padded_length_, fft_length_, fft_size_;
  cufftHandle plan_;
  CudaFeatureWindowFunction window_function_;

  int bin_size_;
  int32 *offsets_, *sizes_;
  CuVector<float> *cu_vecs_;
  float **vecs_;

  // for sanity checking cufft
  int32_t stride_, tmp_stride_;
};
}

#endif

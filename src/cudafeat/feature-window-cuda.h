// cudafeat/feature-window-cuda.h
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

#ifndef KALDI_CUDAFEAT_FEATURE_WINDOW_CUDA_H_
#define KALDI_CUDAFEAT_FEATURE_WINDOW_CUDA_H_

#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/feature-window.h"

namespace kaldi {

// This struct stores a feature window on the device.
// Behind the scense it just computes a feature window on
// the host and then copies it into device memory.
struct CudaFeatureWindowFunction {
  CudaFeatureWindowFunction() {}
  explicit CudaFeatureWindowFunction(const FrameExtractionOptions &opts);
  CuVector<float> cu_window;
};

}  // namespace kaldi

#endif  // KALDI_CUDAFEAT_FEATURE_WINDOW_CUDA_H_

// cudafeat/feature-window-cuda.cu
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

#if HAVE_CUDA == 1
#include <nvToolsExt.h>
#endif
#include "cudafeat/feature-window-cuda.h"
#include "matrix/matrix-functions.h"

namespace kaldi {

CudaFeatureWindowFunction::CudaFeatureWindowFunction(
    const FrameExtractionOptions &opts) {
  nvtxRangePushA("CudaFeatureWindowFunction::CudaFeatureWindowFunction");
  int32 frame_length = opts.WindowSize();

  // Create CPU feature window
  FeatureWindowFunction feature_window(opts);

  // Copy into GPU memory
  cu_window.Resize(frame_length, kUndefined);
  cu_window.CopyFromVec(feature_window.window);
  nvtxRangePop();
}
}  // namespace kaldi

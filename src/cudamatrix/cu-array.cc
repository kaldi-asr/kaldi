// cudamatrix/cu-array.cc

// Copyright 2016  Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#endif

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrixdim.h"
#include "cudamatrix/cu-kernels.h"

#include "cudamatrix/cu-array.h"

namespace kaldi {

template<> 
void CuArray<int32>::Set(const int32 &value) {
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU2DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_int32_set_const(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    for (int32 i = 0; i < dim_; i++) {
      data_[i] = value;
    }
  }
}


template<> 
void CuArray<int32>::Add(const int32 &value) {
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU2DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_int32_add(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    for (int32 i = 0; i < dim_; i++) {
      data_[i] += value;
    }
  }
} 


}  // namespace kaldi

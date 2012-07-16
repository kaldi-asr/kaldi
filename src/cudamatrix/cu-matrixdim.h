// cudamatrix/cu-matrixdim.h

// Copyright 2009-2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CU_MATRIXDIM_H_
#define KALDI_CUDAMATRIX_CU_MATRIXDIM_H_

/*
 * Typedefs needed for ANSI-C interface of CUDA wrappers
 */
#ifdef _MSC_VER
  typedef unsigned __int32 uint32_cuda;
  typedef __int32          int32_cuda;
#else
  #include <stdint.h>
  typedef uint32_t         uint32_cuda;
  typedef int32_t          int32_cuda;
#endif


extern "C" {
  /**
   * Structure containing size of the matrix plus stride.
   * This structure is an argument of most of the CUDA kernels.
   */
  typedef struct MatrixDim_ {
    int32_cuda rows;
    int32_cuda cols;
    int32_cuda stride;
  } MatrixDim;
}

#endif



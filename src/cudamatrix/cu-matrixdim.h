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
  typedef __int32          MatrixIndexT_cuda; // you'd have to change this if you changed MatrixIndexT from int32.
#else
  #include <stdint.h>
  typedef uint32_t         uint32_cuda;
  typedef int32_t          int32_cuda;
  typedef int32_t          MatrixIndexT_cuda; // you'd have to change this if you changed MatrixIndexT from int32.
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

// we define the following constants here because this file is included
// both by the C++ code and also CUDA code.
  
// The size of edge of CUDA square block, e.g. for matrix operations.
// Must be defined the same in cu-kernels-ansi.h
#define CU2DBLOCK 16

// The size of a CUDA 1-d block, e.g. for vector operations..
#define CU1DBLOCK 256
  
}

#endif

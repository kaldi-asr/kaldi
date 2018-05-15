// cudamatrix/cu-common.h

// Copyright 2009-2011  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_COMMON_H_
#define KALDI_CUDAMATRIX_CU_COMMON_H_
#include "cudamatrix/cu-matrixdim.h" // for CU1DBLOCK and CU2DBLOCK

#include <iostream>
#include <sstream>
#include "base/kaldi-error.h"
#include "matrix/matrix-common.h"

#if HAVE_CUDA == 1
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <cuda_runtime_api.h>

#define CU_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "cudaError_t " << ret << " : \"" << cudaGetErrorString((cudaError_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define CUBLAS_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "cublasStatus_t " << ret << " : \"" << cublasGetStatusString((cublasStatus_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define CUSPARSE_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "cusparseStatus_t " << ret << " : \"" << cusparseGetStatusString((cusparseStatus_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define CURAND_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "curandStatus_t " << ret << " : \"" << curandGetStatusString((curandStatus_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
}

#define KALDI_CUDA_ERR(ret, msg) \
{ \
  if (ret != 0) { \
    KALDI_ERR << msg << ", diagnostics: cudaError_t " << ret << " : \"" << cudaGetErrorString((cudaError_t)ret) << "\", in " << __FILE__ << ":" << __LINE__; \
  } \
}


namespace kaldi {

/** Number of blocks in which the task of size 'size' is splitted **/
inline int32 n_blocks(int32 size, int32 block_size) {
  return size / block_size + ((size % block_size == 0)? 0 : 1);
}

cublasOperation_t KaldiTransToCuTrans(MatrixTransposeType kaldi_trans);


/*
  This function gives you suitable dimBlock and dimGrid sizes for a simple
  matrix operation (one that applies to each element of the matrix.  The x
  indexes will be interpreted as column indexes, and the y indexes will be
  interpreted as row indexes; this is based on our interpretation of a matrix as
  being row-major, i.e.  having column-stride = 1, not based on CuBLAS's
  opposite interpretation.  There is a good reason for associating the column
  index with x and not y; this helps memory locality in adjacent kernels.
 */
void GetBlockSizesForSimpleMatrixOperation(int32 num_rows,
                                           int32 num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock);

/** This is analogous to the CUDA function cudaGetErrorString(). **/
const char* cublasGetStatusString(cublasStatus_t status);

/** This is analogous to the CUDA function cudaGetErrorString(). **/
const char* cusparseGetStatusString(cusparseStatus_t status);

/** This is analogous to the CUDA function cudaGetErrorString(). **/
const char* curandGetStatusString(curandStatus_t status);
}

#endif // HAVE_CUDA

namespace kaldi {
// Some forward declarations, needed for friend declarations.
template<typename Real> class CuVectorBase;
template<typename Real> class CuVector;
template<typename Real> class CuSubVector;
template<typename Real> class CuRand;
template<typename Real> class CuMatrixBase;
template<typename Real> class CuMatrix;
template<typename Real> class CuSubMatrix;
template<typename Real> class CuPackedMatrix;
template<typename Real> class CuSpMatrix;
template<typename Real> class CuTpMatrix;
template<typename Real> class CuSparseMatrix;

template<typename Real> class CuBlockMatrix; // this has no non-CU counterpart.


}


#endif

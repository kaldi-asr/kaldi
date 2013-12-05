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
#include <cublas.h>
#include <cuda_runtime_api.h>



#define CU_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "cudaError_t " << ret << " : \"" << cudaGetErrorString((cudaError_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
  cudaThreadSynchronize(); \
} 


namespace kaldi {

/** Number of blocks in which the task of size 'size' is splitted **/
inline int32 n_blocks(int32 size, int32 block_size) { 
  return size / block_size + ((size % block_size == 0)? 0 : 1); 
}

cublasOperation_t KaldiTransToCuTrans(MatrixTransposeType kaldi_trans);
  
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

template<typename Real> class CuBlockMatrix; // this has no non-CU counterpart.


}


#endif

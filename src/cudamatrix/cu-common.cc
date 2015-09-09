// cudamatrix/cu-common.cc

// Copyright      2013  Karel Vesely

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

#ifndef KALDI_CUDAMATRIX_COMMON_H_
#define KALDI_CUDAMATRIX_COMMON_H_

// This file contains some #includes, forward declarations
// and typedefs that are needed by all the main header
// files in this directory.
#include "base/kaldi-common.h"
#include "matrix/kaldi-blas.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-common.h"

namespace kaldi {

#if HAVE_CUDA == 1
cublasOperation_t KaldiTransToCuTrans(MatrixTransposeType kaldi_trans) {
  cublasOperation_t cublas_trans;

  if (kaldi_trans == kNoTrans)
    cublas_trans = CUBLAS_OP_N;
  else if (kaldi_trans == kTrans)
    cublas_trans = CUBLAS_OP_T;
  else
    cublas_trans = CUBLAS_OP_C;
  return cublas_trans;
}
#endif

} // namespace


#endif  // KALDI_CUDAMATRIX_COMMON_H_

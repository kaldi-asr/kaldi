// cudamatrix/cu-common.cc

// Copyright      2013  Karel Vesely
//                2015  Johns Hopkins University (author: Daniel Povey)

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
#include <mutex>
#include "base/kaldi-common.h"
#include "matrix/kaldi-blas.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrixdim.h"


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

void GetBlockSizesForSimpleMatrixOperation(int32 num_rows,
                                           int32 num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock) {
  KALDI_ASSERT(num_rows > 0 && num_cols > 0);
  int32 col_blocksize = 64, row_blocksize = 4;
  while (col_blocksize > 1 &&
         (num_cols + (num_cols / 2) <= col_blocksize ||
          num_rows > 65535 * row_blocksize)) {
    col_blocksize /= 2;
    row_blocksize *= 2;
  }

  dimBlock->x = col_blocksize;
  dimBlock->y = row_blocksize;
  dimBlock->z = 1;
  dimGrid->x = n_blocks(num_cols, col_blocksize);
  dimGrid->y = n_blocks(num_rows, row_blocksize);
  KALDI_ASSERT(dimGrid->y <= 65535 &&
               "Matrix has too many rows to process");
  dimGrid->z = 1;
}

const char* cublasGetStatusString(cublasStatus_t status) {
  switch(status) {
    case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN_ERROR";
}

const char* cusparseGetStatusString(cusparseStatus_t status) {
  // detail info come from http://docs.nvidia.com/cuda/cusparse/index.html#cusparsestatust
  switch(status) {
    case CUSPARSE_STATUS_SUCCESS:                   return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:             return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_ZERO_PIVOT:                return "CUSPARSE_STATUS_ZERO_PIVOT";
  }
  return "CUSPARSE_STATUS_UNKNOWN_ERROR";
}
#endif

} // namespace


#endif  // KALDI_CUDAMATRIX_COMMON_H_

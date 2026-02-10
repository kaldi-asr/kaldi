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

#if HAVE_CUDA

#include "cudamatrix/cu-common.h"

#include <cuda.h>

#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrixdim.h"

namespace kaldi {

#ifdef USE_NVTX
NvtxTracer::NvtxTracer(const char* name) {
  const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
  const int num_colors = sizeof(colors)/sizeof(uint32_t);
  int color_id = ((int)name[0])%num_colors;
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = colors[color_id];
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;
	nvtxRangePushEx(&eventAttrib);
  // nvtxRangePushA(name);
}
NvtxTracer::~NvtxTracer() {
  nvtxRangePop();
}
#endif

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

const char* cublasGetStatusStringK(cublasStatus_t status) {
  // Defined in CUDA include file: cublas.h or cublas_api.h
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
  // Defined in CUDA include file: cusparse.h
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
    #if CUDA_VERSION >= 11000
    case CUSPARSE_STATUS_NOT_SUPPORTED:             return "CUSPARSE_STATUS_NOT_SUPPORTED";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:    return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    #endif
  }
  return "CUSPARSE_STATUS_UNKNOWN_ERROR";
}

const char* curandGetStatusString(curandStatus_t status) {
  // detail info come from http://docs.nvidia.com/cuda/curand/group__HOST.html
  // Defined in CUDA include file: curand.h
  switch(status) {
    case CURAND_STATUS_SUCCESS:                     return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:            return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:             return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:           return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:                  return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:                return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:         return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:   return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:              return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:         return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:       return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:               return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:              return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "CURAND_STATUS_UNKNOWN_ERROR";
}

}  // namespace kaldi

#endif  // HAVE_CUDA

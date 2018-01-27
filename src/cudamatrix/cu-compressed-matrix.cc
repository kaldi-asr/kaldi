// cudamatrix/cu-compressed-matrix.cc

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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


#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-compressed-matrix.h"

namespace kaldi {


CuCompressedMatrixBase *NewCuCompressedMatrix(CuCompressedMatrixType t,
                                              BaseFloat range) {
  if (t == kCompressedMatrixUint8) {
    KALDI_ASSERT(range >= 0);
    return new CuCompressedMatrix<uint8>(range);
  } else if (t == kCompressedMatrixInt16) {
    KALDI_ASSERT(range > 0);
    return new CuCompressedMatrix<int16>(range);
  }
}

template <typename I> CuCompressedMatrix::CuCompressedMatrix(BaseFloat range):
    data_(NULL), scale_(range / std::numeric_limits<I>::max()),
    num_rows_(0), num_cols_(0), stride_(0) {
#if HAVE_CUDA == 1
  KALDI_ASSERT(CuDevice::Instantiate().Enabled());
#endif
  KALDI_ERR << "You instantiated CuCompressedMatrix while GPU use "
      "was not compiled in.";
}


template <typename I> void CuCompressedMatrix::Destroy() {
#if HAVE_CUDA == 1
  if (data_ != NULL) {
    CuTimer tim;
    CuDevice::Instantiate().Free(data_);
    data_ = NULL;
    num_rows_ = 0;
    num_cols_ = 0;
    stride_ = 0;
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  }
#endif
}

template <typename I> void CuCompressedMatrix::CopyFromMat(
    CuMatrixBase<BaseFloat> &mat) {
#if HAVE_CUDA == 1
  KALDI_ASSERT(CuDevice::Instantiate().Enabled());
  Destroy();
  if (mat.NumRows() == 0)
    return;
  num_rows_ = mat.NumRows();
  num_cols_ = mat.NumCold();
  stride_ = num_cols_;

  CuTimer tim;
  data_ = CuDevice::Instantiate().Malloc(sizeof(I) * num_rows_ * num_cols_);

  dim3 dimGrid, dimBlock;
  GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                        &dimGrid, &dimBlock);

  if (scale_ == 0.0) { // scale == 0 calls a different kernel from the others.
    cuda_mat_compress_sign(dimGrid, dimBlock, mat.Data(), mat.Dim(),
                           data_, stride_);
  } else {
    cuda_mat_compress(dimGrid, dimBlock, mat.Data(), mat.Dim(),
                      data_, stride_, 1.0 / scale_);

  }

    CU_SAFE_CALL(cudaGetLastError());



  CuDevice::Instantiate().AccuProfile(CuCompressedMatrix::CopyFromMat(malloc),
                                      tim);


#endif
}


} // namespace kaldi

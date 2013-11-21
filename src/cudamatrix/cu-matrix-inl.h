// cudamatrix/cu-matrix-inl.h

// Copyright 2009-2012  Karel Vesely

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

// Do not include this file directly.  It is included by cu-matrix.h.

#ifndef KALDI_CUDAMATRIX_CU_MATRIX_INL_H_
#define KALDI_CUDAMATRIX_CU_MATRIX_INL_H_

namespace kaldi {

template<typename Real>
inline CuSubMatrix<Real>::CuSubMatrix(const CuMatrixBase<Real> &mat,
                                      const MatrixIndexT row_offset,
                                      const MatrixIndexT num_rows,
                                      const MatrixIndexT col_offset,
                                      const MatrixIndexT num_cols):
    CuMatrixBase<Real>(mat.data_ + (row_offset * mat.stride_) + col_offset,
                       num_rows,
                       num_cols,
                       mat.stride_) {
  KALDI_ASSERT(row_offset >= 0 && col_offset >= 0 &&
               row_offset + num_rows <= mat.num_rows_ &&
               col_offset + num_cols <= mat.num_cols_);
}
  
} // namespace kaldi

#endif

  

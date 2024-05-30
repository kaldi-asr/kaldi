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
                                      const MatrixIndexT num_cols) {
  if (num_rows == 0 || num_cols == 0) {
    KALDI_ASSERT(num_rows == 0 && num_cols == 0);
    // Everything will have been set to zero in CuMastrixBase's default
    // initializer, so nothing to do.
  } else {
    KALDI_ASSERT(row_offset >= 0 && col_offset >= 0 &&
                 num_rows >= 0 && num_cols >= 0 &&
                 row_offset + num_rows <= mat.num_rows_ &&
                 col_offset + num_cols <= mat.num_cols_);
    this->data_ = mat.data_ + static_cast<size_t>(col_offset) +
        static_cast<size_t>(row_offset) * static_cast<size_t>(mat.stride_);
    this->num_cols_ = num_cols;
    this->num_rows_ = num_rows;
    this->stride_ = mat.stride_;
  }
}

template<typename Real>
inline CuSubMatrix<Real>::CuSubMatrix(const Real *data,
                                      const MatrixIndexT num_rows,
                                      const MatrixIndexT num_cols,
                                      const MatrixIndexT stride):
    CuMatrixBase<Real>(const_cast<Real*>(data), num_rows, num_cols, stride) {
  // in general if you use SubMatrix or CuSubMatrix, const-correctness is not
  // preserved (preserving it would require us duplicating the class and it
  // would have been a hassle).

  // Note: we used to check that stride >= num_cols.  We no longer check for
  // this as there are some situations where having stride < num_cols is useful,
  // but beware because most if not all CUBLAS calls will crash when given
  // such an input, even in a situation where it makes sense.
  KALDI_ASSERT((num_rows != 0) == (num_cols != 0) && stride >= 0 &&
               num_rows >= 0 && num_cols >= 0 && stride >= 0);
}


} // namespace kaldi

#endif

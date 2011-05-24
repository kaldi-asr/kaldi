// matrix/tp-matrix.cc
// Copyright 2009-2011  Ondrej Glembek  Lukas Burget  Microsoft Corporation
//                  Arnab Ghoshal (Saarland University)  Yanmin Qian

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "matrix/tp-matrix.h"
#include "matrix/sp-matrix.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

#ifndef HAVE_ATLAS
template<>
void TpMatrix<float>::Invert() {
  // these are CLAPACK types
  KaldiBlasInt result;
  KaldiBlasInt rows = static_cast<int>(num_rows_);

  // clapack call
  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  stptri_(const_cast<char *>("U"), const_cast<char *>("N"), &rows, data_, &result);

  if (result < 0) {
    KALDI_ERR << "Call to CLAPACK stptri_ function failed";
  } else if (result > 0) {
    KALDI_ERR << "Matrix is singular";
  }
}

template<>
void TpMatrix<double>::Invert() {
  // these are CLAPACK types
  KaldiBlasInt result;
  KaldiBlasInt rows = static_cast<int>(num_rows_);

  // clapack call
  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  dtptri_(const_cast<char *>("U"), const_cast<char *>("N"), &rows, data_, &result);

  if (result < 0) {
    KALDI_ERR << "Call to CLAPACK dtptri_ function failed";
  } else if (result > 0) {
    KALDI_ERR << "Matrix is singular";
  }
}
#else
template<class Real>
void TpMatrix<Real>::Invert() {
  Matrix<Real> tmp(*this);
  tmp.Invert();
  (*this).CopyFromMat(tmp);
}

#endif

template<typename Real>
Real TpMatrix<Real>::Determinant() {
  double   det = 1.0;
  for (MatrixIndexT i = 0; i<this->NumRows(); i++) {
    det *= (*this)(i, i);
  }
  return static_cast<Real>(det);
}


template<typename Real>
void TpMatrix<Real>::Swap(TpMatrix<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->num_rows_, other->num_rows_);
}


template<typename Real>
void TpMatrix<Real>::Cholesky(const SpMatrix<Real>& rOrig) {
  KALDI_ASSERT(rOrig.NumRows() == this->NumRows());
  MatrixIndexT n = this->NumRows();
  this->SetZero();

  // Main loop.
  for (MatrixIndexT j = 0; j < n; j++) {
    Real d(0.0);
    for (MatrixIndexT k = 0; k < j; k++) {
      Real s(0.0);
      for (MatrixIndexT i = 0; i < k; i++) {
        s += (*this)(k, i) * (*this)(j, i);
      }
      (*this)(j, k) = s = (rOrig(j, k) - s)/(*this)(k, k);
      d = d + s*s;
    }
    d = rOrig(j, j) - d;
    if (d >= 0.0)
      (*this)(j, j) = sqrt(d);
    else
      KALDI_ERR << "Cholesky decomposition failed. Maybe matrix "
          "is not positive definite.";
  }
}

template<typename Real>
void TpMatrix<Real>::CopyFromMat(MatrixBase<Real> &M,
                                 MatrixTransposeType Trans) {
  if (Trans == kNoTrans) {
    KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
    MatrixIndexT D = this->NumRows();
    for (MatrixIndexT i = 0; i < D; i++) {
      for (MatrixIndexT j = 0; j <= i; j++) {
        (*this)(i, j) = M(i, j);
      }
    }
  } else {
    KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
    MatrixIndexT D = this->NumRows();
    for (MatrixIndexT i = 0; i < D; i++) {
      for (MatrixIndexT j = 0; j <= i; j++) {
        (*this)(i, j) = M(j, i);
      }
    }
  }
}


template class TpMatrix<float>;
template class TpMatrix<double>;

}  // namespace kaldi



// cudamatrix/cu-sp-matrix.h

// Copyright 2009-2013  Karel Vesely
//                2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_CUDAMATRIX_CU_SP_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_SP_MATRIX_H_

#include <sstream>

#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/sp-matrix.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {

/// TraceSpSp returns tr(A B)
template<typename Real, typename OtherReal>
Real TraceSpSp(const CuSpMatrix<Real> &A, const CuSpMatrix<OtherReal> &B);

template<typename Real>
class CuSpMatrix : public CuPackedMatrix<Real> {
  friend class CuMatrixBase<Real>;
  friend class CuVectorBase<Real>;
  friend class CuTpMatrix<Real>;
  friend class CuSubMatrix<Real>;
  friend class CuRand<Real>;

  template<class R, class S>
  friend R TraceSpSp(const CuSpMatrix<R> &A, const CuSpMatrix<S> &B);
 public:
  
  CuSpMatrix(): CuPackedMatrix<Real>() {}
  
  explicit CuSpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
    : CuPackedMatrix<Real>(r, resize_type) {}

  explicit CuSpMatrix(const SpMatrix<Real> &orig)
    : CuPackedMatrix<Real>(orig) {}

  // This constructor lacks the "explicit" keyword so that
  // we can include it in std::vector.
  CuSpMatrix(const CuSpMatrix<Real> &orig)
    : CuPackedMatrix<Real>(orig) {}

  explicit CuSpMatrix(const CuMatrixBase<Real> &orig,
                      SpCopyType copy_type = kTakeLower)
      : CuPackedMatrix<Real>(orig.NumRows(), kUndefined) {
    CopyFromMat(orig, copy_type);
  }

  CuSpMatrix<Real> &operator = (const CuSpMatrix<Real> &in);
  
  ~CuSpMatrix() {}  

  inline void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero) {
    CuPackedMatrix<Real>::Resize(nRows, resize_type);
  }

  Real FrobeniusNorm() const { return sqrt(TraceSpSp(*this, *this)); }

  bool IsUnit(Real tol = 0.001) const;

  bool ApproxEqual(const CuSpMatrix<Real> &other, Real tol = 0.001) const;
  
  void CopyFromSp(const CuSpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }
  void CopyFromSp(const SpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }

  void CopyFromMat(const CuMatrixBase<Real> &orig,
                   SpCopyType copy_type = kTakeLower);
  
  void CopyToSp(SpMatrix<Real> *dst) const { //added const by hxu
    CuPackedMatrix<Real>::CopyToPacked(dst);
  }

  inline CuValue<Real> operator() (MatrixIndexT r, MatrixIndexT c) {
    if (static_cast<UnsignedMatrixIndexT>(c) >
        static_cast<UnsignedMatrixIndexT>(r))
      std::swap(c, r);
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));
    return CuValue<Real>(this->data_ + (r * (r+1)) / 2 + c);
  }
  
  inline Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    if (static_cast<UnsignedMatrixIndexT>(c) >
        static_cast<UnsignedMatrixIndexT>(r))
      std::swap(c, r);
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));
    return CuValue<Real>(this->data_ + (r * (r+1)) / 2 + c); // will be
    // casted to Real.
  }

  /// Note: the CuMatrix version of the Invert() function will only work for
  /// positive definite matrices; it is based on Cholesky.
  void Invert();

  void AddVec2(const Real alpha, const CuVectorBase<Real> &v);

  void AddMat2(const Real alpha, const CuMatrixBase<Real> &M,
               MatrixTransposeType transM, const Real beta);
  
  void AddSp(const Real alpha, const CuSpMatrix<Real> &Ma) {
    this->AddPacked(alpha, Ma);
  }

 protected:
  inline const SpMatrix<Real> &Mat() const {
    return *(reinterpret_cast<const SpMatrix<Real>* >(this));
  }
  inline SpMatrix<Real> &Mat() {
    return *(reinterpret_cast<SpMatrix<Real>* >(this));
  }
};

template<typename Real>
inline bool ApproxEqual(const CuSpMatrix<Real> &A,
                 const CuSpMatrix<Real> &B, Real tol = 0.001) {
  return A.ApproxEqual(B, tol);
}

template<typename Real>
inline void AssertEqual(const CuSpMatrix<Real> &A,
                        const CuSpMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(ApproxEqual(A, B, tol));
}


template<typename Real>
SpMatrix<Real>::SpMatrix(const CuSpMatrix<Real> &cu) {
   Resize(cu.NumRows());
   cu.CopyToSp(this);
}



} // namespace

#endif

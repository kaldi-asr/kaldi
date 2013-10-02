// cudamatrix/cu-tp-matrix.h
// Copyright 2013  Ehsan Variani

// See ../../COPYING for clarification regarding multiple authors
//
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
//
#ifndef KALDI_CUDAMATRIX_CU_TP_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_TP_MATRIX_H_

#include <sstream>

#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/tp-matrix.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {

template<typename Real> class CuTpMatrix;

template<typename Real>
class CuTpMatrix : public CuPackedMatrix<Real> {
  friend class CuMatrixBase<float>;
  friend class CuMatrixBase<double>;
  friend class CuVectorBase<Real>;
  friend class CuSubMatrix<Real>;
  friend class CuRand<Real>;
  friend class CuTpMatrix<float>;
  friend class CuTpMatrix<double>;
 public:
  CuTpMatrix() : CuPackedMatrix<Real>() {}
  explicit CuTpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
      : CuPackedMatrix<Real>(r, resize_type) {}
  explicit CuTpMatrix<Real>(const TpMatrix<Real> &orig)
      : CuPackedMatrix<Real>(orig) {}
  explicit CuTpMatrix<Real>(const CuTpMatrix<Real> &orig)
      : CuPackedMatrix<Real>(orig) {}
  explicit CuTpMatrix<Real>(const CuMatrixBase<Real> &orig,
                            MatrixTransposeType trans = kNoTrans);

  
  ~CuTpMatrix() {}

  void CopyFromMat(const CuMatrixBase<Real> &M,
                   MatrixTransposeType Trans = kNoTrans);

  void CopyFromTp(const CuTpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }
  void CopyFromTp(const TpMatrix<Real> &other) {
    CuPackedMatrix<Real>::CopyFromPacked(other);
  }  
  void Cholesky(const CuSpMatrix<Real>& Orig);
  void Invert();

 protected:
  inline const TpMatrix<Real> &Mat() const {
    return *(reinterpret_cast<const TpMatrix<Real>* >(this));
  }
  inline TpMatrix<Real> &Mat() {
    return *(reinterpret_cast<TpMatrix<Real>* >(this));
  }
  
};

} // namespace

#endif

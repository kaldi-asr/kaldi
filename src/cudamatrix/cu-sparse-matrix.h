// cudamatrix/cu-sparse-matrix.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen

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



#ifndef KALDI_CUDAMATRIX_CU_SPARSE_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_SPARSE_MATRIX_H_

#include <sstream>
#include <vector>

#include "cudamatrix/cu-matrixdim.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-value.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/sparse-matrix.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {

template <typename Real>
Real TraceMatSmat(const CuMatrixBase<Real> &A,
                  const CuSparseMatrix<Real> &B,
                  MatrixTransposeType trans = kNoTrans);

template <class Real>
class CuSparseMatrix {
 public:
  friend class CuMatrixBase<float>;
  friend class CuMatrixBase<double>;
  friend class CuMatrixBase<Real>;
  friend class CuVectorBase<float>;
  friend class CuVectorBase<double>;
  friend class CuVectorBase<Real>;

  friend Real TraceMatSmat<Real>(const CuMatrixBase<Real> &A,
                                 const CuSparseMatrix<Real> &B,
                                 MatrixTransposeType trans);

  MatrixIndexT NumRows() const { return num_rows_; }

  MatrixIndexT NumCols() const { return num_cols_; }

  MatrixIndexT NumElements() const;

  template <typename OtherReal>
  void CopyToMat(CuMatrixBase<OtherReal> *dest,
                 MatrixTransposeType trans = kNoTrans) const;

  Real Sum() const;

  Real FrobeniusNorm() const;

  // returns pointer to element data, or NULL if empty (use with NumElements()).
  // This should only be called when CUDA is enabled.
  MatrixElement<Real> *Data();

  // returns pointer to element data, or NULL if empty (use with NumElements()),
  // const version. This should only be called when CUDA is enabled.
  const MatrixElement<Real> *Data() const;

  /// Copy from CPU-based matrix.
  CuSparseMatrix<Real> &operator = (const SparseMatrix<Real> &smat);

  /// Copy from possibly-GPU-based matrix.
  CuSparseMatrix<Real> &operator = (const CuSparseMatrix<Real> &smat);

  /// Copy from CPU-based matrix. We will add the transpose option later when it
  /// is necessary.
  template <typename OtherReal>
  void CopyFromSmat(const SparseMatrix<OtherReal> &smat);

  /// Copy to CPU-based matrix. We will add the transpose option later when it
  /// is necessary.
  template <typename OtherReal>
  void CopyToSmat(SparseMatrix<OtherReal> *smat) const;

  /// Copy elements to CuVector. It is the caller's responsibility to resize
  /// <*vec>.
  void CopyElementsToVec(CuVectorBase<Real> *vec) const;

  /// Swap with CPU-based matrix.
  void Swap(SparseMatrix<Real> *smat);

  /// Swap with possibly-CPU-based matrix.
  void Swap(CuSparseMatrix<Real> *smat);

  /// Sets up to a pseudo-randomly initialized matrix, with each element zero
  /// with probability zero_prob and else normally distributed- mostly for
  /// purposes of testing.
  void SetRandn(BaseFloat zero_prob);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  // Constructor from CPU-based sparse matrix.
  explicit CuSparseMatrix(const SparseMatrix<Real> &smat) {
    this->CopyFromSmat(smat);
  }

  ~CuSparseMatrix() { }

  // Use the CuMatrix::CopyFromSmat() function to copy from this to
  // CuMatrix.
  // Also see CuMatrix::AddSmat().

 protected:
  // The following two functions should only be called if we did not compile
  // with CUDA or could not get a CUDA card; in that case the contents are
  // interpreted the same as a regular sparse matrix.
  inline const SparseMatrix<Real> &Mat() const {
    return *(reinterpret_cast<const SparseMatrix<Real>* >(this));
  }
  inline SparseMatrix<Real> &Mat() {
    return *(reinterpret_cast<SparseMatrix<Real>* >(this));
  }

 private:
  // This member is only used if we did not compile for the GPU, or if the GPU
  // is not enabled.  It needs to be first because we reinterpret_cast this
  std::vector<SparseVector<Real> > cpu_rows_;

  MatrixIndexT num_rows_;
  MatrixIndexT num_cols_;

  // This is where the data lives if we are using a GPU.  Notice that the format
  // is a little different from on CPU, as there is only one list, of matrix
  // elements, instead of a list for each row.  This is better suited to
  // CUDA code.
  CuArray<MatrixElement<Real> > elements_;
};


}  // namespace

#endif  // KALDI_CUDAMATRIX_CU_SPARSE_MATRIX_H_

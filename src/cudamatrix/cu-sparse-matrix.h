// cudamatrix/cu-sparse-matrix.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen
//                2017  Shiyin Kang

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

template<class Real>
class CuSparseMatrix {
public:
  friend class CuMatrixBase<float> ;
  friend class CuMatrixBase<double> ;
  friend class CuMatrixBase<Real> ;
  friend class CuVectorBase<float> ;
  friend class CuVectorBase<double> ;
  friend class CuVectorBase<Real> ;

  friend Real TraceMatSmat<Real>(const CuMatrixBase<Real> &A,
                                 const CuSparseMatrix<Real> &B,
                                 MatrixTransposeType trans);

  MatrixIndexT NumRows() const;

  MatrixIndexT NumCols() const;

  MatrixIndexT NumElements() const;

  template<typename OtherReal>
  void CopyToMat(CuMatrixBase<OtherReal> *dest, MatrixTransposeType trans =
                     kNoTrans) const;

  Real Sum() const;

  Real FrobeniusNorm() const;

  /// Copy from CPU-based matrix.
  CuSparseMatrix<Real> &operator =(const SparseMatrix<Real> &smat);

  /// Copy from possibly-GPU-based matrix.
  CuSparseMatrix<Real> &operator =(const CuSparseMatrix<Real> &smat);

  /// Copy from CPU-based matrix.  We will add the transpose option later when it
  /// is necessary.  Resizes *this as needed.
  template<typename OtherReal>
  void CopyFromSmat(const SparseMatrix<OtherReal> &smat);

  /// Copy from GPU-based matrix, supporting transposition.  Resizes *this
  /// as needed.
  void CopyFromSmat(const CuSparseMatrix<Real> &smat,
                    MatrixTransposeType trans = kNoTrans);

  /// Select a subset of the rows of a CuSparseMatrix.
  /// Sets *this to only the rows of 'smat_other' that are listed
  /// in 'row_indexes'.
  /// 'row_indexes' must satisfy 0 <= row_indexes[i] < smat_other.NumRows().
  void SelectRows(const CuArray<int32> &row_indexes,
                  const CuSparseMatrix<Real> &smat_other);

  /// Copy to CPU-based matrix. We will add the transpose option later when it
  /// is necessary.
  template<typename OtherReal>
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

  /// Default constructor
  CuSparseMatrix() :
      num_rows_(0), num_cols_(0), nnz_(0), csr_row_ptr_col_idx_(NULL), csr_val_(
          NULL) {
  }

  /// Constructor from CPU-based sparse matrix.
  explicit CuSparseMatrix(const SparseMatrix<Real> &smat) :
      num_rows_(0), num_cols_(0), nnz_(0), csr_row_ptr_col_idx_(NULL), csr_val_(
      NULL) {
    this->CopyFromSmat(smat);
  }

  /// Constructor from GPU-based sparse matrix (supports transposition).
  CuSparseMatrix(const CuSparseMatrix<Real> &smat, MatrixTransposeType trans =
                     kNoTrans) :
      num_rows_(0), num_cols_(0), nnz_(0), csr_row_ptr_col_idx_(NULL), csr_val_(
      NULL) {
    this->CopyFromSmat(smat, trans);
  }

  /// Constructor from an array of indexes.
  /// If trans == kNoTrans, construct a sparse matrix
  /// with num-rows == indexes.Dim() and num-cols = 'dim'.
  /// 'indexes' is expected to contain elements in the
  /// range [0, dim - 1].  Each row 'i' of *this after
  /// calling the constructor will contain  a single
  /// element at column-index indexes[i] with value 1.0.
  ///
  /// If trans == kTrans, the result will be the transpose
  /// of the sparse matrix described above.
  CuSparseMatrix(const CuArray<int32> &indexes, int32 dim,
                 MatrixTransposeType trans = kNoTrans);

  /// Constructor from an array of indexes and an array of
  /// weights; requires indexes.Dim() == weights.Dim().
  /// If trans == kNoTrans, construct a sparse matrix
  /// with num-rows == indexes.Dim() and num-cols = 'dim'.
  /// 'indexes' is expected to contain elements in the
  /// range [0, dim - 1].  Each row 'i' of *this after
  /// calling the constructor will contain a single
  /// element at column-index indexes[i] with value weights[i].
  /// If trans == kTrans, the result will be the transpose
  /// of the sparse matrix described above.
  CuSparseMatrix(const CuArray<int32> &indexes,
                 const CuVectorBase<Real> &weights, int32 dim,
                 MatrixTransposeType trans = kNoTrans);

  ~CuSparseMatrix() {
    Destroy();
  }

protected:
  // The following two functions should only be called if we did not compile
  // with CUDA or could not get a CUDA card; in that case the contents are
  // interpreted the same as a regular sparse matrix.
  inline const SparseMatrix<Real> &Smat() const {
    return *(reinterpret_cast<const SparseMatrix<Real>*>(this));
  }
  inline SparseMatrix<Real> &Smat() {
    return *(reinterpret_cast<SparseMatrix<Real>*>(this));
  }

  /// Users of this class won't normally have to use Resize.
  /// 'nnz' should be determined beforehand when calling this API.
  void Resize(const MatrixIndexT num_rows, const MatrixIndexT num_cols,
              const MatrixIndexT nnz, MatrixResizeType resize_type = kSetZero);

  /// Returns pointer to the data array of length nnz_ that holds all nonzero
  /// values in zero-based CSR format
  const Real* CsrVal() const {
    return csr_val_;
  }
  Real* CsrVal() {
    return csr_val_;
  }

  /// Returns pointer to the integer array of length NumRows()+1 that holds
  /// indices of the first nonzero element in the i-th row, while the last entry
  /// contains nnz_, as zero-based CSR format is used.
  const int* CsrRowPtr() const {
    return csr_row_ptr_col_idx_;
  }
  int* CsrRowPtr() {
    return csr_row_ptr_col_idx_;
  }

  /// Returns pointer to the integer array of length nnz_ that contains
  /// the column indices of the corresponding elements in array CsrVal()
  const int* CsrColIdx() const {
    return csr_row_ptr_col_idx_ + num_rows_ + 1;
  }
  int* CsrColIdx() {
    return csr_row_ptr_col_idx_ + num_rows_ + 1;
  }

private:
  void Destroy();

private:
  // This member is only used if we did not compile for the GPU, or if the GPU
  // is not enabled.  It needs to be first because we reinterpret_cast this
  std::vector<SparseVector<Real> > cpu_rows_;

  // This is where the data lives if we are using a GPU.
  // The sparse matrix is stored in CSR format, as documented here.
  // http://docs.nvidia.com/cuda/cusparse/index.html#compressed-sparse-row-format-csr
  // The 3 arrays are stored in 2 allocated blocks of memory.
  // Row ptr and col idx are both int arrays, thus stored in one block pointed
  // 'by csr_row_ptr_col_idx_'
  // Val are Real array, pointed by `csr_val_`

  // matrix size num_rows_ x num_cols_
  MatrixIndexT num_rows_;
  MatrixIndexT num_cols_;

  // number of non-zeros
  MatrixIndexT nnz_;

  // csr row ptrs and col indices in a single int array
  // of the length (num_rows_ + 1 + nnz_)
  int* csr_row_ptr_col_idx_;

  // csr value array of the length nnz_
  Real* csr_val_;
};


}  // namespace

#endif  // KALDI_CUDAMATRIX_CU_SPARSE_MATRIX_H_

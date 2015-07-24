// matrix/sparse-matrix.h

// Copyright  2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_MATRIX_SPARSE_MATRIX_H_
#define KALDI_MATRIX_SPARSE_MATRIX_H_ 1

#include "matrix/matrix-common.h"

namespace kaldi {


/// \addtogroup matrix_group
/// @{

/// Base class which provides matrix operations not involving resizing
/// or allocation.   Classes Matrix and SubMatrix inherit from it and take care
/// of allocation and resizing.

template <typename Real>
class SparseVector {

  int32 Dim() const;

  void CopyToVector(VectorBase<Real> *other);

  SparseVector<Real> &operator = (const SparseVector<Real> &other); 
      
  SparseVector(const SparseVector<Real> &other) { *this = other; }

  void Swap(SparseVector<Real> *other);

  // initializer.
  SparseVector(const std::vector<std::pair<int32, BaseFloat> > &pairs);

 private:
  // pairs of (row-index, value).  Stored in sorted order with no duplicates.
  // For now we use std::vector, but we could change this.
  std::vector<std::pair<int32, BaseFloat> > pairs_;
};

template <typename Real>
class SparseMatrix {
  int32 NumRows() const;

  int32 NumCols() const;

  void CopyToMatrix(MatrixBase<Real> *other);

  SparseMatrix<Real> &operator = (const SparseMatrix<Real> &other); 
      
  SparseMatrix(const SparseMatrix<Real> &other) { *this = other; }

  void Swap(SparseMatrix<Real> *other);

  // initializer from the type that elsewhere in Kaldi is referred to as type Posterior.
  // indexed first by row-index; the pairs are (column-index, value).
  SparseMatrix(const std::vector<std::vector<std::pair<int32, BaseFloat> > > &pairs);

  /// Sets up to a pseudo-randomly initialized matrix, with each element zero
  /// with probability zero_prob and else normally distributed- mostly for
  /// purposes of testing.
  void SetRandn(BaseFloat zero_prob);
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &os, bool binary);

  // Use the Matrix::CopyFromSmat() function to copy from this to Matrix.  Also
  // see Matrix::AddSmat().  There is not very extensive functionality for SparseMat just yet
  // (e.g. no matrix multiply); we will add things as needed and as it seems necessary.
 private:
  // vector of SparseVector (use an stl vector for now; this could change).
  std::vector<SparseVector<Real> > rows_;
};


template<typename Real>
Real TraceMatSmat(const CuMatrixBase<Real> &A,
                  const CuSparseMatrix<Real> &B,
                  MatrixTransposeType trans = kNoTrans);


/// @} end of \addtogroup matrix_group


}  // namespace kaldi

#endif  // KALDI_MATRIX_KALDI_MATRIX_H_

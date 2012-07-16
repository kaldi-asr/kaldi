// cudamatrix/cu-matrix.h

// Copyright 2009-2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CUMATRIX_H_
#define KALDI_CUDAMATRIX_CUMATRIX_H_

#include <sstream>

#include "cudamatrix/cu-matrixdim.h"

#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"


namespace kaldi {

template<typename Real> class CuVector;

/**
 * Matrix for CUDA computing,
 *
 * It has "polymorphic" behavior. The computation is forwarded to CPU,
 * both when CUDA is not compiled in as well as when missing guitable GPU :
 * (CuDevice::Enabled() != true)
 */
template<typename Real>
class CuMatrix {
 typedef CuMatrix<Real> ThisType;
  
 public:

  /// Default Constructor
  CuMatrix<Real>()
   : num_rows_(0), num_cols_(0), stride_(0), data_(NULL) { 
  }
  /// Constructor with memory initialisation
  CuMatrix<Real>(MatrixIndexT rows, MatrixIndexT cols)
   : num_rows_(0), num_cols_(0), stride_(0), data_(NULL) { 
    Resize(rows, cols); 
  }

  /// Destructor
  ~CuMatrix() {
    Destroy(); 
  }

  /// Dimensions
  MatrixIndexT NumRows() const { 
    return num_rows_; 
  }

  MatrixIndexT NumCols() const { 
    return num_cols_; 
  }

  MatrixIndexT Stride() const { 
    return stride_; 
  }

  ::MatrixDim Dim() const { 
    ::MatrixDim d = { num_rows_, num_cols_, stride_ }; 
    return d; 
  }

  /// Get raw pointer
  const Real* Data() const;
  Real* Data();
  
  /// Get raw row pointer
  const Real* RowData(MatrixIndexT r) const;
  Real* RowData(MatrixIndexT r);

  /// Get size of matrix in bytes
  MatrixIndexT SizeInBytes() const { 
    return num_rows_*stride_*sizeof(Real); 
  }
  
  /// Get size of matrix row in bytes
  MatrixIndexT RowSizeInBytes() const {
    return num_cols_*sizeof(Real); 
  }
  
  /// Get size of matrix stride in bytes
  MatrixIndexT StrideSizeInBytes() const {
    return stride_*sizeof(Real); 
  }

  /// Allocate the memory
  ThisType& Resize(MatrixIndexT rows, MatrixIndexT cols);

  /// Deallocate the memory
  void Destroy();

  /// Copy functions (reallocates when needed)
  ThisType&        CopyFromMat(const CuMatrix<Real> &src);
  ThisType&        CopyFromMat(const Matrix<Real> &src);
  void             CopyToMat(Matrix<Real> *dst) const;

  /// Copy row interval from matrix
  /// @param r      [in] number of rows to copy.
  /// @param src    [in] source matrix.
  /// @param src_ro [in] source matrix row offset.
  /// @param dst_ro [in] destination matrix row offset.
  void             CopyRowsFromMat(int32 r, const CuMatrix<Real> &src, int32 src_ro, int32 dst_ro);

  /// I/O functions
  void             Read(std::istream &is, bool binary);
  void             Write(std::ostream &os, bool binary) const;

  /// Math operations, some calling kernels
  void SetZero();
  void Set(Real value);
  void ApplyLog();
  /// Multiply two matrices elementhwise: C = A .* C
  void MulElements(const CuMatrix<Real>& A);
  /// scale i'th column by scale[i]
  void MulColsVec(const CuVector<Real> &scale); 
  /// scale i'th row by scale[i]
  void MulRowsVec(const CuVector<Real> &scale); 
  /// divide i'th row by scale[i]
  void DivRowsVec(const CuVector<Real> &div);
  /// B = aplha * A + beta * B
  void AddMat(Real alpha, const CuMatrix<Real>& A, Real beta=1.0);
  /// B = aplha * row + beta * B
  void AddVecToCols(Real alpha, const CuVector<Real> &col, Real beta=1.0);
  /// B = aplha * row + beta * B
  void AddVecToRows(Real alpha, const CuVector<Real> &row, Real beta=1.0);
  /// C = alpha * A(^T)*B(^T) + beta * C
  void AddMatMat(Real alpha, const CuMatrix<Real>& A, MatrixTransposeType transA,
                 const CuMatrix<Real>& B, MatrixTransposeType transB, Real beta);

  /// Accessor to the non-CUDA matrix
  const MatrixBase<Real>& Mat() const {
    return mat_;
  }
  MatrixBase<Real>& Mat() {
    return mat_;
  }

 private:
  MatrixIndexT num_rows_;
  MatrixIndexT num_cols_;
  MatrixIndexT stride_;

  Real *data_;       ///< GPU data pointer
  
  Matrix<Real> mat_; ///< non-GPU matrix as back-up


}; // class CuMatrix



/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrix<Real> &mat);


  
} // namespace


#include "cu-matrix-inl.h"

#endif

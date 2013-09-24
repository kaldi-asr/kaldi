// cudamatrix/cu-matrix.h

// Copyright 2009-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_MATRIX_H_

#include <sstream>

#include "cudamatrix/cu-matrixdim.h"
#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {


/**
 * Matrix for CUDA computing.
 * Does the computation on the CUDA card when CUDA is compiled in and
 * we have a suitable GPU (CuDevice::Instantiate().Enabled() == true);
 * otherwise, does it on the CPU.
 */

template<typename Real>
class CuMatrixBase {
 public:
  friend class CuVectorBase<Real>;
  friend class CuSubMatrix<Real>;
  friend class CuRand<Real>;
  friend void cu::RegularizeL1<Real>(CuMatrixBase<Real> *weight,
                                     CuMatrixBase<Real> *grad, Real l1, Real lr);
  friend void cu::Splice<Real>(const CuMatrix<Real> &src,
                               const CuStlVector<int32> &frame_offsets,
                               CuMatrix<Real> *tgt);
  friend void cu::Copy<Real>(const CuMatrix<Real> &src,
                             const CuStlVector<int32> &copy_from_indices,
                             CuMatrix<Real> *tgt);
  friend void cu::Randomize<Real>(const CuMatrixBase<Real> &src,
                                  const CuStlVector<int32> &copy_from_idx,
                                  CuMatrixBase<Real> *tgt);
  
  /// Dimensions
  MatrixIndexT NumRows() const { return num_rows_;  }
  MatrixIndexT NumCols() const { return num_cols_;  }
  MatrixIndexT Stride() const { return stride_; }

  // MatrixDim is a struct containing "rows", "cols" and "stride",
  // that is an argument of most CUDA kernels.
  ::MatrixDim Dim() const { 
    ::MatrixDim d = { num_rows_, num_cols_, stride_ }; 
    return d; 
  }

  /// Get size of matrix in bytes
  MatrixIndexT SizeInBytes() const { return num_rows_*stride_*sizeof(Real); }

  /// Get size of matrix row in bytes
  MatrixIndexT RowSizeInBytes() const { return num_cols_*sizeof(Real); }
  
  /// Get size of matrix stride in bytes
  MatrixIndexT StrideSizeInBytes() const { return stride_*sizeof(Real); }

  
  /// Copy functions (reallocates when needed, but note from Dan: eventually
  /// I'll change it to just die if the sizes don't match, like the Matrix class.)
  void CopyFromMat(const CuMatrixBase<Real> &src);
  void CopyFromMat(const MatrixBase<Real> &src);
  void CopyToMat(MatrixBase<Real> *dst) const;

  /// Set each element to the sigmoid of the corresponding element of "src":
  /// element by element, *this = 1 / (1 + exp(-src)).
  void Sigmoid(const CuMatrixBase<Real> &src);

  /// Compute the hyperbolic tangent (tanh) function; element by element,
  /// *this = tanh(src).
  void Tanh(const CuMatrixBase<Real> &src);

  /// Differentiate backward through the sigmoid function.  Here, "value" is the
  /// sigmoid output.  Does, element-by-element, *this = diff * value * (1 - value).
  void DiffSigmoid(const CuMatrixBase<Real> &value,
                   const CuMatrixBase<Real> &diff);

  /// Differentiate backward through the tanh function.  Here, "value" is the
  /// tanh output.  Does, element-by-element, *this = diff * (1 - value^2).
  void DiffTanh(const CuMatrixBase<Real> &value,
                const CuMatrixBase<Real> &diff);

  /// Differentiate the block [softmax+cross-entropy] :
  /// dE/da = posterior_mat - target_mat, 
  /// 'E' is error function, 'a' is activation on softmax input
  ///
  /// Interface:
  /// tgt ... index vector, encodes the matrix of targets
  /// net_out_or_diff ... before invocation net output, after diff dE/da
  /// log_post_tgt ... per-frame statistics for cross-entropy computations :
  ///                  log(sum_row(posterior_mat .* target_mat))
  void DiffXent(const CuStlVector<int32> &tgt,
                CuVector<Real> *log_post_tgt);  
  
  /// Softmax nonlinearity
  /// Y = Softmax(X) : Yij = e^Xij / sum_k(e^Xik)
  /// for each row, the max value is first subtracted for good numerical stability
  void Softmax(const CuMatrixBase<Real> &src);

  /// Find the id of the maximal element for each row
  void FindRowMaxId(CuStlVector<int32> *id) const;
  
  /*
  // Copy row interval from matrix
  // @param r      [in] number of rows to copy.
  // @param src    [in] source matrix.
  // @param src_ro [in] source matrix row offset.
  // @param dst_ro [in] destination matrix row offset.
  // void CopyRowsFromMat(int32 r, const CuMatrixBase<Real> &src, int32 src_ro, int32 dst_ro);
  */
  
  /// Math operations, some calling kernels
  void SetZero();
  void Set(Real value);
  void Add(Real value);
  void Scale(Real value);
  void ApplyLog();
  /// Multiply two matrices elementhwise: C = A .* C
  void MulElements(const CuMatrixBase<Real>& A);
  /// scale i'th column by scale[i]
  void MulColsVec(const CuVectorBase<Real> &scale); 
  /// scale i'th row by scale[i]
  void MulRowsVec(const CuVectorBase<Real> &scale); 
  /// divide i'th row by scale[i]
  void DivRowsVec(const CuVectorBase<Real> &div);
  /// B = aplha * A + beta * B
  void AddMat(Real alpha, const CuMatrixBase<Real>& A, Real beta=1.0);
  /// B = aplha * row + beta * B
  void AddVecToCols(Real alpha, const CuVectorBase<Real> &col, Real beta=1.0);
  /// B = aplha * row + beta * B
  void AddVecToRows(Real alpha, const CuVectorBase<Real> &row, Real beta=1.0);
  /// C = alpha * A(^T)*B(^T) + beta * C
  void AddMatMat(Real alpha, const CuMatrixBase<Real>& A, MatrixTransposeType transA,
                 const CuMatrixBase<Real>& B, MatrixTransposeType transB, Real beta);


  inline CuSubMatrix<Real> Range(const MatrixIndexT row_offset,
                                 const MatrixIndexT num_rows,
                                 const MatrixIndexT col_offset,
                                 const MatrixIndexT num_cols) const {
    return CuSubMatrix<Real>(*this, row_offset, num_rows,
                             col_offset, num_cols);
  }
  inline CuSubMatrix<Real> RowRange(const MatrixIndexT row_offset,
                                    const MatrixIndexT num_rows) const {
    return CuSubMatrix<Real>(*this, row_offset, num_rows,
                             0, num_cols_);
  }
  inline CuSubMatrix<Real> ColRange(const MatrixIndexT col_offset,
                                    const MatrixIndexT num_cols) const {
    return CuSubMatrix<Real>(*this, 0, num_rows_, col_offset, num_cols); 
  }

  
 protected:
  /// Get raw row pointer
  inline const Real* RowData(MatrixIndexT r) const { return data_ + r * stride_; }
  inline Real* RowData(MatrixIndexT r) { return data_ + r * stride_; }

  
  // The constructors are protected to prevent the user creating an instance of
  // this class.
  
  /// Default constructor
  CuMatrixBase<Real>(): data_(NULL), num_cols_(0), num_rows_(0), stride_(0) { }
  
  /// This constructor takes the #rows, #cols and stride; it's called from
  /// the constructor of CuSubMatrix.
  CuMatrixBase<Real>(Real *data,
                     MatrixIndexT num_rows,
                     MatrixIndexT num_cols,
                     MatrixIndexT stride):
  data_(data), num_cols_(num_cols), num_rows_(num_rows), stride_(stride) { }

  // The following two functions should only be called if we did not compile with CUDA
  // or could not get a CUDA card; in that case the contents are interpreted the
  // same as a regular matrix.
  inline const MatrixBase<Real> &Mat() const {
    return *(reinterpret_cast<const MatrixBase<Real>* >(this));
  }
  inline MatrixBase<Real> &Mat() {
    return *(reinterpret_cast<MatrixBase<Real>* >(this));
  }
  
  Real *data_;       ///< GPU data pointer (or regular matrix data pointer,
                     ///< if either CUDA was not compiled in or we could not
                     ///< acquire the device).
  // Note: it might seem a bit backwards that we have the number of columns
  // first here; it's necessary because we need the data to be laid out the same
  // as for MatrixBase so the Mat() function call will work.  We don't want to
  // change the layout of MatrixBase at this point, or there will be crashes if
  // people don't thoroughly recompile.
  MatrixIndexT num_cols_;
  MatrixIndexT num_rows_;
  MatrixIndexT stride_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(CuMatrixBase);
}; // class CuMatrixBase

/// This class represents a matrix that's stored on the GPU if we have one,
/// and in memory if not.
template<typename Real>
class CuMatrix: public CuMatrixBase<Real> {
 public:

  CuMatrix() { }
    
  /// Constructor with memory initialisation
  CuMatrix(MatrixIndexT rows, MatrixIndexT cols,
           MatrixResizeType resize_type = kSetZero) {
    Resize(rows, cols, resize_type); 
  }

  // Note: we had to remove the "explicit" keyword due
  // to problems with STL vectors of CuMatrixBase.
  CuMatrix(const CuMatrix<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
  }

  explicit CuMatrix(const MatrixBase<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
  }
  
  CuMatrix<Real> &operator = (const CuMatrixBase<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
    return *this;
  }

  CuMatrix<Real> &operator = (const CuMatrix<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
    return *this;
  }  
  
  CuMatrix<Real> &operator = (const MatrixBase<Real> &other) {
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
    this->CopyFromMat(other);
    return *this;
  }   

  /// Allocate the memory
  void Resize(MatrixIndexT rows, MatrixIndexT cols,
              MatrixResizeType resize_type = kSetZero);
  
  
  void Swap(Matrix<Real> *mat);
  
  /// I/O functions
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

  /// Destructor
  ~CuMatrix() { Destroy(); }
 private:
  void Destroy();
};


/// This class is used for a piece of a CuMatrix.
template<typename Real>
class CuSubMatrix: public CuMatrixBase<Real> {
 public:
  inline CuSubMatrix(const CuMatrixBase<Real> &mat,
                     const MatrixIndexT row_offset,
                     const MatrixIndexT num_rows,
                     const MatrixIndexT col_offset,
                     const MatrixIndexT num_cols);
                    
  /// This type of constructor is needed for Range() to work [in CuMatrix base
  /// class]. Cannot make it explicit or that breaks.
  inline CuSubMatrix<Real> (const CuSubMatrix &other):
  CuMatrixBase<Real> (other.data_, other.num_cols_, other.num_rows_,
                      other.stride_) {}
 private:
  /// Disallow assignment.
  CuSubMatrix<Real> &operator = (const CuSubMatrix<Real> &other);
};

template<class Real>
bool SameDim(const CuMatrixBase<Real> &M, const CuMatrixBase<Real> &N) {
  return (M.NumRows() == N.NumRows() && M.NumCols() == N.NumCols());
}

template<class Real>
bool SameDimAndStride(const CuMatrixBase<Real> &M, const CuMatrixBase<Real> &N) {
  return (M.NumRows() == N.NumRows() && M.NumCols() == N.NumCols()
          && M.Stride() == N.Stride());
}


/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrixBase<Real> &mat);


  
} // namespace


#include "cu-matrix-inl.h"

#endif

// cudamatrix/cu-matrix.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//                2013  Johns Hopkins University (author: Guoguo Chen)

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
#include "cudamatrix/cu-value.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {

template<typename Real>
Real TraceMatMat(const CuMatrixBase<Real> &A, const CuMatrixBase<Real> &B,
                 MatrixTransposeType trans = kNoTrans);
/**
 * Matrix for CUDA computing.
 * Does the computation on the CUDA card when CUDA is compiled in and
 * we have a suitable GPU (CuDevice::Instantiate().Enabled() == true);
 * otherwise, does it on the CPU.
 */

/*
template<typename Real>
struct MatrixElement {
  int row;
  int column;
  Real weight;
};
// */

template<typename Real>
class CuMatrixBase {
 public:
  friend class CuMatrixBase<float>;
  friend class CuMatrixBase<double>;
  friend class CuVectorBase<float>;
  friend class CuVectorBase<double>;
  friend class VectorBase<Real>;
  friend class CuSpMatrix<Real>;
  friend class CuTpMatrix<float>;
  friend class CuTpMatrix<double>;
  friend class CuVectorBase<Real>;
  friend class CuSubMatrix<Real>;
  friend class CuRand<Real>;
  friend class CuSubVector<Real>;
  friend class CuBlockMatrix<Real>;
  friend void cu::RegularizeL1<Real>(CuMatrixBase<Real> *weight,
                                     CuMatrixBase<Real> *grad, Real l1, Real lr);
  friend void cu::Splice<Real>(const CuMatrix<Real> &src,
                               const CuArray<int32> &frame_offsets,
                               CuMatrix<Real> *tgt);
  friend void cu::Copy<Real>(const CuMatrix<Real> &src,
                             const CuArray<int32> &copy_from_indices,
                             CuMatrix<Real> *tgt);
  friend void cu::Randomize<Real>(const CuMatrixBase<Real> &src,
                                  const CuArray<int32> &copy_from_idx,
                                  CuMatrixBase<Real> *tgt);

  /// Copies column r from column indices[r] of src.
  /// As a special case, if indexes[i] == -1, sets column i to zero
  /// indices.size() must equal this->NumCols(),
  /// all elements of "reorder" must be in [-1, src.NumCols()-1],
  /// and src.NumRows() must equal this.NumRows()
  void CopyCols(const CuMatrixBase<Real> &src,
                const std::vector<MatrixIndexT> &indices);

  /// Version of CopyCols that takes CuArray argument.
  void CopyCols(const CuMatrixBase<Real> &src,
                const CuArray<MatrixIndexT> &indices);

  
  /// Copies row r from row indices[r] of src.
  /// As a special case, if indexes[i] <== -1, sets row i to zero  
  /// "reorder".size() must equal this->NumRows(), 
  /// all elements of "reorder" must be in [0, src.NumRows()-1],
  /// and src.NumCols() must equal this.NumCols()
  void CopyRows(const CuMatrixBase<Real> &src,
                const std::vector<MatrixIndexT> &indices);


  /// For each row r of this and for each column c, sets (*this)(r, c) to the
  /// sum \sum_j src(r, j), where j ranges from indices[c].first through
  /// indices[c].second - 1.
  void SumColumnRanges(const CuMatrixBase<Real> &src,
                       const CuArray<Int32Pair> &indices);


  friend Real TraceMatMat<Real>(const CuMatrixBase<Real> &A,
                                const CuMatrixBase<Real> &B,
                                MatrixTransposeType trans);

  void AddToDiag(Real value);
  
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

  Real FrobeniusNorm() const { return sqrt(TraceMatMat(*this, *this, kTrans)); }

  bool IsUnit(Real tol = 0.001) const;  

  /// True if ((*this)-other).FrobeniusNorm() <= tol * this->FrobeniusNorm()
  bool ApproxEqual(const CuMatrixBase<Real> &other, float tol = 0.01) const;
  
  /// Get size of matrix in bytes
  MatrixIndexT SizeInBytes() const { return num_rows_*stride_*sizeof(Real); }
  
  // Copy functions.  These do not resize.
  template<typename OtherReal>
  void CopyFromMat(const MatrixBase<OtherReal> &src,
                   MatrixTransposeType trans = kNoTrans);

  void CopyFromMat(const MatrixBase<Real> &src,
                   MatrixTransposeType trans = kNoTrans);
  
  void CopyFromSp(const CuSpMatrix<Real> &M);
  
  template<typename OtherReal>
  void CopyFromTp(const CuTpMatrix<OtherReal> &M,
                  MatrixTransposeType trans = kNoTrans);
  
  template<typename OtherReal>
  void CopyFromMat(const CuMatrixBase<OtherReal> &M,
                   MatrixTransposeType trans = kNoTrans); 

  template<typename OtherReal>
  void CopyToMat(MatrixBase<OtherReal> *dst,
                 MatrixTransposeType trans = kNoTrans) const;
  
  void CopyRowsFromVec(const CuVectorBase<Real> &v);

  void CopyRowsFromVec(const VectorBase<Real> &v);
  
  /// Copy vector into specific column of matrix.
  void CopyColFromVec(const CuVectorBase<Real> &v, const MatrixIndexT col);

  /// Set each element to the sigmoid of the corresponding element of "src":
  /// element by element, x = 1 / (1 + exp(-x))
  void Sigmoid(const CuMatrixBase<Real> &src);

  /// Apply the function y = log(1 + exp(x)), to each element.
  /// Note: the derivative of this function is the sigmoid function.
  /// This is like a soft ReLU.
  void SoftHinge(const CuMatrixBase<Real> &src);

  /// Apply the function y(i) = (sum_{j = i*G}^{(i+1)*G-1} x_j ^ (power)) ^ (1 / p)
  /// where G = x.NumCols() / y.NumCols() must be an integer.
  void GroupPnorm(const CuMatrixBase<Real> &src, Real pow);

  /// Calculate derivatives for the GroupPnorm function above...
  /// if "input" is the input to the GroupPnorm function above (i.e. the "src" variable),
  /// and "output" is the result of the computation (i.e. the "this" of that function
  /// call), and *this has the same dimension as "input", then it sets each element
  /// of *this to the derivative d(output-elem)/d(input-elem) for each element of "input", where
  /// "output-elem" is whichever element of output depends on that input element.
  void GroupPnormDeriv(const CuMatrixBase<Real> &input,
                       const CuMatrixBase<Real> &output, Real power);
  
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
  void DiffXent(const CuArray<int32> &tgt,
                CuVector<Real> *log_post_tgt);  

  /// This method may be only called for symmetric matrices (it accesses the
  /// upper as well as lower triangle).  The result is put in the lower
  /// triangle, and the upper triangle zeroed.
  void Cholesky();
  
  void SymInvertPosDef(); ///< Inversion for positive definite symmetric matrices.
                          ///< Requires that the input is symmetric (we do not check this).
                          ///< The output is symmetric.
  
  void ApplyPow(Real power);
  void ApplyHeaviside(); ///< For each element, sets x = (x > 0 ? 1.0 : 0.0)
  void ApplyFloor(Real floor_val);
  void ApplyCeiling(Real ceiling_val);
  void ApplyExp();
  /// Softmax nonlinearity
  /// Y = Softmax(X) : Yij = e^Xij / sum_k(e^Xik), done to each row
  /// for each row, the max value is first subtracted for good numerical stability
  void ApplySoftMaxPerRow(const CuMatrixBase<Real> &src);

  /// Find the id of the maximal element for each row
  void FindRowMaxId(CuArray<int32> *id) const;
  
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
  void SetZeroUpperDiag();
  void Scale(Real value);
  void ApplyLog();
  
  /// Multiply two matrices elementwise: C = A .* C
  void MulElements(const CuMatrixBase<Real> &A);
  /// Do, elementwise, *this = max(*this, A).
  void Max(const CuMatrixBase<Real> &A);
  /// scale i'th column by scale[i]
  void MulColsVec(const CuVectorBase<Real> &scale); 
  /// scale i'th row by scale[i]
  void MulRowsVec(const CuVectorBase<Real> &scale);
  /// divide each row into src.NumCols() groups, and then scale i'th row's jth group of elements by src[i, j].   
  void MulRowsGroupMat(const CuMatrixBase<Real> &src);
  /// divide i'th row by scale[i]
  void DivRowsVec(const CuVectorBase<Real> &div);
  /// invert the matrix by elements.
  void InvertElements();
  /// B = alpha * A
  void AddMat(Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA = kNoTrans);
  /// B = alpha * row + beta * B
  void AddVecToCols(Real alpha, const CuVectorBase<Real> &col, Real beta = 1.0);
  /// B = alpha * row + beta * B
  void AddVecToRows(Real alpha, const CuVectorBase<Real> &row, Real beta = 1.0);
  /// C = alpha * A(^T)*B(^T) + beta * C
  void AddMatMat(Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
                 const CuMatrixBase<Real> &B, MatrixTransposeType transB, Real beta);
  /// *this = a * b / c (by element; when c = 0, *this = a)
  void AddMatMatDivMat(const CuMatrixBase<Real> &A, const CuMatrixBase<Real> &B, const CuMatrixBase<Real> &C);
  /// *this = beta * *this + alpha * M M^T, for symmetric matrices.  It only
  /// updates the lower triangle of *this.  It will leave the matrix asymmetric;
  /// if you need it symmetric as a regular matrix, do CopyLowerToUpper().
  void SymAddMat2(const Real alpha, const CuMatrixBase<Real> &M,
                  MatrixTransposeType transA, Real beta);

  
  /// This function is like AddMatMat but for where the second argument is of
  /// type CuBlockMatrix (a block-diagonal matrix of blocks).
  void AddMatBlock(Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
                   const CuBlockMatrix<Real> &B, MatrixTransposeType transB, Real beta);
  
  /// *this = beta * *this + alpha * diag(v) * M [or M^T].
  /// The same as adding M but scaling each row M_i by v(i).
  void AddDiagVecMat(const Real alpha, CuVectorBase<Real> &v,
                     const CuMatrixBase<Real> &M, MatrixTransposeType transM, 
                     Real beta = 1.0);  
  
  /// this <-- beta*this + alpha*A*B
  void AddMatSp(const Real alpha,
                const CuMatrixBase<Real> &A, MatrixTransposeType transA,
                const CuSpMatrix<Real> &B,
                const Real beta) {
    CuMatrix<Real> M(B);
    return AddMatMat(alpha, A, transA, M, kNoTrans, beta);
  }

  /// this <-- beta*this + alpha*SpA*B
  void AddSpMat(const Real alpha,
                const CuSpMatrix<Real> &A,
                const CuMatrixBase<Real> &B, MatrixTransposeType transB,
                const Real beta) {
    CuMatrix<Real> M(A);
    return AddMatMat(alpha, M, kNoTrans, B, transB, beta);
  }

  /// this <-- beta*this + alpha*A*B.
  void AddTpMat(const Real alpha,
                const CuTpMatrix<Real> &A, MatrixTransposeType transA,
                const CuMatrixBase<Real> &B, MatrixTransposeType transB,
                const Real beta) {
    CuMatrix<Real> M(A);
    return AddMatMat(alpha, M, transA, B, transB, beta);
  }

  /// this <-- beta*this + alpha*A*B.
  void AddMatTp(const Real alpha,
                const CuMatrixBase<Real> &A, MatrixTransposeType transA,
                const CuTpMatrix<Real> &B, MatrixTransposeType transB,
                const Real beta) {
    CuMatrix<Real> M(B);
    return AddMatMat(alpha, A, transA, M, transB, beta);
  }

  void CopyFromBlock(const CuBlockMatrix<Real> &B,
                     MatrixTransposeType trans = kNoTrans);
  void CopyLowerToUpper();
  void CopyUpperToLower();
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

  inline const CuSubVector<Real> Row(MatrixIndexT i) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return CuSubVector<Real>(data_ + (i * stride_), NumCols());
  }

  inline CuSubVector<Real> Row(MatrixIndexT i) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return CuSubVector<Real>(data_ + (i * stride_), NumCols());
  }

  inline CuValue<Real> operator() (MatrixIndexT r, MatrixIndexT c) {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                          static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                          static_cast<UnsignedMatrixIndexT>(c) <
                          static_cast<UnsignedMatrixIndexT>(num_cols_));
    return CuValue<Real>(data_ + r * stride_ + c);
  }
  
  inline Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                          static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                          static_cast<UnsignedMatrixIndexT>(c) <
                          static_cast<UnsignedMatrixIndexT>(num_cols_));
    return CuValue<Real>(data_ + r * stride_ + c);  // will be casted to Real.
  }

  Real Sum() const;

  /// Return the trace. If check_square = true, will crash if matrix is not square.
  Real Trace(bool check_square = true) const;

  void SetRandn();

  void SetRandUniform();

  void Write(std::ostream &os, bool binary) const;

  // This function, adds a list of MatrixElements (scaled by alpha) to corresponding locations to
  // (*this).
  void AddElements(Real alpha, const std::vector<MatrixElement<Real> >& input);

  // This function resizes the output to indices.size(), and for each element of
  // "indices" it interprets it as a (row, column) index into *this, and puts
  // (*this)(row, column) into the corresponding element of "output".
  void Lookup(const std::vector<Int32Pair> &indices,
              std::vector<Real> *output) const;

  // Creates binary mask with per-element equality predicates of *this, mat.
  // Output stored to 'mask', values : 1.0 = equal, 0.0 = not-equal.
  void EqualElementMask(const CuMatrixBase<Real> &mat, CuMatrix<Real> *mask) const;

 protected:
  // The following two functions should only be called if we did not compile with CUDA
  // or could not get a CUDA card; in that case the contents are interpreted the
  // same as a regular matrix.
  inline const MatrixBase<Real> &Mat() const {
    return *(reinterpret_cast<const MatrixBase<Real>* >(this));
  }
  inline MatrixBase<Real> &Mat() {
    return *(reinterpret_cast<MatrixBase<Real>* >(this));
  }
  
  /// Get raw row pointer
  inline const Real* RowData(MatrixIndexT r) const { return data_ + r * stride_; }
  inline Real* RowData(MatrixIndexT r) { return data_ + r * stride_; }
  inline const Real *Data() const { return data_; }
  inline Real *Data() { return data_; }


  
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
  CuMatrix(const CuMatrix<Real> &other,
           MatrixTransposeType trans = kNoTrans);

  explicit CuMatrix(const CuBlockMatrix<Real> &other,
                    MatrixTransposeType trans = kNoTrans);
  
  explicit CuMatrix(const CuMatrixBase<Real> &other,
                    MatrixTransposeType trans = kNoTrans);

  template<typename OtherReal>
  explicit CuMatrix(const MatrixBase<OtherReal> &other,
                    MatrixTransposeType trans = kNoTrans);

  /// Copy constructor taking SpMatrix... 
  explicit CuMatrix(const CuSpMatrix<Real> &M) : CuMatrixBase<Real>() {
    Resize(M.NumRows(), M.NumRows(), kUndefined);
    this->CopyFromSp(M);
  }

  /// Copy constructor taking TpMatrix...
  template <typename OtherReal>
  explicit CuMatrix(const CuTpMatrix<OtherReal> & M,
                    MatrixTransposeType trans = kNoTrans) : CuMatrixBase<Real>() {
    Resize(M.NumCols(), M.NumRows(), kUndefined);
    this->CopyFromTp(M, trans);
  }

  /// Copy constructor: as above, but from another type.
  template<typename OtherReal>
  explicit CuMatrix(const CuMatrixBase<OtherReal> &M,
                    MatrixTransposeType trans = kNoTrans);
  
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

  void Transpose();

  /// Allocate the memory
  void Resize(MatrixIndexT rows, MatrixIndexT cols,
              MatrixResizeType resize_type = kSetZero);
    
  void Swap(Matrix<Real> *mat);
  void Swap(CuMatrix<Real> *mat);

  template<typename OtherReal>
  void Swap(CuMatrix<OtherReal> *mat);
  
  /// I/O functions
  void Read(std::istream &is, bool binary);

  /// Destructor
  ~CuMatrix() { Destroy(); }

  inline const Matrix<Real> &Mat() const {
    return *(reinterpret_cast<const Matrix<Real>* >(this));
  }
  inline Matrix<Real> &Mat() {
    return *(reinterpret_cast<Matrix<Real>* >(this));
  }

  /// This function does: for each element { row, column, weight } indexed i in
  /// the vector "elements", let x(i) = A(row(i), column(i)); then it does
  /// (*this)(row(i), column(i)) += weight(i) / x(i), and
  /// *tot_objf = \sum_i weight(i) * log(x(i)), and
  /// *tot_weight = \sum_i weight(i)
  /// Preconditions: A must be strictly positive, and no (row, column) pair
  /// may be repeated within "elements"
  void CompObjfAndDeriv(const std::vector<MatrixElement<Real> > &elements,
                        const CuMatrix<Real> &A,
                        Real *tot_objf,
                        Real* tot_weight);

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


template<typename Real>
bool ApproxEqual(const CuMatrixBase<Real> &A,
                 const CuMatrixBase<Real> &B, Real tol = 0.01) {
  return A.ApproxEqual(B, tol);
}

template<typename Real>
inline void AssertEqual(CuMatrixBase<Real> &A, CuMatrixBase<Real> &B,
                        float tol = 0.01) {
  KALDI_ASSERT(A.ApproxEqual(B, tol));
}

template<typename Real>
bool SameDim(const CuMatrixBase<Real> &M, const CuMatrixBase<Real> &N) {
  return (M.NumRows() == N.NumRows() && M.NumCols() == N.NumCols());
}

template<typename Real>
bool SameDimAndStride(const CuMatrixBase<Real> &M, const CuMatrixBase<Real> &N) {
  return (M.NumRows() == N.NumRows() && M.NumCols() == N.NumCols()
          && M.Stride() == N.Stride());
}

/// I/O
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrixBase<Real> &mat);


template<typename Real>
template<typename OtherReal>
Matrix<Real>::Matrix(const CuMatrixBase<OtherReal> &M,
                     MatrixTransposeType trans) {
  if (trans == kNoTrans) Init(M.NumRows(), M.NumCols());
  else Init(M.NumCols(), M.NumRows());
  M.CopyToMat(this, trans);
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromMat(const CuMatrixBase<OtherReal> &cu,
                                   MatrixTransposeType trans) {
  cu.CopyToMat(this, trans);
}


}  // namespace


#include "cudamatrix/cu-matrix-inl.h"

#endif

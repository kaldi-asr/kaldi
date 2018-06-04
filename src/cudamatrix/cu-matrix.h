// cudamatrix/cu-matrix.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//           2013-2015  Guoguo Chen
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



#ifndef KALDI_CUDAMATRIX_CU_MATRIX_H_
#define KALDI_CUDAMATRIX_CU_MATRIX_H_

#include <sstream>
#include <vector>

#include "cudamatrix/cu-matrixdim.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-value.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "cudamatrix/cu-sparse-matrix.h"

namespace kaldi {

template<typename Real>
Real TraceMatMat(const CuMatrixBase<Real> &A, const CuMatrixBase<Real> &B,
                 MatrixTransposeType trans = kNoTrans);

/// Does multiple matrix multiplications, executing them in parallel using
/// cuBLAS's gemmBatched if we are using a GPU. Vectors A, B and C must have
/// the same length; for each i, this function executes the matrix operation
/// C[i] = alpha *  A[i](^T)*B[i](^T) + beta * C[i].
template<typename Real>
void AddMatMatBatched(const Real alpha, std::vector<CuSubMatrix<Real>* > &C,
                      const std::vector<CuSubMatrix<Real>* > &A,
                      MatrixTransposeType transA,
                      const std::vector<CuSubMatrix<Real>* > &B,
                      MatrixTransposeType transB,
                      const Real beta);

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
  friend class CuSparseMatrix<float>;
  friend class CuSparseMatrix<double>;
  friend class CuSparseMatrix<Real>;

  /// Copies column r from column indexes[r] of src.
  /// As a special case, if indexes[i] == -1, sets column i to zero
  /// indexes.size() must equal this->NumCols(),
  /// and src.NumRows() must equal this.NumRows()
  void CopyCols(const CuMatrixBase<Real> &src,
                const CuArrayBase<MatrixIndexT> &indexes);


  /// Add column indices[r] of src to column r.
  /// As a special case, if indexes[i] == -1, skip column i
  /// indices.size() must equal this->NumCols(),
  /// and src.NumRows() must equal this.NumRows()
  void AddCols(const CuMatrixBase<Real> &src,
               const CuArrayBase<MatrixIndexT> &indices);

  /// Copies row r from row indexes[r] of src.
  /// As a special case, if indexes[i] < 0, sets row i to zero.
  /// src.NumCols() must equal this.NumCols()
  void CopyRows(const CuMatrixBase<Real> &src,
                const CuArrayBase<MatrixIndexT> &indexes);

  /// Copies row r of this matrix from an array of floats at the location given
  /// by src[r], where src[r] is assumed to be obtained from the RowData()
  /// function of another CuMatrix, or from CuVector::Data() (the point is: the
  /// data it points to should be on the GPU if we're using a GPU, and on a CPU
  /// otherwise).  src.size() must equal this.NumRows(), and if any src[r] is
  /// NULL then this.Row(r) will be set to zero.
  void CopyRows(const CuArrayBase<const Real*> &src);

  /// For each row r of this matrix, copies it to the array of floats at the
  /// location given by dst[r], where dst[r] is assumed to be obtained from the
  /// RowData() function of another CuMatrix, or from CuVector::Data() (i.e. it
  /// should point to memory on the GPU if we're using a GPU, or on the CPU
  /// otherwise).  If dst[r] is NULL, does not copy anywhere.  Requires that
  /// none of the memory regions pointed to by the pointers in "dst" overlap
  /// (e.g. none of the pointers should be the same).
  void CopyToRows(const CuArrayBase<Real*> &dst) const;

  /// Does for each row r, this.Row(r) += alpha * src.row(indexes[r]).
  /// If indexes[r] < 0, does not add anything.
  /// src.NumCols() must equal this.NumCols()
  void AddRows(Real alpha,
               const CuMatrixBase<Real> &src,
               const CuArrayBase<MatrixIndexT> &indexes);


  /// Does for each row r, this.Row(r) *= alpha * src.row(indexes[r]),
  /// where '*=' is elementwise multiplication.
  /// If indexes[r] < 0, does not add anything.
  /// src.NumCols() must equal this.NumCols()
  void MulRows(const CuMatrixBase<Real> &src,
               const CuArrayBase<MatrixIndexT> &indexes);


  /// Does for each row r, this.Row(r) += alpha * src[r],
  /// treating src[r] as the beginning of a region of memory representing
  /// a vector of floats, of the same length as this.NumCols().
  void AddRows(Real alpha,
               const CuArrayBase<const Real*> &src);


  /// For each row i of *this, adds this->Row(i) to
  /// dst->Row(indexes(i)) if indexes(i) >= 0, else do nothing.
  /// Requires that all the indexes[i] that are >= 0
  /// be distinct, otherwise the behavior is undefined.
  void AddToRows(Real alpha,
                 const CuArrayBase<MatrixIndexT> &indexes,
                 CuMatrixBase<Real> *dst) const;


  /// For each row r of this matrix, adds it (times alpha) to the array of
  /// floats at the location given by dst[r], where dst[r] is assumed to be
  /// obtained from the RowData() function of another CuMatrix, or from
  /// CuVector::Data() (i.e. it should point to memory on the GPU if we're using
  /// a GPU, or on the CPU otherwise).  If dst[r] is NULL, does not do anything
  /// for that row.  Requires that none of the memory regions pointed to by the
  /// pointers in "dst" overlap (e.g. none of the pointers should be the same).
  void AddToRows(Real alpha, const CuArrayBase<Real*> &dst) const;


  /// For each row r of this and for each column c, sets (*this)(r, c) to the
  /// sum \sum_j src(r, j), where j ranges from indexes[c].first through
  /// indexes[c].second - 1.
  void SumColumnRanges(const CuMatrixBase<Real> &src,
                       const CuArrayBase<Int32Pair> &indexes);


  /// For each row r of this and for each column c, do
  /// (*this)(r, c) += \sum_j src(j, c),
  /// where j ranges from indexes[r].first through indexes[r].second - 1.
  /// In general indexes must be >= 0 and < src.NumRows(); but to represent an empty range
  /// you may use the pair (-1, -1) or any pair of numbers (i, j) such that i >= j.
  void AddRowRanges(const CuMatrixBase<Real> &src,
                    const CuArrayBase<Int32Pair> &indexes);


  friend Real TraceMatMat<Real>(const CuMatrixBase<Real> &A,
                                const CuMatrixBase<Real> &B,
                                MatrixTransposeType trans);

  friend Real TraceMatSmat<Real>(const CuMatrixBase<Real> &A,
                                 const CuSparseMatrix<Real> &B,
                                 MatrixTransposeType trans);

  friend void AddMatMatBatched<Real>(const Real alpha,
                                     std::vector<CuSubMatrix<Real>* > &C,
                                     const std::vector<CuSubMatrix<Real>* > &A,
                                     MatrixTransposeType transA,
                                     const std::vector<CuSubMatrix<Real>* > &B,
                                     MatrixTransposeType transB,
                                     const Real beta);

  /// Adds "value" to the diagonal elements of the matrix.  The matrix
  /// *this does not have to be square.
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

  void CopyFromGeneralMat(const GeneralMatrix &src,
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

  /// This function has two modes of operation.  If v.Dim() == NumRows() *
  /// NumCols(), then treats the vector as a row-by-row concatenation of a
  /// matrix and copies to *this.
  /// if v.Dim() == NumCols(), it sets each row of *this to a copy of v.
  void CopyRowsFromVec(const CuVectorBase<Real> &v);

  /// Version of CopyRowsFromVec() that takes a CPU-based vector.
  void CopyRowsFromVec(const VectorBase<Real> &v);

  /// Copies vector into matrix, column-by-column.
  /// Note that rv.Dim() must either equal NumRows()*NumCols() or NumRows();
  /// this has two modes of operation.
  void CopyColsFromVec(const CuVectorBase<Real> &v);

  /// Copy vector into specific column of matrix.
  void CopyColFromVec(const CuVectorBase<Real> &v, const MatrixIndexT col);

  /// Set each element to the sigmoid of the corresponding element of "src":
  /// element by element, x = 1 / (1 + exp(-x))
  void Sigmoid(const CuMatrixBase<Real> &src);

  /// Set each element to the Heaviside function of the corresponding element
  /// of "src", which we define as the function (x > 0 ? 1.0 : 0.0) [note:
  /// in general, there are different ways to deal with the situation when x==0.]
  void Heaviside(const CuMatrixBase<Real> &src);

  /// Apply the function y = log(1 + exp(x)), to each element.
  /// Note: the derivative of this function is the sigmoid function.
  /// This is like a soft ReLU.
  void SoftHinge(const CuMatrixBase<Real> &src);

  /// Apply the function y(i) = (sum_{j = i*G}^{(i+1)*G-1} x_j ^ (power)) ^ (1 / p)
  /// where G = x.NumCols() / y.NumCols() must be an integer.
  /// [note: y corresponds to *this and x to src, so
  ///  src.NumCols() / this->NumCols() must be an integer.
  void GroupPnorm(const CuMatrixBase<Real> &src, Real pow);

  /// Differentiate backward through the GroupPnorm function.
  /// It is a combination of GroupPnormDeriv and MulRowsGroupMat.
  void DiffGroupPnorm(const CuMatrixBase<Real> &in_value,
                      const CuMatrixBase<Real> &out_value,
                      const CuMatrixBase<Real> &out_deriv, Real power);

  /// Apply the function y(i) = (max_{j = i*G}^{(i+1)*G-1} x_j
  /// where G = x.NumCols() / y.NumCols() must be an integer.
  /// [note: y corresponds to *this and x to src, so
  ///  src.NumCols() / this->NumCols() must be an integer.
  void GroupMax(const CuMatrixBase<Real> &src);

  /// Calculate derivatives for the GroupMax function above, where
  /// "input" is the input to the GroupMax function above (i.e. the "src" variable),
  /// and "output" is the result of the computation (i.e. the "this" of that function
  /// call), and *this must have the same dimension as "input". Each element
  /// of *this will be set to 1 if the corresponding input equals the output of
  /// the group, and 0 otherwise. The equals the function derivative where it is
  /// defined (it's not defined where multiple inputs in the group are equal to the output).
  void GroupMaxDeriv(const CuMatrixBase<Real> &input,
                     const CuMatrixBase<Real> &output);

  /// Compute the parametric rectified linear unit function;
  /// element by element, *this = src * (src > 0 ? alpha : beta)
  void ParametricRelu(const CuMatrixBase<Real> &src,
                      const CuVectorBase<Real> &alpha,
                      const CuVectorBase<Real> &beta);

  /// Differentiate backward through the parametric relu function.
  /// Here the "value" is the Relu input. Does, element-by-element.
  /// *this = diff * (value > 0 ? alpha : beta)
  void DiffParametricRelu(const CuMatrixBase<Real> &value,
                          const CuMatrixBase<Real> &diff,
                          const CuVectorBase<Real> &alpha,
                          const CuVectorBase<Real> &beta);

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

  /// Differentiate backward through the softmax function.  Here, "value" is the
  /// softmax output. Does, for each row i,
  /// *this(i) =  diff(i) * diag(value(i)) - diff(i) * (value(i)^T * value(i))
  /// xxxx(i) is row-vector; '*' and '-' are matrix operations.
  /// Supports in-place operation, this  == &diff.
  void DiffSoftmaxPerRow(const CuMatrixBase<Real> &value,
                         const CuMatrixBase<Real> &diff);

  /// Differentiate backward through the log softmax function.
  /// Here, "out_value" is the log softmax output. Does, for each row i,
  /// *this(i) =  out_deriv(i) - sum(out_deriv(i)) .* exp(out_value(i))
  /// xxxx(i) is row-vector.
  /// Supports in-place operation, this == &out_deriv.
  void DiffLogSoftmaxPerRow(const CuMatrixBase<Real> &out_value,
                            const CuMatrixBase<Real> &out_deriv);

  /// Differentiate the block [softmax+cross-entropy] :
  /// dE/da = posterior_mat - target_mat,
  /// 'E' is error function, 'a' is activation on softmax input
  ///
  /// Interface:
  /// tgt ... index vector, encodes the matrix of targets
  /// net_out_or_diff ... before invocation net output, after diff dE/da
  /// log_post_tgt ... per-frame statistics for cross-entropy computations :
  ///                  log(sum_row(posterior_mat .* target_mat))
  void DiffXent(const CuArrayBase<int32> &tgt,
                CuVector<Real> *log_post_tgt);

  /// This function does sets *this to the Cholesky factor of *this (i.e.  the C
  /// satisfying *this = C C^T), and sets "inv_cholesky" (if supplied) to its
  /// inverse.  *this is treated as a symmetric matrix but only the lower triangle
  /// is accessed.
  void Cholesky(CuMatrixBase<Real> *inv_cholesky = NULL);


  /// Inversion for positive definite symmetric matrices.
  /// Treats the input as symmetric but only reads the lower triangle.
  /// The output is symmetric.
  void SymInvertPosDef();

  void ApplyPow(Real power);
  /// Apply power to the absolute value of each element.
  /// If include_sign is true, the result will be multiplied with
  /// the sign of the input value.
  /// If the power is negative and the input to the power is zero,
  /// The output will be set zero. If include_sign is true, it will
  /// multiply the result by the sign of the input.
  void ApplyPowAbs(Real power, bool include_sign=false);
  /// For each element, sets x = (x > 0 ? 1.0 : 0.0).
  /// See also Heaviside().
  void ApplyHeaviside();
  void ApplyFloor(Real floor_val);
  void ApplyCeiling(Real ceiling_val);
  void ApplyExp();


  /// This is equivalent to running:
  /// ApplyFloor(lower_limit);
  /// ApplyCeiling(upper_limit);
  /// ApplyExp()
  void ApplyExpLimited(Real lower_limit, Real upper_limit);

  /// For each element x of the matrix, set it to
  /// (x < 0 ? exp(x) : x + 1).  This function is used
  /// in our RNNLM training.
  void ApplyExpSpecial();

  /// Softmax nonlinearity
  /// Y = Softmax(X) : Yij = e^Xij / sum_k(e^Xik), done to each row,
  /// with attention to avoiding  overflow or underflow.
  /// Supports in-place operation (i.e. this == &src).
  void ApplySoftMaxPerRow(const CuMatrixBase<Real> &src);

  /// LogSoftmax nonlinearity
  /// Y = LogSoftmax(X) : Yij = Xij - log(sum_k(e^Xik)), done to each row,
  /// with attention to avoiding  overflow or underflow.
  /// Supports in-place operation (i.e. this == &src).
  void ApplyLogSoftMaxPerRow(const CuMatrixBase<Real> &src);

  /// Find the id of the maximal element for each row (resizes the 'id'
  /// array to the appropriate size).
  void FindRowMaxId(CuArray<int32> *id) const;

  /// Math operations, some calling kernels
  void SetZero();
  void Set(Real value);
  void Add(Real value);
  /// Zeroes all elements for which col > row.
  void SetZeroAboveDiag();
  void Scale(Real value);
  void ApplyLog();

  /// Multiply two matrices elementwise: C = C .* A
  void MulElements(const CuMatrixBase<Real> &A);
  /// Divide two matrices elementwise: C = A ./ A
  void DivElements(const CuMatrixBase<Real> &A);
  /// Do, elementwise, *this = max(*this, A).
  void Max(const CuMatrixBase<Real> &A);
  /// Do, elementwise, *this = min(*this, A).
  void Min(const CuMatrixBase<Real> &A);
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
  /// *this += alpha * A
  void AddMat(Real alpha, const CuMatrixBase<Real> &A,
              MatrixTransposeType trans = kNoTrans);

  /// *this += alpha * A.
  void AddSmat(Real alpha, const CuSparseMatrix<Real> &A,
              MatrixTransposeType trans = kNoTrans);

  /// (*this) = alpha * op(A) * B + beta * (*this), where A is sparse.
  /// Multiplication of sparse with dense matrix.  See also AddMatSmat.
  /// Note: we recommend, for greatest efficiency, that transA be kNoTrans.
  /// Use AddMatSmat() for better efficiency, as 2 dense mat transpose ops
  /// are called in this API.
  void AddSmatMat(Real alpha, const CuSparseMatrix<Real> &A,
                  MatrixTransposeType transA, const CuMatrixBase<Real> &B,
                  Real beta);

  /// (*this) = alpha * A * op(B) + beta * (*this), where B is sparse
  /// and op(B) is either B or trans(B) depending on the 'transB' argument.
  /// This is multiplication of a dense by a sparse matrix.  See also
  /// AddSmatMat.
  void AddMatSmat(Real alpha, const CuMatrixBase<Real> &A,
                  const CuSparseMatrix<Real> &B, MatrixTransposeType transB,
                  Real beta);


  /// This is a rather special purpose function; we might
  /// generalize it later by adding a transpose-type option.
  /// It expects 'elements.Dim()' to equal NumRows(), and
  /// for each elements[i] to be either -1, or
  /// 0 <= element[i] < NumCols().
  /// It adds alpha to each element (*this)(i, elements[i])
  /// for 0 <= i < NumRows().
  void AddToElements(Real alpha, const CuArrayBase<int32> &elements);


  /// This function is like AddMat (it does *this += alpha * src),
  /// except that it supports cases where *this and src have
  /// different dimension.  There are two allowed cases:
  ///
  ///  (1) *this is larger than src; we do a broadcasting operation.  *this must
  ///       have NumRows() == a * src.NumRows() and NumCols() == b *
  ///       src.NumCols() for integer a >= 1, b >= 1.  *this will be treated as
  ///       a being made up of of blocks with the same size as src, and to each
  ///       block we'll add alpha * src.  This case does not support trans ==
  ///       kTrans.
  ///
  ///  (2) *this is smaller than src; we sum.  src.NumRows() must == a *
  ///      this->NumRows(), and src.NumCols() must == b * this->NumCols(), for a
  ///      >= 1, b >= 1.  In this case, src will be treated as being made up of
  ///      blocks with the same size as *this, and to *this we will add the
  ///      summation of all of those blocks.
  void AddMatBlocks(Real alpha, const CuMatrixBase<Real> &A,
                    MatrixTransposeType trans = kNoTrans);

  /// (for each column c of *this), c = alpha * col + beta * c
  void AddVecToCols(Real alpha, const CuVectorBase<Real> &col, Real beta = 1.0);
  /// (for each row r of *this), r = alpha * row + beta * r
  void AddVecToRows(Real alpha, const CuVectorBase<Real> &row, Real beta = 1.0);
  /// C = alpha * A(^T)*B(^T) + beta * C
  void AddMatMat(Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
                 const CuMatrixBase<Real> &B, MatrixTransposeType transB, Real beta);
  /// A = alpha * x * y^T + A .
  void AddVecVec(Real alpha, const CuVectorBase<Real> &x, const CuVectorBase<Real> &y);
  /// *this = a * b / c (by element; when c = 0, *this = a)
  /// *this can be an alias of a, b or c safely and get expected result.
  void SetMatMatDivMat(const CuMatrixBase<Real> &A, const CuMatrixBase<Real> &B, const CuMatrixBase<Real> &C);

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
  void AddDiagVecMat(const Real alpha, const CuVectorBase<Real> &v,
                     const CuMatrixBase<Real> &M, MatrixTransposeType transM,
                     Real beta = 1.0);

  // *this = beta * *this + alpha * M  * diag(v) [or M^T].
  // The same as adding M but scaling each column M_j by v(j).
  void AddMatDiagVec(const Real alpha,
                     const CuMatrixBase<Real> &M, MatrixTransposeType transM,
                     CuVectorBase<Real> &v,
                     Real beta = 1.0);

  /// *this = beta * *this + alpha * A .* B (.* element by element multiplication)
  void AddMatMatElements(const Real alpha,
                         const CuMatrixBase<Real>& A,
                         const CuMatrixBase<Real>& B,
                         const Real beta);

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
  Real Max() const;
  Real Min() const;

  /// Return the trace. If check_square = true, will crash if matrix is not square.
  Real Trace(bool check_square = true) const;

  void SetRandn();

  void SetRandUniform();

  void Write(std::ostream &os, bool binary) const;

  // This function, adds a list of MatrixElements (scaled by alpha) to corresponding locations to
  // (*this).
  void AddElements(Real alpha, const std::vector<MatrixElement<Real> >& input);

  // For each i, with indexes[i] = (j, k), does (*this)(j, k) += input[i].
  // Requires, but does not check, that the vector of indexes does not contrain
  // repeated elements, 'input' is the start of an array of length equal to
  // indexes.Dim(), which is located on GPU memory if we are using the GPU.
  void AddElements(Real alpha, const CuArrayBase<Int32Pair> &indexes,
                   const Real *input);

  // This function requires that 'output' is a host array and is allocated with size
  // of indexes.size(), and for each element of 'indexes' it interprets it as
  // a (row, column) index into *this, and puts (*this)(row, column) into
  // the corresponding element of 'output'.
  void Lookup(const std::vector<Int32Pair> &indexes,
              Real *output) const;

  // CUDA version of Lookup, would be called internally by the above function.
  void Lookup(const CuArrayBase<Int32Pair> &indexes,
              Real *output) const;

  // Creates binary mask with per-element equality predicates of *this, mat.
  // Output stored to 'mask', values : 1.0 = equal, 0.0 = not-equal.
  void EqualElementMask(const CuMatrixBase<Real> &mat, CuMatrix<Real> *mask) const;


  /// Get raw row pointer (const).  Warning: may return a pointer to GPU memory.  Use at
  /// your own risk.
  inline const Real* RowData(MatrixIndexT r) const { return data_ + r * stride_; }
  /// Get raw row pointer.  Warning: may return a pointer to GPU memory.  Use at
  /// your own risk.
  inline Real* RowData(MatrixIndexT r) { return data_ + r * stride_; }
  /// Return data pointer (const).  Warning: may return a pointer to GPU memory.
  /// Use at your own risk.
  inline const Real *Data() const { return data_; }
  /// Return data pointer.  Warning: may return a pointer to GPU memory.  Use at
  /// your own risk.
  inline Real *Data() { return data_; }

  // The following two functions should only be called if we did not compile
  // with CUDA or could not get a CUDA card; in that case the contents are
  // interpreted the same as a regular matrix.  DON'T USE THESE UNLESS YOU KNOW
  // WHAT YOU ARE DOING!
  inline const MatrixBase<Real> &Mat() const {
    return *(reinterpret_cast<const MatrixBase<Real>* >(this));
  }
  inline MatrixBase<Real> &Mat() {
    return *(reinterpret_cast<MatrixBase<Real>* >(this));
  }

 protected:

  // The constructors are protected to prevent the user creating an instance of
  // this class (you should create a child class CuMatrix or CuSubMatrix.

  CuMatrixBase(): data_(NULL), num_cols_(0), num_rows_(0), stride_(0) { }

  /// This constructor takes the #rows, #cols and stride; it's called from
  /// the constructor of CuSubMatrix.
  CuMatrixBase(Real *data,
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
           MatrixResizeType resize_type = kSetZero,
           MatrixStrideType stride_type = kDefaultStride) {
    Resize(rows, cols, resize_type, stride_type);
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
              MatrixResizeType resize_type = kSetZero,
              MatrixStrideType stride_type = kDefaultStride);

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

  /// Here, A is interpreted as a matrix of probabilities, and "elements" as a list
  /// of posteriors (possibly zero-one), and "*this" as a matrix of derivatives
  /// w.r.t. the log-probs.
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
                        Real *tot_weight);

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

  // This constructor should be used with caution; it can be used for
  // constructing 'fake' submatrices if you want to play with
  // the stride. 'data' should point to GPU data if you're using the
  // GPU.
  inline CuSubMatrix(const Real *data,
                     const MatrixIndexT num_rows,
                     const MatrixIndexT num_cols,
                     const MatrixIndexT stride);

  /// This type of constructor is needed for Range() to work [in CuMatrix base
  /// class]. Cannot make it explicit or that breaks.
  inline CuSubMatrix<Real> (const CuSubMatrix &other):
  CuMatrixBase<Real> (other.data_, other.num_rows_, other.num_cols_,
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
inline void AssertEqual(const CuMatrixBase<Real> &A,
                        const CuMatrixBase<Real> &B, float tol = 0.01) {
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
  if (trans == kNoTrans) Init(M.NumRows(), M.NumCols(), kDefaultStride);
  else Init(M.NumCols(), M.NumRows(), kDefaultStride);
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

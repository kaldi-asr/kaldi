// matrix/kaldi-matrix.h

// Copyright 2009-2011  Ondrej Glembek,  Microsoft Corporation,  Lukas Burget,  Arnab Ghoshal
//                 Petr Schwarz,  Yanmin Qian,  Karel Vesely,  Go Vivace Inc.

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

#ifndef KALDI_MATRIX_KALDI_MATRIX_H_
#define KALDI_MATRIX_KALDI_MATRIX_H_ 1

#include "matrix-common.h"

namespace kaldi {

/// @{ \addtogroup matrix_funcs_scalar

/// We need to declare this here as it will be a friend function.
/// tr(A B), or tr(A B^T).
template<typename Real>
Real TraceMatMat(const MatrixBase<Real> &A, const MatrixBase<Real> &B,
                 MatrixTransposeType trans = kNoTrans);
/// @}

/// \addtogroup matrix_group
/// @{

/// Base class which provides matrix operations not involving resizing
/// or allocation.   Classes Matrix and SubMatrix inherit from it and take care
/// of allocation and resizing.
template<class Real>
class MatrixBase {
 public:
  // so this child can access protected members of other instances.
  friend class Matrix<Real>;

  /// Returns number of rows (or zero for emtpy matrix).
  inline MatrixIndexT  NumRows() const { return num_rows_; }

  /// Returns number of columns (or zero for emtpy matrix).
  inline MatrixIndexT NumCols() const { return num_cols_; }

  /// Stride (distance in memory between each row).  Will be >= NumCols.
  inline MatrixIndexT Stride() const {  return stride_; }

  /// Returns size in bytes of the data held by the matrix.
  MatrixIndexT  SizeInBytes() const { return num_rows_ * stride_ * sizeof(Real); }

  /// Gives pointer to raw data (const).
  inline const Real* Data() const {
    return data_;
  }

  /// Gives pointer to raw data (non-const).
  inline Real* Data() { return data_; }

  /// Returns pointer to data for one row (non-const)
  inline  Real* RowData(MatrixIndexT i) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return data_ + i * stride_;
  }

  /// Returns pointer to data for one row (const)
  inline const Real* RowData(MatrixIndexT i) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return data_ + i * stride_;
  }

  /// Indexing operator, non-const
  /// (only checks sizes if compiled with -DKALDI_PARANOID)
  inline Real&  operator() (MatrixIndexT r, MatrixIndexT c) {
#ifdef KALDI_PARANOID
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                 static_cast<UnsignedMatrixIndexT>(c) <
                 static_cast<UnsignedMatrixIndexT>(num_cols_));
#endif
    return *(data_ + r * stride_ + c);
  }

  /// Indexing operator, const
  /// (only checks sizes if compiled with -DKALDI_PARANOID)
  inline const Real operator() (MatrixIndexT r, MatrixIndexT c) const {
#ifdef KALDI_PARANOID
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                 static_cast<UnsignedMatrixIndexT>(c) <
                 static_cast<UnsignedMatrixIndexT>(num_cols_));
#endif
    return *(data_ + r * stride_ + c);
  }

  /*   Basic setting-to-special values functions. */

  /// Sets matrix to zero.
  void SetZero();
  /// Sets all elements to a specific value.
  void Set(Real);
  /// Sets to zero, except ones along diagonal [for non-square matrices too]
  void SetUnit();
  /// Sets to random values of a normal distribution
  void SetRandn();

  /*  Copying functions.  These do not resize the matrix! */

  /// Copy given matrix. (no resize is done).
  template<typename OtherReal>
  void CopyFromMat(const MatrixBase<OtherReal> & M,
                   MatrixTransposeType Trans = kNoTrans);

  /// Copy given spmatrix. (no resize is done).
  template<typename OtherReal>
  void CopyFromSp(const SpMatrix<OtherReal> &M);

  /// Copy given tpmatrix. (no resize is done).
  template<typename OtherReal>
  void CopyFromTp(const TpMatrix<OtherReal> &M,
                  MatrixTransposeType Trans = kNoTrans);

  /// Inverse of vec() operator. Copies vector into matrix, row-by-row.
  /// Note that rv.Dim() must equal NumRows()*NumCols().
  void CopyRowsFromVec(const VectorBase<Real> &v);
  template<typename OtherReal>
  void CopyRowsFromVec(const VectorBase<OtherReal> &v);

  /// Copies vector into matrix, column-by-column.
  /// Note that rv.Dim() must equal NumRows()*NumCols().
  void CopyColsFromVec(const VectorBase<Real> &v);
  /// Copy vector into specific column of matrix.
  void CopyColFromVec(const VectorBase<Real> &v, const MatrixIndexT col);
  /// Copy vector into specific row of matrix.
  void CopyRowFromVec(const VectorBase<Real> &v, const MatrixIndexT row);
  /// Copy vector into diagonal of matrix.
  void CopyDiagFromVec(const VectorBase<Real> &v);

  /* Accessing of sub-parts of the matrix. */

  /// Return specific row of matrix [const].
  inline const SubVector<Real> Row(MatrixIndexT i) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return SubVector<Real>(data_ + (i * stride_), NumCols());
  }

  /// Return specific row of matrix.
  inline SubVector<Real> Row(MatrixIndexT i) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return SubVector<Real>(data_ + (i * stride_), NumCols());
  }

  /// Return a sub-part of matrix.
  /// @param ro [in] row offset.
  /// @param r [in] num-rows in result.
  /// @param co [in] column offset.
  /// @param c [in] num-columns in result.
  SubMatrix<Real> Range(const MatrixIndexT ro, const MatrixIndexT r, const MatrixIndexT co,
                        const MatrixIndexT c) const {
    return SubMatrix<Real>(*this, ro, r, co, c);
  }

  /* Various special functions. */
  /// Returns sum of all elements in matrix.
  Real Sum() const;
  /// Returns trace of matrix.
  Real Trace(bool check_square = true) const;
  // If check_square = true, will crash if matrix is not square.

  /// Returns maximum element of matrix.
  Real Max() const;
  /// Returns minimum element of matrix.
  Real Min() const;

  /// dot-product, element by element
  void MulElements(const MatrixBase<Real>& a);

  /// Multiply each element with a scalar value.
  void Scale(Real alpha);

  /// Equivalent to (*this) = (*this) * diag(scale).
  void MulColsVec(const VectorBase<Real> &scale);

  /// Equivalent to (*this) = diag(scale) * (*this);
  void MulRowsVec(const VectorBase<Real> &scale);

  /// Returns logdet of matrix.
  Real LogDet(Real *det_sign = NULL) const;

  /// matrix inverse.
  /// if inverse_needed = false, will fill matrix with garbage.
  /// (only useful if logdet wanted).
  void Invert(Real *log_det = NULL, Real *det_sign = NULL,
              bool inverse_needed = true);

  /// matrix inverse [double].
  /// if inverse_needed = false, will fill matrix with garbage
  /// (only useful if logdet wanted).
  /// Does inversion in double precision even if matrix was not double.
  void InvertDouble(Real *LogDet = NULL, Real *det_sign = NULL,
                      bool inverse_needed = true);

  /// Inverts all the elements of the matrix
  void InvertElements();

  /// Transpose the matrix.  This one is only
  /// applicable to square matrices (the one in the
  /// Matrix child class works also for non-square.
  void Transpose();

  /// Applies floor to all matrix elements
  void ApplyFloor(Real floor_val);

  /// Calculates log to all the matrix elemnts
  void ApplyLog();

  /// Applies power to all matrix elements
  void ApplyPow(Real power);

  /// Eigenvalue Decomposition of a square NxN matrix into the form (*this) = P D
  /// P^{-1}.  Be careful: the relationship of D to the eigenvalues we output is
  /// slightly complicated, due to the need for P to be real.  In the symmetric
  /// case D is diagonal and real, but in
  /// the non-symmetric case there may be complex-conjugate pairs of eigenvalues.
  /// In this case, for the equation (*this) = P D P^{-1} to hold, D must actually
  /// be block diagonal, with 2x2 blocks corresponding to any such pairs.  If a
  /// pair is lambda +- i*mu, D will have a corresponding 2x2 block
  /// [lambda, mu; -mu, lambda].
  /// Note that if the input matrix (*this) is non-invertible, P may not be invertible
  /// so in this case instead of the equation (*this) = P D P^{-1} holding, we have
  /// instead (*this) P = P D.
  ///
  /// The non-member function CreateEigenvalueMatrix creates D from eigs_real and eigs_imag.
  void Eig(MatrixBase<Real> *P,
           VectorBase<Real> *eigs_real,
           VectorBase<Real> *eigs_imag) const;

  /// The Power method attempts to take the matrix to a power using a method that
  /// works in general for fractional and negative powers.  The input matrix must
  /// be invertible and have reasonable condition (or we don't guarantee the
  /// results.  The method is based on the eigenvalue decomposition.  It will
  /// return false and leave the matrix unchanged, if at entry the matrix had
  /// real negative eigenvalues (or if it had zero eigenvalues and the power was
  /// negative).
  bool Power(Real pow);

  /** Singular value decomposition
     Major limitations:
     For nonsquare matrices, we assume m>=n (NumRows >= NumCols), and we return
     the "skinny" Svd, i.e. the matrix in the middle is diagonal, and the
     one on the left is rectangular.

     In Svd, *this = U*diag(S)*Vt.
     Null pointers for U and/or Vt at input mean we do not want that output.  We
     expect that S.Dim() == m, U is either NULL or m by n,
     and v is either NULL or n by n. */
  void DestructiveSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                        MatrixBase<Real> *Vt);  // Destroys calling matrix.

  /// Compute SVD (*this) = U diag(s) Vt.   Note that the v in the call is already
  /// transposed; the normal formulation is U diag(s) V^T.
  /// Null pointers for U or V mean we don't want that output (this saves
  /// compute).
  void Svd(VectorBase<Real> *s, MatrixBase<Real> *U,
           MatrixBase<Real> *Vt) const;
  /// Compute SVD but only retain the singular values.
  void Svd(VectorBase<Real> *s) const { Svd(s, NULL, NULL); }


  /// Returns smallest singular value.
  Real MinSingularValue() const {
    Vector<Real> tmp(std::min(NumRows(), NumCols()));
    Svd(&tmp);
    return tmp.Min();
  }

  /// returns condition number by computing Svd.  Works even if cols > rows.
  Real Cond() const;

  /// Returns true if matrix is Symmetric.
  bool IsSymmetric(Real cutoff = 1.0e-05) const;  // replace magic number

  /// Returns true if matrix is Diagonal.
  bool IsDiagonal(Real cutoff = 1.0e-05) const;  // replace magic number

  /// returns true if matrix is all zeros, but ones on diagonal
  /// (not necessarily square).
  bool IsUnit(Real cutoff = 1.0e-05) const;     // replace magic number

  /// Returns true if matrix is all zeros.
  bool IsZero(Real cutoff = 1.0e-05) const;     // replace magic number

  /// Frobenius norm, which is the sqrt of sum of square elements.  Same as Schatten 2-norm,
  /// or just "2-norm".
  Real FrobeniusNorm() const;

  /// Returns true if ((*this)-other).FrobeniusNorm()
  /// <= tol * (*this).FrobeniusNorm().
  bool ApproxEqual(const MatrixBase<Real> &other, float tol = 0.01) const;

  /// Tests for exact equality.  It's usually preferable to use ApproxEqual.
  bool Equal(const MatrixBase<Real> &other) const;

  /// largest absolute value.
  Real LargestAbsElem() const;  // largest absolute value.

  /// Returns log(sum(exp())) without exp overflow
  /// If prune > 0.0, it uses a pruning beam, discarding
  /// terms less than (max - prune).
  Real LogSumExp(Real prune = 0.0) const;

  /// Apply soft-max to the collection of all elements of the
  /// matrix and return normalizer (log sum of exponentials).
  Real ApplySoftMax();

  /** Uses Svd to compute the eigenvalue decomposition of a symmetric positive
   * semi-definite matrix: (*this) = rP * diag(rS) * rP^T, with rP an
   * orthogonal matrix so rP^{-1} = rP^T.   Throws exception if input was not
   * positive semi-definite (check_thresh controls how stringent the check is;
   * set it to 2 to ensure it won't ever complain, but it will zero out negative
   * dimensions in your matrix.
  */
  void SymPosSemiDefEig(VectorBase<Real> *s, MatrixBase<Real> *P,
                        Real check_thresh = 0.001);

  friend Real kaldi::TraceMatMat<Real>(const MatrixBase<Real> &A,
      const MatrixBase<Real> &B, MatrixTransposeType trans);  // tr (A B)

  // so it can get around const restrictions on the pointer to data_.
  friend class SubMatrix<Real>;

  /// Add a scalar to each element
  void Add(const Real alpha);

  /// *this += alpha * a * b^T
  void AddVecVec(const Real alpha, const VectorBase<Real>& a,
                 const VectorBase<Real>& b);

  /// *this += alpha * a * b^T
  template<class OtherReal>
  void AddVecVec(const OtherReal alpha, const VectorBase<OtherReal>& a,
                 const VectorBase<OtherReal>& b);


  /// *this += alpha * M [or M^T]
  void AddMat(const Real alpha, const MatrixBase<Real> &M,
               MatrixTransposeType transA = kNoTrans);

  /// *this += alpha * S
  template<class OtherReal>
  void AddSp(const Real alpha, const SpMatrix<OtherReal> &S);

  /// this <-- beta*this + alpha*A*B.
  void AddMatMat(const Real alpha,
                 const MatrixBase<Real>& A, MatrixTransposeType transA,
                 const MatrixBase<Real>& B, MatrixTransposeType transB,
                 const Real beta);

  /// this <-- beta*this + alpha*A*B*C.
  void AddMatMatMat(const Real alpha,
                    const MatrixBase<Real>& A, MatrixTransposeType transA,
                    const MatrixBase<Real>& B, MatrixTransposeType transB,
                    const MatrixBase<Real>& C, MatrixTransposeType transC,
                    const Real beta);

  /// this <-- beta*this + alpha*SpA*B.
  // This and the routines below are really
  // stubs that need to be made more efficient.
  void AddSpMat(const Real alpha,
                const SpMatrix<Real>& A,
                const MatrixBase<Real>& B, MatrixTransposeType transB,
                const Real beta) {
    Matrix<Real> M(A);
    return AddMatMat(alpha, M, kNoTrans, B, transB, beta);
  }
  /// this <-- beta*this + alpha*A*B.
  void AddTpMat(const Real alpha,
                const TpMatrix<Real>& A, MatrixTransposeType transA,
                const MatrixBase<Real>& B, MatrixTransposeType transB,
                const Real beta) {
    Matrix<Real> M(A);
    return AddMatMat(alpha, M, transA, B, transB, beta);
  }
  /// this <-- beta*this + alpha*A*B.
  void AddMatSp(const Real alpha,
                const MatrixBase<Real>& A, MatrixTransposeType transA,
                const SpMatrix<Real>& B,
                const Real beta) {
    Matrix<Real> M(B);
    return AddMatMat(alpha, A, transA, M, kNoTrans, beta);
  }
  /// this <-- beta*this + alpha*A*B*C.
  void AddSpMatSp(const Real alpha,
                  const SpMatrix<Real> &A,
                  const MatrixBase<Real>& B, MatrixTransposeType transB,
                  const SpMatrix<Real>& C,
                const Real beta) {
    Matrix<Real> M(A), N(C);
    return AddMatMatMat(alpha, M, kNoTrans, B, transB, N, kNoTrans, beta);
  }
  /// this <-- beta*this + alpha*A*B.
  void AddMatTp(const Real alpha,
                const MatrixBase<Real>& A, MatrixTransposeType transA,
                const TpMatrix<Real>& B, MatrixTransposeType transB,
                const Real beta) {
    Matrix<Real> M(B);
    return AddMatMat(alpha, A, transA, M, transB, beta);
  }

  /// this <-- beta*this + alpha*A*B.
  void AddTpTp(const Real alpha,
               const TpMatrix<Real>& A, MatrixTransposeType transA,
               const TpMatrix<Real>& B, MatrixTransposeType transB,
               const Real beta) {
    Matrix<Real> M(A), N(B);
    return AddMatMat(alpha, M, transA, N, transB, beta);
  }

  /// this <-- beta*this + alpha*A*B.
  // This one is more efficient, not like the others above.
  void AddSpSp(const Real alpha,
               const SpMatrix<Real>& A, const SpMatrix<Real>& B,
               const Real beta);

  /// stream read.
  /// Use instead of stream<<*this, if you want to add to existing contents.
  // Will throw exception on failure.
  void Read(std::istream & in, bool binary, bool add = false);
  /// write to stream.
  void Write(std::ostream & out, bool binary) const;

  // Below is internal methods for Svd, user does not have to know about this.
#ifndef HAVE_ATLAS
  // protected:
  // Should be protected but used directly in testing routine.
  // destroys *this!
  void LapackGesvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                     MatrixBase<Real> *Vt);
#else
 protected:
  // destroys *this!
  bool JamaSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
               MatrixBase<Real> *V);

#endif
 protected:

  ///  Initializer, callable only from child.
  explicit MatrixBase(Real* data, MatrixIndexT cols, MatrixIndexT rows, MatrixIndexT stride) :
    data_(data), num_cols_(cols), num_rows_(rows), stride_(stride) {
    KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  }

  ///  Initializer, callable only from child.
  /// Empty initializer, for un-initialized matrix.
  explicit MatrixBase(): data_(NULL) {
    KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  }

  // Make sure pointers to MatrixBase cannot be deleted.
  ~MatrixBase() { }

  /// A workaround that allows SubMatrix to get a pointer to non-const data
  /// for const Matrix. Unfortunately C++ does not allow us to declare a
  /// "public const" inheritance or anything like that, so it would require
  /// a lot of work to make the SubMatrix class totally const-correct--
  /// we would have to override many of the Matrix functions.
  inline Real*  Data_workaround() const {
    return data_;
  }


  /// data memory area
  Real*   data_;

  /// these atributes store the real matrix size as it is stored in memory
  /// including memalignment
  MatrixIndexT    num_cols_;   /// < Number of columns
  MatrixIndexT    num_rows_;   /// < Number of rows
  /** True number of columns for the internal matrix. This number may differ
   * from num_cols_ as memory alignment might be used. */
  MatrixIndexT    stride_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(MatrixBase);
};

/// A class for storing matrices.
template<class Real>
class Matrix : public MatrixBase<Real> {
 public:

  /// Empty constructor.
  Matrix();

  /// Basic constructor.  Sets to zero by default.
  /// if set_zero == false, memory contents are undefined.
  Matrix(const MatrixIndexT r, const MatrixIndexT c, MatrixResizeType resize_type = kSetZero) :
    MatrixBase<Real>() { Resize(r, c, resize_type); }

  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(Matrix<Real> *other);

  /// Constructor from any MatrixBase. Can also copy with transpose.
  /// Allocates new memory.
  explicit Matrix(const MatrixBase<Real> & M,
                    MatrixTransposeType trans = kNoTrans);

  /// Same as above, but need to avoid default copy constructor.
  Matrix(const Matrix<Real> & M);  //  (cannot make explicit)

  /// Copy constructor: as above, but from another type.
  template<typename OtherReal>
  explicit Matrix(const MatrixBase<OtherReal> & M,
                    MatrixTransposeType trans = kNoTrans);

  /// Copy constructor taking SpMatrix...
  /// It is symmetric, so no option for transpose, and NumRows == Cols
  template<typename OtherReal>
  explicit Matrix(const SpMatrix<OtherReal> & M) : MatrixBase<Real>() {
    Resize(M.NumRows(), M.NumRows());
    CopyFromSp(M);
  }

  /// Copy constructor taking TpMatrix...
  template <typename OtherReal>
  explicit Matrix(const TpMatrix<OtherReal> & M,
                    MatrixTransposeType trans = kNoTrans) : MatrixBase<Real>() {
    if (trans == kNoTrans) {
      Resize(M.NumRows(), M.NumCols());
      CopyFromTp(M);
    } else {
      Resize(M.NumCols(), M.NumRows());
      CopyFromTp(M, kTrans);
    }
  }

  /// read from stream.
  // Unlike one in base, allows resizing.
  void Read(std::istream & in, bool binary, bool add = false);

  /// Remove a specified row.
  void  RemoveRow(MatrixIndexT i);

  /// Transpose the matrix.  Works for non-square
  /// matrices as well as square ones.
  void Transpose();

  /// Distructor to free matrices.
  ~Matrix() { Destroy(); }

  /// Deallocates memory and sets to empty matrix.
  void Destroy();

  /// Sets matrix to a specified size (zero is OK as long as both r and c are
  /// zero).  The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  /// This function takes time proportional to the number of data elements.
  void Resize(const MatrixIndexT r,
              const MatrixIndexT c,
              MatrixResizeType resize_type = kSetZero);

  /// Assignment operator. Needed for inclusion in std::vector.
  Matrix<Real> &operator = (const Matrix<Real> &other) {
    if (MatrixBase<Real>::NumRows() != other.NumRows() ||
        MatrixBase<Real>::NumCols() != other.NumCols()) {
      Resize(other.NumRows(), other.NumCols());
    }
    MatrixBase<Real>::CopyFromMat(other);
    return *this;
  }


 private:
  /// Init assumes the current class contents are invalid (i.e. junk or have
  /// already been freed), and it sets the matrix to newly allocated memory with
  /// the specified number of rows and columns.  r == c == 0 is acceptable.  The data
  /// memory contents will be undefined.
  void Init(const MatrixIndexT r,
            const MatrixIndexT c);

#ifdef KALDI_MEMALIGN_MANUAL
  /// data to be freed (in case of manual memory alignment).
  Real*   free_data_;
#endif
};
/// @} end "addtogroup matrix_group"

/// \addtogroup matrix_funcs_io
/// @{

/// A structure containing the HTK header.
struct HtkHeader {
  /// Number of samples.
  int32    mNSamples;
  /// Sample period.
  int32    mSamplePeriod;
  /// Sample size
  int16    mSampleSize;
  /// Sample kind.
  uint16   mSampleKind;
};

// Read HTK formatted features from file into matrix.
template<class Real>
bool ReadHtk(std::istream &is, Matrix<Real> *M, HtkHeader *header_ptr);

// Write (HTK format) features to file from matrix.
template<class Real>
bool WriteHtk(std::ostream &os, const MatrixBase<Real> &M, HtkHeader htk_hdr);


/// @} end of "addtogroup matrix_funcs_io"

/**
  Sub-matrix representation.
  Can work with sub-parts of a matrix using this class.
  Note that SubMatrix is not very const-correct-- it allows you to
  change the contents of a const Matrix.  Be careful!
*/

template<typename Real>
class SubMatrix : public MatrixBase<Real> {
 public:
  // Input matrix is not really const since SubMatrix can change its contents.
  SubMatrix(const MatrixBase<Real>& T,
            const MatrixIndexT ro,  // row offset, 0 < ro < NumRows()
            const MatrixIndexT r,   // number of rows, r > 0
            const MatrixIndexT co,  // column offset, 0 < co < NumCols()
            const MatrixIndexT c);  // number of columns, c > 0

  ~SubMatrix<Real>() {}

  /// This type of constructor is needed for Range() to work [in Matrix base
  /// class]. Cannot make it explicit.
  SubMatrix<Real> (const SubMatrix &other) :
      MatrixBase<Real> (other.data_, other.num_cols_, other.num_rows_,
          other.stride_) {}

 private:
  /// Disallow assignment. If needed can add.
  SubMatrix<Real> &operator = (const SubMatrix<Real> &other);
};
/// @} End of "addtogroup matrix_funcs_io".

/// \addtogroup matrix_funcs_scalar
/// @{

// Some declarations.  These are traces of products.

/// Returns trace of matrix.
template <class Real>
double TraceMat(const MatrixBase<Real> &A) { return A.Trace(); }


/// Returns tr(A B C)
template <class Real>
Real TraceMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                      const MatrixBase<Real> &B, MatrixTransposeType transB,
                      const MatrixBase<Real> &C, MatrixTransposeType transC);

/// Returns tr(A B C D)
template <class Real>
Real TraceMatMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                         const MatrixBase<Real> &B, MatrixTransposeType transB,
                         const MatrixBase<Real> &C, MatrixTransposeType transC,
                         const MatrixBase<Real> &D, MatrixTransposeType transD);

/// @} end "addtogroup matrix_funcs_scalar"

// The following two template specializations don't need to be separately
// documented.
template <>
double TraceMatMat(const MatrixBase<double> &A, const MatrixBase<double> &B,
                   MatrixTransposeType trans);

template <>
float TraceMatMat(const MatrixBase<float> &A, const MatrixBase<float> &B,
                    MatrixTransposeType trans);


/// \addtogroup matrix_funcs_misc
/// @{


/// Function to ensure that SVD is sorted.
template<class Real> void SortSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                                     MatrixBase<Real>* Vt = NULL);

/// Creates the eigenvalue matrix D that is part of the decomposition used Matrix::Eig.
/// D will be block-diagonal with blocks of size 1 (for real eigenvalues) or 2x2
/// for complex pairs.  If a complex pair is lambda +- i*mu, D will have a corresponding
/// 2x2 block [lambda, mu; -mu, lambda].
/// This function will throw if any complex eigenvalues are not in complex conjugate
/// pairs (or the members of such pairs are not consecutively numbered).
template<class Real>
void CreateEigenvalueMatrix(const VectorBase<Real> &real, const VectorBase<Real> &imag,
                            MatrixBase<Real> *D);

/// The following function is used in Matrix::Power, and separately tested, so we
/// declare it here mainly for the testing code to see.  It takes a complex value to
/// a power using a method that will work for noninteger powers (but will fail if the
/// complex value is real and negative).
template<class Real>
bool AttemptComplexPower(Real *x_re, Real *x_im, Real power);



/// @} end of addtogroup matrix_funcs_misc

/// \addtogroup matrix_funcs_io
/// @{
template<typename Real>
std::ostream & operator << (std::ostream & Out, const MatrixBase<Real> & M);

template<typename Real>
std::istream & operator >> (std::istream & In, MatrixBase<Real> & M);

// The Matrix read allows resizing, so we override the MatrixBase one.
template<typename Real>
std::istream & operator >> (std::istream & In, Matrix<Real> & M);

/// @} end of \addtogroup matrix_funcs_io


}  // namespace kaldi



// we need to include the implementation and some
// template specializations.
#include "matrix/kaldi-matrix-inl.h"


#endif  // KALDI_MATRIX_KALDI_MATRIX_H_

// matrix/sp-matrix.h

// Copyright 2009-2011   Ondrej Glembek;  Microsoft Corporation;  Lukas Burget;
//                       Saarland University;  Ariya Rastrow;  Yanmin Qian;
//                       Jan Silovsky

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
#ifndef KALDI_MATRIX_SP_MATRIX_H_
#define KALDI_MATRIX_SP_MATRIX_H_

#include "matrix/packed-matrix.h"

namespace kaldi {



/// \weakgroup matrix_funcs_misc
typedef enum {
  kTakeLower,
  kTakeUpper,
  kTakeMean,
  kTakeMeanAndCheck
} SpCopyType;


/// \addtogroup matrix_group
/// @{
template<typename Real> class SpMatrix;


/**
 * @brief Packed symetric matrix class
*/
template<typename Real>
class SpMatrix : public PackedMatrix<Real> {
 public:
  // so it can use our assignment operator.
  friend class std::vector<Matrix<Real> >;

  SpMatrix(): PackedMatrix<Real>() {}

  explicit SpMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero)
      : PackedMatrix<Real>(r, resize_type) {}

  SpMatrix(const SpMatrix<Real>& Orig)
      : PackedMatrix<Real>(Orig) {}

  template<class OtherReal>
  explicit SpMatrix(const SpMatrix<OtherReal>& Orig)
      : PackedMatrix<Real>(Orig) {}

#ifdef KALDI_PARANOID
  explicit SpMatrix(const MatrixBase<Real> & Orig,
                    SpCopyType copy_type = kTakeMeanAndCheck)
      : PackedMatrix<Real>(Orig.NumRows(), kUndefined) {
    CopyFromMat(Orig, copy_type);
  }
#else
  explicit SpMatrix(const MatrixBase<Real> & Orig,
                    SpCopyType copy_type = kTakeMean)
      : PackedMatrix<Real>(Orig.NumRows(), kUndefined) {
    CopyFromMat(Orig, copy_type);
  }
#endif

  ~SpMatrix() {}

  /// Shallow swap.
  void Swap(SpMatrix *other);

  inline void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero) {
    PackedMatrix<Real>::Resize(nRows, resize_type);
  }

  void CopyFromSp(const SpMatrix<Real> &other) {
    PackedMatrix<Real>::CopyFromPacked(other);
  }

  template<class OtherReal>
  void CopyFromSp(const SpMatrix<OtherReal> &other) {
    PackedMatrix<Real>::CopyFromPacked(other);
  }

#ifdef KALDI_PARANOID
  void CopyFromMat(const MatrixBase<Real> &orig,
                   SpCopyType copy_type = kTakeMeanAndCheck);
#else  // different default arg if non-paranoid mode.
  void CopyFromMat(const MatrixBase<Real> &orig,
                   SpCopyType copy_type = kTakeMean);
#endif

  inline Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    // if column is less than row, then swap these as matrix is stored
    // as upper-triangular...  only allowed for const matrix object.
    if (static_cast<UnsignedMatrixIndexT>(c) >
        static_cast<UnsignedMatrixIndexT>(r))
      std::swap(c, r);
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));  // c<=r now so don't have to check c.
    return *(this->data_ + (r*(r+1)) / 2 + c);
    // Duplicating code from PackedMatrix.h
  }

  inline Real& operator() (MatrixIndexT r, MatrixIndexT c) {
    if (static_cast<UnsignedMatrixIndexT>(c) >
        static_cast<UnsignedMatrixIndexT>(r))
      std::swap(c, r);
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(this->num_rows_));  // c<=r now so don't have to check c.
    return *(this->data_ + (r * (r + 1)) / 2 + c);
    // Duplicating code from PackedMatrix.h
  }

  using PackedMatrix<Real>::operator =;
  using PackedMatrix<Real>::Scale;

  /// matrix inverse.
  /// if inverse_needed = false, will fill matrix with garbage.
  /// (only useful if logdet wanted).
  void Invert(Real *logdet = NULL, Real *det_sign= NULL, bool inverse_needed = true);

  // Below routine does inversion in double precision,
  // even for single-precision object.
  void InvertDouble(Real *logdet = NULL, Real *det_sign = NULL) {
    SpMatrix<double> dmat(*this);
    dmat.Invert();
    (*this).CopyFromSp(dmat);
    // georg: commented this out as function returns void:
    // return *this;
  }

  /// Returns maximum ratio of singular values.
  inline Real Cond() const {
    Matrix<Real> tmp(*this);
    return tmp.Cond();
  }

  /// Takes matrix to a fraction power via Svd.
  /// Will throw exception if matrix is not positive semidefinite
  /// (to within a tolerance)
  void ApplyPow(Real exponent);

  /// This is the version of SVD that we implement for symmetric matrices.
  /// It uses the singular value decomposition algorithm to compute the
  /// eigenvalue decomposition (*this) = U * diag(s) * U^T with U orthogonal.
  /// Will throw exception if input is not positive semidefinite to within a
  /// tolerance. This is the same as SVD, for the +ve semidefinite case.
  /// (The reason we don't have generic symmetric eigenvalue decomposition,
  /// syev in Lapack, is that it's not in the "minimal" ATLAS).  We haven't
  /// yet needed the non-positive-semidefinite case.
  /// Any tolerance >= 1.0 is interpreted as "no checking", and it saves
  /// some computation this way.

  void SymPosSemiDefEig(VectorBase<Real> *s, MatrixBase<Real> *P,
                        Real tolerance = 0.001) const;


  /// Takes log of the matrix (does eigenvalue decomposition then takes
  /// log of eigenvalues and reconstructs).  Will throw of not +ve definite.
  void Log();


  // Takes exponential of the matrix (equivalent to doing eigenvalue
  // decomposition then taking exp of eigenvalues and reconstructing;
  // actually not done that way as we don't have symmetric eigenvalue
  // code).
  void Exp();

  /// Returns the maximum of the absolute values of any of the
  /// eigenvalues.
  Real MaxAbsEig() const;

  void PrintEigs(const char *name) {
    Vector<Real> s((*this).NumRows());
    Matrix<Real> P((*this).NumRows(), (*this).NumCols());
    SymPosSemiDefEig(&s, &P);
    KALDI_LOG << "PrintEigs: " << name << ": " << s;
  }

  bool IsPosDef();  // returns true if Cholesky succeeds.
  void AddSp(const Real alpha, const SpMatrix<Real> &Ma) {
    this->AddPacked(alpha, Ma);
  }

  /// Computes log determinant but only for +ve-def matrices
  /// (it uses Cholesky).
  /// If matrix is not +ve-def, it will throw an exception
  /// was LogPDDeterminant()
  Real LogPosDefDet() const;

  Real LogDet(Real *det_sign = NULL) const;

  /// rank-one update, this <-- this + alpha V V'
  template<class OtherReal>
  void AddVec2(const Real alpha, const VectorBase<OtherReal>& v);

  /// diagonal update, this <-- this + diag(v)
  template<class OtherReal>
  void AddVec(const Real alpha, const VectorBase<OtherReal>& v);
  
  /// rank-N update:
  /// if (transM == kNoTrans)
  /// (*this) = beta*(*this) + alpha * M * M^T,
  /// or  (if transM == kTrans)
  ///  (*this) = beta*(*this) + alpha * M^T * M
  void AddMat2(const Real alpha, const MatrixBase<Real> &M,
               MatrixTransposeType transM, const Real beta = 0.0);

  /// Extension of rank-N update:
  /// this <-- beta*this  +  alpha * M * A * M^T.
  /// (*this) and A are allowed to be the same.
  /// If transM == kTrans, then we do it as M^T * A * M.
  void AddMat2Sp(const Real alpha, const MatrixBase<Real> &M,
                 MatrixTransposeType transM, const SpMatrix<Real> &A,
                 const Real beta = 0.0);

  /// Extension of rank-N update:
  /// this <-- beta*this + alpha * M * diag(v) * M^T.
  /// if transM == kTrans, then
  /// this <-- beta*this + alpha * M^T * diag(v) * M.
  void AddMat2Vec(const Real alpha, const MatrixBase<Real> &M,
                  MatrixTransposeType transM, const VectorBase<Real> &v,
                  const Real beta = 0.0);


  ///  Floors this symmetric matrix to the matrix
  /// alpha * Floor, where Floor must be positive definite.
  /// It is floored in the sense that after flooring,
  ///  x^T (*this) x  >= x^T (alpha*Floor) x.
  /// This is accomplished using an Svd.  It will crash
  /// if Floor is not positive definite. // returns #floored
  int ApplyFloor(const SpMatrix<Real> &Floor, Real alpha = 1.0,
                 bool verbose = false);

  /// Floor: Given a positive semidefinite matrix, floors the eigenvalues
  /// to the specified quantity.  Positive semidefiniteness is only assumed
  /// because a function we call checks for it (to within a tolerance), and
  /// because it tends to be present in situations where doing this would
  /// make sense.  Set the tolerance to 2 to ensure it won't ever complain
  /// about non-+ve-semidefinite matrix (it will zero out negative dimensions)
  /// returns number of floored elements.
  int ApplyFloor(Real floor, BaseFloat tolerance = 0.001); 
  bool IsDiagonal(Real cutoff = 1.0e-05) const;
  bool IsUnit(Real cutoff = 1.0e-05) const;
  bool IsZero(Real cutoff = 1.0e-05) const;

  /// sqrt of sum of square elements.
  Real FrobeniusNorm() const;

  /// Returns true if ((*this)-other).FrobeniusNorm() <=
  ///   tol*(*this).FrobeniusNorma()
  bool ApproxEqual(const SpMatrix<Real> &other, float tol = 0.01) const;

  // LimitCond:
  // Limits the condition of symmetric positive semidefinite matrix to
  // a specified value
  // by flooring all eigenvalues to a positive number which is some multiple
  // of the largest one (or zero if there are no positive eigenvalues).
  // Takes the condition number we are willing to accept, and floors
  // eigenvalues to the largest eigenvalue divided by this.
  //  Returns #eigs floored or already equal to the floor.  This will equal
  // the dimension if the input is negative semidefinite.
  // Throws exception if input is now positive definite.// returns #floored.
  MatrixIndexT LimitCond(Real maxCond = 1.0e+5, bool invert = false);

  // as LimitCond but all done in double precision. // returns #floored.
  MatrixIndexT LimitCondDouble(Real maxCond = 1.0e+5, bool invert = false) {
    SpMatrix<double> dmat(*this);
    MatrixIndexT ans = dmat.LimitCond(maxCond, invert);
    (*this).CopyFromSp(dmat);
    return ans;
  }
 private:
};

/// @} end of "addtogroup matrix_group"

/// \addtogroup matrix_funcs_scalar
/// @{


/// Returns tr(A B).
float TraceSpSp(const SpMatrix<float> &A, const SpMatrix<float> &B);
double TraceSpSp(const SpMatrix<double> &A, const SpMatrix<double> &B);


/// Returns tr(A B).
template<typename Real, typename OtherReal>
Real TraceSpSp(const SpMatrix<Real> &A, const SpMatrix<OtherReal> &B);



// TraceSpSpLower is the same as Trace(A B) except the lower-diagonal elements
// are counted only once not twice as they should be.  It is useful in certain
// optimizations.
template<typename Real>
Real TraceSpSpLower(const SpMatrix<Real> &A, const SpMatrix<Real> &B);


/// Returns tr(A B).
/// No option to transpose B because would make no difference.
template<typename Real>
Real TraceSpMat(const SpMatrix<Real> &A, const MatrixBase<Real> &B);

/// Returns tr(A B C)
/// (A and C may be transposed as specified by transA and transC).
template<typename Real>
Real TraceMatSpMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                   const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                   MatrixTransposeType transC);

/// Returns tr (A B C D)
/// (A and C may be transposed as specified by transA and transB).
template<typename Real>
Real TraceMatSpMatSp(const MatrixBase<Real> &A, MatrixTransposeType transA,
                     const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                     MatrixTransposeType transC, const SpMatrix<Real> &D);

/** Computes v1^T * M * v2.  Not as efficient as it could be where v1 == v2
 * (but no suitable blas routines available).
 */

/// Returns \f$ v_1^T M v_2 \f$
/// Not as efficient as it could be where v1 == v2.
float VecSpVec(const VectorBase<float> &v1, const SpMatrix<float> &M,
               const VectorBase<float> &v2);


/// Returns \f$ v_1^T M v_2 \f$
/// Not as efficient as it could be where v1 == v2.
double VecSpVec(const VectorBase<double> &v1, const SpMatrix<double> &M,
                const VectorBase<double> &v2);


/// @} \addtogroup matrix_funcs_scalar

/// \addtogroup matrix_funcs_misc
/// @{

/// Maximizes the auxiliary function
/// \f[    Q(x) = x.g - 0.5 x^T H x     \f]
/// using a numerically stable method. Like a numerically stable version of
/// \f$  x := Q^{-1} g.    \f$
/// Assumes H positive semidefinite.
template<class Real>
Real SolveQuadraticProblem(const SpMatrix<Real> &H,
                           const VectorBase<Real> &g,
                           VectorBase<Real> *x, Real K = 1.0E4,
                           Real eps = 1.0E-40,
                           const char *debug_str = "unknown",
                           bool optimizeDelta = true);

/// Maximizes the auxiliary function :
/// \f[   Q(x) = tr(M^T P Y) - 0.5 tr(P M Q M^T)        \f]
/// Like a numerically stable version of  \f$  M := Y Q^{-1}   \f$.
/// Assumes Q and P positive semidefinite, and matrix dimensions match
/// enough to make expressions meaningful.
template<class Real>
Real SolveQuadraticMatrixProblem(const SpMatrix<Real> &Q,
                                 const MatrixBase<Real> &Y,
                                 const SpMatrix<Real> &P,
                                 MatrixBase<Real> *M, Real K = 1.0E4,
                                 Real eps = 1.0E-40,
                                 const char *debug_str = "unknown",
                                 bool optimizeDelta = true);

/// Maximizes the auxiliary function :
/// \f[   Q(M) =  tr(M^T G) -0.5 tr(P_1 M Q_1 M^T) -0.5 tr(P_2 M Q_2 M^T).   \f]
/// Encountered in matrix update with a prior. We also apply a limit on the
/// condition but it should be less frequently necessary, and can be set
/// larger. Not unit-tested!  Probably not used.  The prior was abandoned
/// in the recipe that we published.
template<class Real>
Real SolveDoubleQuadraticMatrixProblem(const MatrixBase<Real> &G,
                                       const SpMatrix<Real> &P1,
                                       const SpMatrix<Real> &P2,
                                       const SpMatrix<Real> &Q1,
                                       const SpMatrix<Real> &Q2,
                                       MatrixBase<Real> *M, Real K = 1.0E4,
                                       Real eps = 1.0E-40,
                                       const char *debug_str = "unknown");

/// @} End of "addtogroup matrix_funcs_misc"

}  // namespace kaldi


// Including the implementation (now actually just includes some
// template specializations).
#include "matrix/sp-matrix-inl.h"


#endif


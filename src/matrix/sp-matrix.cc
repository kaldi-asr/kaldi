// matrix/sp-matrix.cc

// Copyright 2009-2011  Lukas Burget;  Ondrej Glembek;  Microsoft Corporation
//                      Saarland University;   Petr Schwarz;   Yanmin Qian

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

#include <limits>

#include "matrix/sp-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-functions.h"
#include "matrix/cblas-wrappers.h"

namespace kaldi {

// ****************************************************************************
// Returns the log-determinant if +ve definite, else KALDI_ERR.
// ****************************************************************************
template<typename Real>
Real SpMatrix<Real>::LogPosDefDet() const {
  TpMatrix<Real> chol(this->NumRows());
  double det = 0.0;
  double diag;
  chol.Cholesky(*this);  // Will throw exception if not +ve definite!

  for (MatrixIndexT i = 0; i < this->NumRows(); i++) {
    diag = static_cast<double>(chol(i, i));
    det += log(diag);
  }
  return static_cast<Real>(2*det);
}


template<typename Real>
void SpMatrix<Real>::Swap(SpMatrix<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->num_rows_, other->num_rows_);
}

template<typename Real>
void SpMatrix<Real>::SymPosSemiDefEig(VectorBase<Real> *s,
                                      MatrixBase<Real> *P,
                                      Real tolerance) const {
  Eig(s, P);
  Real max = s->Max(), min = s->Min();
  KALDI_ASSERT(-min <= tolerance * max);
  s->ApplyFloor(0.0);
}

template<typename Real>
Real SpMatrix<Real>::MaxAbsEig() const {
  Vector<Real> s(this->NumRows());
  this->Eig(&s, static_cast<MatrixBase<Real>*>(NULL));
  return std::max(s.Max(), -s.Min());
}

template<typename Real>
void SpMatrix<Real>::Log() {
  KALDI_ASSERT(this->NumRows() != 0);
  Vector<Real> s(this->NumRows());
  Matrix<Real> P(this->NumRows(), this->NumRows());
  SymPosSemiDefEig(&s, &P);
  s.ApplyLog();  // Per-element log.
  // this <-- P * diag(s) * P^T
  this->AddMat2Vec(1.0, P, kNoTrans, s, 0.0);
}

template<typename Real>
void SpMatrix<Real>::Exp() {
  // The most natural way to do this would be to do a symmetric eigenvalue
  // decomposition, but in order to work with basic ATLAS without CLAPACK, we
  // don't have symmetric eigenvalue decomposition code.  Instead we use the
  // MatrixExponential class which expands it out as a Taylor series (with a
  // pre-scaling trick to make it reasonably fast).

  KALDI_ASSERT(this->NumRows() != 0);
  Matrix<Real> M(*this), expM(this->NumRows(), this->NumRows());
  MatrixExponential<Real> me;
  me.Compute(M, &expM);  // compute exp(M)
  this->CopyFromMat(expM);  // by default, checks it's symmetric.
}

// returns true if positive definite--uses cholesky.
template<typename Real>
bool SpMatrix<Real>::IsPosDef() const {
  MatrixIndexT D = (*this).NumRows();
  KALDI_ASSERT(D > 0);
  try {
    TpMatrix<Real> C(D);
    C.Cholesky(*this);
    for (MatrixIndexT r = 0; r < D; r++)
      if (C(r, r) == 0.0) return false;
    return true;
  }
  catch(...) {  // not positive semidefinite.
    return false;
  }
}

template<typename Real>
void SpMatrix<Real>::ApplyPow(Real power) {
  if (power == 1) return;  // can do nothing.
  MatrixIndexT D = this->NumRows();
  KALDI_ASSERT(D > 0);
  Matrix<Real> U(D, D);
  Vector<Real> l(D);
  (*this).SymPosSemiDefEig(&l, &U);

  Vector<Real> l_copy(l);
  try {
    l.ApplyPow(power * 0.5);
  }
  catch(...) {
    KALDI_ERR << "Error taking power " << (power * 0.5) << " of vector "
              << l_copy;
  }
  U.MulColsVec(l);
  (*this).AddMat2(1.0, U, kNoTrans, 0.0);
}

template<typename Real>
void SpMatrix<Real>::CopyFromMat(const MatrixBase<Real> &M,
                                 SpCopyType copy_type) {
  KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
  MatrixIndexT D = this->NumRows();

  switch (copy_type) {
    case kTakeMeanAndCheck:
      {
        Real good_sum = 0.0, bad_sum = 0.0;
        for (MatrixIndexT i = 0; i < D; i++) {
          for (MatrixIndexT j = 0; j < i; j++) {
            Real a = M(i, j), b = M(j, i), avg = 0.5*(a+b), diff = 0.5*(a-b);
            (*this)(i, j) = avg;
            good_sum += std::abs(avg);
            bad_sum += std::abs(diff);
          }
          good_sum += std::abs(M(i, i));
          (*this)(i, i) = M(i, i);
        }
        if (bad_sum > 0.01 * good_sum) {
          KALDI_ERR << "SpMatrix::Copy(), source matrix is not symmetric: "
                    << bad_sum <<  ">" << good_sum;
        }
        break;
      }
    case kTakeMean:
      {
        for (MatrixIndexT i = 0; i < D; i++) {
          for (MatrixIndexT j = 0; j < i; j++) {
            (*this)(i, j) = 0.5*(M(i, j) + M(j, i));
          }
          (*this)(i, i) = M(i, i);
        }
        break;
      }
    case kTakeLower:
      for (MatrixIndexT i = 0; i < D; i++)
        for (MatrixIndexT j = 0; j <= i; j++)
          (*this)(i, j) = M(i, j);
      break;
    case kTakeUpper:
      for (MatrixIndexT i = 0; i < D; i++)
        for (MatrixIndexT j = 0; j <= i; j++)
          (*this)(i, j) = M(j, i);
      break;
    default:
      KALDI_ASSERT("Invalid argument to SpMatrix::CopyFromMat");
  }
}

template<class Real>
Real SpMatrix<Real>::Trace() const {
  const Real *data = this->data_;
  MatrixIndexT num_rows = this->num_rows_;
  Real ans = 0.0;
  for (int32 i = 1; i <= num_rows; i++, data += i)
    ans += *data;
  return ans;
}

// diagonal update, this <-- this + diag(v)
template<class Real>
template<class OtherReal>
void  SpMatrix<Real>::AddVec(const Real alpha, const VectorBase<OtherReal> &v) {
  int32 num_rows = this->num_rows_;
  KALDI_ASSERT(num_rows == v.Dim() && num_rows > 0);
  const OtherReal *src = v.Data();
  Real *dst = this->data_;
  if (alpha == 1.0)
    for (int32 i = 1; i <= num_rows; i++, src++, dst += i)
      *dst += *src;
  else
    for (int32 i = 1; i <= num_rows; i++, src++, dst += i)
      *dst += alpha * *src;
}

// instantiate the template above.
template
void SpMatrix<float>::AddVec(const float alpha, const VectorBase<double> &v);

template
void SpMatrix<double>::AddVec(const double alpha, const VectorBase<float> &v);

template
void SpMatrix<float>::AddVec(const float alpha, const VectorBase<float> &v);

template
void SpMatrix<double>::AddVec(const double alpha, const VectorBase<double> &v);

template<>
template<>
void SpMatrix<double>::AddVec2(const double alpha, const VectorBase<double> &v);

#ifndef HAVE_ATLAS
template<>
void SpMatrix<float>::Invert(float *logdet, float *det_sign, bool need_inverse);

template<>
void SpMatrix<double>::Invert(double *logdet, double *det_sign, bool need_inverse);
#else
// in the ATLAS case, these are not implemented using a library and we back off to something else.
template<class Real>
void SpMatrix<Real>::Invert(Real *logdet, Real *det_sign, bool need_inverse) {
  Matrix<Real> M(this->NumRows(), this->NumCols());
  M.CopyFromSp(*this);
  M.Invert(logdet, det_sign, need_inverse);
  if (need_inverse)
    for (MatrixIndexT i = 0; i < this->NumRows(); i++)
      for (MatrixIndexT j = 0; j <= i; j++)
        (*this)(i, j) = M(i, j);
}
#endif


double TraceSpSp(const SpMatrix<double> &A, const SpMatrix<double> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  const double *Aptr = A.Data();
  const double *Bptr = B.Data();
  MatrixIndexT R = A.NumRows();
  MatrixIndexT RR = (R * (R + 1)) / 2;
  double all_twice = 2.0 * cblas_ddot(RR, Aptr, 1, Bptr, 1);
  // "all_twice" contains twice the vector-wise dot-product... this is
  // what we want except the diagonal elements are represented
  // twice.
  double diag_once = 0.0;
  for (MatrixIndexT row_plus_two = 2; row_plus_two <= R + 1; row_plus_two++) {
    diag_once += *Aptr * *Bptr;
    Aptr += row_plus_two;
    Bptr += row_plus_two;
  }
  return all_twice - diag_once;
}


float TraceSpSp(const SpMatrix<float> &A, const SpMatrix<float> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  const float *Aptr = A.Data();
  const float *Bptr = B.Data();
  MatrixIndexT R = A.NumRows();
  MatrixIndexT RR = (R * (R + 1)) / 2;
  float all_twice = 2.0 * cblas_sdot(RR, Aptr, 1, Bptr, 1);
  // "all_twice" contains twice the vector-wise dot-product... this is
  // what we want except the diagonal elements are represented
  // twice.
  float diag_once = 0.0;
  for (MatrixIndexT row_plus_two = 2; row_plus_two <= R + 1; row_plus_two++) {
    diag_once += *Aptr * *Bptr;
    Aptr += row_plus_two;
    Bptr += row_plus_two;
  }
  return all_twice - diag_once;
}


template<typename Real, typename OtherReal>
Real TraceSpSp(const SpMatrix<Real> &A, const SpMatrix<OtherReal> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  Real ans = 0.0;
  const Real *Aptr = A.Data();
  const OtherReal *Bptr = B.Data();
  MatrixIndexT row, col, R = A.NumRows();
  for (row = 0; row < R; row++) {
    for (col = 0; col < row; col++)
      ans += 2.0 * *(Aptr++) * *(Bptr++);
    ans += *(Aptr++) * *(Bptr++);  // Diagonal.
  }
  return ans;
}

template
float TraceSpSp<float, double>(const SpMatrix<float> &A, const SpMatrix<double> &B);

template
double TraceSpSp<double, float>(const SpMatrix<double> &A, const SpMatrix<float> &B);


template<typename Real>
Real TraceSpMat(const SpMatrix<Real> &A, const MatrixBase<Real> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols() &&
               "KALDI_ERR: TraceSpMat: arguments have mismatched dimension");
  MatrixIndexT R = A.NumRows();
  Real ans = (Real)0.0;
  const Real *Aptr = A.Data(), *Bptr = B.Data();
  MatrixIndexT bStride = B.Stride();
  for (MatrixIndexT r = 0;r < R;r++) {
    for (MatrixIndexT c = 0;c < r;c++) {
      // ans += A(r, c) * (B(r, c) + B(c, r));
      ans += *(Aptr++) * (Bptr[r*bStride + c] + Bptr[c*bStride + r]);
    }
    // ans += A(r, r) * B(r, r);
    ans += *(Aptr++) * Bptr[r*bStride + r];
  }
  return ans;
}

template
float TraceSpMat(const SpMatrix<float> &A, const MatrixBase<float> &B);

template
double TraceSpMat(const SpMatrix<double> &A, const MatrixBase<double> &B);


template<typename Real>
Real TraceMatSpMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                   const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                   MatrixTransposeType transC) {
  KALDI_ASSERT((transA == kTrans?A.NumCols():A.NumRows()) ==
               (transC == kTrans?C.NumRows():C.NumCols()) &&
               (transA == kTrans?A.NumRows():A.NumCols()) == B.NumRows() &&
               (transC == kTrans?C.NumCols():C.NumRows()) == B.NumRows() &&
               "TraceMatSpMat: arguments have wrong dimension.");
  Matrix<Real> tmp(B.NumRows(), B.NumRows());
  tmp.AddMatMat(1.0, C, transC, A, transA, 0.0);  // tmp = C * A.
  return TraceSpMat(B, tmp);
}

template
float TraceMatSpMat(const MatrixBase<float> &A, MatrixTransposeType transA,
                    const SpMatrix<float> &B, const MatrixBase<float> &C,
                    MatrixTransposeType transC);
template
double TraceMatSpMat(const MatrixBase<double> &A, MatrixTransposeType transA,
                     const SpMatrix<double> &B, const MatrixBase<double> &C,
                     MatrixTransposeType transC);

template<typename Real>
Real TraceMatSpMatSp(const MatrixBase<Real> &A, MatrixTransposeType transA,
                     const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                     MatrixTransposeType transC, const SpMatrix<Real> &D) {
  KALDI_ASSERT((transA == kTrans ?A.NumCols():A.NumRows() == D.NumCols()) &&
               (transA == kTrans ? A.NumRows():A.NumCols() == B.NumRows()) &&
               (transC == kTrans ? A.NumCols():A.NumRows() == B.NumCols()) &&
               (transC == kTrans ? A.NumRows():A.NumCols() == D.NumRows()) &&
               "KALDI_ERR: TraceMatSpMatSp: arguments have mismatched dimension.");
  // Could perhaps optimize this more depending on dimensions of quantities.
  Matrix<Real> tmpAB(transA == kTrans ? A.NumCols():A.NumRows(), B.NumCols());
  tmpAB.AddMatSp(1.0, A, transA, B, 0.0);
  Matrix<Real> tmpCD(transC == kTrans ? C.NumCols():C.NumRows(), D.NumCols());
  tmpCD.AddMatSp(1.0, C, transC, D, 0.0);
  return TraceMatMat(tmpAB, tmpCD, kNoTrans);
}

template
float TraceMatSpMatSp(const MatrixBase<float> &A, MatrixTransposeType transA,
                      const SpMatrix<float> &B, const MatrixBase<float> &C,
                      MatrixTransposeType transC, const SpMatrix<float> &D);
template
double TraceMatSpMatSp(const MatrixBase<double> &A, MatrixTransposeType transA,
                       const SpMatrix<double> &B, const MatrixBase<double> &C,
                       MatrixTransposeType transC, const SpMatrix<double> &D);


template<class Real>
bool SpMatrix<Real>::IsDiagonal(Real cutoff) const {
  MatrixIndexT R = this->NumRows();
  Real bad_sum = 0.0, good_sum = 0.0;
  for (MatrixIndexT i = 0; i < R; i++) {
    for (MatrixIndexT j = 0; j <= i; j++) {
      if (i == j)
        good_sum += std::abs((*this)(i, j));
      else
        bad_sum += std::abs((*this)(i, j));
    }
  }
  return (!(bad_sum > good_sum * cutoff));
}

template<class Real>
bool SpMatrix<Real>::IsUnit(Real cutoff) const {
  MatrixIndexT R = this->NumRows();
  Real max = 0.0;  // max error
  for (MatrixIndexT i = 0; i < R; i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      max = std::max(max, static_cast<Real>(std::abs((*this)(i, j) -
                                                     (i == j ? 1.0 : 0.0))));
  return (max <= cutoff);
}

template<class Real>
bool SpMatrix<Real>::IsTridiagonal(Real cutoff) const {
  MatrixIndexT R = this->NumRows();
  Real max_abs_2diag = 0.0, max_abs_offdiag = 0.0;
  for (MatrixIndexT i = 0; i < R; i++)
    for (MatrixIndexT j = 0; j <= i; j++) {
      if (j+1 < i)
        max_abs_offdiag = std::max(max_abs_offdiag,
                                   std::abs((*this)(i, j)));
      else
        max_abs_2diag = std::max(max_abs_2diag,
                                 std::abs((*this)(i, j)));
    }
  return (max_abs_offdiag <= cutoff * max_abs_2diag);
}

template<class Real>
bool SpMatrix<Real>::IsZero(Real cutoff) const {
  if (this->num_rows_ == 0) return true;
  return (this->Max() <= cutoff && this->Min() >= -cutoff);
}

template<class Real>
Real SpMatrix<Real>::FrobeniusNorm() const {
  Real sum = 0.0;
  MatrixIndexT R = this->NumRows();
  for (MatrixIndexT i = 0; i < R; i++) {
    for (MatrixIndexT j = 0; j < i; j++)
      sum += (*this)(i, j) * (*this)(i, j) * 2;
    sum += (*this)(i, i) * (*this)(i, i) * 2;
  }
  return sqrt(sum);
}

template<class Real>
bool SpMatrix<Real>::ApproxEqual(const SpMatrix<Real> &other, float tol) const {
  if (this->NumRows() != other.NumRows())
    KALDI_ERR << "SpMatrix::AproxEqual, size mismatch, "
              << this->NumRows() << " vs. " << other.NumRows();
  SpMatrix<Real> tmp(*this);
  tmp.AddSp(-1.0, other);
  return (tmp.FrobeniusNorm() <= tol * this->FrobeniusNorm());
}

// function Floor: A = Floor(B, alpha * C) ... see tutorial document.
template<typename Real>
int SpMatrix<Real>::ApplyFloor(const SpMatrix<Real> &C, Real alpha,
                               bool verbose, bool is_psd) {
  MatrixIndexT dim = this->NumRows();
  int nfloored = 0;
  KALDI_ASSERT(C.NumRows() == dim);
  KALDI_ASSERT(alpha > 0);
  TpMatrix<Real> L(dim);
  L.Cholesky(C);
  L.Scale(sqrt(alpha));  // equivalent to scaling C by alpha.
  TpMatrix<Real> LInv(L);
  LInv.Invert();

  SpMatrix<Real> D(dim);
  {  // D = L^{-1} * (*this) * L^{-T}
    Matrix<Real> LInvFull(LInv);
    D.AddMat2Sp(1.0, LInvFull, kNoTrans, (*this), 0.0);
  }

  Vector<Real> l(dim);
  Matrix<Real> U(dim, dim);
  if (is_psd)
    D.SymPosSemiDefEig(&l, &U);
  else
    D.Eig(&l, &U);
  // We added the "Eig" function more recently.  It's not as accurate as in the
  // symmetric positive semidefinite case, so we only use it if the user says
  // the calling matrix is not positive semidefinite.
  // [Note: we since changed it to be more accurate.]
  if (verbose) {
    KALDI_LOG << "ApplyFloor: flooring following diagonal to 1: " << l;
  }
  for (MatrixIndexT i = 0; i < l.Dim(); i++) {
    if (l(i) < 1.0) {
      nfloored++;
      l(i) = 1.0;
    }
  }
  l.ApplyPow(0.5);
  U.MulColsVec(l);
  D.AddMat2(1.0, U, kNoTrans, 0.0);
  {  // D' := U * diag(l') * U^T ... l'=floor(l, 1)
    Matrix<Real> LFull(L);
    (*this).AddMat2Sp(1.0, LFull, kNoTrans, D, 0.0);  // A := L * D' * L^T
  }
  return nfloored;
}

template<class Real>
Real SpMatrix<Real>::LogDet(Real *det_sign) const {
  Real log_det;
  SpMatrix<Real> tmp(*this);
  tmp.Invert(&log_det, det_sign, false);  // false== output not needed (saves some computation).
  return log_det;
}


template<typename Real>
int SpMatrix<Real>::ApplyFloor(Real floor, BaseFloat tolerance) {
  if (floor < 0) {
    KALDI_WARN<< "Cannot have negative floor for a positive semi-definite matrix. "
        "Making the floor zero.";
    floor = 0;
  }
  MatrixIndexT Dim = this->NumRows();
  int nfloored = 0;
  Vector<Real> s(Dim);
  Matrix<Real> P(Dim, Dim);
  (*this).SymPosSemiDefEig(&s, &P, tolerance);
  for (MatrixIndexT i = 0; i < Dim; i++) {
    if (s(i) < floor) nfloored++;
    s(i) = sqrt(std::max(s(i), floor));
  }
  P.MulColsVec(s);
  (*this).AddMat2(1.0, P, kNoTrans);  // (*this) = P*P^T = P*floor(s)*P^T
  return nfloored;
}

template<typename Real>
MatrixIndexT SpMatrix<Real>::LimitCond(Real maxCond, bool invert) {  // e.g. maxCond = 1.0e+05.
  MatrixIndexT Dim = this->NumRows();
  Vector<Real> s(Dim);
  Matrix<Real> P(Dim, Dim);
  (*this).SymPosSemiDefEig(&s, &P);
  KALDI_ASSERT(maxCond > 1);
  Real floor = s.Max() / maxCond;
  if (floor < 0) floor = 0;
  if (floor < 1.0e-40) {
    KALDI_WARN << "LimitCond: limiting " << floor << " to 1.0e-40";
    floor = 1.0e-40;
  }
  MatrixIndexT nfloored = 0;
  for (MatrixIndexT i = 0; i < Dim; i++) {
    if (s(i) <= floor) nfloored++;
    if (invert)
      s(i) = 1.0 / sqrt(std::max(s(i), floor));
    else
      s(i) = sqrt(std::max(s(i), floor));
  }
  P.MulColsVec(s);
  (*this).AddMat2(1.0, P, kNoTrans);  // (*this) = P*P^T.  ... (*this) = P * floor(s) * P^T  ... if P was original P.
  return nfloored;
}

template<> double SolveQuadraticProblem(const SpMatrix<double> &H,
                                        const VectorBase<double> &g,
                                        VectorBase<double> *x, double K,
                                        double eps, const char *debug_str,
                                        bool optimizeDelta) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  KALDI_ASSERT(K>10 && eps<1.0e-10);
  MatrixIndexT dim = x->Dim();
  if (H.IsZero(0.0)) {
    KALDI_WARN << "Zero quadratic term in quadratic vector problem for "
               << debug_str << ": leaving it unchanged.";
    return 0.0;
  }
  Vector<double> gbar(g);
  if (optimizeDelta) gbar.AddSpVec(-1.0, H, *x, 1.0);  // gbar = g - H x
  Matrix<double> U(dim, dim);
  Vector<double> l(dim);
  H.SymPosSemiDefEig(&l, &U);  // does svd H = U L V^T and checks that H == U L U^T to within a tolerance.
  // floor l.
  double f = std::max(eps, l.Max() / K);
  MatrixIndexT nfloored = 0;
  for (MatrixIndexT i = 0; i < dim; i++) {  // floor l.
    if (l(i) < f) {
      nfloored++;
      l(i) = f;
    }
  }
  if (nfloored != 0) {
    KALDI_LOG << "Solving quadratic problem for " << debug_str << ": floored " << nfloored<< " eigenvalues. ";
  }
  Vector<double> tmp(dim);
  tmp.AddMatVec(1.0, U, kTrans, gbar, 0.0);  // tmp = U^T \bar{g}
  tmp.DivElements(l);  // divide each element of tmp by l: tmp = \tilde{L}^{-1} U^T \bar{g}
  Vector<double> delta(dim);
  delta.AddMatVec(1.0, U, kNoTrans, tmp, 0.0);  // delta = U tmp = U \tilde{L}^{-1} U^T \bar{g}
  Vector<double> &xhat(tmp);
  xhat.CopyFromVec(delta);
  if (optimizeDelta) xhat.AddVec(1.0, *x);  // xhat = x + delta.
  double auxf_before = VecVec(g, *x) - 0.5 * VecSpVec(*x, H, *x),
         auxf_after = VecVec(g, xhat) - 0.5 * VecSpVec(xhat, H, xhat);
  if (auxf_after < auxf_before) {  // Reject change.
    if (auxf_after < auxf_before - 1.0e-10)
      KALDI_WARN << "Optimizing vector auxiliary function for "
                 <<debug_str<< ": auxf decreased " << auxf_before
                 << " to "  << auxf_after <<  ", change is "
                 << (auxf_after-auxf_before);
    return 0.0;
  } else {
    x->CopyFromVec(xhat);
    return auxf_after - auxf_before;
  }
}

template<> float SolveQuadraticProblem(const SpMatrix<float> &H,
                                       const VectorBase<float> &g,
                                       VectorBase<float> *x, float K,
                                       float eps, const char *debug_str,
                                       bool optimizeDelta) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  SpMatrix<double> Hd(H);
  Vector<double> gd(g);
  Vector<double> xd(*x);
  float ans = static_cast<float>(SolveQuadraticProblem(Hd, gd, &xd,
      static_cast<double>(K), static_cast<double>(eps), debug_str,
      optimizeDelta));
  x->CopyFromVec(xd);
  return ans;
}

// Maximizes the auxiliary function   Q(x) = tr(M^T SigmaInv Y) - 0.5 tr(SigmaInv M Q M^T).
// Like a numerically stable version of   M := Y Q^{-1}.
template<>
double
SolveQuadraticMatrixProblem(const SpMatrix<double> &Q,
                            const MatrixBase<double> &Y,
                            const SpMatrix<double> &SigmaInv,
                            MatrixBase<double> *M, double K, double eps,
                            const char *debug_str, bool optimizeDelta) {
  KALDI_ASSERT(Q.NumRows() == M->NumCols() && SigmaInv.NumRows() == M->NumRows() && Y.NumRows() == M->NumRows()
      && Y.NumCols() == M->NumCols() && M->NumCols() != 0);
  KALDI_ASSERT(K>10 && eps<1.0e-10);
  MatrixIndexT rows = M->NumRows(), cols = M->NumCols();
  if (Q.IsZero(0.0)) {
    KALDI_WARN << "Zero quadratic term in quadratic matrix problem for "
               << debug_str << ": leaving it unchanged.";
    return 0.0;
  }
  Matrix<double> Ybar(Y);
  if (optimizeDelta) {
    Matrix<double> Qfull(Q);
    Ybar.AddMatMat(-1.0, *M, kNoTrans, Qfull, kNoTrans, 1.0);
  } // Ybar = Y - M Q.
  Matrix<double> U(cols, cols);
  Vector<double> l(cols);
  Q.SymPosSemiDefEig(&l, &U);  // does svd Q = U L V^T and checks that Q == U L U^T to within a tolerance.
  // floor l.
  double f = std::max(eps, l.Max() / K);
  MatrixIndexT nfloored = 0;
  for (MatrixIndexT i = 0;i < cols;i++) {  // floor l.
    if (l(i) < f) { nfloored++; l(i) = f; }
  }
  if (nfloored != 0)
    KALDI_LOG << "Solving matrix problem for " << debug_str
              << ": floored " << nfloored << " eigenvalues. ";
  Matrix<double> tmpDelta(rows, cols);
  tmpDelta.AddMatMat(1.0, Ybar, kNoTrans, U, kNoTrans, 0.0);  // tmpDelta = Ybar * U.
  l.InvertElements(); KALDI_ASSERT(1.0/l.Max() != 0);  // check not infinite.  eps should take care of this.
  tmpDelta.MulColsVec(l);  // tmpDelta = Ybar * U * \tilde{L}^{-1}

  Matrix<double> Delta(rows, cols);
  Delta.AddMatMat(1.0, tmpDelta, kNoTrans, U, kTrans, 0.0);  // Delta = Ybar * U * \tilde{L}^{-1} * U^T

  double auxf_before, auxf_after;
  SpMatrix<double> MQM(rows);
  Matrix<double> &SigmaInvY(tmpDelta);
  { Matrix<double> SigmaInvFull(SigmaInv);  SigmaInvY.AddMatMat(1.0, SigmaInvFull, kNoTrans, Y, kNoTrans, 0.0); }
  {  // get auxf_before.      Q(x) = tr(M^T SigmaInv Y) - 0.5 tr(SigmaInv M Q M^T).
    MQM.AddMat2Sp(1.0, *M, kNoTrans, Q, 0.0);
    auxf_before = TraceMatMat(*M, SigmaInvY, kaldi::kTrans) - 0.5*TraceSpSp(SigmaInv, MQM);
  }

  Matrix<double> Mhat(Delta); if (optimizeDelta) Mhat.AddMat(1.0, *M);  // Mhat = Delta + M.

  {  // get auxf_after.
    MQM.AddMat2Sp(1.0, Mhat, kNoTrans, Q, 0.0);
    auxf_after = TraceMatMat(Mhat, SigmaInvY, kaldi::kTrans) - 0.5*TraceSpSp(SigmaInv, MQM);
  }

  if (auxf_after < auxf_before) {
    if (auxf_after < auxf_before - 1.0e-10)
      KALDI_WARN << "Optimizing matrix auxiliary function for "
                 << debug_str << ", auxf decreased "
                 << auxf_before << " to " << auxf_after << ", change is "
                 << (auxf_after-auxf_before);
    return 0.0;
  } else {
    M->CopyFromMat(Mhat);
    return auxf_after - auxf_before;
  }
}



template<>
float SolveQuadraticMatrixProblem(const SpMatrix<float> &Q,
                                  const MatrixBase<float> &Y,
                                  const SpMatrix<float> &SigmaInv,
                                  MatrixBase<float> *M, float K, float eps,
                                  const char *debug_str, bool optimizeDelta) {
  KALDI_ASSERT(Q.NumRows() == M->NumCols() && SigmaInv.NumRows() == M->NumRows()
               && Y.NumRows() == M->NumRows() && Y.NumCols() == M->NumCols() &&
               M->NumCols() != 0);

  SpMatrix<double> Qd(Q), SigmaInvd(SigmaInv);
  Matrix<double> Yd(Y), Md(*M);
  float ans = SolveQuadraticMatrixProblem(Qd, Yd, SigmaInvd, &Md,
                                          static_cast<double>(K),
                                          static_cast<double>(eps), debug_str,
                                          optimizeDelta);
  M->CopyFromMat(Md);
  return ans;
}


template<>
double SolveDoubleQuadraticMatrixProblem(const MatrixBase<double> &G,
                                         const SpMatrix<double> &P1,
                                         const SpMatrix<double> &P2,
                                         const SpMatrix<double> &Q1,
                                         const SpMatrix<double> &Q2,
                                         MatrixBase<double> *M, double K,
                                         double eps, const char *debug_str) {
  KALDI_ASSERT(Q1.NumRows() == M->NumCols() && P1.NumRows() == M->NumRows() &&
               G.NumRows() == M->NumRows() && G.NumCols() == M->NumCols() &&
               M->NumCols() != 0 && Q2.NumRows() == M->NumCols() &&
               P2.NumRows() == M->NumRows());
  MatrixIndexT rows = M->NumRows(), cols = M->NumCols();
  // The following check should not fail as we stipulate P1, P2 and one of Q1
  // or Q2 must be +ve def and other Q1 or Q2 must be +ve semidef.
  TpMatrix<double> LInv(rows);
  LInv.Cholesky(P1);
  LInv.Invert();  // Will throw exception if fails.
  SpMatrix<double> S(rows);
  Matrix<double> LInvFull(LInv);
  S.AddMat2Sp(1.0, LInvFull, kNoTrans, P2, 0.0);  // S := L^{-1} P_2 L^{-T}
  Matrix<double> U(rows, rows);
  Vector<double> d(rows);
  S.SymPosSemiDefEig(&d, &U);
  Matrix<double> T(rows, rows);
  T.AddMatMat(1.0, U, kTrans, LInvFull, kNoTrans, 0.0);  // T := U^T * L^{-1}

#ifdef KALDI_PARANOID  // checking mainly for errors in the code or math.
  {
    SpMatrix<double> P1Trans(rows);
    P1Trans.AddMat2Sp(1.0, T, kNoTrans, P1, 0.0);
    KALDI_ASSERT(P1Trans.IsUnit(0.01));
  }
  {
    SpMatrix<double> P2Trans(rows);
    P2Trans.AddMat2Sp(1.0, T, kNoTrans, P2, 0.0);
    KALDI_ASSERT(P2Trans.IsDiagonal(0.01));
  }
#endif

  Matrix<double> TInv(T);
  TInv.Invert();
  Matrix<double> Gdash(rows, cols);
  Gdash.AddMatMat(1.0, T, kNoTrans, G, kNoTrans, 0.0);  // G' = T G
  Matrix<double> MdashOld(rows, cols);
  MdashOld.AddMatMat(1.0, TInv, kTrans, *M, kNoTrans, 0.0);  // M' = T^{-T} M
  Matrix<double> MdashNew(MdashOld);
  double objf_impr = 0.0;
  for (MatrixIndexT n = 0; n < rows; n++) {
    SpMatrix<double> Qsum(Q1);
    Qsum.AddSp(d(n), Q2);
    SubVector<double> mdash_n = MdashNew.Row(n);
    SubVector<double> gdash_n = Gdash.Row(n);

    Matrix<double> QsumInv(Qsum);
    try {
      QsumInv.Invert();
      double old_objf = VecVec(mdash_n, gdash_n)
          - 0.5 * VecSpVec(mdash_n, Qsum, mdash_n);
      mdash_n.AddMatVec(1.0, QsumInv, kNoTrans, gdash_n, 0.0); // m'_n := g'_n * (Q_1 + d_n Q_2)^{-1}
      double new_objf = VecVec(mdash_n, gdash_n)
          - 0.5 * VecSpVec(mdash_n, Qsum, mdash_n);
      if (new_objf < old_objf) {
        if (new_objf < old_objf - 1.0e-05) {
          KALDI_WARN << "In double quadratic matrix problem: objective "
              "function decreasing during optimization of " << debug_str
              << ", " << old_objf << "->" << new_objf << ", change is "
              << (new_objf - old_objf);
          KALDI_ERR << "Auxiliary function decreasing."; // Will be caught.
        } else {  // Reset to old value, didn't improve (very close to optimum).
          objf_impr += new_objf - old_objf;
          MdashNew.Row(n).CopyFromVec(MdashOld.Row(n));
        }
      }
    }
    catch (...) {
      KALDI_WARN << "Matrix inversion or optimization failed during double "
          "quadratic problem, solving for" << debug_str
          << ": trying more stable approach.";
      objf_impr += SolveQuadraticProblem(Qsum, gdash_n, &mdash_n, K, eps,
          debug_str);
    }
  }
  M->AddMatMat(1.0, T, kTrans, MdashNew, kNoTrans, 0.0); // M := T^T M'.
  return objf_impr;
}


template<>
float SolveDoubleQuadraticMatrixProblem(const MatrixBase<float> &G,
                                        const SpMatrix<float> &P1,
                                        const SpMatrix<float> &P2,
                                        const SpMatrix<float> &Q1,
                                        const SpMatrix<float> &Q2,
                                        MatrixBase<float> *M, float K, float eps,
                                        const char *debug_str) {
  KALDI_ASSERT(Q1.NumRows() == M->NumCols() && P1.NumRows() == M->NumRows() &&
               G.NumRows() == M->NumRows() && G.NumCols() == M->NumCols() &&
               M->NumCols() != 0 && Q2.NumRows() == M->NumCols() &&
               P2.NumRows() == M->NumRows());

  SpMatrix<double> P1d(P1), Q1d(Q1), P2d(P2), Q2d(Q2);
  Matrix<double> Md(*M), Gd(G);
  float ans = SolveDoubleQuadraticMatrixProblem(Gd, P1d, P2d, Q1d, Q2d, &Md,
                                                static_cast<double>(K),
                                                static_cast<double>(eps),
                                                debug_str);
  M->CopyFromMat(Md);
  return ans;
}

// rank-one update, this <-- this + alpha V V'
template<>
template<>
void SpMatrix<float>::AddVec2(const float alpha, const VectorBase<float> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
  cblas_sspr(CblasRowMajor, CblasLower, v.Dim(), alpha, v.Data(), 1,
             this->data_);
}

// rank-one update, this <-- this + alpha V V'
template<>
template<>
void SpMatrix<double>::AddVec2(const double alpha, const VectorBase<double> &v) {
  KALDI_ASSERT(v.Dim() == num_rows_);
  cblas_dspr(CblasRowMajor, CblasLower, v.Dim(), alpha, v.Data(), 1, data_);
}


template<class Real>
template<class OtherReal>
void SpMatrix<Real>::AddVec2(const Real alpha, const VectorBase<OtherReal> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
  Real *data = this->data_;
  const OtherReal *v_data = v.Data();
  MatrixIndexT nr = this->num_rows_;
  for (MatrixIndexT i = 0; i < nr; i++)
    for (MatrixIndexT j = 0; j <= i; j++, data++)
      *data += alpha * v_data[i] * v_data[j];
}

// instantiate the template above.
template
void SpMatrix<float>::AddVec2(const float alpha, const VectorBase<double> &v);
template
void SpMatrix<double>::AddVec2(const double alpha, const VectorBase<float> &v);


double VecSpVec(const VectorBase<double> &v1, const SpMatrix<double> &M, const VectorBase<double> &v2) {
  MatrixIndexT D = M.NumRows();
  KALDI_ASSERT(v1.Dim() == D && v1.Dim() == v2.Dim());
  Vector<double> tmp_vec(D);
  cblas_dspmv(CblasRowMajor, CblasLower, D, 1.0, M.Data(), v1.Data(), 1, 0.0, tmp_vec.Data(), 1);
  return VecVec(tmp_vec, v2);
}

float VecSpVec(const VectorBase<float> &v1, const SpMatrix<float> &M, const VectorBase<float> &v2) {
  MatrixIndexT D = M.NumRows();
  KALDI_ASSERT(v1.Dim() == D && v1.Dim() == v2.Dim());
  Vector<float> tmp_vec(D);
  cblas_sspmv(CblasRowMajor, CblasLower, D, 1.0, M.Data(), v1.Data(), 1, 0.0, tmp_vec.Data(), 1);
  return VecVec(tmp_vec, v2);
}


// TODO: this is not very efficiently implemented, we could do it using
// lower-level functions.
template<>
void SpMatrix<float>::AddMat2Sp(const float alpha, const MatrixBase<float> &M, MatrixTransposeType transM, const SpMatrix<float> &A, const float beta) {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows() && M.NumCols() == A.NumRows())
         || (transM ==    kTrans && this->NumRows() == M.NumCols() && M.NumRows() == A.NumRows()));

  Vector<float> tmp_vec(A.NumRows());
  SpMatrix<float> tmp_A;
  const float *p_A_data = A.Data();
  float *p_row_data = this->Data();

  if (this->Data() <= A.Data() + A.SizeInBytes() &&
      this->Data() + this->SizeInBytes() >= A.Data()) {
    // Matrices A and *this overlap. Make copy of A
    tmp_A.Resize(A.NumRows());
    tmp_A.CopyFromSp(A);
    p_A_data = tmp_A.Data();
  }
  for (MatrixIndexT r = 0; r < this->NumRows(); r++, p_row_data += r) {
    SubVector<float> out_row_vec(p_row_data, r+1);

    if (transM == kNoTrans) {
      cblas_sspmv(CblasRowMajor, CblasLower, A.NumRows(), 1.0, p_A_data, M.RowData(r), 1, 0.0, tmp_vec.Data(), 1);
      out_row_vec.AddMatVec(alpha, SubMatrix<float>(M, 0, r+1, 0, M.NumCols()), transM, tmp_vec, beta);
    } else {
      cblas_sspmv(CblasRowMajor, CblasLower, A.NumRows(), 1.0, p_A_data, M.Data() + r, M.Stride(), 0.0, tmp_vec.Data(), 1);
      out_row_vec.AddMatVec(alpha, SubMatrix<float>(M, 0, M.NumRows() , 0, r+1), transM, tmp_vec, beta);
    }
  }
}


template<>
void SpMatrix<double>::AddMat2Sp(const double alpha, const MatrixBase<double> &M, MatrixTransposeType transM, const SpMatrix<double> &A, const double beta) {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows() && M.NumCols() == A.NumRows())
         || (transM ==    kTrans && this->NumRows() == M.NumCols() && M.NumRows() == A.NumRows()));

  Vector<double> tmp_vec(A.NumRows());
  SpMatrix<double> tmp_A;
  const double *p_A_data = A.Data();
  double *p_row_data = this->Data();

  if (this->Data() <= A.Data() + A.SizeInBytes() && this->Data() + this->SizeInBytes() >= A.Data()) {
    // Matrices A and *this overlap. Make copy of A
    tmp_A.Resize(A.NumRows());
    tmp_A.CopyFromSp(A);
    p_A_data = tmp_A.Data();
  }
  for (MatrixIndexT r = 0; r < this->NumRows(); r++, p_row_data += r) {
    SubVector<double> out_row_vec(p_row_data, r+1);

    if (transM == kNoTrans) {
      cblas_dspmv(CblasRowMajor, CblasLower, A.NumRows(), 1.0, p_A_data, M.RowData(r), 1,           0.0, tmp_vec.Data(), 1);
      out_row_vec.AddMatVec(alpha, SubMatrix<double>(M, 0, r+1 , 0, M.NumCols()), transM, tmp_vec, beta);
    } else {
      cblas_dspmv(CblasRowMajor, CblasLower, A.NumRows(), 1.0, p_A_data, M.Data() + r, M.Stride(), 0.0, tmp_vec.Data(), 1);
      out_row_vec.AddMatVec(alpha, SubMatrix<double>(M, 0, M.NumRows() , 0, r+1), transM, tmp_vec, beta);
    }
  }
}

template<>
void SpMatrix<float>::AddMat2Vec(const float alpha,
                                 const MatrixBase<float> &M,
                                 MatrixTransposeType transM,
                                 const VectorBase<float> &v,
                                 const float beta) {
  this->Scale(beta);
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows() &&
                M.NumCols() == v.Dim()) ||
               (transM == kTrans && this->NumRows() == M.NumCols() &&
                M.NumRows() == v.Dim()));

  if (transM == kNoTrans) {
    const float *Mdata = M.Data(), *vdata = v.Data();
    float *data = this->data_;
    MatrixIndexT dim = this->NumRows(), mcols = M.NumCols(),
        mstride = M.Stride();
    for (MatrixIndexT col = 0; col < mcols; col++, vdata++, Mdata += 1)
      cblas_sspr(CblasRowMajor, CblasLower, dim, *vdata*alpha, Mdata,
                 mstride, data);
  } else {
    const float *Mdata = M.Data(), *vdata = v.Data();
    float *data = this->data_;
    MatrixIndexT dim = this->NumRows(), mrows = M.NumRows(),
        mstride = M.Stride();
    for (MatrixIndexT row = 0; row < mrows; row++, vdata++, Mdata += mstride)
      cblas_sspr(CblasRowMajor, CblasLower, dim, *vdata*alpha,
                 Mdata, 1, data);
  }
}


template<>
void SpMatrix<double>::AddMat2Vec(const double alpha,
                                  const MatrixBase<double> &M,
                                  MatrixTransposeType transM,
                                  const VectorBase<double> &v,
                                  const double beta) {
  this->Scale(beta);
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows() &&
                M.NumCols() == v.Dim()) ||
               (transM == kTrans && this->NumRows() == M.NumCols() &&
                M.NumRows() == v.Dim()));

  if (transM == kNoTrans) {
    const double *Mdata = M.Data(), *vdata = v.Data();
    double *data = this->data_;
    MatrixIndexT dim = this->NumRows(), mcols = M.NumCols(),
        mstride = M.Stride();
    for (MatrixIndexT col = 0; col < mcols; col++, vdata++, Mdata += 1)
      cblas_dspr(CblasRowMajor, CblasLower, dim, *vdata*alpha, Mdata,
                 mstride, data);
  } else {
    const double *Mdata = M.Data(), *vdata = v.Data();
    double *data = this->data_;
    MatrixIndexT dim = this->NumRows(), mrows = M.NumRows(),
        mstride = M.Stride();
    for (MatrixIndexT row = 0; row < mrows; row++, vdata++, Mdata += mstride)
      cblas_dspr(CblasRowMajor, CblasLower, dim, *vdata*alpha,
                 Mdata, 1, data);
  }
}

template<>
void SpMatrix<float>::AddMat2(const float alpha, const MatrixBase<float> &M, MatrixTransposeType transM, const float beta) {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows())
               || (transM == kTrans && this->NumRows() == M.NumCols()));
  Vector<float> tmp_vec(transM == kTrans ? M.NumRows() : 0);
  SpMatrix<float> tmp_A;
  float *p_row_data = this->Data();

  for (MatrixIndexT r = 0; r < this->NumRows(); r++, p_row_data += r) {
    SubVector<float> out_row_vec(p_row_data, r+1);

    if (transM == kNoTrans) {
      // tmp_vec =
      out_row_vec.AddMatVec(alpha, SubMatrix<float>(M, 0, r+1 , 0, M.NumCols()), transM, M.Row(r), beta);  // M.Row(r) is a SubVector...
    } else {
      tmp_vec.CopyColFromMat(M, r);  // copy the r'th row of M.  A few extra cycles but may help for caching reasons ( + simpler)
      out_row_vec.AddMatVec(alpha, SubMatrix<float>(M, 0, M.NumRows() , 0, r+1), transM, tmp_vec, beta);
    }
  }
}

template<>
void SpMatrix<double>::AddMat2(const double alpha, const MatrixBase<double> &M, MatrixTransposeType transM, const double beta)  {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows())
         || (transM == kTrans && this->NumRows() == M.NumCols()));

  Vector<double> tmp_vec(transM == kTrans ? M.NumRows() : 0);
  SpMatrix<double> tmp_A;
  double *p_row_data = this->Data();

  for (MatrixIndexT r = 0; r < this->NumRows(); r++, p_row_data += r) {
    SubVector<double> out_row_vec(p_row_data, r+1);

    if (transM == kNoTrans) {
      out_row_vec.AddMatVec(alpha, SubMatrix<double>(M, 0, r+1 , 0, M.NumCols()), transM, M.Row(r), beta);  // M.Row(r) is a SubVector...
    } else {
      tmp_vec.CopyColFromMat(M, r);  // copy the r'th row of M.  A few extra cycles but may help for caching reasons ( + simpler)
      out_row_vec.AddMatVec(alpha, SubMatrix<double>(M, 0, M.NumRows() , 0, r+1), transM, tmp_vec, beta);
    }
  }
}


#ifndef HAVE_ATLAS
template<>
void SpMatrix<float>::Invert(float *logdet, float *det_sign, bool need_inverse) {
  // these are CLAPACK types
  KaldiBlasInt   result;
  KaldiBlasInt   rows = static_cast<int>(num_rows_);
  KaldiBlasInt*  p_ipiv = new KaldiBlasInt[rows];
  float*    p_work = new float[rows];   // workspace for the lapack function


  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  ssptrf_(const_cast<char *>("U"), &rows, data_, p_ipiv, &result);


  KALDI_ASSERT(result >= 0 && "Call to CLAPACK ssptrf_ called with wrong arguments");

  if (result > 0) {  // Singular...
    if (det_sign) *det_sign = 0;
    if (logdet) *logdet = -std::numeric_limits<float>::infinity();
    if (need_inverse) KALDI_ERR << "CLAPACK stptrf_ : factorization failed";
  } else {  // Not singular.. compute log-determinant if needed.
    if (logdet != NULL || det_sign != NULL) {
      float prod = 1.0, log_prod = 0.0;
      int sign = 1;
      for (int i = 0; i < (int)num_rows_; i++) {
        if (p_ipiv[i] > 0) {  // not a 2x2 block...
          // if (p_ipiv[i] != i+1) sign *= -1;  // row swap.
          float diag = (*this)(i, i);
          prod *= diag;
        } else {  // negative: 2x2 block. [we are in first of the two].
          i++;  // skip over the first of the pair.
          // each 2x2 block...
          float diag1 = (*this)(i, i), diag2 = (*this)(i-1, i-1),
              offdiag = (*this)(i, i-1);
          float thisdet = diag1*diag2 - offdiag*offdiag;
          // thisdet == determinant of 2x2 block.
          // The following line is more complex than it looks: there are 2 offsets of
          // 1 that cancel.
          prod *= thisdet;
        }
        if (i == (int)(num_rows_-1) || fabs(prod) < 1.0e-10 || fabs(prod) > 1.0e+10) {
          if (prod < 0) { prod = -prod; sign *= -1; }
          log_prod += log(fabs(prod));
          prod = 1.0;
        }
      }
      if (logdet != NULL) *logdet = log_prod;
      if (det_sign != NULL) *det_sign = sign;
    }
  }
  if (!need_inverse) return;  // Don't need what is computed next.

  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  ssptri_(const_cast<char *>("U"), &rows, data_, p_ipiv, p_work, &result);

  KALDI_ASSERT(result >=0 &&"Call to CLAPACK ssptri_ called with wrong arguments");

  if (result != 0) {
    KALDI_ERR << "CLAPACK ssptrf_ : Matrix is singular";
  }

  delete [] p_ipiv;
  delete [] p_work;
}

template<>
void SpMatrix<double>::Invert(double *logdet, double *det_sign, bool need_inverse) {
  // these are CLAPACK types
  KaldiBlasInt   result;
  KaldiBlasInt   rows = static_cast<int>(num_rows_);
  KaldiBlasInt*  p_ipiv = new KaldiBlasInt[rows];
  double*   p_work = new double[rows];   // workspace for the lapack function


  // see e.g. http://www.netlib.org/lapack/lapack-3.1.1/html/dsptrf.f.html#DSPTRF.1
  // for documentation of the output of dsptrf.  It's quite complicated.

  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  dsptrf_(const_cast<char *>("U"), &rows, data_, p_ipiv, &result);

  KALDI_ASSERT(result >= 0 && "Call to CLAPACK dsptrf_ called with wrong arguments");

  if (result > 0) {  // Singular...
    if (det_sign) *det_sign = 0;
    if (logdet) *logdet = -std::numeric_limits<double>::infinity();
    if (need_inverse) KALDI_ERR << "CLAPACK stptrf_ : factorization failed";
  } else {  // Not singular.. compute log-determinant if needed.
    if (logdet != NULL || det_sign != NULL) {
      float prod = 1.0, log_prod = 0.0;
      int sign = 1;
      for (int i = 0; i < (int)num_rows_; i++) {
        if (p_ipiv[i] > 0) {  // not a 2x2 block...
          // if (p_ipiv[i] != i+1) sign *= -1;  // row swap.
          float diag = (*this)(i, i);
          prod *= diag;
        } else {  // negative: 2x2 block. [we are in first of the two].
          i++;  // skip over the first of the pair.
          // each 2x2 block...
          float diag1 = (*this)(i, i), diag2 = (*this)(i-1, i-1),
              offdiag = (*this)(i, i-1);
          float thisdet = diag1*diag2 - offdiag*offdiag;
          // thisdet == determinant of 2x2 block.
          // The following line is more complex than it looks: there are 2 offsets of
          // 1 that cancel.
          prod *= thisdet;
        }
        if (i == (int)(num_rows_-1) || fabs(prod) < 1.0e-10 || fabs(prod) > 1.0e+10) {
          if (prod < 0) { prod = -prod; sign *= -1; }
          log_prod += log(fabs(prod));
          prod = 1.0;
        }
      }
      if (logdet != NULL) *logdet = log_prod;
      if (det_sign != NULL) *det_sign = sign;
    }
  }
  if (!need_inverse) return;  // Don't need what is computed next.

  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  dsptri_(const_cast<char *>("U"), &rows, data_, p_ipiv, p_work, &result);

  KALDI_ASSERT(result >= 0 && "Call to CLAPACK dsptri_ called with wrong arguments");
  if (result > 0) {
    KALDI_ERR << "CLAPACK dsptrf_ : Matrix is singular";
  }
  delete [] p_ipiv;
  delete [] p_work;
}

#endif


template<>
double TraceSpSpLower(const SpMatrix<double> &A, const SpMatrix<double> &B) {
  MatrixIndexT adim = A.NumRows();
  KALDI_ASSERT(adim == B.NumRows());
  MatrixIndexT dim = (adim*(adim+1))/2;
  return cblas_ddot(dim, A.Data(), 1, B.Data(), 1);
}

template<>
float TraceSpSpLower(const SpMatrix<float> &A, const SpMatrix<float> &B) {
  MatrixIndexT adim = A.NumRows();
  KALDI_ASSERT(adim == B.NumRows());
  MatrixIndexT dim = (adim*(adim+1))/2;
  return cblas_sdot(dim, A.Data(), 1, B.Data(), 1);
}


template<class Real>
void SpMatrix<Real>::AddTp2Sp(const Real alpha, const TpMatrix<Real> &T,
                              MatrixTransposeType transM, const SpMatrix<Real> &A,
                              const Real beta) {
  Matrix<Real> Tmat(T);
  AddMat2Sp(alpha, Tmat, transM, A, beta);
}

template<class Real>
void SpMatrix<Real>::AddVecVec(const Real alpha, const VectorBase<Real> &v,
                               const VectorBase<Real> &w) {
  int32 dim = this->NumRows();
  KALDI_ASSERT(dim == v.Dim() && dim == w.Dim() && dim > 0);
  cblas_Xspr2(dim, alpha, v.Data(), 1, w.Data(), 1, this->data_);
}


template<class Real>
void SpMatrix<Real>::AddTp2(const Real alpha, const TpMatrix<Real> &T,
                            MatrixTransposeType transM, const Real beta) {
  Matrix<Real> Tmat(T);
  AddMat2(alpha, Tmat, transM, beta);
}


//Explicit instantiation of the classes
//Apparently, it seems to be necessary that the instantiation 
//happens at the end of the file. Otherwise, not all the member 
//functions will get instantiated.

template class SpMatrix<float>;
template class SpMatrix<double>;

} // namespace kaldi

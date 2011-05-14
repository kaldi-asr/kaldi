// sgmm/estimate-am-sgmm-compress.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include <algorithm>
#include <string>
#include <vector>
using std::vector;

#include "sgmm/estimate-am-sgmm-compress.h"

namespace kaldi {

bool SgmmCompressM::ComputeM() {
  double diff_sum = 0.0, abs_sum = 0.0;
  for (int32 i = 0; i < I_; i++) {
    Matrix<double> M(D_, S_);
    for (int32 g = 0; g < G_; g++)
      M.AddMat(a_(i, g), K_[g]);
    Matrix<double> diff(M);
    diff.AddMat(-1.0, M_[i]);
    diff_sum += diff.FrobeniusNorm();
    abs_sum += M.FrobeniusNorm();
    M_[i].CopyFromMat(M);
  }
  return (diff_sum <= abs_sum * 1.0e-05);  // true if unchanged.
}

double SgmmCompressM::Objf() const {
  // returns \sum_i \tr(\M_i^T \J_i) -0.5 * \tr(\Sigma_i^{-1} \M_i \Q_i \M_i^T)
  double ans = 0.0;
  for (int32 i = 0; i < I_; i++)
    ans += TraceMatMat(M_[i], J_[i], kTrans)
        -0.5 * TraceMatSpMatSp(M_[i], kNoTrans, Q_[i],
                               M_[i], kTrans, SigmaInv_[i]);

  return ans;
}

void SgmmCompressM::PcaInit() {
  SpMatrix<double> X(I_);
  for (int32 i = 0; i < I_; i++)
    for (int32 j = 0; j <= i; j++) // x(i, j) = M_i^T M_j
      X(i, j) = TraceMatMat(M_[i], M_[j], kTrans);

  Matrix<double> U(I_, I_);
  Vector<double> l(I_);
  X.SymPosSemiDefEig(&l, &U);

  KALDI_LOG << "Eigenvalues when initializing basis are "
            << l.Range(0, G_);

  a_.Resize(I_, G_);
  K_.resize(G_);
  for (int32 g = 0; g < G_; g++) {
    Matrix<double> tildeK(D_, S_);
    for (int32 i = 0; i < I_; i++)
      tildeK.AddMat(U(i, g), M_[i]);
    double ng = sqrt(TraceMatMat(tildeK, tildeK, kTrans));
    K_[g].Resize(D_, S_);
    if (ng != 0) {
      K_[g].AddMat(1.0 / ng, tildeK);
    } else {  // initialize to random matrix with unit Frobenius norm
      for (int32 m = 0; m < D_; m++)
        for (int32 n = 0; n < S_; n++)
          K_[g](m, n) = RandGauss();
      K_[g].Scale(1.0 / sqrt(TraceMatMat(K_[g], K_[g], kTrans)));
    }
    for (int32 i = 0; i < I_;i ++)
      a_(i, g) = U(i, g) * ng;
  }
  // Orthonormalize the K_'s (only has an effect on zero eigenvalues...)
  for (int32 g = 0; g < G_; g++) {
    for (int32 h = 0; h < g; h++) {
      double p = TraceMatMat(K_[g], K_[h], kTrans);
      K_[g].AddMat(-p, K_[h]);
    }
    double p = TraceMatMat(K_[g], K_[g], kTrans);
    if (p == 0.0) {
      K_[g](rand() % D_, rand() % S_) = 1.0;  // perturb.
      g--;  // orthonormalize this one again.
    } else {
      K_[g].Scale(1.0 / sqrt(p));
    }
  }
  bool ok = ComputeM();  // ok if the M's unchanged.
  if (!ok)
    KALDI_WARN << "PcaInit(): M's not reconstructed (maybe you didn't compress "
        "the M's on the previous iteration, or you used a larger dim?)";
}

SgmmCompressM::SgmmCompressM(const std::vector<Matrix<BaseFloat> > &M,
                             const std::vector<Matrix<double> > &Y,
                             const std::vector<SpMatrix<BaseFloat> > &SigmaInv,
                             const std::vector<SpMatrix<double> > &Q,
                             const Vector<double> &gamma,
                             int32 G):
    G_(G), Q_(Q), gamma_(gamma) {
  I_ = M.size();
  KALDI_ASSERT(I_ != 0);
  D_ = M[0].NumRows();
  S_ = M[0].NumCols();
  KALDI_ASSERT(G_ < I_ && G_ < S_*D_);
  KALDI_ASSERT(D_ != 0 && S_ != 0);
  KALDI_ASSERT(SigmaInv.size() == I_ && SigmaInv[0].NumRows() == D_
               && Q_.size() == I_ && Q_[0].NumRows() == S_
               && Y.size() == I_ && Y[0].NumRows() == D_
               && Y[0].NumCols() == S_);
  SigmaInv_.resize(I_);
  M_.resize(I_);
  J_.resize(I_);
  for (int32 i = 0; i < I_; i++) {
    SigmaInv_[i].Resize(D_);
    SigmaInv_[i].CopyFromSp(SigmaInv[i]);
    M_[i].Resize(D_, S_);
    M_[i].CopyFromMat(M[i]);
    J_[i].Resize(D_, S_);
    J_[i].AddSpMat(1.0, SigmaInv_[i], Y[i], kNoTrans, 0.0);
  }

  // The #iters parameters are not configurable for now.
  num_outer_iters_ = 2;
  num_cg_iters_a_ = 10;
  num_cg_iters_K_ = 30;
  epsilon_ = 1.0e-05;
  delta_ = 1.0e-04;
}

void SgmmCompressM::Compute(std::vector<Matrix<BaseFloat> > *M,
                           BaseFloat *objf_change_out,
                           BaseFloat *tot_t_out) {
  double objf_at_start = Objf(), tot_t = gamma_.Sum();
  KALDI_LOG << "SgmmCompressM: objf before PCA is " << (objf_at_start/tot_t);
  PcaInit();
  KALDI_LOG << "SgmmCompressM: objf after PCA is " << (Objf()/tot_t);
  for (int32 iter = 0; iter < num_outer_iters_; iter++) {
    // Will first update a's...
    PreconditionForA();
    ComputeA();
    KALDI_LOG << "About to precondition for K (phase 1)";
    PreconditionForKPhase1();
    ComputeK();
  }
  double objf_at_end = Objf();
  for (int32 i = 0; i < I_; i++)
    (*M)[i].CopyFromMat(M_[i]);

  KALDI_LOG << "**SgmmCompressM: Overall objf improvement per frame for M is "
            << ((objf_at_end-objf_at_start)/tot_t) << " over " << tot_t
            << " frames.";
  if (objf_change_out) *objf_change_out = (objf_at_end-objf_at_start);
  if (tot_t_out) *tot_t_out = tot_t;
}

void SgmmCompressM::PreconditionForA() {
  KALDI_LOG << "Preconditioning for A";
  // Preconditioning phase before updating the A's.
  SpMatrix<double> P(G_);
  for (int32 g = 0; g < G_; g++) {
    Matrix<double> L(D_, S_);  // L_g = 1/\gamma \sum_i \Sigma_i^{-1} \K_g \Q_i
    for (int32 i = 0; i < I_; i++) {
      Matrix<double> tmp(D_, S_);
      tmp.AddMatSp(1.0, K_[g], kNoTrans, Q_[i], 0.0);
      L.AddSpMat(1.0, SigmaInv_[i], tmp, kNoTrans, 0.0);
    }
    L.Scale(1.0 / gamma_.Sum());
    for (int32 h = 0; h <= g; h++)
      P(g, h) = TraceMatMat(L, K_[h], kTrans);
  }
  TpMatrix<double> C(G_);
  C.Cholesky(P);
  TpMatrix<double> D(C);  // D is inverse of C.
  D.Invert();
  for (int32 i = 0; i < I_; i++) {
    SubVector<double> a(a_, i);  // vector of size G_ (sometimes x in writeup).
    Vector<double> tmp(G_);
    tmp.AddTpVec(1.0, C, kTrans, a);
    a.CopyFromVec(tmp);  // write back a <-- C^T a
  }
  std::vector<Matrix<double> > Khat(G_);
  for (int32 g = 0; g < G_; g++) {
    Khat[g].Resize(D_, S_);
    for (int32 h = 0; h <= g; h++)
      Khat[g].AddMat(D(g, h), K_[h]);
  }
  for (int32 g = 0; g < G_; g++)
    K_[g].CopyFromMat(Khat[g]);

  bool ok = ComputeM();  // should not have changed.
  if (!ok) KALDI_ERR << "Error preconditioning the basis for \"a\" update.";
}


void SgmmCompressM::HessianMulA(int32 i, const VectorBase<double> &p, VectorBase<double> *q) {
  KALDI_ASSERT(p.Dim() == G_ && q->Dim() == G_);
  // for this function, c.f. eq. (17) in current techreport
  //  (q_g = [blah])
  Matrix<double> Ksum(D_, S_), tmp(D_, S_), X(D_, S_);
  for (int32 h = 0; h < G_; h++)
    Ksum.AddMat(p(h), K_[h]);
  tmp.AddMatSp(1.0, Ksum, kNoTrans, Q_[i], 0.0);  // tmp = Ksum * Q_i
  X.AddSpMat(1.0, SigmaInv_[i], tmp, kNoTrans, 0.0);  // X = \Sigma_i^{-1} * tmp
  for (int32 g = 0; g < G_; g++)
    (*q)(g) = TraceMatMat(K_[g], X, kTrans);
}

void SgmmCompressM::ComputeA() {
  double objf_start = Objf();
  KALDI_LOG << "Computing A";
  for (int32 i = 0; i < I_; i++) {
    Vector<double> b(G_);
    for (int32 g = 0; g < G_; g++)
      b(g) = TraceMatMat(K_[g], J_[i], kTrans);
    SubVector<double> x(a_, i);  // i'th row of a_{ig} coefficients.
    Vector<double> Ax(G_);  // temp variable.
    HessianMulA(i, x, &Ax);
    Vector<double> r(b);
    r.AddVec(-1.0, Ax);  // r = b - A x: initialize "residual"
    Vector<double> f(num_cg_iters_a_+1);
    f(0) = 0.5*(VecVec(r, x)  + VecVec(b, x));  // f = 0.5(x^t r + x^t b)
    Vector<double> p(r);  // initialize p = r
    double s_old = VecVec(r, r);  // s_{old} = r^T r
    for (int32 iter = 0; iter < num_cg_iters_a_; iter++) {
      Vector<double> q(G_);
      HessianMulA(i, p, &q);
      double t = VecVec(p, q);  // t = p^T q
      if (t == 0.0) break;
      double alpha = s_old / t;
      r.AddVec(-alpha, q);  // r = r - alpha q
      x.AddVec(alpha, p);  // x = x + alpha p
      f(iter+1) = f(iter) + 0.5 * s_old*s_old / t;
      double s_new = VecVec(r, r);  // s_new = r^T r
      if (sqrt(s_new) < epsilon_) break;
      p.Scale(s_new / s_old);
      p.AddVec(1.0, r);  // p = r + (s_{new}/s_{old} p).
      s_old = s_new;
    }
    // if (i < 10)
    //  KALDI_LOG << "Computing A: i = " << i << " f(iter) = " << f;
  }
  ComputeM();  // Don't check return status: won't be true.
  KALDI_LOG << "ComputeA(): objf impr per frame is "
            << ((Objf() - objf_start) / gamma_.Sum());
}

void SgmmCompressM::PreconditionForKPhase1() {
  SpMatrix<double> P(G_);
  for (int32 i = 0; i < I_; i++)
    P.AddVec2(gamma_(i), a_.Row(i));
  P.Scale(1.0 / gamma_.Sum());
  TpMatrix<double> C(G_);
  C.Cholesky(P);
  TpMatrix<double> Cinv(C);
  Cinv.Invert();
  for (int32 i = 0; i < I_; i++) {  // do a_i <-- C^{-1} a_i
    Vector<double> ahat(G_);
    ahat.AddTpVec(1.0, Cinv, kNoTrans, a_.Row(i));
    a_.Row(i).CopyFromVec(ahat);
  }
  for (int32 g = 0; g < G_; g++) {
    Matrix<double> K(D_, S_);
    for (int32 h = g; h < G_; h++)
      K.AddMat(C(h, g), K_[h]);
    // Because we only access this or later elements of K_, safe
    // to assign within this loop.
    K_[g].CopyFromMat(K);
  }
  bool ok = ComputeM();  // should not have changed.
  if (!ok) KALDI_ERR << "Error preconditioning the basis for K update.";
}


// using capitals for the names here, as they are actually vectors of matrices...
void SgmmCompressM::HessianMulK(const std::vector<Matrix<double> > &X,
                                 std::vector<Matrix<double> > *U) const {
  KALDI_ASSERT(X.size() == G_ && X[0].NumRows() == D_
               && X[0].NumCols() == S_);
  std::vector<Matrix<double> > V(I_);
  for (int32 i = 0; i < I_; i++) {
    Matrix<double> Wi(D_, S_);
    for (int32 g = 0; g < G_; g++)
      Wi.AddMat(a_(i, g), X[g]);
    V[i].Resize(D_, S_);
    V[i].AddSpMatSp(1.0, SigmaInv_[i], Wi, kNoTrans, Q_[i], 0.0);
  }
  U->resize(G_);
  for (int32 g = 0; g < G_; g++) {
    (*U)[g].Resize(D_, S_);
    for (int32 i = 0; i < I_; i++)
      (*U)[g].AddMat(a_(i, g), V[i]);
  }
}

// Multiply by M^{-1} while estimating K.
void SgmmCompressM::MulMinv(const std::vector<Matrix<double> > &R,
                            const std::vector<SpMatrix<double> > &SigmaG,
                            const std::vector<SpMatrix<double> > &QinvG,
                            std::vector<Matrix<double> > *Z) const {
  KALDI_ASSERT(R.size() == G_ && R[0].NumRows() == D_
               && R[0].NumCols() == S_ && Z != NULL);
  Z->resize(G_);
  for (int32 g = 0; g < G_; g++) {
    (*Z)[g].Resize(D_, S_);
    (*Z)[g].AddSpMatSp(1.0, SigmaG[g], R[g], kNoTrans, QinvG[g], 0.0);
  }
}

// static
double
SgmmCompressM::KInnerProduct(const std::vector<Matrix<double> > &x,
                             const std::vector<Matrix<double> > &y) {
  KALDI_ASSERT(x.size() == y.size());
  double ans = 0.0;
  for (size_t m = 0; m < x.size(); m++)
    ans += TraceMatMat(x[m], y[m], kTrans);
  return ans;
}


void SgmmCompressM::ComputeK() {
  double objf_start = Objf();
  // Optimize the basis elements K_g

  // First work out the SigmaG and QinvG quantities.
  std::vector<SpMatrix<double> > SigmaG(G_), QinvG(G_);
  for (int32 g = 0; g < G_; g++) {
    SigmaG[g].Resize(D_);
    QinvG[g].Resize(S_);
    for (int32 i = 0; i < I_; i++) {
      SigmaG[g].AddSp(a_(i, g)*a_(i, g), SigmaInv_[i]);
      QinvG[g].AddSp(a_(i, g)*a_(i, g), Q_[i]);
    }
    // Now invert (subject to flooring)..
    double smax = SigmaG[g].MaxAbsEig();
    int32 floored_s = SigmaG[g].ApplyFloor(smax*delta_);
    if (floored_s != 0)
      KALDI_LOG << "Floored " << floored_s << " eigs in SigmaInv.";
    SigmaG[g].Invert();
    double qmax = QinvG[g].MaxAbsEig();
    int32 floored_q = QinvG[g].ApplyFloor(qmax*delta_);
    if (floored_q != 0)
      KALDI_LOG << "Floored " << floored_q << " eigs in Q";
    QinvG[g].Invert();
  }

  KALDI_LOG << "Computing K";
  std::vector<Matrix<double> > B(G_);  // corresponds to the "b" term in CG..
  for (int32 g = 0; g < G_; g++) {
    B[g].Resize(D_, S_);
    for (int32 i = 0; i < I_; i++)
      B[g].AddMat(a_(i, g), J_[i]);
  }

  std::vector<Matrix<double> > &R(B);  // Just aliasing residual to B..
  std::vector<Matrix<double> > &X(K_);  // alias to K_
  std::vector<Matrix<double> > AX;
  HessianMulK(X, &AX);  // AX <-- A x.
  Vector<double> f(num_cg_iters_K_+1);
  f(0) = 0.5 * KInnerProduct(X, B);  // f_0 = 0.5*x^t b; will add other term to f(0) below.
  for (int32 g = 0; g < G_; g++) // do: r -= Ax.
    R[g].AddMat(-1.0, AX[g]);
  f(0) += 0.5 * KInnerProduct(X, R);  // f_0 += 0.5*x^t r.
  std::vector<Matrix<double> > Z(G_);
  MulMinv(R, SigmaG, QinvG, &Z);
  std::vector<Matrix<double> > P(G_);
  for (int32 g = 0; g < G_; g++) {  // p <-- r.
    P[g].Resize(D_, S_);
    P[g].CopyFromMat(Z[g]);
  }
  double s_old = KInnerProduct(Z, R);

  for (int iter = 0; iter < num_cg_iters_K_; iter++) {
    std::vector<Matrix<double> > Q;
    HessianMulK(P, &Q);  // q <-- A p
    double t = KInnerProduct(P, Q);  // t <-- p^T q
    if (t == 0.0) break;
    double alpha = s_old / t;
    for (int32 g = 0; g < G_; g++) {
      R[g].AddMat(-alpha, Q[g]);  // r <-- r - alpha q
      X[g].AddMat(alpha, P[g]);  // x <-- x + alpha p.
    }
    f(iter+1) = f(iter) + 0.5 * s_old*s_old / t;
    MulMinv(R, SigmaG, QinvG, &Z);
    double s_new = KInnerProduct(Z, R);  // s_new <-- z^T r
    if (sqrt(s_new) < epsilon_)
      break;
    for (int32 g = 0; g < G_; g++) {  // p <-- r + (s_new/s_old) p
      P[g].Scale(s_new/s_old);
      P[g].AddMat(1.0, Z[g]);
    }
    s_old = s_new;
  }
  f.Scale(1.0 / gamma_.Sum());
  KALDI_LOG << "CompressM: Optimizing K: objective function values are " << f;

  ComputeM();  // Don't check return status-- would have changed.
  KALDI_LOG << "ComputeK(): objf impr per frame is "
            << ((Objf() - objf_start) / gamma_.Sum());

}

CompressVars::CompressVars(const Vector<double> &gamma,
                           const std::vector<SpMatrix<double> > &T,
                           int32 G):
    G_(G), gamma_(gamma), T_(T) {
  KALDI_ASSERT(G > 0);
  I_ = gamma.Dim();
  KALDI_ASSERT(I_ != 0 && I_ == static_cast<int32>(T.size()));
  D_ = T[0].NumRows();

  // set up defaults (these can't be changed at the moment).
  num_outer_iters_ = 2;
  num_cg_iters_a_ = 20;
  num_cg_iters_B_ = 5;
  num_newton_iters_ = 3;
  num_backtrack_iters_ = 10;
  epsilon_ = 1.0e-07;
}

void CompressVars::InitL(const std::vector<SpMatrix<BaseFloat> > &Sigma) {
    // Compute the initial value of L.
  KALDI_ASSERT(static_cast<int32>(Sigma.size()) == I_);
  L_.resize(I_);
  Matrix<double> Cinv(C_);
  Cinv.Invert();
  for (int32 i = 0; i < I_; i++) {
    L_[i].Resize(D_);
    if (gamma_(i) == 0.0) {
      L_[i].SetZero();
    } else {
      SpMatrix<double> SigmaI(Sigma[i]); // convert to double.
      SpMatrix<double> SigmaHat(D_); // transformed variance.
      // \hat{Sigma}_i = C^{-1} Sigma_i C^{-T}
      SigmaHat.AddMat2Sp(1.0, Cinv, kNoTrans, SigmaI, 0.0); 
      Vector<double> s(D_);
      Matrix<double> P(D_, D_);
      SigmaHat.SymPosSemiDefEig(&s, &P);
      KALDI_ASSERT(s.Min() > 0); // or zero variance.
      s.ApplyLog();
      L_[i].AddMat2Vec(1.0, P, kNoTrans, s, 0.0);
      // L_[i] is now log (Sigma[i]), with log
      // being the inverse of the matrix exponential function.
    }
  }
}

void CompressVars::InitS() {
  KALDI_ASSERT(I_!= 0 && D_ != 0 && G_ != 0);
  // Compute global average var.
  SpMatrix<double> Sigma(D_);
  double tot_gamma = gamma_.Sum();
  for (int32 i = 0; i < I_; i++)
    Sigma.AddSp(gamma_(i) / tot_gamma, T_[i]);
  Sigma.Scale(1.0 / gamma_.Sum());
  C_.Resize(D_);  // Cholesky factor of Sigma.
  C_.Cholesky(Sigma); // Compute Cholesky.
  Matrix<double> Cinv(C_);
  Cinv.Invert(); // Compute inverse of C...
  S_.resize(I_);
  for (int32 i = 0; i < I_; i++) {
    S_[i].Resize(D_);
    S_[i].AddMat2Sp(1.0, Cinv, kNoTrans, T_[i], 0.0); // S_i = C^{-1} T_i C^{-T}
  }  
}

void CompressVars::InitPca() {
  int32 DD = (D_ * (D_+1)) / 2;
  // M will be the vectorized form of the L's.
  Matrix<double> M(I_, DD);
  for (int32 i = 0; i < I_; i++) {
    SubVector<double> v(M, i);
    Vectorize(L_[i], &v);
  }
  Matrix<double> B(G_, DD); // the basis elements, as vectors (each row).
  a_.Resize(I_, G_);
  ComputePca(M, &B, &a_, true);
  B_.resize(I_);
  for (int32 g = 0; g < G_; g++) {
    SubVector<double> v(B, g);
    B_[g].Resize(D_);
    UnVectorize(v, &(B_[g]));
  }
  ComputeL(); // don't check return status: L will typically change.
}

bool CompressVars::ComputeL() {
  double tot_diff = 0.0, tot_abs = 0.0;
  SpMatrix<double> L(D_), Ldiff(D_);
  for (int32 i = 0; i < I_; i++) {
    L.SetZero();
    for (int32 g = 0; g < G_; g++)
      L.AddSp(a_(i, g), B_[g]);
    Ldiff.CopyFromSp(L);
    Ldiff.AddSp(-1.0, L_[i]);
    tot_diff += Ldiff.FrobeniusNorm();
    tot_abs += L_[i].FrobeniusNorm();
    L_[i].CopyFromSp(L);
  }
  KALDI_LOG << "ComputeL: relative change is " << (tot_diff / tot_abs);
  return (tot_diff <= 0.001 * tot_abs);
}

// static
void CompressVars::Vectorize(const SpMatrix<double> &S, SubVector<double> *v) {
  int32 D = S.NumRows(), DD = (D*(D+1))/2;
  KALDI_ASSERT(v != NULL && v->Dim() == DD);
  int32 k = 0;
  for (int32 i = 0; i < D; i++) {
    for (int32 j = 0; j < i; j++)
      (*v)(k++) = M_SQRT2 * S(i,j);
    (*v)(k++) = S(i,i);
  }
}

// static
void CompressVars::UnVectorize(const SubVector<double> &v, SpMatrix<double> *S) {
  int32 D = S->NumRows(), DD = (D*(D+1))/2;
  KALDI_ASSERT(D != 0 && v.Dim() == DD);
  int32 k = 0;
  for (int32 i = 0; i < D; i++) {
    for (int32 j = 0; j < i; j++)
      (*S)(i,j) = v(k++) * M_SQRT1_2;
    (*S)(i,i) = v(k++);
  }
}

void CompressVars::Compute(std::vector<SpMatrix<BaseFloat> > *Sigma,
                           BaseFloat *objf_change_out,
                           BaseFloat *count_out) {
  KALDI_ASSERT(Sigma != NULL && static_cast<int32>(Sigma->size()) == I_);
  InitS();
  InitL(*Sigma);
  double objf_at_start = Objf();
  InitPca();
  KALDI_LOG << "CompressVars::Compute, objf change after doing PCA is "
            << ((Objf() - objf_at_start)/gamma_.Sum()) << " per frame over "
            << gamma_.Sum() << " frames.";

  for(int32 iter = 0; iter < num_outer_iters_; iter++) {
    PreconditionForA();
    ComputeA();
    PreconditionForB();
    ComputeB();
  }
  double objf_at_end = Objf();
  KALDI_LOG << "CompressVars::Compute():  Overall objf change is "
            << ((objf_at_end - objf_at_start)/gamma_.Sum()) << " per frame over "
            << gamma_.Sum() << " frames.";
  Finalize(Sigma);
  if (objf_change_out)
    *objf_change_out = (objf_at_end - objf_at_start);
  if(count_out)
    *count_out = gamma_.Sum();
}

void CompressVars::ComputeInv(std::vector<SpMatrix<BaseFloat> > *SigmaInv,
                              BaseFloat *objf_change_out,
                              BaseFloat *count_out) {
  std::vector<SpMatrix<BaseFloat> > Sigma(SigmaInv->size());
  int32 I = Sigma.size();
  KALDI_ASSERT(I == I_);
  for (int32 i = 0; i < I; i++) {
    Sigma[i].Resize(D_);
    Sigma[i].CopyFromSp((*SigmaInv)[i]);
    Sigma[i].Invert();
  }
  Compute(&Sigma, objf_change_out, count_out);
  for (int32 i = 0; i < I; i++) {
    Sigma[i].Invert();
    (*SigmaInv)[i].CopyFromSp(Sigma[i]);
  }
}

// In this stage we make sure the B_g quantities are
// orthonormal (i.e. \tr(\B_f \B_g) = \delta(f,g) )
void CompressVars::PreconditionForA() {
  SpMatrix<double> S(G_);
  for (int32 f = 0; f < G_; f++)
    for (int32 g = 0; g < G_; g++)
      S(f, g) = TraceSpSp(B_[f], B_[g]);
  TpMatrix<double> C(G_);
  C.Cholesky(S);
  // first set a_i <-- C^T a_i
  for(int32 i = 0; i < I_; i++) {
    Vector<double> ahat(G_);
    ahat.AddTpVec(1.0, C, kTrans, a_.Row(i));
    a_.Row(i).CopyFromVec(ahat);
  }
  TpMatrix<double> &D(C); // D will be inverse of C; reuse memory.
  D.Invert();
  for(int32 f = G_-1; f >= 0; f--) {
    // Compute \hat{\B}_f = \sum_{g=1}^f d_{fg} \B_g, in place.
    B_[f].Scale(D(f,f));
    for(int32 g = 0; g < f; g++)
      B_[f].AddSp(D(f,g), B_[g]);
  }
  KALDI_ASSERT(ComputeL()); // check return status.. should not change.
  // put inside assert so would not be done in optimized mode.
}


// In this stage we make sure the a_{ig} quantities have
// unit variance in the space of size G (after weighting by
// count).
void CompressVars::PreconditionForB() {
  double gamma_tot = gamma_.Sum();
  SpMatrix<double> S(G_);
  for (int32 i = 0; i < I_; i++)
    S.AddVec2(gamma_(i) / gamma_tot, a_.Row(i));
  TpMatrix<double> C(G_);
  C.Cholesky(S);
  for(int32 g = 0; g < G_; g++) {
    // Compute \hat{\B}_g = \sum_{f=g}^G c_{fg} \B_g, in place.
    B_[g].Scale(C(g, g));
    for(int32 f = g+1; f < G_; f++)
      B_[g].AddSp(C(f, g), B_[f]);
  }
  TpMatrix<double> &D(C); // D will be inverse of C; reuse memory.
  D.Invert();
  // now set a_i <-- C^{-1} a_i
  for(int32 i = 0; i < I_; i++) {
    Vector<double> ahat(G_);
    ahat.AddTpVec(1.0, D, kNoTrans, a_.Row(i));
    a_.Row(i).CopyFromVec(ahat);
  }
  KALDI_ASSERT(ComputeL()); // check return status.. should not change.
  // put inside assert so would not be done in optimized mode.
}

void CompressVars::ComputeA() {
  double objf_start = Objf();
  for (int32 gauss_i = 0; gauss_i < I_; gauss_i++) { // optimize each i in turn.
    if (gamma_(gauss_i) == 0) continue; // do nothing-- no data.
    double inv_gammai = 1.0 / gamma_(gauss_i);
    SubVector<double> x(a_, gauss_i); // this row of a.
    Vector<double> r(G_); 
    DerivA(gauss_i, x, &r);
    r.Scale(-1.0); // residual r <-- -f'(\x)
    double g = ObjfA(gauss_i, x), g_old;
    Vector<double> s(r);
    s.Scale(inv_gammai); // s <-- M^{-1} r. 
    Vector<double> d(s); // d <-- s
    double delta_new = VecVec(r, d), delta_old;
    int32 m = 0;
    Vector<double> deriv(G_);
    for (int32 i = 0; i < num_cg_iters_a_; i++) {
      g_old = g;
      double delta_d = VecVec(d, d);
      const double sigma_0 = 1.0;
      double alpha = -sigma_0;
      double eta_prev;
      { // compute eta_prev = d^T f'(x + sigma_0 d)
        Vector<double> temp(x);
        temp.AddVec(sigma_0, d); // does temp <-- x + sigma_0 d, with sigma_0=1.0
        DerivA(gauss_i, temp, &deriv);
        eta_prev = VecVec(d, deriv);
      }
      for (int32 j = 0; j < num_newton_iters_; j++) {
        DerivA(gauss_i, x, &deriv);
        double eta = VecVec(d, deriv);
        alpha *= (eta / (eta_prev - eta));
        KALDI_VLOG(3) << "iter = " << j << ", new alpha is " << alpha; // should approach 0.
        int32 k = 0;
        while (k < num_backtrack_iters_) {
          // work out f_x_alpha_d
          Vector<double> temp(x);
          temp.AddVec(alpha, d); // does temp <-- x + sigma_0 d, with sigma_0=1.0
          double f_x_alpha_d = ObjfA(gauss_i, temp);  // f(x + alpha d).
          // Improved, or change is so small that failure to improve must be numeric issue.
          if (f_x_alpha_d <= g || fabs(f_x_alpha_d - g) < 1.0e-10 * fabs(f_x_alpha_d)) {
            g = f_x_alpha_d;
            x.CopyFromVec(temp);
            break;
          }
          KALDI_VLOG(3) << "backtracking as " << f_x_alpha_d << " <= " << g; 
          alpha *= 0.5;
          k++;
        }
        if (k == num_backtrack_iters_) { // need to set x and g.
          // KALDI_WARN << "Backtracked too many times in CG for a!";
          x.AddVec(alpha, d);
          g = ObjfA(gauss_i, x);
        }
        eta_prev = eta;
        if (alpha*alpha*delta_d < epsilon_*epsilon_) break;
      }
      KALDI_VLOG(3) << "On iter " << i << " of optimization of a, objf is "
                    << g << ", change is " << g - g_old;
      if (g_old - g < epsilon_*epsilon_) break;
      delta_old = delta_new;
      double delta_mid = VecVec(r, s);
      DerivA(gauss_i, x, &r);
      r.Scale(-1.0);
      s.CopyFromVec(r);
      s.Scale(inv_gammai); // s <-- M^{-1} r.
      delta_new = VecVec(r, s);
      double beta = ((delta_new - delta_mid) / delta_old);
      m++;
      if (beta <= 0 || m == x.Dim()) {
        d.CopyFromVec(s); // Restart.
        m = 0;
      } else {
        d.Scale(beta);
        d.AddVec(1.0, s); // d <-- beta * d + s.
      }
    }
  }
  ComputeL();
  double objf_end = Objf();
  KALDI_LOG << "CompressVars::ComputeA(), objf change per frame is "
            << ((objf_end-objf_start)/gamma_.Sum());
}


void CompressVars::ComputeB() {
  double objf_start = Objf(); // for diagnostics.  Has the factor of -0.5.
  double gamma_inv = 1.0 / gamma_.Sum();
  std::vector<SpMatrix<double> > &X(B_); // call it X... refers to B_.
  std::vector<SpMatrix<double> > R;
  DerivB(X, &R);
  ScaleB(-1.0, &R); // r <-- -f'(x)
  double g = ObjfB(X), g_old;
  std::vector<SpMatrix<double> > S;
  CopyB(R, &S);
  ScaleB(gamma_inv, &S); // s <-- M^{-1} r
  std::vector<SpMatrix<double> > D;  
  CopyB(S, &D);   // d <-- s
  double delta_new = InnerProductB(R, D), delta_old;
  std::vector<SpMatrix<double> > deriv;
  for (int32 i = 0; i < num_cg_iters_B_; i++) {
    g_old = g;
    double delta_d = InnerProductB(D, D);
    const double sigma_0 = 1.0;
    double alpha = -sigma_0;
    double eta_prev;
    { // compute eta_prev = d^T f'(x + sigma_0 d)
      std::vector<SpMatrix<double> > temp;
      CopyB(X, &temp); // temp = x.
      AddB(D, sigma_0, &temp); // temp = X + sigma_0 D.
      DerivB(temp, &deriv);
      eta_prev = InnerProductB(D, deriv);
    }
    for (int32 j = 0; j < num_newton_iters_; j++) {
      DerivB(X, &deriv);
      double eta = InnerProductB(D, deriv);
      alpha *= (eta / (eta_prev - eta));
      KALDI_VLOG(2) << "iter = " << j << ", new alpha is " << alpha; // should approach 0.
      int32 k = 0;
      while (k < num_backtrack_iters_) {
        // work out f_x_alpha_d
        std::vector<SpMatrix<double> > temp;
        CopyB(X, &temp); // temp = x.
        AddB(D, alpha, &temp); // temp = x + alpha d.
        double f_x_alpha_d = ObjfB(temp);  // f(x + alpha d).
        // Improved, or change is so small that failure to improve must be numeric issue.
        if (f_x_alpha_d <= g || fabs(f_x_alpha_d - g) < 1.0e-10 * fabs(f_x_alpha_d)) {
          g = f_x_alpha_d;
          CopyB(temp, &X); // x = temp.
          break;
        }
        KALDI_VLOG(2) << "backtracking as " << f_x_alpha_d << " <= " << g; 
        alpha *= 0.5;
        k++;
      }
      if (k == num_backtrack_iters_) { // need to set x and g.
        // KALDI_WARN << "Backtracked too many times in CG for a!";
        AddB(D, alpha, &X); // x = x + alpha d.
        g = ObjfB(X);
      }
      eta_prev = eta;
      if (alpha*alpha*delta_d < epsilon_*epsilon_) break;
    }
    KALDI_VLOG(2) << "On iter " << i << " of optimization of B, objf is "
                  << g << ", change is " << g - g_old;
    if (g_old - g < epsilon_*epsilon_) break;
    delta_old = delta_new;
    double delta_mid = InnerProductB(R, S);
    DerivB(X, &R); 
    ScaleB(-1.0, &R); // r = -f'(x)
    CopyB(R, &S); // s = r.
    ScaleB(gamma_inv, &S); // s <-- M^{-1} r
    delta_new = InnerProductB(R, S); // delta_new = r^T s
    double beta = ((delta_new - delta_mid) / delta_old);
    if (beta <= 0) { // Restart only if beta <= 0; dimension of this problem is so large
      // that we will never reach its dimension.
      CopyB(S, &D); // Restart (d = s)
    } else {
      ScaleB(beta, &D); 
      AddB(S, 1.0, &D); // d <-- beta * d + s.
    }
  }
  
  ComputeL();
  double objf_end = Objf();
  KALDI_LOG << "CompressVars::ComputeB(), objf change per frame is "
            << ((objf_end-objf_start)/gamma_.Sum());
}


double CompressVars::ObjfA(int32 i, const VectorBase<double> &a) const {
  SpMatrix<double> MinusLi(D_);
  for (int32 g = 0; g < G_; g++)
    MinusLi.AddSp(-a(g), B_[g]);
  MatrixExponential<double> mexp;
  SpMatrix<double> SigmaInvHat(D_);
  mexp.Compute(MinusLi, &SigmaInvHat);
  return gamma_(i) * -MinusLi.Trace() + TraceSpSp(SigmaInvHat, S_[i]);
}

// compute derivative w.r.t. this a_i.
void CompressVars::DerivA(int32 i, const VectorBase<double> &a, VectorBase<double> *d) const {
  SpMatrix<double> MinusLi(D_);
  for (int32 g = 0; g < G_; g++)
    MinusLi.AddSp(-a(g), B_[g]);
  MatrixExponential<double> mexp;
  SpMatrix<double> SigmaInvHat(D_);
  mexp.Compute(MinusLi, &SigmaInvHat);
  SpMatrix<double> &LiHat(MinusLi); // use the same memory for deriative.
  mexp.Backprop(S_[i], &LiHat);
  LiHat.Scale(-1.0);
  double gamma = gamma_(i);
  for (int32 e = 0; e < D_; e++)
    LiHat(e,e) += gamma;
  for (int32 g = 0; g < G_; g++)
    (*d)(g) = TraceSpSp(LiHat, B_[g]);
}


double CompressVars::Objf() const {
  double objf = 0.0;
  MatrixExponential<double> mexp;
  for (int32 i = 0; i < I_; i++) {
    SpMatrix<double> SigmaInvHat(D_);
    SpMatrix<double> MinusLi(L_[i]);
    MinusLi.Scale(-1.0);
    mexp.Compute(MinusLi, &SigmaInvHat); // Compute SigmaInvHat = exp(L)
    objf += -0.5 * (gamma_(i) * L_[i].Trace()  +  TraceSpSp(S_[i], SigmaInvHat));
  }
  return objf;
}

double CompressVars::ObjfB(const std::vector<SpMatrix<double> > &B) const {
  double objf = 0.0;
  MatrixExponential<double> mexp;
  for (int32 i = 0; i < I_; i++) {
    SpMatrix<double> SigmaInvHat(D_);
    SpMatrix<double> MinusLi(D_);
    for (int32 g = 0; g < G_; g++)
      MinusLi.AddSp(-a_(i, g), B[g]);
    mexp.Compute(MinusLi, &SigmaInvHat); // Compute SigmaInvHat = exp(L)
    objf += (gamma_(i) * -MinusLi.Trace()  +  TraceSpSp(S_[i], SigmaInvHat));
  }
  return objf;
}

void CompressVars::DerivB(const std::vector<SpMatrix<double> > &B,
                          std::vector<SpMatrix<double> > *D) const {
  if (D->size() != G_)
    D->resize(G_);
  for (int32 i = 0; i < G_; i++)
    (*D)[i].Resize(D_); // also zeros them.
  
  MatrixExponential<double> mexp;
  for (int32 i = 0; i < I_; i++) {
    SpMatrix<double> SigmaInvHat(D_);
    SpMatrix<double> MinusLi(D_);
    for (int32 g = 0; g < G_; g++)
      MinusLi.AddSp(-a_(i, g), B[g]);
    mexp.Compute(MinusLi, &SigmaInvHat); // Compute SigmaInvHat = exp(L)
    SpMatrix<double> &LiHat(MinusLi); // re-use this variale.
    mexp.Backprop(S_[i], &LiHat);
    LiHat.Scale(-1.0); // get deriv w.r.t L_i, not -L_i.
    double gamma = gamma_(i);
    for (int32 e = 0; e < D_; e++)
      LiHat(e,e) += gamma;
    for (int32 g = 0; g < G_; g++)
      (*D)[g].AddSp(a_(i,g), LiHat);
  }
}
double CompressVars::InnerProductB(const std::vector<SpMatrix<double> > &B,
                                   const std::vector<SpMatrix<double> > &C) {
  double ans = 0.0;
  for(int32 g = 0; g < G_; g++)
    ans += TraceSpSp(B[g], C[g]);
  return ans;
}

void CompressVars::CopyB(const std::vector<SpMatrix<double> > &B,
                         std::vector<SpMatrix<double> > *C) {
  C->resize(G_);
  for(int32 g = 0; g < G_; g++) {
    (*C)[g].Resize(D_);
    (*C)[g].CopyFromSp(B[g]);
  }
}

void CompressVars::AddB(const std::vector<SpMatrix<double> > &B,
                        double alpha,
                        std::vector<SpMatrix<double> > *C) {
  for(int32 g = 0; g < G_; g++)
    (*C)[g].AddSp(alpha, B[g]);
}

void CompressVars::ScaleB(double alpha, std::vector<SpMatrix<double> > *B) {
  for(int32 g = 0; g < G_; g++)
    (*B)[g].Scale(alpha);
}

void CompressVars::Finalize(std::vector<SpMatrix<BaseFloat> > *Sigma) {
  SpMatrix<double> SigmaHat(D_), ThisSigma(D_);
  MatrixExponential<double> mexp;
  Matrix<double> C(C_); // convert to matrix.
  for(int32 i = 0; i < I_; i++) {
    mexp.Compute(L_[i], &SigmaHat);
    ThisSigma.AddMat2Sp(1.0, C, kNoTrans, SigmaHat, 0.0);
    (*Sigma)[i].CopyFromSp(ThisSigma);
  }
}


} // namespace kaldi

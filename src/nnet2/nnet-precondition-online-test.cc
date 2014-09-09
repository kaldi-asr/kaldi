// nnet2/nnet-precondition-online-test.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet2/nnet-precondition-online.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet2 {

// Simple version of OnlinePreconditioner that we use to make
// sure it is behaving as advertised.
class OnlinePreconditionerSimple {
 public:
  OnlinePreconditionerSimple(): rank_(40), num_samples_history_(2000.0), alpha_(4.0),
                                epsilon_(1.0e-10), delta_(1.0e-05) { }
  
  void SetRank(int32 rank) { rank_ = rank; }

  void PreconditionDirections(
      CuMatrixBase<BaseFloat> *R,
      CuVectorBase<BaseFloat> *row_prod,
      BaseFloat *scale);


 private:
  BaseFloat Eta(int32 N) const;
  
  void PreconditionDirectionsCpu(
      MatrixBase<double> *R,
      VectorBase<double> *row_prod,
      BaseFloat *scale);
  
  
  void Init(const MatrixBase<double> &R0);

  int32 rank_;  
  double num_samples_history_;
  double alpha_;
  double epsilon_;
  double delta_;
  
  Vector<double> d_t_;
  Matrix<double> X_t_;
  double rho_t_;  
};


void OnlinePreconditionerSimple::PreconditionDirections(
      CuMatrixBase<BaseFloat> *R,
      CuVectorBase<BaseFloat> *row_prod,
      BaseFloat *scale) {
  Matrix<BaseFloat> R_cpu(*R);
  Vector<BaseFloat> row_prod_cpu(*row_prod);
  Matrix<double> R_cpu_dbl(R_cpu);
  Vector<double> row_prod_cpu_dbl(row_prod_cpu);
  PreconditionDirectionsCpu(&R_cpu_dbl,
                            &row_prod_cpu_dbl,
                            scale);
  row_prod_cpu.CopyFromVec(row_prod_cpu_dbl);
  R_cpu.CopyFromMat(R_cpu_dbl);
  R->CopyFromMat(R_cpu);
  row_prod->CopyFromVec(row_prod_cpu);
}

void OnlinePreconditionerSimple::Init(const MatrixBase<double> &R0) {
  int32 D = R0.NumCols(), N = R0.NumRows();
  SpMatrix<double> S(D);
  S.AddMat2(1.0 / N, R0, kTrans, 0.0);
  Matrix<double> P(D, D);
  Vector<double> s(D);
  S.Eig(&s, &P);
  SortSvd(&s, &P);
  if (rank_ >= D) {
    KALDI_WARN << "Rank " << rank_ << " of online preconditioner is >= dim " << D
               << ", setting it to "
               << (D - 1) << " (but this is probably still too high)";
    rank_ = D - 1;
  }
  int32 R = rank_;
  X_t_.Resize(R, D);
  P.Transpose();
  X_t_ = P.Range(0, R, 0, D);
  d_t_ = s.Range(0, R);
  KALDI_VLOG(3) << "d_t orig is " << d_t_;
  rho_t_ = (TraceMatMat(R0, R0, kTrans) / N - d_t_.Sum()) / (D - R);
  d_t_.Add(-rho_t_);
  KALDI_VLOG(3) << "rho_0 = " << rho_t_;
  KALDI_VLOG(3) << "d_0 = " << d_t_;
  double floor_val = std::max(epsilon_, delta_ * d_t_.Max());
  if (rho_t_ < floor_val) {
    KALDI_WARN << "Flooring rho_0 to " << floor_val << ", was " << rho_t_;
    rho_t_ = floor_val;
  }
  int32 nf = d_t_.ApplyFloor(epsilon_);
  if (nf > 0) {
    KALDI_WARN << "Floored " << nf << " elements of D_0";
  }  
}

BaseFloat OnlinePreconditionerSimple::Eta(int32 N) const {
  KALDI_ASSERT(num_samples_history_ > 0.0);
  return 1.0 - exp(-N / num_samples_history_);
}


void OnlinePreconditionerSimple::PreconditionDirectionsCpu(
    MatrixBase<double> *R_t,
    VectorBase<double> *row_prod,
    BaseFloat *scale) {
  if (X_t_.NumRows() == 0)
    Init(*R_t);
  int32 R = X_t_.NumRows(), D = X_t_.NumCols(), N = R_t->NumRows();
  BaseFloat eta = Eta(N);
  
  SpMatrix<double> F_t(D);
  // F_t =(def) X_t^T D_t X_t + \rho_t I
  F_t.AddToDiag(rho_t_);
  F_t.AddMat2Vec(1.0, X_t_, kTrans, d_t_, 1.0);

  // S_t =(def) 1/N R_t^T R_t.
  SpMatrix<double> S_t(D);
  S_t.AddMat2(1.0 / N, *R_t, kTrans, 0.0);

  // T_t =(def) \eta S_t + (1-\eta) F_t
  SpMatrix<double> T_t(D);
  T_t.AddSp(eta, S_t);
  T_t.AddSp(1.0 - eta, F_t);

  // Y_t =(def) X_t T_t
  Matrix<double> Y_t(R, D);
  Y_t.AddMatSp(1.0, X_t_, kNoTrans, T_t, 0.0);

  // Z_t =(def) Y_t Y_t^T
  SpMatrix<double> Z_t(R);
  Z_t.AddMat2(1.0, Y_t, kNoTrans, 0.0);

  Matrix<double> U_t(R, R);
  Vector<double> c_t(R);
  // decompose Z_t = U_t C_t U_t^T
  Z_t.Eig(&c_t, &U_t);
  SortSvd(&c_t, &U_t);
  double c_t_floor = pow(rho_t_ * (1.0 - eta), 2);
  int32 nf = c_t.ApplyFloor(c_t_floor);
  if (nf > 0) {
    KALDI_WARN << "Floored " << nf << " elements of c_t.";
  }
  // KALDI_LOG << "c_t is " << c_t;
  // KALDI_LOG << "U_t is " << U_t;
  // KALDI_LOG << "Z_t is " << Z_t;
  
  Vector<double> sqrt_c_t(c_t);
  sqrt_c_t.ApplyPow(0.5);
  Vector<double> inv_sqrt_c_t(sqrt_c_t);
  inv_sqrt_c_t.InvertElements();
  Matrix<double> X_t1(R, D);
  // X_{t+1} = C_t^{-0.5} U_t^T Y_t
  X_t1.AddMatMat(1.0, U_t, kTrans, Y_t, kNoTrans, 0.0);
  X_t1.MulRowsVec(inv_sqrt_c_t);

  double rho_t1 = (1.0 / (D - R)) *
      (eta * S_t.Trace() + (1.0 - eta) * (D * rho_t_ + d_t_.Sum()) - sqrt_c_t.Sum());

  Vector<double> d_t1(sqrt_c_t);
  d_t1.Add(-rho_t1);

  double floor_val = std::max(epsilon_, delta_ * sqrt_c_t.Max());
  if (rho_t1 < floor_val) {
    KALDI_WARN << "flooring rho_{t+1} to " << floor_val << ", was " << rho_t1;
    rho_t1 = floor_val;
  }
  nf = d_t1.ApplyFloor(epsilon_);
  if (nf > 0) {
    KALDI_VLOG(3) << "d_t1 was " << d_t1;
    KALDI_WARN << "Floored " << nf << " elements of d_{t+1}.";
  }  
  // a check.
  if (nf == 0 && rho_t1 > epsilon_) {
    double tr_F_t1 = D * rho_t1 + d_t1.Sum(), tr_T_t = T_t.Trace();
    AssertEqual(tr_F_t1, tr_T_t);
  }

  // G_t = F_t + alpha/D tr(F_t)
  SpMatrix<double> G_t(F_t);
  G_t.AddToDiag(alpha_ / D * F_t.Trace());
  SpMatrix<double> G_t_inv(G_t);
  G_t_inv.Invert();

  double beta_t = rho_t_ + alpha_/D * F_t.Trace();
  // P_t = beta_t R_t G_t^{-1}.
  Matrix<double> P_t(N, D);
  P_t.AddMatSp(beta_t, *R_t, kNoTrans, G_t_inv, 0.0);

  double tr_r_r = TraceMatMat(*R_t, *R_t, kTrans),
      tr_p_p = TraceMatMat(P_t, P_t, kTrans);
  double gamma = (tr_p_p == 0 ? 1.0 : sqrt(tr_r_r / tr_p_p));

  R_t->CopyFromMat(P_t);
  row_prod->AddDiagMat2(1.0, *R_t, kNoTrans, 0.0);
  *scale = gamma;

  // Update the parameters
  rho_t_ = rho_t1;
  d_t_.CopyFromVec(d_t1);
  X_t_.CopyFromMat(X_t1);

  KALDI_VLOG(3) << "rho_t_ = " << rho_t_;
  KALDI_VLOG(3) << "d_t_ = " << d_t_;
  KALDI_VLOG(3) << "X_t_ = " << X_t_;
  

  { // check that X_t_ X_t_^T = I.
    SpMatrix<double> unit(R);
    unit.AddMat2(1.0, X_t_, kNoTrans, 0.0);
    KALDI_ASSERT(unit.IsUnit(1.0e-03));
  }
}


void UnitTestPreconditionDirectionsOnline() {
  MatrixIndexT R = 1 + Rand() % 5,  // rank of correction
      N = (2 * R) + Rand() % 30,  // batch size
      D = R + 1 + Rand() % 20; // problem dimension.  Must be > R.

  // Test sometimes with features that are all-zero or all-one; this will
  // help to make sure low-rank or zero input doesn't crash the code.
  bool zero = false;
  bool one = false;
  if (Rand() % 3 == 0) zero = true;
  else if (Rand() % 2 == 0) one = true;
  
  CuVector<BaseFloat> row_prod1(N), row_prod2(N);
  BaseFloat gamma1, gamma2;
  
  OnlinePreconditionerSimple preconditioner1;
  OnlinePreconditioner preconditioner2;
  preconditioner1.SetRank(R);
  preconditioner2.SetRank(R);
  preconditioner2.TurnOnDebug();

  int32 num_iters = 100;
  for (int32 iter = 0; iter < num_iters; iter++) {
    CuMatrix<BaseFloat> M(N, D);
    if (one) M.Set(1.0);
    else if (!zero)
      M.SetRandn();
    
    CuMatrix<BaseFloat> Mcopy1(M), Mcopy2(M);

    preconditioner1.PreconditionDirections(&Mcopy1, &row_prod1, &gamma1);

    preconditioner2.PreconditionDirections(&Mcopy2, &row_prod2, &gamma2);

    AssertEqual(Mcopy1, Mcopy2);
  }
  return;
}


// outputs eigs to rows of P.
void ExactEigsOfProduct(const CuMatrixBase<BaseFloat> &M,
                        MatrixTransposeType trans,
                        CuMatrixBase<BaseFloat> *P,
                        CuVectorBase<BaseFloat> *s) {
  Matrix<BaseFloat> M_cpu(M);
  int32 D = trans == kTrans ? M.NumCols() : M.NumRows();
  SpMatrix<BaseFloat> S_cpu(D);
  S_cpu.AddMat2(1.0, M_cpu, trans, 0.0);
  Matrix<BaseFloat> P_cpu(D, D);
  Vector<BaseFloat> s_cpu(D);
  S_cpu.Eig(&s_cpu, &P_cpu);
  SortSvd(&s_cpu, &P_cpu);
  P->CopyFromMat(P_cpu.Range(0, D, 0, P->NumRows()), kTrans);
  s->CopyFromVec(s_cpu.Range(0, P->NumRows()));
}
  

void UnitTestApproxEigsOfProduct() {
  int32 dimM = 10 + Rand() % 50,
      dimN = 10 + Rand() % 50;
  MatrixTransposeType trans = (Rand() % 2 == 0 ? kTrans : kNoTrans);
  int32 product_dim = (trans == kTrans ? dimN : dimM),
      other_dim = (trans == kTrans ? dimM : dimN);
  
  CuMatrix<BaseFloat> M(dimM, dimN);
  if (Rand() % 4 == 0) {
    M.SetRandn();
  } else if (Rand() % 3 == 0) {
    M.Row(2).SetRandn();
  } else if (Rand() % 2 == 0) {
    M.Row(2).SetRandn();
    M.Row(4).SetRandn();
  }
  // else leave M at zero.  We want to test
  // full-rank M as well as zero, one or two eigenvalues
  // being nonzero.

  int32 rank = 1 + Rand() % (product_dim - 1);

  CuMatrix<BaseFloat> P_approx(rank, product_dim),
      P_exact(rank, product_dim);
  CuVector<BaseFloat> s_approx(rank),
      s_exact(rank);

  ExactEigsOfProduct(M, trans, &P_exact, &s_exact);
  ApproxEigsOfProduct(M, trans, &P_approx, &s_approx);

  KALDI_LOG << "Approx eig sum is " << s_approx.Sum();
  KALDI_LOG << "Exact eig sum is " << s_exact.Sum();

  CuMatrix<BaseFloat> unit1(rank, rank), unit2(rank, rank);
  unit1.AddMatMat(1.0, P_approx, kNoTrans, P_approx, kTrans, 0.0);
  unit2.AddMatMat(1.0, P_exact, kNoTrans, P_exact, kTrans, 0.0);
  KALDI_ASSERT(unit1.IsUnit());
  KALDI_ASSERT(unit2.IsUnit());

  CuMatrix<BaseFloat> Mproj_approx(rank, other_dim);
  Mproj_approx.AddMatMat(1.0, P_approx, kNoTrans, M, trans, 0.0);
  CuMatrix<BaseFloat> Mproj_exact(rank, other_dim);
  Mproj_exact.AddMatMat(1.0, P_exact, kNoTrans, M, trans, 0.0);

  CuVector<BaseFloat> s2_approx(rank), s2_exact(rank);
  s2_approx.AddDiagMat2(1.0, Mproj_approx, kNoTrans, 0.0);
  s2_exact.AddDiagMat2(1.0, Mproj_exact, kNoTrans, 0.0);  
  KALDI_ASSERT(s_approx.ApproxEqual(s2_approx));
  // KALDI_LOG << "s_exact is " << s_exact;
  // KALDI_LOG << "s2_exact is " << s2_exact;
  // KALDI_LOG << "P_exact is " << P_exact;
  KALDI_ASSERT(s_exact.ApproxEqual(s2_exact));
  
}

/*
  CuSpMatrix<BaseFloat> G(D);
  G.SetUnit();
  G.ScaleDiag(lambda);
  // G += R^T R.
  G.AddMat2(1.0/(N-1), R, kTrans, 1.0);
  
  for (int32 n = 0; n < N; n++) {
    CuSubVector<BaseFloat> rn(R, n);
    CuSpMatrix<BaseFloat> Gn(G);
    Gn.AddVec2(-1.0/(N-1), rn); // subtract the
    // outer product of "this" vector.
    Gn.Invert();
    CuSubVector<BaseFloat> pn(P, n);
    CuVector<BaseFloat> pn_compare(D);
    pn_compare.AddSpVec(1.0, Gn, rn, 0.0);
    KALDI_ASSERT(pn.ApproxEqual(pn_compare, 0.1));
  }
}

*/
} // namespace nnet2
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
    for (int32 i = 0; i < 30; i++) {
      UnitTestPreconditionDirectionsOnline();
      UnitTestApproxEigsOfProduct();
    }
  }
}

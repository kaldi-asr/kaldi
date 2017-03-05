// nnet3/natural-gradient-online-test.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet3/natural-gradient-online.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet3 {

// Simple version of OnlineNaturalGradient that we use to make
// sure it is behaving as advertised.
class OnlineNaturalGradientSimple {
 public:
  OnlineNaturalGradientSimple(): rank_(40), num_samples_history_(2000.0), alpha_(4.0),
                                epsilon_(1.0e-10), delta_(5.0e-04) { }

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

  void InitDefault(int32 D);

  int32 rank_;
  double num_samples_history_;
  double alpha_;
  double epsilon_;
  double delta_;

  // Fisher matrix defined as F_t = R_t^T diag(d_t) R_t + rho_t I.
  Vector<double> d_t_;
  Matrix<double> R_t_;
  double rho_t_;
};


void OnlineNaturalGradientSimple::PreconditionDirections(
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

void OnlineNaturalGradientSimple::InitDefault(int32 D) {
  if (rank_ >= D) {
    KALDI_WARN << "Rank " << rank_ << " of online preconditioner is >= dim " << D
               << ", setting it to "
               << (D - 1) << " (but this is probably still too high)";
    rank_ = D - 1;
  }
  int32 R = rank_;
  R_t_.Resize(R, D);
  for (int32 r = 0; r < R; r++) {
    std::vector<int32> cols;
    for (int32 c = r; c < D; c += R)
      cols.push_back(c);
    for (int32 i = 0; i < cols.size(); i++) {
      int32 c = cols[i];
      R_t_(r, c) = (i == 0 ? 1.1 : 1.0) /
          sqrt(1.1 * 1.1 + cols.size() - 1);
    }
  }
  d_t_.Resize(R);
  d_t_.Set(epsilon_);
  rho_t_ = epsilon_;
}

void OnlineNaturalGradientSimple::Init(const MatrixBase<double> &R0) {
  int32 D = R0.NumCols(), N = R0.NumRows();
  InitDefault(D);
  int32 num_init_iters = 3;
  for (int32 i = 0; i < num_init_iters; i++) {
    CuMatrix<BaseFloat> R0_copy(R0);
    CuVector<BaseFloat> row_products(N);
    BaseFloat scale;
    PreconditionDirections(&R0_copy, &row_products, &scale);
  }
}

BaseFloat OnlineNaturalGradientSimple::Eta(int32 N) const {
  KALDI_ASSERT(num_samples_history_ > 0.0);
  BaseFloat ans = 1.0 - exp(-N / num_samples_history_);
  if (ans > 0.9) ans = 0.9;
  return ans;
}


void OnlineNaturalGradientSimple::PreconditionDirectionsCpu(
    MatrixBase<double> *X_t,
    VectorBase<double> *row_prod,
    BaseFloat *scale) {
  if (R_t_.NumRows() == 0)
    Init(*X_t);
  int32 R = R_t_.NumRows(), D = R_t_.NumCols(), N = X_t->NumRows();
  BaseFloat eta = Eta(N);

  SpMatrix<double> F_t(D);
  // F_t =(def) R_t^T D_t R_t + \rho_t I
  F_t.AddToDiag(rho_t_);
  F_t.AddMat2Vec(1.0, R_t_, kTrans, d_t_, 1.0);

  // Make sure F_t is +ve definite.
  {
    KALDI_ASSERT(d_t_.Min() > 0);
    Vector<double> eigs(D);
    F_t.Eig(&eigs, NULL);
    KALDI_ASSERT(eigs.Min() > 0);
  }

  // S_t =(def) 1/N X_t^T X_t.
  SpMatrix<double> S_t(D);
  S_t.AddMat2(1.0 / N, *X_t, kTrans, 0.0);

  // T_t =(def) \eta S_t + (1-\eta) F_t
  SpMatrix<double> T_t(D);
  T_t.AddSp(eta, S_t);
  T_t.AddSp(1.0 - eta, F_t);

  // Y_t =(def) R_t T_t
  Matrix<double> Y_t(R, D);
  Y_t.AddMatSp(1.0, R_t_, kNoTrans, T_t, 0.0);

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
  Matrix<double> R_t1(R, D);
  // R_{t+1} = C_t^{-0.5} U_t^T Y_t
  R_t1.AddMatMat(1.0, U_t, kTrans, Y_t, kNoTrans, 0.0);
  R_t1.MulRowsVec(inv_sqrt_c_t);

  double rho_t1 = (1.0 / (D - R)) *
      (eta * S_t.Trace() + (1.0 - eta) * (D * rho_t_ + d_t_.Sum()) - sqrt_c_t.Sum());

  Vector<double> d_t1(sqrt_c_t);
  d_t1.Add(-rho_t1);

  double floor_val = std::max(epsilon_, delta_ * sqrt_c_t.Max());
  if (rho_t1 < floor_val) {
    KALDI_WARN << "flooring rho_{t+1} to " << floor_val << ", was " << rho_t1;
    rho_t1 = floor_val;
  }
  nf = d_t1.ApplyFloor(floor_val);
  if (nf > 0) {
    KALDI_VLOG(3) << "d_t1 was " << d_t1;
    KALDI_WARN << "Floored " << nf << " elements of d_{t+1}.";
  }
  // a check.
  if (nf == 0 && rho_t1 > floor_val) {
    double tr_F_t1 = D * rho_t1 + d_t1.Sum(), tr_T_t = T_t.Trace();
    AssertEqual(tr_F_t1, tr_T_t);
  }

  // G_t = F_t + alpha/D tr(F_t)
  SpMatrix<double> G_t(F_t);
  G_t.AddToDiag(alpha_ / D * F_t.Trace());
  SpMatrix<double> G_t_inv(G_t);
  G_t_inv.Invert();

  double beta_t = rho_t_ + alpha_/D * F_t.Trace();
  // X_hat_t = beta_t X_t G_t^{-1}.
  Matrix<double> X_hat_t(N, D);
  X_hat_t.AddMatSp(beta_t, *X_t, kNoTrans, G_t_inv, 0.0);

  double tr_x_x = TraceMatMat(*X_t, *X_t, kTrans),
      tr_Xhat_Xhat = TraceMatMat(X_hat_t, X_hat_t, kTrans);
  double gamma = (tr_Xhat_Xhat == 0 ? 1.0 : sqrt(tr_x_x / tr_Xhat_Xhat));

  X_t->CopyFromMat(X_hat_t);
  row_prod->AddDiagMat2(1.0, *X_t, kNoTrans, 0.0);
  *scale = gamma;

  // Update the parameters
  rho_t_ = rho_t1;
  d_t_.CopyFromVec(d_t1);
  R_t_.CopyFromMat(R_t1);

  KALDI_VLOG(3) << "rho_t_ = " << rho_t_;
  KALDI_VLOG(3) << "d_t_ = " << d_t_;
  KALDI_VLOG(3) << "R_t_ = " << R_t_;


  { // check that R_t_ R_t_^T = I.
    SpMatrix<double> unit(R);
    unit.AddMat2(1.0, R_t_, kNoTrans, 0.0);
    if (!unit.IsUnit(1.0e-03)) {
      KALDI_WARN  << "R is not orthogonal, reorthogonalizing.";
      for (int32 i = 0; i < R; i++) {
        SubVector<double> row(R_t_, i);
        for (int32 j = 0; j < i; j++) {
          SubVector<double> row_j(R_t_, j);
          row.AddVec(-VecVec(row_j, row), row_j);
        }
        row.Scale(1.0 / row.Norm(2.0));
      }
    }
    unit.AddMat2(1.0, R_t_, kNoTrans, 0.0);
    KALDI_ASSERT(unit.IsUnit(1.0e-03));
  }
}


void UnitTestPreconditionDirectionsOnline() {
  MatrixIndexT R = 1 + Rand() % 30,  // rank of correction
      N = (2 * R) + Rand() % 30,  // batch size
      D = R + 1 + Rand() % 20; // problem dimension.  Must be > R.

  // Test sometimes with features that are all-zero or all-one; this will
  // help to make sure low-rank or zero input doesn't crash the code.
  bool zero = false;
  bool one = false;
  if (Rand() % 3 == 0) zero = true;
  //else if (Rand() % 2 == 0) one = true;

  CuVector<BaseFloat> row_prod1(N), row_prod2(N);
  BaseFloat gamma1, gamma2;
  BaseFloat big_eig_factor = RandInt(1, 20);
  big_eig_factor = big_eig_factor * big_eig_factor;
  Vector<BaseFloat> big_eig_vector(D);
  big_eig_vector.SetRandn();
  big_eig_vector.Scale(big_eig_factor);

  OnlineNaturalGradientSimple preconditioner1;
  OnlineNaturalGradient preconditioner2;
  preconditioner1.SetRank(R);
  preconditioner2.SetRank(R);
  preconditioner2.TurnOnDebug();

  int32 num_iters = 100;
  for (int32 iter = 0; iter < num_iters; iter++) {
    Matrix<BaseFloat> M_cpu(N, D);
    if (one) M_cpu.Set(1.0);
    else if (!zero) {
      M_cpu.SetRandn();
      Vector<BaseFloat> rand_vec(N);
      rand_vec.SetRandn();
      M_cpu.AddVecVec(1.0, rand_vec, big_eig_vector);
    }
    CuMatrix<BaseFloat> M(M_cpu);

    CuMatrix<BaseFloat> Mcopy1(M), Mcopy2(M);

    preconditioner1.PreconditionDirections(&Mcopy1, &row_prod1, &gamma1);

    preconditioner2.PreconditionDirections(&Mcopy2, &row_prod2, &gamma2);

    BaseFloat trace1 = TraceMatMat(M, M, kTrans),
        trace2 = TraceMatMat(Mcopy1, Mcopy1, kTrans);
    AssertEqual(trace1, trace2 * gamma2 * gamma2, 1.0e-02);

    AssertEqual(Mcopy1, Mcopy2);
    AssertEqual<BaseFloat>(row_prod1, row_prod2, 1.0e-02);
    AssertEqual(gamma1, gamma2, 1.0e-02);

    // make sure positive definite
    CuVector<BaseFloat> inner_prods(M.NumRows());
    inner_prods.AddDiagMatMat(1.0, M, kNoTrans, Mcopy1, kTrans, 0.0);
    KALDI_ASSERT(inner_prods.Min() >= 0.0);
  }
  return;
}


} // namespace nnet3
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
    for (int32 i = 0; i < 5; i++) {
      UnitTestPreconditionDirectionsOnline();
    }
  }
}

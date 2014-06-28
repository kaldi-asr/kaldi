// nnet2/nnet-precondition-online.cc

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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

namespace kaldi {
namespace nnet2 {




static void CheckOrthogonal(CuMatrixBase<BaseFloat> *N,
                            bool quiet = false,
                            int32 recurse_count = 0) {
  if (recurse_count > 100)
    KALDI_ERR << "CheckOrthogonal recursed >100 times, something is wrong.";
  
  int32 R = N->NumRows();
  CuSpMatrix<BaseFloat> S(R);
  S.AddMat2(1.0, *N, kNoTrans, 0.0);
  if (!S.IsUnit(1.0e-04)) {
    {
      SpMatrix<BaseFloat> S_cpu(S);
      if (!quiet)
        KALDI_WARN << "Matrix N is not orthogonal, fixing it.  N N^T is "
                   << S_cpu;
      Vector<BaseFloat> s(R);
      S_cpu.Eig(&s);
      BaseFloat threshold = 0.001;
      if (s.Min() < threshold) {
        if (!quiet) {
          KALDI_WARN << "Minimum eigenvalue of N N^T is less than " << threshold
                     << ", may be hard to fix: re-initializing from random "
                     << "start. Eigs are" << s;
        }
        N->SetRandn();
        CheckOrthogonal(N, quiet, recurse_count + 1);
        return;
      }
    }
    CuTpMatrix<BaseFloat> Cinv(R);
    Cinv.Cholesky(S);
    Cinv.Invert();
    CuMatrix<BaseFloat> N_copy(*N);
    N->AddTpMat(1.0, Cinv, kNoTrans, N_copy, kNoTrans, 0.0);
    CheckOrthogonal(N, quiet, recurse_count + 1); // Check that it worked.
  }
}

OnlinePreconditioner::OnlinePreconditioner():
    rank_(40), update_period_(1), num_samples_history_(2000.0), alpha_(4.0),
    epsilon_(1.0e-10), delta_(1.0e-05), t_(-1),
    num_updates_skipped_(0), self_debug_(false) { }

void OnlinePreconditioner::Init(const CuMatrixBase<BaseFloat> &R0) {
  int32 D = R0.NumCols(), N = R0.NumRows();
  KALDI_ASSERT(D > 1);
  if (rank_ >= D) {
    KALDI_WARN << "Rank " << rank_ << " of online preconditioner is >= dim " << D
               << ", setting it to "
               << (D - 1) << " (but this is probably still too high)";
    rank_ = D - 1;
  }
  KALDI_ASSERT(num_samples_history_ > 0.0 && num_samples_history_ <= 1.0e+6);
  KALDI_ASSERT(alpha_ >= 0.0);
  KALDI_ASSERT(rank_ > 0);
  KALDI_ASSERT(epsilon_ > 0.0 && epsilon_ <= 1.0e-05);  // plausible values.
  KALDI_ASSERT(delta_ > 0.0 && delta_ <= 1.0e-02);  // plausible values.
  
  int32 R = rank_;
  W_t_.Resize(R, D);
  d_t_.Resize(R);
  CuVector<BaseFloat> L(R);
  ApproxEigsOfProduct(R0, kTrans, &W_t_, &L);
  // want L to be eigenvalues of 1/N R0 R0^T
  L.Scale(1.0 / N);
  
  // \rho_0 = (1/N tr(R0 R0^T) - tr(L)) / (D - R)
  rho_t_ = (TraceMatMat(R0, R0, kTrans) / N - L.Sum()) / (D - R);
  BaseFloat floor_val = std::max(epsilon_, delta_ * L.Max());
  if (rho_t_ < floor_val)
    rho_t_ = floor_val;
  d_t_.CopyFromVec(L);
  d_t_.Add(-rho_t_);  // D_0 = L - \rho_0 I
  d_t_.ApplyFloor(epsilon_);
  
  // beta_t = \rho_t(1+\alpha) + \alpha/D tr(D_t)
  BaseFloat beta_t = rho_t_ * (1.0 + alpha_) + alpha_ * d_t_.Sum() / D;
  Vector<BaseFloat> e_t(R), sqrt_e_t(R), inv_sqrt_e_t(R);
  ComputeEt(d_t_, beta_t, &e_t, &sqrt_e_t, &inv_sqrt_e_t);
  // Compute W_0 by scaling the rows of X_0 with E_0^{0.5}.
  CuVector<BaseFloat> sqrt_e_t_gpu(sqrt_e_t);
  W_t_.MulRowsVec(sqrt_e_t_gpu);
  t_ = 0;
}

void OnlinePreconditioner::PreconditionDirections(
    CuMatrixBase<BaseFloat> *R_t,
    CuVectorBase<BaseFloat> *row_prod,
    BaseFloat *scale) {
  if (row_prod == NULL) {
    CuVector<BaseFloat> row_prod_tmp(R_t->NumRows());
    PreconditionDirections(R_t, &row_prod_tmp, scale);
    return;
  }
  
  read_write_mutex_.Lock();
  if (t_ == -1) // not initialized
    Init(*R_t);
  // Now t_ >= 0.
  // We create local copies  of the class variables... this is intended for
  // multi-threaded safety so we can't read them in an inconsistent state,
  // but we don't really waste anything here (a copy of W_t is needed anyway,
  // if we're to update it).
  int32 t = t_, R = W_t_.NumRows(), D = W_t_.NumCols();
  // space for W_t, J_t, K_t, L_t.
  CuMatrix<BaseFloat> WJKL_t(2 * R, D + R);
  WJKL_t.Range(0, R, 0, D).CopyFromMat(W_t_);
  BaseFloat rho_t(rho_t_);
  Vector<BaseFloat> d_t(d_t_);
  read_write_mutex_.Unlock();
  PreconditionDirectionsInternal(t, rho_t, d_t, &WJKL_t, R_t, row_prod, scale);
}

void OnlinePreconditioner::ReorthogonalizeXt1(
    const VectorBase<BaseFloat> &d_t1,
    BaseFloat rho_t1,
    CuMatrixBase<BaseFloat> *W_t1,
    CuMatrixBase<BaseFloat> *temp_W,
    CuMatrixBase<BaseFloat> *temp_O) {
  // threshold is a configuration value: a desired threshold on orthogonality,
  // below which we won't reorthogonalize.
  const BaseFloat threshold = 1.0e-03;

  int32 R = W_t1->NumRows(), D = W_t1->NumCols();
  BaseFloat beta_t1 = rho_t1 * (1.0 + alpha_) + alpha_ * d_t1.Sum() / D;
  Vector<BaseFloat> e_t1(R, kUndefined), sqrt_e_t1(R, kUndefined),
      inv_sqrt_e_t1(R, kUndefined);
  ComputeEt(d_t1, beta_t1, &e_t1, &sqrt_e_t1, &inv_sqrt_e_t1);  
  
  temp_O->SymAddMat2(1.0, *W_t1, kNoTrans, 0.0);
  // O_t =  E_t^{-0.5} W_t W_t^T E_t^{-0.5}  
  Matrix<BaseFloat> O_mat(*temp_O);
  SpMatrix<BaseFloat> O(O_mat, kTakeLower);
  for (int32 i = 0; i < R; i++) {
    BaseFloat i_factor = inv_sqrt_e_t1(i);
    for (int32 j = 0; j <= i; j++) {
      BaseFloat j_factor = inv_sqrt_e_t1(j);
      O(i, j) *= i_factor * j_factor;
    }
  }
  if (O.IsUnit(threshold)) {
    if (self_debug_) {
      KALDI_WARN << "Not reorthogonalizing since already orthognoal: " << O;
    }
    return;
  }
  TpMatrix<BaseFloat> C(R);
  try {
    C.Cholesky(O);
  } catch (...) {
    // It would be very strange to reach this point, but we try to handle it
    // gracefully anyway.
    KALDI_WARN << "Cholesky failed while re-orthogonalizing X_t. "
               << "Re-initializing as arbitrary orthogonal matrix.";
    // set X_t to [I; 0] which is orthogonal.
    W_t1->SetZero();
    W_t1->AddToDiag(1.0);
    // W_{t+1} = E_{t+1}^{0.5} X_{t+1}
    CuVector<BaseFloat> sqrt_e_t1_gpu(sqrt_e_t1);
    W_t1->MulRowsVec(sqrt_e_t1_gpu);
    return;
  }
  C.Invert();  // Now it's C^{-1}.
  // Next, compute (E_t^{0.5} C^{-1} E_t^{-0.5})
  // but it's really t+1, not t.
  for (int32 i = 0; i < R; i++) {
    BaseFloat i_factor = sqrt_e_t1(i);
    for (int32 j = 0; j < i; j++) {
      // skip j == i because i_factor * j_factor == 1 for j == i.
      BaseFloat j_factor = inv_sqrt_e_t1(j);
      C(i, j) *= i_factor * j_factor;
    }
  }
  O_mat.CopyFromTp(C);
  temp_O->CopyFromMat(O_mat);
  temp_W->CopyFromMat(*W_t1);
  W_t1->AddMatMat(1.0, *temp_O, kNoTrans, *temp_W, kNoTrans, 0.0);
}

void OnlinePreconditioner::PreconditionDirectionsInternal(
    const int32 t,
    const BaseFloat rho_t,
    const Vector<BaseFloat> &d_t,
    CuMatrixBase<BaseFloat> *WJKL_t,
    CuMatrixBase<BaseFloat> *R_t,
    CuVectorBase<BaseFloat> *row_prod,
    BaseFloat *scale) {
  int32 N = R_t->NumRows(),  // Minibatch size.
      D = R_t->NumCols(),  // Dimensions of vectors we're preconditioning
      R = rank_;  // Rank of correction to unit matrix.
  KALDI_ASSERT(R > 0 && R < D);
  BaseFloat eta = Eta(N);

  CuMatrix<BaseFloat> H_t(N, R);
  const CuSubMatrix<BaseFloat> W_t(*WJKL_t, 0, R, 0, D);
  CuSubMatrix<BaseFloat> J_t(*WJKL_t, R, R, 0, D),
      L_t(*WJKL_t, 0, R, D, R),
      K_t(*WJKL_t, R, R, D, R),
      WJ_t(*WJKL_t, 0, 2 * R, 0, D),
      LK_t(*WJKL_t, 0, 2 * R, D, R);
  
  H_t.AddMatMat(1.0, *R_t, kNoTrans, W_t, kTrans, 0.0);  // H_t = R_t W_t^T
  
  bool locked = update_mutex_.TryLock();
  if (locked) {
    // Just hard-code it here that we do 10 updates before skipping any.
    const int num_initial_updates = 10;
    if (t_ > t || (num_updates_skipped_ < update_period_ - 1 &&
                   t_ >= num_initial_updates)) {
      update_mutex_.Unlock();
      // We got the lock but we were already beaten to it by another thread, or
      // we don't want to update yet due to update_period_ > 1 (this saves
      // compute), so release the lock.
      locked = false;
    }
  }
  
  if (!locked) {
    // We're not updating the parameters, either because another thread is
    // working on updating them, or because another thread already did so from
    // the same or later starting point (making our update stale), or because
    // update_period_ > 1.  We just apply the preconditioning and return.

    // note: we don't bother with any locks before incrementing
    // num_updates_skipped_ below, because the worst that could happen is that,
    // on very rare occasions, we could skip one or two more updates than we
    // intended.
    num_updates_skipped_++;
    
    BaseFloat tr_Rt_RtT = TraceMatMat(*R_t, *R_t, kTrans);
    // P_t = R_t - H_t W_t
    R_t->AddMatMat(-1.0, H_t, kNoTrans, W_t, kNoTrans, 1.0); 
    // each element i of row_prod will be inner product of row i of P_t with
    // itself.
    row_prod->AddDiagMat2(1.0, *R_t, kNoTrans, 0.0);
    BaseFloat tr_Pt_PtT = row_prod->Sum();
    KALDI_ASSERT(tr_Pt_PtT == tr_Pt_PtT);  // Check for NaN.
    BaseFloat gamma_t = (tr_Pt_PtT == 0.0 ? 1.0 :
                         sqrt(tr_Rt_RtT / tr_Pt_PtT));
    *scale = gamma_t;
    return;
  }
  J_t.AddMatMat(1.0, H_t, kTrans, *R_t, kNoTrans, 0.0);  // J_t = H_t^T R_t

  bool compute_lk_together = (N > D);
  
  if (compute_lk_together) {
    // do the following two multiplies in one operation...
    // note
    // L_t = W_t J_t^T
    // K_t = J_t J_t^T
    // Note: L_t was defined as L_t = J_t W_t^T, but it's actually symmetric,
    // so we can compute it as L_t = W_t J_t^T.
    LK_t.AddMatMat(1.0, WJ_t, kNoTrans, J_t, kTrans, 0.0);
  } else {
    K_t.SymAddMat2(1.0, J_t, kNoTrans, 0.0);
    L_t.SymAddMat2(1.0, H_t, kTrans, 0.0);
  }

  Matrix<BaseFloat> LK_cpu(LK_t);  // contains L and K on the CPU.
  SubMatrix<BaseFloat> L_t_cpu(LK_cpu, 0, R, 0, R),
      K_t_cpu(LK_cpu, R, R, 0, R);
  if (!compute_lk_together) {
    // the SymAddMat2 operations only set the lower triangle and diagonal.
    L_t_cpu.CopyLowerToUpper();
    K_t_cpu.CopyLowerToUpper();
  }

  // beta_t = \rho_t(1+\alpha) + \alpha/D tr(D_t)
  BaseFloat beta_t = rho_t * (1.0 + alpha_) + alpha_ * d_t.Sum() / D;
  Vector<BaseFloat> e_t(R), sqrt_e_t(R), inv_sqrt_e_t(R);
  ComputeEt(d_t, beta_t, &e_t, &sqrt_e_t, &inv_sqrt_e_t);
  KALDI_VLOG(5) << "e_t = " << e_t;
  
  SpMatrix<BaseFloat> Z_t(R);
  ComputeZt(N, rho_t, d_t, inv_sqrt_e_t, K_t_cpu, L_t_cpu, &Z_t);

  Matrix<BaseFloat> U_t(R, R);
  Vector<BaseFloat> c_t(R);
  // do the symmetric eigenvalue decomposition Z_t = U_t C_t U_t^T.
  Z_t.Eig(&c_t, &U_t);
  SortSvd(&c_t, &U_t);

  const BaseFloat condition_threshold = 1.0e+06;
  // must_reorthogonalize will be true if the last diagonal element of c_t is
  // negative, since we don't take the absolute value, but this is the right
  // thing anyway.
  bool must_reorthogonalize = (c_t(0) > condition_threshold * c_t(R - 1));
  
  BaseFloat c_t_floor = pow(rho_t * (1 - eta), 2);
  int32 nf = c_t.ApplyFloor(c_t_floor);
  if (nf > 0)
    must_reorthogonalize = true;
  if (nf > 0 && self_debug_) {
    KALDI_WARN << "Floored " << nf << " elements of C_t.";
  }
  BaseFloat tr_Rt_RtT_check;
  if (self_debug_)
    tr_Rt_RtT_check = TraceMatMat(*R_t, *R_t, kTrans);
  
  R_t->AddMatMat(-1.0, H_t, kNoTrans, W_t, kNoTrans, 1.0);  // P_t = R_t - H_t W_t
  // set *row_prod to inner products of each row of P_t with itself.
  row_prod->AddDiagMat2(1.0, *R_t, kNoTrans, 0.0);

  BaseFloat tr_Pt_PtT = row_prod->Sum();
  //  tr(R_t R_t^T) = tr(P_t P_t^T) - tr(L_t E_t) + 2 tr(L_t)  
  double tr_Rt_RtT = tr_Pt_PtT;
  for (int32 i = 0; i < R; i++)
    tr_Rt_RtT += L_t_cpu(i, i) * (2.0 - e_t(i));
  if (self_debug_) {
    KALDI_ASSERT(ApproxEqual(tr_Rt_RtT, tr_Rt_RtT_check));
  }
  BaseFloat gamma_t = (tr_Pt_PtT == 0.0 ? 1.0 :
                       sqrt(tr_Rt_RtT / tr_Pt_PtT));
  *scale = gamma_t;

  Vector<BaseFloat> sqrt_c_t(c_t);
  sqrt_c_t.ApplyPow(0.5);
  
  // \rho_{t+1} = 1/(D - R) (\eta/N tr(R_t R_t^T) + (1-\eta)(D \rho_t + tr(D_t)) - tr(C_t^{0.5})).  
  BaseFloat rho_t1 = 1.0 / (D - R) * (eta / N * tr_Rt_RtT
                                      + (1-eta)*(D * rho_t + d_t.Sum())
                                      - sqrt_c_t.Sum());
  // D_{t+1} = C_t^{0.5} - \rho_{t+1} I
  Vector<BaseFloat> d_t1(sqrt_c_t);
  d_t1.Add(-rho_t1);
  BaseFloat floor_val = std::max(epsilon_, delta_ * sqrt_c_t.Max());
  if (rho_t1 < floor_val)
    rho_t1 = floor_val;
  d_t1.ApplyFloor(epsilon_);

  CuMatrix<BaseFloat> W_t1(R, D);  // W_{t+1}
  ComputeWt1(N, d_t, d_t1, rho_t, rho_t1, U_t, sqrt_c_t, inv_sqrt_e_t,
             W_t, &J_t, &W_t1);

  if (must_reorthogonalize) {
    if (self_debug_) {
      KALDI_WARN << "Reorthogonalizing.";
    }
    ReorthogonalizeXt1(d_t1,
                       rho_t1,
                       &W_t1,
                       &J_t,
                       &L_t);
  }


  // Commit the new parameters.
  read_write_mutex_.Lock();
  KALDI_ASSERT(t_ == t);  // we already ensured this.
  t_ = t + 1;
  num_updates_skipped_ = 0;
  W_t_.Swap(&W_t1);
  d_t_.CopyFromVec(d_t1);
  rho_t_ = rho_t1;
  
  read_write_mutex_.Unlock();
  update_mutex_.Unlock();
}

BaseFloat OnlinePreconditioner::Eta(int32 N) const {
  KALDI_ASSERT(num_samples_history_ > 0.0);
  return 1.0 - exp(-N / num_samples_history_);
}

void OnlinePreconditioner::ComputeWt1(int32 N,
                                      const VectorBase<BaseFloat> &d_t,
                                      const VectorBase<BaseFloat> &d_t1,
                                      BaseFloat rho_t,
                                      BaseFloat rho_t1,
                                      const MatrixBase<BaseFloat> &U_t,
                                      const VectorBase<BaseFloat> &sqrt_c_t,
                                      const VectorBase<BaseFloat> &inv_sqrt_e_t,                                      
                                      const CuMatrixBase<BaseFloat> &W_t,
                                      CuMatrixBase<BaseFloat> *J_t,
                                      CuMatrixBase<BaseFloat> *W_t1) const {
  
  int32 R = d_t.Dim(), D = W_t.NumCols();
  BaseFloat eta = Eta(N);

  // \beta_{t+1} = \rho_{t+1} (1+\alpha) + \alpha/D tr(D_{t+1})
  BaseFloat beta_t1 = rho_t1 * (1.0 + alpha_) + alpha_ * d_t1.Sum() / D;
  KALDI_ASSERT(beta_t1 > 0.0);
  Vector<BaseFloat> e_t1(R, kUndefined), sqrt_e_t1(R, kUndefined),
      inv_sqrt_e_t1(R, kUndefined);
  ComputeEt(d_t1, beta_t1, &e_t1, &sqrt_e_t1, &inv_sqrt_e_t1);
  Vector<BaseFloat> inv_sqrt_c_t(sqrt_c_t);
  inv_sqrt_c_t.InvertElements();
  
  Vector<BaseFloat> w_t_coeff(R);
  for (int32 i = 0; i < R; i++)
    w_t_coeff(i) = (1.0 - eta) / (eta/N) * (d_t(i) + rho_t);
  CuVector<BaseFloat> w_t_coeff_gpu(w_t_coeff);
  // B_t = J_t + (1-\eta)/(\eta/N) (D_t + \rho_t I) W_t
  J_t->AddDiagVecMat(1.0, w_t_coeff_gpu, W_t, kNoTrans, 1.0);

  // A_t = (\eta/N) E_{t+1}^{0.5} C_t^{-0.5} U_t^T E_t^{-0.5} B_t
  Matrix<BaseFloat> A_t(U_t, kTrans);
  for (int32 i = 0; i < R; i++) {
    BaseFloat i_factor = (eta / N) * sqrt_e_t1(i) * inv_sqrt_c_t(i);
    for (int32 j = 0; j < R; j++) {
      BaseFloat j_factor = inv_sqrt_e_t(j);
      A_t(i, j) *= i_factor * j_factor;
    }
  }
  // W_{t+1} = A_t B_t
  CuMatrix<BaseFloat> A_t_gpu(A_t);
  W_t1->AddMatMat(1.0, A_t_gpu, kNoTrans, *J_t, kNoTrans, 0.0);

  if (self_debug_) {
    CuMatrix<BaseFloat> W_t1_prod(R, R);
    W_t1_prod.SymAddMat2(1.0, *W_t1, kNoTrans, 0.0);
    W_t1_prod.CopyLowerToUpper();
    Matrix<BaseFloat> W_t1_prod_cpu(W_t1_prod);
    // Verifying that W_{t+1} W_{t+1}^T == E_t, via
    // E_{-0.5} W_{t+1} W_{t+1}^T E_{-0.5} == I.
    for (int32 i = 0; i < R; i++)
      for (int32 j = 0; j < R; j++)
        W_t1_prod_cpu(i, j) *= inv_sqrt_e_t1(i) * inv_sqrt_e_t1(j);
    for (int32 i = 0; i < R; i++) {
      for (int32 j = 0; j < R; j++) {
        BaseFloat elem = W_t1_prod_cpu(i, j);
        if ((i == j && fabs(elem - 1.0) > 0.1) ||
            (i != j && fabs(elem) > 1.0e-02)) {
          KALDI_WARN << "Failed to verify W_{t+1}, the following should be unit: "
                     << W_t1_prod_cpu;
        }
      }
    }
  }
}

void OnlinePreconditioner::ComputeZt(int32 N,
                                     BaseFloat rho_t,
                                     const VectorBase<BaseFloat> &d_t,
                                     const VectorBase<BaseFloat> &inv_sqrt_e_t,
                                     const MatrixBase<BaseFloat> &K_t,
                                     const MatrixBase<BaseFloat> &L_t,
                                     SpMatrix<BaseFloat> *Z_t) const {
  BaseFloat eta = Eta(N);
  Vector<BaseFloat> d_t_rho_t(d_t);
  d_t_rho_t.Add(rho_t);  // now d_t_rho_t is diag(D_t + \rho_t I).
  BaseFloat etaN = eta / N, eta1 = 1.0 - eta,
      etaN_sq = etaN * etaN, eta1_sq = eta1 * eta1,
      etaN_eta1 = etaN * eta1;
  int32 R = d_t.Dim();
  for (int32 i = 0; i < R; i++) {
    BaseFloat inv_sqrt_e_t_i = inv_sqrt_e_t(i), d_t_rho_t_i = d_t_rho_t(i);
    for (int32 j = 0; j <= i; j++) {
      BaseFloat inv_sqrt_e_t_j = inv_sqrt_e_t(j), d_t_rho_t_j = d_t_rho_t(j),
          L_t_i_j = 0.5 * (L_t(i, j) + L_t(j, i)),
          K_t_i_j = 0.5 * (K_t(i, j) + K_t(j, i));
      // See (eqn:Zt) in header.
      (*Z_t)(i, j) = etaN_sq * inv_sqrt_e_t_i * K_t_i_j * inv_sqrt_e_t_j
          + etaN_eta1 * inv_sqrt_e_t_i * L_t_i_j * inv_sqrt_e_t_j * d_t_rho_t_j
          + etaN_eta1 * d_t_rho_t_i * inv_sqrt_e_t_i * L_t_i_j * inv_sqrt_e_t_j
          + (i == j ? eta1_sq * d_t_rho_t_i * d_t_rho_t_i : 0.0);
    }
  }
}

void OnlinePreconditioner::ComputeEt(const VectorBase<BaseFloat> &d_t,
                                     BaseFloat beta_t,
                                     VectorBase<BaseFloat> *e_t,
                                     VectorBase<BaseFloat> *sqrt_e_t,
                                     VectorBase<BaseFloat> *inv_sqrt_e_t) const {
  // e_{tii} = 1/(\beta_t/d_{tii} + 1)
  int32 D = d_t.Dim();
  const BaseFloat *d = d_t.Data();
  BaseFloat *e = e_t->Data();
  for (int32 i = 0; i < D; i++)
    e[i] = 1.0 / (beta_t / d[i]  +  1);
  sqrt_e_t->CopyFromVec(*e_t);
  sqrt_e_t->ApplyPow(0.5);
  inv_sqrt_e_t->CopyFromVec(*sqrt_e_t);
  inv_sqrt_e_t->InvertElements();
}


/**
   I'm not very satisfied with the implementation of this function, but a
   careful GPU-oriented version would take a while to do correctly, mainly due
   to the necessity to implement orthogonalization of a matrix where the matrix
   might have a reduced rank and we might have to "complete" it with random
   rows.  Anyway, in the current implementation we just move an inner-product
   matrix to the CPU and compute the approximate top eigenvalues there.
 */
void ApproxEigsOfProduct(const CuMatrixBase<BaseFloat> &M,
                         MatrixTransposeType trans,
                         CuMatrixBase<BaseFloat> *P,
                         CuVectorBase<BaseFloat> *s) {
  int32 R = P->NumRows(), D = P->NumCols();
  
  // First make sure, for simplicity, that trans == kNoTrans.
  if (trans == kTrans) {
    CuMatrix<BaseFloat> M_trans(M, kTrans);
    ApproxEigsOfProduct(M_trans, kNoTrans, P, s);
    return;
  }
  // Next, make sure we can handle the case when the number of requested
  // eigenvalues is more than smaller of (#columns/#rows)... this makes sense
  // in a situation where we are asked for a number eigenvalues of R R^T that
  // is greater than the #cols of R.  The remaining eigenvectors should be zero.
  if (R > std::min(M.NumRows(), M.NumCols())) {
    KALDI_ASSERT(R <= D);
    int32 R_tmp = std::min(M.NumRows(), M.NumCols());
    CuSubMatrix<BaseFloat> P_part(*P, 0, R_tmp, 0, D);
    CuSubVector<BaseFloat> s_part(*s, 0, R_tmp);
    s->SetZero();
    ApproxEigsOfProduct(M, trans, &P_part, &s_part);
    Matrix<BaseFloat> P_cpu(*P);
    P_cpu.OrthogonalizeRows();  // Will fill the remaining rows of P_cpu with
                                // random vectors and ensure P P^T = I.
    P->CopyFromMat(P_cpu);
    return;
  }
  
  KALDI_ASSERT(R <= D && R > 0 && s->Dim() == R);
  if (trans == kNoTrans) {
    KALDI_ASSERT(D == M.NumRows());
  } else {
    KALDI_ASSERT(D == M.NumCols());
  }

  if (M.NumRows() < M.NumCols()) {
    // Quicker to compute eigenvalues of M M^T
    CuMatrix<BaseFloat> MMT(M.NumRows(), M.NumRows());
    MMT.SymAddMat2(1.0, M, kNoTrans, 0.0);
    CuSpMatrix<BaseFloat> MMT_sp(MMT, kTakeLower);
    SpMatrix<BaseFloat> MMT_cpu(MMT_sp);

    Vector<BaseFloat> s_cpu(R);
    Matrix<BaseFloat> P_cpu(D, R);  // It's actually the columns of P that are
                                    // the eigenvectors.
    // Uses default configuration to get top eigenvalues approximately.
    MMT_cpu.TopEigs(&s_cpu, &P_cpu);  
    P->CopyFromMat(P_cpu, kTrans);
    s->CopyFromVec(s_cpu);
  } else {
    // Quicker to compute eigenvalues of M^T M
    int32 D = M.NumCols();
    CuMatrix<BaseFloat> MTM(D, D);
    MTM.SymAddMat2(1.0, M, kTrans, 0.0);
    CuSpMatrix<BaseFloat> MTM_sp(MTM, kTakeLower);
    SpMatrix<BaseFloat> MTM_cpu(MTM_sp);

    Vector<BaseFloat> s_cpu(R);
    Matrix<BaseFloat> Q_cpu(D, R);  // It's actually the columns of Q that are
                                    // the eigenvectors.
    MTM_cpu.TopEigs(&s_cpu, &Q_cpu);  // Uses default configuration.
    
    // OK, suppose we have some eigenvector v, so M^T M v = \lambda v.  Define w
    // = M v.  Then M M^T M v = M (M^T M v) = M (\lambda v) = \lambda M M^T w.
    // Then w = M v is also an eigenvector of M M^T, with the same eigenvalue
    // \lambda.
    // However, we might have a problem if M v == 0 (this is only possible if
    // some eigenvalues are zero); in this case we won't be able to renormalize
    // w to have unit norm.  We'll let OrthogonalizeRows() take care of that,
    // though.  Anyway, just to avoid having to think about it to hard,
    // we'll recompute the eigenvalues after computing P = Q^T M^T below
    // and orthogonalizing its rows.
    // Note: the tranpose on Q in the above equation is because our Q_cpu
    // has its columns, not rows, as the eigenvectors.

    Matrix<BaseFloat> P_cpu(R, M.NumRows());
    Matrix<BaseFloat> M_cpu(M);
    P_cpu.AddMatMat(1.0, Q_cpu, kTrans, M_cpu, kTrans, 0.0);
    P_cpu.OrthogonalizeRows();
    P->CopyFromMat(P_cpu);

    // we will set s according to diag(s) = P M M^T P^T,
    // which we can get by computing P M, and doing AddDiagMat2
    CuMatrix<BaseFloat> PM(R, M.NumCols());
    PM.AddMatMat(1.0, *P, kNoTrans, M, kNoTrans, 0.0);
    s->SetZero();  // In case it had NaN's in it.
    s->AddDiagMat2(1.0, PM, kNoTrans, 0.0);
  }
}

OnlinePreconditioner::OnlinePreconditioner(const OnlinePreconditioner &other):
    rank_(other.rank_), update_period_(other.update_period_),
    num_samples_history_(other.num_samples_history_),
    alpha_(other.alpha_), epsilon_(other.epsilon_), delta_(other.delta_),
    t_(other.t_), num_updates_skipped_(other.num_updates_skipped_),
    self_debug_(other.self_debug_), W_t_(other.W_t_),
    rho_t_(other.rho_t_), d_t_(other.d_t_) {
  // use default constructor for the mutextes.
}

OnlinePreconditioner& OnlinePreconditioner::operator = (
    const OnlinePreconditioner &other) {
  rank_ = other.rank_;
  update_period_ = other.update_period_;
  num_samples_history_ = other.num_samples_history_;
  alpha_ = other.alpha_;
  epsilon_ = other.epsilon_;
  t_ = other.t_;
  self_debug_ = other.self_debug_;
  W_t_ = other.W_t_;
  rho_t_ = other.rho_t_;
  d_t_ = other.d_t_;
  return *this;
}

void OnlinePreconditioner::SetRank(int32 rank) {
  KALDI_ASSERT(rank > 0);
  rank_ = rank;  
}
void OnlinePreconditioner::SetUpdatePeriod(int32 update_period) {
  KALDI_ASSERT(update_period > 0);
  update_period_ = update_period;
}
void OnlinePreconditioner::SetNumSamplesHistory(BaseFloat num_samples_history) {
  KALDI_ASSERT(num_samples_history > 0.0 &&
               num_samples_history < 1.0e+6);
  num_samples_history_ = num_samples_history;
}
void OnlinePreconditioner::SetAlpha(BaseFloat alpha) {
  KALDI_ASSERT(alpha >= 0.0);
  alpha_ = alpha;
}


}
}

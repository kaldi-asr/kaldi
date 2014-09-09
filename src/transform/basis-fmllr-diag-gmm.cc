// transform/basis-fmllr-diag-gmm.cc

// Copyright 2012  Carnegie Mellon University (author: Yajie Miao)
//           2014  Johns Hopkins University (author: Daniel Povey)
//           2014  IMSL, PKU-HKUST (Author: Wei Shi)

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

#include <algorithm>
#include <utility>
#include <vector>
using std::vector;
#include <string>
using std::string;

#include "transform/fmllr-diag-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "transform/basis-fmllr-diag-gmm.h"

namespace kaldi {


/// This function takes the step direction (delta) of fMLLR matrix as argument,
/// and optimize step size using Newton's method. This is an iterative method,
/// where each iteration should not decrease the auxiliary function. Note that
/// the resulting step size \k should be close to 1. If \k <<1 or >>1, there
/// maybe problems with preconditioning or the speaker stats.
static BaseFloat CalBasisFmllrStepSize(
    const AffineXformStats &spk_stats,
    const Matrix<BaseFloat> &spk_stats_tmp_K,
    const std::vector<SpMatrix<BaseFloat> > &spk_stats_tmp_G,
    const Matrix<BaseFloat> &delta,
    const Matrix<BaseFloat> &A,
    const Matrix<BaseFloat> &S,
    int32 max_iters);


void BasisFmllrAccus::Write(std::ostream &os, bool binary) const {

  WriteToken(os, binary, "<BASISFMLLRACCUS>");
  WriteToken(os, binary, "<BETA>");
  WriteBasicType(os, binary, beta_);
  if (!binary) os << '\n';
  if (grad_scatter_.NumCols() != 0) {
    WriteToken(os, binary, "<GRADSCATTER>");
    grad_scatter_.Write(os, binary);
  }
  WriteToken(os, binary, "</BASISFMLLRACCUS>");
}

void BasisFmllrAccus::Read(std::istream &is, bool binary,
                           bool add) {
  ExpectToken(is, binary, "<BASISFMLLRACCUS>");
  ExpectToken(is, binary, "<BETA>");
  double tmp_beta = 0;
  ReadBasicType(is, binary, &tmp_beta);
  if (add) {
    beta_ += tmp_beta;
  } else {
    beta_ = tmp_beta;
  }
  ExpectToken(is, binary, "<GRADSCATTER>");
  grad_scatter_.Read(is, binary, add);
  ExpectToken(is, binary, "</BASISFMLLRACCUS>");
}

void BasisFmllrAccus::ResizeAccus(int32 dim) {
  if (dim <= 0) {
    KALDI_ERR << "Invalid feature dimension " << dim; // dim=0 is not allowed
  } else {
    // 'kSetZero' may not be necessary, but makes computation safe
    grad_scatter_.Resize((dim + 1) * dim, kSetZero);
  }
}

void BasisFmllrAccus::AccuGradientScatter(
                      const AffineXformStats &spk_stats) {

  // Gradient of auxf w.r.t. xform_spk
  // Eq. (33)
  Matrix<double> grad_mat(dim_, dim_ + 1);
  grad_mat.SetUnit();
  grad_mat.Scale(spk_stats.beta_);
  grad_mat.AddMat(1.0, spk_stats.K_);
  for (int d = 0; d < dim_; ++d) {
      Matrix<double> G_d_mat(spk_stats.G_[d]);
       grad_mat.Row(d).AddVec(-1.0, G_d_mat.Row(d));
  }
  // Row stack of gradient matrix
  Vector<BaseFloat> grad_vec((dim_+1) * dim_);
  grad_vec.CopyRowsFromMat(grad_mat);
  // The amount of data beta_ is likely to be ZERO, especially
  // when silence-weight is set to be 0 and we are using the
  // per-utt mode.
  if (spk_stats.beta_ > 0) {
    beta_ += spk_stats.beta_;
    grad_scatter_.AddVec2(BaseFloat(1.0 / spk_stats.beta_), grad_vec);
  }
}

void BasisFmllrEstimate::Write(std::ostream &os, bool binary) const {
  uint32 tmp_uint32;

  WriteToken(os, binary, "<BASISFMLLRPARAM>");

  WriteToken(os, binary, "<NUMBASIS>");
  tmp_uint32 = static_cast<uint32>(basis_size_);
  WriteBasicType(os, binary, tmp_uint32);
  if (fmllr_basis_.size() != 0) {
    WriteToken(os, binary, "<BASIS>");
    for (int32 n = 0; n < basis_size_; ++n) {
      fmllr_basis_[n].Write(os, binary);
    }
  }
  WriteToken(os, binary, "</BASISFMLLRPARAM>");
}

void BasisFmllrEstimate::Read(std::istream &is, bool binary) {
  uint32 tmp_uint32;
  string token;

  ExpectToken(is, binary, "<BASISFMLLRPARAM>");

  ExpectToken(is, binary, "<NUMBASIS>");
  ReadBasicType(is, binary, &tmp_uint32);
  basis_size_ = static_cast<int32>(tmp_uint32);
  KALDI_ASSERT(basis_size_ > 0);
  ExpectToken(is, binary, "<BASIS>");
  fmllr_basis_.resize(basis_size_);
  for (int32 n = 0; n < basis_size_; ++n) {
    fmllr_basis_[n].Read(is, binary);
    if (n == 0)
      dim_ = fmllr_basis_[n].NumRows();
    else {
      KALDI_ASSERT(dim_ == fmllr_basis_[n].NumRows());
    }
  }
  ExpectToken(is, binary, "</BASISFMLLRPARAM>");
}

void BasisFmllrEstimate::ComputeAmDiagPrecond(const AmDiagGmm &am_gmm,
                                              SpMatrix<double> *pre_cond) {
  KALDI_ASSERT(am_gmm.Dim() == dim_);
  if (pre_cond->NumRows() != (dim_ + 1) * dim_)
    pre_cond->Resize((dim_ + 1) * dim_, kSetZero);

  int32 num_pdf = am_gmm.NumPdfs();
  Matrix<double> H_mat((dim_ + 1) * dim_, (dim_ + 1) * dim_);
  // expected values of fMLLR G statistics
  vector< SpMatrix<double> > G_hat(dim_);
  for (int32 d = 0; d < dim_; ++d)
       G_hat[d].Resize(dim_ + 1, kSetZero);

  // extend mean vectors with 1  [mule_jm 1]
  Vector<double> extend_mean(dim_ + 1);
  // extend covariance matrix with a row and column of 0
  Vector<double> extend_var(dim_ + 1);
  for (int32 j = 0; j < num_pdf; ++j) {
    const DiagGmm &diag_gmm = am_gmm.GetPdf(j);
    int32 num_comp = diag_gmm.NumGauss();
    // means, covariance and mixture weights for this diagonal GMM
    Matrix<double> means(num_comp, dim_);
    Matrix<double> vars(num_comp, dim_);
    diag_gmm.GetMeans(&means); diag_gmm.GetVars(&vars);
    Vector<BaseFloat> weights(diag_gmm.weights());

    for (int32 m = 0; m < num_comp; ++m) {
      extend_mean.Range(0, dim_).CopyFromVec(means.Row(m));
      extend_mean(dim_) = 1.0;
      extend_var.Range(0, dim_).CopyFromVec(vars.Row(m));
      extend_var(dim_) = 0;
      // loop over feature dimension
      // Eq. (28): G_hat {d} = \sum_{j, m} P_{j}{m} Inv_Sigma{j, m, d}
      // (mule_extend mule_extend^T + Sigma_extend)
      // where P_{j}{m} = P_{j} c_{j}{m}
      for (int32 d = 0; d < dim_; ++d) {
        double alpha = (1.0 / num_pdf) * weights(m) * (1.0 / vars.Row(m)(d));
        G_hat[d].AddVec2(alpha, extend_mean);
        // add vector to the diagonal elements of the matrix
        // not work for full covariance matrices
        G_hat[d].AddDiagVec(alpha, extend_var);
      } // loop over dimension
    } //  loop over Gaussians
  }  // loop over states

  // fill H_ with G_hat[i]; build the block diagonal structure
  // Eq. (31)
  for (int32 d = 0; d < dim_; d++) {
    H_mat.Range(d * (dim_ + 1), (dim_ + 1), d * (dim_ + 1), (dim_ + 1))
              .CopyFromSp(G_hat[d]);
  }

  // add the extra H(1) elements
  // Eq. (30) and Footnote 1 (0-based index)
  for (int32 i = 0; i < dim_; ++i)
    for (int32 j = 0; j < dim_; ++j)
      H_mat(i * (dim_ + 1) + j, j * (dim_ + 1) + i) += 1;
  // the final H should be symmetric
  if (!H_mat.IsSymmetric())
    KALDI_ERR << "Preconditioner matrix H = H(1) + H(2) is not symmetric";
  pre_cond->CopyFromMat(H_mat, kTakeLower);
}

void BasisFmllrEstimate::EstimateFmllrBasis(
                              const AmDiagGmm &am_gmm,
                              const BasisFmllrAccus &basis_accus) {
  // Compute the preconditioner
  SpMatrix<double> precond_mat((dim_ + 1) * dim_);
  ComputeAmDiagPrecond(am_gmm, &precond_mat);
  // H = C C^T
  TpMatrix<double> C((dim_+1) * dim_);
  C.Cholesky(precond_mat);
  TpMatrix<double> C_inv(C);
  C_inv.InvertDouble();
  // From TpMatrix to Matrix
  Matrix<double> C_inv_full((dim_ + 1) * dim_, (dim_ + 1) * dim_);
  C_inv_full.CopyFromTp(C_inv);

  // Convert to the preconditioned coordinates
  // Eq. (35)  M_hat = C^{-1} grad_scatter C^{-T}
  SpMatrix<double> M_hat((dim_ + 1) * dim_);
  {
    SpMatrix<double> grad_scatter_d(basis_accus.grad_scatter_);
    M_hat.AddMat2Sp(1.0, C_inv_full, kNoTrans, grad_scatter_d, 0.0);
  }
  Vector<double> Lvec((dim_ + 1) * dim_);
  Matrix<double> U((dim_ + 1) * dim_, (dim_ + 1) * dim_);
  // SVD of M_hat; sort eigenvalues from greatest to smallest
  M_hat.SymPosSemiDefEig(&Lvec, &U);
  SortSvd(&Lvec, &U);
  // After transpose, each row is one base
  U.Transpose();

  fmllr_basis_.resize(basis_size_);
  for (int32 n = 0; n < basis_size_; ++n) {
    fmllr_basis_[n].Resize(dim_, dim_ + 1, kSetZero);
    Vector<double> basis_vec((dim_ + 1) * dim_);
    // Convert eigenvectors back to unnormalized space
    basis_vec.AddMatVec(1.0, C_inv_full, kTrans, U.Row(n), 0.0);
    // Convert stacked vectors to matrix
    fmllr_basis_[n].CopyRowsFromVec(basis_vec);
  }
  // Output the eigenvalues of the gradient scatter matrix
  // The eigenvalues are divided by twice the number of frames
  // in the training data, to get the per-frame values.
  Vector<double> Lvec_scaled(Lvec);
  Lvec_scaled.Scale(1.0 / (2 * basis_accus.beta_));
  KALDI_LOG << "The [per-frame] eigenvalues sorted from largest to smallest: " << Lvec_scaled;
  /// The sum of the [per-frame] eigenvalues is roughly equal to
  /// the improvement of log-likelihood of the training data.
  KALDI_LOG << "Sum of the [per-frame] eigenvalues, that is"
          " the log-likelihood improvement, is " << Lvec_scaled.Sum();
}

double BasisFmllrEstimate::ComputeTransform(
    const AffineXformStats &spk_stats,
    Matrix<BaseFloat> *out_xform,
    Vector<BaseFloat> *coefficient,
    BasisFmllrOptions options) const {
  if (coefficient == NULL) {
    Vector<BaseFloat> tmp;
    return ComputeTransform(spk_stats, out_xform, &tmp, options);
  }
  KALDI_ASSERT(dim_ == spk_stats.dim_);
  if (spk_stats.beta_ < options.min_count) {
    KALDI_WARN << "Not updating fMLLR since count is below min-count: "
               << spk_stats.beta_;
    coefficient->Resize(0);
    return 0.0;
  } else {
    if (out_xform->NumRows() != dim_ || out_xform->NumCols() != (dim_ +1)) {
      out_xform->Resize(dim_, dim_ + 1, kSetZero);
    }
    // Initialized either as [I;0] or as the current transform
    Matrix<BaseFloat> W_mat(dim_, dim_ + 1);
    if (out_xform->IsZero()) {
      W_mat.SetUnit();
    } else {
      W_mat.CopyFromMat(*out_xform);
    }

    // Create temporary K and G quantities. Add for efficiency,
    // avoid repetitions of converting the stats from double
    // precision to single precision
    Matrix<BaseFloat> stats_tmp_K(spk_stats.K_);
    std::vector<SpMatrix<BaseFloat> > stats_tmp_G(dim_);
    for (int32 d = 0; d < dim_; d++)
      stats_tmp_G[d] = SpMatrix<BaseFloat>(spk_stats.G_[d]);

    // Number of bases for this speaker, according to the available
    // adaptation data
    int32 basis_size = int32 (std::min( double(basis_size_),
                               options.size_scale * spk_stats.beta_));

    coefficient->Resize(basis_size, kSetZero);

    BaseFloat impr_spk = 0;
    for (int32 iter = 1; iter <= options.num_iters; ++iter) {
      // Auxf computation based on FmllrAuxFuncDiagGmm from fmllr-diag-gmm.cc
      BaseFloat start_obj = FmllrAuxFuncDiagGmm(W_mat, spk_stats);

      // Contribution of quadratic terms to derivative
      // Eq. (37)  s_{d} = G_{d} w_{d}
      Matrix<BaseFloat> S(dim_, dim_ + 1);
      for (int32 d = 0; d < dim_; ++d)
        S.Row(d).AddSpVec(1.0, stats_tmp_G[d], W_mat.Row(d), 0.0);


      // W_mat = [A; b]
      Matrix<BaseFloat> A(dim_, dim_);
      A.CopyFromMat(W_mat.Range(0, dim_, 0, dim_));
      Matrix<BaseFloat> A_inv(A);
      A_inv.InvertDouble();
      Matrix<BaseFloat> A_inv_trans(A_inv);
      A_inv_trans.Transpose();
      // Compute gradient of auxf w.r.t. W_mat
      // Eq. (38)  P = beta [A^{-T}; 0] + K - S
      Matrix<BaseFloat> P(dim_, dim_ + 1);
      P.SetZero();
      P.Range(0, dim_, 0, dim_).CopyFromMat(A_inv_trans);
      P.Scale(spk_stats.beta_);
      P.AddMat(1.0, stats_tmp_K);
      P.AddMat(-1.0, S);

      // Compute directional gradient restricted by bases. Here we only use
      // the simple gradient method, rather than conjugate gradient. Finding
      // the optimal transformation W_mat is equivalent to optimizing weights
      // d_{1,2,...,N}.
      // Eq. (39)  delta(W) = \sum_n tr(\fmllr_basis_{n}^T \P) \fmllr_basis_{n}
      // delta(d_{n}) = tr(\fmllr_basis_{n}^T \P)
      Matrix<BaseFloat> delta_W(dim_, dim_ + 1);
      Vector<BaseFloat> delta_d(basis_size);
      for (int32 n = 0; n < basis_size; ++n) {
        delta_d(n) = TraceMatMat(fmllr_basis_[n], P, kTrans);
        delta_W.AddMat(delta_d(n), fmllr_basis_[n]);
      }

      BaseFloat step_size = CalBasisFmllrStepSize(spk_stats, stats_tmp_K,
        stats_tmp_G, delta_W, A, S, options.step_size_iters);
      W_mat.AddMat(step_size, delta_W, kNoTrans);
      coefficient->AddVec(step_size, delta_d);
      // Check auxiliary function
      BaseFloat end_obj = FmllrAuxFuncDiagGmm(W_mat, spk_stats);

      KALDI_VLOG(4) << "Objective function (iter=" << iter << "): "
                    << start_obj / spk_stats.beta_  << " -> "
                    << (end_obj / spk_stats.beta_) << " over "
                    << spk_stats.beta_ << " frames";

      impr_spk += (end_obj - start_obj);
    }  // loop over iters

    out_xform->CopyFromMat(W_mat, kNoTrans);
    return impr_spk;
  }
}

// static
BaseFloat CalBasisFmllrStepSize(const AffineXformStats &spk_stats,
  const Matrix<BaseFloat> &spk_stats_tmp_K,
  const std::vector<SpMatrix<BaseFloat> > &spk_stats_tmp_G,
  const Matrix<BaseFloat> &delta,
  const Matrix<BaseFloat> &A,
  const Matrix<BaseFloat> &S,
  int32 max_iters) {

  int32 dim = spk_stats.dim_;
  KALDI_ASSERT(dim == delta.NumRows() && dim == S.NumRows());
  // The first D columns of delta_W
  SubMatrix<BaseFloat> delta_Dim(delta, 0, dim, 0, dim);
  // Eq. (46): b = tr(delta K^T) - tr(delta S^T)
  BaseFloat b = TraceMatMat(delta, spk_stats_tmp_K, kTrans)
                 - TraceMatMat(delta, S, kTrans);
  // Eq. (47): c = sum_d tr(delta_{d} G_{d} delta_{d})
  BaseFloat c = 0;
  Vector<BaseFloat> G_row_delta(dim + 1);
  for (int32 d = 0; d < dim; ++d) {
    G_row_delta.AddSpVec(1.0, spk_stats_tmp_G[d], delta.Row(d), 0.0);
    c += VecVec(G_row_delta, delta.Row(d));
  }

  // Sometimes, the change of step size, d1/d2, may get tiny
  // Due to numerical precision, we compute everything in double
  BaseFloat step_size = 0.0;
  BaseFloat obj_old, obj_new = 0.0;
  Matrix<BaseFloat> N(dim, dim);
  for (int32 iter_step = 1; iter_step <= max_iters; ++iter_step) {
    if (iter_step == 1) {
      // k = 0, auxf = beta logdet(A)
      obj_old = spk_stats.beta_ * A.LogDet();
    } else {
      obj_old = obj_new;
    }

    // Eq. (49): N = (A + k * delta_Dim)^{-1} delta_Dim
    // In case of bad condition, careful preconditioning should be done. Maybe safer
    // to use SolveQuadraticMatrixProblem. Future work for Yajie.
    Matrix<BaseFloat> tmp_A(A);
    tmp_A.AddMat(step_size, delta_Dim, kNoTrans);
    tmp_A.InvertDouble();
    N.AddMatMat(1.0, tmp_A, kNoTrans, delta_Dim, kNoTrans, 0.0);
    // first-order derivative w.r.t. k
    // Eq. (50): d1 = beta * trace(N) + b - k * c
    BaseFloat d1 = spk_stats.beta_ * TraceMat(N) + b - step_size * c;
    // second-order derivative w.r.t. k
    // Eq. (51): d2 = -beta * tr(N N) - c
    BaseFloat d2 = -c - spk_stats.beta_ * TraceMatMat(N, N, kNoTrans);
    d2 = std::min((double)d2, -c / 10.0);
    // convergence judgment from fmllr-sgmm.cc
    // it seems to work well, though not sure whether 1e-06 is appropriate
    // note from Dan: commenting this out after someone complained it was
    // causing a test to behave weirdly.  This doesn't dominate computation
    // anyway, I don't think.
    // if (std::fabs(d1 / d2) < 0.000001) { break; }

    // Eq. (52): update step_size
    BaseFloat step_size_change = -(d1 / d2);
    step_size += step_size_change;

    // Repeatedly check auxiliary function; halve step size change if auxf decreases.
    // According to the paper, we should limit the number of repetitions. The
    // following implementation seems to work well. But the termination condition/judgment
    // should be optimized later.
    do {
      // Eq. (48): auxf = beta * logdet(A + k * delta_Dim) + kb - 0.5 * k * k * c
      tmp_A.CopyFromMat(A);
      tmp_A.AddMat(step_size, delta_Dim, kNoTrans);
      obj_new = spk_stats.beta_ * tmp_A.LogDet() + step_size * b -
          0.5 * step_size * step_size * c;

      if (obj_new - obj_old < -1.0e-04 * spk_stats.beta_) {  // deal with numerical issues
        KALDI_WARN << "Objective function decreased (" << obj_old << "->"
                   << obj_new << "). Halving step size change ( step size "
                   << step_size << " -> " << (step_size - (step_size_change/2))
                   << ")";
        step_size_change /= 2;
        step_size -= step_size_change;
      }
    } while (obj_new - obj_old < -1.0e-04 * spk_stats.beta_ && step_size_change > 1e-05);
  }
  return step_size;
}

} // namespace kaldi

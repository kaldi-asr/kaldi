// sgmm/fmllr-sgmm.cc

// Copyright 2009-2011       Saarland University  (author:  Arnab Ghoshal)
//                2012  Johns Hopkins University (author: Daniel Povey)

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
#include <string>
#include <vector>
using std::vector;

#include "sgmm/fmllr-sgmm.h"
#include "util/parse-options.h"

namespace kaldi {

static void ApplyPreXformToGradient(const SgmmFmllrGlobalParams &globals,
                                    const Matrix<BaseFloat> &gradient_in,
                                    Matrix<BaseFloat> *gradient_out) {
  // Eq. (B.14): P' = A_{inv}^T P {W_{pre}^+}^T
  int32 dim = gradient_in.NumRows();
  Matrix<BaseFloat> Wpre_plus(dim + 1, dim + 1, kSetZero);
  Wpre_plus.Range(0, dim, 0, dim + 1).CopyFromMat(globals.pre_xform_);
  Wpre_plus(dim, dim) = 1;
  SubMatrix<BaseFloat> Ainv(globals.inv_xform_, 0, dim, 0, dim);
  Matrix<BaseFloat> AinvP(dim, dim + 1, kUndefined);
  AinvP.AddMatMat(1.0, Ainv, kTrans, gradient_in, kNoTrans, 0.0);
  gradient_out->AddMatMat(1.0, AinvP, kNoTrans, Wpre_plus, kTrans, 0.0);
}

static void ApplyInvPreXformToChange(const SgmmFmllrGlobalParams &globals,
                                     const Matrix<BaseFloat> &delta_in,
                                     Matrix<BaseFloat> *delta_out) {
  // Eq. (B.25): \Delta = A_{inv} \Delta' W_{pre}^+
  int32 dim = delta_in.NumRows();
  Matrix<BaseFloat> Wpre_plus(dim + 1, dim + 1, kSetZero);
  Wpre_plus.Range(0, dim, 0, dim + 1).CopyFromMat(globals.pre_xform_);
  Wpre_plus(dim, dim) = 1;
  SubMatrix<BaseFloat> Ainv(globals.inv_xform_, 0, dim, 0, dim);
  Matrix<BaseFloat> AinvD(dim, dim + 1, kUndefined);
  AinvD.AddMatMat(1.0, Ainv, kNoTrans, delta_in, kNoTrans, 0.0);
  delta_out->AddMatMat(1.0, AinvD, kNoTrans, Wpre_plus, kNoTrans, 0.0);
}

static void ApplyHessianXformToGradient(const SgmmFmllrGlobalParams &globals,
                                        const Matrix<BaseFloat> &gradient_in,
                                        Matrix<BaseFloat> *gradient_out) {
  int32 dim = gradient_in.NumRows();
  const Vector<BaseFloat> &D = globals.mean_scatter_;
  if (D.Min() <= 0.0)
    KALDI_ERR << "Cannot estimate FMLLR: mean scatter has 0 eigenvalues.";
  for (int32 r = 0; r < dim; r++) {
    for (int32 c = 0; c < r; c++) {
      // Eq. (B.15)
      (*gradient_out)(r, c) = gradient_in(r, c) / std::sqrt(1 + D(c));
      // Eq. (B.16)
      (*gradient_out)(c, r) = gradient_in(c, r) / std::sqrt(1 + D(r) -
          1 / (1 + D(c))) - gradient_in(r, c) / ((1 + D(c)) *
              std::sqrt(1 + D(r) - 1 / (1 + D(c))));
    }
    // Eq. (B.17) & (B.18)
    (*gradient_out)(r, r) = gradient_in(r, r) / std::sqrt(2 + D(r));
    (*gradient_out)(r, dim) = gradient_in(r, dim);
  }
}

static void ApplyInvHessianXformToChange(const SgmmFmllrGlobalParams &globals,
                                         const Matrix<BaseFloat> &delta_in,
                                         Matrix<BaseFloat> *delta_out) {
  int32 dim = delta_in.NumRows();
  const Vector<BaseFloat> &D = globals.mean_scatter_;
  if (D.Min() <= 0.0)
    KALDI_ERR << "Cannot estimate FMLLR: mean scatter has 0 eigenvalues.";
  for (int32 r = 0; r < dim; r++) {
    for (int32 c = 0; c < r; c++) {
      // Eq. (B.21)
      (*delta_out)(r, c) = delta_in(r, c) / std::sqrt(1 + D(c)) -
          delta_in(c, r) / ((1 + D(c)) * std::sqrt(1 + D(r) - 1 / (1 + D(c))));
      // Eq. (B.22)
      (*delta_out)(c, r) = delta_in(c, r) / std::sqrt(1 + D(r) - 1/ (1 + D(c)));
    }
    // Eq. (B.23) & (B.24)
    (*delta_out)(r, r) = delta_in(r, r) / std::sqrt(2 + D(r));
    (*delta_out)(r, dim) = delta_in(r, dim);
  }
}


void SgmmFmllrGlobalParams::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<SGMM_FMLLR_GLOBAL_PARAMS>");
  WriteToken(out, binary, "<PRE_XFORM>");
  pre_xform_.Write(out, binary);
  WriteToken(out, binary, "<INV_XFORM>");
  inv_xform_.Write(out, binary);
  WriteToken(out, binary, "<MEAN_SCATTER>");
  mean_scatter_.Write(out, binary);
  if (fmllr_bases_.size() != 0) {
    WriteToken(out, binary, "<FMLLR_BASIS>");
    uint32 tmp = static_cast<uint32>(fmllr_bases_.size());
    WriteBasicType(out, binary, tmp);
    for (uint32 i = 0; i < tmp; i++) {
      fmllr_bases_[i].Write(out, binary);
    }
  }
  WriteToken(out, binary, "</SGMM_FMLLR_GLOBAL_PARAMS>");
}

void SgmmFmllrGlobalParams::Read(std::istream &in, bool binary) {
  ExpectToken(in, binary, "<SGMM_FMLLR_GLOBAL_PARAMS>");
  ExpectToken(in, binary, "<PRE_XFORM>");
  pre_xform_.Read(in, binary);
  ExpectToken(in, binary, "<INV_XFORM>");
  inv_xform_.Read(in, binary);
  ExpectToken(in, binary, "<MEAN_SCATTER>");
  mean_scatter_.Read(in, binary);
  std::string token;
  ReadToken(in, binary, &token);
  if (token == "<FMLLR_BASIS>") {
    uint32 tmp;
    ReadBasicType(in, binary, &tmp);
    fmllr_bases_.resize(tmp);
    for (uint32 i = 0; i < tmp; i++) {
      fmllr_bases_[i].Read(in, binary);
    }
  } else {
    if (token != "</SGMM_FMLLR_GLOBAL_PARAMS>")
      KALDI_ERR << "Unexpected token '" << token << "' found.";
  }
}


void FmllrSgmmAccs::Init(int32 dim, int32 num_gaussians) {
  if (dim == 0) {  // empty stats
    dim_ = 0;  // non-zero dimension is meaningless in empty stats
    stats_.Init(0, 0);  // clear the stats
  } else {
    dim_ = dim;
    stats_.Init(dim, num_gaussians);
  }
}

BaseFloat FmllrSgmmAccs::Accumulate(const AmSgmm &model,
                                    const SgmmPerSpkDerivedVars &spk,
                                    const VectorBase<BaseFloat> &data,
                                    const SgmmPerFrameDerivedVars &frame_vars,
                                    int32 pdf_index, BaseFloat weight) {
  // Calulate Gaussian posteriors and collect statistics
  Matrix<BaseFloat> posteriors;
  BaseFloat log_like = model.ComponentPosteriors(frame_vars, pdf_index,
                                                 &posteriors);
  posteriors.Scale(weight);
  AccumulateFromPosteriors(model, spk, data, frame_vars.gselect, posteriors,
                           pdf_index);
  return log_like;
}

void
FmllrSgmmAccs::AccumulateFromPosteriors(const AmSgmm &model,
                                        const SgmmPerSpkDerivedVars &spk,
                                        const VectorBase<BaseFloat> &data,
                                        const vector<int32> &gselect,
                                        const Matrix<BaseFloat> &posteriors,
                                        int32 pdf_index) {
  Vector<double> var_scaled_mean(dim_), extended_data(dim_+1);
  extended_data.Range(0, dim_).CopyFromVec(data);
  extended_data(dim_) = 1.0;
  SpMatrix<double> scatter(dim_+1, kSetZero);
  scatter.AddVec2(1.0, extended_data);

  for (int32 ki = 0, ki_max = gselect.size(); ki < ki_max; ki++) {
    int32 i = gselect[ki];

    for (int32 m = 0; m < model.NumSubstates(pdf_index); m++) {
      // posterior gamma_{jkmi}(t)                             eq.(39)
      BaseFloat gammat_jmi = posteriors(ki, m);

      // Accumulate statistics for non-zero gaussian posterior
      if (gammat_jmi > 0.0) {
        stats_.beta_ += gammat_jmi;
        model.GetVarScaledSubstateSpeakerMean(pdf_index, m, i, spk,
                                              &var_scaled_mean);
        // Eq. (52): K += \gamma_{jmi} \Sigma_{i}^{-1} \mu_{jmi}^{(s)} x^{+T}
        stats_.K_.AddVecVec(gammat_jmi, var_scaled_mean, extended_data);
        // Eq. (53): G_{i} += \gamma_{jmi} x^{+} x^{+T}
        stats_.G_[i].AddSp(gammat_jmi, scatter);
      }  // non-zero posteriors
    }  // loop over substates
  }  // loop over selected Gaussians
}

void FmllrSgmmAccs::AccumulateForFmllrSubspace(const AmSgmm &sgmm,
    const SgmmFmllrGlobalParams &globals, SpMatrix<double> *grad_scatter) {
  if (stats_.beta_ <= 0.0) {
    KALDI_WARN << "Not committing any stats since no stats accumulated.";
    return;
  }
  int32 dim = sgmm.FeatureDim();
  Matrix<BaseFloat> xform(dim, dim + 1, kUndefined);
  xform.SetUnit();
  Matrix<BaseFloat> grad(dim, dim + 1, kSetZero);
  this->FmllrObjGradient(sgmm, xform, &grad, NULL);
  Matrix<BaseFloat> pre_xformed_grad(dim, dim + 1, kSetZero);
  ApplyPreXformToGradient(globals, grad, &pre_xformed_grad);
  Matrix<BaseFloat> hess_xformed_grad(dim, dim + 1, kSetZero);
  ApplyHessianXformToGradient(globals, pre_xformed_grad, &hess_xformed_grad);
  Vector<double> grad_vec(dim * (dim + 1));
  grad_vec.CopyRowsFromMat(hess_xformed_grad);
  grad_vec.Scale(1 / std::sqrt(stats_.beta_));
  grad_scatter->AddVec2(1.0, grad_vec);
  KALDI_LOG << "Frame counts for when committing fMLLR subspace stats are "
            << stats_.beta_;
}


BaseFloat FmllrSgmmAccs::FmllrObjGradient(const AmSgmm &sgmm,
                                          const Matrix<BaseFloat> &xform,
                                          Matrix<BaseFloat> *grad_out,
                                          Matrix<BaseFloat> *G_out) const {
  int32 dim = sgmm.FeatureDim(),
      num_gauss = sgmm.NumGauss();
  KALDI_ASSERT(stats_.G_.size() == static_cast<size_t>(num_gauss));
  Matrix<double> xform_d(xform);
  SubMatrix<double> A(xform_d, 0, dim, 0, dim);
  Matrix<double> xform_g(dim, dim + 1), total_g(dim, dim + 1);
  SpMatrix<double> inv_covar(dim);
  double obj = stats_.beta_ * A.LogDet() +
      TraceMatMat(xform_d, stats_.K_, kTrans);
  for (int32 i = 0; i < num_gauss; i++) {
    sgmm.GetInvCovars(i, &inv_covar);
    xform_g.AddMatSp(1.0, xform_d, kNoTrans, stats_.G_[i], 0.0);
    total_g.AddSpMat(1.0, inv_covar, xform_g, kNoTrans, 1.0);
  }
  obj -= 0.5 * TraceMatMat(xform_d, total_g, kTrans);
  if (G_out != NULL) G_out->CopyFromMat(total_g);

  // Compute the gradient: P = \beta [(A^{-1})^{T} , 0] + K - S
  if (grad_out != NULL) {
    Matrix<double> grad_d(dim, dim + 1, kSetZero);
    grad_d.Range(0, dim, 0, dim).CopyFromMat(A);
    grad_d.Range(0, dim, 0, dim).InvertDouble();
    grad_d.Range(0, dim, 0, dim).Transpose();
    grad_d.Scale(stats_.beta_);
    grad_d.AddMat(-1.0, total_g, kNoTrans);
    grad_d.AddMat(1.0, stats_.K_, kNoTrans);
    grad_out->CopyFromMat(grad_d);
  }

  return obj;
}


void FmllrSgmmAccs::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<FMLLRACCS>");
  WriteToken(out, binary, "<DIMENSION>");
  WriteBasicType(out, binary, dim_);
  WriteToken(out, binary, "<STATS>");
  stats_.Write(out, binary);
  WriteToken(out, binary, "</FMLLRACCS>");
}

void FmllrSgmmAccs::Read(std::istream &in, bool binary, bool add) {
  ExpectToken(in, binary, "<FMLLRACCS>");
  ExpectToken(in, binary, "<DIMENSION>");
  ReadBasicType(in, binary, &dim_);
  KALDI_ASSERT(dim_ > 0);
  ExpectToken(in, binary, "<STATS>");
  stats_.Read(in, binary, add);
  ExpectToken(in, binary, "</FMLLRACCS>");
}


static BaseFloat CalcFmllrStepSize(const AffineXformStats &stats,
                                   const AmSgmm &sgmm,
                                   const MatrixBase<BaseFloat> &Delta,
                                   const MatrixBase<BaseFloat> &A,
                                   const Matrix<BaseFloat> &G,
                                   int32 max_iters) {
  int32 dim = sgmm.FeatureDim();
  Matrix<double> Delta_d(Delta);
  Matrix<double> G_d(G);
  SubMatrix<double> Delta_C(Delta_d, 0, dim, 0, dim);

  // Eq. (B.28): m = tr(\Delta K^T) - tr(\Delta S^T)
  BaseFloat m = TraceMatMat(Delta_d, stats.K_, kTrans)
                    - TraceMatMat(Delta_d, G_d, kTrans);
  // Eq. (B.29): n = \sum_i tr(\Delta \Sigma_{i}^{-1} \Delta S_{i})
  BaseFloat n = 0;
  SpMatrix<double> inv_covar;
  for (int32 i = 0, num_gauss = sgmm.NumGauss(); i < num_gauss; i++) {
    sgmm.GetInvCovars(i, &inv_covar);
    n += TraceMatSpMatSp(Delta_d, kTrans, inv_covar, Delta_d, kNoTrans,
                         stats.G_[i]);
  }

  BaseFloat step_size = 0.0;
  // initialize just to get rid of compile errors.
  BaseFloat obj_step_old, obj_step_new = 0.0;
  Matrix<double> new_A(dim, dim);
  Matrix<double> B(dim, dim);
  for (int32 iter_step = 0; iter_step < max_iters; iter_step++) {
    if (iter_step == 0) {
      obj_step_old = stats.beta_ * A.LogDet();  // Q_0 = \beta * log det(A)
    } else {
      obj_step_old = obj_step_new;
    }

    // Eq. (B.30); B = (A + k\Delta^{-C})^{-1} \Delta^{-C}
    new_A.CopyFromMat(A);
    new_A.AddMat(step_size, Delta_C, kNoTrans);
    new_A.InvertDouble();
    B.AddMatMat(1.0, new_A, kNoTrans, Delta_C, kNoTrans, 0.0);

    BaseFloat d = m - step_size * n + stats.beta_ * TraceMat(B);
    BaseFloat d2 = -n - stats.beta_ * TraceMatMat(B, B, kNoTrans);
    if (std::fabs(d / d2) < 0.000001) { break; }  // converged

    BaseFloat step_size_change = -(d / d2);
    step_size += step_size_change;  // Eq. (B.33)

    // Halve step size when the auxiliary function decreases.
    do {
      new_A.CopyFromMat(A);
      new_A.AddMat(step_size, Delta_C, kNoTrans);
      BaseFloat logdet = new_A.LogDet();
      obj_step_new = stats.beta_ * logdet + step_size * m -
          0.5 * step_size * step_size * n;

      if (obj_step_new - obj_step_old < -0.001) {
        KALDI_WARN << "Objective function decreased (" << obj_step_old << "->"
                   << obj_step_new << "). Halving step size change ("
                   << step_size << " -> " << (step_size - (step_size_change/2))
                   << ")";
        step_size_change /= 2;
        step_size -= step_size_change;  // take away half of our step
      }  // Facing numeric precision issues. Compute in double?
    } while (obj_step_new - obj_step_old < -0.001 && step_size_change > 1e-05);
  }
  return step_size;
}


bool FmllrSgmmAccs::Update(const AmSgmm &sgmm,
                           const SgmmFmllrGlobalParams &globals,
                           const SgmmFmllrConfig &opts,
                           Matrix<BaseFloat> *out_xform,
                           BaseFloat *frame_count, BaseFloat *auxf_out) const {
  BaseFloat auxf_improv = 0.0, logdet = 0.0;
  KALDI_ASSERT(out_xform->NumRows() == dim_ && out_xform->NumCols() == dim_+1);
  BaseFloat mincount = (globals.HasBasis() ?
      std::min(opts.fmllr_min_count_basis, opts.fmllr_min_count_full) :
      opts.fmllr_min_count);
  bool using_subspace = (globals.HasBasis() ?
      (stats_.beta_ < opts.fmllr_min_count_full) : false);

  if (globals.IsEmpty())
    KALDI_ERR << "Must set up pre-transforms before estimating FMLLR.";

  KALDI_VLOG(1) << "Mincount = " << mincount << "; Basis: "
                << std::string(globals.HasBasis()? "yes; " : "no; ")
                << "Using subspace: " << std::string(using_subspace? "yes; "
                    : "no; ");

  int32 num_bases = 0;
  if (using_subspace) {
    KALDI_ASSERT(globals.fmllr_bases_.size() != 0);
    int32 max_bases = std::min(static_cast<int32>(globals.fmllr_bases_.size()),
                               opts.num_fmllr_bases);
    num_bases = (opts.bases_occ_scale <= 0.0)? max_bases :
        std::min(max_bases, static_cast<int32>(std::floor(opts.bases_occ_scale
                                                          * stats_.beta_)));
    KALDI_VLOG(1) << "Have " << stats_.beta_ << " frames for speaker: Using "
                  << num_bases << " fMLLR bases.";
  }

  // initialization just to get rid of compile errors.
  BaseFloat auxf_old = 0, auxf_new = 0;
  if (frame_count != NULL) *frame_count = stats_.beta_;

  // If occupancy is greater than the min count, update the transform
  if (stats_.beta_ >= mincount) {
    for (int32 iter = 0; iter < opts.fmllr_iters; iter++) {
      Matrix<BaseFloat> grad(dim_, dim_ + 1, kSetZero);
      Matrix<BaseFloat> G(dim_, dim_ + 1, kSetZero);
      auxf_new = this->FmllrObjGradient(sgmm, *out_xform, &grad, &G);

      // For diagnostic purposes
      KALDI_VLOG(3) << "Iter " << iter << ": Auxiliary function = "
          << (auxf_new / stats_.beta_) << " per frame over " << stats_.beta_
          << " frames";

      if (iter > 0) {
        // For diagnostic purposes
        KALDI_VLOG(2) << "Iter " << iter << ": Auxiliary function improvement: "
            << ((auxf_new - auxf_old) / stats_.beta_) << " per frame over "
            << (stats_.beta_) << " frames";
        auxf_improv += auxf_new - auxf_old;
      }

      Matrix<BaseFloat> pre_xformed_grad(dim_, dim_ + 1, kSetZero);
      ApplyPreXformToGradient(globals, grad, &pre_xformed_grad);
//      std::cout << "Pre-X Grad = " << pre_xformed_grad << std::endl;

      // Transform P_sk with the Hessian
      Matrix<BaseFloat> hess_xformed_grad(dim_, dim_ + 1, kSetZero);
      ApplyHessianXformToGradient(globals, pre_xformed_grad,
                                  &hess_xformed_grad);
//      std::cout << "Hess-X Grad = " << hess_xformed_grad << std::endl;

      // Update the actual FMLLR transform matrices
      Matrix<BaseFloat> hess_xformed_delta(dim_, dim_ + 1, kUndefined);
      if (using_subspace) {
        // Note that in this case we can simply store the speaker-specific
        // coefficients for each of the basis matrices. The current
        // implementation stores the computed transform to simplify the code!
        hess_xformed_delta.SetZero();
        for (int32 b = 0; b < num_bases; b++) {  // Eq (B.20)
          hess_xformed_delta.AddMat(TraceMatMat(globals.fmllr_bases_[b],
                                                hess_xformed_grad, kTrans),
                                    globals.fmllr_bases_[b], kNoTrans);
        }
        hess_xformed_delta.Scale(1 / stats_.beta_);
      } else {
        hess_xformed_delta.CopyFromMat(hess_xformed_grad);
        hess_xformed_delta.Scale(1 / stats_.beta_);  // Eq. (B.19)
      }

//      std::cout << "Hess-X Delta = " << hess_xformed_delta << std::endl;

      // Transform Delta with the Hessian
      Matrix<BaseFloat> pre_xformed_delta(dim_, dim_ + 1, kSetZero);
      ApplyInvHessianXformToChange(globals, hess_xformed_delta,
                                   &pre_xformed_delta);

      // Apply inverse pre-transform to Delta
      Matrix<BaseFloat> delta(dim_, dim_ + 1, kSetZero);
      ApplyInvPreXformToChange(globals, pre_xformed_delta, &delta);

#ifdef KALDI_PARANOID
      // Check whether co-ordinate transformation is correct.
      {
        BaseFloat tr1 = TraceMatMat(delta, grad, kTrans);
        BaseFloat tr2 = TraceMatMat(pre_xformed_delta, pre_xformed_grad,
                                    kTrans);
        BaseFloat tr3 = TraceMatMat(hess_xformed_delta, hess_xformed_grad,
                                    kTrans);
        AssertEqual(tr1, tr2, 1e-5);
        AssertEqual(tr2, tr3, 1e-5);
      }
#endif

      // Calculate the optimal step size
      SubMatrix<BaseFloat> A(*out_xform, 0, dim_, 0, dim_);
      BaseFloat step_size = CalcFmllrStepSize(stats_, sgmm, delta, A, G,
                                              opts.fmllr_iters);

      // Update: W <-- W + k \Delta   Eq. (B.34)
      out_xform->AddMat(step_size, delta, kNoTrans);
      auxf_old = auxf_new;

      // Check the objective function change for the last iteration
      if (iter == opts.fmllr_iters - 1) {
        auxf_new = this->FmllrObjGradient(sgmm, *out_xform, NULL, NULL);
        logdet = A.LogDet();
        // SubMatrix A points to the memory location of out_xform, and so will
        // contain the updated value

        KALDI_VLOG(2) << "Iter " << iter << ": Auxiliary function improvement: "
            << ((auxf_new - auxf_old) / stats_.beta_) << " per frame over "
            << (stats_.beta_) << " frames";
        auxf_improv += auxf_new - auxf_old;
      }
    }
    if (auxf_out != NULL) *auxf_out = auxf_improv;
    auxf_improv /= (stats_.beta_ + 1.0e-10);

    KALDI_LOG << "Auxiliary function improvement for FMLLR = " << auxf_improv
        << " per frame over " << stats_.beta_ << " frames. Log-determinant = "
        << logdet;
    return true;
  } else {
    KALDI_ASSERT(stats_.beta_ < mincount);
//    std::cerr.precision(10);
//    std::cerr.setf(std::ios::fixed,std::ios::floatfield);
    KALDI_WARN << "Not updating FMLLR because count is " << stats_.beta_
        << " < " << (mincount);
    if (auxf_out != NULL) *auxf_out = 0.0;
    return false;
  }  // Do not use the transform if it does not have enough counts
  KALDI_ASSERT(false);  // Should never be reached.
}

void EstimateSgmmFmllrSubspace(const SpMatrix<double> &fmllr_grad_scatter,
                               int32 num_fmllr_bases, int32 feat_dim,
                               SgmmFmllrGlobalParams *globals, double min_eig) {
  KALDI_ASSERT(num_fmllr_bases > 0 && feat_dim > 0);
  if (num_fmllr_bases >  feat_dim * (feat_dim + 1)) {
    num_fmllr_bases = feat_dim * (feat_dim + 1);
    KALDI_WARN << "Limiting number of fMLLR bases to be the same as transform "
               << "dimension.";
  }

  vector< Matrix<BaseFloat> > &fmllr_bases(globals->fmllr_bases_);

  Vector<double> s(fmllr_grad_scatter.NumRows());
  Matrix<double> U(fmllr_grad_scatter.NumRows(),
                   fmllr_grad_scatter.NumRows());
  try {
    fmllr_grad_scatter.Eig(&s, &U);
    SortSvd(&s, &U);  // in case was not exactly sorted.
    KALDI_VLOG(1) << "Eigenvalues (max 200) of CMLLR scatter are: "
                  << (SubVector<double>(s, 0,
                                        std::min(static_cast<MatrixIndexT>(200),
                                                 s.Dim())));
    
//    for (int32 b = 2; b < num_fmllr_bases; b++) {
//      if (s(b) < min_eig) {
//        num_fmllr_bases = b;
//        KALDI_WARN << "Limiting number of fMLLR bases to " << num_fmllr_bases
//                   << " because of small eigenvalues.";
//        break;
//      }
//    }

    U.Transpose();  // Now the rows of U correspond to the basis vectors.
    fmllr_bases.resize(num_fmllr_bases);
    for (int32 b = 0; b < num_fmllr_bases; b++) {
      fmllr_bases[b].Resize(feat_dim, feat_dim + 1, kSetZero);
      fmllr_bases[b].CopyRowsFromVec(U.Row(b));
    }
    KALDI_LOG << "Estimated " << num_fmllr_bases << " fMLLR basis matrices.";
  } catch(const std::exception &e) {
    KALDI_WARN << "Not estimating FMLLR bases because of a thrown exception:\n"
               << e.what();
    fmllr_bases.resize(0);
  }
}  // End of EstimateSgmmFmllrSubspace


}  // namespace kaldi


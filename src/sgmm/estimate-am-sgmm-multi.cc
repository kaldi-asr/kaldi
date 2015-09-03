// sgmm/estimate-am-sgmm-multi.cc

// Copyright 2012       Arnab Ghoshal

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
using std::string;
#include <vector>
using std::vector;

#include "sgmm/am-sgmm.h"
#include "sgmm/estimate-am-sgmm-multi.h"
#include "thread/kaldi-thread.h"

namespace kaldi {

void MleAmSgmmGlobalAccs::ResizeAccumulators(const AmSgmm &model,
                                             SgmmUpdateFlagsType flags) {
  num_gaussians_ = model.NumGauss();
  feature_dim_ = model.FeatureDim();
  phn_space_dim_ = model.PhoneSpaceDim();
  spk_space_dim_ = model.SpkSpaceDim();

  if (flags & (kSgmmPhoneProjections | kSgmmCovarianceMatrix)) {
    Y_.resize(num_gaussians_);
    Q_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; ++i) {
      Y_[i].Resize(feature_dim_, phn_space_dim_, kSetZero);
      Q_[i].Resize(phn_space_dim_, kSetZero);
    }
  } else {
    Y_.clear();
    Q_.clear();
  }

  if (flags & kSgmmCovarianceMatrix) {
    S_.resize(num_gaussians_);
    S_means_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++) {
      S_[i].Resize(feature_dim_, kSetZero);
      S_means_[i].Resize(feature_dim_, kSetZero);
    }
  } else {
    S_.clear();
  }

  if (flags & kSgmmSpeakerProjections) {
    if (spk_space_dim_ == 0) {
      KALDI_ERR << "Cannot set up accumulators for speaker projections "
                << "because speaker subspace has not been set up";
    }
    Z_.resize(num_gaussians_);
    R_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; ++i) {
      Z_[i].Resize(feature_dim_, spk_space_dim_, kSetZero);
      R_[i].Resize(spk_space_dim_, kSetZero);
    }
  } else {
    Z_.clear();
    R_.clear();
  }

  gamma_i_.Resize(num_gaussians_, kSetZero);
}

void MleAmSgmmGlobalAccs::ZeroAccumulators(SgmmUpdateFlagsType flags) {
  if (flags & (kSgmmPhoneProjections | kSgmmCovarianceMatrix)) {
    for (int32 i = 0, end = Y_.size(); i < end; ++i)
      Y_[i].SetZero();
  }
  if (flags & kSgmmCovarianceMatrix) {
    for (int32 i = 0, end = S_.size(); i < end; ++i) {
      S_[i].SetZero();
      S_means_[i].SetZero();
    }
  }

  if (flags & kSgmmSpeakerProjections) {
    for (int32 i = 0, end = Z_.size(); i < end; ++i) {
      Z_[i].SetZero();
      R_[i].SetZero();
    }
  }
  gamma_i_.SetZero();
}

void MleAmSgmmGlobalAccs::AddAccumulators(const AmSgmm &model,
                                          const MleAmSgmmAccs &accs,
                                          SgmmUpdateFlagsType flags) {
  total_frames_ += accs.total_frames_;
  total_like_ += accs.total_like_;
  for (int32 i = 0; i < num_gaussians_; ++i) {
    if (flags & (kSgmmPhoneProjections | kSgmmCovarianceMatrix)) {
      Y_[i].AddMat(1.0, accs.Y_[i], kNoTrans);
    }
    if (flags & kSgmmSpeakerProjections) {
      Z_[i].AddMat(1.0, accs.Z_[i], kNoTrans);
      R_[i].AddSp(1.0, accs.R_[i]);
    }
    if (flags & kSgmmCovarianceMatrix)
      S_[i].AddSp(1.0, accs.S_[i]);
  }

  // gamma_i
  for (int32 j = 0; j < model.NumPdfs(); ++j) {
    for (int32 m = 0; m < model.NumSubstates(j); ++m) {
      gamma_i_.AddVec(1.0, accs.gamma_[j].Row(m));
    }
  }

  //  Compute the Q_i quantities (Eq. 64).
  if (flags & kSgmmPhoneProjections) {
    for (int32 i = 0; i < num_gaussians_; ++i) {
      for (int32 j = 0; j < accs.num_states_; ++j) {
        const Matrix<BaseFloat> &state_vec(model.StateVectors(j));
        for (int32 m = 0; m < model.NumSubstates(j); ++m) {
          if (accs.gamma_[j](m, i) > 0.0) {
            Q_[i].AddVec2(static_cast<BaseFloat>(accs.gamma_[j](m, i)),
                          state_vec.Row(m));
          }
        }
      }
    }
  }

  // Compute the S_i^{(means)} quantities (Eq. 74).
  if (flags & kSgmmCovarianceMatrix) {
    Matrix<double> YM_MY(feature_dim_, feature_dim_);
    SpMatrix<double> tmp_S_means(feature_dim_);
    Vector<BaseFloat> mu_jmi(feature_dim_);
    for (int32 i = 0; i < num_gaussians_; ++i) {
      // YM_MY = - (Y_{i} M_{i}^T)
      Matrix<double> M(model.GetPhoneProjection(i));
      YM_MY.AddMatMat(-1.0, accs.Y_[i], kNoTrans, M, kTrans, 0.0);
      // Add its own transpose: YM_MY = - (Y_{i} M_{i}^T + M_{i} Y_{i}^T)
      {
        Matrix<double> M(YM_MY, kTrans);
        YM_MY.AddMat(1.0, M);
      }
      tmp_S_means.CopyFromMat(YM_MY);  // Sigma_{i} = -(YM' + MY')

      for (int32 j = 0; j < accs.num_states_; ++j) {
        for (int32 m = 0; m < model.NumSubstates(j); ++m) {
          // Sigma_{i} += gamma_{jmi} * mu_{jmi}*mu_{jmi}^T
          model.GetSubstateMean(j, m, i, &mu_jmi);
          tmp_S_means.AddVec2(static_cast<BaseFloat>(accs.gamma_[j](m, i)), mu_jmi);
        }
      }
      S_means_[i].AddSp(1.0, tmp_S_means);
      KALDI_ASSERT(1.0 / S_means_[i](0, 0) != 0.0);
    }
  }
}

BaseFloat MleAmSgmmUpdaterMulti::UpdateGlobals(const MleAmSgmmGlobalAccs &accs,
                                               SgmmUpdateFlagsType flags) {
  BaseFloat tot_impr = 0.0;
  if (flags & kSgmmPhoneProjections) {
    tot_impr += UpdateM(accs);
  }
  if (flags & kSgmmCovarianceMatrix) {
    tot_impr += UpdateVars(accs);
  }
  if (flags & kSgmmSpeakerProjections) {
    tot_impr += UpdateN(accs);
    if (update_options_.renormalize_N)
      KALDI_WARN << "Not renormalizing N";
  }

  KALDI_LOG << "**Total auxf improvement for phone projections & covariances is "
            << (tot_impr) << " over " << accs.total_frames_ << " frames.";
  return tot_impr;
}

void MleAmSgmmUpdaterMulti::Update(const std::vector<MleAmSgmmAccs*> &accs,
                                   const std::vector<AmSgmm*> &models,
                                   SgmmUpdateFlagsType flags) {
  KALDI_ASSERT((flags & (kSgmmPhoneVectors | kSgmmPhoneProjections |
                         kSgmmPhoneWeightProjections | kSgmmCovarianceMatrix |
                         kSgmmSubstateWeights | kSgmmSpeakerProjections)) != 0);
  if (accs.size() != models.size()) {
    KALDI_ERR << "Found " << accs.size() << " accs and " << models.size()
              << " models. Must have same number of models and accs.";
  }

  SgmmUpdateFlagsType global_flags = (flags & (kSgmmPhoneProjections |
                                               kSgmmPhoneWeightProjections |
                                               kSgmmSpeakerProjections |
                                               kSgmmCovarianceMatrix));
  SgmmUpdateFlagsType state_spec_flags = (flags & ~global_flags);
  MleAmSgmmGlobalAccs glob_accs;
  BaseFloat tot_impr = 0.0;
  int32 num_models = models.size();

  std::vector< SpMatrix<double> > H;
  if (update_options_.renormalize_V)
    models[0]->ComputeH(&H);

  if (global_flags != 0) {  // expected operating case
    glob_accs.ResizeAccumulators(*models[0], global_flags);
    for (int32 i = 0; i < num_models; ++i) {
      glob_accs.AddAccumulators(*models[i], *accs[i], global_flags);
    }
    UpdateGlobals(glob_accs, global_flags);

    // Weight projection needs access to all models
    if (global_flags & kSgmmPhoneWeightProjections) {
      if (update_options_.use_sequential_weight_update)
        KALDI_ERR << "Sequential weight update not implemented, using parallel";
//        tot_impr += UpdateWSequential(accs, model);
//      } else {
        tot_impr += UpdateWParallel(accs, models);
//      }
    }
  } else {  // Shouldn't be using this class without updating global params
    KALDI_WARN << "Using MleAmSgmmUpdaterMulti class without updating global "
               << " parameters.";
  }

  // Update the state-specific parameters: phone vectors & substate weights
  if (state_spec_flags != 0) {
    MleAmSgmmOptions state_spec_opts = update_options_;
    state_spec_opts.renormalize_V = false;
    state_spec_opts.renormalize_N = false;

    MleAmSgmmUpdater sgmm_updater(state_spec_opts);
    for (int32 i = 0; i < num_models; ++i)
      tot_impr += sgmm_updater.Update(*accs[i], models[i], state_spec_flags);
  }


  if (update_options_.renormalize_V && (global_flags != 0)) {
    SpMatrix<double> H_sm;
    this->ComputeSmoothingTerms(glob_accs, H, &H_sm);
    RenormalizeV(H_sm, models);
  }

  KALDI_LOG << "**Total auxf improvement, combining all parameters, over "
            << "all model is " << tot_impr << " per frame.";

  // The following is just for diagnostics
  double total_frames = 0, total_like = 0;
  for (int32 i = 0; i < num_models; ++i) {
    total_frames += accs[i]->TotalFrames();
    total_like += accs[i]->TotalLike();
  }
  KALDI_LOG << "***Total data likelihood, over all models, is "
            << (total_like/total_frames) << " over " << total_frames
            << " frames.";

  // Now, copy the global parameters to the models
  for (int32 i = 0; i < num_models; ++i) {
    if ((flags & kSgmmPhoneProjections) || update_options_.renormalize_V)
      models[i]->M_ = global_M_;
    if (flags & kSgmmCovarianceMatrix)
      models[i]->SigmaInv_ = global_SigmaInv_;
    if ((flags & kSgmmSpeakerProjections) || update_options_.renormalize_N)
      models[i]->N_ = global_N_;
    if ((flags & kSgmmPhoneWeightProjections) || update_options_.renormalize_V)
      models[i]->w_ = global_w_;
    models[i]->ComputeNormalizers();  // So that the models are ready to use.
  }
}

// Compute H^{(sm)}, the "smoothing" matrices.
void MleAmSgmmUpdaterMulti::ComputeSmoothingTerms(
    const MleAmSgmmGlobalAccs &accs,
    const std::vector< SpMatrix<double> > &H,
    SpMatrix<double> *H_sm) const {
  KALDI_ASSERT(H_sm != NULL);
  H_sm->Resize(PhoneSpaceDim());

  double sum = 0.0;
  for (int32 i = 0; i < NumGauss(); ++i) {
    if (accs.gamma_i_(i) > 0) {
      H_sm->AddSp(accs.gamma_i_(i), H[i]);
      sum += accs.gamma_i_(i);
    }
  }

  if (sum == 0.0) {
    KALDI_WARN << "Sum of counts is zero. Smoothing matrix set to unit";
    H_sm->SetUnit();  // arbitrary non-singular matrix
  } else {
    H_sm->Scale(1.0 / sum);
    int32 tmp = H_sm->LimitCondDouble(update_options_.max_cond_H_sm);
    if (tmp > 0) {
      KALDI_WARN << "Limited " << tmp << " eigenvalues of H_sm.";
    }
  }
}

double MleAmSgmmUpdaterMulti::UpdateM(const MleAmSgmmGlobalAccs &accs) {
  double totcount = 0.0, tot_like_impr = 0.0;
  for (int32 i = 0; i < accs.num_gaussians_; ++i) {
    if (accs.gamma_i_(i) < accs.feature_dim_) {
      KALDI_WARN << "For component " << i << ": not updating M due to very "
                 << "small count (=" << accs.gamma_i_(i) << ").";
      continue;
    }


    SolverOptions opts;
    opts.name = "M";
    opts.K = update_options_.max_cond;
    opts.eps = update_options_.epsilon;
    
    Matrix<double> Mi(global_M_[i]);
    double impr =
        SolveQuadraticMatrixProblem(accs.Q_[i], accs.Y_[i],
                                    SpMatrix<double>(global_SigmaInv_[i]),
                                    opts, &Mi);
    global_M_[i].CopyFromMat(Mi);

    if (i % 50 == 0) {
      KALDI_VLOG(2) << "Objf impr for projection M for i = " << i << ", is "
                    << (impr/(accs.gamma_i_(i) + 1.0e-20)) << " over "
                    << accs.gamma_i_(i) << " frames";
    }
    totcount += accs.gamma_i_(i);
    tot_like_impr += impr;
  }
  tot_like_impr /= (totcount + 1.0e-20);
  KALDI_LOG << "Overall objective function improvement for model projections "
            << "M is " << tot_like_impr << " over " << totcount << " frames";
  return tot_like_impr;
}

double MleAmSgmmUpdaterMulti::UpdateN(const MleAmSgmmGlobalAccs &accs) {
  double totcount = 0.0, tot_like_impr = 0.0;
  if (accs.spk_space_dim_ == 0 || accs.R_.size() == 0 || accs.Z_.size() == 0) {
    KALDI_ERR << "Speaker subspace dim is zero or no stats accumulated";
  }

  for (int32 i = 0; i < accs.num_gaussians_; ++i) {
    if (accs.gamma_i_(i) < 2 * accs.spk_space_dim_) {
      KALDI_WARN << "Not updating speaker basis for i = " << (i)
                 << " because count is too small " << (accs.gamma_i_(i));
      continue;
    }

    SolverOptions opts;
    opts.name = "N";
    opts.K = update_options_.max_cond;
    opts.eps = update_options_.epsilon;
    
    Matrix<double> Ni(global_N_[i]);
    double impr =
        SolveQuadraticMatrixProblem(accs.R_[i], accs.Z_[i],
                                    SpMatrix<double>(global_SigmaInv_[i]),
                                    opts, &Ni);
    global_N_[i].CopyFromMat(Ni);
    if (i < 10) {
      KALDI_LOG << "Objf impr for spk projection N for i = " << (i)
                << ", is " << (impr / (accs.gamma_i_(i) + 1.0e-20)) << " over "
                << (accs.gamma_i_(i)) << " frames";
    }
    totcount += accs.gamma_i_(i);
    tot_like_impr += impr;
  }

  tot_like_impr /= (totcount+1.0e-20);
  KALDI_LOG << "**Overall objf impr for N is " << tot_like_impr << " over "
            << totcount << " frames";
  return tot_like_impr;
}


double MleAmSgmmUpdaterMulti::UpdateVars(const MleAmSgmmGlobalAccs &accs) {
  SpMatrix<double> Sigma_i(accs.feature_dim_), Sigma_i_ml(accs.feature_dim_);
  double tot_objf_impr = 0.0, tot_t = 0.0;
  SpMatrix<double> covfloor(accs.feature_dim_);
  Vector<double> objf_improv(accs.num_gaussians_);

  // First pass over all (shared) Gaussian components to calculate the
  // ML estimate of the covariances, and the total covariance for flooring.
  for (int32 i = 0; i < accs.num_gaussians_; ++i) {
    // Eq. (75): Sigma_{i}^{ml} = 1/gamma_{i} [S_{i} + S_{i}^{(means)} - ...
    //                                          Y_{i} M_{i}^T - M_{i} Y_{i}^T]
    // Note the S_means_ already contains the Y_{i} M_{i}^T terms.
    Sigma_i_ml.CopyFromSp(accs.S_means_[i]);
    Sigma_i_ml.AddSp(1.0, accs.S_[i]);
    covfloor.AddSp(1.0, Sigma_i_ml);
    // inverting  small values e.g. 4.41745328e-40 seems to generate inf,
    // although would be fixed up later.
    if (accs.gamma_i_(i) > 1.0e-20) {
      Sigma_i_ml.Scale(1 / (accs.gamma_i_(i) + 1.0e-20));
    } else {
      Sigma_i_ml.SetUnit();
    }
    KALDI_ASSERT(1.0 / Sigma_i_ml(0, 0) != 0.0);
    // Eq. (76): Compute the objective function with the old parameter values
    objf_improv(i) = global_SigmaInv_[i].LogPosDefDet() -
        TraceSpSp(SpMatrix<double>(global_SigmaInv_[i]), Sigma_i_ml);

    global_SigmaInv_[i].CopyFromSp(Sigma_i_ml);  // inverted in the next loop.
  }

  // Compute the covariance floor.
  if (accs.gamma_i_.Sum() == 0) {  // If no count, use identity.
    KALDI_WARN << "Updating variances: zero counts. Setting floor to unit.";
    covfloor.SetUnit();
  } else {  // else, use the global average covariance.
    covfloor.Scale(update_options_.cov_floor / accs.gamma_i_.Sum());
    int32 tmp;
    if ((tmp = covfloor.LimitCondDouble(update_options_.max_cond)) != 0) {
      KALDI_WARN << "Covariance flooring matrix is poorly conditioned. Fixed "
                 << "up " << (tmp) << " eigenvalues.";
    }
  }

  if (update_options_.cov_diag_ratio > 1000) {
    KALDI_LOG << "Assuming you want to build a diagonal system since "
              << "cov_diag_ratio is large: making diagonal covFloor.";
    for (int32 i = 0; i < covfloor.NumRows(); i++)
      for (int32 j = 0; j < i; j++)
        covfloor(i, j) = 0.0;
  }

  // Second pass over all (shared) Gaussian components to calculate the
  // floored estimate of the covariances, and update the model.
  for (int32 i = 0; i < accs.num_gaussians_; ++i) {
    Sigma_i.CopyFromSp(global_SigmaInv_[i]);
    Sigma_i_ml.CopyFromSp(Sigma_i);
    // In case of insufficient counts, make the covariance matrix diagonal.
    // cov_diag_ratio is 2 by default, set to very large to always get diag-cov
    if (accs.gamma_i_(i) < update_options_.cov_diag_ratio * accs.feature_dim_) {
      KALDI_WARN << "For Gaussian component " << i << ": Too low count "
                 << accs.gamma_i_(i) << " for covariance matrix estimation. "
                 << "Setting to diagonal";
      for (int32 d = 0; d < accs.feature_dim_; d++)
        for (int32 e = 0; e < d; e++)
          Sigma_i(d, e) = 0.0;  // SpMatrix, can only set lower traingular part

      int floored = Sigma_i.ApplyFloor(covfloor);
      if (floored > 0) {
        KALDI_WARN << "For Gaussian component " << i << ": Floored " << floored
                   << " covariance eigenvalues.";
      }
      global_SigmaInv_[i].CopyFromSp(Sigma_i);
      global_SigmaInv_[i].InvertDouble();
    } else {  // Updating the full covariance matrix.
      try {
        int floored = Sigma_i.ApplyFloor(covfloor);
        if (floored > 0) {
          KALDI_WARN << "For Gaussian component " << i << ": Floored "
                     << floored << " covariance eigenvalues.";
        }
        global_SigmaInv_[i].CopyFromSp(Sigma_i);
        global_SigmaInv_[i].InvertDouble();

        objf_improv(i) += Sigma_i.LogPosDefDet() +
            TraceSpSp(SpMatrix<double>(global_SigmaInv_[i]), Sigma_i_ml);
        objf_improv(i) *= (-0.5 * accs.gamma_i_(i));  // Eq. (76)

        tot_objf_impr += objf_improv(i);
        tot_t += accs.gamma_i_(i);
        if (i < 5) {
          KALDI_VLOG(2) << "objf impr from variance update =" << objf_improv(i)
              / (accs.gamma_i_(i) + 1.0e-20) << " over " << (accs.gamma_i_(i))
                        << " frames for i = " << (i);
        }
      } catch(...) {
        KALDI_WARN << "Updating within-class covariance matrix i = " << (i)
                   << ", numerical problem";
        // This is a catch-all thing in case of unanticipated errors, but
        // flooring should prevent this occurring for the most part.
        global_SigmaInv_[i].SetUnit();  // Set to unit.
      }
    }
  }
  KALDI_LOG << "**Overall objf impr for variance update = "
            << (tot_objf_impr / (tot_t+ 1.0e-20))
            << " over " << (tot_t) << " frames";
  return tot_objf_impr / (tot_t + 1.0e-20);
}


// The parallel weight update, in the paper.
double MleAmSgmmUpdaterMulti::UpdateWParallel(
    const std::vector<MleAmSgmmAccs*> &accs,
    const std::vector<AmSgmm*> &models) {
  KALDI_LOG << "Updating weight projections";

  int32 phn_dim = models[0]->PhoneSpaceDim(),
      num_gauss = models[0]->NumGauss(),
      num_models = models.size();
  SpMatrix<double> v_vT(phn_dim);
  // tot_like_{after, before} are totals over multiple iterations,
  // not valid likelihoods. but difference is valid (when divided by tot_count).
  double tot_predicted_like_impr = 0.0, tot_like_before = 0.0,
      tot_like_after = 0.0, tot_count = 0.0;

  Vector<double> w_jm(num_gauss);
  Matrix<double> g_i(num_gauss, phn_dim);
  std::vector< SpMatrix<double> > F_i(num_gauss);

  Matrix<double> w(global_w_);
  for (int iter = 0; iter < update_options_.weight_projections_iters; iter++) {
    for (int32 i = 0; i < num_gauss; ++i) {
      F_i[i].Resize(phn_dim, kSetZero);
    }
    double k_like_before = 0.0, k_count = 0.0;
    g_i.SetZero();

    // Unlike in the report the inner most loop is over Gaussians, where
    // per-gaussian statistics are accumulated. This is more memory demanding
    // but more computationally efficient, as outer product v_{jvm} v_{jvm}^T
    // is computed only once for all gaussians.

    for (int32 mdl_idx = 0; mdl_idx < num_models; ++mdl_idx) {
      std::vector< Matrix<double> > gamma(accs[mdl_idx]->GetOccs());
      for (int32 j = 0; j < models[mdl_idx]->NumPdfs(); j++) {
        for (int32 m = 0; m < models[mdl_idx]->NumSubstates(j); m++) {
          double gamma_jm = gamma[j].Row(m).Sum();
          k_count += gamma_jm;

          // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
          w_jm.AddMatVec(1.0, w, kNoTrans,
                         Vector<double>(models[mdl_idx]->v_[j].Row(m)), 0.0);
          w_jm.Add((-1.0) * w_jm.LogSumExp());
          k_like_before += VecVec(w_jm, gamma[j].Row(m));
          w_jm.ApplyExp();
          v_vT.SetZero();
          // v_vT := v_{jkm} v_{jkm}^T
          v_vT.AddVec2(1.0, models[mdl_idx]->v_[j].Row(m));

          for (int32 i = 0; i < num_gauss; i++) {
            // Suggestion: g_jkm can be computed more efficiently
            // using the Vector/Matrix routines for all i at once
            // linear term around cur value.
            double linear_term = gamma[j](m, i) - gamma_jm * w_jm(i);
            double quadratic_term = std::max(gamma[j](m, i), gamma_jm * w_jm(i));
            g_i.Row(i).AddVec(linear_term, models[mdl_idx]->v_[j].Row(m));
            // Now I am calling this F_i in the document. [dan]
            F_i[i].AddSp(quadratic_term, v_vT);
          }
        }  // loop over substates
      }  // loop over states
    }  // loop over model/acc pairs

    Matrix<double> w_orig(w);
    double k_predicted_like_impr = 0.0, k_like_after = 0.0;
    double min_step = 0.001, step_size;

    SolverOptions opts;
    opts.name = "w";
    opts.K = update_options_.max_cond;
    opts.eps = update_options_.epsilon;
    
    for (step_size = 1.0; step_size >= min_step; step_size /= 2) {
      k_predicted_like_impr = 0.0;
      k_like_after = 0.0;

      for (int32 i = 0; i < num_gauss; i++) {
        // auxf is formulated in terms of change in w.
        Vector<double> delta_w(phn_dim);
        // returns objf impr with step_size = 1,
        // but it may not be 1 so we recalculate it.
        SolveQuadraticProblem(F_i[i], g_i.Row(i), opts, &delta_w);

        delta_w.Scale(step_size);
        double predicted_impr = VecVec(delta_w, g_i.Row(i)) -
            0.5 * VecSpVec(delta_w,  F_i[i], delta_w);

        // should never be negative because
        // we checked inside SolveQuadraticProblem.
        KALDI_ASSERT(predicted_impr >= -1.0e-05);

        if (i < 10) {
          KALDI_LOG << "Predicted objf impr for w (not per frame), iter = " <<
              (iter) << ", i = " << (i) << " is " << (predicted_impr);
        }
        k_predicted_like_impr += predicted_impr;
        w.Row(i).AddVec(1.0, delta_w);
      }

      for (int32 mdl_idx = 0; mdl_idx < num_models; ++mdl_idx) {
        std::vector< Matrix<double> > gamma(accs[mdl_idx]->GetOccs());
        for (int32 j = 0; j < models[mdl_idx]->NumPdfs(); j++) {
          for (int32 m = 0; m < models[mdl_idx]->NumSubstates(j); m++) {
            w_jm.AddMatVec(1.0, w, kNoTrans,
                           Vector<double>(models[mdl_idx]->v_[j].Row(m)), 0.0);
            w_jm.Add((-1.0) * w_jm.LogSumExp());
            k_like_after += VecVec(w_jm, gamma[j].Row(m));
          }
        }
      }
      KALDI_VLOG(2) << "For iteration " << (iter) << ", updating w gives "
                    << "predicted per-frame like impr "
                    << (k_predicted_like_impr / k_count) << ", actual "
                    << ((k_like_after - k_like_before) / k_count) << ", over "
                    << (k_count) << " frames";
      if (k_like_after < k_like_before) {
        w.CopyFromMat(w_orig);  // Undo what we computed.
        if (fabs(k_like_after - k_like_before) / k_count < 1.0e-05) {
          k_like_after = k_like_before;
          KALDI_WARN << "Not updating weights as not increasing auxf and "
                     << "probably due to numerical issues (since small change).";
          break;
        } else {
          KALDI_WARN << "Halving step size for weights as likelihood did "
                     << "not increase";
        }
      } else {
        break;
      }
    }
    if (step_size < min_step) {
      // Undo any step as we have no confidence that this is right.
      w.CopyFromMat(w_orig);
    } else {
      if (iter == 0) {
        tot_count += k_count;
      }
      tot_predicted_like_impr += k_predicted_like_impr;
      tot_like_after += k_like_after;
      tot_like_before += k_like_before;
    }
  }

  global_w_.CopyFromMat(w);

  tot_predicted_like_impr /= tot_count;
  tot_like_after = (tot_like_after - tot_like_before) / tot_count;
  KALDI_LOG << "**Overall objf impr for w is " << tot_predicted_like_impr
            << ", actual " << tot_like_after << ", over "
            << tot_count << " frames";
  return tot_like_after;
}

void MleAmSgmmUpdaterMulti::RenormalizeV(const SpMatrix<double> &H_sm,
                                        const vector<AmSgmm*> &models) {
  int32 phn_dim = PhoneSpaceDim(),
      feat_dim = FeatureDim(),
      num_models = models.size();
  SpMatrix<double> Sigma(phn_dim);
  int32 count = 0;
  for (int32 mdl = 0; mdl < num_models; ++mdl) {
    for (int32 j = 0; j < models[mdl]->NumPdfs(); ++j) {
      for (int32 m = 0; m < models[mdl]->NumSubstates(j); ++m) {
        count++;
        Sigma.AddVec2(static_cast<BaseFloat>(1.0), models[mdl]->v_[j].Row(m));
      }
    }
  }
  Sigma.Scale(1.0 / count);
  int32 fixed_eigs = Sigma.LimitCondDouble(update_options_.max_cond);
  if (fixed_eigs != 0) {
    KALDI_WARN << "Scatter of vectors v is poorly conditioned. Fixed up "
               << fixed_eigs << " eigenvalues.";
  }
  KALDI_LOG << "Eigenvalues of scatter of vectors v is : ";
  Sigma.PrintEigs("Sigma");
  if (!Sigma.IsPosDef()) {
    KALDI_LOG << "Not renormalizing v because scatter is not positive definite"
              << " -- maybe first iter?";
    return;
  }

  // Want to make variance of v unit and H_sm (like precision matrix) diagonal.
  TpMatrix<double> L(phn_dim);
  L.Cholesky(Sigma);
  TpMatrix<double> LInv(L);
  LInv.Invert();

  Matrix<double> tmpL(phn_dim, phn_dim);
  tmpL.CopyFromTp(L);

  SpMatrix<double> H_sm_proj(phn_dim);
  H_sm_proj.AddMat2Sp(1.0, tmpL, kTrans, H_sm, 0.0);
  // H_sm_proj := L^{T} * H_sm * L.
  // This is right because we would transform the vectors themselves
  // by L^{-1}, and H_sm is like the inverse of the vectors,
  // so it's {L^{-1}}^{-T} = L^T.

  Matrix<double> U(phn_dim, phn_dim);
  Vector<double> eigs(phn_dim);
  H_sm_proj.SymPosSemiDefEig(&eigs, &U, 1.0);  // 1.0 means no checking +ve def -> faster
  KALDI_LOG << "Note on the next diagnostic: the first number is generally not "
            << "that meaningful as it relates to the static offset";
  H_sm_proj.PrintEigs("H_sm_proj (Significance of dims in vector space.. note)");

  // Transform on vectors is U^T L^{-1}.
  // Why?  Because transform on H_sm is T =U^T L^T
  // and we want T^{-T} by normal rules of vector/covector and we
  // have (U^T L^T)^{-T} = (L U)^{-1} = U^T L^{-1}.
  Matrix<double> Trans(phn_dim, phn_dim);  // T^{-T}
  Matrix<double> tmpLInv(phn_dim, phn_dim);
  tmpLInv.CopyFromTp(LInv);
  Trans.AddMatMat(1.0, U, kTrans, tmpLInv, kNoTrans, 0.0);
  Matrix<double> TransInv(Trans);
  TransInv.Invert();  // T in above...

#ifdef KALDI_PARANOID
  {
    SpMatrix<double> H_sm_tmp(phn_dim);
    H_sm_tmp.AddMat2Sp(1.0, TransInv, kTrans, H_sm, 0.0);
    KALDI_ASSERT(H_sm_tmp.IsDiagonal(0.1));
  }
  {
    SpMatrix<double> Sigma_tmp(phn_dim);
    Sigma_tmp.AddMat2Sp(1.0, Trans, kNoTrans, Sigma, 0.0);
    KALDI_ASSERT(Sigma_tmp.IsUnit(0.1));
  }
#endif

  for (int32 mdl = 0; mdl < num_models; ++mdl) {
    for (int32 j = 0; j < models[mdl]->NumPdfs(); ++j) {
      for (int32 m = 0; m < models[mdl]->NumSubstates(j); ++m) {
        Vector<double> tmp(phn_dim);
        tmp.AddMatVec(1.0, Trans, kNoTrans, Vector<double>(models[mdl]->v_[j].Row(m)), 0.0);
        models[mdl]->v_[j].Row(m).CopyFromVec(tmp);
      }
    }
  }
  for (int32 i = 0; i < NumGauss(); ++i) {
    Vector<double> tmp(phn_dim);
    tmp.AddMatVec(1.0, TransInv, kTrans, Vector<double>(global_w_.Row(i)), 0.0);
    global_w_.Row(i).CopyFromVec(tmp);

    Matrix<double> tmpM(feat_dim, phn_dim);
    // Multiplying on right not left so must not transpose TransInv.
    tmpM.AddMatMat(1.0, Matrix<double>(global_M_[i]), kNoTrans,
                   TransInv, kNoTrans, 0.0);
    global_M_[i].CopyFromMat(tmpM);
  }
  KALDI_LOG << "Renormalized subspace.";
}

}  // namespace kaldi

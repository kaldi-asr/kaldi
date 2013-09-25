// sgmm/estimate-am-sgmm.cc

// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Ondrej Glembek;  Yanmin Qian;
// Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)
//                      Liang Lu;  Arnab Ghoshal

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


#include "sgmm/am-sgmm.h"
#include "sgmm/estimate-am-sgmm.h"
#include "thread/kaldi-thread.h"

namespace kaldi {
using std::string;
using std::vector;

void MleAmSgmmAccs::Write(std::ostream &out_stream, bool binary) const {
  uint32 tmp_uint32;

  WriteToken(out_stream, binary, "<SGMMACCS>");

  WriteToken(out_stream, binary, "<NUMSTATES>");
  tmp_uint32 = static_cast<uint32>(num_states_);
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteToken(out_stream, binary, "<NUMGaussians>");
  tmp_uint32 = static_cast<uint32>(num_gaussians_);
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteToken(out_stream, binary, "<FEATUREDIM>");
  tmp_uint32 = static_cast<uint32>(feature_dim_);
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteToken(out_stream, binary, "<PHONESPACEDIM>");
  tmp_uint32 = static_cast<uint32>(phn_space_dim_);
  WriteBasicType(out_stream, binary, tmp_uint32);
  WriteToken(out_stream, binary, "<SPKSPACEDIM>");
  tmp_uint32 = static_cast<uint32>(spk_space_dim_);
  WriteBasicType(out_stream, binary, tmp_uint32);
  if (!binary) out_stream << "\n";

  if (Y_.size() != 0) {
    KALDI_ASSERT(gamma_.size() != 0);
    WriteToken(out_stream, binary, "<Y>");
    for (int32 i = 0; i < num_gaussians_; i++) {
      Y_[i].Write(out_stream, binary);
    }
  }
  if (Z_.size() != 0) {
    KALDI_ASSERT(R_.size() != 0);
    WriteToken(out_stream, binary, "<Z>");
    for (int32 i = 0; i < num_gaussians_; i++) {
      Z_[i].Write(out_stream, binary);
    }
    WriteToken(out_stream, binary, "<R>");
    for (int32 i = 0; i < num_gaussians_; i++) {
      R_[i].Write(out_stream, binary);
    }
  }
  if (S_.size() != 0) {
    KALDI_ASSERT(gamma_.size() != 0);
    WriteToken(out_stream, binary, "<S>");
    for (int32 i = 0; i < num_gaussians_; i++) {
      S_[i].Write(out_stream, binary);
    }
  }
  if (y_.size() != 0) {
    KALDI_ASSERT(gamma_.size() != 0);
    WriteToken(out_stream, binary, "<y>");
    for (int32 j = 0; j < num_states_; j++) {
      y_[j].Write(out_stream, binary);
    }
  }
  if (gamma_.size() != 0) {
    WriteToken(out_stream, binary, "<gamma>");
    for (int32 j = 0; j < num_states_; j++) {
      gamma_[j].Write(out_stream, binary);
    }
  }
  WriteToken(out_stream, binary, "<total_like>");
  WriteBasicType(out_stream, binary, total_like_);

  WriteToken(out_stream, binary, "<total_frames>");
  WriteBasicType(out_stream, binary, total_frames_);

  WriteToken(out_stream, binary, "</SGMMACCS>");
}

void MleAmSgmmAccs::Read(std::istream &in_stream, bool binary,
                         bool add) {
  uint32 tmp_uint32;
  string token;

  ExpectToken(in_stream, binary, "<SGMMACCS>");

  ExpectToken(in_stream, binary, "<NUMSTATES>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  num_states_ = static_cast<int32>(tmp_uint32);
  ExpectToken(in_stream, binary, "<NUMGaussians>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  num_gaussians_ = static_cast<int32>(tmp_uint32);
  ExpectToken(in_stream, binary, "<FEATUREDIM>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  feature_dim_ = static_cast<int32>(tmp_uint32);
  ExpectToken(in_stream, binary, "<PHONESPACEDIM>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  phn_space_dim_ = static_cast<int32>(tmp_uint32);
  ExpectToken(in_stream, binary, "<SPKSPACEDIM>");
  ReadBasicType(in_stream, binary, &tmp_uint32);
  spk_space_dim_ = static_cast<int32>(tmp_uint32);

  ReadToken(in_stream, binary, &token);

  while (token != "</SGMMACCS>") {
    if (token == "<Y>") {
      Y_.resize(num_gaussians_);
      for (size_t i = 0; i < Y_.size(); i++) {
        Y_[i].Read(in_stream, binary, add);
      }
    } else if (token == "<Z>") {
      Z_.resize(num_gaussians_);
      for (size_t i = 0; i < Z_.size(); i++) {
        Z_[i].Read(in_stream, binary, add);
      }
    } else if (token == "<R>") {
      R_.resize(num_gaussians_);
      if (gamma_s_.Dim() == 0) gamma_s_.Resize(num_gaussians_);
      for (size_t i = 0; i < R_.size(); i++) {
        R_[i].Read(in_stream, binary, add);
      }
    } else if (token == "<S>") {
      S_.resize(num_gaussians_);
      for (size_t i = 0; i < S_.size(); i++) {
        S_[i].Read(in_stream, binary, add);
      }
    } else if (token == "<y>") {
      y_.resize(num_states_);
      for (int32 j = 0; j < num_states_; j++) {
        y_[j].Read(in_stream, binary, add);
      }
    } else if (token == "<gamma>") {
      gamma_.resize(num_states_);
      for (int32 j = 0; j < num_states_; j++) {
        gamma_[j].Read(in_stream, binary, add);
      }
      // Don't read gamma_s, it's just a temporary variable and
      // not part of the permanent (non-speaker-specific) accs.
    } else if (token == "<total_like>") {
      double total_like;
      ReadBasicType(in_stream, binary, &total_like);
      if (add)
        total_like_ += total_like;
      else
        total_like_ = total_like;
    } else if (token == "<total_frames>") {
      double total_frames;
      ReadBasicType(in_stream, binary, &total_frames);
      if (add)
        total_frames_ += total_frames;
      else
        total_frames_ = total_frames;
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadToken(in_stream, binary, &token);
  }
}

void MleAmSgmmAccs::Check(const AmSgmm &model,
                          bool show_properties) const {
  if (show_properties) {
    KALDI_LOG << "SgmmPdfModel: J = " << num_states_ << ", D = " <<
        feature_dim_ << ", S = " << phn_space_dim_ << ", T = " <<
        spk_space_dim_ << ", I = " << num_gaussians_;
  }
  KALDI_ASSERT(num_states_ == model.NumPdfs() && num_states_ > 0);
  KALDI_ASSERT(num_gaussians_ == model.NumGauss() && num_gaussians_ > 0);
  KALDI_ASSERT(feature_dim_ == model.FeatureDim() && feature_dim_ > 0);
  KALDI_ASSERT(phn_space_dim_ == model.PhoneSpaceDim() && phn_space_dim_ > 0);
  KALDI_ASSERT(spk_space_dim_ == model.SpkSpaceDim());

  std::ostringstream debug_str;

  if (Y_.size() == 0) {
    debug_str << "Y: no.  ";
  } else {
    KALDI_ASSERT(gamma_.size() != 0);
    KALDI_ASSERT(Y_.size() == static_cast<size_t>(num_gaussians_));
    bool nz = false;
    for (int32 i = 0; i < num_gaussians_; i++) {
      KALDI_ASSERT(Y_[i].NumRows() == feature_dim_ &&
                   Y_[i].NumCols() == phn_space_dim_);
      if (!nz && Y_[i](0, 0) != 0) { nz = true; }
    }
    debug_str << "Y: yes, " << string(nz ? "nonzero. " : "zero. ");
  }

  if (Z_.size() == 0) {
    KALDI_ASSERT(R_.size() == 0);
    debug_str << "Z, R: no.  ";
  } else {
    KALDI_ASSERT(gamma_s_.Dim() == num_gaussians_);
    KALDI_ASSERT(Z_.size() == static_cast<size_t>(num_gaussians_));
    KALDI_ASSERT(R_.size() == static_cast<size_t>(num_gaussians_));
    bool Z_nz = false, R_nz = false;
    for (int32 i = 0; i < num_gaussians_; i++) {
      KALDI_ASSERT(Z_[i].NumRows() == feature_dim_ &&
                   Z_[i].NumCols() == spk_space_dim_);
      KALDI_ASSERT(R_[i].NumRows() == spk_space_dim_);
      if (!Z_nz && Z_[i](0, 0) != 0) { Z_nz = true; }
      if (!R_nz && R_[i](0, 0) != 0) { R_nz = true; }
    }
    bool gamma_s_nz = !gamma_s_.IsZero();
    debug_str << "Z: yes, " << string(Z_nz ? "nonzero. " : "zero. ");
    debug_str << "R: yes, " << string(R_nz ? "nonzero. " : "zero. ");
    debug_str << "gamma_s: yes, " << string(gamma_s_nz ? "nonzero. " : "zero. ");
  }

  if (S_.size() == 0) {
    debug_str << "S: no.  ";
  } else {
    KALDI_ASSERT(gamma_.size() != 0);
    bool S_nz = false;
    KALDI_ASSERT(S_.size() == static_cast<size_t>(num_gaussians_));
    for (int32 i = 0; i < num_gaussians_; i++) {
      KALDI_ASSERT(S_[i].NumRows() == feature_dim_);
      if (!S_nz && S_[i](0, 0) != 0) { S_nz = true; }
    }
    debug_str << "S: yes, " << string(S_nz ? "nonzero. " : "zero. ");
  }

  if (y_.size() == 0) {
    debug_str << "y: no.  ";
  } else {
    KALDI_ASSERT(gamma_.size() != 0);
    bool nz = false;
    KALDI_ASSERT(y_.size() == static_cast<size_t>(num_states_));
    for (int32 j = 0; j < num_states_; j++) {
      KALDI_ASSERT(y_[j].NumRows() == model.NumSubstates(j));
      KALDI_ASSERT(y_[j].NumCols() == phn_space_dim_);
      if (!nz && y_[j](0, 0) != 0) { nz = true; }
    }
    debug_str << "y: yes, " << string(nz ? "nonzero. " : "zero. ");
  }

  if (gamma_.size() == 0) {
    debug_str << "gamma: no.  ";
  } else {
    debug_str << "gamma: yes.  ";
    bool nz = false;
    KALDI_ASSERT(gamma_.size() == static_cast<size_t>(num_states_));
    for (int32 j = 0; j < num_states_; j++) {
      KALDI_ASSERT(gamma_[j].NumRows() == model.NumSubstates(j) &&
                   gamma_[j].NumCols() == num_gaussians_);
      // Just test the first substate for nonzero, else it would take too long.
      if (!nz && gamma_[j].Row(0).Norm(1.0) != 0) { nz = true; }
    }
    debug_str << "gamma: yes, " << string(nz ? "nonzero. " : "zero. ");
  }

  if (show_properties)
    KALDI_LOG << "Subspace GMM model properties: " << debug_str.str() << '\n';
}

void MleAmSgmmAccs::ResizeAccumulators(const AmSgmm &model,
                                       SgmmUpdateFlagsType flags) {
  num_states_ = model.NumPdfs();
  num_gaussians_ = model.NumGauss();
  feature_dim_ = model.FeatureDim();
  phn_space_dim_ = model.PhoneSpaceDim();
  spk_space_dim_ = model.SpkSpaceDim();

  if (flags & (kSgmmPhoneProjections | kSgmmCovarianceMatrix)) {
    Y_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++) {
      Y_[i].Resize(feature_dim_, phn_space_dim_);
    }
  } else {
    Y_.clear();
  }

  if (flags & kSgmmSpeakerProjections) {
    if (spk_space_dim_ == 0) {
      KALDI_ERR << "Cannot set up accumulators for speaker projections "
                << "because speaker subspace has not been set up";
    }
    gamma_s_.Resize(num_gaussians_);
    Z_.resize(num_gaussians_);
    R_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++) {
      Z_[i].Resize(feature_dim_, spk_space_dim_);
      R_[i].Resize(spk_space_dim_);
    }
  } else {
    gamma_s_.Resize(0);
    Z_.clear();
    R_.clear();
  }

  if (flags & kSgmmCovarianceMatrix) {
    S_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++) {
      S_[i].Resize(feature_dim_);
    }
  } else {
    S_.clear();
  }

  if (flags & (kSgmmPhoneVectors | kSgmmPhoneWeightProjections |
               kSgmmCovarianceMatrix | kSgmmSubstateWeights |
               kSgmmPhoneProjections)) {
    gamma_.resize(num_states_);
    total_frames_ = total_like_ = 0;
    for (int32 j = 0; j < num_states_; j++) {
      gamma_[j].Resize(model.NumSubstates(j), num_gaussians_);
    }
  } else {
    gamma_.clear();
    total_frames_ = total_like_ = 0;
  }

  if (flags & kSgmmPhoneVectors) {
    y_.resize(num_states_);
    for (int32 j = 0; j < num_states_; j++) {
      y_[j].Resize(model.NumSubstates(j), phn_space_dim_);
    }
  } else {
    y_.clear();
  }
}

BaseFloat MleAmSgmmAccs::Accumulate(const AmSgmm &model,
                                    const SgmmPerFrameDerivedVars &frame_vars,
                                    const VectorBase<BaseFloat> &v_s,  // may be empty
                                    int32 j, BaseFloat weight,
                                    SgmmUpdateFlagsType flags) {
  // Calculate Gaussian posteriors and collect statistics
  Matrix<BaseFloat> posteriors;
  BaseFloat log_like = model.ComponentPosteriors(frame_vars, j, &posteriors);
  posteriors.Scale(weight);
  BaseFloat count = AccumulateFromPosteriors(model, frame_vars, posteriors,
                                             v_s, j, flags);
  // Note: total_frames_ is incremented in AccumulateFromPosteriors().
  total_like_ += count * log_like;
  return log_like;
}


BaseFloat MleAmSgmmAccs::AccumulateFromPosteriors(
    const AmSgmm &model,
    const SgmmPerFrameDerivedVars &frame_vars,
    const Matrix<BaseFloat> &posteriors,
    const VectorBase<BaseFloat> &v_s,  // may be empty
    int32 j,
    SgmmUpdateFlagsType flags) {
  double tot_count = 0.0;
  const vector<int32> &gselect = frame_vars.gselect;
  // Intermediate variables
  Vector<BaseFloat> gammat(gselect.size());
  Vector<BaseFloat> xt_jmi(feature_dim_), mu_jmi(feature_dim_),
      zt_jmi(spk_space_dim_);

  int32 num_substates = model.NumSubstates(j);
  for (int32 ki = 0; ki < static_cast<int32>(gselect.size()); ki++) {
    int32 i = gselect[ki];

    for (int32 m = 0; m < num_substates; m++) {
      // Eq. (39): gamma_{jmi}(t) = p (j, m, i|t)
      BaseFloat gammat_jmi = RandPrune(posteriors(ki, m), rand_prune_);

      // Accumulate statistics for non-zero gaussian posterior
      if (gammat_jmi != 0.0) {
        tot_count += gammat_jmi;
        if (flags & (kSgmmPhoneVectors | kSgmmPhoneWeightProjections |
                     kSgmmCovarianceMatrix | kSgmmSubstateWeights |
                     kSgmmPhoneProjections)) {
          // Eq. (40): gamma_{jmi} = \sum_t gamma_{jmi}(t)
          gamma_[j](m, i) += gammat_jmi;
        }

        if (flags & kSgmmPhoneVectors) {
          // Eq. (41): y_{jm} = \sum_{t, i} \gamma_{jmi}(t) z_{i}(t)
          // Suggestion:  move this out of the loop over m
          y_[j].Row(m).AddVec(gammat_jmi, frame_vars.zti.Row(ki));
        }

        if (flags & (kSgmmPhoneProjections | kSgmmCovarianceMatrix)) {
          // Eq. (42): Y_{i} = \sum_{t, j, m} \gamma_{jmi}(t) x_{i}(t) v_{jm}^T
          Y_[i].AddVecVec(gammat_jmi, frame_vars.xti.Row(ki),
                          model.StateVectors(j).Row(m));
        }

        if (flags & kSgmmCovarianceMatrix)
          gammat(ki) += gammat_jmi;

        // Accumulate for speaker projections
        if (flags & kSgmmSpeakerProjections) {
          KALDI_ASSERT(spk_space_dim_ > 0);
          // Eq. (43): x_{jmi}(t) = x_k(t) - M{i} v_{jm}
          model.GetSubstateMean(j, m, i, &mu_jmi);
          xt_jmi.CopyFromVec(frame_vars.xt);
          xt_jmi.AddVec(-1.0, mu_jmi);
          // Eq. (44): Z_{i} = \sum_{t, j, m} \gamma_{jmi}(t) x_{jmi}(t) v^{s}'
          if (v_s.Dim() != 0)  // interpret empty v_s as zero.
            Z_[i].AddVecVec(gammat_jmi, xt_jmi, v_s);
          // Eq. (49): \gamma_{i}^{(s)} = \sum_{t\in\Tau(s), j, m} gamma_{jmi}
          // Will be used when you call CommitStatsForSpk(), to update R_.
          gamma_s_(i) += gammat_jmi;
        }
      }  // non-zero posteriors
    }  // loop over substates
  }  // loop over selected Gaussians

  if (flags & kSgmmCovarianceMatrix) {
    for (int32 ki = 0; ki < static_cast<int32>(gselect.size()); ki++) {
      int32 i = gselect[ki];
      // Eq. (47): S_{i} = \sum_{t, j, m} \gamma_{jmi}(t) x_{i}(t) x_{i}(t)^T
      if (gammat(ki) != 0.0)
        S_[i].AddVec2(gammat(ki), frame_vars.xti.Row(ki));
    }
  }
  total_frames_ += tot_count;
  return tot_count;
}

void MleAmSgmmAccs::CommitStatsForSpk(const AmSgmm &model,
                                      const VectorBase<BaseFloat> &v_s) {
  if (v_s.Dim() != 0 && spk_space_dim_ > 0 && gamma_s_.Dim() != 0) {
    if (!v_s.IsZero())
      for (int32 i = 0; i < num_gaussians_; i++)
        // Accumulate Statistics R_{ki}
        if (gamma_s_(i) != 0.0)
          R_[i].AddVec2(static_cast<BaseFloat>(gamma_s_(i)), v_s);
  }
  gamma_s_.SetZero();
}

void MleAmSgmmAccs::GetStateOccupancies(Vector<BaseFloat> *occs) const {
  occs->Resize(gamma_.size());
  for (int32 j = 0, end = gamma_.size(); j < end; j++) {
    (*occs)(j) = gamma_[j].Sum();
  }
}

BaseFloat MleAmSgmmUpdater::Update(const MleAmSgmmAccs &accs,
                                   AmSgmm *model,
                                   SgmmUpdateFlagsType flags) {
  KALDI_ASSERT((flags & (kSgmmPhoneVectors | kSgmmPhoneProjections |
                         kSgmmPhoneWeightProjections | kSgmmCovarianceMatrix |
                         kSgmmSubstateWeights | kSgmmSpeakerProjections)) != 0);

  if (flags & kSgmmPhoneProjections)
    ComputeQ(accs, *model, &Q_);
  if (flags & kSgmmCovarianceMatrix)
    ComputeSMeans(accs, *model, &S_means_);

  // quantities used in both vector and weights updates...
  vector< SpMatrix<double> > H;
  // "smoothing" matrices, weighted sums of above.
  SpMatrix<double> H_sm;
  Vector<double> y_sm;  // "smoothing" vectors
  if ((flags & (kSgmmPhoneVectors | kSgmmPhoneWeightProjections))
      || update_options_.renormalize_V) {
    model->ComputeH(&H);
    ComputeSmoothingTerms(accs, *model, H, &H_sm,
                          (flags & kSgmmPhoneVectors) ? &y_sm : NULL);
  }

  BaseFloat tot_impr = 0.0;

  if (flags & kSgmmPhoneVectors) {
    if (update_options_.check_v) {
      KALDI_ASSERT(update_options_.tau_vec == 0 &&
                   "You cannot combine the check-v and tau-vec options.");
      tot_impr += UpdatePhoneVectorsChecked(accs, model, H);
    } else {
      tot_impr += UpdatePhoneVectors(accs, model, H, H_sm, y_sm);
    }
  }
  if (flags & kSgmmPhoneProjections) {
    if (update_options_.tau_map_M > 0.0)
      tot_impr += MapUpdateM(accs, model);  // MAP adaptation of M
    else
      tot_impr += UpdateM(accs, model);
  }

  if (flags & kSgmmPhoneWeightProjections) {
    if (update_options_.use_sequential_weight_update) {
      tot_impr += UpdateWSequential(accs, model);
    } else {
      tot_impr += UpdateWParallel(accs, model);
    }
  }
  if (flags & kSgmmCovarianceMatrix)
    tot_impr += UpdateVars(accs, model);
  if (flags & kSgmmSubstateWeights)
    tot_impr += UpdateSubstateWeights(accs, model);
  if (flags & kSgmmSpeakerProjections) {
    tot_impr += UpdateN(accs, model);
    if (update_options_.renormalize_N)
      RenormalizeN(accs, model); // if you renormalize N you have to
    // alter any speaker vectors you're keeping around, as well.
  }

  if (update_options_.renormalize_V)
    RenormalizeV(accs, model, H_sm);

  KALDI_LOG << "*Overall auxf improvement, combining all parameters, is "
            << tot_impr;

  KALDI_LOG << "***Overall data likelihood is "
            << (accs.total_like_/accs.total_frames_)
            << " over " << (accs.total_frames_) << " frames.";

  model->ComputeNormalizers();  // So that the model is ready to use.
  return tot_impr;
}

// Compute the Q_{i} (Eq. 64)
void MleAmSgmmUpdater::ComputeQ(const MleAmSgmmAccs &accs,
                                const AmSgmm &model,
                                std::vector< SpMatrix<double> > *Q) {
  Q->resize(accs.num_gaussians_);
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    (*Q)[i].Resize(accs.phn_space_dim_);
    for (int32 j = 0; j < accs.num_states_; j++) {
      for (int32 m = 0; m < model.NumSubstates(j); m++) {
        if (accs.gamma_[j](m, i) > 0.0) {
          (*Q)[i].AddVec2(static_cast<BaseFloat>(accs.gamma_[j](m, i)),
                        model.v_[j].Row(m));
        }
      }
    }
  }
}

// Compute the S_i^{(means)} quantities (Eq. 74).
// Note: we seem to have also included in this variable
// the term - (Y_i M_I^T + M_i Y_i^T).
void MleAmSgmmUpdater::ComputeSMeans(const MleAmSgmmAccs &accs,
                                     const AmSgmm &model,
                                     std::vector< SpMatrix<double> > *S_means) {
  S_means->resize(accs.num_gaussians_);
  Matrix<double> YM_MY(accs.feature_dim_, accs.feature_dim_);
  Vector<BaseFloat> mu_jmi(accs.feature_dim_);
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    // YM_MY = - (Y_{i} M_{i}^T)
    YM_MY.AddMatMat(-1.0, accs.Y_[i], kNoTrans,
                    Matrix<double>(model.M_[i]), kTrans, 0.0);
    // Add its own transpose: YM_MY = - (Y_{i} M_{i}^T + M_{i} Y_{i}^T)
    {
      Matrix<double> M(YM_MY, kTrans);
      YM_MY.AddMat(1.0, M);
    }
    (*S_means)[i].Resize(accs.feature_dim_, kUndefined);
    (*S_means)[i].CopyFromMat(YM_MY);  // Sigma_{i} = -(YM' + MY')

    for (int32 j = 0; j < accs.num_states_; j++) {
      for (int32 m = 0; m < model.NumSubstates(j); m++) {
        if (accs.gamma_[j](m, i) != 0.0) {
          // Sigma_{i} += gamma_{jmi} * mu_{jmi}*mu_{jmi}^T
          mu_jmi.AddMatVec(1.0, model.M_[i], kNoTrans, model.v_[j].Row(m), 0.0);
          (*S_means)[i].AddVec2(static_cast<BaseFloat>(accs.gamma_[j](m, i)), mu_jmi);
        }
      }
    }
    KALDI_ASSERT(1.0 / (*S_means)[i](0, 0) != 0.0);
  }
}

// Compute H^{(sm)}, the "smoothing" matrices.
void MleAmSgmmUpdater::ComputeSmoothingTerms(const MleAmSgmmAccs &accs,
                                             const AmSgmm &model,
                                             const vector<SpMatrix<double> > &H,
                                             SpMatrix<double> *H_sm,
                                             Vector<double> *y_sm) const {
  KALDI_ASSERT(H_sm != NULL);
  H_sm->Resize(accs.phn_space_dim_);
  if (y_sm != NULL) y_sm->Resize(accs.phn_space_dim_);
  Vector<double> gamma_i(accs.num_gaussians_);

  for (int32 j = 0; j < accs.num_states_; j++) {
    for (int32 m = 0, end = model.NumSubstates(j); m < end; m++) {
      gamma_i.AddVec(1.0, accs.gamma_[j].Row(m));
      if (y_sm != NULL) (*y_sm).AddVec(1.0, accs.y_[j].Row(m));
    }
  }

  double sum = 0.0;
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    if (gamma_i(i) > 0) {
      H_sm->AddSp(gamma_i(i), H[i]);
      sum += gamma_i(i);
    }
  }

  if (sum == 0.0) {
    KALDI_WARN << "Sum of counts is zero. Smoothing matrix set to unit"
               << string((y_sm != NULL)? " & smoothing vector set to 0." : ".");
    H_sm->SetUnit();  // arbitrary non-singular matrix
  } else {
    if (y_sm != NULL) {
      (*y_sm).Scale(1.0 / sum);
      KALDI_VLOG(3) << "y_sm is " << (*y_sm);
    }
    H_sm->Scale(1.0 / sum);
    Matrix<double> H_sm_old(*H_sm);
    int32 tmp = H_sm->LimitCondDouble(update_options_.max_cond_H_sm);
    if (tmp > 0) {
      KALDI_WARN << "Limited " << tmp << " eigenvalues of H_sm.";
      if (update_options_.fixup_H_sm && y_sm != NULL) {
        Vector<double> avgVec(accs.phn_space_dim_);
        SpMatrix<double> HInv(H_sm_old);
        HInv.Invert();
        avgVec.AddSpVec(1.0, HInv, (*y_sm), 0.0);
        (*y_sm).AddSpVec(1.0, (*H_sm), avgVec, 0.0);
        KALDI_VLOG(3) << "y_sm [fixed up] is " << (*y_sm);
      }
    }
  }
}


class UpdatePhoneVectorsClass: public MultiThreadable { // For multi-threaded.
 public:
  UpdatePhoneVectorsClass(const MleAmSgmmUpdater &updater,
                          const MleAmSgmmAccs &accs,
                          AmSgmm *model,
                          const std::vector<SpMatrix<double> > &H,
                          const SpMatrix<double> &H_sm,
                          const Vector<double> &y_sm,
                          double *auxf_impr,
                          double *like_impr):
      updater_(updater), accs_(accs), model_(model), 
      H_(H), H_sm_(H_sm), y_sm_(y_sm), auxf_impr_ptr_(auxf_impr),
      auxf_impr_(0.0), like_impr_ptr_(like_impr), like_impr_(0.0) { }
    
  ~UpdatePhoneVectorsClass() {
    *auxf_impr_ptr_ += auxf_impr_;
    *like_impr_ptr_ += like_impr_;
  }
  
  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to the total sums in the destructor.
    updater_.UpdatePhoneVectorsInternal(accs_, model_, H_, H_sm_, y_sm_,
                                        &auxf_impr_, &like_impr_,
                                        num_threads_, thread_id_);
  }
 private:
  const MleAmSgmmUpdater &updater_;
  const MleAmSgmmAccs &accs_;
  AmSgmm *model_;
  const std::vector<SpMatrix<double> > &H_;
  const SpMatrix<double> &H_sm_;
  const Vector<double> &y_sm_;
  double *auxf_impr_ptr_;
  double auxf_impr_;
  double *like_impr_ptr_;
  double like_impr_;
};


// Runs the phone vectors update for a subset of states (called
// multi-threaded).
void MleAmSgmmUpdater::UpdatePhoneVectorsInternal(
    const MleAmSgmmAccs &accs,
    AmSgmm *model,
    const std::vector<SpMatrix<double> > &H,
    const SpMatrix<double> &H_sm,
    const Vector<double> &y_sm,
    double *auxf_impr,
    double *like_impr,
    int32 num_threads,
    int32 thread_id) const {

  int32 block_size = (accs.num_states_ + (num_threads-1)) / num_threads,
      j_start = block_size * thread_id,
      j_end = std::min(accs.num_states_, j_start + block_size);
  
  for (int32 j = j_start; j < j_end; j++) {
    double state_count = 0.0, state_auxf_impr = 0.0, state_like_impr = 0.0;
    Vector<double> w_jm(accs.num_gaussians_);
    for (int32 m = 0; m < model->NumSubstates(j); m++) {
      double gamma_jm = accs.gamma_[j].Row(m).Sum();
      state_count += gamma_jm;
      Vector<double> g_jm(accs.phn_space_dim_);  // computed using eq. 58
      SpMatrix<double> H_jm(accs.phn_space_dim_);  // computed using eq. 59
      // First compute normal H_jm.

      // need weights for this ...
      // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
      w_jm.AddMatVec(1.0, Matrix<double>(model->w_), kNoTrans,
                     Vector<double>(model->v_[j].Row(m)), 0.0);
      w_jm.ApplySoftMax();
      g_jm.CopyFromVec(accs.y_[j].Row(m));

      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        double gamma_jmi = accs.gamma_[j](m, i);
        double quadratic_term = std::max(gamma_jmi, gamma_jm * w_jm(i));
        double scalar = gamma_jmi - gamma_jm * w_jm(i) + quadratic_term
            * VecVec(model->w_.Row(i), model->v_[j].Row(m));
        g_jm.AddVec(scalar, model->w_.Row(i));
        if (gamma_jmi != 0.0) {
          H_jm.AddSp(gamma_jmi, H[i]);  // The most important term..
        }
        if (quadratic_term > 1.0e-10) {
          H_jm.AddVec2(static_cast<BaseFloat>(quadratic_term), model->w_.Row(i));
        }
      }
      SpMatrix<double> H_jm_dash(H_jm);  // with ad-hoc smoothing term.
      Vector<double> g_jm_dash(g_jm);  // with ad-hoc smoothing term.

      //  H_jm_dash = H_jm + (smoothing term)
      H_jm_dash.AddSp(update_options_.tau_vec, H_sm);
      // g_jm_dash.BlasGemv(update_options_.mTauVec, H_sm, kNoTrans, e_1, 1.0);
      // g_jm_dash = g_jm + (smoothing term)
      g_jm_dash.AddVec(update_options_.tau_vec, y_sm);

      // if (gamma_jm == 0) continue;
      // no, we still want to update even with zero count.
#ifdef KALDI_PARANOID
      if (update_options_.tau_vec > 0)
        KALDI_ASSERT(H_jm_dash.IsPosDef());
#endif
      Vector<double> vhat_jm(model->v_[j].Row(m));
      SolverOptions opts;
      opts.name = "v";
      opts.K = update_options_.max_cond;
      opts.eps = update_options_.epsilon;
      double objf_impr_with_prior =
          SolveQuadraticProblem(H_jm_dash,
                                g_jm_dash,
                                opts,
                                &vhat_jm);
      
      SpMatrix<BaseFloat> H_jm_flt(H_jm);

      double objf_impr_noprior =
          (VecVec(vhat_jm, g_jm)
           - 0.5 * VecSpVec(vhat_jm, H_jm, vhat_jm))
          - (VecVec(model->v_[j].Row(m), g_jm)
             - 0.5 * VecSpVec(model->v_[j].Row(m), H_jm_flt, model->v_[j].Row(m)));
      model->v_[j].Row(m).CopyFromVec(vhat_jm);
      if (j < 3 && m < 2 && thread_id == 0) {
        KALDI_LOG << "Objf impr for j = " << (j) << " m = " << (m) << " is "
                  << (objf_impr_with_prior / (gamma_jm + 1.0e-20))
                  << " (with ad-hoc prior) "
                  << (objf_impr_noprior / (gamma_jm + 1.0e-20))
                  << " (no prior) over " << (gamma_jm) << " frames";
      }
      state_auxf_impr += objf_impr_with_prior;
      state_like_impr += objf_impr_noprior;
    }

    *auxf_impr += state_auxf_impr;
    *like_impr += state_like_impr;
    if (j < 10 && thread_id == 0) {
      KALDI_LOG << "Objf impr for state j = " << (j) << "  is "
                << (state_auxf_impr / (state_count + 1.0e-20))
                << " (with ad-hoc prior) "
                << (state_like_impr / (state_count + 1.0e-20))
                << " (no prior) over " << (state_count) << " frames";
    }
  }
}

double MleAmSgmmUpdater::UpdatePhoneVectors(const MleAmSgmmAccs &accs,
                                            AmSgmm *model,
                                            const vector< SpMatrix<double> > &H,
                                            const SpMatrix<double> &H_sm,
                                            const Vector<double> &y_sm) {
  KALDI_LOG << "Updating phone vectors";

  double count = 0.0, auxf_impr = 0.0, like_impr = 0.0;  // sum over all states

  for (int32 j = 0; j < accs.num_states_; j++) count += accs.gamma_[j].Sum();

  UpdatePhoneVectorsClass c(*this, accs, model, H, H_sm, y_sm,
                            &auxf_impr, &like_impr);
  RunMultiThreaded(c);

  auxf_impr /= (count + 1.0e-20);
  like_impr /= (count + 1.0e-20);
  KALDI_LOG << "**Overall objf impr for v is " << auxf_impr
            << "(with ad-hoc prior) " << like_impr << " (no prior) over "
            << (count) << " frames";
  // Choosing to return actual likelihood impr here.
  return like_impr;
}


/**
   This is as UpdatePhoneVectors but does not support smoothing terms or
   parallelization.  However, it does compute the auxiliary function
   after doing the update, and backtracks if it did not increase (due
   to the weight terms, increase is not mathematically guaranteed). */

double MleAmSgmmUpdater::UpdatePhoneVectorsChecked(const MleAmSgmmAccs &accs,
                                                   AmSgmm *model,
                                                   const vector< SpMatrix<double> > &H) {
  KALDI_LOG << "Updating phone vectors (and checking auxiliary function)";

  double tot_count = 0.0, tot_objf_impr = 0.0, tot_auxf_impr = 0.0;  // sum over all states

  for (int32 j = 0; j < accs.num_states_; j++) {
    for (int32 m = 0; m < model->NumSubstates(j); m++) {
      double gamma_jm = accs.gamma_[j].Row(m).Sum();
      SpMatrix<double> X_jm(accs.phn_space_dim_);  // = \sum_i \gamma_{jmi} H_i

      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        double gamma_jmi = accs.gamma_[j](m, i);
        if (gamma_jmi != 0.0)
          X_jm.AddSp(gamma_jmi, H[i]);
      }

      Vector<double> v_jm_orig(model->v_[j].Row(m)),
          v_jm(v_jm_orig);

      double exact_objf_start = 0.0, exact_objf = 0.0, auxf_impr = 0.0;
      int32 backtrack_iter, max_backtrack = 10;
      for (backtrack_iter = 0; backtrack_iter < max_backtrack; backtrack_iter++) {
        // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
        Vector<double> w_jm(accs.num_gaussians_);
        w_jm.AddMatVec(1.0, Matrix<double>(model->w_), kNoTrans,
                       v_jm, 0.0);
        w_jm.Add(-w_jm.LogSumExp());  // it is now log w_jm

        exact_objf = VecVec(w_jm, accs.gamma_[j].Row(m))
            + VecVec(v_jm, accs.y_[j].Row(m))
            -0.5 * VecSpVec(v_jm, X_jm, v_jm);

        if (backtrack_iter == 0.0) {
          exact_objf_start = exact_objf;
        } else {
          if (exact_objf >= exact_objf_start) {
            break;  // terminate backtracking.
          } else {
            KALDI_LOG << "Backtracking computation of v_jm for j = " << j
                      << " and m = " << m << " because objf changed by "
                      << (exact_objf-exact_objf_start) << " [vs. predicted:] "
                      << auxf_impr;
            v_jm.AddVec(1.0, v_jm_orig);
            v_jm.Scale(0.5);
          }
        }

        if (backtrack_iter == 0) {  // computing updated value.
          w_jm.ApplyExp();  // it is now w_jm
          SpMatrix<double> H_jm(X_jm);
          Vector<double> g_jm(accs.y_[j].Row(m));
          for (int32 i = 0; i < accs.num_gaussians_; i++) {
            double gamma_jmi = accs.gamma_[j](m, i);
            double quadratic_term = std::max(gamma_jmi, gamma_jm * w_jm(i));
            double scalar = gamma_jmi - gamma_jm * w_jm(i) + quadratic_term
                * VecVec(model->w_.Row(i), model->v_[j].Row(m));
            g_jm.AddVec(scalar, model->w_.Row(i));
            if (quadratic_term > 1.0e-10) {
              H_jm.AddVec2(static_cast<BaseFloat>(quadratic_term), model->w_.Row(i));
            }
          }
          SolverOptions opts;
          opts.name = "v";
          opts.K = update_options_.max_cond;
          opts.eps = update_options_.epsilon;
          auxf_impr = SolveQuadraticProblem(H_jm, g_jm, opts, &v_jm);
        }
      }
      double objf_impr = exact_objf - exact_objf_start;
      tot_count += gamma_jm;
      tot_objf_impr += objf_impr;
      tot_auxf_impr += auxf_impr;
      if (backtrack_iter == max_backtrack) {
        KALDI_WARN << "Backtracked " << max_backtrack << " times [not updating]\n";
      } else {
        model->v_[j].Row(m).CopyFromVec(v_jm);
      }

      if (j < 3 && m < 2) {
        KALDI_LOG << "Objf impr for j = " << (j) << " m = " << (m) << " is "
                  << objf_impr << " vs. quadratic auxf impr (before backtrack) "
                  << auxf_impr;
      }
    }
  }

  tot_objf_impr /= (tot_count + 1.0e-20);
  tot_auxf_impr /= (tot_count + 1.0e-20);
  KALDI_LOG << "**Overall objf impr for v is " << tot_objf_impr
            << " (auxf impr before backtracking:) " << tot_auxf_impr
            << " over " << tot_count << " frames";
  // Choosing to return actual likelihood impr here.
  return tot_objf_impr;
}



class UpdatePhoneVectorsCheckedFromClusterableClass: public MultiThreadable { // For multi-threaded.
 public:
  UpdatePhoneVectorsCheckedFromClusterableClass(
      MleAmSgmmUpdater *updater,
      const std::vector<SgmmClusterable*> &stats,
      const std::vector<SpMatrix<double> > &H,
      AmSgmm *model,
      double *count,
      double *like_impr):
      updater_(updater), stats_(stats), H_(H), model_(model),
      count_ptr_(count), count_(0.0),
      like_impr_ptr_(like_impr), like_impr_(0.0)
      { }
  
  ~UpdatePhoneVectorsCheckedFromClusterableClass() {
    *count_ptr_ += count_;
    *like_impr_ptr_ += like_impr_;
  }
  
  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to the total sums in the destructor.
    updater_->UpdatePhoneVectorsCheckedFromClusterableInternal(
        stats_, H_, model_, &count_, &like_impr_, num_threads_, thread_id_);
  }
 private:
  MleAmSgmmUpdater *updater_;
  const std::vector<SgmmClusterable*> &stats_;
  const std::vector<SpMatrix<double> > &H_;
  AmSgmm *model_;
  double *count_ptr_;
  double count_;
  double *like_impr_ptr_;
  double like_impr_;
};


double MleAmSgmmUpdater::UpdatePhoneVectorsCheckedFromClusterable(
    const std::vector<SgmmClusterable*> &stats,
    const vector< SpMatrix<double> > &H,
    AmSgmm *model) {
  KALDI_LOG << "Updating phone vectors using stats from Clusterable class "
      "(and checking auxiliary function)";
  double count = 0.0, like_impr = 0.0;
  
  UpdatePhoneVectorsCheckedFromClusterableClass c(this, stats, H, model,
                                                  &count, &like_impr);
  RunMultiThreaded(c);

  KALDI_LOG << "**Overall objf impr for v is " << (like_impr / count)
            << " over " << count << " frames.";
  
  return like_impr / count;
}


void MleAmSgmmUpdater::UpdatePhoneVectorsCheckedFromClusterableInternal(
    const std::vector<SgmmClusterable*> &stats,
    const vector< SpMatrix<double> > &H,
    AmSgmm *model,
    double *count_ptr,
    double *like_impr_ptr,
    int32 num_threads,
    int32 thread_id) {
  
  int32 block_size = (model->NumPdfs() + (num_threads-1)) / num_threads,
                j_start = block_size * thread_id,
                j_end = std::min(model->NumPdfs(), j_start + block_size);
  
  double tot_count = 0.0, tot_objf_impr = 0.0, tot_auxf_impr = 0.0;  // sum over all states

  KALDI_ASSERT(model->NumPdfs() == static_cast<int32>(stats.size()));
  int32 num_gauss = model->NumGauss();
  for (int32 j = j_start; j < j_end; j++) {
    KALDI_ASSERT(model->NumSubstates(j) == 1 &&
                 "This function only works if there is 1 substate per state.");
    int32 m = 0; // sub-state index.
    const Vector<double> &gamma = stats[j]->gamma();
    const Vector<double> &y = stats[j]->y();
      
    double gamma_jm = gamma.Sum();
    SpMatrix<double> X_jm(model->PhoneSpaceDim());  // = \sum_i \gamma_{jmi} H_i

    for (int32 i = 0; i < num_gauss; i++) {
      double gamma_jmi = gamma(i);
      if (gamma_jmi != 0.0)
        X_jm.AddSp(gamma_jmi, H[i]);
    }

    Vector<double> v_jm_orig(model->v_[j].Row(m)),
        v_jm(v_jm_orig);
    
    double exact_objf_start = 0.0, exact_objf = 0.0, auxf_impr = 0.0;
    int32 backtrack_iter, max_backtrack = 10;
    for (backtrack_iter = 0; backtrack_iter < max_backtrack; backtrack_iter++) {
      // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
      Vector<double> w_jm(num_gauss);
      w_jm.AddMatVec(1.0, Matrix<double>(model->w_), kNoTrans,
                     v_jm, 0.0);
      w_jm.Add(-w_jm.LogSumExp());  // it is now log w_jm
      
      exact_objf = VecVec(w_jm, gamma)
          + VecVec(v_jm, y)
          -0.5 * VecSpVec(v_jm, X_jm, v_jm);
      
      if (backtrack_iter == 0.0) {
        exact_objf_start = exact_objf;
      } else {
        if (exact_objf >= exact_objf_start) {
          break;  // terminate backtracking.
        } else {
          KALDI_LOG << "Backtracking computation of v_jm for j = " << j
                    << " and m = " << m << " because objf changed by "
                    << (exact_objf-exact_objf_start) << " [vs. predicted:] "
                    << auxf_impr;
          v_jm.AddVec(1.0, v_jm_orig);
          v_jm.Scale(0.5);
        }
      }

      if (backtrack_iter == 0) {  // computing updated value.
        w_jm.ApplyExp();  // it is now w_jm
        SpMatrix<double> weight_2nd_deriv(model->PhoneSpaceDim()); // actually
        // negatived 2nd derivative.
        Vector<double> num_deriv(model->PhoneSpaceDim());
        Vector<double> den_deriv(model->PhoneSpaceDim());
        
        // We modify the optimization to use the exact 2nd derivative.
        // Because we do checking and backtracking, the loss of
        // natural stability is OK.
        for (int32 i = 0; i < num_gauss; i++) {
          double gamma_jmi = gamma(i);
          SubVector<BaseFloat> wi(model->w_, i);
          num_deriv.AddVec(gamma_jmi, wi);
          double scalar = gamma_jm * w_jm(i); // expected count.
          den_deriv.AddVec(scalar, wi);
          if (scalar > 1.0e-10) // if-statement is a speedup
            weight_2nd_deriv.AddVec2(static_cast<BaseFloat>(scalar), wi);
        }
        Vector<double> total_linear_term(y);
        total_linear_term.AddVec(1.0, num_deriv);
        total_linear_term.AddVec(-1.0, den_deriv);
        if (gamma_jm > 0.0)
          weight_2nd_deriv.AddVec2(-1.0/gamma_jm, den_deriv);
        
        total_linear_term.AddSpVec(1.0, weight_2nd_deriv, v_jm, 1.0);
        // we want the derivatives around zero, not around the current point.
        // Correction for this.
        
        SpMatrix<double> total_quadratic_term(weight_2nd_deriv);
        total_quadratic_term.AddSp(1.0, X_jm);

        SolverOptions opts;
        opts.name = "v";
        opts.K = update_options_.max_cond;
        opts.eps = update_options_.epsilon;
        auxf_impr = SolveQuadraticProblem(total_quadratic_term,
                                          total_linear_term, opts, &v_jm);
      }
    }
    double objf_impr = exact_objf - exact_objf_start;
    tot_count += gamma_jm;
    tot_objf_impr += objf_impr;
    tot_auxf_impr += auxf_impr;
    if (backtrack_iter == max_backtrack) {
      KALDI_WARN << "Backtracked " << max_backtrack << " times [not updating]\n";
    } else {
      model->v_[j].Row(m).CopyFromVec(v_jm);
    }
    if (j < 3) {
      KALDI_LOG << "Objf impr for j = " << (j) << " m = " << (m) << " is "
                << objf_impr << " vs. quadratic auxf impr (before backtrack) "
                << auxf_impr;
    }
  }

  *like_impr_ptr = tot_objf_impr;  
  *count_ptr = tot_count;
  
  tot_objf_impr /= (tot_count + 1.0e-20);
  tot_auxf_impr /= (tot_count + 1.0e-20);

  if (j_start == 0)
    KALDI_LOG << "**For first batch: objf impr for v is " << tot_objf_impr
              << " (auxf impr before backtracking:) " << tot_auxf_impr
              << " over " << tot_count << " frames";
}


void MleAmSgmmUpdater::RenormalizeV(const MleAmSgmmAccs &accs,
                                    AmSgmm *model,
                                    const SpMatrix<double> &H_sm) {
  SpMatrix<double> Sigma(accs.phn_space_dim_);
  int32 count = 0;
  for (int32 j = 0; j < accs.num_states_; j++) {
    for (int32 m = 0; m < model->NumSubstates(j); m++) {
      count++;
      Sigma.AddVec2(static_cast<BaseFloat>(1.0), model->v_[j].Row(m));
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
  TpMatrix<double> L(accs.phn_space_dim_);
  L.Cholesky(Sigma);
  TpMatrix<double> LInv(L);
  LInv.Invert();

  Matrix<double> tmpL(accs.phn_space_dim_, accs.phn_space_dim_);
  tmpL.CopyFromTp(L);

  SpMatrix<double> H_sm_proj(accs.phn_space_dim_);
  H_sm_proj.AddMat2Sp(1.0, tmpL, kTrans, H_sm, 0.0);
  // H_sm_proj := L^{T} * H_sm * L.
  // This is right because we would transform the vectors themselves
  // by L^{-1}, and H_sm is like the inverse of the vectors,
  // so it's {L^{-1}}^{-T} = L^T.

  Matrix<double> U(accs.phn_space_dim_, accs.phn_space_dim_);
  Vector<double> eigs(accs.phn_space_dim_);
  H_sm_proj.SymPosSemiDefEig(&eigs, &U, 1.0);  // 1.0 means no checking +ve def -> faster
  KALDI_LOG << "Note on the next diagnostic: the first number is generally not "
            << "that meaningful as it relates to the static offset";
  H_sm_proj.PrintEigs("H_sm_proj (Significance of dims in vector space.. note)");

  // Transform on vectors is U^T L^{-1}.
  // Why?  Because transform on H_sm is T =U^T L^T
  // and we want T^{-T} by normal rules of vector/covector and we
  // have (U^T L^T)^{-T} = (L U)^{-1} = U^T L^{-1}.
  Matrix<double> Trans(accs.phn_space_dim_, accs.phn_space_dim_);  // T^{-T}
  Matrix<double> tmpLInv(accs.phn_space_dim_, accs.phn_space_dim_);
  tmpLInv.CopyFromTp(LInv);
  Trans.AddMatMat(1.0, U, kTrans, tmpLInv, kNoTrans, 0.0);
  Matrix<double> TransInv(Trans);
  TransInv.Invert();  // T in above...

#ifdef KALDI_PARANOID
  {
    SpMatrix<double> H_sm_tmp(accs.phn_space_dim_);
    H_sm_tmp.AddMat2Sp(1.0, TransInv, kTrans, H_sm, 0.0);
    KALDI_ASSERT(H_sm_tmp.IsDiagonal(0.1));
  }
  {
    SpMatrix<double> Sigma_tmp(accs.phn_space_dim_);
    Sigma_tmp.AddMat2Sp(1.0, Trans, kNoTrans, Sigma, 0.0);
    KALDI_ASSERT(Sigma_tmp.IsUnit(0.1));
  }
#endif

  for (int32 j = 0; j < accs.num_states_; j++) {
    for (int32 m = 0; m < model->NumSubstates(j); m++) {
      Vector<double> tmp(accs.phn_space_dim_);
      tmp.AddMatVec(1.0, Trans, kNoTrans, Vector<double>(model->v_[j].Row(m)), 0.0);
      model->v_[j].Row(m).CopyFromVec(tmp);
    }
  }
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    Vector<double> tmp(accs.phn_space_dim_);
    tmp.AddMatVec(1.0, TransInv, kTrans, Vector<double>(model->w_.Row(i)), 0.0);
    model->w_.Row(i).CopyFromVec(tmp);

    Matrix<double> tmpM(accs.feature_dim_, accs.phn_space_dim_);
    // Multiplying on right not left so must not transpose TransInv.
    tmpM.AddMatMat(1.0, Matrix<double>(model->M_[i]), kNoTrans,
                   TransInv, kNoTrans, 0.0);
    model->M_[i].CopyFromMat(tmpM);
  }
  KALDI_LOG << "Renormalized subspace.";
}

double MleAmSgmmUpdater::UpdateM(const MleAmSgmmAccs &accs,
                                 AmSgmm *model) {
  double tot_count = 0.0, tot_like_impr = 0.0;
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    double gamma_i = 0.0;
    for (int32 j = 0; j < accs.num_states_; j++)
      for (int32 m = 0; m < model->NumSubstates(j); m++)
        gamma_i += accs.gamma_[j](m, i);

    if (gamma_i < accs.feature_dim_) {
      KALDI_WARN << "For component " << i << ": not updating M due to very "
                 << "small count (=" << gamma_i << ").";
      continue;
    }

    SolverOptions opts;
    opts.name = "M";
    opts.K = update_options_.max_cond;
    opts.eps = update_options_.epsilon;
    
    Matrix<double> Mi(model->M_[i]);
    double impr = SolveQuadraticMatrixProblem(Q_[i], accs.Y_[i],
                                              SpMatrix<double>(model->SigmaInv_[i]),
                                              opts, &Mi);
    model->M_[i].CopyFromMat(Mi);

    if (i < 10) {
      KALDI_VLOG(2) << "Objf impr for projection M for i = " << i << ", is "
                    << (impr/(gamma_i + 1.0e-20)) << " over " << gamma_i
                    << " frames";
    }
    tot_count += gamma_i;
    tot_like_impr += impr;
  }
  tot_like_impr /= (tot_count + 1.0e-20);
  KALDI_LOG << "Overall objective function improvement for model projections "
            << "M is " << tot_like_impr << " over " << tot_count << " frames";
  return tot_like_impr;
}

// Estimate the parameters of a Gaussian prior over the M matrices. There are
// as many mean matrices as UBM size and two covariance matrices for the rows
// of M and columns of M. The prior means M_i are fixed to the unadapted values.
// This is what was done in Lu, et al. "Maximum a posteriori adaptation of
// subspace Gaussian mixture models for cross-lingual speech recognition",
// ICASSP 2012.
void MleAmSgmmUpdater::ComputeMPrior(AmSgmm *model) {
  KALDI_ASSERT(update_options_.map_M_prior_iters > 0);
  int32 Ddim = model->FeatureDim();
  int32 Sdim = model->PhoneSpaceDim();
  int32 nGaussians = model->NumGauss();

  // inverse variance of the columns of M: dim is # of rows
  model->col_cov_inv_.Resize(Ddim);
  // inverse covariance of the rows of M: dim is # of columns
  model->row_cov_inv_.Resize(Sdim);

  model->col_cov_inv_.SetUnit();
  model->row_cov_inv_.SetUnit();

  if (model->M_prior_.size() == 0) {
    model->M_prior_.resize(nGaussians);
    for (int32 i = 0; i < nGaussians; i++) {
      model->M_prior_[i].Resize(Ddim, Sdim);
      model->M_prior_[i].CopyFromMat(model->M_[i]); // We initialize Mpri as this
    }
  }

  if (update_options_.full_col_cov || update_options_.full_row_cov) {
    Matrix<double> avg_M(Ddim, Sdim);  // average of the Gaussian prior means
    for (int32 i = 0; i < nGaussians; i++)
      avg_M.AddMat(1.0, Matrix<double>(model->M_prior_[i]));
    avg_M.Scale(1.0 / nGaussians);

    Matrix<double> MDiff(Ddim, Sdim);
    for (int32 iter = 0; iter < update_options_.map_M_prior_iters; iter++) {
      { // diagnostic block.
        double prior_like = -0.5 * nGaussians * (Ddim * Sdim * log(2 * M_PI)
                + Sdim * (-model->row_cov_inv_.LogPosDefDet())
                + Ddim * (-model->col_cov_inv_.LogPosDefDet()));
        for (int32 i = 0; i < nGaussians; i++) {
          MDiff.CopyFromMat(Matrix<double>(model->M_prior_[i]));
          MDiff.AddMat(-1.0, avg_M);  // MDiff = M_{i} - avg(M)
          SpMatrix<double> tmp(Ddim);
          // tmp = MDiff.Omega_r^{-1}*MDiff^T.
          tmp.AddMat2Sp(1.0, MDiff, kNoTrans,
                        SpMatrix<double>(model->row_cov_inv_), 0.0);
          prior_like -= 0.5 * TraceSpSp(tmp, SpMatrix<double>(model->col_cov_inv_));
        }
        KALDI_LOG << "Before iteration " << iter
            << " of updating prior over M, log like per dimension modeled is "
            << prior_like / (nGaussians * Ddim * Sdim);
      }

      // First estimate the column covariances (\Omega_r in paper)
      if (update_options_.full_col_cov) {
        size_t limited;
        model->col_cov_inv_.SetZero();
        for (int32 i = 0; i < nGaussians; i++) {
          MDiff.CopyFromMat(Matrix<double>(model->M_prior_[i]));
          MDiff.AddMat(-1.0, avg_M);  // MDiff = M_{i} - avg(M)
          // Omega_r += 1/(D*I) * Mdiff * Omega_c^{-1} * Mdiff^T
          model->col_cov_inv_.AddMat2Sp(1.0 / (Ddim * nGaussians),
                                        Matrix<BaseFloat>(MDiff), kNoTrans,
                                        model->row_cov_inv_, 1.0);
        }
        model->col_cov_inv_.PrintEigs("col_cov");
        limited = model->col_cov_inv_.LimitCond(update_options_.max_cond,
                                                true /*invert the matrix*/);
        if (limited != 0) {
          KALDI_LOG << "Computing column covariances for M: limited " << limited
                    << " singular values, max condition is "
                    << update_options_.max_cond;
        }
      }

      // Now estimate the row covariances (\Omega_c in paper)
      if (update_options_.full_row_cov) {
        size_t limited;
        model->row_cov_inv_.SetZero();
        for (int32 i = 0; i < nGaussians; i++) {
          MDiff.CopyFromMat(Matrix<double>(model->M_prior_[i]));
          MDiff.AddMat(-1.0, avg_M);  // MDiff = M_{i} - avg(M)
          // Omega_c += 1/(S*I) * Mdiff^T * Omega_r^{-1} * Mdiff.
          model->row_cov_inv_.AddMat2Sp(1.0 / (Sdim * nGaussians),
                                        Matrix<BaseFloat>(MDiff), kTrans,
                                        model->col_cov_inv_, 1.0);
        }
        model->row_cov_inv_.PrintEigs("row_cov");
        limited = model->row_cov_inv_.LimitCond(update_options_.max_cond,
                                                true /*invert the matrix*/);
        if (limited != 0) {
          KALDI_LOG << "Computing row covariances for M: limited " << limited
                    << " singular values, max condition is "
                    << update_options_.max_cond;
        }
      }
    }  // end iterations
  }
}


// MAP adaptation of M with a matrix-variate Gaussian prior
double MleAmSgmmUpdater::MapUpdateM(const MleAmSgmmAccs &accs, AmSgmm *model) {
  int32 Ddim = model->FeatureDim();
  int32 Sdim = model->PhoneSpaceDim();
  int32 nGaussians = model->NumGauss();

  KALDI_LOG << "Prior smoothing parameter: Tau = " << update_options_.tau_map_M;
  if (model->M_prior_.size() == 0 || model->col_cov_inv_.NumRows() == 0
      || model->row_cov_inv_.NumRows() == 0) {
    KALDI_LOG << "Computing the prior first";
    ComputeMPrior(model);
  }

  Matrix<double> G(Ddim, Sdim);
  // \tau \Omega_c^{-1} avg(M) \Omega_r^{-1}, depends on Gaussian index
  Matrix<double> prior_term_i(Ddim, Sdim);
  SpMatrix<double> P2(model->col_cov_inv_);
  SpMatrix<double> Q2(model->row_cov_inv_);
  Q2.Scale(update_options_.tau_map_M);

  double totcount = 0.0, tot_like_impr = 0.0;
  for (int32 i = 0; i < nGaussians; ++i) {
    double gamma_i = 0.0;
    for (int32 j = 0; j < accs.num_states_; ++j)
      for (int32 m = 0; m < model->NumSubstates(j); ++m)
        gamma_i += accs.gamma_[j](m, i);

    if (gamma_i < accs.feature_dim_) {
      KALDI_WARN << "For component " << i << ": not updating M due to very "
                 << "small count (=" << gamma_i << ").";
      continue;
    }

    Matrix<double> tmp(Ddim, Sdim, kSetZero);
    tmp.AddSpMat(1.0, SpMatrix<double>(model->col_cov_inv_),
                 Matrix<double>(model->M_prior_[i]), kNoTrans, 0.0);
    prior_term_i.AddMatSp(update_options_.tau_map_M, tmp, kNoTrans,
                          SpMatrix<double>(model->row_cov_inv_), 0.0);

    Matrix<double> SigmaY(Ddim, Sdim, kSetZero);
    SigmaY.AddSpMat(1.0, SpMatrix<double>(model->SigmaInv_[i]), accs.Y_[i],
                    kNoTrans, 0.0);
    G.CopyFromMat(SigmaY);  // G = \Sigma_{i}^{-1} Y_{i}
    G.AddMat(1.0, prior_term_i); // G += \tau \Omega_c^{-1} avg(M) \Omega_r^{-1}
    SpMatrix<double> P1(model->SigmaInv_[i]);
    Matrix<double> Mi(model->M_[i]);

    SolverOptions opts;
    opts.name = "M";
    opts.K = update_options_.max_cond;
    opts.eps = update_options_.epsilon;
    
    double impr = SolveDoubleQuadraticMatrixProblem(G, P1, P2, Q_[i], Q2, opts, &Mi);
    model->M_[i].CopyFromMat(Mi);
    if (i < 10) {
      KALDI_LOG << "Objf impr for projection M for i = " << i << ", is "
                << (impr / (gamma_i + 1.0e-20)) << " over " << gamma_i
                << " frames";
    }
    totcount += gamma_i;
    tot_like_impr += impr;
  }
  tot_like_impr /= (totcount + 1.0e-20);
  KALDI_LOG << "Overall objective function improvement for model projections "
            << "M is " << tot_like_impr << " over " << totcount << " frames";
  return tot_like_impr;
}


/// This function gets stats used inside UpdateWParallel, where it accumulates
/// the F_i and g_i quantities.  Note: F_i is viewed as a vector of SpMatrix
/// (one for each i); each row of F_i is viewed as an SpMatrix even though
/// it's stored as a vector....
/// Note: w is just a double-precision copy of the matrix model->w_

// static
void MleAmSgmmUpdater::UpdateWParallelGetStats(const MleAmSgmmAccs &accs,
                                               const AmSgmm &model,
                                               const Matrix<double> &w,
                                               Matrix<double> *F_i,
                                               Matrix<double> *g_i,
                                               double *tot_like,
                                               int32 num_threads, 
                                               int32 thread_id) {

  // Accumulate stats from a block of states (this gets called in parallel).
  int32 block_size = (accs.num_states_ + (num_threads-1)) / num_threads,
      j_start = block_size * thread_id,
      j_end = std::min(accs.num_states_, j_start + block_size);

  // Unlike in the report the inner most loop is over Gaussians, where
  // per-gaussian statistics are accumulated. This is more memory demanding
  // but more computationally efficient, as outer product v_{jvm} v_{jvm}^T
  // is computed only once for all gaussians.

  SpMatrix<double> v_vT(accs.phn_space_dim_);
  
  for (int32 j = j_start; j < j_end; j++) {
    int32 num_substates = model.NumSubstates(j);
    Matrix<double> w_jm(num_substates, accs.num_gaussians_);
    // The linear term and quadratic term for each Gaussian-- two scalars
    // for each Gaussian, they appear in the accumulation formulas.
    Matrix<double> linear_term(num_substates, accs.num_gaussians_);
    Matrix<double> quadratic_term(num_substates, accs.num_gaussians_);
    Matrix<double> v_vT_m(num_substates,
                          (accs.phn_space_dim_*(accs.phn_space_dim_+1))/2);

    // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
    Matrix<double> v_j_double(model.v_[j]);
    w_jm.AddMatMat(1.0, v_j_double, kNoTrans, w, kTrans, 0.0);
      
    for (int32 m = 0; m < model.NumSubstates(j); m++) {
      double gamma_jm = accs.gamma_[j].Row(m).Sum();

      w_jm.Row(m).Add(-1.0 * w_jm.Row(m).LogSumExp());
      *tot_like += VecVec(w_jm.Row(m), accs.gamma_[j].Row(m));
      w_jm.Row(m).ApplyExp();
      v_vT.SetZero();
      // v_vT := v_{jkm} v_{jkm}^T
      v_vT.AddVec2(static_cast<BaseFloat>(1.0), v_j_double.Row(m));
      v_vT_m.Row(m).CopyFromPacked(v_vT); // a bit wasteful, but does not dominate.
        
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        // Suggestion: g_jkm can be computed more efficiently
        // using the Vector/Matrix routines for all i at once
        // linear term around cur value.
        linear_term(m, i) = accs.gamma_[j](m, i) - gamma_jm * w_jm(m, i);
        quadratic_term(m, i) = std::max(accs.gamma_[j](m, i),
                                        gamma_jm * w_jm(m, i));
      }
    } // loop over substates
    g_i->AddMatMat(1.0, linear_term, kTrans, v_j_double, kNoTrans, 1.0);
    F_i->AddMatMat(1.0, quadratic_term, kTrans, v_vT_m, kNoTrans, 1.0);
  } // loop over states
}

// The parallel weight update, in the paper.
double MleAmSgmmUpdater::UpdateWParallel(const MleAmSgmmAccs &accs,
                                         AmSgmm *model) {
  KALDI_LOG << "Updating weight projections";

  // tot_like_{after, before} are totals over multiple iterations,
  // not valid likelihoods. but difference is valid (when divided by tot_count).
  double tot_predicted_like_impr = 0.0, tot_like_before = 0.0,
      tot_like_after = 0.0;
  
  Matrix<double> g_i(accs.num_gaussians_, accs.phn_space_dim_);
  // View F_i as a vector of SpMatrix.
  Matrix<double> F_i(accs.num_gaussians_,
                     (accs.phn_space_dim_*(accs.phn_space_dim_+1))/2);
  
  Matrix<double> w(model->w_);
  double tot_count = 0.0;
  for (int32 j = 0; j < accs.num_states_; j++) tot_count += accs.gamma_[j].Sum();
  
  for (int iter = 0; iter < update_options_.weight_projections_iters; iter++) {
    F_i.SetZero();
    g_i.SetZero();
    double k_like_before = 0.0;

    UpdateWParallelClass c(accs, *model, w, &F_i, &g_i, &k_like_before);
    RunMultiThreaded(c);
    
    Matrix<double> w_orig(w);
    double k_predicted_like_impr = 0.0, k_like_after = 0.0;
    double min_step = 0.001, step_size;
    for (step_size = 1.0; step_size >= min_step; step_size /= 2) {
      k_predicted_like_impr = 0.0;
      k_like_after = 0.0;

      SolverOptions opts;
      opts.name = "w";
      opts.K = update_options_.max_cond;
      opts.eps = update_options_.epsilon;
      
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        // auxf is formulated in terms of change in w.
        Vector<double> delta_w(accs.phn_space_dim_);
        // returns objf impr with step_size = 1,
        // but it may not be 1 so we recalculate it.
        SpMatrix<double> this_F_i(accs.phn_space_dim_);
        this_F_i.CopyFromVec(F_i.Row(i));
        SolveQuadraticProblem(this_F_i, g_i.Row(i), opts, &delta_w);

        delta_w.Scale(step_size);
        double predicted_impr = VecVec(delta_w, g_i.Row(i)) -
            0.5 * VecSpVec(delta_w,  this_F_i, delta_w);

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
      Vector<double> w_jm_vec(accs.num_gaussians_);
      for (int32 j = 0; j < accs.num_states_; j++) {
        for (int32 m = 0; m < model->NumSubstates(j); m++) {
          w_jm_vec.AddMatVec(1.0, w, kNoTrans, Vector<double>(model->v_[j].Row(m)), 0.0);
          w_jm_vec.Add((-1.0) * w_jm_vec.LogSumExp());
          k_like_after += VecVec(w_jm_vec, accs.gamma_[j].Row(m));
        }
      }
      KALDI_VLOG(2) << "For iteration " << (iter) << ", updating w gives "
                    << "predicted per-frame like impr "
                    << (k_predicted_like_impr / tot_count) << ", actual "
                    << ((k_like_after - k_like_before) / tot_count) << ", over "
                    << (tot_count) << " frames";
      if (k_like_after < k_like_before) {
        w.CopyFromMat(w_orig);  // Undo what we computed.
        if (fabs(k_like_after - k_like_before) / tot_count < 1.0e-05) {
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
      tot_predicted_like_impr += k_predicted_like_impr;
      tot_like_after += k_like_after;
      tot_like_before += k_like_before;
    }
  }

  model->w_.CopyFromMat(w);

  tot_predicted_like_impr /= tot_count;
  tot_like_after = (tot_like_after - tot_like_before) / tot_count;
  KALDI_LOG << "**Overall objf impr for w is " << tot_predicted_like_impr
            << ", actual " << tot_like_after << ", over "
            << tot_count << " frames";
  return tot_like_after;
}

double MleAmSgmmUpdater::UpdateWSequential(
    const MleAmSgmmAccs &accs, AmSgmm *model) {
  // Sequential version, in paper.
  /* This is the approach for the weight projections that
   * I originally implemented, in which we test the auxiliary function
   improvement for each i that we update. This requires some
   careful bookkeeping.  It means that we need to store the
   total of the un-normalized weights for each j, m.  */

  KALDI_LOG << "Updating weight projections [original approach, checking each"
            << "Gaussian component].";

  SpMatrix<double> v_vT(accs.phn_space_dim_);
  // tot_like_{after, before} are totals over multiple iterations,
  // not valid likelihoods...
  // but difference is valid (when divided by tot_count).
  double tot_delta_predicted = 0.0, tot_delta_observed = 0.0,
      tot_count = 0.0;

  Vector<double> w_jm(accs.num_gaussians_);
  Vector<double> g_i(accs.phn_space_dim_);
  SpMatrix<double> F_i(accs.phn_space_dim_);

  double k_count = 0.0;
  // Total count in each substate.
  std::vector< Vector<double> > gamma_jm(accs.num_states_);
  for (int32 j = 0; j < accs.num_states_; j++) {   // Initialize gamma_jm
    gamma_jm[j].Resize(model->NumSubstates(j));
    for (int32 m = 0; m < model->NumSubstates(j); m++) {
      k_count += (gamma_jm[j](m) = accs.gamma_[j].Row(m).Sum());
    }
  }

  Matrix<double> w(model->w_);

  for (int iter = 0; iter < update_options_.weight_projections_iters; iter++) {
    double k_delta_predicted = 0.0, k_delta_observed = 0.0;

    // log total of un-normalized weights for each j, m
    std::vector< Vector<double> > weight_tots(accs.num_states_);

    // Initialize weight_tots
    for (int32 j = 0; j < accs.num_states_; j++) {
      weight_tots[j].Resize(model->NumSubstates(j));
      for (int32 m = 0; m < model->NumSubstates(j); m++) {
        w_jm.AddMatVec(1.0, w, kNoTrans, Vector<double>(model->v_[j].Row(m)), 0.0);
        weight_tots[j](m) = w_jm.LogSumExp();
      }
    }

    for (int32 i = 0; i < accs.num_gaussians_; i++) {
      F_i.SetZero();
      g_i.SetZero();
      SubVector<double> w_i = w.Row(i);

      for (int32 j = 0; j < accs.num_states_; j++) {
        for (int32 m = 0; m < model->NumSubstates(j); m++) {
          double this_unnormalized_weight = VecVec(w_i, model->v_[j].Row(m));
          double normalizer = weight_tots[j](m);
          double this_log_w = this_unnormalized_weight - normalizer,
              this_w = exp(this_log_w),
              substate_count = gamma_jm[j](m),
              this_count = accs.gamma_[j](m, i);

          double linear_term = this_count - substate_count * this_w;
          double quadratic_term = std::max(this_count, substate_count * this_w);

          g_i.AddVec(linear_term, model->v_[j].Row(m));
          // should not ever be zero, but check anyway.
          if (quadratic_term != 0.0)
            F_i.AddVec2(static_cast<BaseFloat>(quadratic_term), model->v_[j].Row(m));
        }
      }

      SolverOptions opts;
      opts.name = "w";
      opts.K = update_options_.max_cond;
      opts.eps = update_options_.epsilon;
      
      // auxf is formulated in terms of change in w.
      Vector<double> delta_w(accs.phn_space_dim_);
      // returns objf impr with step_size = 1,
      // but it may not be 1 so we recalculate it.
      SolveQuadraticProblem(F_i,
                            g_i,
                            opts,
                            &delta_w);

      try {  // In case we have a problem in LogSub.
        double step_size, min_step = 0.0001;
        for (step_size = 1.0; step_size >= min_step; step_size /= 2) {
          Vector<double> new_w_i(w_i);
          // copy it in case we do not commit this change.
          std::vector<Vector<double> > new_weight_tots(weight_tots);
          new_w_i.AddVec(step_size, delta_w);
          double predicted_impr = step_size * VecVec(delta_w, g_i) -
              0.5 * step_size * step_size * VecSpVec(delta_w,  F_i, delta_w);
          if (predicted_impr < -0.1) {
            KALDI_WARN << "Negative predicted auxf improvement " <<
                (predicted_impr) << ", not updating this gaussian " <<
                "(either numerical problems or a code mistake.";
            break;
          }
          // Now compute observed objf change.
          double observed_impr = 0.0, this_tot_count = 0.0;

          for (int32 j = 0; j < accs.num_states_; j++) {
            for (int32 m = 0; m < model->NumSubstates(j); m++) {
              double old_unnorm_weight = VecVec(w_i, model->v_[j].Row(m)),
                  new_unnorm_weight = VecVec(new_w_i, model->v_[j].Row(m)),
                  substate_count = gamma_jm[j](m),
                  this_count = accs.gamma_[j](m, i);
              this_tot_count += this_count;
              observed_impr += this_count *  // from numerator.
                  (new_unnorm_weight - old_unnorm_weight);
              double old_normalizer = new_weight_tots[j](m), delta;
              if (new_unnorm_weight > old_unnorm_weight) {
                delta = LogAdd(0, LogSub(new_unnorm_weight - old_normalizer,
                                         old_unnorm_weight - old_normalizer));
              } else {
                delta = LogSub(0, LogSub(old_unnorm_weight - old_normalizer,
                                         new_unnorm_weight - old_normalizer));
                // The if-statement above is equivalent to:
                // delta = LogAdd(LogSub(0,
                // old_unnorm_weight-old_normalizer),
                // new_unnorm_weight-old_normalizer)
                // but has better behaviour numerically.
              }
              observed_impr -= substate_count * delta;
              new_weight_tots[j](m) += delta;
            }
          }
          if (observed_impr < 0.0) {  // failed, so we reduce step size.
            KALDI_LOG << "Updating weights, for i = " << (i) << ", predicted "
                "auxf: " << (predicted_impr/(this_tot_count + 1.0e-20))
                      << ", observed " << observed_impr/(this_tot_count + 1.0e-20)
                      << " over " << this_tot_count << " frames. Reducing step size "
                      << "to " << (step_size/2);
            if (predicted_impr / (this_tot_count + 1.0e-20) < 1.0e-07) {
              KALDI_WARN << "Not updating this weight vector as auxf decreased"
                         << " probably due to numerical issues (since small change).";
              break;
            }
          } else {
            if (i < 10)
              KALDI_LOG << "Updating weights, for i = " << (i)
                        << ", auxf change per frame is" << ": predicted " <<
                  (predicted_impr /(this_tot_count + 1.0e-20)) << ", observed "
                        << (observed_impr / (this_tot_count + 1.0e-20))
                        << " over " << (this_tot_count) << " frames.";

            k_delta_predicted += predicted_impr;
            k_delta_observed += observed_impr;
            w.Row(i).CopyFromVec(new_w_i);
            weight_tots = new_weight_tots;  // Copy over normalizers.
            break;
          }
        }
      } catch(...) {
        KALDI_LOG << "Warning: weight update for i = " << i
                  << " failed, possible numerical problem.";
      }
    }
    KALDI_LOG << "For iteration " << iter << ", updating w gives predicted "
              << "per-frame like impr " << (k_delta_predicted / k_count) <<
        ", observed " << (k_delta_observed / k_count) << ", over " << (k_count)
              << " frames";
    if (iter == 0) tot_count += k_count;
    tot_delta_predicted += k_delta_predicted;
    tot_delta_observed += k_delta_observed;
  }

  model->w_.CopyFromMat(w);

  tot_delta_observed /= tot_count;
  tot_delta_predicted /= tot_count;
  KALDI_LOG << "**Overall objf impr for w is " << tot_delta_predicted
            << ", observed " << tot_delta_observed << ", over "
            << tot_count << " frames";
  return tot_delta_observed;
}

double MleAmSgmmUpdater::UpdateN(const MleAmSgmmAccs &accs,
                                 AmSgmm *model) {
  double tot_count = 0.0, tot_like_impr = 0.0;
  if (accs.spk_space_dim_ == 0 || accs.R_.size() == 0 || accs.Z_.size() == 0) {
    KALDI_ERR << "Speaker subspace dim is zero or no stats accumulated";
  }

  Vector<double> gamma_i(accs.num_gaussians_);
  for (int32 j = 0; j < accs.num_states_; j++) {
    for (int32 m = 0; m < model->NumSubstates(j); m++) {
      gamma_i.AddVec(1.0, accs.gamma_[j].Row(m));
    }
  }

  SolverOptions opts;
  opts.name = "N";
  opts.K = update_options_.max_cond;
  opts.eps = update_options_.epsilon;
  
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    if (gamma_i(i) < 2 * accs.spk_space_dim_) {
      KALDI_WARN << "Not updating speaker basis for i = " << (i)
                 << " because count is too small " << (gamma_i(i));
      continue;
    }
    Matrix<double> Ni(model->N_[i]);
    double impr =
        SolveQuadraticMatrixProblem(accs.R_[i], accs.Z_[i],
                                    SpMatrix<double>(model->SigmaInv_[i]),
                                    opts, &Ni);
    model->N_[i].CopyFromMat(Ni);
    if (i < 10) {
      KALDI_LOG << "Objf impr for spk projection N for i = " << (i)
                << ", is " << (impr / (gamma_i(i) + 1.0e-20)) << " over "
                << (gamma_i(i)) << " frames";
    }
    tot_count += gamma_i(i);
    tot_like_impr += impr;
  }

  tot_like_impr /= (tot_count+1.0e-20);
  KALDI_LOG << "**Overall objf impr for N is " << tot_like_impr << " over "
            << tot_count << " frames";
  return tot_like_impr;
}

void MleAmSgmmUpdater::RenormalizeN(
    const MleAmSgmmAccs &accs, AmSgmm *model) {
  KALDI_ASSERT(accs.R_.size() != 0);
  Vector<double> gamma_i(accs.num_gaussians_);
  for (int32 j = 0; j < accs.num_states_; j++) {
    for (int32 m = 0; m < model->NumSubstates(j); m++) {
      gamma_i.AddVec(1.0, accs.gamma_[j].Row(m));
    }
  }
  double tot_count = gamma_i.Sum();
  if (tot_count == 0) {
    KALDI_WARN << "Not renormalizing N, since there are no counts.";
    return;
  }

  SpMatrix<double> RTot(accs.spk_space_dim_);
  //  for (int32 i = 0; i < accs.num_gaussians_; i++) {
  //    RTot.AddSp(1.0, accs.R_[i]);
  //  }
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    RTot.AddSp(gamma_i(i), accs.R_[i]);
  }
  RTot.Scale(1.0 / tot_count);
  Matrix<double> U(accs.spk_space_dim_, accs.spk_space_dim_);
  Vector<double> eigs(accs.spk_space_dim_);
  RTot.SymPosSemiDefEig(&eigs, &U);
  KALDI_LOG << "Renormalizing N, eigs are: " << (eigs);
  Vector<double> sqrteigs(accs.spk_space_dim_);
  for (int32 t = 0; t < accs.spk_space_dim_; t++) {
    sqrteigs(t) = sqrt(eigs(t));
  }
  // e.g.   diag(eigs)^{-0.5} * U' * RTot * U * diag(eigs)^{-0.5}  = 1
  // But inverse transpose of this transformation needs to take place on R,
  // i.e. not (on left: diag(eigs)^{-0.5} * U')
  // but: (inverse it: U . diag(eigs)^{0.5},
  // transpose it: diag(eigs)^{0.5} U^T. Need to do this on the right to N
  // (because N has the spk vecs on the right), so N := N U diag(eigs)^{0.5}
  U.MulColsVec(sqrteigs);
  Matrix<double> Ntmp(accs.feature_dim_, accs.spk_space_dim_);
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    Ntmp.AddMatMat(1.0, Matrix<double>(model->N_[i]), kNoTrans, U, kNoTrans, 0.0);
    model->N_[i].CopyFromMat(Ntmp);
  }
}


double MleAmSgmmUpdater::UpdateVars(const MleAmSgmmAccs &accs,
                                    AmSgmm *model) {
  KALDI_ASSERT(S_means_.size() == static_cast<size_t>(accs.num_gaussians_) &&
               "Must call PreComputeStats before updating the covariances.");
  SpMatrix<double> Sigma_i(accs.feature_dim_), Sigma_i_ml(accs.feature_dim_);
  double tot_objf_impr = 0.0, tot_t = 0.0;
  SpMatrix<double> covfloor(accs.feature_dim_);
  Vector<double> gamma_vec(accs.num_gaussians_);
  Vector<double> objf_improv(accs.num_gaussians_);

  // First pass over all (shared) Gaussian components to calculate the
  // ML estimate of the covariances, and the total covariance for flooring.
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    double gamma_i = 0;
    for (int32 j = 0; j < accs.num_states_; j++)
      for (int32 m = 0, end = model->NumSubstates(j); m < end; m++)
        gamma_i += accs.gamma_[j](m, i);

    // Eq. (75): Sigma_{i}^{ml} = 1/gamma_{i} [S_{i} + S_{i}^{(means)} - ...
    //                                          Y_{i} M_{i}^T - M_{i} Y_{i}^T]
    // Note the S_means_ already contains the Y_{i} M_{i}^T terms.
    Sigma_i_ml.CopyFromSp(S_means_[i]);
    Sigma_i_ml.AddSp(1.0, accs.S_[i]);

    gamma_vec(i) = gamma_i;
    covfloor.AddSp(1.0, Sigma_i_ml);
    // inverting  small values e.g. 4.41745328e-40 seems to generate inf,
    // although would be fixed up later.
    if (gamma_i > 1.0e-20) {
      Sigma_i_ml.Scale(1 / (gamma_i + 1.0e-20));
    } else {
      Sigma_i_ml.SetUnit();
    }
    KALDI_ASSERT(1.0 / Sigma_i_ml(0, 0) != 0.0);
    // Eq. (76): Compute the objective function with the old parameter values
    objf_improv(i) = model->SigmaInv_[i].LogPosDefDet() -
        TraceSpSp(SpMatrix<double>(model->SigmaInv_[i]), Sigma_i_ml);

    model->SigmaInv_[i].CopyFromSp(Sigma_i_ml);  // inverted in the next loop.
  }

  // Compute the covariance floor.
  if (gamma_vec.Sum() == 0) {  // If no count, use identity.
    KALDI_WARN << "Updating variances: zero counts. Setting floor to unit.";
    covfloor.SetUnit();
  } else {  // else, use the global average covariance.
    covfloor.Scale(update_options_.cov_floor / gamma_vec.Sum());
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
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    Sigma_i.CopyFromSp(model->SigmaInv_[i]);
    Sigma_i_ml.CopyFromSp(Sigma_i);
    // In case of insufficient counts, make the covariance matrix diagonal.
    // cov_diag_ratio is 2 by default, set to very large to always get diag-cov
    if (gamma_vec(i) < update_options_.cov_diag_ratio * accs.feature_dim_) {
      KALDI_WARN << "For Gaussian component " << i << ": Too low count "
                 << gamma_vec(i) << " for covariance matrix estimation. Setting to "
                 << "diagonal";
      for (int32 d = 0; d < accs.feature_dim_; d++)
        for (int32 e = 0; e < d; e++)
          Sigma_i(d, e) = 0.0;  // SpMatrix, can only set lower traingular part

      int floored = Sigma_i.ApplyFloor(covfloor);
      if (floored > 0) {
        KALDI_WARN << "For Gaussian component " << i << ": Floored " << floored
                   << " covariance eigenvalues.";
      }
      model->SigmaInv_[i].CopyFromSp(Sigma_i);
      model->SigmaInv_[i].InvertDouble();
    } else {  // Updating the full covariance matrix.
      try {
        int floored = Sigma_i.ApplyFloor(covfloor);
        if (floored > 0) {
          KALDI_WARN << "For Gaussian component " << i << ": Floored "
                     << floored << " covariance eigenvalues.";
        }
        model->SigmaInv_[i].CopyFromSp(Sigma_i);
        model->SigmaInv_[i].InvertDouble();

        objf_improv(i) += Sigma_i.LogPosDefDet() +
            TraceSpSp(SpMatrix<double>(model->SigmaInv_[i]), Sigma_i_ml);
        objf_improv(i) *= (-0.5 * gamma_vec(i));  // Eq. (76)

        tot_objf_impr += objf_improv(i);
        tot_t += gamma_vec(i);
        if (i < 5) {
          KALDI_VLOG(2) << "objf impr from variance update =" << objf_improv(i)
              / (gamma_vec(i) + 1.0e-20) << " over " << (gamma_vec(i))
                        << " frames for i = " << (i);
        }
      } catch(...) {
        KALDI_WARN << "Updating within-class covariance matrix i = " << (i)
                   << ", numerical problem";
        // This is a catch-all thing in case of unanticipated errors, but
        // flooring should prevent this occurring for the most part.
        model->SigmaInv_[i].SetUnit();  // Set to unit.
      }
    }
  }
  KALDI_LOG << "**Overall objf impr for variance update = "
            << (tot_objf_impr / (tot_t+ 1.0e-20))
            << " over " << (tot_t) << " frames";
  return tot_objf_impr / (tot_t + 1.0e-20);
}


double MleAmSgmmUpdater::UpdateSubstateWeights(
    const MleAmSgmmAccs &accs, AmSgmm *model) {
  KALDI_LOG << "Updating substate mixture weights";
  // Also set the vector gamma_j which is a cache of the state occupancies.
  gamma_j_.Resize(accs.num_states_);

  double tot_gamma = 0.0, objf_impr = 0.0;
  for (int32 j = 0; j < accs.num_states_; j++) {
    double gamma_j_sm = 0.0;
    int32 num_substates = model->NumSubstates(j);
    Vector<double> occs(num_substates),
        smoothed_occs(num_substates);
    for (int32 m = 0; m < num_substates; m++) {
      occs(m) = accs.gamma_[j].Row(m).Sum();  // \sum_i gamma_{jmi}
      gamma_j_(j) += occs(m);  // actual state occupancy.
      smoothed_occs(m) = occs(m) + update_options_.tau_c;
      gamma_j_sm += smoothed_occs(m);  // smoothed state occupancy for update.
    }

    for (int32 m = 0; m < num_substates; m++) {
      double cur_weight = model->c_[j](m);
      if (cur_weight <= 0) {
        KALDI_WARN << "Zero or negative weight, flooring";
        cur_weight = 1.0e-10;  // future work(arnab): remove magic numbers
      }
      model->c_[j](m) = smoothed_occs(m) / gamma_j_sm;
      objf_impr += log(model->c_[j](m) / cur_weight) * occs(m);
    }
    tot_gamma += gamma_j_(j);
  }
  objf_impr /= (tot_gamma + 1.0e-20);
  KALDI_LOG << "**Overall objf impr for c is " << objf_impr << ", over "
            << tot_gamma << " frames.";
  return objf_impr;
}


MleSgmmSpeakerAccs::MleSgmmSpeakerAccs(const AmSgmm &model, BaseFloat prune)
    : rand_prune_(prune) {
  KALDI_ASSERT(model.SpkSpaceDim() != 0);
  H_spk_.resize(model.NumGauss());
  for (int32 i = 0; i < model.NumGauss(); i++) {
    // Eq. (82): H_{i}^{spk} = N_{i}^T \Sigma_{i}^{-1} N_{i}
    H_spk_[i].Resize(model.SpkSpaceDim());
    H_spk_[i].AddMat2Sp(1.0, Matrix<double>(model.N_[i]),
                        kTrans, SpMatrix<double>(model.SigmaInv_[i]), 0.0);
  }

  model.GetNtransSigmaInv(&NtransSigmaInv_);

  gamma_s_.Resize(model.NumGauss());
  y_s_.Resize(model.SpkSpaceDim());
}

void MleSgmmSpeakerAccs::Clear() {
  y_s_.SetZero();
  gamma_s_.SetZero();
}


BaseFloat
MleSgmmSpeakerAccs::Accumulate(const AmSgmm &model,
                               const SgmmPerFrameDerivedVars &frame_vars,
                               int32 j,
                               BaseFloat weight) {
  // Calculate Gaussian posteriors and collect statistics
  Matrix<BaseFloat> posteriors;
  BaseFloat log_like = model.ComponentPosteriors(frame_vars, j, &posteriors);
  posteriors.Scale(weight);
  AccumulateFromPosteriors(model, frame_vars, posteriors, j);
  return log_like;
}

BaseFloat
MleSgmmSpeakerAccs::AccumulateFromPosteriors(const AmSgmm &model,
                                             const SgmmPerFrameDerivedVars &frame_vars,
                                             const Matrix<BaseFloat> &posteriors,
                                             int32 j) {
  double tot_count = 0.0;
  int32 feature_dim = model.FeatureDim(),
      spk_space_dim = model.SpkSpaceDim();
  KALDI_ASSERT(spk_space_dim != 0);
  const vector<int32> &gselect = frame_vars.gselect;

  // Intermediate variables
  Vector<double> xt_jmi(feature_dim), mu_jmi(feature_dim),
      zt_jmi(spk_space_dim);
  int32 num_substates = model.NumSubstates(j);
  for (int32 ki = 0; ki < static_cast<int32>(gselect.size()); ki++) {
    int32 i = gselect[ki];
    for (int32 m = 0; m < num_substates; m++) {
      // Eq. (39): gamma_{jmi}(t) = p (j, m, i|t)
      BaseFloat gammat_jmi = RandPrune(posteriors(ki, m), rand_prune_);
      if (gammat_jmi != 0.0) {
        tot_count += gammat_jmi;
        model.GetSubstateMean(j, m, i, &mu_jmi);
        xt_jmi.CopyFromVec(frame_vars.xt);
        xt_jmi.AddVec(-1.0, mu_jmi);
        // Eq. (48): z{jmi}(t) = N_{i}^{T} \Sigma_{i}^{-1} x_{jmi}(t)
        zt_jmi.AddMatVec(1.0, NtransSigmaInv_[i], kNoTrans, xt_jmi, 0.0);
        // Eq. (49): \gamma_{i}^{(s)} = \sum_{t\in\Tau(s), j, m} gamma_{jmi}
        gamma_s_(i) += gammat_jmi;
        // Eq. (50): y^{(s)} = \sum_{t, j, m, i} gamma_{jmi}(t) z_{jmi}(t)
        y_s_.AddVec(gammat_jmi, zt_jmi);
      }
    }
  }
  return tot_count;
}

void MleSgmmSpeakerAccs::Update(BaseFloat min_count,
                                Vector<BaseFloat> *v_s,
                                BaseFloat *objf_impr_out,
                                BaseFloat *count_out) {
  double tot_gamma = gamma_s_.Sum();
  KALDI_ASSERT(y_s_.Dim() != 0);
  int32 T = y_s_.Dim();  // speaker-subspace dim.
  int32 num_gauss = gamma_s_.Dim();
  if (v_s->Dim() != T) v_s->Resize(T);  // will set it to zero.

  if (tot_gamma < min_count) {
    KALDI_WARN << "Updating speaker vectors, count is " << tot_gamma
               << " < " << min_count << "not updating.";
    if (objf_impr_out) *objf_impr_out = 0.0;
    if (count_out) *count_out = 0.0;
    return;
  }

  // Eq. (84): H^{(s)} = \sum_{i} \gamma_{i}(s) H_{i}^{spk}
  SpMatrix<double> H_s(T);

  for (int32 i = 0; i < num_gauss; i++)
    H_s.AddSp(gamma_s_(i), H_spk_[i]);


  // Don't make these options to SolveQuadraticProblem configurable...
  // they really don't make a difference at all unless the matrix in
  // question is singular, which wouldn't happen in this case.
  Vector<double> v_s_dbl(*v_s);
  double tot_objf_impr =
      SolveQuadraticProblem(H_s, y_s_, SolverOptions("v_s"), &v_s_dbl);
  v_s->CopyFromVec(v_s_dbl);

  KALDI_LOG << "*Objf impr for speaker vector is " << (tot_objf_impr / tot_gamma)
            << " over " << (tot_gamma) << " frames.";

  if (objf_impr_out) *objf_impr_out = tot_objf_impr;
  if (count_out) *count_out = tot_gamma;
}


MleAmSgmmAccs::~MleAmSgmmAccs() {
  if (gamma_s_.Sum() != 0.0)
    KALDI_ERR << "In destructor of MleAmSgmmAccs: detected that you forgot to "
        "call CommitStatsForSpk()";
}


}  // namespace kaldi

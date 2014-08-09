// sgmm2/estimate-am-sgmm2.cc

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


#include "sgmm2/am-sgmm2.h"
#include "sgmm2/estimate-am-sgmm2.h"
#include "thread/kaldi-thread.h"

namespace kaldi {

using std::string;
using std::vector;

void MleAmSgmm2Accs::Write(std::ostream &out_stream, bool binary) const {

  WriteToken(out_stream, binary, "<SGMMACCS>");
  WriteToken(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs_);
  WriteToken(out_stream, binary, "<NUMGROUPS>");
  WriteBasicType(out_stream, binary, num_groups_);
  WriteToken(out_stream, binary, "<NUMGaussians>");
  WriteBasicType(out_stream, binary, num_gaussians_);
  WriteToken(out_stream, binary, "<FEATUREDIM>");
  WriteBasicType(out_stream, binary, feature_dim_);
  WriteToken(out_stream, binary, "<PHONESPACEDIM>");
  WriteBasicType(out_stream, binary, phn_space_dim_);
  WriteToken(out_stream, binary, "<SPKSPACEDIM>");
  WriteBasicType(out_stream, binary, spk_space_dim_);
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
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      y_[j1].Write(out_stream, binary);
    }
  }
  if (gamma_.size() != 0) { // These stats are large
    // -> write as single precision.
    WriteToken(out_stream, binary, "<gamma>");
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      Matrix<BaseFloat> gamma_j1(gamma_[j1]);
      gamma_j1.Write(out_stream, binary);
    }
  }
  if (t_.NumRows() != 0) {
    WriteToken(out_stream, binary, "<t>");
    t_.Write(out_stream, binary);
  }
  if (U_.size() != 0) {
    WriteToken(out_stream, binary, "<U>");
    for (int32 i = 0; i < num_gaussians_; i++) {
      U_[i].Write(out_stream, binary);
    }
  }
  if (gamma_c_.size() != 0) {
    WriteToken(out_stream, binary, "<gamma_c>");
    for (int32 j2 = 0; j2 < num_pdfs_; j2++) {
      gamma_c_[j2].Write(out_stream, binary);
    }
  }
  if (a_.size() != 0) {
    WriteToken(out_stream, binary, "<a>");
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      a_[j1].Write(out_stream, binary);
    }
  }
  WriteToken(out_stream, binary, "<total_like>");
  WriteBasicType(out_stream, binary, total_like_);

  WriteToken(out_stream, binary, "<total_frames>");
  WriteBasicType(out_stream, binary, total_frames_);

  WriteToken(out_stream, binary, "</SGMMACCS>");
}

void MleAmSgmm2Accs::Read(std::istream &in_stream, bool binary,
                         bool add) {
  ExpectToken(in_stream, binary, "<SGMMACCS>");
  ExpectToken(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs_);
  ExpectToken(in_stream, binary, "<NUMGROUPS>");
  ReadBasicType(in_stream, binary, &num_groups_);
  ExpectToken(in_stream, binary, "<NUMGaussians>");
  ReadBasicType(in_stream, binary, &num_gaussians_);
  ExpectToken(in_stream, binary, "<FEATUREDIM>");
  ReadBasicType(in_stream, binary, &feature_dim_);
  ExpectToken(in_stream, binary, "<PHONESPACEDIM>");
  ReadBasicType(in_stream, binary, &phn_space_dim_);
  ExpectToken(in_stream, binary, "<SPKSPACEDIM>");
  ReadBasicType(in_stream, binary, &spk_space_dim_);

  string token;
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
      y_.resize(num_groups_);
      for (int32 j1 = 0; j1 < num_groups_; j1++) {
        y_[j1].Read(in_stream, binary, add);
      }
    } else if (token == "<gamma>") {
      gamma_.resize(num_groups_);
      for (int32 j1 = 0; j1 < num_groups_; j1++) {
        gamma_[j1].Read(in_stream, binary, add);
      }
      // Don't read gamma_s, it's just a temporary variable and
      // not part of the permanent (non-speaker-specific) accs.
    } else if (token == "<a>") {
      a_.resize(num_groups_);
      for (int32 j1 = 0; j1 < num_groups_; j1++) {
        a_[j1].Read(in_stream, binary, add);
      }
    } else if (token == "<gamma_c>") {
      gamma_c_.resize(num_pdfs_);
      for (int32 j2 = 0; j2 < num_pdfs_; j2++) {
        gamma_c_[j2].Read(in_stream, binary, add);
      }
    } else if (token == "<t>") {
      t_.Read(in_stream, binary, add);
    } else if (token == "<U>") {
      U_.resize(num_gaussians_);
      for (int32 i = 0; i < num_gaussians_; i++) {
        U_[i].Read(in_stream, binary, add);
      }
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

void MleAmSgmm2Accs::Check(const AmSgmm2 &model,
                          bool show_properties) const {
  if (show_properties)
    KALDI_LOG << "Sgmm2PdfModel: J1 = " << num_groups_ << ", J2 = "
              << num_pdfs_ << ", D = " << feature_dim_ << ", S = "
              << phn_space_dim_ << ", T = " << spk_space_dim_ << ", I = "
              << num_gaussians_;
  
  KALDI_ASSERT(num_pdfs_ == model.NumPdfs() && num_pdfs_ > 0);
  KALDI_ASSERT(num_groups_ == model.NumGroups() && num_groups_ > 0);
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
    KALDI_ASSERT(y_.size() == static_cast<size_t>(num_groups_));
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      KALDI_ASSERT(y_[j1].NumRows() == model.NumSubstatesForGroup(j1));
      KALDI_ASSERT(y_[j1].NumCols() == phn_space_dim_);
      if (!nz && y_[j1](0, 0) != 0) { nz = true; }
    }
    debug_str << "y: yes, " << string(nz ? "nonzero. " : "zero. ");
  }

  if (a_.size() == 0) {
    debug_str << "a: no.  ";
  } else {
    debug_str << "a: yes.  ";
    bool nz = false;
    KALDI_ASSERT(a_.size() == static_cast<size_t>(num_groups_));
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      KALDI_ASSERT(a_[j1].NumRows() == model.NumSubstatesForGroup(j1) &&
                   a_[j1].NumCols() == num_gaussians_);
      if (!nz && a_[j1].Sum() != 0) nz = true;
    }
    debug_str << "a: yes, " << string(nz ? "nonzero. " : "zero. "); // TODO: take out "string"
  }

  double tot_gamma = 0.0;
  if (gamma_.size() == 0) {
    debug_str << "gamma: no.  ";
  } else {
    debug_str << "gamma: yes.  ";
    KALDI_ASSERT(gamma_.size() == static_cast<size_t>(num_groups_));
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      KALDI_ASSERT(gamma_[j1].NumRows() == model.NumSubstatesForGroup(j1) &&
                   gamma_[j1].NumCols() == num_gaussians_);
      tot_gamma += gamma_[j1].Sum();
    }
    bool nz = (tot_gamma != 0.0);
    KALDI_ASSERT(gamma_c_.size() == num_pdfs_ && "gamma_ set up but not gamma_c_.");
    debug_str << "gamma: yes, " << string(nz ? "nonzero. " : "zero. ");
  }
  
  if (gamma_c_.size() == 0) {
    KALDI_ERR << "gamma_c_ not set up."; // required for all accs.
  } else {
    KALDI_ASSERT(gamma_c_.size() == num_pdfs_);
    double tot_gamma_c = 0.0;
    for (int32 j2 = 0; j2 < num_pdfs_; j2++) {
      KALDI_ASSERT(gamma_c_[j2].Dim() == model.NumSubstatesForPdf(j2));
      tot_gamma_c += gamma_c_[j2].Sum();
    }
    bool nz = (tot_gamma_c != 0.0);
    debug_str << "gamma_c: yes, " << string(nz ? "nonzero. " : "zero. ");
    if (!gamma_.empty() && !ApproxEqual(tot_gamma_c, tot_gamma))
      KALDI_WARN << "Counts from gamma and gamma_c differ "
                 << tot_gamma << " vs. " << tot_gamma_c;
  }

  if (t_.NumRows() == 0) {
    debug_str << "t: no.  ";
  } else {    
    KALDI_ASSERT(t_.NumRows() == num_gaussians_ &&
                 t_.NumCols() == spk_space_dim_);
    KALDI_ASSERT(!U_.empty()); // t and U are used together.
    bool nz = (t_.FrobeniusNorm() != 0);
    debug_str << "t: yes, " << string(nz ? "nonzero. " : "zero. ");
  }

  if (U_.size() == 0) {
    debug_str << "U: no.  ";
  } else {    
    bool nz = false;
    KALDI_ASSERT(U_.size() == num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++) {
      if (!nz && U_[i].FrobeniusNorm() != 0) nz = true;
      KALDI_ASSERT(U_[i].NumRows() == spk_space_dim_);
    }
    KALDI_ASSERT(t_.NumRows() != 0); // t and U are used together.
    debug_str << "t: yes, " << string(nz ? "nonzero. " : "zero. ");
  }
  
  if (show_properties)
    KALDI_LOG << "Subspace GMM model properties: " << debug_str.str();
}

void MleAmSgmm2Accs::ResizeAccumulators(const AmSgmm2 &model,
                                        SgmmUpdateFlagsType flags,
                                        bool have_spk_vecs) {
  num_pdfs_ = model.NumPdfs();
  num_groups_ = model.NumGroups();
  num_gaussians_ = model.NumGauss();
  feature_dim_ = model.FeatureDim();
  phn_space_dim_ = model.PhoneSpaceDim();
  spk_space_dim_ = model.SpkSpaceDim();
  total_frames_ = total_like_ = 0;
    
  if (flags & (kSgmmPhoneProjections | kSgmmCovarianceMatrix)) {
    Y_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++) {
      Y_[i].Resize(feature_dim_, phn_space_dim_);
    }
  } else {
    Y_.clear();
  }

  if (flags & (kSgmmSpeakerProjections | kSgmmSpeakerWeightProjections)) {
    gamma_s_.Resize(num_gaussians_);
  } else {
    gamma_s_.Resize(0);
  }
  
  if (flags & kSgmmSpeakerProjections) {
    if (spk_space_dim_ == 0) {
      KALDI_ERR << "Cannot set up accumulators for speaker projections "
                << "because speaker subspace has not been set up";
    }
    Z_.resize(num_gaussians_);
    R_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++) {
      Z_[i].Resize(feature_dim_, spk_space_dim_);
      R_[i].Resize(spk_space_dim_);
    }
  } else {
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
               kSgmmCovarianceMatrix | kSgmmPhoneProjections)) {
    gamma_.resize(num_groups_);
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      gamma_[j1].Resize(model.NumSubstatesForGroup(j1), num_gaussians_);
    }
  } else {
    gamma_.clear();
  }

  if (flags & (kSgmmPhoneVectors | kSgmmPhoneWeightProjections)
      && model.HasSpeakerDependentWeights() && have_spk_vecs) { // SSGMM code.
    a_.resize(num_groups_);
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      a_[j1].Resize(model.NumSubstatesForGroup(j1),
                    num_gaussians_);
    }
  } else {
    a_.clear();
  }

  if (flags & kSgmmSpeakerWeightProjections) {
    KALDI_ASSERT(model.HasSpeakerDependentWeights() &&
                 "remove the flag \"u\" if you don't have u set up.");
    a_s_.Resize(num_gaussians_);
    t_.Resize(num_gaussians_, spk_space_dim_);
    U_.resize(num_gaussians_);
    for (int32 i = 0; i < num_gaussians_; i++)
      U_[i].Resize(spk_space_dim_);
  } else {
    a_s_.Resize(0);
    t_.Resize(0, 0);
    U_.resize(0);
  }
  
  if (true) { // always set up gamma_c_; it's nominally for
    // estimation of substate weights, but it's also required when
    // GetStateOccupancies() is called.
    gamma_c_.resize(num_pdfs_);
    for (int32 j2 = 0; j2 < num_pdfs_; j2++) {
      gamma_c_[j2].Resize(model.NumSubstatesForPdf(j2));
    }
  }


  if (flags & kSgmmPhoneVectors) {
    y_.resize(num_groups_);
    for (int32 j1 = 0; j1 < num_groups_; j1++) {
      y_[j1].Resize(model.NumSubstatesForGroup(j1), phn_space_dim_);
    }
  } else {
    y_.clear();
  }
}

BaseFloat MleAmSgmm2Accs::Accumulate(const AmSgmm2 &model,
                                    const Sgmm2PerFrameDerivedVars &frame_vars,
                                    int32 j2,
                                    BaseFloat weight,
                                    Sgmm2PerSpkDerivedVars *spk_vars) {
  // Calculate Gaussian posteriors and collect statistics
  Matrix<BaseFloat> posteriors;
  BaseFloat log_like = model.ComponentPosteriors(frame_vars, j2, spk_vars, &posteriors);
  posteriors.Scale(weight);
  BaseFloat count = AccumulateFromPosteriors(model, frame_vars, posteriors,
                                             j2, spk_vars);
  // Note: total_frames_ is incremented in AccumulateFromPosteriors().
  total_like_ += count * log_like;
  return log_like;
}

BaseFloat MleAmSgmm2Accs::AccumulateFromPosteriors(
    const AmSgmm2 &model,
    const Sgmm2PerFrameDerivedVars &frame_vars,
    const Matrix<BaseFloat> &posteriors,
    int32 j2,
    Sgmm2PerSpkDerivedVars *spk_vars) {
  double tot_count = 0.0;
  const vector<int32> &gselect = frame_vars.gselect;
  // Intermediate variables
  Vector<BaseFloat> gammat(gselect.size()), // sum of gammas over mix-weight.
      a_is_part(gselect.size()); //
  Vector<BaseFloat> xt_jmi(feature_dim_), mu_jmi(feature_dim_),
      zt_jmi(spk_space_dim_);

  int32 j1 = model.Pdf2Group(j2);
  int32 num_substates = model.NumSubstatesForGroup(j1);

  for (int32 m = 0; m < num_substates; m++) {
    BaseFloat d_jms = model.GetDjms(j1, m, spk_vars);
    BaseFloat gammat_jm = 0.0;
    for (int32 ki = 0; ki < static_cast<int32>(gselect.size()); ki++) {
      int32 i = gselect[ki];
      
      // Eq. (39): gamma_{jmi}(t) = p (j, m, i|t)
      BaseFloat gammat_jmi = RandPrune(posteriors(ki, m), rand_prune_);
      if (gammat_jmi == 0.0) continue;
      gammat(ki) += gammat_jmi;
      if (gamma_s_.Dim() != 0)
        gamma_s_(i) += gammat_jmi;
      gammat_jm += gammat_jmi;
      
      // Accumulate statistics for non-zero gaussian posteriors
      tot_count += gammat_jmi;
      if (!gamma_.empty()) {
        // Eq. (40): gamma_{jmi} = \sum_t gamma_{jmi}(t)
        gamma_[j1](m, i) += gammat_jmi;
      }
      if (!y_.empty()) {
        // Eq. (41): y_{jm} = \sum_{t, i} \gamma_{jmi}(t) z_{i}(t)
        // Suggestion:  move this out of the loop over m
        y_[j1].Row(m).AddVec(gammat_jmi, frame_vars.zti.Row(ki));
      }
      if (!Y_.empty()) {
        // Eq. (42): Y_{i} = \sum_{t, j, m} \gamma_{jmi}(t) x_{i}(t) v_{jm}^T
        Y_[i].AddVecVec(gammat_jmi, frame_vars.xti.Row(ki),
                        model.v_[j1].Row(m));
      }
      // Accumulate for speaker projections
      if (!Z_.empty()) {
        KALDI_ASSERT(spk_space_dim_ > 0);
        // Eq. (43): x_{jmi}(t) = x_k(t) - M{i} v_{jm}
        model.GetSubstateMean(j1, m, i, &mu_jmi);
        xt_jmi.CopyFromVec(frame_vars.xt);
        xt_jmi.AddVec(-1.0, mu_jmi);
        // Eq. (44): Z_{i} = \sum_{t, j, m} \gamma_{jmi}(t) x_{jmi}(t) v^{s}'
        if (spk_vars->v_s.Dim() != 0)  // interpret empty v_s as zero.
          Z_[i].AddVecVec(gammat_jmi, xt_jmi, spk_vars->v_s);
        // Eq. (49): \gamma_{i}^{(s)} = \sum_{t\in\Tau(s), j, m} gamma_{jmi}
        // Will be used when you call CommitStatsForSpk(), to update R_.
      }
    } // loop over selected Gaussians
    if (gammat_jm != 0.0) {
      if (!a_.empty()) { // SSGMM code.
        KALDI_ASSERT(d_jms > 0);
        // below is eq. 40 in the MSR techreport.  Caution: there
        // was an error in the original techreport.  The index i
        // in the summation and the quantity \gamma_{jmi}^{(t)}
        // should be differently named, e.g. i'.
        a_[j1].Row(m).AddVec(gammat_jm / d_jms, spk_vars->b_is);
      }
      if (a_s_.Dim() != 0) { // [SSGMM]
        KALDI_ASSERT(d_jms > 0);
        KALDI_ASSERT(!model.w_jmi_.empty());
        a_s_.AddVec(gammat_jm / d_jms, model.w_jmi_[j1].Row(m));
      }
      if (!gamma_c_.empty())
        gamma_c_[j2](m) += gammat_jm;
    }
  } // loop over substates

  if (!S_.empty()) {
    for (int32 ki = 0; ki < static_cast<int32>(gselect.size()); ki++) {
      // Eq. (47): S_{i} = \sum_{t, j, m} \gamma_{jmi}(t) x_{i}(t) x_{i}(t)^T
      if (gammat(ki) != 0.0) {
        int32 i = gselect[ki];
        S_[i].AddVec2(gammat(ki), frame_vars.xti.Row(ki));
      }
    }
  }
  total_frames_ += tot_count;
  return tot_count;
}

void MleAmSgmm2Accs::CommitStatsForSpk(const AmSgmm2 &model,
                                       const Sgmm2PerSpkDerivedVars &spk_vars) {
  const VectorBase<BaseFloat> &v_s = spk_vars.v_s;
  if (v_s.Dim() != 0 && !v_s.IsZero() && !R_.empty()) {
    for (int32 i = 0; i < num_gaussians_; i++)
      // Accumulate Statistics R_{ki}
      if (gamma_s_(i) != 0.0)
        R_[i].AddVec2(gamma_s_(i),
                      Vector<double>(v_s));
  }
  if (a_s_.Dim() != 0) {
    Vector<BaseFloat> tmp(gamma_s_);
    // tmp(i) = gamma_s^{(i)} - a_i^{(s)} b_i^{(s)}.
    tmp.AddVecVec(-1.0, Vector<BaseFloat>(a_s_), spk_vars.b_is, 1.0);
    t_.AddVecVec(1.0, tmp, v_s); // eq. 53 of techreport.
    for (int32 i = 0; i < num_gaussians_; i++) {
      U_[i].AddVec2(a_s_(i) * spk_vars.b_is(i),
                    Vector<double>(v_s)); // eq. 54 of techreport.
    }
  }
  gamma_s_.SetZero();
  a_s_.SetZero();
}

void MleAmSgmm2Accs::GetStateOccupancies(Vector<BaseFloat> *occs) const {
  int32 J2 = gamma_c_.size();
  occs->Resize(J2);
  for (int32 j2 = 0; j2 < J2; j2++) {
    (*occs)(j2) = gamma_c_[j2].Sum();
  }
}

void MleAmSgmm2Updater::Update(const MleAmSgmm2Accs &accs,
                               AmSgmm2 *model,
                               SgmmUpdateFlagsType flags) {
  // Q_{i}, quadratic term for phonetic subspace estimation. Dim is [I][S][S]
  std::vector< SpMatrix<double> > Q;

  // Eq (74): S_{i}^{(means)}, scatter of substate mean vectors for estimating
  // the shared covariance matrices. [Actually this variable contains also the
  // term -(Y_i M_i^T + M_i Y_I^T).]  Dimension is [I][D][D].
  std::vector< SpMatrix<double> > S_means;
  std::vector<Matrix<double> > log_a;
  
  Vector<double> gamma_i(accs.num_gaussians_);
  for (int32 j1 = 0; j1 < accs.num_groups_; j1++)
    gamma_i.AddRowSumMat(1.0, accs.gamma_[j1]); // add sum of rows of
  // accs.gamma_[j1], to gamma_i.
  
  if (flags & kSgmmPhoneProjections)
    ComputeQ(accs, *model, &Q);
  if (flags & kSgmmCovarianceMatrix)
    ComputeSMeans(accs, *model, &S_means);
  if (!accs.a_.empty())
    ComputeLogA(accs, &log_a);

  // quantities used in both vector and weights updates...
  vector< SpMatrix<double> > H;
  // "smoothing" matrices, weighted sums of above.
  SpMatrix<double> H_sm; // weighted sum of H.  Used e.g. in renormalizing phonetic space.
  if ((flags & (kSgmmPhoneVectors | kSgmmPhoneWeightProjections))
      || options_.renormalize_V)
    model->ComputeH(&H);
  
  BaseFloat tot_impr = 0.0;

  if (flags & kSgmmPhoneVectors)
    tot_impr += UpdatePhoneVectors(accs, H, log_a, model);
  if (flags & kSgmmPhoneProjections) {
    if (options_.tau_map_M > 0.0)
      tot_impr += MapUpdateM(accs, Q, gamma_i, model);  // MAP adaptation of M
    else
      tot_impr += UpdateM(accs, Q, gamma_i, model);
  }
  if (flags & kSgmmPhoneWeightProjections)
    tot_impr += UpdateW(accs, log_a, gamma_i, model);
  if (flags & kSgmmCovarianceMatrix)
    tot_impr += UpdateVars(accs, S_means, gamma_i, model);
  if (flags & kSgmmSubstateWeights)
    tot_impr += UpdateSubstateWeights(accs, model);
  if (flags & kSgmmSpeakerProjections)
    tot_impr += UpdateN(accs, gamma_i, model);
  if (flags & kSgmmSpeakerWeightProjections)
    tot_impr += UpdateU(accs, gamma_i, model);
  
  if ((flags & kSgmmSpeakerProjections) && (options_.renormalize_N)) 
    RenormalizeN(accs, gamma_i, model); // if you renormalize N you have to
  // alter any speaker vectors you're keeping around, as well.
  // So be careful with this option.
  
  if (options_.renormalize_V)
    RenormalizeV(accs, model, gamma_i, H);

  KALDI_LOG << "*Overall auxf improvement, combining all parameters, is "
            << tot_impr;

  KALDI_LOG << "***Overall data likelihood is "
            << (accs.total_like_/accs.total_frames_)
            << " over " << accs.total_frames_ << " frames.";

  model->n_.clear(); // has become invalid.
  model->w_jmi_.clear(); // has become invalid.
  // we updated the v or w quantities.
}

// Compute the Q_{i} (Eq. 64)
void MleAmSgmm2Updater::ComputeQ(const MleAmSgmm2Accs &accs,
                                const AmSgmm2 &model,
                                std::vector< SpMatrix<double> > *Q) {
  Q->resize(accs.num_gaussians_);
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    (*Q)[i].Resize(accs.phn_space_dim_);
    for (int32 j1 = 0; j1 < accs.num_groups_; j1++) {
      for (int32 m = 0; m < model.NumSubstatesForGroup(j1); m++) {
        if (accs.gamma_[j1](m, i) > 0.0) {
          (*Q)[i].AddVec2(static_cast<BaseFloat>(accs.gamma_[j1](m, i)),
                          model.v_[j1].Row(m));
        }
      }
    }
  }
}

// Compute the S_i^{(means)} quantities (Eq. 74).
// Note: we seem to have also included in this variable
// the term - (Y_i M_I^T + M_i Y_i^T).
void MleAmSgmm2Updater::ComputeSMeans(const MleAmSgmm2Accs &accs,
                                     const AmSgmm2 &model,
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

    for (int32 j1 = 0; j1 < accs.num_groups_; j1++) {
      for (int32 m = 0; m < model.NumSubstatesForGroup(j1); m++) {
        if (accs.gamma_[j1](m, i) != 0.0) {
          // Sigma_{i} += gamma_{jmi} * mu_{jmi}*mu_{jmi}^T
          mu_jmi.AddMatVec(1.0, model.M_[i], kNoTrans, model.v_[j1].Row(m), 0.0);
          (*S_means)[i].AddVec2(static_cast<BaseFloat>(accs.gamma_[j1](m, i)), mu_jmi);
        }
      }
    }
    KALDI_ASSERT(1.0 / (*S_means)[i](0, 0) != 0.0);
  }
}


class UpdatePhoneVectorsClass: public MultiThreadable { // For multi-threaded.
 public:
  UpdatePhoneVectorsClass(const MleAmSgmm2Updater &updater,
                          const MleAmSgmm2Accs &accs,
                          const std::vector<SpMatrix<double> > &H,
                          const std::vector<Matrix<double> > &log_a,
                          AmSgmm2 *model,
                          double *auxf_impr):
      updater_(updater), accs_(accs), model_(model), 
      H_(H), log_a_(log_a), auxf_impr_ptr_(auxf_impr),
      auxf_impr_(0.0) { }
 
  ~UpdatePhoneVectorsClass() {
    *auxf_impr_ptr_ += auxf_impr_;
  }
  
  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to the total sums in the destructor.
    updater_.UpdatePhoneVectorsInternal(accs_, H_, log_a_, model_,
                                        &auxf_impr_, num_threads_, thread_id_);
  }
 private:
  const MleAmSgmm2Updater &updater_;
  const MleAmSgmm2Accs &accs_;
  AmSgmm2 *model_;
  const std::vector<SpMatrix<double> > &H_;
  const std::vector<Matrix<double> > &log_a_;
  double *auxf_impr_ptr_;
  double auxf_impr_;
};

/**
   In this update, smoothing terms are not supported.  However, it does compute
   the auxiliary function after doing the update, and backtracks if it did not
   increase (due to the weight terms, increase is not mathematically
   guaranteed). */

double MleAmSgmm2Updater::UpdatePhoneVectors(
    const MleAmSgmm2Accs &accs,    
    const vector< SpMatrix<double> > &H,
    const vector< Matrix<double> > &log_a,
    AmSgmm2 *model) const {

  KALDI_LOG << "Updating phone vectors";

  double count = 0.0, auxf_impr = 0.0;  // sum over all states
  
  for (int32 j1 = 0; j1 < accs.num_groups_; j1++)
    count += accs.gamma_[j1].Sum();

  UpdatePhoneVectorsClass c(*this, accs, H, log_a, model, &auxf_impr);
  RunMultiThreaded(c);

  double auxf_per_frame = auxf_impr / (count + 1.0e-20);
  
  KALDI_LOG << "**Overall auxf impr for v is " << auxf_per_frame << " over "
            << count << " frames";
  return auxf_per_frame;
}

//static
void MleAmSgmm2Updater::ComputeLogA(const MleAmSgmm2Accs &accs,
                                    std::vector<Matrix<double> > *log_a) {
  // This computes the logarithm of the statistics a_{jmi} defined
  // in Eq. 40 of the SSGMM techreport.  Although the log of a_{jmi} never
  // explicitly appears in the techreport, it happens to be more convenient
  // in the code to use the log of it.
  // Note: because of the way a is computed, for each (j,m) the
  // entries over i should always be all zero or all nonzero.
  int32 num_zeros = 0;
  KALDI_ASSERT(accs.a_.size() == accs.num_groups_);
  log_a->resize(accs.num_groups_);
  for (int32 j1 = 0; j1 < accs.num_groups_; j1++) {
    int32 num_substates = accs.a_[j1].NumRows();
    KALDI_ASSERT(num_substates > 0);
    (*log_a)[j1].Resize(num_substates, accs.num_gaussians_);
    for (int32 m = 0; m < num_substates; m++) {
      if (accs.a_[j1](m, 0) == 0.0) { // Zero accs. 
        num_zeros++;
        if (accs.gamma_[j1].Row(m).Sum() != 0.0)
          KALDI_WARN << "Inconsistency between a and gamma stats. [BAD!]";
        // leave the row zero.  This means the sub-state saw no stats.
      } else {
        (*log_a)[j1].Row(m).CopyFromVec(accs.a_[j1].Row(m));
        (*log_a)[j1].Row(m).ApplyLog();
      }
    }
  }
  if (num_zeros != 0)
    KALDI_WARN << num_zeros
               << " sub-states with zero \"a\" (and presumably gamma) stats.";
}

void MleAmSgmm2Updater::UpdatePhoneVectorsInternal(
    const MleAmSgmm2Accs &accs,
    const vector< SpMatrix<double> > &H,
    const vector< Matrix<double> > &log_a,
    AmSgmm2 *model,
    double *auxf_impr_ptr,
    int32 num_threads,
    int32 thread_id) const {
  
  int32 J1 = accs.num_groups_, block_size = (J1 + (num_threads-1)) / num_threads,
      j1_start = block_size * thread_id,
      j1_end = std::min(accs.num_groups_, j1_start + block_size);

  double tot_auxf_impr = 0.0;
  
  for (int32 j1 = j1_start; j1 < j1_end; j1++) {
    for (int32 m = 0; m < model->NumSubstatesForGroup(j1); m++) {
      double gamma_jm = accs.gamma_[j1].Row(m).Sum();
      SpMatrix<double> X_jm(accs.phn_space_dim_);  // = \sum_i \gamma_{jmi} H_i
      
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        double gamma_jmi = accs.gamma_[j1](m, i);
        if (gamma_jmi != 0.0)
          X_jm.AddSp(gamma_jmi, H[i]);
      }

      Vector<double> v_jm_orig(model->v_[j1].Row(m)),
          v_jm(v_jm_orig);

      double exact_auxf_start = 0.0, exact_auxf = 0.0, approx_auxf_impr = 0.0;
      int32 backtrack_iter, max_backtrack = 10;
      for (backtrack_iter = 0; backtrack_iter < max_backtrack; backtrack_iter++) {
        // Note: the 1st time we go through this loop we have not yet updated
        // v_jm and it has the old value; the 2nd time, it has the updated value
        // and we will typically break at this point, after verifying that
        // the auxf has improved.
        
        // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
        Vector<double> w_jm(accs.num_gaussians_);
        w_jm.AddMatVec(1.0, Matrix<double>(model->w_), kNoTrans,
                       v_jm, 0.0);
        if (!log_a.empty()) w_jm.AddVec(1.0, log_a[j1].Row(m)); // SSGMM techreport eq. 42
        w_jm.Add(-w_jm.LogSumExp());  // it is now log w_jm
        

        exact_auxf = VecVec(w_jm, accs.gamma_[j1].Row(m))
            + VecVec(v_jm, accs.y_[j1].Row(m))
            -0.5 * VecSpVec(v_jm, X_jm, v_jm);

        if (backtrack_iter == 0) {
          exact_auxf_start = exact_auxf;
        } else {
          if (exact_auxf >= exact_auxf_start) {
            break;  // terminate backtracking.
          } else {
            KALDI_LOG << "Backtracking computation of v_jm for j = " << j1
                      << " and m = " << m << " because auxf changed by "
                      << (exact_auxf-exact_auxf_start) << " [vs. predicted:] "
                      << approx_auxf_impr;
            v_jm.AddVec(1.0, v_jm_orig);
            v_jm.Scale(0.5);
          }
        }

        if (backtrack_iter == 0) {  // computing updated value.
          w_jm.ApplyExp();  // it is now w_jm
          SpMatrix<double> H_jm(X_jm);
          Vector<double> g_jm(accs.y_[j1].Row(m));
          for (int32 i = 0; i < accs.num_gaussians_; i++) {
            double gamma_jmi = accs.gamma_[j1](m, i);
            double quadratic_term = std::max(gamma_jmi, gamma_jm * w_jm(i));
            double scalar = gamma_jmi - gamma_jm * w_jm(i) + quadratic_term
                * VecVec(model->w_.Row(i), model->v_[j1].Row(m));
            g_jm.AddVec(scalar, model->w_.Row(i));
            if (quadratic_term > 1.0e-10) {
              H_jm.AddVec2(static_cast<BaseFloat>(quadratic_term), model->w_.Row(i));
            }
          }

          SolverOptions opts;
          opts.name = "v";
          opts.K = options_.max_cond;
          opts.eps = options_.epsilon;

          approx_auxf_impr = SolveQuadraticProblem(H_jm, g_jm, opts, &v_jm);
        }
      }
      double exact_auxf_impr = exact_auxf - exact_auxf_start;
      tot_auxf_impr += exact_auxf_impr; 
      if (backtrack_iter == max_backtrack) {
        KALDI_WARN << "Backtracked " << max_backtrack << " times [not updating]";
      } else {
        model->v_[j1].Row(m).CopyFromVec(v_jm);
      }

      if (j1 < 3 && m < 3) {
        KALDI_LOG << "Auxf impr for j = " << j1 << " m = " << m << " is "
                  << (exact_auxf_impr/gamma_jm+1.0e-20) << " per frame over "
                  << gamma_jm << " frames.";
      }
    }
  }
  *auxf_impr_ptr = tot_auxf_impr;
}


void MleAmSgmm2Updater::RenormalizeV(const MleAmSgmm2Accs &accs,
                                    AmSgmm2 *model,
                                    const Vector<double> &gamma_i,
                                    const vector<SpMatrix<double> > &H) {
  // Compute H^{(sm)}, the "smoothing" matrix-- average of H's.
  SpMatrix<double> H_sm(accs.phn_space_dim_);
  for (int32 i = 0; i < accs.num_gaussians_; i++)
    H_sm.AddSp(gamma_i(i), H[i]);
  KALDI_ASSERT(gamma_i.Sum() > 0.0);
  H_sm.Scale(1.0 / gamma_i.Sum());
  
  SpMatrix<double> Sigma(accs.phn_space_dim_);
  int32 count = 0;
  for (int32 j1 = 0; j1 < accs.num_groups_; j1++) {
    for (int32 m = 0; m < model->NumSubstatesForGroup(j1); m++) {
      count++;
      Sigma.AddVec2(static_cast<BaseFloat>(1.0), model->v_[j1].Row(m));
    }
  }
  if (!Sigma.IsPosDef()) {
    KALDI_LOG << "Not renormalizing v because scatter is not positive definite"
              << " -- maybe first iter?";
    return;
  }
  Sigma.Scale(1.0 / count);
  KALDI_LOG << "Scatter of vectors v is : ";
  Sigma.PrintEigs("Sigma");

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

  for (int32 j1 = 0; j1 < accs.num_groups_; j1++) {
    for (int32 m = 0; m < model->NumSubstatesForGroup(j1); m++) {
      Vector<double> tmp(accs.phn_space_dim_);
      tmp.AddMatVec(1.0, Trans, kNoTrans, Vector<double>(model->v_[j1].Row(m)), 0.0);
      model->v_[j1].Row(m).CopyFromVec(tmp);
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

double MleAmSgmm2Updater::UpdateM(const MleAmSgmm2Accs &accs,
                                 const std::vector< SpMatrix<double> > &Q,
                                 const Vector<double> &gamma_i,
                                 AmSgmm2 *model) {
  double tot_count = 0.0, tot_like_impr = 0.0;
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    if (gamma_i(i) < accs.feature_dim_) {
      KALDI_WARN << "For component " << i << ": not updating M due to very "
                 << "small count (=" << gamma_i(i) << ").";
      continue;
    }

    SolverOptions opts;
    opts.name = "M";
    opts.K = options_.max_cond;
    opts.eps = options_.epsilon;
    
    Matrix<double> Mi(model->M_[i]);
    double impr =
        SolveQuadraticMatrixProblem(Q[i], accs.Y_[i],
                                    SpMatrix<double>(model->SigmaInv_[i]),
                                    opts, &Mi);

    model->M_[i].CopyFromMat(Mi);

    if (i < 10) {
      KALDI_VLOG(2) << "Objf impr for projection M for i = " << i << ", is "
                    << (impr/(gamma_i(i) + 1.0e-20)) << " over " << gamma_i(i)
                    << " frames";
    }
    tot_count += gamma_i(i);
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
void MleAmSgmm2Updater::ComputeMPrior(AmSgmm2 *model) {
  KALDI_ASSERT(options_.map_M_prior_iters > 0);
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

  if (options_.full_col_cov || options_.full_row_cov) {
    Matrix<double> avg_M(Ddim, Sdim);  // average of the Gaussian prior means
    for (int32 i = 0; i < nGaussians; i++)
      avg_M.AddMat(1.0, Matrix<double>(model->M_prior_[i]));
    avg_M.Scale(1.0 / nGaussians);

    Matrix<double> MDiff(Ddim, Sdim);
    for (int32 iter = 0; iter < options_.map_M_prior_iters; iter++) {
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
      if (options_.full_col_cov) {
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
        limited = model->col_cov_inv_.LimitCond(options_.max_cond,
                                                true /*invert the matrix*/);
        if (limited != 0) {
          KALDI_LOG << "Computing column covariances for M: limited " << limited
                    << " singular values, max condition is "
                    << options_.max_cond;
        }
      }

      // Now estimate the row covariances (\Omega_c in paper)
      if (options_.full_row_cov) {
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
        limited = model->row_cov_inv_.LimitCond(options_.max_cond,
                                                true /*invert the matrix*/);
        if (limited != 0) {
          KALDI_LOG << "Computing row covariances for M: limited " << limited
                    << " singular values, max condition is "
                    << options_.max_cond;
        }
      }
    }  // end iterations
  }
}


// MAP adaptation of M with a matrix-variate Gaussian prior
double MleAmSgmm2Updater::MapUpdateM(const MleAmSgmm2Accs &accs,
                                     const std::vector< SpMatrix<double> > &Q,
                                     const Vector<double> &gamma_i,
                                     AmSgmm2 *model) {
  int32 Ddim = model->FeatureDim();
  int32 Sdim = model->PhoneSpaceDim();
  int32 nGaussians = model->NumGauss();

  KALDI_LOG << "Prior smoothing parameter: Tau = " << options_.tau_map_M;
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
  Q2.Scale(options_.tau_map_M);

  double totcount = 0.0, tot_like_impr = 0.0;
  for (int32 i = 0; i < nGaussians; ++i) {
    if (gamma_i(i) < accs.feature_dim_) {
      KALDI_WARN << "For component " << i << ": not updating M due to very "
                 << "small count (=" << gamma_i(i) << ").";
      continue;
    }

    Matrix<double> tmp(Ddim, Sdim, kSetZero);
    tmp.AddSpMat(1.0, SpMatrix<double>(model->col_cov_inv_),
                 Matrix<double>(model->M_prior_[i]), kNoTrans, 0.0);
    prior_term_i.AddMatSp(options_.tau_map_M, tmp, kNoTrans,
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
    opts.K = options_.max_cond;
    opts.eps = options_.epsilon;
    double impr =
        SolveDoubleQuadraticMatrixProblem(G, P1, P2, Q[i], Q2, opts, &Mi);
    model->M_[i].CopyFromMat(Mi);
    if (i < 10) {
      KALDI_LOG << "Objf impr for projection M for i = " << i << ", is "
                << (impr / (gamma_i(i) + 1.0e-20)) << " over " << gamma_i(i)
                << " frames";
    }
    totcount += gamma_i(i);
    tot_like_impr += impr;
  }
  tot_like_impr /= (totcount + 1.0e-20);
  KALDI_LOG << "Overall objective function improvement for model projections "
            << "M is " << tot_like_impr << " over " << totcount << " frames";
  return tot_like_impr;
}


/// This function gets stats used inside UpdateW, where it accumulates
/// the F_i and g_i quantities.  Note: F_i is viewed as a vector of SpMatrix
/// (one for each i); each row of F_i is viewed as an SpMatrix even though
/// it's stored as a vector....
/// Note: on the first iteration w is just a double-precision copy of the matrix
/// model->w_; thereafter it may differ.
/// log_a relates to the SSGMM.

// static
void MleAmSgmm2Updater::UpdateWGetStats(const MleAmSgmm2Accs &accs,
                                        const AmSgmm2 &model,
                                        const Matrix<double> &w,
                                        const std::vector<Matrix<double> > &log_a,
                                        Matrix<double> *F_i,
                                        Matrix<double> *g_i,
                                        double *tot_like,
                                        int32 num_threads, 
                                        int32 thread_id) {

  // Accumulate stats from a block of states (this gets called in parallel).
  int32 block_size = (accs.num_groups_ + (num_threads-1)) / num_threads,
      j1_start = block_size * thread_id,
      j1_end = std::min(accs.num_groups_, j1_start + block_size);
  
  // Unlike in the report the inner most loop is over Gaussians, where
  // per-gaussian statistics are accumulated. This is more memory demanding
  // but more computationally efficient, as outer product v_{jvm} v_{jvm}^T
  // is computed only once for all gaussians.

  SpMatrix<double> v_vT(accs.phn_space_dim_);
  
  for (int32 j1 = j1_start; j1 < j1_end; j1++) {
    int32 num_substates = model.NumSubstatesForGroup(j1);
    Matrix<double> w_j(num_substates, accs.num_gaussians_);
    // The linear term and quadratic term for each Gaussian-- two scalars
    // for each Gaussian, they appear in the accumulation formulas.
    Matrix<double> linear_term(num_substates, accs.num_gaussians_);
    Matrix<double> quadratic_term(num_substates, accs.num_gaussians_);
    Matrix<double> v_vT_m(num_substates,
                          (accs.phn_space_dim_*(accs.phn_space_dim_+1))/2);

    // w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm})  eq.(7)
    Matrix<double> v_j_double(model.v_[j1]);
    w_j.AddMatMat(1.0, v_j_double, kNoTrans, w, kTrans, 0.0);
    if (!log_a.empty()) w_j.AddMat(1.0, log_a[j1]); // SSGMM techreport eq. 42
    
    for (int32 m = 0; m < model.NumSubstatesForGroup(j1); m++) {
      SubVector<double> w_jm(w_j, m);
      double gamma_jm = accs.gamma_[j1].Row(m).Sum();
      w_jm.Add(-1.0 * w_jm.LogSumExp());
      *tot_like += VecVec(w_jm, accs.gamma_[j1].Row(m));
      w_jm.ApplyExp();
      v_vT.SetZero();
      // v_vT := v_{jkm} v_{jkm}^T
      v_vT.AddVec2(static_cast<BaseFloat>(1.0), v_j_double.Row(m));
      v_vT_m.Row(m).CopyFromPacked(v_vT); // a bit wasteful, but does not dominate.
      
      for (int32 i = 0; i < accs.num_gaussians_; i++) {
        // Suggestion: g_jkm can be computed more efficiently
        // using the Vector/Matrix routines for all i at once
        // linear term around cur value.
        linear_term(m, i) = accs.gamma_[j1](m, i) - gamma_jm * w_jm(i);
        quadratic_term(m, i) = std::max(accs.gamma_[j1](m, i),
                                        gamma_jm * w_jm(i));
      }
    } // loop over substates
    g_i->AddMatMat(1.0, linear_term, kTrans, v_j_double, kNoTrans, 1.0);
    F_i->AddMatMat(1.0, quadratic_term, kTrans, v_vT_m, kNoTrans, 1.0);
  } // loop over states
}

// The parallel weight update, in the paper.
double MleAmSgmm2Updater::UpdateW(const MleAmSgmm2Accs &accs,
                                  const std::vector<Matrix<double> > &log_a,
                                  const Vector<double> &gamma_i,
                                  AmSgmm2 *model) {
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
  double tot_count = gamma_i.Sum();
  
  for (int iter = 0; iter < options_.weight_projections_iters; iter++) {
    F_i.SetZero();
    g_i.SetZero();
    double k_like_before = 0.0;
    
    UpdateWClass c(accs, *model, w, log_a, &F_i, &g_i, &k_like_before);
    RunMultiThreaded(c);
    
    Matrix<double> w_orig(w);
    double k_predicted_like_impr = 0.0, k_like_after = 0.0;
    double min_step = 0.001, step_size;

    SolverOptions opts;
    opts.name = "w";
    opts.K = options_.max_cond;
    opts.eps = options_.epsilon;
    
    for (step_size = 1.0; step_size >= min_step; step_size /= 2) {
      k_predicted_like_impr = 0.0;
      k_like_after = 0.0;
      
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

        if (i < 10)
          KALDI_LOG << "Predicted objf impr for w, iter = " << iter
                    << ", i = " << i << " is "
                    << (predicted_impr/gamma_i(i)+1.0e-20)
                    << " per frame over " << gamma_i(i) << " frames.";
        k_predicted_like_impr += predicted_impr;
        w.Row(i).AddVec(1.0, delta_w);
      }
      for (int32 j1 = 0; j1 < accs.num_groups_; j1++) {
        int32 M = model->NumSubstatesForGroup(j1);
        Matrix<double> w_j(M, accs.num_gaussians_);
        w_j.AddMatMat(1.0, Matrix<double>(model->v_[j1]), kNoTrans,
                       w, kTrans, 0.0);
        if (!log_a.empty()) w_j.AddMat(1.0, log_a[j1]); // SSGMM techreport eq. 42
        for (int32 m = 0; m < M; m++) {
          SubVector<double> w_jm(w_j, m);
          w_jm.Add(-1.0 * w_jm.LogSumExp());
        }
        k_like_after += TraceMatMat(w_j, accs.gamma_[j1], kTrans);
      }
      KALDI_VLOG(2) << "For iteration " << iter << ", updating w gives "
                    << "predicted per-frame like impr "
                    << (k_predicted_like_impr / tot_count) << ", actual "
                    << ((k_like_after - k_like_before) / tot_count) << ", over "
                    << tot_count << " frames";
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
  model->w_jmi_.clear(); // invalidated.
      
  tot_predicted_like_impr /= tot_count;
  tot_like_after = (tot_like_after - tot_like_before) / tot_count;
  KALDI_LOG << "**Overall objf impr for w is " << tot_predicted_like_impr
            << ", actual " << tot_like_after << ", over "
            << tot_count << " frames";
  return tot_like_after;
}

double MleAmSgmm2Updater::UpdateU(const MleAmSgmm2Accs &accs,
                                 const Vector<double> &gamma_i,
                                 AmSgmm2 *model) {
  double tot_impr = 0.0;
  SolverOptions opts;
  opts.name = "u";
  opts.K = options_.max_cond;
  opts.eps = options_.epsilon;
  
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    if (gamma_i(i) < 200.0) {
      KALDI_LOG << "Count is small " << gamma_i(i) << " for gaussian "
                << i << ", not updating u_i.";
      continue;
    }
    Vector<double> u_i(model->u_.Row(i));
    Vector<double> delta_u(accs.spk_space_dim_);
    double impr =
        SolveQuadraticProblem(accs.U_[i], accs.t_.Row(i), opts, &delta_u);
    double impr_per_frame = impr / gamma_i(i);
    if (impr_per_frame > options_.max_impr_u) {
      KALDI_WARN << "Updating speaker weight projections u, for Gaussian index "
                 << i << ", impr/frame is " << impr_per_frame << " over "
                 << gamma_i(i) << " frames, scaling back to not exceed "
                 << options_.max_impr_u;
      double scale = options_.max_impr_u / impr_per_frame;
      impr *= scale;
      delta_u.Scale(scale);
      // Note: a linear scaling of "impr" with "scale" is not quite accurate
      // in depicting how the quadratic auxiliary function varies as we change
      // the scale on "delta", but this does not really matter-- the goal is
      // to limit the auxiliary-function change to not be too large.
    }
    if (i < 10) {
      KALDI_LOG << "Objf impr for spk weight-projection u for i = " << (i)
                << ", is " << (impr / (gamma_i(i) + 1.0e-20)) << " over "
                << gamma_i(i) << " frames";
    }
    u_i.AddVec(1.0, delta_u);
    model->u_.Row(i).CopyFromVec(u_i);
    tot_impr += impr;
  }
  KALDI_LOG << "**Overall objf impr for u is " << (tot_impr/gamma_i.Sum())
            << ", over " << gamma_i.Sum() << " frames";
  return tot_impr / gamma_i.Sum();
}

double MleAmSgmm2Updater::UpdateN(const MleAmSgmm2Accs &accs,
                                 const Vector<double>  &gamma_i,
                                 AmSgmm2 *model) {
  double tot_count = 0.0, tot_like_impr = 0.0;
  if (accs.spk_space_dim_ == 0 || accs.R_.size() == 0 || accs.Z_.size() == 0) {
    KALDI_ERR << "Speaker subspace dim is zero or no stats accumulated";
  }
  SolverOptions opts;
  opts.name = "N";
  opts.K = options_.max_cond;
  opts.eps = options_.epsilon;

  
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
                << gamma_i(i) << " frames";
    }
    tot_count += gamma_i(i);
    tot_like_impr += impr;
  }

  KALDI_LOG << "**Overall objf impr for N is " << (tot_like_impr/tot_count)
            << " over " << tot_count << " frames";
  return (tot_like_impr/tot_count);
}

void MleAmSgmm2Updater::RenormalizeN(const MleAmSgmm2Accs &accs,
                                    const Vector<double> &gamma_i,
                                    AmSgmm2 *model) {
  KALDI_ASSERT(accs.R_.size() != 0);
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


double MleAmSgmm2Updater::UpdateVars(const MleAmSgmm2Accs &accs,
                                    const std::vector< SpMatrix<double> > &S_means,
                                    const Vector<double> &gamma_i,
                                    AmSgmm2 *model) {
  KALDI_ASSERT(S_means.size() == static_cast<size_t>(accs.num_gaussians_));

  SpMatrix<double> Sigma_i(accs.feature_dim_), Sigma_i_ml(accs.feature_dim_);
  double tot_objf_impr = 0.0, tot_t = 0.0;
  SpMatrix<double> covfloor(accs.feature_dim_);
  Vector<double> objf_improv(accs.num_gaussians_);

  // First pass over all (shared) Gaussian components to calculate the
  // ML estimate of the covariances, and the total covariance for flooring.
  for (int32 i = 0; i < accs.num_gaussians_; i++) {
    // Eq. (75): Sigma_{i}^{ml} = 1/gamma_{i} [S_{i} + S_{i}^{(means)} - ...
    //                                          Y_{i} M_{i}^T - M_{i} Y_{i}^T]
    // Note the S_means already contains the Y_{i} M_{i}^T terms.
    Sigma_i_ml.CopyFromSp(S_means[i]);
    Sigma_i_ml.AddSp(1.0, accs.S_[i]);
    
    covfloor.AddSp(1.0, Sigma_i_ml);
    // inverting  small values e.g. 4.41745328e-40 seems to generate inf,
    // although would be fixed up later.
    if (gamma_i(i) > 1.0e-20) {
      Sigma_i_ml.Scale(1 / (gamma_i(i) + 1.0e-20));
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
  if (gamma_i.Sum() == 0) {  // If no count, use identity.
    KALDI_WARN << "Updating variances: zero counts. Setting floor to unit.";
    covfloor.SetUnit();
  } else {  // else, use the global average covariance.
    covfloor.Scale(options_.cov_floor / gamma_i.Sum());
    int32 tmp;
    if ((tmp = covfloor.LimitCondDouble(options_.max_cond)) != 0) {
      KALDI_WARN << "Covariance flooring matrix is poorly conditioned. Fixed "
                 << "up " << tmp << " eigenvalues.";
    }
  }

  if (options_.cov_diag_ratio > 1000) {
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
    if (gamma_i(i) < options_.cov_diag_ratio * accs.feature_dim_) {
      KALDI_WARN << "For Gaussian component " << i << ": Too low count "
                 << gamma_i(i) << " for covariance matrix estimation. Setting to "
                 << "diagonal";
      for (int32 d = 0; d < accs.feature_dim_; d++)
        for (int32 e = 0; e < d; e++)
          Sigma_i(d, e) = 0.0;  // SpMatrix, can only set lower triangular part

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
        objf_improv(i) *= (-0.5 * gamma_i(i));  // Eq. (76)

        tot_objf_impr += objf_improv(i);
        tot_t += gamma_i(i);
        if (i < 5) {
          KALDI_VLOG(2) << "objf impr from variance update =" << objf_improv(i)
              / (gamma_i(i) + 1.0e-20) << " over " << (gamma_i(i))
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
            << " over " << tot_t << " frames";
  return tot_objf_impr / (tot_t + 1.0e-20);
}


double MleAmSgmm2Updater::UpdateSubstateWeights(
    const MleAmSgmm2Accs &accs, AmSgmm2 *model) {
  KALDI_LOG << "Updating substate mixture weights";
  // Also set the vector gamma_j which is a cache of the state occupancies.

  double tot_gamma = 0.0, objf_impr = 0.0;
  for (int32 j2 = 0; j2 < accs.num_pdfs_; j2++) {
    double gamma_j_sm = 0.0;
    int32 num_substates = model->NumSubstatesForPdf(j2);
    const Vector<double> &occs(accs.gamma_c_[j2]);
    Vector<double> smoothed_occs(occs);
    smoothed_occs.Add(options_.tau_c);
    gamma_j_sm += smoothed_occs.Sum();
    tot_gamma += occs.Sum();
    
    for (int32 m = 0; m < num_substates; m++) {
      double cur_weight = model->c_[j2](m);
      if (cur_weight <= 0) {
        KALDI_WARN << "Zero or negative weight, flooring";
        cur_weight = 1.0e-10;  // future work(arnab): remove magic numbers
      }
      model->c_[j2](m) = smoothed_occs(m) / gamma_j_sm;
      objf_impr += log(model->c_[j2](m) / cur_weight) * occs(m);
    }
  }
  KALDI_LOG << "**Overall objf impr for c is " << (objf_impr/tot_gamma)
            << ", over " << tot_gamma << " frames.";
  return (objf_impr/tot_gamma);
}


MleSgmm2SpeakerAccs::MleSgmm2SpeakerAccs(const AmSgmm2 &model,
                                         BaseFloat prune)
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
  if (model.HasSpeakerDependentWeights())
    a_s_.Resize(model.NumGauss());
}

void MleSgmm2SpeakerAccs::Clear() {
  y_s_.SetZero();
  gamma_s_.SetZero();
  if (a_s_.Dim() != 0) a_s_.SetZero();
}

BaseFloat
MleSgmm2SpeakerAccs::Accumulate(const AmSgmm2 &model,
                               const Sgmm2PerFrameDerivedVars &frame_vars,
                               int32 j2,
                               BaseFloat weight,
                               Sgmm2PerSpkDerivedVars *spk_vars) {
  // Calculate Gaussian posteriors and collect statistics
  Matrix<BaseFloat> posteriors;
  BaseFloat log_like = model.ComponentPosteriors(frame_vars, j2, spk_vars,
                                                 &posteriors);
  posteriors.Scale(weight);
  AccumulateFromPosteriors(model, frame_vars, posteriors, j2, spk_vars);
  return log_like;
}

BaseFloat
MleSgmm2SpeakerAccs::AccumulateFromPosteriors(const AmSgmm2 &model,
                                             const Sgmm2PerFrameDerivedVars &frame_vars,
                                             const Matrix<BaseFloat> &posteriors,
                                             int32 j2,
                                             Sgmm2PerSpkDerivedVars *spk_vars) {
  double tot_count = 0.0;
  int32 feature_dim = model.FeatureDim(),
      spk_space_dim = model.SpkSpaceDim();
  KALDI_ASSERT(spk_space_dim != 0);
  const vector<int32> &gselect = frame_vars.gselect;

  // Intermediate variables
  Vector<double> xt_jmi(feature_dim), mu_jmi(feature_dim),
      zt_jmi(spk_space_dim);
  int32 num_substates = model.NumSubstatesForPdf(j2),
      j1 = model.Pdf2Group(j2);
  bool have_spk_dep_weights = (a_s_.Dim() != 0);

  for (int32 m = 0; m < num_substates; m++) {
    BaseFloat gammat_jm = 0.0;
    for (int32 ki = 0; ki < static_cast<int32>(gselect.size()); ki++) {
      int32 i = gselect[ki];
      // Eq. (39): gamma_{jmi}(t) = p (j, m, i|t)
      BaseFloat gammat_jmi = RandPrune(posteriors(ki, m), rand_prune_);
      if (gammat_jmi != 0.0) {
        gammat_jm += gammat_jmi;
        tot_count += gammat_jmi;
        model.GetSubstateMean(j1, m, i, &mu_jmi);
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
    if (have_spk_dep_weights) {
      KALDI_ASSERT(!model.w_jmi_.empty());
      BaseFloat d_jms = model.GetDjms(j1, m, spk_vars);
      if (d_jms == -1.0) d_jms = 1.0; // Explanation: d_jms is set to -1 when we didn't have
      // speaker vectors in training.  We treat this the same as the speaker vector being
      // zero, and d_jms becomes 1 in this case.
      a_s_.AddVec(gammat_jm/d_jms, model.w_jmi_[j1].Row(m));
    }
  }
  return tot_count;
}

void MleSgmm2SpeakerAccs::Update(const AmSgmm2 &model,
                                BaseFloat min_count,
                                Vector<BaseFloat> *v_s,
                                BaseFloat *objf_impr_out,
                                BaseFloat *count_out) {
  double tot_gamma = gamma_s_.Sum();
  if (tot_gamma < min_count) {
    KALDI_WARN << "Updating speaker vectors, count is " << tot_gamma
               << " < " << min_count << "not updating.";
    if (objf_impr_out) *objf_impr_out = 0.0;
    if (count_out) *count_out = 0.0;
    return;
  }
  if (a_s_.Dim() == 0) // No speaker-dependent weights...
    UpdateNoU(v_s, objf_impr_out, count_out);
  else
    UpdateWithU(model, v_s, objf_impr_out, count_out);
}


// Basic update, no SSGMM.
void MleSgmm2SpeakerAccs::UpdateNoU(Vector<BaseFloat> *v_s,
                                BaseFloat *objf_impr_out,
                                BaseFloat *count_out) {
  double tot_gamma = gamma_s_.Sum();
  KALDI_ASSERT(y_s_.Dim() != 0);
  int32 T = y_s_.Dim();  // speaker-subspace dim.
  int32 num_gauss = gamma_s_.Dim();
  if (v_s->Dim() != T) v_s->Resize(T);  // will set it to zero.

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
            << " over " << tot_gamma << " frames.";

  if (objf_impr_out) *objf_impr_out = tot_objf_impr;
  if (count_out) *count_out = tot_gamma;
}

// Basic update, no SSGMM.
void MleSgmm2SpeakerAccs::UpdateWithU(const AmSgmm2 &model,
                                     Vector<BaseFloat> *v_s_ptr,
                                     BaseFloat *objf_impr_out,
                                     BaseFloat *count_out) {
  double tot_gamma = gamma_s_.Sum();
  KALDI_ASSERT(y_s_.Dim() != 0);
  int32 T = y_s_.Dim();  // speaker-subspace dim.
  int32 num_gauss = gamma_s_.Dim();
  if (v_s_ptr->Dim() != T) v_s_ptr->Resize(T);  // will set it to zero.

  // Eq. (84): H^{(s)} = \sum_{i} \gamma_{i}(s) H_{i}^{spk}
  SpMatrix<double> H_s(T);
  
  for (int32 i = 0; i < num_gauss; i++)
    H_s.AddSp(gamma_s_(i), H_spk_[i]);
  
  Vector<double> v_s(*v_s_ptr);
  int32 num_iters = 5, // don't set this to 1, as we discard last iter.
      num_backtracks = 0,
      max_backtracks = 10;
  Vector<double> auxf(num_iters);
  Matrix<double> v_s_per_iter(num_iters, T);
  // The update for v^{(s)} is the one described in the technical report
  // section 5.1 (eq. 33 and below).
  
  for (int32 iter = 0; iter < num_iters; iter++) { // converges very fast,
    // and each iteration is fast, so don't need to make this configurable.
    v_s_per_iter.Row(iter).CopyFromVec(v_s);
    
    SpMatrix<double> F(H_s); // the 2nd-order quadratic term on this iteration...
    // F^{(p)} in the techerport.
    Vector<double> g(y_s_); // g^{(p)} in the techreport.
    g.AddSpVec(-1.0, H_s, v_s, 1.0);
    Vector<double> log_b_is(num_gauss); // b_i^{(s)}, indexed by i.
    log_b_is.AddMatVec(1.0, Matrix<double>(model.u_), kNoTrans, v_s, 0.0);
    Vector<double> tilde_w_is(log_b_is);
    Vector<double> log_a_s_(a_s_);
    log_a_s_.ApplyLog();
    tilde_w_is.AddVec(1.0, log_a_s_);
    tilde_w_is.Add(-1.0 * tilde_w_is.LogSumExp()); // normalize.
    // currently tilde_w_is is in log form.
    auxf(iter) = VecVec(v_s, y_s_) - 0.5 * VecSpVec(v_s, H_s, v_s)
        + VecVec(gamma_s_, tilde_w_is); // "new" term (weights)
    
    if (iter > 0 && auxf(iter) < auxf(iter-1) &&
        !ApproxEqual(auxf(iter), auxf(iter-1))) { // auxf did not improve.
      // backtrack halfway, and do this iteration again.
      KALDI_WARN << "Backtracking in speaker vector update, on iter "
                 << iter << ", auxfs are " << auxf(iter-1) << " -> "
                 << auxf(iter);
      v_s.Scale(0.5);
      v_s.AddVec(0.5, v_s_per_iter.Row(iter-1));
      if (++num_backtracks >= max_backtracks) {
        KALDI_WARN << "Backtracked " << max_backtracks
                   << " times in speaker-vector update.";
        // backtrack all the way, and terminate:
        v_s_per_iter.Row(num_iters-1).CopyFromVec(v_s_per_iter.Row(iter-1));
        // the following statement ensures we will get
        // the appropriate auxiliary-function.
        auxf(num_iters-1) = auxf(iter-1);
        break;
      }
      iter--;
    }        
    tilde_w_is.ApplyExp();
    for (int32 i = 0; i < num_gauss; i++) {
      g.AddVec(gamma_s_(i) - tot_gamma * tilde_w_is(i), model.u_.Row(i));
      F.AddVec2(tot_gamma * tilde_w_is(i), model.u_.Row(i));
    }
    Vector<double> delta(v_s.Dim());
    SolveQuadraticProblem(F, g, SolverOptions("v_s"), &delta);
    v_s.AddVec(1.0, delta);
  }
  // so that we only accept things where the auxf has been checked, we
  // actually take the penultimate speaker-vector. --> don't set
  // num-iters = 1.
  v_s_ptr->CopyFromVec(v_s_per_iter.Row(num_iters-1));  
  
  double auxf_change = auxf(num_iters-1) - auxf(0);
  KALDI_LOG << "*Objf impr for speaker vector is " << (auxf_change / tot_gamma)
            << " per frame, over " << tot_gamma << " frames.";
  
  if (objf_impr_out) *objf_impr_out = auxf_change;
  if (count_out) *count_out = tot_gamma;
}


MleAmSgmm2Accs::~MleAmSgmm2Accs() {
  if (gamma_s_.Sum() != 0.0)
    KALDI_ERR << "In destructor of MleAmSgmm2Accs: detected that you forgot to "
        "call CommitStatsForSpk()";
}


}  // namespace kaldi

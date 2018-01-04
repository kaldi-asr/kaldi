// sgmm2/am-sgmm2.cc

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

#include <functional>

#include "sgmm2/am-sgmm2.h"
#include "util/kaldi-thread.h"

namespace kaldi {
using std::vector;

// This function needs to be added because std::generate is complaining
// about RandGauss(), which takes an optional arguments.
static inline float _RandGauss()
{
  return RandGauss();
}

void Sgmm2LikelihoodCache::NextFrame() {
  t++;
  if (t == 0) {
    t++; // skip over zero; zero is used to invalidate frames.
    for (size_t i = 0; i < substate_cache.size(); i++)
      substate_cache[i].t = 0;
    for (size_t i = 0; i < pdf_cache.size(); i++)
      pdf_cache[i].t = 0;
  }
}

void AmSgmm2::ComputeGammaI(const Vector<BaseFloat> &state_occupancies,
                            Vector<BaseFloat> *gamma_i) const {
  KALDI_ASSERT(state_occupancies.Dim() == NumPdfs());
  Vector<BaseFloat> w_jm(NumGauss());
  gamma_i->Resize(NumGauss());
  for (int32 j1 = 0; j1 < NumGroups(); j1++) {
    int32 M = NumSubstatesForGroup(j1);
    const std::vector<int32> &pdfs = group2pdf_[j1];
    Vector<BaseFloat> substate_weight(M); // total weight for each substate.
    for (size_t i = 0; i < pdfs.size(); i++) {
      int32 j2 = pdfs[i];
      substate_weight.AddVec(state_occupancies(j2), c_[j2]);
    }
    for (int32 m = 0; m < M; m++) {
      w_jm.AddMatVec(1.0, w_, kNoTrans, v_[j1].Row(m), 0.0);
      w_jm.ApplySoftMax();
      gamma_i->AddVec(substate_weight(m), w_jm);
    }
  }
}


void AmSgmm2::ComputePdfMappings() {
  if (pdf2group_.empty()) {
    KALDI_WARN << "ComputePdfMappings(): no pdf2group_ map, assuming you "
        "are reading in old model.";
    KALDI_ASSERT(v_.size() != 0);
    pdf2group_.resize(v_.size());
    for (int32 j2 = 0; j2 < static_cast<int32>(pdf2group_.size()); j2++)
      pdf2group_[j2] = j2;
  }
  group2pdf_.clear();
  for (int32 j2 = 0; j2 < static_cast<int32>(pdf2group_.size()); j2++) {
    int32 j1 = pdf2group_[j2];
    if (group2pdf_.size() <= j1) group2pdf_.resize(j1+1);
    group2pdf_[j1].push_back(j2);
  }
}

void AmSgmm2::Read(std::istream &in_stream, bool binary) {
  { // We want this to work even if the object was previously
    // populated, so we clear the items that are more likely
    // to cause problems.
    pdf2group_.clear();
    group2pdf_.clear();
    u_.Resize(0,0);
    w_jmi_.clear();
    v_.clear();
  }
  // removing anything that was in the object before.
  int32 num_pdfs = -1, feat_dim, num_gauss;
  std::string token;

  ExpectToken(in_stream, binary, "<SGMM>");
  ExpectToken(in_stream, binary, "<NUMSTATES>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  ExpectToken(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &feat_dim);
  ExpectToken(in_stream, binary, "<NUMGAUSS>");
  ReadBasicType(in_stream, binary, &num_gauss);

  KALDI_ASSERT(num_pdfs > 0 && feat_dim > 0);

  ReadToken(in_stream, binary, &token);

  while (token != "</SGMM>") {
    if (token == "<PDF2GROUP>") {
      ReadIntegerVector(in_stream, binary, &pdf2group_);
      ComputePdfMappings();
    } else if (token == "<WEIGHTIDX2GAUSS>") {  // TEMP!   Will remove.
      std::vector<int32> garbage;
      ReadIntegerVector(in_stream, binary, &garbage);
    } else if (token == "<DIAG_UBM>") {
      diag_ubm_.Read(in_stream, binary);
    } else if (token == "<FULL_UBM>") {
      full_ubm_.Read(in_stream, binary);
    } else if (token == "<SigmaInv>") {
      SigmaInv_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        SigmaInv_[i].Read(in_stream, binary);
      }
    } else if (token == "<M>") {
      M_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        M_[i].Read(in_stream, binary);
      }
    } else if (token == "<N>") {
      N_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        N_[i].Read(in_stream, binary);
      }
    } else if (token == "<w>") {
      w_.Read(in_stream, binary);
    } else if (token == "<u>") {
      u_.Read(in_stream, binary);
    } else if (token == "<v>") {
      int32 num_groups = group2pdf_.size();
      if (num_groups == 0) {
        KALDI_WARN << "Reading old model with new code (should still work)";
        num_groups = num_pdfs;
      }
      v_.resize(num_groups);
      for (int32 j1 = 0; j1 < num_groups; j1++) {
        v_[j1].Read(in_stream, binary);
      }
    } else if (token == "<c>") {
      c_.resize(num_pdfs);
      for (int32 j2 = 0; j2 < num_pdfs; j2++) {
        c_[j2].Read(in_stream, binary);
      }
    } else if (token == "<n>") {
      int32 num_groups = group2pdf_.size();
      if (num_groups == 0) num_groups = num_pdfs;
      n_.resize(num_groups);
      for (int32 j1 = 0; j1 < num_groups; j1++) {
        n_[j1].Read(in_stream, binary);
      }
      // The following are the Gaussian prior parameters for MAP adaptation of M
      // They may be moved to somewhere else eventually.
    } else if (token == "<M_Prior>") {
      ExpectToken(in_stream, binary, "<NUMGaussians>");
      ReadBasicType(in_stream, binary, &num_gauss);
      M_prior_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        M_prior_[i].Read(in_stream, binary);
      }
    } else if (token == "<Row_Cov_Inv>") {
      row_cov_inv_.Read(in_stream, binary);
    } else if (token == "<Col_Cov_Inv>") {
      col_cov_inv_.Read(in_stream, binary);
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadToken(in_stream, binary, &token);
  }

  if (pdf2group_.empty())
    ComputePdfMappings(); // sets up group2pdf_, and pdf2group_ if reading
  // old model.

  if (n_.empty())
    ComputeNormalizers();
  if (HasSpeakerDependentWeights())
    ComputeWeights();
}

int32 AmSgmm2::Pdf2Group(int32 j2) const {
  KALDI_ASSERT(static_cast<size_t>(j2) < pdf2group_.size());
  int32 j1 = pdf2group_[j2];
  return j1;
}


void AmSgmm2::Write(std::ostream &out_stream,
                   bool binary,
                   SgmmWriteFlagsType write_params) const {
  int32 num_pdfs = NumPdfs(),
      feat_dim = FeatureDim(),
      num_gauss = NumGauss();

  WriteToken(out_stream, binary, "<SGMM>");
  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<NUMSTATES>");
  WriteBasicType(out_stream, binary, num_pdfs);
  WriteToken(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, feat_dim);
  WriteToken(out_stream, binary, "<NUMGAUSS>");
  WriteBasicType(out_stream, binary, num_gauss);
  if (!binary) out_stream << "\n";

  if (write_params & kSgmmBackgroundGmms) {
    WriteToken(out_stream, binary, "<DIAG_UBM>");
    diag_ubm_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<FULL_UBM>");
    full_ubm_.Write(out_stream, binary);
  }

  if (write_params & kSgmmGlobalParams) {
    WriteToken(out_stream, binary, "<SigmaInv>");
    if (!binary) out_stream << "\n";
    for (int32 i = 0; i < num_gauss; i++) {
      SigmaInv_[i].Write(out_stream, binary);
    }
    WriteToken(out_stream, binary, "<M>");
    if (!binary) out_stream << "\n";
    for (int32 i = 0; i < num_gauss; i++) {
      M_[i].Write(out_stream, binary);
    }
    if (N_.size() != 0) {
      WriteToken(out_stream, binary, "<N>");
      if (!binary) out_stream << "\n";
      for (int32 i = 0; i < num_gauss; i++) {
        N_[i].Write(out_stream, binary);
      }
    }
    WriteToken(out_stream, binary, "<w>");
    w_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<u>");
    u_.Write(out_stream, binary);
  }

  if (write_params & kSgmmStateParams) {
    WriteToken(out_stream, binary, "<PDF2GROUP>");
    WriteIntegerVector(out_stream, binary, pdf2group_);
    WriteToken(out_stream, binary, "<v>");
    for (int32 j1 = 0; j1 < NumGroups(); j1++) {
      v_[j1].Write(out_stream, binary);
    }
    WriteToken(out_stream, binary, "<c>");
    for (int32 j2 = 0; j2 < num_pdfs; j2++) {
      c_[j2].Write(out_stream, binary);
    }
  }

  if (write_params & kSgmmNormalizers) {
    WriteToken(out_stream, binary, "<n>");
    if (n_.empty())
      KALDI_WARN << "Not writing normalizers since they are not present.";
    else
      for (int32 j1 = 0; j1 < NumGroups(); j1++)
        n_[j1].Write(out_stream, binary);
  }
  WriteToken(out_stream, binary, "</SGMM>");
}


void AmSgmm2::Check(bool show_properties) {
  int32 J1 = NumGroups(),
      J2 = NumPdfs(),
      num_gauss = NumGauss(),
      feat_dim = FeatureDim(),
      phn_dim = PhoneSpaceDim(),
      spk_dim = SpkSpaceDim();

  if (show_properties)
    KALDI_LOG << "AmSgmm2: #pdfs = " << J2 << ", #pdf-groups = "
              << J1 << ", #Gaussians = "
              << num_gauss << ", feature dim = " << feat_dim
              << ", phone-space dim =" << phn_dim
              << ", speaker-space dim =" << spk_dim;
  KALDI_ASSERT(J1 > 0 && num_gauss > 0 && feat_dim > 0 && phn_dim > 0
               && J2 > 0 && J2 >= J1);

  std::ostringstream debug_str;

  // First check the diagonal-covariance UBM.
  KALDI_ASSERT(diag_ubm_.NumGauss() == num_gauss);
  KALDI_ASSERT(diag_ubm_.Dim() == feat_dim);

  // Check the full-covariance UBM.
  KALDI_ASSERT(full_ubm_.NumGauss() == num_gauss);
  KALDI_ASSERT(full_ubm_.Dim() == feat_dim);

  // Check the globally-shared covariance matrices.
  KALDI_ASSERT(SigmaInv_.size() == static_cast<size_t>(num_gauss));
  for (int32 i = 0; i < num_gauss; i++) {
    KALDI_ASSERT(SigmaInv_[i].NumRows() == feat_dim &&
                 SigmaInv_[i](0, 0) > 0.0);  // or it wouldn't be +ve definite.
  }

  if (spk_dim != 0) {
    KALDI_ASSERT(N_.size() == static_cast<size_t>(num_gauss));
    for (int32 i = 0; i < num_gauss; i++)
      KALDI_ASSERT(N_[i].NumRows() == feat_dim && N_[i].NumCols() == spk_dim);
    if (u_.NumRows() == 0) {
      debug_str << "Speaker-weight projections: no.";
    } else {
      KALDI_ASSERT(u_.NumRows() == num_gauss && u_.NumCols() == spk_dim);
      debug_str << "Speaker-weight projections: yes.";
    }
  } else {
    KALDI_ASSERT(N_.size() == 0 && u_.NumRows() == 0);
  }

  KALDI_ASSERT(M_.size() == static_cast<size_t>(num_gauss));
  for (int32 i = 0; i < num_gauss; i++) {
    KALDI_ASSERT(M_[i].NumRows() == feat_dim && M_[i].NumCols() == phn_dim);
  }

  KALDI_ASSERT(w_.NumRows() == num_gauss && w_.NumCols() == phn_dim);

  {  // check v, c.
    KALDI_ASSERT(v_.size() == static_cast<size_t>(J1) &&
                 c_.size() == static_cast<size_t>(J2));
    int32 nSubstatesTot = 0;
    for (int32 j1 = 0; j1 < J1; j1++) {
      int32 M_j = NumSubstatesForGroup(j1);
      nSubstatesTot += M_j;
      KALDI_ASSERT(M_j > 0 && v_[j1].NumRows() == M_j &&
                   v_[j1].NumCols() == phn_dim);
    }
    debug_str << "Substates: "<< (nSubstatesTot) << ".  ";
    int32 nSubstateWeights = 0;
    for (int32 j2 = 0; j2 < J2; j2++) {
      int32 j1 = Pdf2Group(j2);
      int32 M = NumSubstatesForPdf(j2);
      KALDI_ASSERT(M == NumSubstatesForGroup(j1));
      nSubstateWeights += M;
    }
    KALDI_ASSERT(nSubstateWeights >= nSubstatesTot);
    debug_str << "SubstateWeights: "<< (nSubstateWeights) << ".  ";
  }

  // check normalizers.
  if (n_.size() == 0) {
    debug_str << "Normalizers: no.  ";
  } else {
    debug_str << "Normalizers: yes.  ";
    KALDI_ASSERT(n_.size() == static_cast<size_t>(J1));
    for (int32 j1 = 0; j1 < J1; j1++) {
      KALDI_ASSERT(n_[j1].NumRows() == num_gauss &&
                   n_[j1].NumCols() == NumSubstatesForGroup(j1));
    }
  }

  // check w_jmi_.
  if (w_jmi_.size() == 0) {
    debug_str << "Computed weights: no.  ";
  } else {
    debug_str << "Computed weights: yes.  ";
    KALDI_ASSERT(w_jmi_.size() == static_cast<size_t>(J1));
    for (int32 j1 = 0; j1 < J1; j1++) {
      KALDI_ASSERT(w_jmi_[j1].NumRows() == NumSubstatesForGroup(j1) &&
                   w_jmi_[j1].NumCols() == num_gauss);
    }
  }

  if (show_properties)
    KALDI_LOG << "Subspace GMM model properties: " << debug_str.str();
}

void AmSgmm2::InitializeFromFullGmm(const FullGmm &full_gmm,
                                    const std::vector<int32> &pdf2group,
                                    int32 phn_subspace_dim,
                                    int32 spk_subspace_dim,
                                    bool speaker_dependent_weights,
                                    BaseFloat self_weight) {
  pdf2group_ = pdf2group;
  ComputePdfMappings();
  full_ubm_.CopyFromFullGmm(full_gmm);
  diag_ubm_.CopyFromFullGmm(full_gmm);
  if (phn_subspace_dim < 1 || phn_subspace_dim > full_gmm.Dim() + 1) {
    KALDI_WARN << "Initial phone-subspace dimension must be >= 1, value is "
               << phn_subspace_dim << "; setting to " << full_gmm.Dim() + 1;
    phn_subspace_dim = full_gmm.Dim() + 1;
  }
  KALDI_ASSERT(spk_subspace_dim >= 0);

  w_.Resize(0, 0);
  N_.clear();
  c_.clear();
  v_.clear();
  SigmaInv_.clear();

  KALDI_LOG << "Initializing model";
  Matrix<BaseFloat> norm_xform;
  ComputeFeatureNormalizingTransform(full_gmm, &norm_xform);
  InitializeMw(phn_subspace_dim, norm_xform);
  if (spk_subspace_dim > 0)
    InitializeNu(spk_subspace_dim, norm_xform, speaker_dependent_weights);
  InitializeVecsAndSubstateWeights(self_weight);
  KALDI_LOG << "Initializing variances";
  InitializeCovars();
}

void AmSgmm2::CopyFromSgmm2(const AmSgmm2 &other,
                          bool copy_normalizers,
                          bool copy_weights) {
  KALDI_LOG << "Copying AmSgmm2";
  pdf2group_ = other.pdf2group_;
  group2pdf_ = other.group2pdf_;

  // Copy background GMMs
  diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
  full_ubm_.CopyFromFullGmm(other.full_ubm_);

  // Copy global params
  SigmaInv_ = other.SigmaInv_;
  M_ = other.M_;
  w_ = other.w_;
  N_ = other.N_;
  u_ = other.u_;

  // Copy state-specific params, but only copy normalizers if requested.
  v_ = other.v_;
  c_ = other.c_;
  if (copy_normalizers) n_ = other.n_;
  if (copy_weights) w_jmi_ = other.w_jmi_;

  KALDI_LOG << "Done.";
}

void AmSgmm2::ComputePerFrameVars(const VectorBase<BaseFloat> &data,
                                 const std::vector<int32> &gselect,
                                 const Sgmm2PerSpkDerivedVars &spk_vars,
                                 Sgmm2PerFrameDerivedVars *per_frame_vars) const {
  KALDI_ASSERT(!n_.empty() && "ComputeNormalizers() must be called.");

  per_frame_vars->Resize(gselect.size(), FeatureDim(), PhoneSpaceDim());

  per_frame_vars->gselect = gselect;
  per_frame_vars->xt.CopyFromVec(data);

  for (int32 ki = 0, last = gselect.size(); ki < last; ki++) {
    int32 i = gselect[ki];
    per_frame_vars->xti.Row(ki).CopyFromVec(per_frame_vars->xt);
    if (spk_vars.v_s.Dim() != 0)
      per_frame_vars->xti.Row(ki).AddVec(-1.0, spk_vars.o_s.Row(i));
  }
  Vector<BaseFloat> SigmaInv_xt(FeatureDim());

  bool speaker_dep_weights =
      (spk_vars.v_s.Dim() != 0 && HasSpeakerDependentWeights());
  for (int32 ki = 0, last = gselect.size(); ki < last; ki++) {
    int32 i = gselect[ki];
    BaseFloat ssgmm_term = (speaker_dep_weights ? spk_vars.log_b_is(i) : 0.0);
    SigmaInv_xt.AddSpVec(1.0, SigmaInv_[i], per_frame_vars->xti.Row(ki), 0.0);
    // Eq (35): z_{i}(t) = M_{i}^{T} \Sigma_{i}^{-1} x_{i}(t)
    per_frame_vars->zti.Row(ki).AddMatVec(1.0, M_[i], kTrans, SigmaInv_xt, 0.0);
    // Eq.(36): n_{i}(t) = -0.5 x_{i}^{T} \Sigma_{i}^{-1} x_{i}(t)
    per_frame_vars->nti(ki) = -0.5 * VecVec(per_frame_vars->xti.Row(ki),
                                            SigmaInv_xt) + ssgmm_term;
  }
}

// inline
void AmSgmm2::ComponentLogLikes(const Sgmm2PerFrameDerivedVars &per_frame_vars,
                               int32 j1,
                               Sgmm2PerSpkDerivedVars *spk_vars,
                               Matrix<BaseFloat> *loglikes) const {
  const vector<int32> &gselect = per_frame_vars.gselect;
  int32 num_gselect = gselect.size(), num_substates = v_[j1].NumRows();

  // Eq.(37): log p(x(t), m, i|j)  [indexed by j, ki]
  // Although the extra memory allocation of storing this as a
  // matrix might seem unnecessary, we save time in the LogSumExp()
  // via more effective pruning.
  loglikes->Resize(num_gselect, num_substates);
  bool speaker_dep_weights =
      (spk_vars->v_s.Dim() != 0 && HasSpeakerDependentWeights());
  if (speaker_dep_weights) {
    KALDI_ASSERT(static_cast<int32>(spk_vars->log_d_jms.size()) == NumGroups());
    KALDI_ASSERT(static_cast<int32>(w_jmi_.size()) == NumGroups() ||
                 "You need to call ComputeWeights().");
  }
  for (int32 ki = 0;  ki < num_gselect; ki++) {
    SubVector<BaseFloat> logp_xi(*loglikes, ki);
    int32 i = gselect[ki];
    // for all substates, compute z_{i}^T v_{jm}
    logp_xi.AddMatVec(1.0, v_[j1], kNoTrans, per_frame_vars.zti.Row(ki), 0.0);
    logp_xi.AddVec(1.0, n_[j1].Row(i));  // for all substates, add n_{jim}
    logp_xi.Add(per_frame_vars.nti(ki));  // for all substates, add n_{i}(t)
  }
  if (speaker_dep_weights) { // [SSGMM]
    Vector<BaseFloat> &log_d = spk_vars->log_d_jms[j1];
    if (log_d.Dim() == 0) { // have not yet cached this quantity.
      log_d.Resize(num_substates);
      log_d.AddMatVec(1.0, w_jmi_[j1], kNoTrans, spk_vars->b_is, 0.0);
      log_d.ApplyLog();
    }
    loglikes->AddVecToRows(-1.0, log_d); // [SSGMM] this is the term
    // - log d_{jm}^{(s)} in the likelihood function [eq. 25 in
    // the techreport]
  }
}


BaseFloat AmSgmm2::LogLikelihood(const Sgmm2PerFrameDerivedVars &per_frame_vars,
                                int32 j2,
                                Sgmm2LikelihoodCache *cache,
                                Sgmm2PerSpkDerivedVars *spk_vars,
                                BaseFloat log_prune) const {
  int32 t = cache->t; // not a real time; used to uniquely identify frames.
  // Forgo asserts here, as this is frequently called.
  // We'll probably get a segfault if an error is made.
  Sgmm2LikelihoodCache::PdfCacheElement &pdf_cache =
      cache->pdf_cache[j2];
#ifdef KALDI_PARANOID
  bool random_test = (Rand() % 1000 == 1); // to check that the user is
  // calling Next() on the cache, as they should.
#else
  bool random_test = false; // compiler will ignore test branches.
#endif
  if (pdf_cache.t == t) {
    if (!random_test) return pdf_cache.log_like;
  } else {
    random_test = false;
  }
  // if random_test == true at this point, it was already cached, and we will
  // verify that we return the same value as the cached one.
  pdf_cache.t = t;

  int32 j1 = pdf2group_[j2];
  Sgmm2LikelihoodCache::SubstateCacheElement &substate_cache =
      cache->substate_cache[j1];
  if (substate_cache.t != t) { // Need to compute sub-state likelihoods.
    substate_cache.t = t;
    Matrix<BaseFloat> loglikes; // indexed [gselect-index][substate-index]
    ComponentLogLikes(per_frame_vars, j1, spk_vars, &loglikes);
    BaseFloat max = loglikes.Max(); // use this to keep things in good numerical range.
    loglikes.Add(-max);
    loglikes.ApplyExp();
    substate_cache.remaining_log_like = max;
    int32 num_substates = loglikes.NumCols();
    substate_cache.likes.Resize(num_substates); // zeroes it.
    substate_cache.likes.AddRowSumMat(1.0, loglikes); // add likelihoods [not in log!] for
    // each column [i.e. summing over the rows], so we get the sum for
    // each substate index.  You have to multiply by exp(remaining_log_like)
    // to get a real likelihood.
  }

  BaseFloat log_like = substate_cache.remaining_log_like
      + Log(VecVec(substate_cache.likes, c_[j2]));

  if (random_test)
    KALDI_ASSERT(ApproxEqual(pdf_cache.log_like, log_like));

  pdf_cache.log_like = log_like;
  KALDI_ASSERT(log_like == log_like && log_like - log_like == 0); // check
  // that it's not NaN or infinity.
  return log_like;
}

BaseFloat
AmSgmm2::ComponentPosteriors(const Sgmm2PerFrameDerivedVars &per_frame_vars,
                            int32 j2,
                            Sgmm2PerSpkDerivedVars *spk_vars,
                            Matrix<BaseFloat> *post) const {
  KALDI_ASSERT(j2 < NumPdfs() && post != NULL);
  int32 j1 = pdf2group_[j2];
  ComponentLogLikes(per_frame_vars, j1, spk_vars, post); // now
  // post is a matrix of log-likelihoods indexed by [gaussian-selection index]
  // [sub-state index].  It doesn't include the sub-state weights,
  // though.
  BaseFloat loglike = post->Max();
  post->Add(-loglike); // get it to nicer numeric range.
  post->ApplyExp(); // so we're dealing with likelihoods (with an arbitrary offset
  // "loglike" removed to make it in a nice numeric range)
  post->MulColsVec(c_[j2]); // include the sub-state weights.

  BaseFloat tot_like = post->Sum();
  KALDI_ASSERT(tot_like != 0.0); // note: not valid to have zero weights.
  loglike += Log(tot_like);
  post->Scale(1.0 / tot_like); // so "post" now sums to one, and "loglike"
  // contains the correct log-likelihood of the data given the pdf.

  return loglike;
}

void AmSgmm2::SplitSubstatesInGroup(const Vector<BaseFloat> &pdf_occupancies,
                                    const Sgmm2SplitSubstatesConfig &opts,
                                    const SpMatrix<BaseFloat> &sqrt_H_sm,
                                    int32 j1,
                                    int32 tgt_M) {
  const std::vector<int32> &pdfs = group2pdf_[j1];
  int32 phn_dim = PhoneSpaceDim(), cur_M = NumSubstatesForGroup(j1),
      num_pdfs_for_group = pdfs.size();
  Vector<BaseFloat> rand_vec(phn_dim), v_shift(phn_dim);

  KALDI_ASSERT(tgt_M >= cur_M);
  if (cur_M == tgt_M) return;
  // Resize v[j1] to fit new substates
  {
    Matrix<BaseFloat> tmp_v_j(v_[j1]);
    v_[j1].Resize(tgt_M, phn_dim);
    v_[j1].Range(0, cur_M, 0, phn_dim).CopyFromMat(tmp_v_j);
  }

  // we'll use a temporary matrix for the c quantities.
  Matrix<BaseFloat> c_j(num_pdfs_for_group, tgt_M);
  for (int32 i = 0; i < num_pdfs_for_group; i++) {
    int32 j2 = pdfs[i];
    c_j.Row(i).Range(0, cur_M).CopyFromVec(c_[j2]);
  }

  // Keep splitting substates until obtaining the desired number
  for (; cur_M < tgt_M; cur_M++) {
    int32 split_m; // substate to split.
    {
      Vector<BaseFloat> substate_count(tgt_M);
      substate_count.AddRowSumMat(1.0, c_j);
      BaseFloat *data = substate_count.Data();
      split_m = std::max_element(data, data+cur_M) - data;
    }
    for (int32 i = 0; i < num_pdfs_for_group; i++) { // divide count of split
      // substate. [extended for SCTM]
      // c_{jkm} := c_{jmk}' := c_{jkm} / 2
      c_j(i, split_m) = c_j(i, cur_M) = c_j(i, split_m) / 2;
    }
    // v_{jkm} := +/- split_perturb * H_k^{(sm)}^{-0.5} * rand_vec
    std::generate(rand_vec.Data(), rand_vec.Data() + rand_vec.Dim(),
                  _RandGauss);
    v_shift.AddSpVec(opts.perturb_factor, sqrt_H_sm, rand_vec, 0.0);
    v_[j1].Row(cur_M).CopyFromVec(v_[j1].Row(split_m));
    v_[j1].Row(cur_M).AddVec(1.0, v_shift);
    v_[j1].Row(split_m).AddVec(-1.0, v_shift);
  }
  // copy the temporary matrix for the c_ (sub-state weight)
  // quantities back to the place it belongs.
  for (int32 i = 0; i < num_pdfs_for_group; i++) {
    int32 j2 = pdfs[i];
    c_[j2].Resize(tgt_M);
    c_[j2].CopyFromVec(c_j.Row(i));
  }
}


void AmSgmm2::SplitSubstates(const Vector<BaseFloat> &pdf_occupancies,
                             const Sgmm2SplitSubstatesConfig &opts) {
  KALDI_ASSERT(pdf_occupancies.Dim() == NumPdfs());
  int32 J1 = NumGroups(), J2 = NumPdfs();
  Vector<BaseFloat> group_occupancies(J1);
  for (int32 j2 = 0; j2 < J2; j2++)
    group_occupancies(Pdf2Group(j2)) += pdf_occupancies(j2);

  vector<int32> tgt_num_substates;

  GetSplitTargets(group_occupancies, opts.split_substates,
                  opts.power, opts.min_count, &tgt_num_substates);

  int32 tot_num_substates_old = 0, tot_num_substates_new = 0;
  vector< SpMatrix<BaseFloat> > H_i;
  SpMatrix<BaseFloat> sqrt_H_sm;

  ComputeH(&H_i);  // set up that array.
  ComputeHsmFromModel(H_i, pdf_occupancies, &sqrt_H_sm, opts.max_cond);
  H_i.clear();
  sqrt_H_sm.ApplyPow(-0.5);

  for (int32 j1 = 0; j1 < J1; j1++) {
    int32 cur_M = NumSubstatesForGroup(j1),
        tgt_M = tgt_num_substates[j1];
    tot_num_substates_old += cur_M;
    tot_num_substates_new += std::max(cur_M, tgt_M);
    if (cur_M < tgt_M)
      SplitSubstatesInGroup(pdf_occupancies, opts, sqrt_H_sm, j1, tgt_M);
  }
  if (tot_num_substates_old == tot_num_substates_new) {
    KALDI_LOG << "Not splitting substates; current #substates is "
              << tot_num_substates_old << " and target is "
              << opts.split_substates;
  } else {
    KALDI_LOG << "Getting rid of normalizers as they will no longer be valid";
    n_.clear();
    KALDI_LOG << "Split " << tot_num_substates_old << " substates to "
              << tot_num_substates_new;
  }
}

void AmSgmm2::IncreasePhoneSpaceDim(int32 target_dim,
                                   const Matrix<BaseFloat> &norm_xform) {
  KALDI_ASSERT(!M_.empty());
  int32 initial_dim = PhoneSpaceDim(),
      feat_dim = FeatureDim();
  KALDI_ASSERT(norm_xform.NumRows() == feat_dim);

  if (target_dim < initial_dim)
    KALDI_ERR << "You asked to increase phn dim to a value lower than the "
              << " current dimension, " << target_dim << " < " << initial_dim;

  if (target_dim > initial_dim + feat_dim) {
    KALDI_WARN << "Cannot increase phone subspace dimensionality from "
               << initial_dim << " to " << target_dim << ", increasing to "
               << initial_dim + feat_dim;
    target_dim = initial_dim + feat_dim;
  }

  if (initial_dim < target_dim) {
    Matrix<BaseFloat> tmp_M(feat_dim, initial_dim);
    for (int32 i = 0; i < NumGauss(); i++) {
      tmp_M.CopyFromMat(M_[i]);
      M_[i].Resize(feat_dim, target_dim);
      M_[i].Range(0, feat_dim, 0, tmp_M.NumCols()).CopyFromMat(tmp_M);
      M_[i].Range(0, feat_dim, tmp_M.NumCols(),
          target_dim - tmp_M.NumCols()).CopyFromMat(norm_xform.Range(0,
              feat_dim, 0, target_dim-tmp_M.NumCols()));
    }
    Matrix<BaseFloat> tmp_w = w_;
    w_.Resize(tmp_w.NumRows(), target_dim);
    w_.Range(0, tmp_w.NumRows(), 0, tmp_w.NumCols()).CopyFromMat(tmp_w);

    for (int32 j1 = 0; j1 < NumGroups(); j1++) {
      // Resize phonetic-subspce vectors.
      Matrix<BaseFloat> tmp_v_j = v_[j1];
      v_[j1].Resize(tmp_v_j.NumRows(), target_dim);
      v_[j1].Range(0, tmp_v_j.NumRows(), 0, tmp_v_j.NumCols()).CopyFromMat(
          tmp_v_j);
    }
    KALDI_LOG << "Phone subspace dimensionality increased from " <<
        initial_dim << " to " << target_dim;
  } else {
    KALDI_LOG << "Phone subspace dimensionality unchanged, since target " <<
        "dimension (" << target_dim << ") <= initial dimansion (" <<
        initial_dim << ")";
  }
}

void AmSgmm2::IncreaseSpkSpaceDim(int32 target_dim,
                                 const Matrix<BaseFloat> &norm_xform,
                                 bool speaker_dependent_weights) {
  int32 initial_dim = SpkSpaceDim(),
      feat_dim = FeatureDim();
  KALDI_ASSERT(norm_xform.NumRows() == feat_dim);

  if (N_.size() == 0)
    N_.resize(NumGauss());

  if (target_dim < initial_dim)
    KALDI_ERR << "You asked to increase spk dim to a value lower than the "
              << " current dimension, " << target_dim << " < " << initial_dim;

  if (target_dim > initial_dim + feat_dim) {
    KALDI_WARN << "Cannot increase speaker subspace dimensionality from "
               << initial_dim << " to " << target_dim << ", increasing to "
               << initial_dim + feat_dim;
    target_dim = initial_dim + feat_dim;
  }

  if (initial_dim < target_dim) {
    int32 dim_change = target_dim - initial_dim;
    Matrix<BaseFloat> tmp_N((initial_dim != 0) ? feat_dim : 0,
                            initial_dim);
    for (int32 i = 0; i < NumGauss(); i++) {
      if (initial_dim != 0) tmp_N.CopyFromMat(N_[i]);
      N_[i].Resize(feat_dim, target_dim);
      if (initial_dim != 0) {
        N_[i].Range(0, feat_dim, 0, tmp_N.NumCols()).CopyFromMat(tmp_N);
      }
      N_[i].Range(0, feat_dim, tmp_N.NumCols(), dim_change).CopyFromMat(
          norm_xform.Range(0, feat_dim, 0, dim_change));
    }
    // if we already have speaker-dependent weights or we are increasing
    // spk-dim from zero and are asked to add them...
    if (u_.NumRows() != 0 || (initial_dim == 0 && speaker_dependent_weights))
      u_.Resize(NumGauss(), target_dim, kCopyData); // extend dim of u_i's
    KALDI_LOG << "Speaker subspace dimensionality increased from " <<
        initial_dim << " to " << target_dim;
    if (initial_dim == 0 && speaker_dependent_weights)
      KALDI_LOG << "Added parameters u for speaker-dependent weights.";
  } else {
    KALDI_LOG << "Speaker subspace dimensionality unchanged, since target " <<
        "dimension (" << target_dim << ") <= initial dimansion (" <<
        initial_dim << ")";
  }
}

void AmSgmm2::ComputeWeights() {
  int32 J1 = NumGroups();
  w_jmi_.resize(J1);
  int32 i = NumGauss();
  for (int32 j1 = 0; j1 < J1; j1++) {
    int32 M = NumSubstatesForGroup(j1);
    w_jmi_[j1].Resize(M, i);
    w_jmi_[j1].AddMatMat(1.0, v_[j1], kNoTrans, w_, kTrans, 0.0);
    // now w_jmi_ contains un-normalized log weights.
    for (int32 m = 0; m < M; m++)
      w_jmi_[j1].Row(m).ApplySoftMax(); // get the actual weights.
  }
}

void AmSgmm2::ComputeDerivedVars() {
  if (n_.empty()) ComputeNormalizers();
  if (diag_ubm_.NumGauss() != full_ubm_.NumGauss()
      || diag_ubm_.Dim() != full_ubm_.Dim()) {
    diag_ubm_.CopyFromFullGmm(full_ubm_);
  }
  if (w_jmi_.empty() && HasSpeakerDependentWeights())
    ComputeWeights();
}

class ComputeNormalizersClass: public MultiThreadable { // For multi-threaded.
 public:
  ComputeNormalizersClass(AmSgmm2 *am_sgmm,
                          int32 *entropy_count_ptr,
                          double *entropy_sum_ptr):
      am_sgmm_(am_sgmm), entropy_count_ptr_(entropy_count_ptr),
      entropy_sum_ptr_(entropy_sum_ptr), entropy_count_(0),
      entropy_sum_(0.0) { }

  ComputeNormalizersClass(const ComputeNormalizersClass &other):
      MultiThreadable(other),
      am_sgmm_(other.am_sgmm_), entropy_count_ptr_(other.entropy_count_ptr_),
      entropy_sum_ptr_(other.entropy_sum_ptr_), entropy_count_(0),
      entropy_sum_(0.0) { }

  ~ComputeNormalizersClass() {
    *entropy_count_ptr_ += entropy_count_;
    *entropy_sum_ptr_ += entropy_sum_;
  }

  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to original pointer in the destructor.
    am_sgmm_->ComputeNormalizersInternal(num_threads_, thread_id_,
                                         &entropy_count_,
                                         &entropy_sum_);
  }
 private:
  ComputeNormalizersClass() { } // Disallow empty constructor.
  AmSgmm2 *am_sgmm_;
  int32 *entropy_count_ptr_;
  double *entropy_sum_ptr_;
  int32 entropy_count_;
  double entropy_sum_;

};

void AmSgmm2::ComputeNormalizers() {
  KALDI_LOG << "Computing normalizers";
  n_.resize(NumPdfs());
  int32 entropy_count = 0;
  double entropy_sum = 0.0;
  ComputeNormalizersClass c(this, &entropy_count, &entropy_sum);
  RunMultiThreaded(c);

  KALDI_LOG << "Entropy of weights in substates is "
            << (entropy_sum / entropy_count) << " over " << entropy_count
            << " substates, equivalent to perplexity of "
            << (Exp(entropy_sum /entropy_count));
  KALDI_LOG << "Done computing normalizers";
}


void AmSgmm2::ComputeNormalizersInternal(int32 num_threads, int32 thread,
                                         int32 *entropy_count,
                                         double *entropy_sum) {

  BaseFloat DLog2pi = FeatureDim() * Log(2 * M_PI);
  Vector<BaseFloat> log_det_Sigma(NumGauss());

  for (int32 i = 0; i < NumGauss(); i++) {
    try {
      log_det_Sigma(i) = - SigmaInv_[i].LogPosDefDet();
    } catch(...) {
      if (thread == 0) // just for one thread, print errors [else, duplicates]
        KALDI_WARN << "Covariance is not positive definite, setting to unit";
      SigmaInv_[i].SetUnit();
      log_det_Sigma(i) = 0.0;
    }
  }

  int32 J1 = NumGroups();

  int block_size = (NumPdfs() + num_threads-1) / num_threads;
  int j_start = thread * block_size, j_end = std::min(J1, j_start + block_size);

  int32 I = NumGauss();
  for (int32 j1 = j_start; j1 < j_end; j1++) {
    int32 M = NumSubstatesForGroup(j1);
    Matrix<BaseFloat> log_w_jm(M, I);
    n_[j1].Resize(I, M);
    Matrix<BaseFloat> mu_jmi(M, FeatureDim());
    Matrix<BaseFloat> SigmaInv_mu(M, FeatureDim());

    // (in logs): w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm}) eq.(7)
    log_w_jm.AddMatMat(1.0, v_[j1], kNoTrans, w_, kTrans, 0.0);
    for (int32 m = 0; m < M; m++) {
      log_w_jm.Row(m).Add(-1.0 * log_w_jm.Row(m).LogSumExp());
      {  // DIAGNOSTIC CODE
        (*entropy_count)++;
        for (int32 i = 0; i < NumGauss(); i++) {
          (*entropy_sum) -= log_w_jm(m, i) * Exp(log_w_jm(m, i));
        }
      }
    }

    for (int32 i = 0; i < I; i++) {
      // mu_jmi = M_{i} * v_{jm}
      mu_jmi.AddMatMat(1.0, v_[j1], kNoTrans, M_[i], kTrans, 0.0);
      SigmaInv_mu.AddMatSp(1.0, mu_jmi, kNoTrans, SigmaInv_[i], 0.0);

      for (int32 m = 0; m < M; m++) {
        // mu_{jmi} * \Sigma_{i}^{-1} * mu_{jmi}
        BaseFloat mu_SigmaInv_mu = VecVec(mu_jmi.Row(m), SigmaInv_mu.Row(m));
        // Previously had:
        // BaseFloat logc = log(c_[j](m));
        // but because of STCM aspect, we can't include the sub-state mixture weights
        // at this point [included later on.]

        // eq.(31)
        n_[j1](i, m) = log_w_jm(m, i) - 0.5 * (log_det_Sigma(i) + DLog2pi
            + mu_SigmaInv_mu);
        {  // Mainly diagnostic code.  Not necessary.
          BaseFloat tmp = n_[j1](i, m);
          if (!KALDI_ISFINITE(tmp)) {  // NaN or inf
            KALDI_LOG << "Warning: normalizer for j1 = " << j1 << ", m = " << m
                      << ", i = " << i << " is infinite or NaN " << tmp << "= "
                      << log_w_jm(m, i) << "+"
                      << (-0.5 * log_det_Sigma(i)) << "+" << (-0.5 * DLog2pi)
                      << "+" << (mu_SigmaInv_mu) << ", setting to finite.";
            n_[j1](i, m) = -1.0e+40;  // future work(arnab): get rid of magic number
          }
        }
      }
    }
  }
}

BaseFloat AmSgmm2::GetDjms(int32 j1, int32 m,
                          Sgmm2PerSpkDerivedVars *spk_vars) const {
  // This relates to SSGMMs (speaker-dependent weights).
  if (spk_vars->log_d_jms.empty()) return -1; // this would be
  // because we don't have speaker-dependent weights ("u" not set up).

  KALDI_ASSERT(!w_jmi_.empty() && "You need to call ComputeWeights() on SGMM.");
  Vector<BaseFloat> &log_d = spk_vars->log_d_jms[j1];
  if (log_d.Dim() == 0) {
    log_d.Resize(NumSubstatesForGroup(j1));
    log_d.AddMatVec(1.0, w_jmi_[j1], kNoTrans, spk_vars->b_is, 0.0);
    log_d.ApplyLog();
  }
  return Exp(log_d(m));
}


void AmSgmm2::ComputeFmllrPreXform(const Vector<BaseFloat> &state_occs,
                                  Matrix<BaseFloat> *xform,
                                   Matrix<BaseFloat> *inv_xform,
                                  Vector<BaseFloat> *diag_mean_scatter) const {
  int32 num_pdfs = NumPdfs(),
      num_gauss = NumGauss(),
      dim = FeatureDim();
  KALDI_ASSERT(state_occs.Dim() == num_pdfs);

  BaseFloat total_occ = state_occs.Sum();

  // Degenerate case: unlikely to ever happen.
  if (total_occ == 0) {
    KALDI_WARN << "Zero probability (computing transform). Using unit "
               << "pre-transform";
    xform->Resize(dim, dim + 1, kUndefined);
    xform->SetUnit();
    inv_xform->Resize(dim, dim + 1, kUndefined);
    inv_xform->SetUnit();
    diag_mean_scatter->Resize(dim, kSetZero);
    return;
  }

  // Convert state occupancies to posteriors; Eq. (B.1)
  Vector<BaseFloat> state_posteriors(state_occs);
  state_posteriors.Scale(1/total_occ);

  Vector<BaseFloat> mu_jmi(dim), global_mean(dim);
  SpMatrix<BaseFloat> within_class_covar(dim), between_class_covar(dim);
  Vector<BaseFloat> gauss_weight(num_gauss);  // weights for within-class vars.
  Vector<BaseFloat> w_jm(num_gauss);
  for (int32 j1 = 0; j1 < NumGroups(); j1++) {
    const std::vector<int32> &pdfs = group2pdf_[j1];
    int32 M = NumSubstatesForGroup(j1);
    Vector<BaseFloat> substate_weight(M); // total weight for each substate.
    for (size_t i = 0; i < pdfs.size(); i++) {
      int32 j2 = pdfs[i];
      substate_weight.AddVec(state_posteriors(j2), c_[j2]);
    }
    for (int32 m = 0; m < M; m++) {
      BaseFloat this_substate_weight = substate_weight(m);
      // Eq. (7): w_jm = softmax([w_{1}^T ... w_{D}^T] * v_{jm})
      w_jm.AddMatVec(1.0, w_, kNoTrans, v_[j1].Row(m), 0.0);
      w_jm.ApplySoftMax();

      for (int32 i = 0; i < num_gauss; i++) {
        BaseFloat weight = this_substate_weight * w_jm(i);
        mu_jmi.AddMatVec(1.0, M_[i], kNoTrans, v_[j1].Row(m), 0.0);  // Eq. (6)
        // Eq. (B.3): \mu_avg = \sum_{jmi} p(j) c_{jm} w_{jmi} \mu_{jmi}
        global_mean.AddVec(weight, mu_jmi);
        // \Sigma_B = \sum_{jmi} p(j) c_{jm} w_{jmi} \mu_{jmi} \mu_{jmi}^T
        between_class_covar.AddVec2(weight, mu_jmi);  // Eq. (B.4)
        gauss_weight(i) += weight;
      }
    }
  }
  between_class_covar.AddVec2(-1.0, global_mean);  // Eq. (B.4)

  for (int32 i = 0; i < num_gauss; i++) {
    SpMatrix<BaseFloat> Sigma(SigmaInv_[i]);
    Sigma.InvertDouble();
    // Eq. (B.2): \Sigma_W = \sum_{jmi} p(j) c_{jm} w_{jmi} \Sigma_i
    within_class_covar.AddSp(gauss_weight(i), Sigma);
  }

  TpMatrix<BaseFloat> tmpL(dim);
  Matrix<BaseFloat> tmpLInvFull(dim, dim);
  tmpL.Cholesky(within_class_covar);  // \Sigma_W = L L^T
  tmpL.InvertDouble();  // L^{-1}
  tmpLInvFull.CopyFromTp(tmpL);  // get as full matrix.

  // B := L^{-1} * \Sigma_B * L^{-T}
  SpMatrix<BaseFloat> tmpB(dim);
  tmpB.AddMat2Sp(1.0, tmpLInvFull, kNoTrans, between_class_covar, 0.0);

  Matrix<BaseFloat> U(dim, dim);
  diag_mean_scatter->Resize(dim);
  xform->Resize(dim, dim + 1);
  inv_xform->Resize(dim, dim + 1);

  tmpB.Eig(diag_mean_scatter, &U);  // Eq. (B.5): B = U D V^T

  int32 n;
  diag_mean_scatter->ApplyFloor(1.0e-04, &n);
  if (n != 0)
    KALDI_WARN << "Floored " << n << " elements of the mean-scatter matrix.";

  // Eq. (B.6): A_{pre} = U^T * L^{-1}
  SubMatrix<BaseFloat> Apre(*xform, 0, dim, 0, dim);
  Apre.AddMatMat(1.0, U, kTrans, tmpLInvFull, kNoTrans, 0.0);

#ifdef KALDI_PARANOID
  {
    SpMatrix<BaseFloat> tmp(dim);
    tmp.AddMat2Sp(1.0, Apre, kNoTrans, within_class_covar, 0.0);
    KALDI_ASSERT(tmp.IsUnit(0.01));
  }
  {
    SpMatrix<BaseFloat> tmp(dim);
    tmp.AddMat2Sp(1.0, Apre, kNoTrans, between_class_covar, 0.0);
    KALDI_ASSERT(tmp.IsDiagonal(0.01));
  }
#endif

  // Eq. (B.7): b_{pre} = - A_{pre} \mu_{avg}
  Vector<BaseFloat> b_pre(dim);
  b_pre.AddMatVec(-1.0, Apre, kNoTrans, global_mean, 0.0);
  for (int32 r = 0; r < dim; r++) {
    xform->Row(r)(dim) = b_pre(r);  // W_{pre} = [ A_{pre}, b_{pre} ]
  }

  // Eq. (B.8) & (B.9): W_{inv} = [ A_{pre}^{-1}, \mu_{avg} ]
  inv_xform->CopyFromMat(*xform);
  inv_xform->Range(0, dim, 0, dim).InvertDouble();
  for (int32 r = 0; r < dim; r++)
    inv_xform->Row(r)(dim) = global_mean(r);
}  // End of ComputePreXform()

template<typename Real>
void AmSgmm2::GetNtransSigmaInv(vector< Matrix<Real> > *out) const {
  KALDI_ASSERT(SpkSpaceDim() > 0 &&
      "Cannot compute N^{T} \\Sigma_{i}^{-1} without speaker projections.");
  out->resize(NumGauss());
  Matrix<Real> tmpcov(FeatureDim(), FeatureDim());
  Matrix<Real> tmp_n(FeatureDim(), SpkSpaceDim());
  for (int32 i = 0; i < NumGauss(); i++) {
    tmpcov.CopyFromSp(SigmaInv_[i]);
    tmp_n.CopyFromMat(N_[i]);
    (*out)[i].Resize(SpkSpaceDim(), FeatureDim());
    (*out)[i].AddMatMat(1.0, tmp_n, kTrans, tmpcov, kNoTrans, 0.0);
  }
}

// Instantiate the above template.
template
void AmSgmm2::GetNtransSigmaInv(vector< Matrix<float> > *out) const;
template
void AmSgmm2::GetNtransSigmaInv(vector< Matrix<double> > *out) const;

///////////////////////////////////////////////////////////////////////////////

template<class Real>
void AmSgmm2::ComputeH(std::vector< SpMatrix<Real> > *H_i) const {
  KALDI_ASSERT(NumGauss() != 0);
  (*H_i).resize(NumGauss());
  SpMatrix<BaseFloat> H_i_tmp(PhoneSpaceDim());
  for (int32 i = 0; i < NumGauss(); i++) {
    (*H_i)[i].Resize(PhoneSpaceDim());
    H_i_tmp.AddMat2Sp(1.0, M_[i], kTrans, SigmaInv_[i], 0.0);
    (*H_i)[i].CopyFromSp(H_i_tmp);
  }
}

// Instantiate the template.
template
void AmSgmm2::ComputeH(std::vector< SpMatrix<float> > *H_i) const;
template
void AmSgmm2::ComputeH(std::vector< SpMatrix<double> > *H_i) const;


// Initializes the matrices M_{i} and w_i
void AmSgmm2::InitializeMw(int32 phn_subspace_dim,
                           const Matrix<BaseFloat> &norm_xform) {
  int32 ddim = full_ubm_.Dim();
  KALDI_ASSERT(phn_subspace_dim <= ddim + 1);
  KALDI_ASSERT(phn_subspace_dim <= norm_xform.NumCols() + 1);
  KALDI_ASSERT(ddim <= norm_xform.NumRows());

  Vector<BaseFloat> mean(ddim);
  int32 num_gauss = full_ubm_.NumGauss();
  w_.Resize(num_gauss, phn_subspace_dim);
  M_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    full_ubm_.GetComponentMean(i, &mean);
    Matrix<BaseFloat> &thisM(M_[i]);
    thisM.Resize(ddim, phn_subspace_dim);
    // Eq. (27): M_{i} = [ \bar{\mu}_{i} (J)_{1:D, 1:(S-1)}]
    thisM.CopyColFromVec(mean, 0);
    int32 nonrandom_dim = std::min(phn_subspace_dim - 1, ddim),
        random_dim = phn_subspace_dim - 1 - nonrandom_dim;
    thisM.Range(0, ddim, 1, nonrandom_dim).CopyFromMat(
        norm_xform.Range(0, ddim, 0, nonrandom_dim), kNoTrans);
    // The following extension to the original paper allows us to
    // initialize the model with a larger dimension of phone-subspace vector.
    if (random_dim > 0)
      thisM.Range(0, ddim, nonrandom_dim + 1, random_dim).SetRandn();
  }
}

// Initializes the matrices N_i, and [if speaker_dependent_weights==true] u_i.
void AmSgmm2::InitializeNu(int32 spk_subspace_dim,
                          const Matrix<BaseFloat> &norm_xform,
                          bool speaker_dependent_weights) {
  int32 ddim = full_ubm_.Dim();

  int32 num_gauss = full_ubm_.NumGauss();
  N_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    N_[i].Resize(ddim, spk_subspace_dim);
    // Eq. (28): N_{i} = [ (J)_{1:D, 1:T)}]

    int32 nonrandom_dim = std::min(spk_subspace_dim, ddim),
        random_dim = spk_subspace_dim - nonrandom_dim;

    N_[i].Range(0, ddim, 0, nonrandom_dim).
        CopyFromMat(norm_xform.Range(0, ddim, 0, nonrandom_dim), kNoTrans);
    // The following extension to the original paper allows us to
    // initialize the model with a larger dimension of speaker-subspace vector.
    if (random_dim > 0)
      N_[i].Range(0, ddim, nonrandom_dim, random_dim).SetRandn();
  }
  if (speaker_dependent_weights) {
    u_.Resize(num_gauss, spk_subspace_dim); // will set to zero.
  } else {
    u_.Resize(0, 0);
  }
}

void AmSgmm2::CopyGlobalsInitVecs(const AmSgmm2 &other,
                                  const std::vector<int32> &pdf2group,
                                  BaseFloat self_weight) {
  KALDI_LOG << "Initializing model";
  pdf2group_ = pdf2group;
  ComputePdfMappings();

  // Copy background GMMs
  diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
  full_ubm_.CopyFromFullGmm(other.full_ubm_);

  // Copy global params
  SigmaInv_ = other.SigmaInv_;

  M_ = other.M_;
  w_ = other.w_;
  u_ = other.u_;
  N_ = other.N_;

  InitializeVecsAndSubstateWeights(self_weight);
}


// Initializes the vectors v_{j1,m} and substate weights c_{j2,m}.
void AmSgmm2::InitializeVecsAndSubstateWeights(BaseFloat self_weight) {
  int32 J1 = NumGroups(), J2 = NumPdfs();
  KALDI_ASSERT(J1 > 0 && J2 >= J1);
  int32 phn_subspace_dim = PhoneSpaceDim();
  KALDI_ASSERT(phn_subspace_dim > 0 && "Initialize M and w first.");

  v_.resize(J1);
  if (self_weight == 1.0) {
    for (int32 j1 = 0; j1 < J1; j1++) {
      v_[j1].Resize(1, phn_subspace_dim);
      v_[j1](0, 0) = 1.0;  // Eq. (26): v_{j1} = [1 0 0 ... 0]
    }
    c_.resize(J2);
    for (int32 j2 = 0; j2 < J2; j2++) {
      c_[j2].Resize(1);
      c_[j2](0) = 1.0;    // Eq. (25): c_{j1} = 1.0
    }
  } else {
    for (int32 j1 = 0; j1 < J1; j1++) {
      int32 npdfs = group2pdf_[j1].size();
      v_[j1].Resize(npdfs, phn_subspace_dim);
      for (int32 m = 0; m < npdfs; m++)
        v_[j1](m, 0) = 1.0;  // Eq. (26): v_{j1} = [1 0 0 ... 0]
    }
    c_.resize(J2);
    for (int32 j2 = 0; j2 < J2; j2++) {
      int32 j1 = pdf2group_[j2], npdfs = group2pdf_[j1].size();
      c_[j2].Resize(npdfs);
      if (npdfs == 1) c_[j2].Set(1.0);
      else {
        // note: just avoid NaNs if npdfs-1... value won't matter.
        double other_weight = (1.0 - self_weight) / std::max((1-npdfs), 1);
        c_[j2].Set(other_weight);
        for (int32 k = 0; k < npdfs; k++)
          if(group2pdf_[j1][k] == j2) c_[j2](k) = self_weight;
      }
    }
  }
}

// Initializes the within-class vars Sigma_{ki}
void AmSgmm2::InitializeCovars() {
  std::vector< SpMatrix<BaseFloat> > &inv_covars(full_ubm_.inv_covars());
  int32 num_gauss = full_ubm_.NumGauss();
  int32 dim = full_ubm_.Dim();
  SigmaInv_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    SigmaInv_[i].Resize(dim);
    SigmaInv_[i].CopyFromSp(inv_covars[i]);
  }
}

// Compute the "smoothing" matrix H^{(sm)} from expected counts given the model.
void AmSgmm2::ComputeHsmFromModel(
    const std::vector< SpMatrix<BaseFloat> > &H,
    const Vector<BaseFloat> &state_occupancies,
    SpMatrix<BaseFloat> *H_sm,
    BaseFloat max_cond) const {
  int32 num_gauss = NumGauss();
  BaseFloat tot_sum = 0.0;
  KALDI_ASSERT(state_occupancies.Dim() == NumPdfs());
  Vector<BaseFloat> w_jm(num_gauss);
  H_sm->Resize(PhoneSpaceDim());
  H_sm->SetZero();
  Vector<BaseFloat> gamma_i;
  ComputeGammaI(state_occupancies, &gamma_i);

  BaseFloat sum = 0.0;
  for (int32 i = 0; i < num_gauss; i++) {
    if (gamma_i(i) > 0) {
      H_sm->AddSp(gamma_i(i), H[i]);
      sum += gamma_i(i);
    }
  }
  if (sum == 0.0) {
    KALDI_WARN << "Sum of counts is zero. ";
    // set to unit matrix--arbitrary non-singular matrix.. won't ever matter.
    H_sm->SetUnit();
  } else {
    H_sm->Scale(1.0 / sum);
    int32 tmp = H_sm->LimitCondDouble(max_cond);
    if (tmp > 0) {
      KALDI_WARN << "Limited " << (tmp) << " eigenvalues of H_sm";
    }
  }
  tot_sum += sum;

  KALDI_LOG << "total count is " << tot_sum;
}

void ComputeFeatureNormalizingTransform(const FullGmm &gmm, Matrix<BaseFloat> *xform) {
  int32 dim = gmm.Dim();
  int32 num_gauss = gmm.NumGauss();
  SpMatrix<BaseFloat> within_class_covar(dim);
  SpMatrix<BaseFloat> between_class_covar(dim);
  Vector<BaseFloat> global_mean(dim);

  // Accumulate LDA statistics from the GMM parameters.
  {
    BaseFloat total_weight = 0.0;
    Vector<BaseFloat> tmp_weight(num_gauss);
    Matrix<BaseFloat> tmp_means;
    std::vector< SpMatrix<BaseFloat> > tmp_covars;
    tmp_weight.CopyFromVec(gmm.weights());
    gmm.GetCovarsAndMeans(&tmp_covars, &tmp_means);
    for (int32 i = 0; i < num_gauss; i++) {
      BaseFloat w_i = tmp_weight(i);
      total_weight += w_i;
      within_class_covar.AddSp(w_i, tmp_covars[i]);
      between_class_covar.AddVec2(w_i, tmp_means.Row(i));
      global_mean.AddVec(w_i, tmp_means.Row(i));
    }
    KALDI_ASSERT(total_weight > 0);
    if (fabs(total_weight - 1.0) > 0.001) {
      KALDI_WARN << "Total weight across the GMMs is " << (total_weight)
          << ", renormalizing.";
      global_mean.Scale(1.0 / total_weight);
      within_class_covar.Scale(1.0 / total_weight);
      between_class_covar.Scale(1.0 / total_weight);
    }
    between_class_covar.AddVec2(-1.0, global_mean);
  }

  TpMatrix<BaseFloat> chol(dim);
  chol.Cholesky(within_class_covar);  // Sigma_W = L L^T
  TpMatrix<BaseFloat> chol_inv(chol);
  chol_inv.InvertDouble();
  Matrix<BaseFloat> chol_full(dim, dim);
  chol_full.CopyFromTp(chol_inv);
  SpMatrix<BaseFloat> LBL(dim);
  // LBL = L^{-1} \Sigma_B L^{-T}
  LBL.AddMat2Sp(1.0, chol_full, kNoTrans, between_class_covar, 0.0);
  Vector<BaseFloat> Dvec(dim);
  Matrix<BaseFloat> U(dim, dim);
  LBL.Eig(&Dvec, &U);
  SortSvd(&Dvec, &U);

  xform->Resize(dim, dim);
  chol_full.CopyFromTp(chol);
  // T := L U, eq (23)
  xform->AddMatMat(1.0, chol_full, kNoTrans, U, kNoTrans, 0.0);

#ifdef KALDI_PARANOID
  Matrix<BaseFloat> inv_xform(*xform);
  inv_xform.InvertDouble();
  {  // Check that T*within_class_covar*T' = I.
    Matrix<BaseFloat> wc_covar_full(dim, dim), tmp(dim, dim);
    wc_covar_full.CopyFromSp(within_class_covar);
    tmp.AddMatMat(1.0, inv_xform, kNoTrans, wc_covar_full, kNoTrans, 0.0);
    wc_covar_full.AddMatMat(1.0, tmp, kNoTrans, inv_xform, kTrans, 0.0);
    KALDI_ASSERT(wc_covar_full.IsUnit(0.01));
  }
  {  // Check that T*between_class_covar*T' = diagonal.
    Matrix<BaseFloat> bc_covar_full(dim, dim), tmp(dim, dim);
    bc_covar_full.CopyFromSp(between_class_covar);
    tmp.AddMatMat(1.0, inv_xform, kNoTrans, bc_covar_full, kNoTrans, 0.0);
    bc_covar_full.AddMatMat(1.0, tmp, kNoTrans, inv_xform, kTrans, 0.0);
    KALDI_ASSERT(bc_covar_full.IsDiagonal(0.01));
  }
#endif
}

void AmSgmm2::ComputePerSpkDerivedVars(Sgmm2PerSpkDerivedVars *vars) const {
  KALDI_ASSERT(vars != NULL);
  if (vars->v_s.Dim() != 0) {
    KALDI_ASSERT(vars->v_s.Dim() == SpkSpaceDim());
    vars->o_s.Resize(NumGauss(), FeatureDim());
    int32 num_gauss = NumGauss();
    // first compute the o_i^{(s)} quantities.
    for (int32 i = 0; i < num_gauss; i++) {
       // Eqn. (32): o_i^{(s)} = N_i v^{(s)}
      vars->o_s.Row(i).AddMatVec(1.0, N_[i], kNoTrans, vars->v_s, 0.0);
    }
    // the rest relates to the SSGMM.  We only need to to this
    // if we're using speaker-dependent weights.
    if (HasSpeakerDependentWeights()) {
      vars->log_d_jms.clear();
      vars->log_d_jms.resize(NumGroups());
      vars->log_b_is.Resize(NumGauss());
      vars->log_b_is.AddMatVec(1.0, u_, kNoTrans, vars->v_s, 0.0);
      vars->b_is.Resize(NumGauss());
      vars->b_is.CopyFromVec(vars->log_b_is);
      vars->b_is.ApplyExp();
      for (int32 i = 0; i < vars->b_is.Dim(); i++) {
        if (vars->b_is(i) - vars->b_is(i) != 0.0) { // NaN.
          vars->b_is(i) = 1.0;
          KALDI_WARN << "Set NaN in b_is to 1.0";
        }
      }
    } else {
      vars->b_is.Resize(0);
      vars->log_b_is.Resize(0);
      vars->log_d_jms.resize(0);
    }
  } else {
    vars->Clear(); // make sure everything is cleared.
  }
}

BaseFloat AmSgmm2::GaussianSelection(const Sgmm2GselectConfig &config,
                                    const VectorBase<BaseFloat> &data,
                                    std::vector<int32> *gselect) const {
  KALDI_ASSERT(diag_ubm_.NumGauss() != 0 &&
               diag_ubm_.NumGauss() == full_ubm_.NumGauss() &&
               diag_ubm_.Dim() == data.Dim());
  KALDI_ASSERT(config.diag_gmm_nbest > 0 && config.full_gmm_nbest > 0 &&
               config.full_gmm_nbest < config.diag_gmm_nbest);
  int32 num_gauss = diag_ubm_.NumGauss();

  std::vector< std::pair<BaseFloat, int32> > pruned_pairs;
  if (config.diag_gmm_nbest < num_gauss) {    Vector<BaseFloat> loglikes(num_gauss);
    diag_ubm_.LogLikelihoods(data, &loglikes);
    Vector<BaseFloat> loglikes_copy(loglikes);
    BaseFloat *ptr = loglikes_copy.Data();
    std::nth_element(ptr, ptr+num_gauss-config.diag_gmm_nbest, ptr+num_gauss);
    BaseFloat thresh = ptr[num_gauss-config.diag_gmm_nbest];
    for (int32 g = 0; g < num_gauss; g++)
      if (loglikes(g) >= thresh)  // met threshold for diagonal phase.
        pruned_pairs.push_back(
            std::make_pair(full_ubm_.ComponentLogLikelihood(data, g), g));
  } else {
    Vector<BaseFloat> loglikes(num_gauss);
    full_ubm_.LogLikelihoods(data, &loglikes);
    for (int32 g = 0; g < num_gauss; g++)
      pruned_pairs.push_back(std::make_pair(loglikes(g), g));
  }
  KALDI_ASSERT(!pruned_pairs.empty());
  if (pruned_pairs.size() > static_cast<size_t>(config.full_gmm_nbest)) {
    std::nth_element(pruned_pairs.begin(),
                     pruned_pairs.end() - config.full_gmm_nbest,
                     pruned_pairs.end());
    pruned_pairs.erase(pruned_pairs.begin(),
                       pruned_pairs.end() - config.full_gmm_nbest);
  }
  Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
  KALDI_ASSERT(gselect != NULL);
  gselect->resize(pruned_pairs.size());
  // Make sure pruned Gaussians appear from best to worst.
  std::sort(pruned_pairs.begin(), pruned_pairs.end(),
            std::greater< std::pair<BaseFloat, int32> >());
  for (size_t i = 0; i < pruned_pairs.size(); i++) {
    loglikes_tmp(i) = pruned_pairs[i].first;
    (*gselect)[i] = pruned_pairs[i].second;
  }
  return loglikes_tmp.LogSumExp();
}

void Sgmm2GauPost::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Sgmm2GauPost>");
  int32 T = this->size();
  WriteBasicType(os, binary, T);
  for (int32 t = 0; t < T; t++) {
    WriteToken(os, binary, "<gselect>");
    WriteIntegerVector(os, binary, (*this)[t].gselect);
    WriteToken(os, binary, "<tids>");
    WriteIntegerVector(os, binary, (*this)[t].tids);
    KALDI_ASSERT((*this)[t].tids.size() == (*this)[t].posteriors.size());
    for (size_t i = 0; i < (*this)[t].posteriors.size(); i++) {
      (*this)[t].posteriors[i].Write(os, binary);
    }
  }
  WriteToken(os, binary, "</Sgmm2GauPost>");
}


void Sgmm2GauPost::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Sgmm2GauPost>");
  int32 T;
  ReadBasicType(is, binary, &T);
  KALDI_ASSERT(T >= 0);
  this->resize(T);
  for (int32 t = 0; t < T; t++) {
    ExpectToken(is, binary, "<gselect>");
    ReadIntegerVector(is, binary, &((*this)[t].gselect));
    ExpectToken(is, binary, "<tids>");
    ReadIntegerVector(is, binary, &((*this)[t].tids));
    size_t sz = (*this)[t].tids.size();
    (*this)[t].posteriors.resize(sz);
    for (size_t i = 0; i < sz; i++)
      (*this)[t].posteriors[i].Read(is, binary);
  }
  ExpectToken(is, binary, "</Sgmm2GauPost>");
}



}  // namespace kaldi

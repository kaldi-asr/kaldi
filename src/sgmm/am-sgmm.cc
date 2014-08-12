// sgmm/am-sgmm.cc

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
#include <queue>
#include "sgmm/am-sgmm.h"
#include "thread/kaldi-thread.h"

namespace kaldi {
using std::vector;

// This function needs to be added because std::generate is complaining
// about RandGauss(), which takes an optional arguments.
static inline float _RandGauss()
{
  return RandGauss();
}

void AmSgmm::Read(std::istream &in_stream, bool binary) {
  int32 num_states, feat_dim, num_gauss;
  std::string token;

  ExpectToken(in_stream, binary, "<SGMM>");
  ExpectToken(in_stream, binary, "<NUMSTATES>");
  ReadBasicType(in_stream, binary, &num_states);
  ExpectToken(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &feat_dim);
  KALDI_ASSERT(num_states > 0 && feat_dim > 0);

  ReadToken(in_stream, binary, &token);

  while (token != "</SGMM>") {
    if (token == "<DIAG_UBM>") {
      diag_ubm_.Read(in_stream, binary);
    } else if (token == "<FULL_UBM>") {
      full_ubm_.Read(in_stream, binary);
    } else if (token == "<SigmaInv>") {
      ExpectToken(in_stream, binary, "<NUMGaussians>");
      ReadBasicType(in_stream, binary, &num_gauss);
      SigmaInv_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        SigmaInv_[i].Read(in_stream, binary);
      }
    } else if (token == "<M>") {
      ExpectToken(in_stream, binary, "<NUMGaussians>");
      ReadBasicType(in_stream, binary, &num_gauss);
      M_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        M_[i].Read(in_stream, binary);
      }
    } else if (token == "<N>") {
      ExpectToken(in_stream, binary, "<NUMGaussians>");
      ReadBasicType(in_stream, binary, &num_gauss);
      N_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) {
        N_[i].Read(in_stream, binary);
      }
    } else if (token == "<w>") {
      w_.Read(in_stream, binary);
    } else if (token == "<v>") {
      v_.resize(num_states);
      for (int32 j = 0; j < num_states; j++) {
        v_[j].Read(in_stream, binary);
      }
    } else if (token == "<c>") {
      c_.resize(num_states);
      for (int32 j = 0; j < num_states; j++) {
        c_[j].Read(in_stream, binary);
      }
    } else if (token == "<n>") {
      n_.resize(num_states);
      for (int32 j = 0; j < num_states; j++) {
        n_[j].Read(in_stream, binary);
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

  if (n_.empty()) {
    ComputeNormalizers();
  }
}

void AmSgmm::Write(std::ostream &out_stream, bool binary,
                   SgmmWriteFlagsType write_params) const {
  int32 num_states = NumPdfs(),
      feat_dim = FeatureDim(),
      num_gauss = NumGauss();

  WriteToken(out_stream, binary, "<SGMM>");
  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<NUMSTATES>");
  WriteBasicType(out_stream, binary, num_states);
  WriteToken(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, feat_dim);
  if (!binary) out_stream << "\n";

  if (write_params & kSgmmBackgroundGmms) {
    WriteToken(out_stream, binary, "<DIAG_UBM>");
    diag_ubm_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<FULL_UBM>");
    full_ubm_.Write(out_stream, binary);
  }

  if (write_params & kSgmmGlobalParams) {
    WriteToken(out_stream, binary, "<SigmaInv>");
    WriteToken(out_stream, binary, "<NUMGaussians>");
    WriteBasicType(out_stream, binary, num_gauss);
    if (!binary) out_stream << "\n";
    for (int32 i = 0; i < num_gauss; i++) {
      SigmaInv_[i].Write(out_stream, binary);
    }
    WriteToken(out_stream, binary, "<M>");
    WriteToken(out_stream, binary, "<NUMGaussians>");
    WriteBasicType(out_stream, binary, num_gauss);
    if (!binary) out_stream << "\n";
    for (int32 i = 0; i < num_gauss; i++) {
      M_[i].Write(out_stream, binary);
    }
    if (N_.size() != 0) {
      WriteToken(out_stream, binary, "<N>");
      WriteToken(out_stream, binary, "<NUMGaussians>");
      WriteBasicType(out_stream, binary, num_gauss);
      if (!binary) out_stream << "\n";
      for (int32 i = 0; i < num_gauss; i++) {
        N_[i].Write(out_stream, binary);
      }
    }
    WriteToken(out_stream, binary, "<w>");
    w_.Write(out_stream, binary);

    // The following are the Gaussian prior parameters for MAP adaptation of M.
    // They may be moved to somewhere else eventually.
    if (M_prior_.size() != 0) {
      WriteToken(out_stream, binary, "<M_Prior>");
      WriteToken(out_stream, binary, "<NUMGaussians>");
      WriteBasicType(out_stream, binary, num_gauss);
      if (!binary) out_stream << "\n";
      for (int32 i = 0; i < num_gauss; i++) {
        M_prior_[i].Write(out_stream, binary);
      }

      KALDI_ASSERT(row_cov_inv_.NumRows() != 0 &&
                   "Empty row covariance for MAP prior");
      WriteToken(out_stream, binary, "<Row_Cov_Inv>");
      if (!binary) out_stream << "\n";
      row_cov_inv_.Write(out_stream, binary);

      KALDI_ASSERT(col_cov_inv_.NumRows() != 0 &&
                   "Empty column covariance for MAP prior");
      WriteToken(out_stream, binary, "<Col_Cov_Inv>");
      if (!binary) out_stream << "\n";
      col_cov_inv_.Write(out_stream, binary);
    }
    // end priors for MAP adaptation
  }

  if (write_params & kSgmmStateParams) {
    WriteToken(out_stream, binary, "<v>");
    for (int32 j = 0; j < num_states; j++) {
      v_[j].Write(out_stream, binary);
    }
    WriteToken(out_stream, binary, "<c>");
    for (int32 j = 0; j < num_states; j++) {
      c_[j].Write(out_stream, binary);
    }
  }

  if (write_params & kSgmmNormalizers) {
    WriteToken(out_stream, binary, "<n>");
    if (n_.empty())
      KALDI_WARN << "Not writing normalizers since they are not present.";
    else
      for (int32 j = 0; j < num_states; j++)
        n_[j].Write(out_stream, binary);
  }

  WriteToken(out_stream, binary, "</SGMM>");
}

void AmSgmm::Check(bool show_properties) {
  int32 num_states = NumPdfs(),
      num_gauss = NumGauss(),
      feat_dim = FeatureDim(),
      phn_dim = PhoneSpaceDim(),
      spk_dim = SpkSpaceDim();

  if (show_properties)
    KALDI_LOG << "AmSgmm: #states = " << num_states << ", #Gaussians = "
              << num_gauss << ", feature dim = " << feat_dim
              << ", phone-space dim =" << phn_dim
              << ", speaker-space dim =" << spk_dim;
  KALDI_ASSERT(num_states > 0 && num_gauss > 0 && feat_dim > 0 && phn_dim > 0);

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

  KALDI_ASSERT(M_.size() == static_cast<size_t>(num_gauss));
  for (int32 i = 0; i < num_gauss; i++) {
    KALDI_ASSERT(M_[i].NumRows() == feat_dim && M_[i].NumCols() == phn_dim);
  }

  KALDI_ASSERT(w_.NumRows() == num_gauss && w_.NumCols() == phn_dim);

  {  // check v, c.
    KALDI_ASSERT(v_.size() == static_cast<size_t>(num_states) &&
                 c_.size() == static_cast<size_t>(num_states));
    int32 nSubstatesTot = 0;
    for (int32 j = 0; j < num_states; j++) {
      int32 M_j = NumSubstates(j);
      nSubstatesTot += M_j;
      KALDI_ASSERT(M_j > 0 && v_[j].NumRows() == M_j &&
          c_[j].Dim() == M_j && v_[j].NumCols() == phn_dim);
    }
    debug_str << "Substates: "<< (nSubstatesTot) << ".  ";
  }

  // check n.
  if (n_.size() == 0) {
    debug_str << "Normalizers: no.  ";
  } else {
    debug_str << "Normalizers: yes.  ";
    KALDI_ASSERT(n_.size() == static_cast<size_t>(num_states));
    for (int32 j = 0; j < num_states; j++) {
      KALDI_ASSERT(n_[j].NumRows() == num_gauss &&
          n_[j].NumCols() == NumSubstates(j));
    }
  }

  if (show_properties)
    KALDI_LOG << "Subspace GMM model properties: " << debug_str.str();
}

void AmSgmm::InitializeFromFullGmm(const FullGmm &full_gmm,
                                   int32 num_states,
                                   int32 phn_subspace_dim,
                                   int32 spk_subspace_dim) {
  full_ubm_.CopyFromFullGmm(full_gmm);
  diag_ubm_.CopyFromFullGmm(full_gmm);
  if (phn_subspace_dim < 1 || phn_subspace_dim > full_gmm.Dim() + 1) {
    KALDI_WARN << "Initial phone-subspace dimension must be in [1, "
               << full_gmm.Dim() + 1 << "]. Changing from " << phn_subspace_dim
               << " to " << full_gmm.Dim() + 1;
    phn_subspace_dim = full_gmm.Dim() + 1;
  }
  if (spk_subspace_dim < 0 || spk_subspace_dim > full_gmm.Dim()) {
    KALDI_WARN << "Initial spk-subspace dimension must be in [1, "
               << full_gmm.Dim() << "]. Changing from " << spk_subspace_dim
               << " to " << full_gmm.Dim();
    spk_subspace_dim = full_gmm.Dim();
  }
  w_.Resize(0, 0);
  N_.clear();
  c_.clear();
  v_.clear();
  SigmaInv_.clear();

  KALDI_LOG << "Initializing model";
  Matrix<BaseFloat> norm_xform;
  ComputeFeatureNormalizer(full_gmm, &norm_xform);
  InitializeMw(phn_subspace_dim, norm_xform);
  if (spk_subspace_dim > 0) InitializeN(spk_subspace_dim, norm_xform);
  InitializeVecs(num_states);
  KALDI_LOG << "Initializing variances";
  InitializeCovars();
}

void AmSgmm::CopyFromSgmm(const AmSgmm &other,
                          bool copy_normalizers) {
  KALDI_LOG << "Copying AmSgmm";

  // Copy background GMMs
  diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
  full_ubm_.CopyFromFullGmm(other.full_ubm_);

  // Copy global params
  SigmaInv_ = other.SigmaInv_;
  M_ = other.M_;
  w_ = other.w_;
  N_ = other.N_;

  // Copy state-specific params, but only copy normalizers if requested.
  v_ = other.v_;
  c_ = other.c_;
  if (copy_normalizers) n_ = other.n_;

  KALDI_LOG << "Done.";
}

void AmSgmm::CopyGlobalsInitVecs(const AmSgmm &other,
                                 int32 phn_subspace_dim,
                                 int32 spk_subspace_dim,
                                 int32 num_pdfs) {
  if (phn_subspace_dim < 1 || phn_subspace_dim > other.PhoneSpaceDim()) {
    KALDI_WARN << "Initial phone-subspace dimension must be in [1, "
        << other.PhoneSpaceDim() << "]. Changing from " << phn_subspace_dim
        << " to " << other.PhoneSpaceDim();
    phn_subspace_dim = other.PhoneSpaceDim();
  }
  if (spk_subspace_dim < 0 || spk_subspace_dim > other.SpkSpaceDim()) {
    KALDI_WARN << "Initial spk-subspace dimension must be in [1, "
               << other.SpkSpaceDim() << "]. Changing from " << spk_subspace_dim
               << " to " << other.SpkSpaceDim();
    spk_subspace_dim = other.SpkSpaceDim();
  }

  KALDI_LOG << "Initializing model";

  // Copy background GMMs
  diag_ubm_.CopyFromDiagGmm(other.diag_ubm_);
  full_ubm_.CopyFromFullGmm(other.full_ubm_);

  // Copy global params
  SigmaInv_ = other.SigmaInv_;
  int32 num_gauss = diag_ubm_.NumGauss(),
      data_dim = other.FeatureDim();
  M_.resize(num_gauss);
  w_.Resize(num_gauss, phn_subspace_dim);
  for (int32 i = 0; i < num_gauss; i++) {
    M_[i].Resize(data_dim, phn_subspace_dim);
    M_[i].CopyFromMat(other.M_[i].Range(0, data_dim, 0, phn_subspace_dim),
                      kNoTrans);
  }
  w_.CopyFromMat(other.w_.Range(0, num_gauss, 0, phn_subspace_dim), kNoTrans);

  if (spk_subspace_dim > 0) {
    N_.resize(num_gauss);
    for (int32 i = 0; i < num_gauss; i++) {
      N_[i].Resize(data_dim, spk_subspace_dim);
      N_[i].CopyFromMat(other.N_[i].Range(0, data_dim, 0, spk_subspace_dim),
                        kNoTrans);
    }
  } else {
    N_.clear();
  }
  InitializeVecs(num_pdfs);
}


void AmSgmm::ComputePerFrameVars(const VectorBase<BaseFloat> &data,
                                 const std::vector<int32> &gselect,
                                 const SgmmPerSpkDerivedVars &spk_vars,
                                 BaseFloat logdet_s,
                                 SgmmPerFrameDerivedVars *per_frame_vars) const {
  KALDI_ASSERT(!n_.empty() && "ComputeNormalizers() must be called.");

  if (per_frame_vars->NeedsResizing(gselect.size(),
                                    FeatureDim(),
                                    PhoneSpaceDim()))
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
  for (int32 ki = 0, last = gselect.size(); ki < last; ki++) {
    int32 i = gselect[ki];
    SigmaInv_xt.AddSpVec(1.0, SigmaInv_[i], per_frame_vars->xti.Row(ki), 0.0);
    // Eq (35): z_{i}(t) = M_{i}^{T} \Sigma_{i}^{-1} x_{i}(t)
    per_frame_vars->zti.Row(ki).AddMatVec(1.0, M_[i], kTrans, SigmaInv_xt, 0.0);
    // Eq.(36): n_{i}(t) = -0.5 x_{i}^{T} \Sigma_{i}^{-1} x_{i}(t)
    per_frame_vars->nti(ki) = -0.5 * VecVec(per_frame_vars->xti.Row(ki),
                                            SigmaInv_xt) + logdet_s;
  }
}

BaseFloat AmSgmm::LogLikelihood(const SgmmPerFrameDerivedVars &per_frame_vars,
                                int32 j, BaseFloat log_prune) const {
  KALDI_ASSERT(j < NumPdfs());
  const vector<int32> &gselect = per_frame_vars.gselect;


  // Eq.(37): log p(x(t), m, i|j)  [indexed by j, ki]
  // Although the extra memory allocation of storing this as a
  // matrix might seem unnecessary, we save time in the LogSumExp()
  // via more effective pruning.
  Matrix<BaseFloat> logp_x(gselect.size(), NumSubstates(j));

  for (int32 ki = 0, last = gselect.size();  ki < last; ki++) {
    SubVector<BaseFloat> logp_xi(logp_x, ki);
    int32 i = gselect[ki];
    // for all substates, compute z_{i}^T v_{jm}
    logp_xi.AddMatVec(1.0, v_[j], kNoTrans, per_frame_vars.zti.Row(ki), 0.0);
    logp_xi.AddVec(1.0, n_[j].Row(i));  // for all substates, add n_{jim}
    logp_xi.Add(per_frame_vars.nti(ki));  // for all substates, add n_{i}(t)
  }
  // Eq. (38): log p(x(t)|j) = log \sum_{m, i} p(x(t), m, i|j)
  return logp_x.LogSumExp(log_prune);
}

BaseFloat
AmSgmm::ComponentPosteriors(const SgmmPerFrameDerivedVars &per_frame_vars,
                            int32 j,
                            Matrix<BaseFloat> *post) const {
  KALDI_ASSERT(j < NumPdfs());
  if (post == NULL) KALDI_ERR << "NULL pointer passed as return argument.";
  const vector<int32> &gselect = per_frame_vars.gselect;
  int32 num_gselect = gselect.size();
  post->Resize(num_gselect, NumSubstates(j));

  // Eq.(37): log p(x(t), m, i|j) = z_{i}^T v_{jm} (for all substates)
  post->AddMatMat(1.0, per_frame_vars.zti, kNoTrans, v_[j], kTrans, 0.0);
  for (int32 ki = 0; ki < num_gselect; ki++) {
    int32 i = gselect[ki];
    // Eq. (37): log p(x(t), m, i|j) += n_{jim} + n_{i}(t) (for all substates)
    post->Row(ki).AddVec(1.0, n_[j].Row(i));
    post->Row(ki).Add(per_frame_vars.nti(ki));
  }

  // Eq. (38): log p(x(t)|j) = log \sum_{m, i} p(x(t), m, i|j)
  return post->ApplySoftMax();
}

struct SubstateCounter {
  SubstateCounter(int32 j, int32 num_substates, BaseFloat occ)
      : state_index(j), num_substates(num_substates), occupancy(occ) {}

  int32 state_index;
  int32 num_substates;
  BaseFloat occupancy;

  bool operator < (const SubstateCounter &r) const {
    return occupancy/num_substates < r.occupancy/r.num_substates;
  }
};

void AmSgmm::SplitSubstates(const Vector<BaseFloat> &state_occupancies,
                                   int32 target_nsubstates, BaseFloat perturb,
                                   BaseFloat power, BaseFloat max_cond) {
  // power == p in document.  target_nsubstates == T in document.
  KALDI_ASSERT(state_occupancies.Dim() == NumPdfs());
  int32 tot_n_substates_old = 0;
  int32 phn_dim = PhoneSpaceDim();
  std::priority_queue<SubstateCounter> substate_counts;
  vector< SpMatrix<BaseFloat> > H_i;
  SpMatrix<BaseFloat> sqrt_H_sm;
  Vector<BaseFloat> rand_vec(phn_dim), v_shift(phn_dim);

  for (int32 j = 0; j < NumPdfs(); j++) {
    BaseFloat gamma_p = pow(state_occupancies(j), power);
    substate_counts.push(SubstateCounter(j, NumSubstates(j), gamma_p));
    tot_n_substates_old += NumSubstates(j);
  }
  if (target_nsubstates <= tot_n_substates_old || tot_n_substates_old == 0) {
    KALDI_WARN << "Cannot split from " << (tot_n_substates_old) <<
        " to " << (target_nsubstates) << " substates.";
    return;
  }

  ComputeH(&H_i);  // set up that array.
  ComputeSmoothingTermsFromModel(H_i, state_occupancies, &sqrt_H_sm, max_cond);
  H_i.clear();
  sqrt_H_sm.ApplyPow(-0.5);

  for (int32 n_states = tot_n_substates_old;
       n_states < target_nsubstates; n_states++) {
    SubstateCounter state_to_split = substate_counts.top();
    substate_counts.pop();
    state_to_split.num_substates++;
    substate_counts.push(state_to_split);
  }

  while (!substate_counts.empty()) {
    int32 j = substate_counts.top().state_index;
    int32 tgt_n_substates_j = substate_counts.top().num_substates;
    int32 n_substates_j     = NumSubstates(j);
    substate_counts.pop();

    if (n_substates_j == tgt_n_substates_j) continue;

    // Resize v[j] and c[j] to fit new substates
    Matrix<BaseFloat> tmp_v_j(v_[j]);
    v_[j].Resize(tgt_n_substates_j, phn_dim);
    v_[j].Range(0, n_substates_j, 0, phn_dim).CopyFromMat(tmp_v_j);
    tmp_v_j.Resize(0, 0);

    Vector<BaseFloat> tmp_c_j(c_[j]);
    c_[j].Resize(tgt_n_substates_j);
    c_[j].Range(0, n_substates_j).CopyFromVec(tmp_c_j);
    tmp_c_j.Resize(0);

    // Keep splitting substates until obtaining the desired number
    for (; n_substates_j < tgt_n_substates_j; n_substates_j++) {
      int32 split_substate = std::max_element(c_[j].Data(), c_[j].Data()
          + n_substates_j) - c_[j].Data();

      // c_{jkm} := c_{jmk}' := c_{jkm} / 2
      c_[j](split_substate) = c_[j](n_substates_j) = c_[j](split_substate) / 2;

      // v_{jkm} := +/- split_perturb * H_k^{(sm)}^{-0.5} * rand_vec
      std::generate(rand_vec.Data(), rand_vec.Data() + rand_vec.Dim(),
                    _RandGauss);
      v_shift.AddSpVec(perturb, sqrt_H_sm, rand_vec, 0.0);
      v_[j].Row(n_substates_j).CopyFromVec(v_[j].Row(split_substate));
      v_[j].Row(n_substates_j).AddVec(1.0, v_shift);
      v_[j].Row(split_substate).AddVec((-1.0), v_shift);
    }
  }
  KALDI_LOG << "Getting rid of normalizers as they will no longer be valid";

  n_.clear();
  KALDI_LOG << "Split " << (tot_n_substates_old) << " substates to "
      << (target_nsubstates);
}

void AmSgmm::IncreasePhoneSpaceDim(int32 target_dim,
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

    for (int32 j = 0; j < NumPdfs(); j++) {
      // Resize v[j]
      Matrix<BaseFloat> tmp_v_j = v_[j];
      v_[j].Resize(tmp_v_j.NumRows(), target_dim);
      v_[j].Range(0, tmp_v_j.NumRows(), 0, tmp_v_j.NumCols()).CopyFromMat(
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

void AmSgmm::IncreaseSpkSpaceDim(int32 target_dim,
                                 const Matrix<BaseFloat> &norm_xform) {
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
    KALDI_LOG << "Speaker subspace dimensionality increased from " <<
        initial_dim << " to " << target_dim;
  } else {
    KALDI_LOG << "Speaker subspace dimensionality unchanged, since target " <<
        "dimension (" << target_dim << ") <= initial dimansion (" <<
        initial_dim << ")";
  }
}

void AmSgmm::ComputeDerivedVars() {
  if (n_.empty()) {
    ComputeNormalizers();
  }
  if (diag_ubm_.NumGauss() != full_ubm_.NumGauss()
      || diag_ubm_.Dim() != full_ubm_.Dim()) {
    diag_ubm_.CopyFromFullGmm(full_ubm_);
  }
}

class ComputeNormalizersClass: public MultiThreadable { // For multi-threaded.
 public:
  ComputeNormalizersClass(AmSgmm *am_sgmm,
                          int32 *entropy_count_ptr,
                          double *entropy_sum_ptr):
      am_sgmm_(am_sgmm), entropy_count_ptr_(entropy_count_ptr),
      entropy_sum_ptr_(entropy_sum_ptr), entropy_count_(0),
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
  AmSgmm *am_sgmm_;
  int32 *entropy_count_ptr_;
  double *entropy_sum_ptr_;
  int32 entropy_count_;
  double entropy_sum_;

};

void AmSgmm::ComputeNormalizers() {
  KALDI_LOG << "Computing normalizers";
  n_.resize(NumPdfs());
  int32 entropy_count = 0;
  double entropy_sum = 0.0;
  ComputeNormalizersClass c(this, &entropy_count, &entropy_sum);
  RunMultiThreaded(c);

  KALDI_LOG << "Entropy of weights in substates is "
            << (entropy_sum / entropy_count) << " over " << entropy_count
            << " substates, equivalent to perplexity of "
            << (exp(entropy_sum /entropy_count));
  KALDI_LOG << "Done computing normalizers";
}


void AmSgmm::ComputeNormalizersInternal(int32 num_threads, int32 thread,
                                        int32 *entropy_count,
                                        double *entropy_sum) {

  BaseFloat DLog2pi = FeatureDim() * log(2 * M_PI);
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


  int block_size = (NumPdfs() + num_threads-1) / num_threads;
  int j_start = thread * block_size, j_end = std::min(NumPdfs(), j_start + block_size);
  
  for (int32 j = j_start; j < j_end; j++) {
    Matrix<BaseFloat> log_w_jm(NumSubstates(j), NumGauss());
    n_[j].Resize(NumGauss(), NumSubstates(j));
    Matrix<BaseFloat> mu_jmi(NumSubstates(j), FeatureDim());
    Matrix<BaseFloat> SigmaInv_mu(NumSubstates(j), FeatureDim());
        
    // (in logs): w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm}) eq.(7)
    log_w_jm.AddMatMat(1.0, v_[j], kNoTrans, w_, kTrans, 0.0);
    for (int32 m = 0; m < NumSubstates(j); m++) {
      log_w_jm.Row(m).Add(-1.0 * log_w_jm.Row(m).LogSumExp());    
      {  // DIAGNOSTIC CODE
        (*entropy_count)++;
        for (int32 i = 0; i < NumGauss(); i++) {
          (*entropy_sum) -= log_w_jm(m, i) * exp(log_w_jm(m, i));
        }
      }
    }      
    
    for (int32 i = 0; i < NumGauss(); i++) {    
      // mu_jmi = M_{i} * v_{jm}
      mu_jmi.AddMatMat(1.0, v_[j], kNoTrans, M_[i], kTrans, 0.0);
      SigmaInv_mu.AddMatSp(1.0, mu_jmi, kNoTrans, SigmaInv_[i], 0.0);
    
      for (int32 m = 0; m < NumSubstates(j); m++) {
        // mu_{jmi} * \Sigma_{i}^{-1} * mu_{jmi}
        BaseFloat mu_SigmaInv_mu = VecVec(mu_jmi.Row(m), SigmaInv_mu.Row(m));
        BaseFloat logc = log(c_[j](m));

        // Suggestion: Both mu_jmi and SigmaInv_mu could
        // have been computed at once for i,
        // if M[i] was concatenated to single matrix over i indices
        
        // eq.(31)
        n_[j](i, m) = logc + log_w_jm(m, i) - 0.5 * (log_det_Sigma(i) + DLog2pi
            + mu_SigmaInv_mu);
        {  // Mainly diagnostic code.  Not necessary.
          BaseFloat tmp = n_[j](i, m);
          if (!KALDI_ISFINITE(tmp)) {  // NaN or inf
            KALDI_LOG << "Warning: normalizer for j = " << j << ", m = " << m
                      << ", i = " << i << " is infinite or NaN " << tmp << "= "
                      << (logc) << "+" << (log_w_jm(m, i)) << "+" << (-0.5 *
                          log_det_Sigma(i)) << "+" << (-0.5 * DLog2pi)
                      << "+" << (mu_SigmaInv_mu) << ", setting to finite.";
            n_[j](i, m) = -1.0e+40;  // future work(arnab): get rid of magic number
          }
        }
      }
    }
  }
}


void AmSgmm::ComputeNormalizersNormalized(
    const std::vector< std::vector<int32> > &normalize_sets) {
  { // Check sets in normalize_sets are disjoint and cover all Gaussians.
    std::set<int32> all;
    for (int32 i = 0; i < normalize_sets.size(); i++)
      for (int32 j = 0; static_cast<size_t>(j) < normalize_sets[i].size(); j++) {
        int32 n = normalize_sets[i][j];
        KALDI_ASSERT(all.count(n) == 0 && n >= 0 && n < NumGauss());
        all.insert(n);
      }
    KALDI_ASSERT(all.size() == NumGauss());
  }

  KALDI_LOG << "Computing normalizers [normalized]";
  BaseFloat DLog2pi = FeatureDim() * log(2 * M_PI);
  Vector<BaseFloat> mu_jmi(FeatureDim());
  Vector<BaseFloat> SigmaInv_mu(FeatureDim());
  Vector<BaseFloat> log_det_Sigma(NumGauss());

  for (int32 i = 0; i < NumGauss(); i++) {
    try {
      log_det_Sigma(i) = - SigmaInv_[i].LogPosDefDet();
    } catch(...) {
      KALDI_WARN << "Covariance is not positive definite, setting to unit";
      SigmaInv_[i].SetUnit();
      log_det_Sigma(i) = 0.0;
    }
  }

  n_.resize(NumPdfs());
  for (int32 j = 0; j < NumPdfs(); j++) {
    Vector<BaseFloat> log_w_jm(NumGauss());

    n_[j].Resize(NumGauss(), NumSubstates(j));
    for (int32 m = 0; m < NumSubstates(j); m++) {
      BaseFloat logc = log(c_[j](m));

      // (in logs): w_jm = softmax([w_{k1}^T ... w_{kD}^T] * v_{jkm}) eq.(7)
      log_w_jm.AddMatVec(1.0, w_, kNoTrans, v_[j].Row(m), 0.0);
      log_w_jm.Add((-1.0) * log_w_jm.LogSumExp());

      for (int32 n = 0; n < normalize_sets.size(); n++) {
        const std::vector<int32> &this_set(normalize_sets[n]);
        double sum = 0.0;
        for (int32 p = 0; p < this_set.size(); p++)
          sum += exp(log_w_jm(this_set[p]));
        double offset = -log(sum);  // add "offset", to normalize weights.
        for (int32 p = 0; p < this_set.size(); p++)
          log_w_jm(this_set[p]) += offset;
      }

      for (int32 i = 0; i < NumGauss(); i++) {
        // mu_jmi = M_{i} * v_{jm}
        mu_jmi.AddMatVec(1.0, M_[i], kNoTrans, v_[j].Row(m), 0.0);

        // mu_{jmi} * \Sigma_{i}^{-1} * mu_{jmi}
        SigmaInv_mu.AddSpVec(1.0, SigmaInv_[i], mu_jmi, 0.0);
        BaseFloat mu_SigmaInv_mu = VecVec(mu_jmi, SigmaInv_mu);

        // Suggestion: Both mu_jmi and SigmaInv_mu could
        // have been computed at once  for i ,
        // if M[i] was concatenated to single matrix over i indeces

        // eq.(31)
        n_[j](i, m) = logc + log_w_jm(i) - 0.5 * (log_det_Sigma(i) + DLog2pi
            + mu_SigmaInv_mu);
        {  // Mainly diagnostic code.  Not necessary.
          BaseFloat tmp = n_[j](i, m);
          if (!KALDI_ISFINITE(tmp)) {  // NaN or inf
            KALDI_LOG << "Warning: normalizer for j = " << j << ", m = " << m
                      << ", i = " << i << " is infinite or NaN " << tmp << "= "
                      << (logc) << "+" << (log_w_jm(i)) << "+" << (-0.5 *
                          log_det_Sigma(i)) << "+" << (-0.5 * DLog2pi)
                      << "+" << (mu_SigmaInv_mu) << ", setting to finite.";
            n_[j](i, m) = -1.0e+40;  // future work(arnab): get rid of magic number
          }
        }
      }
    }
  }

  KALDI_LOG << "Done computing normalizers (normalized over subsets)";
}


void AmSgmm::ComputeFmllrPreXform(const Vector<BaseFloat> &state_occs,
    Matrix<BaseFloat> *xform, Matrix<BaseFloat> *inv_xform,
    Vector<BaseFloat> *diag_mean_scatter) const {
  int32 num_states = NumPdfs(),
      num_gauss = NumGauss(),
      dim = FeatureDim();
  KALDI_ASSERT(state_occs.Dim() == num_states);

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
  BaseFloat substate_weight;
  for (int32 j = 0; j < num_states; j++) {
    for (int32 m = 0; m < NumSubstates(j); m++) {
      // Eq. (7): w_jm = softmax([w_{1}^T ... w_{D}^T] * v_{jm})
      w_jm.AddMatVec(1.0, w_, kNoTrans, v_[j].Row(m), 0.0);
      w_jm.ApplySoftMax();

      for (int32 i = 0; i < num_gauss; i++) {
        substate_weight = state_posteriors(j) * c_[j](m) * w_jm(i);
        mu_jmi.AddMatVec(1.0, M_[i], kNoTrans, v_[j].Row(m), 0.0);  // Eq. (6)
        // Eq. (B.3): \mu_avg = \sum_{jmi} p(j) c_{jm} w_{jmi} \mu_{jmi}
        global_mean.AddVec(substate_weight, mu_jmi);
        // \Sigma_B = \sum_{jmi} p(j) c_{jm} w_{jmi} \mu_{jmi} \mu_{jmi}^T
        between_class_covar.AddVec2(substate_weight, mu_jmi);  // Eq. (B.4)
        gauss_weight(i) += substate_weight;
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
  if ((n = diag_mean_scatter->ApplyFloor(1.0e-04)) != 0)
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
void AmSgmm::GetNtransSigmaInv(vector< Matrix<Real> > *out) const {
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
void AmSgmm::GetNtransSigmaInv(vector< Matrix<float> > *out) const;
template
void AmSgmm::GetNtransSigmaInv(vector< Matrix<double> > *out) const;

///////////////////////////////////////////////////////////////////////////////

template<class Real>
void AmSgmm::ComputeH(std::vector< SpMatrix<Real> > *H_i) const {
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
void AmSgmm::ComputeH(std::vector< SpMatrix<float> > *H_i) const;
template
void AmSgmm::ComputeH(std::vector< SpMatrix<double> > *H_i) const;


// Initializes the matrices M_{i} and w_i
void AmSgmm::InitializeMw(int32 phn_subspace_dim,
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
    thisM.Range(0, ddim, 1, phn_subspace_dim-1).CopyFromMat(
        norm_xform.Range(0, ddim, 0, phn_subspace_dim-1), kNoTrans);
  }
}

// Initializes the matrices N_{i}
void AmSgmm::InitializeN(int32 spk_subspace_dim,
                         const Matrix<BaseFloat> &norm_xform) {
  int32 ddim = full_ubm_.Dim();
  KALDI_ASSERT(spk_subspace_dim <= ddim);
  KALDI_ASSERT(spk_subspace_dim <= norm_xform.NumCols());
  KALDI_ASSERT(ddim <= norm_xform.NumRows());

  int32 num_gauss = full_ubm_.NumGauss();
  N_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    N_[i].Resize(ddim, spk_subspace_dim);
    // Eq. (28): N_{i} = [ (J)_{1:D, 1:T)}]
    N_[i].CopyFromMat(norm_xform.Range(0, ddim, 0, spk_subspace_dim), kNoTrans);
  }
}

// Initializes the vectors v_{jm}
void AmSgmm::InitializeVecs(int32 num_states) {
  KALDI_ASSERT(num_states >= 0);
  int32 phn_subspace_dim = PhoneSpaceDim();
  KALDI_ASSERT(phn_subspace_dim > 0 && "Initialize M and w first.");

  v_.resize(num_states);
  c_.resize(num_states);
  for (int32 j = 0; j < num_states; j++) {
    v_[j].Resize(1, phn_subspace_dim);
    c_[j].Resize(1);
    v_[j](0, 0) = 1.0;  // Eq. (26): v_{j1} = [1 0 0 ... 0]
    c_[j](0) = 1.0;     // Eq. (25): c_{j1} = 1.0
  }
}

// Initializes the within-class vars Sigma_{ki}
void AmSgmm::InitializeCovars() {
  std::vector< SpMatrix<BaseFloat> > &inv_covars(full_ubm_.inv_covars());
  int32 num_gauss = full_ubm_.NumGauss();
  int32 dim = full_ubm_.Dim();
  SigmaInv_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    SigmaInv_[i].Resize(dim);
    SigmaInv_[i].CopyFromSp(inv_covars[i]);
  }
}

// Compute the "smoothing" matrices from expected counts given the model.
void AmSgmm::ComputeSmoothingTermsFromModel(
    const std::vector< SpMatrix<BaseFloat> > &H,
    const Vector<BaseFloat> &state_occupancies, SpMatrix<BaseFloat> *H_sm,
    BaseFloat max_cond) const {
  int32 num_gauss = NumGauss();
  BaseFloat tot_sum = 0.0;
  KALDI_ASSERT(state_occupancies.Dim() == NumPdfs());
  Vector<BaseFloat> w_jm(num_gauss);
  H_sm->Resize(PhoneSpaceDim());
  H_sm->SetZero();
  Vector<BaseFloat> gamma_i(num_gauss);
  gamma_i.SetZero();
  for (int32 j = 0; j < NumPdfs(); j++) {
    int32 M_j = NumSubstates(j);
    KALDI_ASSERT(M_j > 0);
    for (int32 m = 0; m < M_j; m++) {
      w_jm.AddMatVec(1.0, w_, kNoTrans, v_[j].Row(m), 0.0);
      w_jm.ApplySoftMax();
      gamma_i.AddVec(state_occupancies(j) * c_[j](m), w_jm);
    }
  }
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

  KALDI_LOG << "ComputeSmoothingTermsFromModel: total count is " << tot_sum;
}

void ComputeFeatureNormalizer(const FullGmm &gmm, Matrix<BaseFloat> *xform) {
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

void AmSgmm::ComputePerSpkDerivedVars(SgmmPerSpkDerivedVars *vars) const {
  KALDI_ASSERT(vars != NULL);
  if (vars->v_s.Dim() != 0) {
    KALDI_ASSERT(vars->v_s.Dim() == SpkSpaceDim());
    vars->o_s.Resize(NumGauss(), FeatureDim());
    int32 num_gauss = NumGauss();
    for (int32 i = 0; i < num_gauss; i++) {
      // Eqn. (32): o_i^{(s)} = N_i v^{(s)}
      vars->o_s.Row(i).AddMatVec(1.0, N_[i], kNoTrans, vars->v_s, 0.0);
    }
  } else {
    vars->o_s.Resize(0, 0);
  }
}

BaseFloat AmSgmm::GaussianSelection(const SgmmGselectConfig &config,
                                    const VectorBase<BaseFloat> &data,
                                    std::vector<int32> *gselect) const {
  KALDI_ASSERT(diag_ubm_.NumGauss() != 0 &&
               diag_ubm_.NumGauss() == full_ubm_.NumGauss() &&
               diag_ubm_.Dim() == data.Dim());
  KALDI_ASSERT(config.diag_gmm_nbest > 0 && config.full_gmm_nbest > 0 &&
               config.full_gmm_nbest < config.diag_gmm_nbest);
  int32 num_gauss = diag_ubm_.NumGauss();

  std::vector< std::pair<BaseFloat, int32> > pruned_pairs;
  if (config.diag_gmm_nbest < num_gauss) {
    Vector<BaseFloat> loglikes(num_gauss);
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

BaseFloat AmSgmm::GaussianSelectionPreselect(const SgmmGselectConfig &config,
                                             const VectorBase<BaseFloat> &data,
                                             const std::vector<int32> &preselect,
                                             std::vector<int32> *gselect) const {
  KALDI_ASSERT(IsSortedAndUniq(preselect) && !preselect.empty());
  KALDI_ASSERT(diag_ubm_.NumGauss() != 0 &&
               diag_ubm_.NumGauss() == full_ubm_.NumGauss() &&
               diag_ubm_.Dim() == data.Dim());

  int32 num_preselect = preselect.size();

  KALDI_ASSERT(config.diag_gmm_nbest > 0 && config.full_gmm_nbest > 0 &&
               config.full_gmm_nbest < num_preselect);

  std::vector<std::pair<BaseFloat, int32> > pruned_pairs;
  if (config.diag_gmm_nbest < num_preselect) {
    Vector<BaseFloat> loglikes(num_preselect);
    diag_ubm_.LogLikelihoodsPreselect(data, preselect, &loglikes);
    Vector<BaseFloat> loglikes_copy(loglikes);
    BaseFloat *ptr = loglikes_copy.Data();
    std::nth_element(ptr, ptr+num_preselect-config.diag_gmm_nbest,
                     ptr+num_preselect);
    BaseFloat thresh = ptr[num_preselect-config.diag_gmm_nbest];
    for (int32 p = 0; p < num_preselect; p++) {
      if (loglikes(p) >= thresh) {  // met threshold for diagonal phase.
        int32 g = preselect[p];
        pruned_pairs.push_back(
            std::make_pair(full_ubm_.ComponentLogLikelihood(data, g), g));
      }
    }
  } else {
    for (int32 p = 0; p < num_preselect; p++) {
      int32 g = preselect[p];
      pruned_pairs.push_back(
          std::make_pair(full_ubm_.ComponentLogLikelihood(data, g), g));
    }
  }
  KALDI_ASSERT(!pruned_pairs.empty());
  if (pruned_pairs.size() > static_cast<size_t>(config.full_gmm_nbest)) {
    std::nth_element(pruned_pairs.begin(),
                     pruned_pairs.end() - config.full_gmm_nbest,
                     pruned_pairs.end());
    pruned_pairs.erase(pruned_pairs.begin(),
                       pruned_pairs.end() - config.full_gmm_nbest);
  }
  // Make sure pruned Gaussians appear from best to worst.
  std::sort(pruned_pairs.begin(), pruned_pairs.end(),
            std::greater<std::pair<BaseFloat, int32> >());
  Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
  KALDI_ASSERT(gselect != NULL);
  gselect->resize(pruned_pairs.size());
  for (size_t i = 0; i < pruned_pairs.size(); i++) {
    loglikes_tmp(i) = pruned_pairs[i].first;
    (*gselect)[i] = pruned_pairs[i].second;
  }
  return loglikes_tmp.LogSumExp();
}



void SgmmGauPost::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SgmmGauPost>");
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
  WriteToken(os, binary, "</SgmmGauPost>");
}

void SgmmGauPost::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<SgmmGauPost>");
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
  ExpectToken(is, binary, "</SgmmGauPost>");
}


void AmSgmmFunctions::ComputeDistances(const AmSgmm &model,
                                       const Vector<BaseFloat> &state_occs,
                                       MatrixBase<BaseFloat> *dists) {
  int32 num_states = model.NumPdfs(),
      phn_space_dim = model.PhoneSpaceDim(),
      num_gauss = model.NumGauss();
  KALDI_ASSERT(dists != NULL && dists->NumRows() == num_states
               && dists->NumCols() == num_states);
  Vector<double> prior(state_occs);
  KALDI_ASSERT(prior.Sum() != 0.0);
  prior.Scale(1.0 / prior.Sum());  // Normalize.
  SpMatrix<BaseFloat> H(phn_space_dim);  // The same as H_sm in some other code.
  for (int32 i = 0; i < num_gauss; i++) {
    SpMatrix<BaseFloat> Hi(phn_space_dim);
    Hi.AddMat2Sp(1.0, model.M_[i], kTrans, model.SigmaInv_[i], 0.0);
    H.AddSp(prior(i), Hi);
  }
  bool warned = false;
  for (int32 j1 = 0; j1 < num_states; ++j1) {
    if (model.NumSubstates(j1) != 1 && !warned) {
      KALDI_WARN << "ComputeDistances() can only give meaningful output if you "
                 << "have one substate per state.";
      warned = true;
    }
    for (int32 j2 = 0; j2 <= j1; ++j2) {
      Vector<BaseFloat> v_diff(model.v_[j1].Row(0));
      v_diff.AddVec(-1.0, model.v_[j2].Row(0));
      (*dists)(j1, j2) = (*dists)(j2, j1) = VecSpVec(v_diff, H, v_diff);
    }
  }
}

}  // namespace kaldi

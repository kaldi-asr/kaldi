// gmm/mle-am-diag-gmm.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal);
//                      Microsoft Corporation;  Georg Stemmer;  Yanmin Qian

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

#include "gmm/am-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

const AccumDiagGmm& AccumAmDiagGmm::GetAcc(int32 index) const {
  KALDI_ASSERT(index >= 0 && index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[index]);
}

AccumDiagGmm& AccumAmDiagGmm::GetAcc(int32 index) {
  KALDI_ASSERT(index >= 0 && index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[index]);
}

AccumAmDiagGmm::~AccumAmDiagGmm() {
  DeletePointers(&gmm_accumulators_);
}

void AccumAmDiagGmm::Init(const AmDiagGmm &model,
                              GmmFlagsType flags) {
  DeletePointers(&gmm_accumulators_);  // in case was non-empty when called.
  gmm_accumulators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    gmm_accumulators_[i] = new AccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i), flags);
  }
}

void AccumAmDiagGmm::Init(const AmDiagGmm &model,
                              int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(dim > 0);
  DeletePointers(&gmm_accumulators_);  // in case was non-empty when called.
  gmm_accumulators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    gmm_accumulators_[i] = new AccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i).NumGauss(),
                                 dim, flags);
  }
}

void AccumAmDiagGmm::SetZero(GmmFlagsType flags) {
  for (size_t i = 0; i < gmm_accumulators_.size(); i++) {
    gmm_accumulators_[i]->SetZero(flags);
  }
}

BaseFloat AccumAmDiagGmm::AccumulateForGmm(
    const AmDiagGmm &model, const VectorBase<BaseFloat> &data,
    int32 gmm_index, BaseFloat weight) {
  KALDI_ASSERT(static_cast<size_t>(gmm_index) < gmm_accumulators_.size());
  BaseFloat log_like =
      gmm_accumulators_[gmm_index]->AccumulateFromDiag(model.GetPdf(gmm_index),
                                                       data, weight);
  total_log_like_ += log_like * weight;
  total_frames_ += weight;
  return log_like;
}

BaseFloat AccumAmDiagGmm::AccumulateForGmmTwofeats(
    const AmDiagGmm &model,
    const VectorBase<BaseFloat> &data1,
    const VectorBase<BaseFloat> &data2,
    int32 gmm_index,
    BaseFloat weight) {
  KALDI_ASSERT(static_cast<size_t>(gmm_index) < gmm_accumulators_.size());
  const DiagGmm &gmm = model.GetPdf(gmm_index);
  AccumDiagGmm &acc = *(gmm_accumulators_[gmm_index]);
  Vector<BaseFloat> posteriors;
  BaseFloat log_like = gmm.ComponentPosteriors(data1, &posteriors);
  posteriors.Scale(weight);
  acc.AccumulateFromPosteriors(data2, posteriors);
  total_log_like_ += log_like * weight;
  total_frames_ += weight;
  return log_like;
}


void AccumAmDiagGmm::AccumulateFromPosteriors(
    const AmDiagGmm &model, const VectorBase<BaseFloat> &data,
    int32 gmm_index, const VectorBase<BaseFloat> &posteriors) {
  KALDI_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());
  gmm_accumulators_[gmm_index]->AccumulateFromPosteriors(data, posteriors);
  total_frames_ += posteriors.Sum();
}

void AccumAmDiagGmm::AccumulateForGaussian(
    const AmDiagGmm &am, const VectorBase<BaseFloat> &data,
    int32 gmm_index, int32 gauss_index, BaseFloat weight) {
  KALDI_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());
  KALDI_ASSERT(gauss_index >= 0
      && gauss_index < am.GetPdf(gmm_index).NumGauss());
  gmm_accumulators_[gmm_index]->AccumulateForComponent(data, gauss_index, weight);
}

void AccumAmDiagGmm::Read(std::istream &in_stream, bool binary,
                          bool add) {
  int32 num_pdfs;
  ExpectToken(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && gmm_accumulators_.empty())) {
    gmm_accumulators_.resize(num_pdfs, NULL);
    for (std::vector<AccumDiagGmm*>::iterator it = gmm_accumulators_.begin(),
             end = gmm_accumulators_.end(); it != end; ++it) {
      delete *it;
      *it = new AccumDiagGmm();
      (*it)->Read(in_stream, binary, add);
    }
  } else {
    if (gmm_accumulators_.size() != static_cast<size_t> (num_pdfs))
      KALDI_ERR << "Adding accumulators but num-pdfs do not match: "
                << (gmm_accumulators_.size()) << " vs. "
                << (num_pdfs);
    for (std::vector<AccumDiagGmm*>::iterator it = gmm_accumulators_.begin(),
             end = gmm_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);
  }
  // TODO(arnab): Bad hack! Need to make this self-delimiting.
  in_stream.peek();  // This will set the EOF bit for older accs.
  if (!in_stream.eof()) {
    double like, frames;
    ExpectToken(in_stream, binary, "<total_like>");
    ReadBasicType(in_stream, binary, &like);
    total_log_like_ = (add)? total_log_like_ + like : like;
    ExpectToken(in_stream, binary, "<total_frames>");
    ReadBasicType(in_stream, binary, &frames);
    total_frames_ = (add)? total_frames_ + frames : frames;
  }
}

void AccumAmDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  int32 num_pdfs = gmm_accumulators_.size();
  WriteToken(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (std::vector<AccumDiagGmm*>::const_iterator it =
      gmm_accumulators_.begin(), end = gmm_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
  WriteToken(out_stream, binary, "<total_like>");
  WriteBasicType(out_stream, binary, total_log_like_);

  WriteToken(out_stream, binary, "<total_frames>");
  WriteBasicType(out_stream, binary, total_frames_);
}


// BaseFloat AccumAmDiagGmm::TotCount() const {
//  BaseFloat ans = 0.0;
//  for (int32 pdf = 0; pdf < NumAccs(); pdf++)
//    ans += gmm_accumulators_[pdf]->occupancy().Sum();
//  return ans;
// }

void ResizeModel (int32 dim, AmDiagGmm *am_gmm) {
  for (int32 pdf_id = 0; pdf_id < am_gmm->NumPdfs(); pdf_id++) {
    DiagGmm &pdf = am_gmm->GetPdf(pdf_id);
    pdf.Resize(pdf.NumGauss(), dim);
    Matrix<BaseFloat> inv_vars(pdf.NumGauss(), dim);
    inv_vars.Set(1.0); // make all vars 1.
    pdf.SetInvVars(inv_vars);
    pdf.ComputeGconsts();
  }
}

void MleAmDiagGmmUpdate (const MleDiagGmmOptions &config,
                         const AccumAmDiagGmm &am_diag_gmm_acc,
                         GmmFlagsType flags,
                         AmDiagGmm *am_gmm,
                         BaseFloat *obj_change_out,
                         BaseFloat *count_out) {
  if (am_diag_gmm_acc.Dim() != am_gmm->Dim()) {
    KALDI_ASSERT(am_diag_gmm_acc.Dim() != 0);
    KALDI_WARN << "Dimensions of accumulator " << am_diag_gmm_acc.Dim()
               << " and gmm " << am_gmm->Dim() << " do not match, resizing "
               << " GMM and setting to zero-mean, unit-variance.";
    ResizeModel(am_diag_gmm_acc.Dim(), am_gmm);
  }
  
  KALDI_ASSERT(am_gmm != NULL);
  KALDI_ASSERT(am_diag_gmm_acc.NumAccs() == am_gmm->NumPdfs());
  if (obj_change_out != NULL) *obj_change_out = 0.0;
  if (count_out != NULL) *count_out = 0.0;

  BaseFloat tot_obj_change = 0.0, tot_count = 0.0;
  int32 tot_elems_floored = 0, tot_gauss_floored = 0,
      tot_gauss_removed = 0;
  for (int32 i = 0; i < am_diag_gmm_acc.NumAccs(); i++) {
    BaseFloat obj_change, count;
    int32 elems_floored, gauss_floored, gauss_removed;
    
    MleDiagGmmUpdate(config, am_diag_gmm_acc.GetAcc(i), flags,
                     &(am_gmm->GetPdf(i)),
                     &obj_change, &count, &elems_floored,
                     &gauss_floored, &gauss_removed);
    tot_obj_change += obj_change;
    tot_count += count;
    tot_elems_floored += elems_floored;
    tot_gauss_floored += gauss_floored;
    tot_gauss_removed += gauss_removed;
  }
  if (obj_change_out != NULL) *obj_change_out = tot_obj_change;
  if (count_out != NULL) *count_out = tot_count;
  KALDI_LOG << tot_elems_floored << " variance elements floored in "
            << tot_gauss_floored << " Gaussians, out of "
            <<  am_gmm->NumGauss();
  if (config.remove_low_count_gaussians) {
    KALDI_LOG << "Removed " << tot_gauss_removed
              << " Gaussians due to counts < --min-gaussian-occupancy="
              <<  config.min_gaussian_occupancy
              << " and --remove-low-count-gaussians=true";
  }
}


void MapAmDiagGmmUpdate (const MapDiagGmmOptions &config,
                         const AccumAmDiagGmm &am_diag_gmm_acc,
                         GmmFlagsType flags,
                         AmDiagGmm *am_gmm,
                         BaseFloat *obj_change_out,
                         BaseFloat *count_out) {
  KALDI_ASSERT(am_gmm != NULL && am_diag_gmm_acc.Dim() == am_gmm->Dim() &&
               am_diag_gmm_acc.NumAccs() == am_gmm->NumPdfs());
  if (obj_change_out != NULL) *obj_change_out = 0.0;
  if (count_out != NULL) *count_out = 0.0;
  BaseFloat tmp_obj_change, tmp_count;
  BaseFloat *p_obj = (obj_change_out != NULL) ? &tmp_obj_change : NULL,
      *p_count   = (count_out != NULL) ? &tmp_count : NULL;

  for (int32 i = 0; i < am_diag_gmm_acc.NumAccs(); i++) {
    MapDiagGmmUpdate(config, am_diag_gmm_acc.GetAcc(i), flags,
                     &(am_gmm->GetPdf(i)), p_obj, p_count);

    if (obj_change_out != NULL) *obj_change_out += tmp_obj_change;
    if (count_out != NULL) *count_out += tmp_count;
  }
}


BaseFloat AccumAmDiagGmm::TotStatsCount() const {
  double ans = 0.0;
  for (int32 i = 0; i < NumAccs(); i++) {
    const AccumDiagGmm &acc = GetAcc(i);
    ans += acc.occupancy().Sum();
  }
  return ans;
}

void AccumAmDiagGmm::Scale(BaseFloat scale) {
  for (int32 i = 0; i < NumAccs(); i++) {
    AccumDiagGmm &acc = GetAcc(i);
    acc.Scale(scale, acc.Flags());
  }
  total_frames_ *= scale;
  total_log_like_ *= scale;
}

void AccumAmDiagGmm::Add(BaseFloat scale, const AccumAmDiagGmm &other) {
  total_frames_ += scale * other.total_frames_;
  total_log_like_ += scale * other.total_log_like_;
  
  int32 num_accs = NumAccs();
  KALDI_ASSERT(num_accs == other.NumAccs());
  for (int32 i = 0; i < num_accs; i++)
    gmm_accumulators_[i]->Add(scale, *(other.gmm_accumulators_[i]));
}

}  // namespace kaldi

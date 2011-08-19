// tied/mle-am-tied-diag-gmm.cc

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#include "tied/am-tied-diag-gmm.h"
#include "tied/mle-am-tied-diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

const AccumDiagGmm& AccumAmTiedDiagGmm::GetDiagAcc(int32 index) const {
  assert(index >= 0 && index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[index]);
}

const AccumTiedGmm& AccumAmTiedDiagGmm::GetTiedAcc(int32 pdf_index) const {
  assert(pdf_index >= 0 && pdf_index < static_cast<int32>(tied_gmm_accumulators_.size()));
  return *(tied_gmm_accumulators_[pdf_index]);
}

AccumAmTiedDiagGmm::~AccumAmTiedDiagGmm() {
  DeletePointers(&gmm_accumulators_);
  DeletePointers(&tied_gmm_accumulators_);
}

void AccumAmTiedDiagGmm::Init(const AmTiedDiagGmm &model,
                              GmmFlagsType flags) {
  DeletePointers(&gmm_accumulators_);
  DeletePointers(&tied_gmm_accumulators_);
  
  gmm_accumulators_.resize(model.NumPdfs(), NULL);
  tied_gmm_accumulators_.resize(model.NumTiedPdfs(), NULL);
  
  /// codebook accumulators
  for (int32 i = 0; i < model.NumPdfs(); ++i) {
    gmm_accumulators_[i] = new AccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i), flags);
  }
  
  /// tied gmm accumulators
  for (int32 i = 0; i < model.NumTiedPdfs(); ++i) {
    tied_gmm_accumulators_[i] = new AccumTiedGmm();
    tied_gmm_accumulators_[i]->Resize(model.GetTiedPdf(i), flags);
  }
}

void AccumAmTiedDiagGmm::SetZero(GmmFlagsType flags) {
  for (size_t i = 0; i < gmm_accumulators_.size(); ++i)
    gmm_accumulators_[i]->SetZero(flags);
  for (size_t i = 0; i < tied_gmm_accumulators_.size(); ++i)
    tied_gmm_accumulators_[i]->SetZero(flags);
}

BaseFloat AccumAmTiedDiagGmm::Accumulate(const AmTiedDiagGmm &model, 
                                         const TiedGmmPerFrameVars &per_frame_vars,
                                         int32 pdf_index, 
                                         BaseFloat frame_posterior) {
  KALDI_ASSERT(static_cast<size_t>(pdf_index) < tied_gmm_accumulators_.size());  
  const TiedGmm &tied = model.GetTiedPdf(pdf_index);
  
  KALDI_ASSERT(static_cast<size_t>(tied.pdf_index()) < gmm_accumulators_.size());
    
  Vector<BaseFloat> posteriors(per_frame_vars.ll[tied.pdf_index()]->Dim());
  
  // use the loglikelihoods from the per_frame_vars and the weights of the target
  // pdf to compute the ll and the posteriors
  tied.LogLikelihoods(*(per_frame_vars.ll[tied.pdf_index()]), &posteriors);
  BaseFloat logl = posteriors.ApplySoftMax();
  posteriors.Scale(frame_posterior);
  
  // accumulate for codebook and tied pdf
  AccumulateFromPosteriors(per_frame_vars.x, tied.pdf_index(), pdf_index, posteriors);
  
  return logl;
}

BaseFloat AccumAmTiedDiagGmm::AccumulateForGmm(const AmTiedDiagGmm &model, 
                             const VectorBase<BaseFloat> &data,
                             int32 pdf_index,
                             BaseFloat frame_posterior) {
  KALDI_ASSERT(static_cast<size_t>(pdf_index) < tied_gmm_accumulators_.size());  
  const TiedGmm &tied = model.GetTiedPdf(pdf_index);
  
  KALDI_ASSERT(static_cast<size_t>(tied.pdf_index()) < gmm_accumulators_.size());
  const DiagGmm &diag = model.GetPdf(tied.pdf_index());

  Vector<BaseFloat> scores(diag.Dim());
  Vector<BaseFloat> posteriors(diag.Dim());

  diag.LogLikelihoods(data, &scores);
  tied.LogLikelihoods(scores, &posteriors);

  BaseFloat logl = posteriors.ApplySoftMax();
  posteriors.Scale(frame_posterior);
  
  /// dont forget to accumulate :)
  AccumulateFromPosteriors(data, tied.pdf_index(), pdf_index, posteriors);
  
  return logl;
}

void AccumAmTiedDiagGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat> &data,
    int32 pdf_index,
    int32 tied_pdf_index,
    const VectorBase<BaseFloat> &posteriors) {
  /// accumulate for codebook...
  gmm_accumulators_[pdf_index]->AccumulateFromPosteriors(data, posteriors);
  
  /// and tied gmm
  tied_gmm_accumulators_[tied_pdf_index]->AccumulateFromPosteriors(posteriors);
}

void AccumAmTiedDiagGmm::Read(std::istream& in_stream, bool binary, bool add) {
  int32 num_pdfs, num_tied;
  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && gmm_accumulators_.empty())) {
    gmm_accumulators_.resize(num_pdfs, NULL);
    for (std::vector<AccumDiagGmm*>::iterator it = gmm_accumulators_.begin(),
             end = gmm_accumulators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
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
  
  ExpectMarker(in_stream, binary, "<NUMTIEDPDFS>");
  ReadBasicType(in_stream, binary, &num_tied);
  KALDI_ASSERT(num_tied > 0);
  if (!add || (add && tied_gmm_accumulators_.empty())) {
    tied_gmm_accumulators_.resize(num_tied, NULL);
    for (std::vector<AccumTiedGmm*>::iterator it = tied_gmm_accumulators_.begin(),
             end = tied_gmm_accumulators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
      *it = new AccumTiedGmm();
      (*it)->Read(in_stream, binary, add);
    }
  } else {
    if (tied_gmm_accumulators_.size() != static_cast<size_t> (num_tied))
      KALDI_ERR << "Adding accumulators but num-tied-pdfs do not match: "
                << (tied_gmm_accumulators_.size()) << " vs. "
                << (num_tied);
    for (std::vector<AccumTiedGmm*>::iterator it = tied_gmm_accumulators_.begin(),
             end = tied_gmm_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);
  }
}

void AccumAmTiedDiagGmm::Write(std::ostream& out_stream, bool binary) const {
  int32 num_pdfs = gmm_accumulators_.size();
  int32 num_tied = tied_gmm_accumulators_.size();
  WriteMarker(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (std::vector<AccumDiagGmm*>::const_iterator it =
      gmm_accumulators_.begin(), end = gmm_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
  WriteMarker(out_stream, binary, "<NUMTIEDPDFS>");
  WriteBasicType(out_stream, binary, num_tied);
  for (std::vector<AccumTiedGmm*>::const_iterator it =
      tied_gmm_accumulators_.begin(), end = tied_gmm_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}

void MleAmTiedDiagGmmUpdate(
            const MleDiagGmmOptions &config_diag,
            const MleTiedGmmOptions &config_tied,
            const AccumAmTiedDiagGmm &acc,
            GmmFlagsType flags,
            AmTiedDiagGmm *model,
            BaseFloat *obj_change_out,
            BaseFloat *count_out) {
  KALDI_ASSERT(model != NULL);
  KALDI_ASSERT(acc.NumDiagAccs() == model->NumPdfs());
  KALDI_ASSERT(acc.NumTiedAccs() == model->NumTiedPdfs());
  
  if (obj_change_out != NULL) *obj_change_out = 0.0;
  if (count_out != NULL) *count_out = 0.0;
  
  BaseFloat tmp_obj_change, tmp_count;
  BaseFloat *p_obj = (obj_change_out != NULL) ? &tmp_obj_change : NULL,
            *p_count   = (count_out != NULL) ? &tmp_count : NULL;

  /// reestimate the codebooks
  for (size_t i = 0; i < acc.NumDiagAccs(); i++) {
    MleDiagGmmUpdate(config_diag, acc.GetDiagAcc(i), flags & !kGmmWeights, &(model->GetPdf(i)), 
        p_obj, p_count);

    if (obj_change_out != NULL) *obj_change_out += tmp_obj_change;
    if (count_out != NULL) *count_out += tmp_count;
  }

  if (!(flags & kGmmWeights)) {
    KALDI_WARN << "no weight update as desired by flags";
    return;
  }
  
  /// reestimate the tied gmms
  for (size_t i = 0; i < acc.NumTiedAccs(); i++) {
    MleTiedGmmUpdate(config_tied, acc.GetTiedAcc(i), flags, &(model->GetTiedPdf(i)), 
        p_obj, p_count);

    if (obj_change_out != NULL) *obj_change_out += tmp_obj_change;
    if (count_out != NULL) *count_out += tmp_count;
  }
}

}  // namespace kaldi

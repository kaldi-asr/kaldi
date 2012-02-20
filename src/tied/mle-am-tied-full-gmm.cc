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

#include "tied/am-tied-full-gmm.h"
#include "tied/mle-am-tied-full-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

using std::vector;

AccumFullGmm& AccumAmTiedFullGmm::GetFullAcc(int32 codebook_index) const {
  KALDI_ASSERT(codebook_index >= 0 &&
               codebook_index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[codebook_index]);
}

AccumTiedGmm& AccumAmTiedFullGmm::GetTiedAcc(int32 tied_pdf_index) const {
  KALDI_ASSERT(tied_pdf_index >= 0 && tied_pdf_index
               < static_cast<int32>(tied_gmm_accumulators_.size()));
  return *(tied_gmm_accumulators_[tied_pdf_index]);
}

AccumAmTiedFullGmm::~AccumAmTiedFullGmm() {
  DeletePointers(&gmm_accumulators_);
  DeletePointers(&tied_gmm_accumulators_);
}

void AccumAmTiedFullGmm::Init(const AmTiedFullGmm &model,
                              GmmFlagsType flags) {
  DeletePointers(&gmm_accumulators_);
  DeletePointers(&tied_gmm_accumulators_);

  gmm_accumulators_.resize(model.NumCodebooks(), NULL);
  tied_gmm_accumulators_.resize(model.NumTiedPdfs(), NULL);

  /// codebook accumulators
  for (int32 i = 0; i < model.NumCodebooks(); ++i) {
    gmm_accumulators_[i] = new AccumFullGmm();
    gmm_accumulators_[i]->Resize(model.GetCodebook(i), flags);
  }

  /// tied gmm accumulators
  for (int32 i = 0; i < model.NumTiedPdfs(); ++i) {
    tied_gmm_accumulators_[i] = new AccumTiedGmm();
    tied_gmm_accumulators_[i]->Resize(model.GetTiedPdf(i), flags);
  }
}

void AccumAmTiedFullGmm::SetZero(GmmFlagsType flags) {
  for (size_t i = 0; i < gmm_accumulators_.size(); ++i)
    gmm_accumulators_[i]->SetZero(flags);
  for (size_t i = 0; i < tied_gmm_accumulators_.size(); ++i)
    tied_gmm_accumulators_[i]->SetZero(flags);
}

BaseFloat AccumAmTiedFullGmm::AccumulateForTied(
            const AmTiedFullGmm &model,
            const VectorBase<BaseFloat> &data,
            int32 tied_pdf_index,
            BaseFloat frame_posterior) {
  KALDI_ASSERT(static_cast<size_t>(tied_pdf_index) <
               tied_gmm_accumulators_.size());
  const TiedGmm &tied = model.GetTiedPdf(tied_pdf_index);

  int32 i = tied.codebook_index();
  KALDI_ASSERT(static_cast<size_t>(i) < gmm_accumulators_.size());

  Vector<BaseFloat> svq;
  BaseFloat c = model.ComputePerFrameVars(data, i, &svq);

  Vector<BaseFloat> posteriors;
  BaseFloat logl = tied.ComponentPosteriors(c, svq, &posteriors);

  // scale by frame posterior
  posteriors.Scale(frame_posterior);

  // dont forget to accumulate :)
  AccumulateFromPosteriors(data, i, tied_pdf_index, posteriors);

  return logl;
}

void AccumAmTiedFullGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat> &data,
    int32 codebook_index,
    int32 tied_pdf_index,
    const VectorBase<BaseFloat> &posteriors) {
  KALDI_ASSERT(codebook_index < NumFullAccs());
  KALDI_ASSERT(tied_pdf_index < NumTiedAccs());

  /// accumulate for codebook...
  gmm_accumulators_[codebook_index]->
    AccumulateFromPosteriors(data, posteriors);

  /// and tied gmm
  tied_gmm_accumulators_[tied_pdf_index]->
    AccumulateFromPosteriors(posteriors);
}

void AccumAmTiedFullGmm::Read(std::istream& in_stream, bool binary, bool add) {
  int32 num_pdfs, num_tied;
  ExpectToken(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && gmm_accumulators_.empty())) {
    gmm_accumulators_.resize(num_pdfs, NULL);
    for (vector<AccumFullGmm*>::iterator it = gmm_accumulators_.begin(),
             end = gmm_accumulators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
      *it = new AccumFullGmm();
      (*it)->Read(in_stream, binary, add);
    }
  } else {
    if (gmm_accumulators_.size() != static_cast<size_t> (num_pdfs))
      KALDI_ERR << "Adding accumulators but num-pdfs do not match: "
                << (gmm_accumulators_.size()) << " vs. "
                << (num_pdfs);
    for (vector<AccumFullGmm*>::iterator it = gmm_accumulators_.begin(),
             end = gmm_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);
  }

  ExpectToken(in_stream, binary, "<NUMTIEDPDFS>");
  ReadBasicType(in_stream, binary, &num_tied);
  KALDI_ASSERT(num_tied > 0);
  if (!add || (add && tied_gmm_accumulators_.empty())) {
    tied_gmm_accumulators_.resize(num_tied, NULL);
    for (vector<AccumTiedGmm*>::iterator it = tied_gmm_accumulators_.begin(),
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
    for (vector<AccumTiedGmm*>::iterator it = tied_gmm_accumulators_.begin(),
             end = tied_gmm_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);
  }
}

void AccumAmTiedFullGmm::Write(std::ostream& out_stream, bool binary) const {
  int32 num_pdfs = gmm_accumulators_.size();
  int32 num_tied = tied_gmm_accumulators_.size();
  WriteToken(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (vector<AccumFullGmm*>::const_iterator it = gmm_accumulators_.begin(),
       end = gmm_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
  WriteToken(out_stream, binary, "<NUMTIEDPDFS>");
  WriteBasicType(out_stream, binary, num_tied);
  for (vector<AccumTiedGmm*>::const_iterator it =
       tied_gmm_accumulators_.begin(), end = tied_gmm_accumulators_.end();
       it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}

void MleAmTiedFullGmmUpdate(
       const MleFullGmmOptions &config_full,
       const MleTiedGmmOptions &config_tied,
       const AccumAmTiedFullGmm &acc,
       GmmFlagsType flags,
       AmTiedFullGmm *model,
       BaseFloat *obj_change_out_cb,
       BaseFloat *count_out_cb,
       BaseFloat *obj_change_out_tied,
       BaseFloat *count_out_tied) {
  KALDI_ASSERT(model != NULL);
  KALDI_ASSERT(acc.NumFullAccs() == model->NumCodebooks());
  KALDI_ASSERT(acc.NumTiedAccs() == model->NumTiedPdfs());

  KALDI_ASSERT(!config_full.remove_low_count_gaussians);

  // clear output stats
  if (obj_change_out_cb != NULL) *obj_change_out_cb = 0.0;
  if (count_out_cb != NULL) *count_out_cb = 0.0;
  if (obj_change_out_tied != NULL) *obj_change_out_tied = 0.0;
  if (count_out_tied != NULL) *count_out_tied = 0.0;

  AmTiedFullGmm *oldm = NULL;
  if (config_tied.interpolate()) {
    // make a copy of the model
    oldm = new AmTiedFullGmm();
    oldm->CopyFromAmTiedFullGmm(*model);
  }

  BaseFloat tmp_obj_change = 0.0, tmp_count = 0.0;
  BaseFloat *p_obj = (obj_change_out_cb != NULL) ? &tmp_obj_change : NULL,
            *p_count = (count_out_cb != NULL) ? &tmp_count : NULL;

  // reestimate the codebooks
  for (int32 i = 0; i < acc.NumFullAccs(); i++) {
    // modify flags by enforcing no weight update
    MleFullGmmUpdate(config_full, acc.GetFullAcc(i), flags & ~kGmmWeights,
                     &(model->GetCodebook(i)), p_obj, p_count);

    KALDI_VLOG(1) << "MleFullGmmUpdate " << i << " delta-obj="
                  << (tmp_obj_change/tmp_count) << " count=" << tmp_count;

    if (obj_change_out_cb != NULL) *obj_change_out_cb += tmp_obj_change;
    if (count_out_cb != NULL) *count_out_cb += tmp_count;
  }

  // only reestimate the tied Gmms if we have a weight update requested
  if (flags & kGmmWeights) {
    tmp_obj_change = 0.0;
    tmp_count = 0.0;
    p_obj = (obj_change_out_tied != NULL) ? &tmp_obj_change : NULL;
    p_count = (count_out_tied != NULL) ? &tmp_count : NULL;

    // reestimate the tied gmms
    for (int32 i = 0; i < acc.NumTiedAccs(); i++) {
      MleTiedGmmUpdate(config_tied, acc.GetTiedAcc(i), flags,
                       &(model->GetTiedPdf(i)), p_obj, p_count);

      KALDI_VLOG(1) << "MleTiedGmmUpdate tied-pdf " << i << " delta-obj="
                    << (tmp_obj_change/tmp_count) << " count=" << tmp_count;

      if (obj_change_out_tied != NULL) *obj_change_out_cb += tmp_obj_change;
      if (count_out_tied != NULL) *count_out_tied += tmp_count;
    }

  } else {
    KALDI_LOG << "No weight update for tied states as requested by flags.";
  }

  // smooth new model with old: new <- wt*est + (1-est)old
  if (config_tied.interpolate()) {
    BaseFloat wt = config_tied.interpolation_weight;

    KALDI_LOG << "Interpolating MLE estimate with prior iteration, (rho="
              << wt << "): "
              << (config_tied.interpolate_weights ? "weights " : "")
              << (config_tied.interpolate_weights ? "means " : "")
              << (config_tied.interpolate_weights ? "variances" : "");

    // smooth the weights for tied...
    if ((flags & kGmmWeights) && config_tied.interpolate_weights) {
      for (int32 i = 0; i < model->NumTiedPdfs(); ++i)
        model->GetTiedPdf(i).Interpolate(wt, oldm->GetTiedPdf(i));
    }

    // ...and mean/var for codebooks
    if ((flags & kGmmMeans) && config_tied.interpolate_means) {
      for (int32 i = 0; i < model->NumCodebooks(); ++i)
        model->GetCodebook(i).Interpolate(wt, oldm->GetCodebook(i),
                                          kGmmMeans);
    }

    if ((flags & kGmmVariances) && config_tied.interpolate_variances) {
      for (int32 i = 0; i < model->NumCodebooks(); ++i)
        model->GetCodebook(i).Interpolate(wt, oldm->GetCodebook(i),
                                          kGmmVariances);
    }

    delete oldm;
  }
}

}  // namespace kaldi

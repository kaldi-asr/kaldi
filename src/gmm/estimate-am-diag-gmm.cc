// gmm/estimate-am-diag-gmm.cc

// Copyright 2009-2011  Arnab Ghoshal (Saarland University)  Microsoft Corporation  Georg Stemmer

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
#include "gmm/estimate-am-diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

MlEstimateDiagGmm& MlEstimateAmDiagGmm::GetAcc(int32 index) {
  assert(index >= 0 && index < static_cast<int32>(gmm_estimators_.size()));
  return *(gmm_estimators_[index]);
}

const MlEstimateDiagGmm& MlEstimateAmDiagGmm::GetAcc(int32 index) const {
  assert(index >= 0 && index < static_cast<int32>(gmm_estimators_.size()));
  return *(gmm_estimators_[index]);
}

MlEstimateAmDiagGmm::~MlEstimateAmDiagGmm() {
  DeletePointers(&gmm_estimators_);
}

void MlEstimateAmDiagGmm::InitAccumulators(const AmDiagGmm &model,
                                           GmmFlagsType flags) {
  DeletePointers(&gmm_estimators_);  // in case was non-empty when called.
  gmm_estimators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    gmm_estimators_[i] = new MlEstimateDiagGmm();
    gmm_estimators_[i]->ResizeAccumulators(model.GetPdf(i), flags);
  }
}

void MlEstimateAmDiagGmm::InitAccumulators(const AmDiagGmm &model,
                                           int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(dim > 0);
  DeletePointers(&gmm_estimators_);  // in case was non-empty when called.
  gmm_estimators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    gmm_estimators_[i] = new MlEstimateDiagGmm();
    gmm_estimators_[i]->ResizeAccumulators(model.GetPdf(i).NumGauss(),
                                           dim, flags);
  }
}

void MlEstimateAmDiagGmm::ZeroAccumulators(GmmFlagsType flags) {
  for (size_t i = 0; i < gmm_estimators_.size(); ++i) {
    gmm_estimators_[i]->ZeroAccumulators(flags);
  }
}

BaseFloat MlEstimateAmDiagGmm::AccumulateForGmm(
    const AmDiagGmm &model, const VectorBase<BaseFloat>& data,
    int32 gmm_index, BaseFloat weight) {
  KALDI_ASSERT(static_cast<size_t>(gmm_index) < gmm_estimators_.size());
  return gmm_estimators_[gmm_index]->AccumulateFromDiag(model.GetPdf(
      gmm_index), data, weight);
}

BaseFloat MlEstimateAmDiagGmm::AccumulateForGmmTwofeats(
    const AmDiagGmm &model,
    const VectorBase<BaseFloat>& data1,
    const VectorBase<BaseFloat>& data2,
    int32 gmm_index,
    BaseFloat weight) {
  assert(static_cast<size_t>(gmm_index) < gmm_estimators_.size());
  const DiagGmm &gmm = model.GetPdf(gmm_index);
  MlEstimateDiagGmm &acc = *(gmm_estimators_[gmm_index]);
  Vector<BaseFloat> posteriors;
  BaseFloat log_like = gmm.ComponentPosteriors(data1, &posteriors);
  acc.AccumulateFromPosteriors(data2, posteriors);
  return log_like;
}


void MlEstimateAmDiagGmm::AccumulateFromPosteriors(
    const AmDiagGmm &model, const VectorBase<BaseFloat>& data,
    int32 gmm_index, const VectorBase<BaseFloat>& posteriors) {
  KALDI_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());
  gmm_estimators_[gmm_index]->AccumulateFromPosteriors(data, posteriors);
}

void MlEstimateAmDiagGmm::AccumulateForGaussian(
    const AmDiagGmm &am, const VectorBase<BaseFloat>& data,
    int32 gmm_index, int32 gauss_index, BaseFloat weight) {
  KALDI_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());
  KALDI_ASSERT(gauss_index >= 0
      && gauss_index < am.GetPdf(gmm_index).NumGauss());
  gmm_estimators_[gmm_index]->AccumulateForComponent(data, gauss_index, weight);
}

void MlEstimateAmDiagGmm::Update(const MleDiagGmmOptions &config,
                                 GmmFlagsType flags,
                                 AmDiagGmm *am_gmm,
                                 BaseFloat *obj_change_out,
                                 BaseFloat *count_out) const {
  KALDI_ASSERT(am_gmm != NULL);
  KALDI_ASSERT(static_cast<int32>(gmm_estimators_.size()) == am_gmm->NumPdfs());
  if (obj_change_out != NULL) *obj_change_out = 0.0;
  if (count_out != NULL) *count_out = 0.0;
  BaseFloat tmp_obj_change, tmp_count;
  BaseFloat *p_obj = (obj_change_out != NULL) ? &tmp_obj_change : NULL,
            *p_count   = (count_out != NULL) ? &tmp_count : NULL;

  for (size_t i = 0; i < gmm_estimators_.size(); i++) {
    gmm_estimators_[i]->Update(config, flags, &(am_gmm->GetPdf(i)), p_obj,
        p_count);

    if (obj_change_out != NULL) *obj_change_out += tmp_obj_change;
    if (count_out != NULL) *count_out += tmp_count;
  }
}

void MlEstimateAmDiagGmm::Read(std::istream& in_stream, bool binary,
                               bool add) {
  int32 num_pdfs;
  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && gmm_estimators_.empty())) {
    gmm_estimators_.resize(num_pdfs, NULL);
    for (std::vector<MlEstimateDiagGmm*>::iterator it = gmm_estimators_.begin(),
             end = gmm_estimators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
      *it = new MlEstimateDiagGmm();
      (*it)->Read(in_stream, binary, add);
    }
  } else {
    if (gmm_estimators_.size() != static_cast<size_t> (num_pdfs))
      KALDI_ERR << "Adding accumulators but num-pdfs do not match: "
                << (gmm_estimators_.size()) << " vs. "
                << (num_pdfs);
    for (std::vector<MlEstimateDiagGmm*>::iterator it = gmm_estimators_.begin(),
             end = gmm_estimators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);
  }
}

void MlEstimateAmDiagGmm::Write(std::ostream& out_stream, bool binary) const {
  int32 num_pdfs = gmm_estimators_.size();
  WriteMarker(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (std::vector<MlEstimateDiagGmm*>::const_iterator it =
      gmm_estimators_.begin(), end = gmm_estimators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}

}  // namespace kaldi

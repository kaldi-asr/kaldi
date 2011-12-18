// gmm/mmie-am-diag-gmm.cc

// Copyright 2009-2011  Petr Motlicek

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
#include "gmm/mmie-am-diag-gmm.h"
#include "gmm/mmie-diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

/// Reads accumulators of MMI Numerators
AccumDiagGmm& MmieAccumAmDiagGmm::GetNumAcc(int32 index) const {
  assert(index >= 0 && index < static_cast<int32>(num_accumulators_.size()));
  return *(num_accumulators_[index]);
}

/// Reads accumulators of MMI Denominators
AccumDiagGmm& MmieAccumAmDiagGmm::GetDenAcc(int32 index) const {
  assert(index >= 0 && index < static_cast<int32>(den_accumulators_.size()));
  return *(den_accumulators_[index]);
}

/// Reads accumulators of I-Smooth 
AccumDiagGmm& MmieAccumAmDiagGmm::GetISmoothAcc(int32 index) const {
  assert(index >= 0 && index < static_cast<int32>(i_smooth_accumulators_.size()));
  return *(i_smooth_accumulators_[index]);
}

MmieAccumAmDiagGmm::~MmieAccumAmDiagGmm() {
  DeletePointers(&num_accumulators_);
  DeletePointers(&den_accumulators_);
}

void MmieAccumAmDiagGmm::Init(const AmDiagGmm &model,
                              GmmFlagsType flags) {
  DeletePointers(&num_accumulators_);  // in case was non-empty when called.
  DeletePointers(&den_accumulators_);  // in case was non-empty when called.
    
  num_accumulators_.resize(model.NumPdfs(), NULL);
  den_accumulators_.resize(model.NumPdfs(), NULL);
    
  for (int32 i = 0; i < model.NumPdfs(); i++) {
    num_accumulators_[i] = new AccumDiagGmm();
    num_accumulators_[i]->Resize(model.GetPdf(i), flags);
    den_accumulators_[i] = new AccumDiagGmm();
    den_accumulators_[i]->Resize(model.GetPdf(i), flags);

  }
}

void MmieAccumAmDiagGmm::Init(const AmDiagGmm &model,
                              int32 dim, GmmFlagsType flags) {
  KALDI_ASSERT(dim > 0);
  DeletePointers(&num_accumulators_);  // in case was non-empty when called.
  DeletePointers(&den_accumulators_);  // in case was non-empty when called.  
  num_accumulators_.resize(model.NumPdfs(), NULL);
  den_accumulators_.resize(model.NumPdfs(), NULL);

  for (int32 i = 0; i < model.NumPdfs(); i++) {
    num_accumulators_[i] = new AccumDiagGmm();
    num_accumulators_[i]->Resize(model.GetPdf(i).NumGauss(),
                                 dim, flags);
    den_accumulators_[i] = new AccumDiagGmm();
    den_accumulators_[i]->Resize(model.GetPdf(i).NumGauss(),
                                 dim, flags);

  }
}

void MmieAccumAmDiagGmm::SetZero(GmmFlagsType flags) {
  for (size_t i = 0; i < num_accumulators_.size(); ++i) {
    num_accumulators_[i]->SetZero(flags);
    den_accumulators_[i]->SetZero(flags);
  }
}


void MmieAccumAmDiagGmm::ReadNum(std::istream& in_stream, bool binary,
                               bool add) {
  int32 num_pdfs;
  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && num_accumulators_.empty())) {
    num_accumulators_.resize(num_pdfs, NULL);
    for (std::vector<AccumDiagGmm*>::iterator it = num_accumulators_.begin(),
             end = num_accumulators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
      *it = new AccumDiagGmm();
      (*it)->Read(in_stream, binary, add);
    }


  } else {
    if (num_accumulators_.size() != static_cast<size_t> (num_pdfs))
      KALDI_ERR << "Adding NUM accumulators but num-pdfs do not match: "
                << (num_accumulators_.size()) << " vs. "
                << (num_pdfs);

    for (std::vector<AccumDiagGmm*>::iterator it = num_accumulators_.begin(),
             end = num_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);

  }
}

void MmieAccumAmDiagGmm::ReadDen(std::istream& in_stream, bool binary,
                               bool add) {
  int32 num_pdfs;
  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && den_accumulators_.empty())) {
    den_accumulators_.resize(num_pdfs, NULL);
    for (std::vector<AccumDiagGmm*>::iterator it = den_accumulators_.begin(),
             end = den_accumulators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
      *it = new AccumDiagGmm();
      (*it)->Read(in_stream, binary, add);
    }


  } else {
    if (den_accumulators_.size() != static_cast<size_t> (num_pdfs))
      KALDI_ERR << "Adding DEN accumulators but num-pdfs do not match: "
                << (den_accumulators_.size()) << " vs. "
                << (num_pdfs);

    for (std::vector<AccumDiagGmm*>::iterator it = den_accumulators_.begin(),
             end = den_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);

  }
}

void MmieAccumAmDiagGmm::ReadISmooth(std::istream& in_stream, bool binary,
                               bool add) {
  int32 num_pdfs;
  ExpectMarker(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && i_smooth_accumulators_.empty())) {
    i_smooth_accumulators_.resize(num_pdfs, NULL);
    for (std::vector<AccumDiagGmm*>::iterator it = i_smooth_accumulators_.begin(),
             end = i_smooth_accumulators_.end(); it != end; ++it) {
      if (*it != NULL) delete *it;
      *it = new AccumDiagGmm();
      (*it)->Read(in_stream, binary, add);
    }


  } else {
    if (i_smooth_accumulators_.size() != static_cast<size_t> (num_pdfs))
      KALDI_ERR << "Adding DEN accumulators but num-pdfs do not match: "
                << (i_smooth_accumulators_.size()) << " vs. "
                << (num_pdfs);

    for (std::vector<AccumDiagGmm*>::iterator it = i_smooth_accumulators_.begin(),
             end = i_smooth_accumulators_.end(); it != end; ++it)
      (*it)->Read(in_stream, binary, add);

  }
}

void MmieAccumAmDiagGmm::WriteNum(std::ostream& out_stream, bool binary) const {
  int32 num_pdfs = num_accumulators_.size();
  WriteMarker(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (std::vector<AccumDiagGmm*>::const_iterator it =
      num_accumulators_.begin(), end = num_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}


void MmieAccumAmDiagGmm::WriteDen(std::ostream& out_stream, bool binary) const {
  int32 num_pdfs = den_accumulators_.size();
  WriteMarker(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (std::vector<AccumDiagGmm*>::const_iterator it =
      den_accumulators_.begin(), end = den_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
}


void MmieAmDiagGmmUpdate(const MmieDiagGmmOptions &config,
                         const MmieAccumAmDiagGmm &mmieamdiaggmm_acc,
                         GmmFlagsType flags,
                         AmDiagGmm *am_gmm,
                         BaseFloat *auxf_change_gauss,
                         BaseFloat *auxf_change_weights,
                         BaseFloat *count_out,
                         int32 *num_floored_out) {
  KALDI_ASSERT(am_gmm != NULL);
  KALDI_ASSERT(mmieamdiaggmm_acc.NumAccs() == am_gmm->NumPdfs());
  if (auxf_change_gauss != NULL) *auxf_change_gauss = 0.0;
  if (auxf_change_weights != NULL) *auxf_change_weights = 0.0;
  if (count_out != NULL) *count_out = 0.0;
  if (num_floored_out != NULL) *num_floored_out = 0.0;
  BaseFloat tmp_auxf_change_gauss, tmp_auxf_change_weights, tmp_count;
  int32 tmp_num_floored;

  MmieAccumDiagGmm mmie_gmm;

  for (size_t i = 0; i < mmieamdiaggmm_acc.NumAccs(); i++) {
     mmie_gmm.Resize(am_gmm->GetPdf(i).NumGauss(), am_gmm->GetPdf(i).Dim(), flags);
     mmie_gmm.SubtractAccumulatorsISmoothing(mmieamdiaggmm_acc.GetNumAcc(i),
                                             mmieamdiaggmm_acc.GetDenAcc(i),
                                             config,
                                             config.has_i_smooth_stats ?
                                             mmieamdiaggmm_acc.GetISmoothAcc(i):
                                             mmieamdiaggmm_acc.GetNumAcc(i));
     mmie_gmm.Update(config, flags, &(am_gmm->GetPdf(i)),
                     &tmp_auxf_change_gauss, &tmp_auxf_change_weights,
                     &tmp_count, &tmp_num_floored);
     if (auxf_change_gauss != NULL) *auxf_change_gauss += tmp_auxf_change_gauss;
     if (auxf_change_weights != NULL) *auxf_change_weights += tmp_auxf_change_weights;
     if (count_out != NULL) *count_out += tmp_count;
     if (num_floored_out != NULL) *num_floored_out += tmp_num_floored;
  }
}

BaseFloat MmieAccumAmDiagGmm::TotNumCount() {
  BaseFloat ans = 0.0;
  for (size_t i = 0; i < num_accumulators_.size(); i++)
    if (num_accumulators_[i])
      ans += num_accumulators_[i]->occupancy().Sum();
  return ans;
}

BaseFloat MmieAccumAmDiagGmm::TotDenCount() {
  BaseFloat ans = 0.0;
  for (size_t i = 0; i < den_accumulators_.size(); i++)
    if (den_accumulators_[i])
      ans += den_accumulators_[i]->occupancy().Sum();
  return ans;
}



}  // namespace kaldi

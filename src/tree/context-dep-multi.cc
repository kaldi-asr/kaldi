// tree/context-dep-multi.cc

// Copyright 2009-2011  Microsoft Corporation
//           2015       Hainan Xu

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

#include "tree/context-dep-multi.h"
#include "base/kaldi-math.h"
#include "tree/build-tree.h"
#include "tree/build-tree-virtual.h"

namespace kaldi {

ContextDependencyMulti::ContextDependencyMulti(
                         const vector<std::pair<int32, int32> > &NPs,
                         const vector<EventMap*> &single_trees,
                         const HmmTopology &topo):
    topo_(topo) {
  KALDI_ASSERT(NPs.size() == single_trees.size());
  int num_left_context = 0;
  int num_right_context = 0;
  for (int i = 0; i < NPs.size(); i++) {
    if (NPs[i].second > num_left_context) {
      num_left_context = NPs[i].second;
    }
    if (NPs[i].first - 1 - NPs[i].second > num_right_context) {
      num_right_context = NPs[i].first - 1 - NPs[i].second;
    }
  }
  N_ = num_left_context + 1 + num_right_context;
  P_ = num_left_context;
  
  for (int i = 0; i < NPs.size(); i++) {
    ConvertTreeContext(NPs[i].second, P_, single_trees[i]);
    single_trees_.push_back(single_trees[i]); // not changing it from now
  }

  BuildVirtualTree();
}

void ContextDependencyMulti::ConvertTreeContext(int32 old_P, int32 new_P,
                                                EventMap* tree) {
  if (old_P == new_P) {
    return; // no need to do anything
  }
  tree->ApplyOffsetToCentralPhone(new_P - old_P);
}

bool ContextDependencyMulti::Compute(const std::vector<int32> &phoneseq,
                                     int32 pdf_class,
                                     int32 *pdf_id) const {
  KALDI_ASSERT(static_cast<int32>(phoneseq.size()) == N_);
  EventType event_vec;
  event_vec.reserve(N_+1);
  event_vec.push_back(std::make_pair
                      (static_cast<EventKeyType>(kPdfClass),  // -1
                       static_cast<EventValueType>(pdf_class)));
  KALDI_COMPILE_TIME_ASSERT(kPdfClass < 0);  // or it would not be sorted.
  for (int32 i = 0;i < N_;i++) {
    event_vec.push_back(std::make_pair
                        (static_cast<EventKeyType>(i),
                         static_cast<EventValueType>(phoneseq[i])));
    KALDI_ASSERT(static_cast<EventAnswerType>(phoneseq[i]) != -1);  // >=0 ?
  }
  KALDI_ASSERT(pdf_id != NULL);
  return to_pdf_->Map(event_vec, pdf_id);
}

void ContextDependencyMulti::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "ContextDependencyMulti");
  WriteBasicType(os, binary, N_);
  WriteBasicType(os, binary, P_);
  WriteToken(os, binary, "Trees");
  WriteBasicType(os, binary, single_trees_.size());
  for (int i = 0; i < single_trees_.size(); i++) {
    single_trees_[i]->Write(os, binary);
  }
  WriteToken(os, binary, "EndContextDependencyMulti");
}

//*
void ContextDependencyMulti::WriteVirtualTree(std::ostream &os,
                                              bool binary) const {
  //to_pdf_->Write(os, binary);
  WriteToken(os, binary, "ContextDependency");
  WriteBasicType(os, binary, N_);
  WriteBasicType(os, binary, P_);
  WriteToken(os, binary, "ToPdf");
  to_pdf_->Write(os, binary);
  WriteToken(os, binary, "EndContextDependency");
}
//*/

void ContextDependencyMulti::WriteMapping(std::ostream &os,
                                          bool binary) const {
  WriteMultiTreeMapping(mappings_, os, binary, single_trees_.size());
}

void ContextDependencyMulti::Read(std::istream &is, bool binary) {
  // first clear the pointers
  if (to_pdf_) {
    delete to_pdf_;
    to_pdf_ = NULL;
  }
  for (int i = 0; i < single_trees_.size(); i++) {
    if (single_trees_[i]) {
      delete single_trees_[i];
    }
  }

  ExpectToken(is, binary, "ContextDependencyMulti");
  ReadBasicType(is, binary, &N_);
  ReadBasicType(is, binary, &P_);
  ExpectToken(is, binary, "Trees");
  size_t size;
  ReadBasicType(is, binary, &size);
  single_trees_.resize(size);

  for (int i = 0; i < size; i++) {
    single_trees_[i] = EventMap::Read(is, binary);
  }
  ExpectToken(is, binary, "EndContextDependencyMulti");

  BuildVirtualTree();
}

void ContextDependencyMulti::GetPdfInfo(
         const std::vector<int32> &phones,
         const std::vector<int32> &num_pdf_classes,  // indexed by phone,
         std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const {  

  EventType vec;
  KALDI_ASSERT(pdf_info != NULL);
  pdf_info->resize(NumPdfs());
  for (size_t i = 0 ; i < phones.size(); i++) {
    int32 phone = phones[i];
    vec.clear();
    vec.push_back(std::make_pair(static_cast<EventKeyType>(P_),
                                 static_cast<EventValueType>(phone)));
    // Now get length.
    KALDI_ASSERT(static_cast<size_t>(phone) < num_pdf_classes.size());
    EventAnswerType len = num_pdf_classes[phone];

    for (int32 pos = 0; pos < len; pos++) {
      vec.resize(2);
      vec[0] = std::make_pair(static_cast<EventKeyType>(P_),
                              static_cast<EventValueType>(phone));
      vec[1] = std::make_pair(kPdfClass, static_cast<EventValueType>(pos));
      std::sort(vec.begin(), vec.end());
      std::vector<EventAnswerType> pdfs;  // pdfs that can be at this pos as this phone.
      to_pdf_->MultiMap(vec, &pdfs);
      SortAndUniq(&pdfs);
      if (pdfs.empty()) {
        KALDI_WARN << "ContextDependencyMulti::GetPdfInfo, "
                   "no pdfs returned for position "
                   << pos << " of phone " << phone
                   << ".   Continuing but this is a serious error.";
      }
      for (size_t j = 0; j < pdfs.size(); j++) {
        KALDI_ASSERT(static_cast<size_t>(pdfs[j]) < pdf_info->size());
        (*pdf_info)[pdfs[j]].push_back(std::make_pair(phone, pos));
      }
    }
  }
  for (size_t i = 0; i < pdf_info->size(); i++) {
    std::sort( ((*pdf_info)[i]).begin(),  ((*pdf_info)[i]).end());
    KALDI_ASSERT(IsSortedAndUniq( ((*pdf_info)[i])));  // should have no dups.
  }
}

void ContextDependencyMulti::BuildVirtualTree() {
  vector<int32> phone2num_pdf_classes;
  topo_.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
  MultiTreePdfMap m(single_trees_, N_, P_, phone2num_pdf_classes);
  to_pdf_ = m.GenerateVirtualTree(mappings_);
}
} // end namespace kaldi.


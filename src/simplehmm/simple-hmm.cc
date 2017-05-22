// hmm/simple-hmm.cc

// Copyright 2016   Vimal Manohar

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

#include "simplehmm/simple-hmm.h"

namespace kaldi {

void SimpleHmm::FakeContextDependency::GetPdfInfo(
    const std::vector<int32> &phones,  // list of phones
    const std::vector<int32> &num_pdf_classes,  // indexed by phone,
    std::vector<std::vector<std::pair<int32, int32> > > *pdf_info) const {
  KALDI_ASSERT(phones.size() == 1 && phones[0] == 1);
  KALDI_ASSERT(num_pdf_classes.size() == 2 && 
      num_pdf_classes[1] == NumPdfs());
  KALDI_ASSERT(pdf_info);
  pdf_info->resize(NumPdfs(),
      std::vector<std::pair<int32, int32> >());

  for (int32 pdf = 0; pdf < NumPdfs(); pdf++) {
    (*pdf_info)[pdf].push_back(std::make_pair(1, pdf));
  }
}

void SimpleHmm::FakeContextDependency::GetPdfInfo(
    const std::vector<int32> &phones,
    const std::vector<std::vector<std::pair<int32, int32> > > &pdf_class_pairs,
    std::vector<std::vector<std::vector<std::pair<int32, int32> > > > *pdf_info) const {
  KALDI_ASSERT(pdf_info);
  KALDI_ASSERT(phones.size() == 1 && phones[0] == 1);
  KALDI_ASSERT(pdf_class_pairs.size() == 2);
  
  pdf_info->resize(2);
  (*pdf_info)[1].resize(pdf_class_pairs[1].size());

  for (size_t j = 0; j < pdf_class_pairs[1].size(); j++) {
    int32 pdf_class = pdf_class_pairs[1][j].first,
          self_loop_pdf_class = pdf_class_pairs[1][j].second;
    KALDI_ASSERT(pdf_class == self_loop_pdf_class && 
                 pdf_class < NumPdfs());

    (*pdf_info)[1][j].push_back(std::make_pair(pdf_class, pdf_class));
  }
}

void SimpleHmm::Read(std::istream &is, bool binary) {
  TransitionModel::Read(is, binary);
  ctx_dep_.Init(NumPdfs());
  CheckSimpleHmm();
}

void SimpleHmm::CheckSimpleHmm() const {
  KALDI_ASSERT(NumPhones() == 1);
  KALDI_ASSERT(GetPhones()[0] == 1);
  const HmmTopology::TopologyEntry &entry = GetTopo().TopologyForPhone(1);
  for (int32 j = 0; j < static_cast<int32>(entry.size()); j++) {  // for each state...
    int32 forward_pdf_class = entry[j].forward_pdf_class,
          self_loop_pdf_class = entry[j].self_loop_pdf_class;
    KALDI_ASSERT(forward_pdf_class == self_loop_pdf_class &&
                 forward_pdf_class < NumPdfs());
  }
}

}  // end namespace kaldi

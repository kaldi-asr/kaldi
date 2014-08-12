// tree/context-dep-test.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/kaldi-io.h"

namespace kaldi {
void TestContextDep() {
  BaseFloat varFloor = 0.1;
  size_t dim = 1 + Rand() % 20;
  size_t nGauss = 1 + Rand() % 10;
  std::vector< GaussClusterable * > v(nGauss);
  for (size_t i = 0;i < nGauss;i++) {
    v[i] = new GaussClusterable(dim, varFloor);
  }
  for (size_t i = 0;i < nGauss;i++) {
    size_t nPoints = 1 + Rand() % 30;
    for (size_t j = 0;j < nPoints;j++) {
      BaseFloat post = 0.5 *(Rand()%3);
      Vector<BaseFloat> vec(dim);
      for (size_t k = 0;k < dim;k++) vec(k) = RandGauss();
      v[i]->AddStats(vec, post);
    }
  }
  for (size_t i = 0;i+1 < nGauss;i++) {
    BaseFloat like_before = (v[i]->Objf() + v[i+1]->Objf()) / (v[i]->Normalizer() + v[i+1]->Normalizer());
    Clusterable *tmp = v[i]->Copy();
    tmp->Add(*(v[i+1]));
    BaseFloat like_after = tmp->Objf() / tmp->Normalizer();
    std::cout << "Like_before = " << like_before <<", after = "<<like_after <<" over "<<tmp->Normalizer()<<" frames.\n";
    if (tmp->Normalizer() > 0.1)
      KALDI_ASSERT(like_after <= like_before);  // should get worse after combining stats.
    delete tmp;
  }
  for (size_t i = 0;i < nGauss;i++)
    delete v[i];
}

void TestMonophoneContextDependency() {
  std::set<int32> phones_set;
  for (size_t i = 1; i <= 20; i++) phones_set.insert(1 + Rand() % 30);
  std::vector<int32> phones;
  CopySetToVector(phones_set, &phones);
  std::vector<int32> phone2num_classes(1 + *std::max_element(phones.begin(), phones.end()));
  for (size_t i = 0; i < phones.size(); i++)
    phone2num_classes[phones[i]] = 3;
  ContextDependency *cd = MonophoneContextDependency(phones,
                                                     phone2num_classes);

  std::vector<std::vector<std::pair<int32, int32> > >  pdf_info;
  cd->GetPdfInfo(phones, phone2num_classes, &pdf_info);
  KALDI_ASSERT(pdf_info.size() == phones.size() * 3 &&
       pdf_info[Rand() % pdf_info.size()].size() == 1);
  delete cd;
}
// Also tests I/O of ContextDependency
void TestGenRandContextDependency() {
  bool binary = (Rand()%2 == 0);
  size_t num_phones = 1 + Rand() % 10;
  std::set<int32> phones_set;
  while (phones_set.size() < num_phones) phones_set.insert(Rand() % (num_phones + 5));
  std::vector<int32> phones;
  CopySetToVector(phones_set, &phones);
  bool ensure_all_covered = (Rand() % 2 == 0);
  std::vector<int32> phone2num_pdf_classes;
  ContextDependency *dep = GenRandContextDependency(phones,
                                                    ensure_all_covered,  // false == don't ensure all phones covered.
                                                    &phone2num_pdf_classes);
  // stuff here.
  const char *filename = "tmpf";
  {
    Output ko(filename, binary);
    std::ostream &outfile = ko.Stream();
    {  // Test GetPdfInfo
      std::vector<std::vector<std::pair<int32, int32> > > pdf_info;
      dep->GetPdfInfo(phones, phone2num_pdf_classes, &pdf_info);
      std::vector<bool> all_phones(phones.back()+1, false);  // making sure all covered.
      for (size_t i = 0; i < pdf_info.size(); i++) {
        KALDI_ASSERT(!pdf_info[i].empty());  // make sure pdf seen.
        for (size_t j = 0; j < pdf_info[i].size(); j++) {
          int32 idx = pdf_info[i][j].first;
          KALDI_ASSERT(static_cast<size_t>(idx) < all_phones.size());
          all_phones[pdf_info[i][j].first] = true;
        }
      }
      if (ensure_all_covered)
        for (size_t k = 0; k < phones.size(); k++) KALDI_ASSERT(all_phones[phones[k]]);
    }

    dep->Write(outfile, binary);
    ko.Close();
  }
  {
    bool binary_in;
    Input ki(filename, &binary_in);
    std::istream &infile = ki.Stream();
    ContextDependency dep2;
    dep2.Read(infile, binary_in);

    std::ostringstream ostr1, ostr2;
    dep->Write(ostr1, false);
    dep2.Write(ostr2, false);
    KALDI_ASSERT(ostr1.str() == ostr2.str());
  }

  delete dep;
  std::cout << "Note: any \"serious error\" warnings preceding this line are OK.\n";
}

} // end namespace kaldi

int main() {
  for (size_t i = 0;i < 10;i++) {
    kaldi::TestContextDep();
    kaldi::TestGenRandContextDependency();  // Also tests I/O of ContextDependency
    kaldi::TestMonophoneContextDependency();
  }
}

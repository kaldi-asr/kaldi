// hmm/simple-hmm-test.cc

// Copyright 2016  Vimal Manohar

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
#include "hmm/hmm-test-utils.h"

namespace kaldi {
namespace simple_hmm {


SimpleHmm *GenRandSimpleHmm() {
  std::vector<int32> phones;
  phones.push_back(1);

  std::vector<int32> num_pdf_classes;
  num_pdf_classes.push_back(rand() + 1);

  HmmTopology topo = GenRandTopology(phones, num_pdf_classes);

  SimpleHmm *model = new SimpleHmm(topo);

  return model;
}


void TestSimpleHmm() {

  SimpleHmm *model = GenRandSimpleHmm();

  bool binary = (rand() % 2 == 0);

  std::ostringstream os;
  model->Write(os, binary);

  SimpleHmm model2;
  std::istringstream is2(os.str());
  model2.Read(is2, binary);

  {
    std::ostringstream os1, os2;
    model->Write(os1, false);
    model2.Write(os2, false);
    KALDI_ASSERT(os1.str() == os2.str());
    KALDI_ASSERT(model->Compatible(model2));
  }
  delete model;
}


}  // end namespace simple_hmm
}  // end namespace kaldi


int main() {
  for (int i = 0; i < 2; i++)
    kaldi::TestSimpleHmm();
  KALDI_LOG << "Test OK.\n";
}



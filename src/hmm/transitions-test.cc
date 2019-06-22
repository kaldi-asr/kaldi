// hmm/transition-model-test.cc

// Copyright 2014  Johns Hopkins University

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

#include "hmm/transitions.h"
#include "hmm/hmm-test-utils.h"

namespace kaldi {


void TestTransitions() {
  Transitions *trans_model = GenRandTransitions(NULL);
  bool binary = (rand() % 2 == 0);

  std::ostringstream os;
  trans_model->Write(os, binary);

  Transitions trans_model2;
  std::istringstream is2(os.str());
  trans_model2.Read(is2, binary);

  {
    std::ostringstream os1, os2;
    trans_model->Write(os1, false);
    trans_model2.Write(os2, false);
    KALDI_ASSERT(os1.str() == os2.str());
    KALDI_ASSERT(*trans_model == trans_model2);
  }
  delete trans_model;
}

}

int main() {
  for (int i = 0; i < 2; i++)
    kaldi::TestTransitions();
  KALDI_LOG << "Test OK.\n";
}

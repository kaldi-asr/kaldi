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

#include "hmm/transition-model.h"

namespace kaldi {


void TestTransitionModel() {
  std::vector<int32> phones;
  phones.push_back(1);
  for (int32 i = 2; i < 20; i++)
    if (rand() % 2 == 0)
      phones.push_back(i);
  int32 N = 2 + rand() % 2, // context-size N is 2 or 3.
      P = rand() % N;  // Central-phone is random on [0, N)

  std::vector<int32> num_pdf_classes;

  ContextDependency *ctx_dep =
      GenRandContextDependencyLarge(phones, N, P,
                                    true, &num_pdf_classes);
  
  HmmTopology topo = GetDefaultTopology(phones);
  
  TransitionModel trans_model(*ctx_dep, topo);
  
  delete ctx_dep; // We won't need this further.
  ctx_dep = NULL;


  bool binary = (rand() % 2 == 0);
  
  std::ostringstream os;
  trans_model.Write(os, binary);

  TransitionModel trans_model2;
  std::istringstream is2(os.str());  
  trans_model2.Read(is2, binary);

  {
    std::ostringstream os1, os2;
    trans_model.Write(os1, false);
    trans_model2.Write(os2, false);
    KALDI_ASSERT(os1.str() == os2.str());
    KALDI_ASSERT(trans_model.Compatible(trans_model2));
  }

}
  
}

int main() {
  for (int i = 0; i < 2; i++)
    kaldi::TestTransitionModel();
  KALDI_LOG << "Test OK.\n";
}


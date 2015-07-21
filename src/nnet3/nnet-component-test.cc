// nnet3/nnet-component-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {

void TestNnetComponentIo(Component *c) {
  bool binary = (Rand() % 2 == 0);
  std::ostringstream os1;
  c->Write(os1, binary);
  std::istringstream is(os1.str());
  Component *c2 = Component::ReadNew(is, binary);
  std::ostringstream os2;
  c2->Write(os2, binary);
  if (!binary) {
    KALDI_ASSERT(os2.str() == os1.str());
  }
  delete c2;
  
}


void UnitTestNnetComponent() {
  for (int32 n = 0; n < 200; n++) {
    Component *c = GenerateRandomSimpleComponent();
    TestNnetComponentIo(c);
    // More tests here.
    delete c;
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  //SetVerboseLevel(2);

  UnitTestNnetComponent();

  KALDI_LOG << "Nnet component ntests succeeded.";

  return 0;
}

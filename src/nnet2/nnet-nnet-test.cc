// nnet2/nnet-nnet-test.cc

// Copyright 2014  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet2/nnet-nnet.h"

namespace kaldi {
namespace nnet2 {


void UnitTestNnet() {
  int32 input_dim = 40, output_dim = 500;
  Nnet *nnet = GenRandomNnet(input_dim, output_dim);

  bool binary = (rand() % 2 == 0);
  std::ostringstream os;
  nnet->Write(os, binary);
  Nnet nnet2;
  std::istringstream is(os.str());
  nnet2.Read(is, binary);

  std::ostringstream os2;
  nnet2.Write(os2, binary);

  KALDI_ASSERT(os2.str() == os.str());
  delete nnet;
}

} // namespace nnet2
} // namespace kaldi

#include "matrix/matrix-functions.h"


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet2;

  UnitTestNnet();
  return 0;
}
  

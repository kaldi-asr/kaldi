// nnet3/nnet-nnet-test.cc

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
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {


void UnitTestNnetIo() {
  for (int32 n = 0; n < 100; n++) {
    struct NnetGenerationOptions gen_config;
    
    bool binary = (Rand() % 2 == 0);
    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    std::istringstream is(configs[0]);
    nnet.ReadConfig(is);

    std::ostringstream os;
    nnet.Write(os, binary);
    const std::string &original_output = os.str();
    std::istringstream nnet_is(original_output);
    nnet.Read(nnet_is, binary);
    std::istringstream nnet_is2(original_output);
    Nnet nnet2;
    nnet2.Read(nnet_is2, binary);
      
    std::ostringstream os2, os3;
    nnet.Write(os2, binary);
    
    nnet2.Write(os3, binary);
    if (binary) {
      KALDI_ASSERT(os2.str() == original_output);
      KALDI_ASSERT(os3.str() == original_output);
    }
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  UnitTestNnetIo();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}

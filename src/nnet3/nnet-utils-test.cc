// nnet3/nnet-utils-test.cc

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


void UnitTestNnetContext() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationOptions gen_config;
    
    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    std::istringstream is(configs[0]);
    nnet.ReadConfig(is);

    // this test doesn't really test anything except that it runs;
    // we manually inspect the output.
    int32 left_context, right_context;
    ComputeSimpleNnetContext(nnet, &left_context, &right_context);
    KALDI_LOG << "Left,right-context= " << left_context << ","
              << right_context << " for config: " << configs[0];
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  SetVerboseLevel(2);

  UnitTestNnetContext();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}

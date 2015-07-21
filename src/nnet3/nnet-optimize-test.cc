// nnet3/nnet-optimize-test.cc

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
#include "nnet3/nnet-compile.h"
#include "nnet3/nnet-analyze.h"
#include "nnet3/nnet-test-utils.h"
#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace nnet3 {


void UnitTestNnetOptimize() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationConfig gen_config;
    
    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    for (size_t j = 0; j < configs.size(); j++) {
      KALDI_LOG << "Input config[" << j << "] is: " << configs[j];
      std::istringstream is(configs[j]);
      nnet.ReadConfig(is);
    }

    ComputationRequest request;
    std::vector<Matrix<BaseFloat> > inputs;
    ComputeExampleComputationRequestSimple(nnet, &request, &inputs);
    
    NnetComputation computation;
    Compiler compiler(request, nnet);

    CompilerOptions opts;
    compiler.CreateComputation(opts, &computation);
    {
      std::ostringstream os;
      computation.Print(os, nnet);
      KALDI_LOG << "Generated computation is: " << os.str();
    }
    CheckComputationConfig check_config;
    // we can do the rewrite check since it's before optimization.
    check_config.check_rewrite = true;  
    ComputationChecker checker(check_config, nnet, request, computation);
    checker.Check();

    NnetOptimizeConfig opt_config;
    opt_config.initialize_undefined = false;
    opt_config.propagate_in_place = false;
    opt_config.backprop_in_place = false;
    opt_config.remove_assignments = false;
    
    Optimize(opt_config, nnet, request, &computation);
    {
      std::ostringstream os;
      computation.Print(os, nnet);
      KALDI_LOG << "Optimized computation is: " << os.str();
    }

    {
      CheckComputationConfig check_config;
      ComputationChecker checker(check_config, nnet, request, computation);
      checker.Check();
    }
    {
      Analyzer analyzer;
      analyzer.Init(nnet, computation);
      KALDI_LOG << "Matrix accesses are: ";
      PrintMatrixAccesses(std::cerr, analyzer.matrix_accesses);
      KALDI_LOG << "Command attributes are: ";
      PrintCommandAttributes(std::cerr, analyzer.command_attributes);
    }
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  //SetVerboseLevel(2);

  UnitTestNnetOptimize();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}

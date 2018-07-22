// nnet3/nnet-analyze-test.cc

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

namespace kaldi {
namespace nnet3 {

std::string PrintCommand(int32 num_commands,
                         int32 command) {
  std::ostringstream os;
  if (command < 0 || command >= num_commands)
    os << command;
  else
    os << 'c' << command;
  return os.str();
}


void UnitTestNnetAnalyze() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationOptions gen_config;

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

    std::ostringstream os;
    computation.Print(os, nnet);
    KALDI_LOG << "Generated computation is: " << os.str();

    CheckComputationOptions check_config;
    // we can do the rewrite check since it's before optimization.
    check_config.check_rewrite = true;
    ComputationChecker checker(check_config, nnet, computation);
    checker.Check();

    Analyzer analyzer;
    analyzer.Init(nnet, computation);
    ComputationAnalysis analysis(computation, analyzer);
    // The following output is to be eyeballed by a person.
    std::vector<std::string> submatrix_strings;
    computation.GetSubmatrixStrings(nnet, &submatrix_strings);
    int32 nc = computation.commands.size();
    for (int32 n = 0; n < 30; n++) {
      int32 s = RandInt(1, computation.submatrices.size() - 1);
      int32 c = RandInt(0, nc - 1);
      KALDI_LOG << "First nontrivial access of submatrix " << submatrix_strings[s]
                << " is command "
                << PrintCommand(nc, analysis.FirstNontrivialAccess(s));
      KALDI_LOG << "Last access of submatrix " << submatrix_strings[s]
                << " is command " << PrintCommand(nc, analysis.LastAccess(s));
      KALDI_LOG << "Last write access of submatrix " << submatrix_strings[s]
                << " is command " << PrintCommand(nc, analysis.LastWriteAccess(s));
      KALDI_LOG << "Data present in " << submatrix_strings[s]
                << " at command " << c << " is invalidated at command "
                << PrintCommand(nc, analysis.DataInvalidatedCommand(c, s));
    }
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  //SetVerboseLevel(2);

  UnitTestNnetAnalyze();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}

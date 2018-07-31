// nnet3/nnet-compile-test.cc

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
#include "nnet3/nnet-compile-looped.h"
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {


void UnitTestNnetCompile() {
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
    KALDI_LOG << "Computation request is:";
    request.Print(std::cerr);

    NnetComputation computation;
    Compiler compiler(request, nnet);

    CompilerOptions opts;
    compiler.CreateComputation(opts, &computation);

    std::ostringstream os;
    computation.Print(os, nnet);
    KALDI_LOG << "Generated computation is: " << os.str();
  }
}


// this tests compilation where there are more than one
// computation-request... this is to test some of the
// low-level utilities that will be used in looped computation.
void UnitTestNnetCompileMulti() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationOptions gen_config;
    gen_config.allow_use_of_x_dim = false;

    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    for (size_t j = 0; j < configs.size(); j++) {
      KALDI_LOG << "Input config[" << j << "] is: " << configs[j];
      std::istringstream is(configs[j]);
      nnet.ReadConfig(is);
    }

    ComputationRequest request1, request2;
    std::vector<Matrix<BaseFloat> > inputs1, inputs2;
    ComputeExampleComputationRequestSimple(nnet, &request1, &inputs1);
    ComputeExampleComputationRequestSimple(nnet, &request2, &inputs2);


    KALDI_LOG << "Computation request 1 is:";
    request1.Print(std::cerr);
    KALDI_LOG << "Computation request 2 is:";
    request2.Print(std::cerr);

    std::vector<const ComputationRequest*> requests;
    request2.store_component_stats = request1.store_component_stats;
    request1.need_model_derivative = false;
    request2.need_model_derivative = false;
    requests.push_back(&request1);
    requests.push_back(&request2);

    // set all the x indexes to 1 for request 2 (they would otherwise
    // be zero).  This ensures that there is no overlap
    // between the inputs and outputs on the two requests.
    for (int32 i = 0; i < request2.inputs.size(); i++)
      for (int32 j = 0; j < request2.inputs[i].indexes.size(); j++)
        request2.inputs[i].indexes[j].x = 1;
    for (int32 i = 0; i < request2.outputs.size(); i++)
      for (int32 j = 0; j < request2.outputs[i].indexes.size(); j++)
        request2.outputs[i].indexes[j].x = 1;


    NnetComputation computation;
    Compiler compiler(requests, nnet);

    CompilerOptions opts;
    compiler.CreateComputation(opts, &computation);

    std::ostringstream os;
    computation.Print(os, nnet);
    KALDI_LOG << "Generated computation is: " << os.str();
  }
}



void UnitTestNnetCompileLooped() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationOptions gen_config;
    gen_config.allow_ivector = true;

    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    for (size_t j = 0; j < configs.size(); j++) {
      KALDI_LOG << "Input config[" << j << "] is: " << configs[j];
      std::istringstream is(configs[j]);
      nnet.ReadConfig(is);
    }

    ComputationRequest request1, request2, request3;
    int32 chunk_size_min = RandInt(5, 15);
    int32 frame_subsampling_factor = RandInt(1, 3),
        extra_left_context_begin = RandInt(0, 10),
        extra_right_context = RandInt(0, 10),
        num_sequences = RandInt(1, 2);
    int32 chunk_size = GetChunkSize(nnet, frame_subsampling_factor,
                                    chunk_size_min),
        ivector_period = chunk_size;



    ModifyNnetIvectorPeriod(ivector_period, &nnet);
    KALDI_LOG << "Nnet info after modifying ivector period is: "
              << nnet.Info();
    CreateLoopedComputationRequestSimple(
        nnet, chunk_size, frame_subsampling_factor,
        ivector_period, extra_left_context_begin, extra_right_context,
        num_sequences, &request1, &request2, &request3);

    KALDI_LOG << "Computation request 1 is:";
    request1.Print(std::cerr);
    KALDI_LOG << "Computation request 2 is:";
    request2.Print(std::cerr);
    KALDI_LOG << "Computation request 3 is:";
    request3.Print(std::cerr);

    NnetOptimizeOptions optimize_opts;
    // todo: set optimize-looped=true.
    NnetComputation computation;
    CompileLooped(nnet, optimize_opts,
                  request1, request2, request3,
                  &computation);
    KALDI_LOG << "Compiled looped computation is ";
    computation.Print(std::cerr, nnet);
  }
}



} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  SetVerboseLevel(4);

  UnitTestNnetCompileLooped();
  UnitTestNnetCompile();
  UnitTestNnetCompileMulti();


  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}

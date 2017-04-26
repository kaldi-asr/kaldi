// nnet3/nnet-compute-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)
//           2015  Xiaohui Zhang

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
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/decodable-simple-looped.h"

namespace kaldi {
namespace nnet3 {

void UnitTestNnetComputationIo(NnetComputation *computation) {
  bool binary = (Rand() % 2 == 0);
  std::ostringstream os;
  computation->Write(os, binary);
  const std::string &original_output = os.str();
  std::istringstream computation_is(original_output);
  computation->Read(computation_is, binary);
  std::istringstream computation_is2(original_output);
  NnetComputation computation2;
  computation2.Read(computation_is2, binary);

  std::ostringstream os2, os3;
  computation->Write(os2, binary);
  computation2.Write(os3, binary);

  if (binary) {
    if (!(os2.str() == original_output)) {
      KALDI_ERR << "Outputs differ for computation";
    }
  }
}

void UnitTestComputationRequestIo(ComputationRequest *request) {
  bool binary = (Rand() % 2 == 0);
  std::ostringstream os;
  request->Write(os, binary);
  const std::string &original_output = os.str();
  std::istringstream request_is(original_output);
  request->Read(request_is, binary);
  std::istringstream request_is2(original_output);
  ComputationRequest request2;
  request2.Read(request_is2, binary);

  std::ostringstream os2, os3;
  request->Write(os2, binary);
  request2.Write(os3, binary);
  KALDI_ASSERT(*request == request2);

  if (binary) {
    KALDI_ASSERT(os2.str() == original_output);
    KALDI_ASSERT(os3.str() == original_output);
  }
}

// this checks that a couple of different decodable objects give the same
// answer.
void TestNnetDecodable(Nnet *nnet) {
  int32 num_frames = 5 + RandInt(1, 100),
      input_dim = nnet->InputDim("input"),
      output_dim = nnet->OutputDim("output"),
      ivector_dim = std::max<int32>(0, nnet->InputDim("ivector"));
  Matrix<BaseFloat> input(num_frames, input_dim);

  SetBatchnormTestMode(true, nnet);
  SetDropoutTestMode(true, nnet);

  input.SetRandn();
  Vector<BaseFloat> ivector(ivector_dim);
  ivector.SetRandn();

  Vector<BaseFloat> priors(RandInt(0, 1) == 0 ? output_dim : 0);
  if (priors.Dim() != 0) {
    priors.SetRandn();
    priors.ApplyExp();
  }

  Matrix<BaseFloat> output1(num_frames, output_dim),
      output2(num_frames, output_dim);

  {
    NnetSimpleComputationOptions opts;
    opts.frames_per_chunk = RandInt(5, 25);
    CachingOptimizingCompiler compiler(*nnet);
    DecodableNnetSimple decodable(opts, *nnet, priors, input, &compiler,
                                  (ivector_dim != 0 ? &ivector : NULL));
    for (int32 t = 0; t < num_frames; t++) {
      SubVector<BaseFloat> row(output1, t);
      decodable.GetOutputForFrame(t, &row);
    }
  }

  {
    NnetSimpleLoopedComputationOptions opts;
    // caution: this may modify nnet, by changing how it consumes iVectors.
    DecodableNnetSimpleLoopedInfo info(opts, priors, nnet);
    DecodableNnetSimpleLooped decodable(info, input,
                                        (ivector_dim != 0 ? &ivector : NULL));
    for (int32 t = 0; t < num_frames; t++) {
      SubVector<BaseFloat> row(output2, t);
      decodable.GetOutputForFrame(t, &row);
    }
  }


  if (!NnetIsRecurrent(*nnet) &&
      nnet->Info().find("statistics-extraction") == std::string::npos &&
      nnet->Info().find("TimeHeightConvolutionComponent") == std::string::npos) {
    // this equivalence will not hold for recurrent nnets, or those that
    // have the statistics-extraction/statistics-pooling layers,
    // or in general for nnets with convolution components (because these
    // might have 'optional' context if required-time-offsets != time-offsets.
    for (int32 t = 0; t < num_frames; t++) {
      SubVector<BaseFloat> row1(output1, t),
          row2(output2, t);
      KALDI_ASSERT(row1.ApproxEqual(row2));
    }
  }
}

void UnitTestNnetCompute() {
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
    {
      std::ostringstream os;
      computation.Print(os, nnet);
      KALDI_LOG << "Generated computation is: " << os.str();
      UnitTestNnetComputationIo(&computation);
      UnitTestComputationRequestIo(&request);
    }
    CheckComputationOptions check_config;
    // we can do the rewrite check since it's before optimization.
    check_config.check_rewrite = true;
    ComputationChecker checker(check_config, nnet, computation);
    checker.Check();

    if (RandInt(0, 1) == 0) {
      NnetOptimizeOptions opt_config;

      Optimize(opt_config, nnet,
               MaxOutputTimeInRequest(request),
               &computation);
      {
        std::ostringstream os;
        computation.Print(os, nnet);
        KALDI_LOG << "Optimized computation is: " << os.str();
      }
    }

    NnetComputeOptions compute_opts;
    if (RandInt(0, 1) == 0)
      compute_opts.debug = true;

    computation.ComputeCudaIndexes();
    NnetComputer computer(compute_opts,
                          computation,
                          nnet,
                          &nnet);
    // provide the input to the computation.
    for (size_t i = 0; i < request.inputs.size(); i++) {
      CuMatrix<BaseFloat> temp(inputs[i]);
      KALDI_LOG << "Input sum is " << temp.Sum();
      computer.AcceptInput(request.inputs[i].name, &temp);
    }
    computer.Run();
    const CuMatrixBase<BaseFloat> &output(computer.GetOutput("output"));

    KALDI_LOG << "Output sum is " << output.Sum();
    CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
    output_deriv.SetRandn();
    // output_deriv sum won't be informative so don't print it.
    if (request.outputs[0].has_deriv) {
      computer.AcceptInput("output", &output_deriv);
      computer.Run();
      for (size_t i = 0; i < request.inputs.size(); i++) {
        if (request.inputs[i].has_deriv) {
          const CuMatrixBase<BaseFloat> &in_deriv =
              computer.GetOutput(request.inputs[i].name);
          KALDI_LOG << "Input-deriv sum for input '"
                    << request.inputs[i].name << "' is " << in_deriv.Sum();
        }
      }
    }
    TestNnetDecodable(&nnet);
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  // uncommenting the following activates extra checks during optimization, that
  // can help narrow down the source of problems.
  // SetVerboseLevel(4);


  for (kaldi::int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    UnitTestNnetCompute();
  }

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}

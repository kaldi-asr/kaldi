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
#include "nnet3/nnet-compute.h"

namespace kaldi {
namespace nnet3 {

// Run the test wothout optimizations and with optimizations specified by the
// parameter. Only print warnings; we'll fail the whole test later.
static bool UnitTestNnetOptimizeWithOptions(NnetOptimizeOptions opt_config) {
  //opt_config.convert_addition = false;
  //opt_config.remove_assignments = false;
  //opt_config.move_sizing_commands = false;
  //opt_config.allocate_from_other = false;

  srand(0);  // Every run must be deterministic.
  for (int32 n = 0; n < 40; n++) {
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
    }
    CheckComputationOptions check_config;
    // we can do the rewrite check since it's before optimization.
    check_config.check_rewrite = true;
    ComputationChecker checker(check_config, nnet, computation);
    checker.Check();

    NnetComputation computation_opt(computation);

    {
      Optimize(opt_config, nnet, request, &computation_opt);
      std::ostringstream os;
      computation_opt.Print(os, nnet);
      KALDI_LOG << "Optimized computation is: " << os.str();
    }

    NnetComputeOptions compute_opts;
    if (RandInt(0, 1) == 0)
      compute_opts.debug = true;

    computation.ComputeCudaIndexes();
    computation_opt.ComputeCudaIndexes();
    Nnet nnet_to_update(nnet);  // copy of the nnet that we update...  needed to
                                // test the consolidation of backprop commands,
                                // otherwise the optimized and non-optimized
                                // comptuations differ.
    bool is_gradient = true;  // with natural gradient, the consolidation would
                              // affect the final model params -> test just the
                              // gradient.
    SetZero(is_gradient, &nnet_to_update);

    NnetComputer computer(compute_opts,
                          computation,
                          nnet,
                          &nnet_to_update);

    Nnet nnet_opt(nnet);  // copy of the nnet for the optimized computation.
                          // necessary in case backprop changes parameters.
    Nnet nnet_opt_to_update(nnet_opt);
    SetZero(is_gradient, &nnet_opt_to_update);

    // NnetComputer for the optimized version of the computation.
    NnetComputer computer_opt(compute_opts,
                              computation_opt,
                              nnet_opt,
                              &nnet_opt_to_update);

    // provide the input to the computations.
    for (size_t i = 0; i < request.inputs.size(); i++) {
      CuMatrix<BaseFloat> temp(inputs[i]);
      KALDI_LOG << "Input sum is " << temp.Sum();
      computer.AcceptInput(request.inputs[i].name, &temp);
      CuMatrix<BaseFloat> temp2(inputs[i]);
      computer_opt.AcceptInput(request.inputs[i].name, &temp2);
    }
    KALDI_LOG << "Running non-optimized forward computation";
    computer.Forward();
    KALDI_LOG << "Running optimized forward computation";
    computer_opt.Forward();

    const CuMatrixBase<BaseFloat> &output(computer.GetOutput("output"));
    KALDI_LOG << "Output sum (not optimized) is " << output.Sum();
    const CuMatrixBase<BaseFloat> &output_opt(computer_opt.GetOutput("output"));
    KALDI_LOG << "Output sum (optimized) is " << output_opt.Sum();
    if (!ApproxEqual(output, output_opt)) {
      KALDI_WARN << "Non-optimized and optimized versions of the computation give "
                 << "different outputs.";
      return false;
    }

    CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
    output_deriv.SetRandn();
    CuMatrix<BaseFloat> output_deriv_opt(output_deriv);

    if (request.outputs[0].has_deriv) {
      computer.AcceptOutputDeriv("output", &output_deriv);
      computer_opt.AcceptOutputDeriv("output", &output_deriv_opt);
    }

    KALDI_LOG << "Running non-optimized backward computation";
    computer.Backward();
    KALDI_LOG << "Running optimized backward computation";
    computer_opt.Backward();
    for (size_t i = 0; i < request.inputs.size(); i++) {
      if (request.inputs[i].has_deriv) {
        const CuMatrixBase<BaseFloat> &in_deriv =
            computer.GetInputDeriv(request.inputs[i].name);
        const CuMatrixBase<BaseFloat> &in_deriv_opt =
            computer_opt.GetInputDeriv(request.inputs[i].name);
        KALDI_LOG << "Input-deriv sum for input '" << request.inputs[i].name
                  << "' (non-optimized) is " << in_deriv.Sum();
        KALDI_LOG << "Input-deriv sum for input '" << request.inputs[i].name
                  << "' (optimized) is " << in_deriv_opt.Sum();
        if (!ApproxEqual(in_deriv, in_deriv_opt)) {
          KALDI_WARN << "Non-optimized and optimized versions of the "
                     << "computation give different input-derivs.";
          return false;
        }
      }
    }

    if (!NnetParametersAreIdentical(nnet_to_update,
                                    nnet_opt_to_update, 1.0e-05)) {
      KALDI_WARN << "Neural networks differ after training, between "
                 << "optimized and non-optimized computation.";
      return false;
    }
  }
  return true;
}


// This test runs the computation with and without optimization, and checks that
// the outputs are the same.
static void UnitTestNnetOptimize() {
  NnetOptimizeOptions optimize_all;
  // randomly sometimes set min_deriv and max_deriv to small/large values,
  // which will cause some of the LimitDerivativeTimes() code to be called
  // (without really changing anything).
  if (RandInt(0, 3) == 0) optimize_all.min_deriv_time = -200;
  if (RandInt(0, 3) == 0) optimize_all.max_deriv_time = 1000;

  // this is useful for debugging as it removes nans:
  // optimize_all.initialize_undefined = false;
  bool success = UnitTestNnetOptimizeWithOptions(optimize_all);
  if (success)
    return;

  // Test failed with full optimization. Slowly retry with various
  // optimizations switched off.
  NnetOptimizeOptions optimize = optimize_all;
  optimize.propagate_in_place = false;
  bool succ_no_propagate_in_place = UnitTestNnetOptimizeWithOptions(optimize);

  optimize = optimize_all;
  optimize.backprop_in_place = false;
  bool succ_no_backprop_in_place = UnitTestNnetOptimizeWithOptions(optimize);

  optimize = optimize_all;
  optimize.remove_assignments = false;
  bool succ_no_remove_assignments = UnitTestNnetOptimizeWithOptions(optimize);

  optimize = optimize_all;
  optimize.initialize_undefined = false;
  bool succ_no_initialize_undefined = UnitTestNnetOptimizeWithOptions(optimize);

  optimize = optimize_all;
  optimize.move_sizing_commands = false;
  bool succ_no_move_sizing_commands = UnitTestNnetOptimizeWithOptions(optimize);

#define KALDI_SUCCFAIL(b) ((b) ? "SUCCESS" : "FAILURE")
  KALDI_ERR
    << "Test failed with all optimizations enabled. Retried test with the "
    << "following optimizations turned off:"
    << "\n  propagate_in_place   ... " << KALDI_SUCCFAIL(succ_no_propagate_in_place)
    << "\n  backprop_in_place    ... " << KALDI_SUCCFAIL(succ_no_backprop_in_place)
    << "\n  remove_assignments   ... " << KALDI_SUCCFAIL(succ_no_remove_assignments)
    << "\n  initialize_undefined ... " << KALDI_SUCCFAIL(succ_no_initialize_undefined)
    << "\n  move_sizing_commands ... " << KALDI_SUCCFAIL(succ_no_move_sizing_commands);
#undef KALDI_SUCCFAIL
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  //SetVerboseLevel(2);

#if HAVE_CUDA == 1
  CuDevice::Instantiate().SetDebugStrideMode(true);
  CuDevice::Instantiate().SelectGpuId("no");
  UnitTestNnetOptimize();
  CuDevice::Instantiate().SelectGpuId("yes");
#endif
  UnitTestNnetOptimize();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}

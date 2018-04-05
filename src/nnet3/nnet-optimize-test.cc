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

// Run the test without optimizations and with optimizations specified by the
// configs (the optimized version is done with class CachingOptimizingCompiler).
// Only print warnings; we'll fail the whole test later.
static bool UnitTestNnetOptimizeWithOptions(int32 srand_seed,
                                            NnetOptimizeOptions opt_config,
                                            CachingOptimizingCompilerOptions compiler_config) {

  //opt_config.convert_addition = false;
  //opt_config.remove_assignments = false;
  //opt_config.move_sizing_commands = false;
  //opt_config.allocate_from_other = false;

  srand(srand_seed);  // so that we can compare between differnt optimization types
                      // with the randomly generated network staying the same.

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
    KALDI_LOG << "Generated computation with no optimization or shortcut is: " << os.str();
  }
  CheckComputationOptions check_config;
  // we can do the rewrite check since it's before optimization.
  check_config.check_rewrite = true;
  ComputationChecker checker(check_config, nnet, computation);
  checker.Check();

  CachingOptimizingCompiler opt_compiler(nnet, opt_config, compiler_config);

  const NnetComputation &computation_opt = *opt_compiler.Compile(request);

  {
    std::ostringstream os;
    computation_opt.Print(os, nnet);
    KALDI_LOG << "Optimized computation is: " << os.str();
  }

  NnetComputeOptions compute_opts;
  if (RandInt(0, 1) == 0)
    compute_opts.debug = true;

  computation.ComputeCudaIndexes();
  // computation_opt has already had this function called.

  Nnet nnet_to_update(nnet);  // copy of the nnet that we update...  needed to
  // test the consolidation of backprop commands,
  // otherwise the optimized and non-optimized
  // comptuations differ.
  ScaleNnet(0.0, &nnet_to_update);
  // with natural gradient, the consolidation would affect the final model
  // params -> test just the gradient.
  SetNnetAsGradient(&nnet_to_update);

  NnetComputer computer(compute_opts,
                        computation,
                        nnet,
                        &nnet_to_update);

  Nnet nnet_opt(nnet);  // copy of the nnet for the optimized computation.
  // necessary in case backprop changes parameters.
  Nnet nnet_opt_to_update(nnet_opt);
  ScaleNnet(0.0, &nnet_opt_to_update);
  SetNnetAsGradient(&nnet_opt_to_update);

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
  srand(srand_seed);
  ResetGenerators(&nnet);
  computer.Run();
  KALDI_LOG << "Running optimized forward computation";
  srand(srand_seed);
  ResetGenerators(&nnet_opt);
  computer_opt.Run();

  const CuMatrixBase<BaseFloat> &output(computer.GetOutput("output"));
  KALDI_LOG << "Output sum (not optimized) is " << output.Sum();
  const CuMatrixBase<BaseFloat> &output_opt(computer_opt.GetOutput("output"));
  KALDI_LOG << "Output sum (optimized) is " << output_opt.Sum();
  if (!ApproxEqual(output, output_opt)) {
    KALDI_WARN << "Non-optimized and optimized versions of the computation give "
               << "different outputs: " << output << " vs. " << output_opt;
    return false;
  }

  CuMatrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
  output_deriv.SetRandn();
  CuMatrix<BaseFloat> output_deriv_opt(output_deriv);

  if (request.outputs[0].has_deriv) {
    computer.AcceptInput("output", &output_deriv);
    computer_opt.AcceptInput("output", &output_deriv_opt);

    KALDI_LOG << "Running non-optimized backward computation";
    computer.Run();
    KALDI_LOG << "Running optimized backward computation";
    computer_opt.Run();
    for (size_t i = 0; i < request.inputs.size(); i++) {
      if (request.inputs[i].has_deriv) {
        const CuMatrixBase<BaseFloat> &in_deriv =
            computer.GetOutput(request.inputs[i].name);
        const CuMatrixBase<BaseFloat> &in_deriv_opt =
            computer_opt.GetOutput(request.inputs[i].name);
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
  }

  if (!NnetParametersAreIdentical(nnet_to_update,
                                  nnet_opt_to_update, 1.0e-05)) {
    KALDI_WARN << "Neural networks differ after training, between "
               << "optimized and non-optimized computation.";
    return false;
  } else {
    return true;
  }
}


// This test runs the computation with and without optimization, and checks that
// the outputs are the same.
static void UnitTestNnetOptimizeInternal(int32 srand_seed) {
  NnetOptimizeOptions optimize_all;
  CachingOptimizingCompilerOptions compiler_all;

  // randomly sometimes set min_deriv and max_deriv to small/large values,
  // which will cause some of the LimitDerivativeTimes() code to be called
  // (without really changing anything).
  if (RandInt(0, 3) == 0) optimize_all.min_deriv_time = -200;
  if (RandInt(0, 3) == 0) optimize_all.max_deriv_time = 1000;

  // this is useful for debugging as it removes nans:
  // optimize_all.initialize_undefined = false;
  bool success = UnitTestNnetOptimizeWithOptions(srand_seed, optimize_all,
                                                 compiler_all);
  if (success)
    return;

  // Test failed with full optimization. Slowly retry with various
  // optimizations switched off.
  NnetOptimizeOptions optimize = optimize_all;
  CachingOptimizingCompilerOptions compiler = compiler_all;


  compiler.use_shortcut = false;
  bool succ_no_shortcut = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                          compiler);
  compiler = compiler_all;


  optimize.propagate_in_place = false;
  bool succ_no_propagate_in_place = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                                    compiler);
  optimize = optimize_all;

  optimize.backprop_in_place = false;
  bool succ_no_backprop_in_place = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                                   compiler);
  optimize = optimize_all;

  optimize.optimize_row_ops = false;
  bool succ_no_row_ops = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                         compiler);
  optimize = optimize_all;

  optimize.convert_addition = false;
  bool succ_no_convert_addition = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                                  compiler);
  optimize = optimize_all;

  optimize.remove_assignments = false;
  bool succ_no_remove_assignments = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                                    compiler);
  optimize = optimize_all;

  optimize.initialize_undefined = false;
  bool succ_no_initialize_undefined = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                                      compiler);
  optimize = optimize_all;

  optimize.allocate_from_other = false;
  bool succ_no_allocate_from_other = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                                     compiler);
  optimize = optimize_all;

  optimize.move_sizing_commands = false;
  bool succ_no_move_sizing_commands = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                                      compiler);
  optimize = optimize_all;

  optimize.snip_row_ops = false;
  bool succ_no_snip_row_ops = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                              compiler);
  optimize = optimize_all;


  optimize.min_deriv_time = std::numeric_limits<int32>::min();
  optimize.max_deriv_time = std::numeric_limits<int32>::max();
  optimize.max_deriv_time_relative = std::numeric_limits<int32>::max();
  bool succ_no_deriv_time = UnitTestNnetOptimizeWithOptions(srand_seed, optimize,
                                                            compiler);
  optimize = optimize_all;


#define KALDI_SUCCFAIL(b) ((b) ? "SUCCESS" : "FAILURE")
  KALDI_ERR
    << "Test failed with all optimizations enabled. Retried test with the "
    << "following optimizations turned off:"
    << "\n  use_shortcut         ... " << KALDI_SUCCFAIL(succ_no_shortcut)
    << "\n  propagate_in_place   ... " << KALDI_SUCCFAIL(succ_no_propagate_in_place)
    << "\n  backprop_in_place    ... " << KALDI_SUCCFAIL(succ_no_backprop_in_place)
    << "\n  optimize_row_ops     ... " << KALDI_SUCCFAIL(succ_no_row_ops)
    << "\n  convert_addition     ... " << KALDI_SUCCFAIL(succ_no_convert_addition)
    << "\n  remove_assignments   ... " << KALDI_SUCCFAIL(succ_no_remove_assignments)
    << "\n  initialize_undefined ... " << KALDI_SUCCFAIL(succ_no_initialize_undefined)
    << "\n  allocate_from_other  ... " << KALDI_SUCCFAIL(succ_no_allocate_from_other)
    << "\n  move_sizing_commands ... " << KALDI_SUCCFAIL(succ_no_move_sizing_commands)
    << "\n  snip_row_ops         ... " << KALDI_SUCCFAIL(succ_no_snip_row_ops)
    << "\n  no_deriv_time        ... " << KALDI_SUCCFAIL(succ_no_deriv_time);
#undef KALDI_SUCCFAIL
}

static void UnitTestNnetOptimize() {
  for (int32 srand_seed = 0; srand_seed < 40; srand_seed++) {
    KALDI_LOG << "About to run UnitTestNnetOptimizeInternal with srand_seed = "
              << srand_seed;
    UnitTestNnetOptimizeInternal(srand_seed);
  }
}



} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  SetVerboseLevel(3);

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

// nnet3/nnet-derivative-test.cc

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


// This test makes sure that the model-derivatives are correct.
void UnitTestNnetModelDerivatives() {
  int32 num_tries = 20, num_success = 0;
  for (int32 n = 0; n < num_tries; n++) {
    struct NnetGenerationOptions gen_config;
    //gen_config.allow_nonlinearity = false;
    //gen_config.allow_recursion = false;
    //gen_config.allow_final_nonlinearity = true;
    bool allow_optimization = true;
    
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

    // make sure that a model-derivative is requested, and an output-derivative
    // is supplied.
    request.need_model_derivative = true;
    request.outputs[0].has_deriv = true;
    // whether input-derivatives are required or not does not matter,
    // so leave it as it is in that regard.
    
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
    ComputationChecker checker(check_config, nnet, request, computation);
    checker.Check();
    
    if (RandInt(0, 3) != 0 && allow_optimization) {
      NnetOptimizeOptions opt_config;
      Optimize(opt_config, nnet, request, &computation);
      std::ostringstream os;
      computation.Print(os, nnet);
      KALDI_LOG << "Optimized computation is: " << os.str();
    }

    NnetComputeOptions compute_opts;
    if (RandInt(0, 1) == 0)
      compute_opts.debug = true;
    computation.ComputeCudaIndexes();


    Nnet nnet_deriv(nnet);
    bool is_gradient = true;
    SetZero(is_gradient, &nnet_deriv);  // forces "simple" update and unit
                                        // learning rate.
    
    int32 num_directions = 4;  // must be >= 1.  Best if it's >1, will reduce
                               // the probability of random failures.

    // the order of these vectors is:
    // [ un-perturbed, perturbed-1, perturbed-2, perturbed-3 ].
    std::vector<BaseFloat> measured_objf(num_directions + 1, 0.0),
        predicted_objf_change(num_directions + 1, 0.0);
    BaseFloat delta = 5.0e-04;

    // output_deriv is the derivative of the objective function w.r.t. the
    // (single) output.  We make the objf a linear function of the output and
    // just set the output_deriv to be a random matrix, which defines the
    // objective function.
    CuMatrix<BaseFloat> output_deriv;
    output_deriv.Resize(request.outputs[0].indexes.size(),
                        nnet.OutputDim("output"));
    output_deriv.SetRandn();

    // pass 0 is the forward pass with the un-perturbed model.
    // Other passes are with various differently-perturbed versions of
    // the model.
    for (int32 pass = 0; pass <= num_directions; pass++) {
      Nnet nnet_copy(nnet);
      if (pass > 0)
        PerturbParams(delta, &nnet_copy);

      NnetComputer computer(compute_opts,
                            computation,
                            nnet_copy,
                            (pass == 0 ? &nnet_deriv : &nnet_copy));


      // provide the input to the computation.
      for (size_t i = 0; i < request.inputs.size(); i++) {
        CuMatrix<BaseFloat> temp(inputs[i]);
        computer.AcceptInput(request.inputs[i].name, &temp);
      }
      
      KALDI_LOG << "Running forward computation";
      computer.Forward();
      
      const CuMatrixBase<BaseFloat> &output(computer.GetOutput("output"));
      KALDI_LOG << "Output sum for pass " << pass << " is " << output.Sum();
      BaseFloat objf = TraceMatMat(output, output_deriv, kTrans);
      measured_objf[pass] = objf;

      if (pass == 0) {
        // we need to do the backward computation (to get the model derivative)
        CuMatrix<BaseFloat> temp(output_deriv);
        computer.AcceptOutputDeriv("output", &temp);
        KALDI_LOG << "Running backward computation";
        computer.Backward();
      } else {
        // work out the predicted objf-change as dot-product of deriv and
        // parameter-change.  The expression below can be interpreted as
        // DotProduct(nnet_copy - nnet, nnet_deriv).
        predicted_objf_change[pass] = DotProduct(nnet_copy, nnet_deriv) -
                                      DotProduct(nnet, nnet_deriv);
      }
    }
    
    Vector<BaseFloat> predicted_objf_change_vec(num_directions),
        measured_objf_change_vec(num_directions);
    for (int32 d = 0; d < num_directions; d++) {
      BaseFloat predicted_change = predicted_objf_change[d+1],
                 measured_change = measured_objf[d+1] - measured_objf[0];
      predicted_objf_change_vec(d) = predicted_change;
      measured_objf_change_vec(d) = measured_change;
    }
    KALDI_LOG << "Vector of predicted objf-change is: "
              << predicted_objf_change_vec;
    KALDI_LOG << "Vector of measured objf-change is: "
              << measured_objf_change_vec;
    BaseFloat delta_thresh = 0.05;
    if (!ApproxEqual(predicted_objf_change_vec,
                     measured_objf_change_vec, delta_thresh)) {
      KALDI_WARN << "Predicted and measured objf-changes differ too much.";
    } else {
      num_success++;
    }
  }
  KALDI_LOG << "Model-derivative check succeeded for " << num_success << " out of "
            << num_tries << " tries.";
  if (num_success < num_tries - (2 + num_tries / 5))
    KALDI_ERR << "Failed too many times.";
}


// This test makes sure that the input-derivatives are correct.
void UnitTestNnetInputDerivatives() {
  int32 num_tries = 20, num_success = 0;
  for (int32 n = 0; n < num_tries; n++) {
    struct NnetGenerationOptions gen_config;
    //gen_config.allow_nonlinearity = false;
    //gen_config.allow_recursion = false;
    //gen_config.allow_final_nonlinearity = true;
    bool allow_optimization = true;
    
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

    // make sure that all inputs and outputs have derivatives requested/provided,
    // and that the model-update (need_model_derivative) is not requested.
    request.need_model_derivative = false;
    for (int32 i = 0; i < request.inputs.size(); i++)
      request.inputs[i].has_deriv = true;
    request.outputs[0].has_deriv = true;
    
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
    ComputationChecker checker(check_config, nnet, request, computation);
    checker.Check();
    
    if (RandInt(0, 3) != 0 && allow_optimization) {
      NnetOptimizeOptions opt_config;
      Optimize(opt_config, nnet, request, &computation);
      std::ostringstream os;
      computation.Print(os, nnet);
      KALDI_LOG << "Optimized computation is: " << os.str();
    }

    NnetComputeOptions compute_opts;
    if (RandInt(0, 1) == 0)
      compute_opts.debug = true;
    computation.ComputeCudaIndexes();

    // the only reason we might need to provide the &nnet parameter is if the
    // StoreStats() operation had been requested.  We made sure no model update
    // is being performed.
    NnetComputer computer(compute_opts,
                          computation,
                          nnet,
                          &nnet);

    int32 num_directions = 3;  // must be >= 1.  Best if it's >1, will reduce
                               // the probability of random failures.

    // the order of these vectors is:
    // [ un-perturbed, perturbed-1, perturbed-2, perturbed-3, un-perturbed ].
    // we compute un-perturbed twice to double-check the model did not change.
    std::vector<BaseFloat> measured_objf(num_directions + 2, 0.0),
        predicted_objf_change(num_directions + 2, 0.0);
    BaseFloat delta = 1.0e-03;

    // output_deriv is the derivative of the objective function w.r.t. the
    // (single) output.  We make the objf a linear function of the output and
    // just set the output_deriv to be a random matrix, which defines the
    // objective function.
    CuMatrix<BaseFloat> output_deriv;
    output_deriv.Resize(request.outputs[0].indexes.size(),
                        nnet.OutputDim("output"));
    output_deriv.SetRandn();

    std::vector<CuMatrix<BaseFloat> > delta_inputs(inputs.size());
    std::vector<CuMatrix<BaseFloat> > input_derivs(inputs.size());

    // pass 0 is the forward pass with the un-perturbed features; so is
    // pass num_directions + 1.
    // Other passes are with various differently-perturbed versions of
    // the features.
    for (int32 pass = 0; pass <= num_directions + 1; pass++) {  
      // provide the input to the computations.
      for (size_t i = 0; i < request.inputs.size(); i++) {
        CuMatrix<BaseFloat> temp(inputs[i]);
        if (pass > 0 && pass <= num_directions) {  // Perturb the input randomly.
          delta_inputs[i].Resize(inputs[i].NumRows(), inputs[i].NumCols());
          delta_inputs[i].SetRandn();
          delta_inputs[i].Scale(delta);
          // if there are >1 inputs, sometimes set the delta for input 0 to
          // zero.  might sometimes give more accurate test of error in iVector
          // derivative computation.
          if (i == 0 && request.inputs.size() > 1 && RandInt(0, 1) == 0)
            delta_inputs[i].SetZero();
          temp.AddMat(1.0, delta_inputs[i]);
          predicted_objf_change[pass] += TraceMatMat(input_derivs[i],
                                                     delta_inputs[i], kTrans);
        }
        computer.AcceptInput(request.inputs[i].name, &temp);
      }

      KALDI_LOG << "Running forward computation";
      computer.Forward();
      
      const CuMatrixBase<BaseFloat> &output(computer.GetOutput("output"));
      KALDI_LOG << "Output sum for pass " << pass << " is " << output.Sum();
      BaseFloat objf = TraceMatMat(output, output_deriv, kTrans);
      measured_objf[pass] = objf;

      if (pass == 0) {
        // We need to compute the input derivatives.
        CuMatrix<BaseFloat> temp(output_deriv);
        computer.AcceptOutputDeriv("output", &temp);
        KALDI_LOG << "Running backward computation";
        computer.Backward();
        for (size_t i = 0; i < request.inputs.size(); i++) {
          input_derivs[i] = computer.GetInputDeriv(request.inputs[i].name);
          KALDI_LOG << "Input-deriv norm for '" << request.inputs[i].name
                    << "' is " << input_derivs[i].FrobeniusNorm();
        }
      }
    }
    KALDI_ASSERT(ApproxEqual(measured_objf[0],
                             measured_objf[num_directions + 1]));
    
    Vector<BaseFloat> predicted_objf_change_vec(num_directions),
        measured_objf_change_vec(num_directions);
    for (int32 d = 0; d < num_directions; d++) {
      BaseFloat predicted_change = predicted_objf_change[d+1],
                 measured_change = measured_objf[d+1] - measured_objf[0];
      predicted_objf_change_vec(d) = predicted_change;
      measured_objf_change_vec(d) = measured_change;
    }
    KALDI_LOG << "Vector of predicted objf-change is: "
              << predicted_objf_change_vec;
    KALDI_LOG << "Vector of measured objf-change is: "
              << measured_objf_change_vec;
    BaseFloat delta_thresh = 0.1;
    if (!ApproxEqual(predicted_objf_change_vec,
                     measured_objf_change_vec, delta_thresh)) {
      KALDI_WARN << "Predicted and measured objf-changes differ too much.";
    } else {
      num_success++;
    }
  }
  KALDI_LOG << "Input-derivative check succeeded for " << num_success << " out of "
            << num_tries << " tries.";
  if (num_success < num_tries - (2 + num_tries / 5))
    KALDI_ERR << "Failed too many times.";
}


} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  //SetVerboseLevel(2);


  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    UnitTestNnetModelDerivatives();
    UnitTestNnetInputDerivatives();
  }

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}


// nnet3bin/nnet3-combine.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2017  Yiming Wang

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-diagnostics.h"


namespace kaldi {
namespace nnet3 {

// Computes and returns the objective function for the examples in 'egs' given
// the model in 'nnet'. If either of batchnorm/dropout test modes is true, we
// make a copy of 'nnet', set test modes on that and evaluate its objective.
// Note: the object that prob_computer->nnet_ refers to should be 'nnet'.
double ComputeObjf(bool batchnorm_test_mode, bool dropout_test_mode,
                   const std::vector<NnetExample> &egs, const Nnet &nnet,
                   NnetComputeProb *prob_computer) {
  if (batchnorm_test_mode || dropout_test_mode) {
    Nnet nnet_copy(nnet);
    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet_copy);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet_copy);
    NnetComputeProbOptions compute_prob_opts;
    NnetComputeProb prob_computer_test(compute_prob_opts, nnet_copy);
    return ComputeObjf(false, false, egs, nnet_copy, &prob_computer_test);
  } else {
    prob_computer->Reset();
    std::vector<NnetExample>::const_iterator iter = egs.begin(),
                                              end = egs.end();
    for (; iter != end; ++iter)
      prob_computer->Compute(*iter);
    double tot_weights,
        tot_objf = prob_computer->GetTotalObjective(&tot_weights);
    KALDI_ASSERT(tot_weights > 0.0);
    // inf/nan tot_objf->return -inf objective.
    if (!(tot_objf == tot_objf && tot_objf - tot_objf == 0))
      return -std::numeric_limits<double>::infinity();
    // we prefer to deal with normalized objective functions.
    return tot_objf / tot_weights;
  }
}

// Updates moving average over num_models nnets, given the average over
// previous (num_models - 1) nnets, and the new nnet.
void UpdateNnetMovingAverage(int32 num_models,
    const Nnet &nnet, Nnet *moving_average_nnet) {
  KALDI_ASSERT(NumParameters(nnet) == NumParameters(*moving_average_nnet));
  ScaleNnet((num_models - 1.0) / num_models, moving_average_nnet);
  AddNnet(nnet, 1.0 / num_models, moving_average_nnet);
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a subset of training or held-out examples, compute the average\n"
        "over the first n nnet3 models where we maximize the objective function\n"
        "for n. Note that the order of models has been reversed before\n"
        "being fed into this binary. So we are actually combining last n models.\n"
        "Inputs and outputs are 'raw' nnets.\n"
        "\n"
        "Usage:  nnet3-combine [options] <nnet-in1> <nnet-in2> ... <nnet-inN> <valid-examples-in> <nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-combine 1.1.raw 1.2.raw 1.3.raw ark:valid.egs 2.raw\n";

    bool binary_write = true;
    int32 max_objective_evaluations = 30;
    bool batchnorm_test_mode = false,
        dropout_test_mode = true;
    std::string use_gpu = "yes";

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("max-objective-evaluations", &max_objective_evaluations, "The "
                "maximum number of objective evaluations in order to figure "
                "out the best number of models to combine. It helps to speedup "
                "if the number of models provided to this binary is quite "
                "large (e.g. several hundred)."); 
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents "
                "while evaluating objectives.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents while evaluating objectives.");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string
        nnet_rxfilename = po.GetArg(1),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    Nnet moving_average_nnet(nnet), best_nnet(nnet);
    NnetComputeProbOptions compute_prob_opts;
    NnetComputeProb prob_computer(compute_prob_opts, moving_average_nnet);

    std::vector<NnetExample> egs;
    egs.reserve(10000);  // reserve a lot of space to minimize the chance of
                         // reallocation.

    { // This block adds training examples to "egs".
      SequentialNnetExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(example_reader.Value());
      KALDI_LOG << "Read " << egs.size() << " examples.";
      KALDI_ASSERT(!egs.empty());
    }

    // first evaluates the objective using the last model.
    int32 best_num_to_combine = 1;
    double
        init_objf = ComputeObjf(batchnorm_test_mode, dropout_test_mode,
            egs, moving_average_nnet, &prob_computer),
        best_objf = init_objf;
    KALDI_LOG << "objective function using the last model is " << init_objf;

    int32 num_nnets = po.NumArgs() - 2;
    // then each time before we re-evaluate the objective function, we will add
    // num_to_add models to the moving average.
    int32 num_to_add = (num_nnets + max_objective_evaluations - 1) /
                       max_objective_evaluations;
    for (int32 n = 1; n < num_nnets; n++) {
      ReadKaldiObject(po.GetArg(1 + n), &nnet);
      // updates the moving average
      UpdateNnetMovingAverage(n + 1, nnet, &moving_average_nnet);
      // evaluates the objective everytime after adding num_to_add model or
      // all the models to the moving average.
      if ((n - 1) % num_to_add == num_to_add - 1 || n == num_nnets - 1) {
        double objf = ComputeObjf(batchnorm_test_mode, dropout_test_mode,
            egs, moving_average_nnet, &prob_computer);
        KALDI_LOG << "Combining last " << n + 1
                  << " models, objective function is " << objf;
        if (objf > best_objf) {
          best_objf = objf;
          best_nnet = moving_average_nnet;
          best_num_to_combine = n + 1;
        }
      }
    }
    KALDI_LOG << "Combining " << best_num_to_combine
              << " nnets, objective function changed from " << init_objf
              << " to " << best_objf;

    if (HasBatchnorm(nnet))
      RecomputeStats(egs, &best_nnet);

#if HAVE_CUDA==1
      CuDevice::Instantiate().PrintProfile();
#endif

    WriteKaldiObject(best_nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

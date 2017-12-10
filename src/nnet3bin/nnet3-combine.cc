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

double ComputeObjf(const std::vector<NnetExample> &egs,
                   NnetComputeProb *prob_computer) {
  prob_computer->Reset();
  std::vector<NnetExample>::const_iterator iter = egs.begin(),
                                            end = egs.end();
  for (; iter != end; ++iter)
    prob_computer->Compute(*iter);
  double tot_weights,
      tot_objf = prob_computer->GetTotalObjective(&tot_weights);
  KALDI_ASSERT(tot_weights > 0.0);
  // we prefer to deal with normalized objective functions.
  return tot_objf / tot_weights;
}

// Note: the object that prob_computer.nnet_ refers to should be
// *moving_average_nnet.
double UpdateNnetMovingAverageAndComputeObjf(int32 num_models,
    const std::vector<NnetExample> &egs,
    const Nnet &nnet, Nnet *moving_average_nnet,
    NnetComputeProb *prob_computer) {
  int32 num_params = NumParameters(nnet);
  KALDI_ASSERT(num_params == NumParameters(*moving_average_nnet));
  Vector<BaseFloat> nnet_params(num_params, kUndefined),
      moving_average_nnet_params(num_params, kUndefined);
  VectorizeNnet(nnet, &nnet_params);
  VectorizeNnet(*moving_average_nnet, &moving_average_nnet_params);
  moving_average_nnet_params.Scale((num_models - 1.0) / num_models);
  moving_average_nnet_params.AddVec(1.0 / num_models,  nnet_params);

  BaseFloat sum = moving_average_nnet_params.Sum();
  // inf/nan parameters->return -inf objective.
  if (!(sum == sum && sum - sum == 0))
    return -std::numeric_limits<double>::infinity();

  UnVectorizeNnet(moving_average_nnet_params, moving_average_nnet);
  return ComputeObjf(egs, prob_computer);
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
        "over the first n nnet3 models where we maxize the objective function\n"
        "for n. Note that the order of models has been reversed before\n"
        "feeding into this binary. So we are actually combining last n models.\n"
        "Inputs and outputs are 'raw' nnets.\n"
        "\n"
        "Usage:  nnet3-combine [options] <nnet-in1> <nnet-in2> ... <nnet-inN> <valid-example-in> <nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-combine 1.1.raw 1.2.raw 1.3.raw ark:valid.egs 2.raw\n";

    bool binary_write = true;
    bool batchnorm_test_mode = false,
        dropout_test_mode = true;
    std::string use_gpu = "yes";

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");
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
    NnetComputeProb *prob_computer = new NnetComputeProb(compute_prob_opts,
                                                         moving_average_nnet);

    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &nnet);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet);

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

    int32 best_n = 1;
    double best_objf = ComputeObjf(egs, prob_computer);
    KALDI_LOG << "objective function using the last model is " << best_objf;

    int32 num_inputs = po.NumArgs() - 2;
    if (num_inputs > 1) {
      for (int32 n = 1; n < num_inputs; n++) {
        ReadKaldiObject(po.GetArg(1 + n), &nnet);
        double objf = UpdateNnetMovingAverageAndComputeObjf(n + 1, egs, nnet,
            &moving_average_nnet, prob_computer);
        KALDI_LOG << "Combining last " << n + 1
                  << " models, objective function is " << objf;
        if (objf > best_objf) {
          best_objf = objf;
          best_nnet = moving_average_nnet;
          best_n = n + 1;
        }
      }
      KALDI_LOG << "Using the model averaged over last " << best_n
                << " models, objective function is " << best_objf;

#if HAVE_CUDA==1
      CuDevice::Instantiate().PrintProfile();
#endif
      if (HasBatchnorm(nnet))
        RecomputeStats(egs, &best_nnet);
      WriteKaldiObject(best_nnet, nnet_wxfilename, binary_write);
    } else {
      KALDI_LOG << "Copying the single input model directly to the output, "
                << "without any combination.";
      if (HasBatchnorm(nnet))
        RecomputeStats(egs, &nnet);
      WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    }
    KALDI_LOG << "Finished combining neural nets, wrote model to "
              << nnet_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

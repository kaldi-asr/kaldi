// nnet3bin/nnet3-discriminative-train.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-discriminative-training.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3 neural network parameters with discriminative sequence objective \n"
        "gradient descent.  Minibatches are to be created by nnet3-discriminative-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU).\n"
        "\n"
        "Usage:  nnet3-discriminative-train [options] <nnet-in> <discriminative-training-examples-in> <raw-nnet-out>\n"
        "\n"
        "nnet3-discriminative-train 1.mdl 'ark:nnet3-merge-egs 1.degs ark:-|' 2.raw\n";

    bool binary_write = true;
    std::string use_gpu = "yes";
    bool dropout_test_mode = true;
    
    NnetDiscriminativeOptions opts;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");

    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string model_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        model_wxfilename = po.GetArg(3);

    TransitionModel tmodel;
    AmNnetSimple am_nnet;

    bool binary;
    Input ki(model_rxfilename, &binary);
    
    tmodel.Read(ki.Stream(), binary);
    am_nnet.Read(ki.Stream(), binary);
    
    Nnet nnet = am_nnet.GetNnet();

    if (dropout_test_mode)
      SetDropoutTestMode(true, &nnet);
    
    const VectorBase<BaseFloat> &priors = am_nnet.Priors();

    NnetDiscriminativeTrainer trainer(opts, tmodel, priors, &nnet);

    SequentialNnetDiscriminativeExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      trainer.Train(example_reader.Value());

    bool ok = trainer.PrintTotalStats();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    Output ko(model_wxfilename, binary_write);
    nnet.Write(ko.Stream(), binary_write);
    
    KALDI_LOG << "Wrote raw nnet model to " << model_wxfilename;
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


// nnet3bin/nnet3-ctc-train.cc

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
#include "nnet3/nnet-cctc-training.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3+ctc neural network parameters with backprop and stochastic\n"
        "gradient descent.  Minibatches are to be created by nnet3-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU).  The --write-raw option controls whether the entire\n"
        "model including the transition-model, or just the neural net, is output.\n"
        "\n"
        "Usage:  nnet3-ctc-train [options] <model-in> <ctc-training-examples-in> (<model-out>|<raw-model-out>)\n"
        "\n"
        "nnet3-ctc-train 1.mdl 'ark:nnet3-merge-egs 1.cegs ark:-|' 2.raw\n";

    bool binary_write = true;
    bool write_raw = false;
    std::string use_gpu = "yes";
    NnetCctcTrainerOptions train_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("write-raw", &write_raw, "If true, write just the raw neural-net "
                "and not also the transition-model");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    train_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string cctc_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);

    CctcTransitionModel trans_model;
    Nnet nnet;
    {
      bool binary;
      Input input(cctc_nnet_rxfilename, &binary);
      trans_model.Read(input.Stream(), binary);
      nnet.Read(input.Stream(), binary);
    }
    
    NnetCctcTrainer trainer(train_config, trans_model, &nnet);

    SequentialNnetCctcExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      trainer.Train(example_reader.Value());

    bool ok = trainer.PrintTotalStats();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    if (write_raw) {
      WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
      KALDI_LOG << "Wrote raw model to " << nnet_wxfilename;
    } else {
      Output output(nnet_wxfilename, binary_write);
      trans_model.Write(output.Stream(), binary_write);
      nnet.Write(output.Stream(), binary_write);
      KALDI_LOG << "Wrote model to " << nnet_wxfilename;
    }
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

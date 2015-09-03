// nnet2bin/nnet-train-perturbed.cc

// Copyright 2012-2014  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/train-nnet-perturbed.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train the neural network parameters with backprop and stochastic\n"
        "gradient descent using minibatches.  As nnet-train-simple, but\n"
        "perturbs the input features by going a certain distance down the\n"
        "backprop-ed gradient.  Can be helpful for small datasets.\n"
        "\n"
        "Usage:  nnet-train-perturbed [options] <model-in> <training-examples-in> <model-out>\n"
        "note: the option --within-covar=<file> is needed\n"
        "\n"
        "e.g.:\n"
        "nnet-train-perturbed --within-covar=within.spmat 1.nnet ark:1.1.egs 2.nnet\n";
    
    bool binary_write = true;
    bool zero_stats = true;
    int32 srand_seed = 0;
    std::string use_gpu = "yes";
    std::string within_covar_rxfilename;
    NnetPerturbedTrainerConfig train_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("within-covar", &within_covar_rxfilename,
                "rxfilename of within-class covariance-matrix, written as "
                "SpMatrix.  Must be specified.");
    po.Register("zero-stats", &zero_stats, "If true, zero occupation "
                "counts stored with the neural net (only affects mixing up).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(relevant if you have layers of type AffineComponentPreconditioned "
                "with l2-penalty != 0.0");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    
    train_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    srand(srand_seed);
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    if (within_covar_rxfilename == "") {
      KALDI_ERR << "The option --within-covar is required.";
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    int64 num_examples = 0;

    {
      TransitionModel trans_model;
      AmNnet am_nnet;
      {
        bool binary_read;
        Input ki(nnet_rxfilename, &binary_read);
        trans_model.Read(ki.Stream(), binary_read);
        am_nnet.Read(ki.Stream(), binary_read);
      }

      SpMatrix<BaseFloat> within_covar;
      ReadKaldiObject(within_covar_rxfilename, &within_covar);

      if (zero_stats) am_nnet.GetNnet().ZeroStats();
    
      { // want to make sure this object deinitializes before
        // we write the model, as it does something in the destructor.
        NnetPerturbedTrainer trainer(train_config,
                                     within_covar,
                                     &(am_nnet.GetNnet()));
      
        SequentialNnetExampleReader example_reader(examples_rspecifier);

        for (; !example_reader.Done(); example_reader.Next(), num_examples++)
          trainer.TrainOnExample(example_reader.Value());  // It all happens here!
      }
    
      {
        Output ko(nnet_wxfilename, binary_write);
        trans_model.Write(ko.Stream(), binary_write);
        am_nnet.Write(ko.Stream(), binary_write);
      }
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.  Wrote model to "
              << nnet_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



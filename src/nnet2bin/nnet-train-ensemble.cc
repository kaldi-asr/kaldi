// nnet2bin/nnet-train-ensemble.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//           2014  Xiaohui Zhang

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
#include "nnet2/train-nnet-ensemble.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train an ensemble of neural networks with backprop and stochastic\n"
        "gradient descent using minibatches.  Modified version of nnet-train-simple.\n"
        "Implements parallel gradient descent with a term that encourages the nnets to\n"
        "produce similar outputs.\n"
        "\n"
        "Usage:  nnet-train-ensemble [options] <model-in-1> <model-in-2> ... <model-in-n>"
        "  <training-examples-in> <model-out-1> <model-out-2> ... <model-out-n> \n"
        "\n"
        "e.g.:\n"
        " nnet-train-ensemble 1.1.nnet 2.1.nnet ark:egs.ark 2.1.nnet 2.2.nnet \n";
    
    bool binary_write = true;
    bool zero_stats = true;
    int32 srand_seed = 0;
    std::string use_gpu = "yes";
    NnetEnsembleTrainerConfig train_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("zero-stats", &zero_stats, "If true, zero occupation "
                "counts stored with the neural net (only affects mixing up).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(relevant if you have layers of type AffineComponentPreconditioned "
                "with l2-penalty != 0.0");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
 
    train_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() <= 3) {
      po.PrintUsage();
      exit(1);
    }
    srand(srand_seed);
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    
    int32 num_nnets = (po.NumArgs() - 1) / 2;
    std::string nnet_rxfilename = po.GetArg(1);
    std::string examples_rspecifier = po.GetArg(num_nnets + 1);

    std::string nnet1_rxfilename = po.GetArg(1);
    
    TransitionModel trans_model;
    std::vector<AmNnet> am_nnets(num_nnets);
    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      KALDI_LOG << nnet1_rxfilename;
      am_nnets[0].Read(ki.Stream(), binary_read);
    }

    std::vector<Nnet*> nnets(num_nnets);
    nnets[0] = &(am_nnets[0].GetNnet());

    for (int32 n = 1; n < num_nnets; n++) {
      TransitionModel trans_model;
      bool binary_read;
      Input ki(po.GetArg(1 + n), &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnets[n].Read(ki.Stream(), binary_read);
      nnets[n] = &am_nnets[n].GetNnet();
    }      
    

    int64 num_examples = 0;

    {
      if (zero_stats) {
        for (int32 n = 1; n < num_nnets; n++) 
          nnets[n]->ZeroStats();
      }
      { // want to make sure this object deinitializes before
        // we write the model, as it does something in the destructor.
        NnetEnsembleTrainer trainer(train_config,
                                    nnets);
      
        SequentialNnetExampleReader example_reader(examples_rspecifier);

        for (; !example_reader.Done(); example_reader.Next(), num_examples++)
          trainer.TrainOnExample(example_reader.Value());  // It all happens here!
      }
    
      {
        for (int32 n = 0; n < num_nnets; n++) {
          Output ko(po.GetArg(po.NumArgs() - num_nnets + n + 1), binary_write);
          trans_model.Write(ko.Stream(), binary_write);
          am_nnets[n].Write(ko.Stream(), binary_write);
        }
      }
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.";
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



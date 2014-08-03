// nnet2bin/nnet-train-parallel-perturbed.cc

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
#include "thread/kaldi-thread.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train the neural network parameters with backprop and stochastic\n"
        "gradient descent using minibatches.  The training frames and labels\n"
        "are read via a pipe from nnet-randomize-frames.  This is like nnet-train-parallel,\n"
        "using multiple threads in a Hogwild type of update, but also adding\n"
        "perturbed training (see src/nnet2/train-nnet-perturbed.h for info)\n"
        "\n"
        "Usage:  nnet-train-parallel-perturbed [options] <model-in> <training-examples-in> <model-out>\n"
        "\n"
        "e.g.:\n"
        "nnet-randomize-frames [args] | nnet-train-parallel-pertured \\\n"
        " --within-covar=within.spmat --num-threads=8 --target-objf-change=0.2 1.nnet ark:- 2.nnet\n";
    
    bool binary_write = true;
    bool zero_stats = true;
    int32 srand_seed = 0;
    std::string within_covar_rxfilename;
    NnetPerturbedTrainerConfig train_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("within-covar", &within_covar_rxfilename,
                "rxfilename of within-class covariance-matrix, written as "
                "SpMatrix.  Must be specified.");
    po.Register("zero-stats", &zero_stats, "If true, zero stats "
                "stored with the neural net (only affects mixing up).");
    po.Register("srand", &srand_seed,
                "Seed for random number generator (e.g., for dropout)");
    po.Register("num-threads", &g_num_threads, "Number of training threads to use "
                "in the parallel update. [Note: if you use a parallel "
                "implementation of BLAS, the actual number of threads may be larger.]");
    train_config.Register(&po);
    
    po.Read(argc, argv);
    srand(srand_seed);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);

    if (within_covar_rxfilename == "") {
      KALDI_ERR << "The option --within-covar is required.";
    }
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    KALDI_ASSERT(train_config.minibatch_size > 0);

    SpMatrix<BaseFloat> within_covar;
    ReadKaldiObject(within_covar_rxfilename, &within_covar);
    
    if (zero_stats) am_nnet.GetNnet().ZeroStats();

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    
    
    double tot_objf_orig, tot_objf_perturbed, tot_weight;
    // logging info will be printed from within the next call.
    DoBackpropPerturbedParallel(train_config,
                                within_covar,
                                &example_reader,
                                &tot_objf_orig,
                                &tot_objf_perturbed,
                                &tot_weight,
                                &(am_nnet.GetNnet()));
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Finished training, processed " << tot_weight
              << " training examples (weighted).  Wrote model to "
              << nnet_wxfilename;
    return (tot_weight == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



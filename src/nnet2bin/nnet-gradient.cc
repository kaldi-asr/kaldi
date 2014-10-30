// nnet2bin/nnet-gradient.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet2/nnet-update-parallel.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Compute neural net gradient using backprop.  Can use multiple threads\n"
        "using --num-threads option.   Note: this is in addition to any BLAS-level\n"
        "multi-threading.  Note: the model gradient is written in the same format\n"
        "as the model, with the transition-model included.\n"
        "\n"
        "Usage:  nnet-gradient [options] <model-in> <training-examples-in> <model-gradient-out>\n"
        "\n"
        "e.g.:  nnet-gradient 1.nnet ark:1.egs 1.gradient\n";
    
    bool binary_write = true;
    int32 minibatch_size = 1024;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("num-threads", &g_num_threads, "Number of training threads to use "
                "in the parallel update. [Note: if you use a parallel "
                "implementation of BLAS, the actual number of threads may be larger.]");
    po.Register("minibatch-size", &minibatch_size, "Number of examples to use for "
                "each minibatch during training.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        gradient_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    KALDI_ASSERT(minibatch_size > 0);


    double num_examples = 0.0;
    SequentialNnetExampleReader example_reader(examples_rspecifier);

    AmNnet am_gradient(am_nnet);
    bool is_gradient = true;
    am_gradient.GetNnet().SetZero(is_gradient);
    
    // BaseFloat tot_loglike =
    DoBackpropParallel(am_nnet.GetNnet(),
                       minibatch_size,
                       &example_reader,
                       &num_examples,
                       &(am_gradient.GetNnet()));
    // This function will have produced logging output, so we have no
    // need for that here.
    
    {
      Output ko(gradient_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gradient.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Finished computing gradient, processed " << num_examples
              << " training examples.  Wrote model to "
              << gradient_wxfilename;
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



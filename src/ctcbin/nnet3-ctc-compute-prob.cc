// nnet3bin/nnet3-ctc-compute-prob.cc

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
#include "nnet3/nnet-cctc-diagnostics.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints to in logging messages the average log-prob per frame of\n"
        "the given data with an nnet3+ctc neural net.  The input of this is the output of\n"
        "e.g. nnet3-ctc-get-egs | nnet3-ctc-merge-egs.\n"
        "\n"
        "Usage:  nnet3-ctc-compute-prob [options] <nnet3-ctc-model-in> <training-examples-in>\n"
        "e.g.: nnet3-ctc-compute-prob 0.mdl ark:valid.egs\n";

    
    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.
    // It wouldn't be hard to make it support GPU, though.

    NnetCctcComputeProbOptions opts;
    
    ParseOptions po(usage);

    opts.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string ctc_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    ctc::CctcTransitionModel trans_model;
    Nnet nnet;
    {
      bool binary;
      Input input(ctc_nnet_rxfilename, &binary);
      trans_model.Read(input.Stream(), binary);
      nnet.Read(input.Stream(), binary);
    }

    NnetCctcComputeProb cctc_prob_computer(opts, trans_model, nnet);
    
    SequentialNnetCctcExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      cctc_prob_computer.Compute(example_reader.Value());

    bool ok = cctc_prob_computer.PrintTotalStats();
    
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



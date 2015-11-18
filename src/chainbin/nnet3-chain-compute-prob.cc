// nnet3bin/nnet3-chain-compute-prob.cc

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
#include "nnet3/nnet-chain-diagnostics.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints to in logging messages the average log-prob per frame of\n"
        "the given data with an nnet3+chain neural net.  The input of this is the output of\n"
        "e.g. nnet3-chain-get-egs | nnet3-chain-merge-egs.\n"
        "\n"
        "Usage:  nnet3-chain-compute-prob [options] <raw-nnet3-model-in> <denominator-fst> <training-examples-in>\n"
        "e.g.: nnet3-chain-compute-prob 0.mdl den.fst ark:valid.egs\n";


    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.
    // It wouldn't be hard to make it support GPU, though.

    NnetComputeProbOptions nnet_opts;
    chain::ChainTrainingOptions chain_opts;

    ParseOptions po(usage);

    nnet_opts.Register(&po);
    chain_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        den_fst_rxfilename = po.GetArg(2),
        examples_rspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);

    NnetChainComputeProb chain_prob_computer(nnet_opts, chain_opts, den_fst,
                                            nnet);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      chain_prob_computer.Compute(example_reader.Value());

    bool ok = chain_prob_computer.PrintTotalStats();

    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



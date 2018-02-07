// nnet3bin/nnet3-discriminative-compute-objf.cc

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//           2014-2015  Vimal Manohar

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
#include "nnet3/nnet-discriminative-diagnostics.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints to in logging messages the objective function per frame of\n"
        "the given data with an nnet3 neural net.  The input of this is the output of\n"
        "e.g. nnet3-discriminative-get-egs | nnet3-discriminative-merge-egs.\n"
        "\n"
        "Usage:  nnet3-discrminative-compute-objf [options] <nnet3-model-in> <training-examples-in>\n"
        "e.g.: nnet3-discriminative-compute-objf 0.mdl ark:valid.degs\n";

    bool batchnorm_test_mode = true, dropout_test_mode = true;

    // This program doesn't support using a GPU, because these probabilities are
    // used for diagnostics, and you can just compute them with a small enough
    // amount of data that a CPU can do it within reasonable time.
    // It wouldn't be hard to make it support GPU, though.

    NnetComputeProbOptions nnet_opts;
    discriminative::DiscriminativeOptions discriminative_opts;

    ParseOptions po(usage);

    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");

    nnet_opts.Register(&po);
    discriminative_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    TransitionModel tmodel;
    AmNnetSimple am_nnet;

    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      tmodel.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }
    
    Nnet* nnet = &(am_nnet.GetNnet());

    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, nnet);

    if (dropout_test_mode)
      SetDropoutTestMode(true, nnet);

    NnetDiscriminativeComputeObjf discriminative_objf_computer(nnet_opts, 
                                              discriminative_opts, 
                                              tmodel, am_nnet.Priors(), 
                                              *nnet);

    SequentialNnetDiscriminativeExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      discriminative_objf_computer.Compute(example_reader.Value());

    bool ok = discriminative_objf_computer.PrintTotalStats();

    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



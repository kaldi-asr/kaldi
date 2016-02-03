// nnet3bin/nnet3-compute-prob.cc

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
#include "nnet3/nnet-diagnostics.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints to in logging messages the average log-prob per frame of\n"
        "the given data with an nnet3 neural net.  The input of this is the output of\n"
        "e.g. nnet3-get-egs | nnet3-merge-egs.\n"
        "\n"
        "Usage:  nnet3-compute-prob [options] <raw-model-in> <training-examples-in>\n"
        "e.g.: nnet3-compute-prob 0.raw ark:valid.egs\n";

    
    std::string use_gpu = "no";

    NnetComputeProbOptions opts;
    
    ParseOptions po(usage);
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    opts.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string raw_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);

    NnetComputeProb prob_computer(opts, nnet);
    
    SequentialNnetExampleReader example_reader(examples_rspecifier);

    for (; !example_reader.Done(); example_reader.Next())
      prob_computer.Compute(example_reader.Value());

    bool ok = prob_computer.PrintTotalStats();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



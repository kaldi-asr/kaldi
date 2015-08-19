// nnet3bin/nnet3-copy.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)
//           2015  Vimal Manohar

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

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy nnet3 raw neural-net file\n"
        "Also supports multiplying all the learning rates by a factor\n"
        "(the --learning-rate-factor option) and setting them all to supplied\n"
        "values (the --learning-rate and --learning-rates options),\n"
        "\n"
        "Usage:  nnet3-copy [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-copy --binary=false 1.raw text.raw\n";

    bool binary_write = true;
    BaseFloat learning_rate = -1;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("learning-rate", &learning_rate,
                "If supplied, all the learning rates of updatable components"
                "are set to this value.");


    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(2);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    
    if (learning_rate >= 0)
      SetLearningRate(learning_rate, &nnet);

    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    KALDI_LOG << "Copied neural net from " << nnet_rxfilename
              << " to " << nnet_wxfilename;
      
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


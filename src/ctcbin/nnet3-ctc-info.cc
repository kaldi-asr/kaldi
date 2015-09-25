// ctcbin/nnet3-ctc-info.cc

// Copyright 2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "ctc/cctc-transition-model.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print some text information about an nnet3+ctc neural network, to\n"
        "standard output\n"
        "\n"
        "Usage:  nnet3-ctc-info [options] <nnet>\n"
        "e.g.:\n"
        " nnet3-ctc-info 0.mdl\n"
        "See also: nnet3-ctc-info\n";

    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string ctc_nnet_rxfilename = po.GetArg(1);
    
    CctcTransitionModel trans_model;
    Nnet nnet;
    {
      bool binary;
      Input input(ctc_nnet_rxfilename, &binary);
      trans_model.Read(input.Stream(), binary);
      if (set_raw_nnet.empty())
        nnet.Read(input.Stream(), binary);
    }

    std::cout << trans_model.Info()
              << NnetInfo(nnet);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

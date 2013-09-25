// nnet2bin/nnet-am-mixup.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/am-nnet.h"
#include "nnet2/mixup-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy a (cpu-based) neural net and its associated transition model,\n"
        "possibly changing the binary mode\n"
        "Also supports multiplying all the learning rates by a factor\n"
        "(the --learning-rate-factor option) and setting them all to a given\n"
        "value (the --learning-rate options)\n"
        "\n"
        "Usage:  nnet-am-mixup [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-mixup --power=0.3 --num-mixtures=5000 1.mdl 2.mdl\n";

    NnetMixupConfig config;
    bool binary_write = true;
    
    ParseOptions po(usage);
    config.Register(&po);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(2);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    MixupNnet(config, &(am_nnet.GetNnet()));

    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Mixed up neural net from " << nnet_rxfilename
              << " and wrote it to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

// nnet2bin/nnet-am-stats.cc

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
#include "nnet2/nnet-stats.h"
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print some statistics about the average derivatives of the sigmoid layers\n"
        "of the neural net, that are stored in the net\n"
        "\n"
        "Usage:  nnet-am-stats [options] <nnet-in>\n"
        "e.g.:\n"
        " nnet-am-stats 1.mdl 1_fixed.mdl\n";
    
    NnetStatsConfig config;
    
    ParseOptions po(usage);
    config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    std::vector<NnetStats> stats;
    GetNnetStats(config, am_nnet.GetNnet(), &stats);
    if (stats.empty()) {
      KALDI_WARN << "No stats obtained (possibly nnet has wrong topology,"
                 << "expect affine followed by (nonlinear but not softmax)";
    }
    for (size_t i = 0; i < stats.size(); i++)
      stats[i].PrintStats(std::cout);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

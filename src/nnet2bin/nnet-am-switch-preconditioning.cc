// nnet2bin/nnet-am-switch-preconditioning.cc

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
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy a (cpu-based) neural net and its associated transition model,\n"
        "and switch it to online preconditioning, i.e. change any components\n"
        "derived from AffineComponent to components of type\n"
        "AffineComponentPreconditionedOnline.\n"
        "\n"
        "Usage:  nnet-am-switch-preconditioning [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-switch-preconditioning --binary=false 1.mdl text.mdl\n";

    int32 rank_in = 20, rank_out = 80, update_period = 4;
    BaseFloat num_samples_history = 2000.0;
    BaseFloat alpha = 4.0;
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("rank-in", &rank_in,
                "Rank used in online-preconditioning on input side of each layer");
    po.Register("rank-out", &rank_out,
                "Rank used in online-preconditioning on output side of each layer");
    po.Register("update-period", &update_period,
                "Affects how frequently we update the Fisher-matrix estimate (every "
                "this-many minibatches).");
    po.Register("num-samples-history", &num_samples_history,
                "Number of samples of history to use in online preconditioning "
                "(affects speed vs accuracy of update of Fisher matrix)");
    po.Register("alpha", &alpha,
                "Parameter that affects amount of smoothing with unit matrix "
                "in online preconditioning (larger -> more smoothing)");
    
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

    am_nnet.GetNnet().SwitchToOnlinePreconditioning(rank_in, rank_out, update_period,
                                                    num_samples_history, alpha);
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Copied neural net from " << nnet_rxfilename
              << " to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

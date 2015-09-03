// nnet2bin/nnet-am-rescale.cc

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
#include "nnet2/rescale-nnet.h"
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Rescale the parameters in a neural net to achieve certain target\n"
        "statistics, relating to the average derivative of the sigmoids\n"
        "measured at some supplied data.  This relates to how saturated\n"
        "the sigmoids are (we try to match the statistics of `good' neural\n"
        "nets).\n"
        "\n"
        "Usage:  nnet-am-rescale [options] <nnet-in> <examples-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-rescale 1.mdl valid.egs 1_rescaled.mdl\n";

    bool binary_write = true;
    NnetRescaleConfig config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        egs_rspecifier = po.GetArg(2), 
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    std::vector<NnetExample> egs;

    // This block adds samples to "egs".
    SequentialNnetExampleReader example_reader(
        egs_rspecifier);
    for (; !example_reader.Done(); example_reader.Next())
      egs.push_back(example_reader.Value());
    KALDI_LOG << "Read " << egs.size() << " examples.";
    KALDI_ASSERT(!egs.empty());
    
    RescaleNnet(config, egs, &am_nnet.GetNnet());
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Rescaled neural net and wrote it to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

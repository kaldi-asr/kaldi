// nnet2bin/nnet-to-raw-nnet.cc

// Copyright 2013  Johns Hopkins University (author:  Daniel Povey)

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
        "Copy a (cpu-based) neural net: reads the AmNnet with its transition model, but\n"
        "writes just the Nnet with no transition model (i.e. the raw neural net.)\n"
        "\n"
        "Usage:  nnet-to-raw-nnet [options] <nnet-in> <raw-nnet-out>\n"
        "e.g.:\n"
        " nnet-to-raw-nnet --binary=false 1.mdl 1.raw\n";

    int32 truncate = -1;
    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("truncate", &truncate, "If set, will truncate the neural net "
                "to this many components by removing the last components.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        raw_nnet_wxfilename = po.GetArg(2);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    if (truncate >= 0) {
      KALDI_LOG << "Truncating neural net to " << truncate << " layers.";
      am_nnet.GetNnet().Resize(truncate);
    }

    const Nnet &nnet = am_nnet.GetNnet();
    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);
    
    KALDI_LOG << "Read neural net from " << nnet_rxfilename
              << " and wrote raw neural net to " << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

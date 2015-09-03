// nnet2bin/nnet-am-fix.cc

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
#include "nnet2/nnet-fix.h"
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
        "but modify it to remove certain pathologies.  We use the average\n"
        "derivative statistics stored with the layers derived from\n"
        "NonlinearComponent.  Note: some processes, such as nnet-combine-fast,\n"
        "may not process these statistics correctly, and you may have to recover\n"
        "them using the --stats-from option of nnet-am-copy before you use.\n"
        "this program.\n"
        "\n"
        "Usage:  nnet-am-fix [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-fix 1.mdl 1_fixed.mdl\n"
        "or:\n"
        " nnet-am-fix --get-counts-from=1.gradient 1.mdl 1_shrunk.mdl\n";

    bool binary_write = true;
    NnetFixConfig config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
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

    FixNnet(config, &am_nnet.GetNnet());
    
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

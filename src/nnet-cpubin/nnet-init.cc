// nnet-cpubin/nnet-init.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet-cpu/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    // TODO: specify in the usage message where the example scripts are.
    const char *usage =
        "Initialize the neural network and its associated transition-model, from\n"
        "a tree, a topology file, and a neural-net config file where each line\n"
        "specifies one component (part of a layer).  See example scripts to see\n"
        "how this works in practice.\n"
        "\n"
        "Usage:  nnet-init [options] <tree-in> <topology-in> <config-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-init tree topo nnet.config 1.nnet\n";
    
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        topo_rxfilename = po.GetArg(2),
        config_rxfilename = po.GetArg(3),
        nnet_wxfilename = po.GetArg(4);
    
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_rxfilename, &ctx_dep);
    
    HmmTopology topo;
    ReadKaldiObject(topo_rxfilename, &topo);

    // Construct the transition model from the tree and the topology file.
    TransitionModel trans_model(ctx_dep, topo);

    AmNnet am_nnet;
    {
      bool binary;
      Input ki(config_rxfilename, &binary);
      KALDI_ASSERT(!binary && "Expect config file to contain text.");
      am_nnet.Init(ki.Stream());
    }

    if (am_nnet.NumPdfs() != trans_model.NumPdfs())
      KALDI_ERR << "Mismatch in number of pdfs, neural net has "
                << am_nnet.NumPdfs() << ", transition model has "
                << trans_model.NumPdfs();

    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Initialized neural net and wrote it to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



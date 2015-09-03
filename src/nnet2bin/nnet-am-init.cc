// nnet2bin/nnet-am-init.cc

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

    // TODO: specify in the usage message where the example scripts are.
    const char *usage =
        "Initialize the neural network acoustic model and its associated\n"
        "transition-model, from a tree, a topology file, and a neural-net\n"
        "without an associated acoustic model.\n"
        "See example scripts to see how this works in practice.\n"
        "\n"
        "Usage:  nnet-am-init [options] <tree-in> <topology-in> <raw-nnet-in> <nnet-am-out>\n"
        "or:  nnet-am-init [options] <transition-model-in> <raw-nnet-in> <nnet-am-out>\n"
        "e.g.:\n"
        " nnet-am-init tree topo \"nnet-init nnet.config - |\" 1.mdl\n";
        
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename, nnet_wxfilename;
    
    TransitionModel *trans_model = NULL;

    if (po.NumArgs() == 4) {
      std::string tree_rxfilename = po.GetArg(1),
          topo_rxfilename = po.GetArg(2);
      raw_nnet_rxfilename = po.GetArg(3);
      nnet_wxfilename = po.GetArg(4);
    
      ContextDependency ctx_dep;
      ReadKaldiObject(tree_rxfilename, &ctx_dep);
    
      HmmTopology topo;
      ReadKaldiObject(topo_rxfilename, &topo);

      // Construct the transition model from the tree and the topology file.
      trans_model = new TransitionModel(ctx_dep, topo);
    } else {
      std::string trans_model_rxfilename = po.GetArg(1);
      raw_nnet_rxfilename = po.GetArg(2);
      nnet_wxfilename = po.GetArg(3);
      trans_model = new TransitionModel();
      ReadKaldiObject(trans_model_rxfilename, trans_model);
    }
    
    AmNnet am_nnet;    
    {
      Nnet nnet;
      bool binary;
      Input ki(raw_nnet_rxfilename, &binary);
      nnet.Read(ki.Stream(), binary);
      am_nnet.Init(nnet);
    }
    
    if (am_nnet.NumPdfs() != trans_model->NumPdfs())
      KALDI_ERR << "Mismatch in number of pdfs, neural net has "
                << am_nnet.NumPdfs() << ", transition model has "
                << trans_model->NumPdfs();

    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model->Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    delete trans_model;
    KALDI_LOG << "Initialized neural net and wrote it to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}



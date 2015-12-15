// nnet3bin/nnet3-am-init.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "tree/context-dep.h"
#include "nnet3/am-nnet-multi.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize nnet3 am-nnet (i.e. neural network-based acoustic model, with\n"
        "associated transition model) from an existing transition model and nnet..\n"
        "Search for examples in scripts in /egs/wsj/s5/steps/nnet3/\n"
        "Set priors using nnet3-am-train-transitions or nnet3-am-adjust-priors\n"
        "\n"
        "Usage:  nnet3-am-init [options] <tree-in> <topology-in> <input-raw-nnet> <output-am-nnet>\n"
        "  or:  nnet3-am-init [options] <trans-model-in> <input-raw-nnet> <output-am-nnet>\n"
        "e.g.:\n"
        " nnet3-am-init tree topo 0.raw 0.mdl\n"
        "See also: nnet3-init, nnet3-am-copy, nnet3-am-info, nnet3-am-train-transitions,\n"
        " nnet3-am-adjust-priors\n";
    
    bool binary_write = true;
    int num_outputs = 2;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("num-outputs", &num_outputs, "number of outputs");
    
    po.Read(argc, argv);
    
/*    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }
*/
    std::string raw_nnet_rxfilename,
        am_nnet_wxfilename;
    vector<TransitionModel*> trans_models;
    
/*    if (false && po.NumArgs() == 4) {
      std::string tree_rxfilename = po.GetArg(1),
          topo_rxfilename = po.GetArg(2);
      raw_nnet_rxfilename = po.GetArg(3);
      am_nnet_wxfilename = po.GetArg(4);
    
      ContextDependency ctx_dep;
      ReadKaldiObject(tree_rxfilename, &ctx_dep);
    
      HmmTopology topo;
      ReadKaldiObject(topo_rxfilename, &topo);
      
      // Construct the transition model from the tree and the topology file.
      trans_model = new TransitionModel(ctx_dep, topo);
    }  else */ {
      std::vector<std::string> trans_model_rxfilenames;
      int i = 1;
      for (; i < po.NumArgs() - 1;) {
        trans_model_rxfilenames.push_back(po.GetArg(i++));
      }
      raw_nnet_rxfilename = po.GetArg(i++);
      am_nnet_wxfilename = po.GetArg(i++);
      
      trans_models.resize(num_outputs);
      for (int i = 0; i < num_outputs; i++) {
        trans_models[i] = new TransitionModel();
        ReadKaldiObject(trans_model_rxfilenames[i], trans_models[i]);
      }
    }

    KALDI_ASSERT(num_outputs == trans_models.size());

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);

    // priors won't be set yet.
    AmNnetMulti am_nnet(nnet, num_outputs);
    
    {
      Output ko(am_nnet_wxfilename, binary_write);
      WriteBasicType(ko.Stream(), binary_write, num_outputs);
      for (int i = 0; i < num_outputs; i++) {
        trans_models[i]->Write(ko.Stream(), binary_write);
        delete trans_models[i];
      }
      am_nnet.Write(ko.Stream(), binary_write);
    }
//    delete trans_model;
    for (int i = 0; i < num_outputs; i++) {
      delete trans_models[i];
    }
    KALDI_LOG << "Initialized am-nnet (neural net acoustic model) and wrote to "
              << am_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


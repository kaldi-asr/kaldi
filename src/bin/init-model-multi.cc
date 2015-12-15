// gmmbin/gmm-init-model-multi.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//                     Johns Hopkins University  (author: Guoguo Chen)
//           2015 Hainan Xu

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"
#include "tree/build-tree-utils.h"
#include "tree/build-tree-virtual.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Writes transition model for multi-decision tree\n"
        "Usage:  init-model-multi [options] <tree-in> "
        "<mapping-file> <topo-file> <model-in-prefix> <model-out> \n"
        "e.g.: \n"
        "  init-model-multi num-trees=2 tree mapping topo model out.mdl\n"
        "Files model-0, model-1 should be present in the path where 'model' is";

    bool binary = true;
    int32 num_trees;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-trees", &num_trees, "num trees");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        tree_filename = po.GetArg(1),
        mapping_file = po.GetArg(2),
        topo_filename = po.GetArg(3),
        model_in_filename = po.GetArg(4),
        model_out_filename = po.GetArg(5);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);


    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    unordered_map<int32, vector<int32> > mapping;
    {
      bool binary;
      Input input(mapping_file, &binary);
      ReadMultiTreeMapping(mapping, input.Stream(), binary);
    }
    std::vector<TransitionModel> trans_models(num_trees);
    for (size_t i = 0; i < num_trees; i++)
    {
      char temp[4];
      sprintf(temp, "-%d", (int)i);
      std::string file_affix(temp);
      bool binary;
      Input ki(model_in_filename + file_affix, &binary);
      trans_models[i].Read(ki.Stream(), binary);
    }

    TransitionModel trans_model(ctx_dep, mapping, trans_models);

    Output ko(model_out_filename, binary);
    trans_model.Write(ko.Stream(), binary);
    KALDI_LOG << "Wrote only transition model.";
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


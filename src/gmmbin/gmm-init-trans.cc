// gmmbin/gmm-init-trans.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "tree/build-tree.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize transition model given topo, tree and GMM (used for format conversion from HTK)\n"
        "Usage:  gmm-init-trans <topology-in> <gmm-in> <tree-in> <model-out>\n";

    bool binary = true;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_filename = po.GetArg(1);
    std::string gmm_filename = po.GetArg(2);
    std::string tree_filename = po.GetArg(3);
    std::string model_out_filename = po.GetArg(4);

    // Read toppology.
    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    // Read model.
    AmDiagGmm am_gmm;
    ReadKaldiObject(gmm_filename, &am_gmm);

    // Now the tree
    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    TransitionModel trans_model(ctx_dep, topo);

    {  // Write transition-model and GMM to one file in the normal Kaldi way.
      Output out(model_out_filename, binary);
      trans_model.Write(out.Stream(), binary);
      am_gmm.Write(out.Stream(), binary);
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


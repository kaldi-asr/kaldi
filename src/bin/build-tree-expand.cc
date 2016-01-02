// bin/build-tree-virtual.cc

// Copyright 2014 Hainan XU

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/context-dep-multi.h"
#include "tree/build-tree.h"
#include "tree/build-tree-virtual.h"
#include "tree/build-tree-expand.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

using std::string;
using std::vector;
using std::pair;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Expand the leaves of the input decision tree, add N questions\n"
        "Usage:  build-tree-virtual [options]"
        " <tree-in> <topo-file> <stats> <tree-out-prefix> \n";

    bool binary = true;
    int32 num_qst = 1;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-questions", &num_qst,
        "number of questions to expand the tree");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_in = po.GetArg(1),
        topo_filename = po.GetArg(2),
        questions_filename = po.GetArg(3),
        stats_filename = po.GetArg(4),
        tree_out_prefix = po.GetArg(5);

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    Questions qo;
    {
      bool binary_in;
      try {
        Input ki(questions_filename, &binary_in);
        qo.Read(ki.Stream(), binary_in);
      } catch (const std::exception &e) {
        KALDI_ERR << "Error reading questions file " << questions_filename
                  << ", error is: " << e.what();
      }
    }

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_in, &ctx_dep);
    vector<EventMap*> out = 
         ExpandDecisionTree(ctx_dep, stats, qo, num_qst);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

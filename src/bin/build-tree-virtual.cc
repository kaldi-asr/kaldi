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
        "Build one virtual decision tree from multiple decision trees\n"
        "Usage:  build-tree-virtual [options]"
        " <tree-prefix-in> <topo-file> <tree-out> <mapping-out> [ <stats> ]\n"
        "e.g.: \n"
        " build-tree-virtual num-trees=2 tree topo tree-out mapping stats\n"
        "The last argument is not neccesary. If provided, this program will"
        " show the joint-entropy of the virtual on the stats, as well as"
        " check the correctness of the virtual tree along with the mapping"
        " by mapping each stat to single/virtual trees and compare with the"
        " mapping file.";

    bool binary = true;
    int32 num_trees = 1;

    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-trees", &num_trees, "Number of source trees that we will"
                " use to build the virtual tree");

    po.Read(argc, argv);

    if (po.NumArgs() != 4 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_prefix = po.GetArg(1),
        topo_filename = po.GetArg(2),
        tree_out_filename = po.GetArg(3),
        mapping_out_filename = po.GetArg(4),
        stats_filename;

    if (po.NumArgs() == 5) {
      stats_filename = po.GetArg(5);
    }
    else {
      stats_filename = "";
    }

    BuildTreeStatsType stats;
    if (stats_filename != "") {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    // pointers owned
    vector<ContextDependency*> ctx_deps(num_trees);

    // pointers not owned,
    vector<EventMap*> trees(num_trees);
    vector<pair<int32, int32> > NPs(num_trees);

    for (int j = 0; j < num_trees; j++) {
      char temp[4];
      sprintf(temp, "-%d", j);
      std::string tree_affix(temp);

      ctx_deps[j] = new ContextDependency();
      ReadKaldiObject(tree_prefix + tree_affix, (ctx_deps[j]));

      NPs[j].first = ctx_deps[j]->ContextWidth();
      NPs[j].second = ctx_deps[j]->CentralPosition();

      trees[j] = &ctx_deps[j]->ToPdfMap();
    }

    ContextDependencyMulti ctx_dep_multi(NPs, trees, topo);

    unordered_map<int32, vector<int32> > mappings;
    EventMap* tree_out;
    ctx_dep_multi.GetVirtualTreeAndMapping(&tree_out, &mappings);

    if (num_trees == 1) {
      bool same = (trees[0]->IsSameTree(tree_out));
      KALDI_ASSERT(same);
    }

    if (stats_filename != "") {
      GaussClusterable *sum_of_all_stats = 
                dynamic_cast<GaussClusterable*>(SumStats(stats)); 

      std::vector<BuildTreeStatsType> split_stats;
      SplitStatsByMap(stats, *tree_out, &split_stats);
      double entropy = 0.0;
      for (size_t i = 0; i < split_stats.size(); i++) {
        GaussClusterable *sum = 
          dynamic_cast<GaussClusterable*>(SumStats(split_stats[i]));
        if (sum == NULL) continue;  // sometimes there is no stats
        entropy += (sum->count() / sum_of_all_stats->count())
           * log(sum_of_all_stats->count() / sum->count());
        delete sum;
      }
      delete sum_of_all_stats;

      KALDI_LOG << "the entropy for the virtual tree is " << entropy << "\n";

      std::vector<std::vector<BuildTreeStatsType> > vec_splits;

      KALDI_LOG << "Test correctness of virtual tree: "
                << "size is " << stats.size();
      for (size_t i = 0; i < stats.size(); i++) {
        EventType e = stats[i].first;
        EventAnswerType ans;
        // emap is of type EventMap; e is EventType
        bool ret = tree_out->Map(e, &ans);
        KALDI_ASSERT(ret == true);
        vector<int32> m = mappings[ans];
        for (size_t j = 0; j < num_trees; j++) {
          ret = trees[j]->Map(e, &ans);
          KALDI_ASSERT(ret && m[j] == ans);
        }
      }
    }

    Output treefile_out(tree_out_filename, binary);
    ctx_dep_multi.WriteVirtualTree(treefile_out.Stream(), binary);

    Output mapfile_output(mapping_out_filename, binary);
    WriteMultiTreeMapping(mappings, mapfile_output.Stream(), binary, num_trees);
    treefile_out.Close();
    mapfile_output.Close();
    // tree files are like "tree-2"

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

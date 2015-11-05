// bin/sum-bool-vector-stats.cc

// Copyright 2015   Vimal Manohar

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
#include "decision-tree/tree-stats.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::decision_tree_classifier;

  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Sum statistics for phonetic-context tree building.\n"
        "Usage:  sum-tree-stats [options] tree-accs-out tree-accs-in1 tree-accs-in2 ...\n"
        "e.g.: \n"
        " sum-tree-stats treeacc 1.treeacc 2.treeacc 3.treeacc\n";

    ParseOptions po(usage);
    bool binary = true;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    BooleanTreeStats tree_stats;
    
    std::string tree_stats_wxfilename = po.GetArg(1);

    // A reminder on what BuildTreeStatsType is:
    // typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;
    
    for (int32 arg = 2; arg <= po.NumArgs(); arg++) {
      std::string tree_stats_rxfilename = po.GetArg(arg);
      BooleanTreeStats other_tree_stats;
      ReadKaldiObject(tree_stats_rxfilename, &other_tree_stats);
      tree_stats.AddStats(other_tree_stats);
    }

    Output ko(tree_stats_wxfilename, binary);
    tree_stats.Write(ko.Stream(), binary);

    KALDI_LOG << "Wrote summed accs ( " << tree_stats.NumStats() << " individual stats)";
    return (tree_stats.NumStats() != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


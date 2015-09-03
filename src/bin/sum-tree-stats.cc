// bin/sum-tree-stats.cc

// Copyright 2009-2011  Microsoft Corporation, GoVivace Inc.

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
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "tree/build-tree-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
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

    std::map<EventType, Clusterable*> tree_stats;
    
    std::string tree_stats_wxfilename = po.GetArg(1);

    // A reminder on what BuildTreeStatsType is:
    // typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;
    
    for (int32 arg = 2; arg <= po.NumArgs(); arg++) {
      std::string tree_stats_rxfilename = po.GetArg(arg);
      bool binary_in;
      Input ki(tree_stats_rxfilename, &binary_in);
      BuildTreeStatsType stats_array;
      GaussClusterable example; // Lets ReadBuildTreeStats know which type to read..
      ReadBuildTreeStats(ki.Stream(), binary_in, example, &stats_array);
      for (BuildTreeStatsType::iterator iter = stats_array.begin();
           iter != stats_array.end(); ++iter) {
        EventType e = iter->first;
        Clusterable *c = iter->second;
        std::map<EventType, Clusterable*>::iterator map_iter = tree_stats.find(e);
        if (map_iter == tree_stats.end()) { // Not already present.
          tree_stats[e] = c;
        } else {
          map_iter->second->Add(*c);
          delete c;
        }
      }
    }

    BuildTreeStatsType stats;  // vectorized form.

    for (std::map<EventType, Clusterable*>::const_iterator iter = tree_stats.begin();  
        iter != tree_stats.end();
        iter++ ) {
      stats.push_back(std::make_pair(iter->first, iter->second));
    }
    tree_stats.clear();

    {
      Output ko(tree_stats_wxfilename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }
    KALDI_LOG << "Wrote summed accs ( " << stats.size() << " individual stats)";
    DeleteBuildTreeStats(&stats);
    return (stats.size() != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



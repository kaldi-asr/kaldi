// sgmmbin/sgmm-sum-tree-stats.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "tree/build-tree-utils.h"
#include "sgmm/sgmm-clusterable.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Sum SGMM-type statistics used for phonetic decision tree building.\n"
        "Usage:  sgmm-sum-tree-stats [options] tree-accs-out trea-accs-in1 tree-accs-in2 ...\n"
        "e.g.: sgmm-sum-tree-stats treeacc 1.streeacc 2.streeacc 3.streeacc\n";
    
    ParseOptions po(usage);
    bool binary = true;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string treeacc_wxfilename = po.GetArg(1);
    
    std::map<EventType, Clusterable*> tree_stats;

    AmSgmm am_sgmm; // dummy variable needed to initialize stats.
    std::vector<SpMatrix<double> > H; // also needed to initialize stats,
    // but never accessed in this program.
    
    // typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;    
    for (int32 arg = 2; arg <= po.NumArgs(); arg++) {
      std::string treeacc_rxfilename = po.GetArg(arg);
      bool binary_in;
      Input ki(treeacc_rxfilename, &binary_in);
      BuildTreeStatsType stats_array;
      SgmmClusterable example(am_sgmm, H); // Needed for its type information.
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

    BuildTreeStatsType stats;  // all the stats, in vectorized form.

    for (std::map<EventType, Clusterable*>::const_iterator iter = tree_stats.begin();  
        iter != tree_stats.end();
        iter++ ) {
      stats.push_back(std::make_pair(iter->first, iter->second));
    }
    tree_stats.clear();

    {
      Output ko(treeacc_wxfilename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }
    KALDI_LOG << "Wrote summed sgmm-treeaaccs: number of separate objects was "
              << stats.size();
    DeleteBuildTreeStats(&stats);
    return (stats.size() != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



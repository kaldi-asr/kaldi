// bin/build-tree-two-level.cc

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
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

namespace kaldi {
void GetSeenPhones(BuildTreeStatsType &stats, int P, std::vector<int32> *phones_out) {
  // Get list of phones that we saw (in the central position P, although it
  // shouldn't matter what position).

  std::set<int32> phones_set;
  for (size_t i = 0 ; i < stats.size(); i++) {
    const EventType &evec = stats[i].first;
    for (size_t j = 0; j < evec.size(); j++) {
      if (evec[j].first == P) {  // "key" is position P
        KALDI_ASSERT(evec[j].second != 0);
        phones_set.insert(evec[j].second);  // insert "value" of this
        // phone.
      }
    }
    CopySetToVector(phones_set, phones_out);
  }
}


}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Trains two-level decision tree.  Outputs the larger tree, and a mapping from the\n"
        "leaf-ids of the larger tree to those of the smaller tree.  Useful, for instance,\n"
        "in tied-mixture systems with multiple codebooks.\n"
        "\n"
        "Usage:  build-tree-two-level [options] <tree-stats-in> <roots-file> <questions-file> <topo-file> <tree-out> <mapping-out>\n"
        "e.g.: \n"
        " build-tree-two-level treeacc roots.txt 1.qst topo tree tree.map\n";

    bool binary = true;
    int32 P = 1, N = 3;

    bool cluster_leaves = true;
    int32 max_leaves_first = 1000;
    int32 max_leaves_second = 5000;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("context-width", &N, "Context window size [must match "
                "acc-tree-stats]");
    po.Register("central-position", &P, "Central position in context window "
                "[must match acc-tree-stats]");
    po.Register("max-leaves-first", &max_leaves_first, "Maximum number of "
                "leaves in first-level decision tree.");
    po.Register("max-leaves-second", &max_leaves_second, "Maximum number of "
                "leaves in second-level decision tree.");
    po.Register("cluster-leaves", &cluster_leaves, "If true, do a post-clustering"
                " of the leaves of the final decision tree.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_filename = po.GetArg(1),
        roots_filename = po.GetArg(2),
        questions_filename = po.GetArg(3),
        topo_filename = po.GetArg(4),
        tree_out_filename = po.GetArg(5),
        map_out_filename = po.GetArg(6);


    // Following 2 variables derived from roots file.
    // phone_sets is sets of phones that share their roots.
    // Just one phone each for normal systems.
    std::vector<std::vector<int32> > phone_sets;
    std::vector<bool> is_shared_root;
    std::vector<bool> is_split_root;
    {
      Input ki(roots_filename.c_str());
      ReadRootsFile(ki.Stream(), &phone_sets, &is_shared_root, &is_split_root);
    }

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
    std::cerr << "Number of separate statistics is " << stats.size() << '\n';

    Questions qo;
    {
      bool binary_in;
      try {
        Input ki(questions_filename, &binary_in);
        qo.Read(ki.Stream(), binary_in);
      } catch (const std::exception &e) {
        KALDI_ERR << "Error reading questions file "<<questions_filename<<", error is: " << e.what();
      }
    }


    std::vector<int32> phone2num_pdf_classes;
    topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

    EventMap *to_pdf = NULL;

    std::vector<int32> mapping;

    //////// Build the tree. ////////////

    to_pdf = BuildTreeTwoLevel(qo,
                               phone_sets,
                               phone2num_pdf_classes,
                               is_shared_root,
                               is_split_root,
                               stats,
                               max_leaves_first,
                               max_leaves_second,
                               cluster_leaves,
                               P,
                               &mapping);

    ContextDependency ctx_dep(N, P, to_pdf);  // takes ownership
    // of pointer "to_pdf", so set it NULL.
    to_pdf = NULL;

    WriteKaldiObject(ctx_dep, tree_out_filename, binary);

    {
      Output ko(map_out_filename, binary);
      WriteIntegerVector(ko.Stream(), binary, mapping); 
    }
    
    {  // This block is just doing some checks.

      std::vector<int32> all_phones;
      for (size_t i = 0; i < phone_sets.size(); i++)
        all_phones.insert(all_phones.end(),
                          phone_sets[i].begin(), phone_sets[i].end());
      SortAndUniq(&all_phones);
      if (all_phones != topo.GetPhones()) {
        std::ostringstream ss;
        WriteIntegerVector(ss, false, all_phones);
        ss << " vs. ";
        WriteIntegerVector(ss, false, topo.GetPhones());
        KALDI_WARN << "Mismatch between phone sets provided in roots file, and those in topology: " << ss.str();
      }
      std::vector<int32> phones_vec;  // phones we saw.
      GetSeenPhones(stats, P, &phones_vec);

      std::vector<int32> unseen_phones;  // diagnostic.
      for (size_t i = 0; i < all_phones.size(); i++)
        if (!std::binary_search(phones_vec.begin(), phones_vec.end(), all_phones[i]))
          unseen_phones.push_back(all_phones[i]);
      for (size_t i = 0; i < phones_vec.size(); i++)
        if (!std::binary_search(all_phones.begin(), all_phones.end(), phones_vec[i]))
          KALDI_ERR << "Phone "<< (phones_vec[i]) << " appears in stats but is not listed in roots file.";
      if (!unseen_phones.empty()) {
        std::ostringstream ss;
        for (size_t i = 0; i < unseen_phones.size(); i++)
          ss << unseen_phones[i] << ' ';
        // Note, unseen phones is just a warning as in certain kinds of
        // systems, this can be OK (e.g. where phone encodes position and
        // stress information).
        KALDI_WARN << "Saw no stats for following phones: " << ss.str();
      }
    }

    std::cerr << "Wrote tree and mapping\n";
    
    DeleteBuildTreeStats(&stats);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

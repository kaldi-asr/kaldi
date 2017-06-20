// bin/acc-tree-stats.cc

// Copyright 2009-2011  Microsoft Corporation, GoVivace Inc.
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "hmm/tree-accu.h"

/** @brief Accumulate tree statistics for decision tree training. The
program reads in a feature archive, and the corresponding alignments,
and generates the sufficient statistics for the decision tree
creation. Context width and central phone position are used to
identify the contexts.Transition model is used as an input to identify
the PDF's and the phones.  */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate statistics for phonetic-context tree building.\n"
        "Usage:  acc-tree-stats [options] <model-in> <features-rspecifier> <alignments-rspecifier> <tree-accs-out>\n"
        "e.g.: \n"
        " acc-tree-stats 1.mdl scp:train.scp ark:1.ali 1.tacc\n";

    bool binary = true;
    AccumulateTreeStatsOptions opts;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignment_rspecifier = po.GetArg(3),
        accs_out_wxfilename = po.GetOptArg(4);


    AccumulateTreeStatsInfo acc_tree_stats_info(opts);

    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      // There is more in this file but we don't need it.
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);

    std::map<EventType, GaussClusterable*> tree_stats;

    int num_done = 0, num_no_alignment = 0, num_other_error = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!alignment_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignment_reader.Value(key);

        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (alignment.size())<<" vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        AccumulateTreeStats(trans_model,
                            acc_tree_stats_info,
                            alignment,
                            mat,
                            &tree_stats);
        num_done++;
        if (num_done % 1000 == 0)
          KALDI_LOG << "Processed " << num_done << " utterances.";
      }
    }

    BuildTreeStatsType stats;  // vectorized form.

    for (std::map<EventType, GaussClusterable*>::const_iterator iter = tree_stats.begin();
         iter != tree_stats.end();
         ++iter) {
      stats.push_back(std::make_pair(iter->first, iter->second));
    }
    tree_stats.clear();

    {
      Output ko(accs_out_wxfilename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }
    KALDI_LOG << "Accumulated stats for " << num_done << " files, "
              << num_no_alignment << " failed due to no alignment, "
              << num_other_error << " failed for other reasons.";
    KALDI_LOG << "Number of separate stats (context-dependent states) is "
              << stats.size();
    DeleteBuildTreeStats(&stats);
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

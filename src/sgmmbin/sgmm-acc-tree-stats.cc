// sgmmbin/sgmm-acc-tree-stats.cc

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
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate statistics for decision tree training.\n"
        "This version accumulates statistics in the form of state-specific "
        "SGMM stats; you need to use the program sgmm-build-tree to build "
        "the tree (and sgmm-sum-tree-accs to sum the stats).\n"
        "Usage:  sgmm-acc-tree-stats [options] sgmm-model-in features-rspecifier "
        "alignments-rspecifier [tree-accs-out]\n"
        "e.g.: sgmm-acc-tree-stats --ci-phones=48:49 1.mdl scp:train.scp ark:1.ali 1.tacc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    string ci_phones_str;
    int N = 3, P = 1;
    SgmmGselectConfig sgmm_opts;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices (rspecifier)");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("ci-phones", &ci_phones_str, "Colon-separated list of integer "
                "indices of context-independent phones.");
    po.Register("context-width", &N, "Context window size.");
    po.Register("central-position", &P,
                "Central context-window position (zero-based)");
    sgmm_opts.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string sgmm_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignment_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetOptArg(4);
    
    std::vector<int32> ci_phones;
    if (ci_phones_str != "") {
      SplitStringToIntegers(ci_phones_str, ":", false, &ci_phones);
      std::sort(ci_phones.begin(), ci_phones.end());
      if (!IsSortedAndUniq(ci_phones) || ci_phones[0] == 0) {
        KALDI_ERR << "Invalid set of ci_phones: " << ci_phones_str;
      }
    }

    TransitionModel trans_model;
    AmSgmm am_sgmm;
    std::vector<SpMatrix<double> > H; // Not initialized in this program-- not needed
    // as we don't call Objf() from stats.
    {
      bool binary;
      Input ki(sgmm_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    if (gselect_rspecifier.empty())
      KALDI_ERR << "--gselect option is required.";
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);
    
    std::map<EventType, SgmmClusterable*> tree_stats;

    int num_done = 0, num_err = 0;  
    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!alignment_reader.HasKey(utt)) {
        num_err++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignment_reader.Value(utt);

        if (!gselect_reader.HasKey(utt) ||

            gselect_reader.Value(utt).size() != mat.NumRows()) {
          KALDI_WARN << "No gselect information for utterance " << utt
                     << " (or wrong size)";
          num_err++;
          continue;
        }

        const std::vector<std::vector<int32> > &gselect =
            gselect_reader.Value(utt);
        
        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (alignment.size())<<" vs. "<< (mat.NumRows());
          num_err++;
          continue;
        }

        SgmmPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
          }
        } // else spk_vars is "empty"
        
        
        //  The work gets done here.
        if (!AccumulateSgmmTreeStats(trans_model,
                                     am_sgmm,
                                     H,
                                     N, P, 
                                     ci_phones,
                                     alignment,
                                     gselect,
                                     spk_vars,
                                     mat,
                                     &tree_stats)) {
          num_err++;
        } else {
          num_done++;
          if (num_done % 1000 == 0)
            KALDI_LOG << "Processed " << num_done << " utterances.";
        }
      }
    }

    BuildTreeStatsType stats; // Converting from a map to a vector of pairs.
    
    for (std::map<EventType, SgmmClusterable*>::const_iterator iter = tree_stats.begin();  
        iter != tree_stats.end();
        iter++ ) {
      stats.push_back(std::make_pair(iter->first, static_cast<Clusterable*>(iter->second)));
    }
    tree_stats.clear();

    {
      Output ko(accs_wxfilename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }
    KALDI_LOG << "Accumulated stats for " << num_done << " files, "
              << num_err << " failed.";
    KALDI_LOG << "Number of separate stats (context-dependent states) is "
              << stats.size();
    DeleteBuildTreeStats(&stats);
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



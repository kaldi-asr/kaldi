// latbin/lattice-add-nnlmscore.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)
//                2021  Johns Hopkins University (author: Ke Li)
//
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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Add estimated neural language model scores of all arcs in a lattice\n"
        "back to the lattice for rescoring.\n"
        "Usage: lattice-add-nnlmscore [options] <lattice-rspecifier> <nnlm-scores-rspecifier> <lattice-wspecifier>\n"
        " e.g.: lattice-add-nnlmscore --lm-scale=0.8 ark:in.lats nnlm_scores.txt ark:out.lats\n"; 
    ParseOptions po(usage);
    BaseFloat lm_scale = 1.0;
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "scores.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1); 
    }
    
    std::string lats_rspecifier = po.GetArg(1),
        scores_rxfilename = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);
    
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier); // write as compact.

    // Read estimated neural language model scores for each arc of a lattice.
    typedef unordered_map<std::string, unordered_map<std::pair<int32, int32>,
            double, PairHasher<int32> >, StringHasher > ScoreMapType;
    ScoreMapType nnlm_scores;
    std::ifstream read_scores(scores_rxfilename);
    if (!read_scores) {
      KALDI_ERR << "Cannot open input file.";
    }
    std::string line;
    while (std::getline(read_scores, line)) {
      std::istringstream scores(line);
      std::string key;
      int32 arc_start_state, arc_end_state;
      double score;
      scores >> key >> arc_start_state >> arc_end_state >> score;
      std::pair<int32, int32> arc_index =
        std::make_pair(arc_start_state, arc_end_state);
      nnlm_scores[key][arc_index] = lm_scale * score;
    }

    typedef ScoreMapType::const_iterator ScoreIter;
    ScoreIter iter;
    int32 n_done = 0;

    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string key = compact_lattice_reader.Key();
      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();
      
      iter = nnlm_scores.find(key);
      KALDI_ASSERT(iter != nnlm_scores.end()); 
      unordered_map<std::pair<int32, int32>, double,
        PairHasher<int32> > arc_to_score;
      arc_to_score = nnlm_scores[key];
      AddNnlmScoreToCompactLattice(arc_to_score, &clat); 
      compact_lattice_writer.Write(key, clat);
      n_done++; 
    }
    KALDI_LOG << "Done with adding neural language model scores to " <<
      n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

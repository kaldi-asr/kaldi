// latbin/lattice-path-cover.cc

// Copyright 2021 Johns Hopkins University (author: Ke Li) 

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

namespace kaldi {

// This class computes a minimal set of paths (path cover) from a lattice.
// The paths have two properties:
// 1) Every arc in the lattice must be covered by the set of paths.
// 2) Each path is the best path that includes at least one arc on the lattice.

class PathCoverComputer {
 public:
  typedef CompactLatticeArc::StateId StateId;
  typedef std::pair<std::vector<StateId>, double> PathType;
  
  // Note: 'clat' must be topologically sorted.
  PathCoverComputer(const CompactLattice &clat):clat_(clat) {
    KALDI_ASSERT(clat.NumStates() > 1);
    // Each arc is indexed by a state pair (start_state, end_state)
    for (StateId s = 0; s < clat.NumStates(); s++) {
      for (fst::ArcIterator<CompactLattice> aiter(clat, s);
           !aiter.Done(); aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        StatePair arc_index = std::make_pair(s, arc.nextstate);
        arc_stats_[arc_index] = false; // 'false' means the best path including
                                       // this arc has not been generated yet. 
      }
    } 
  }
  
  // Compute the best path from the start state to a state.
  void GetFirstPartialBestPath(StateId cur_state,
                               std::vector<StateId> *partial_best_path) {
    partial_best_path->clear();
    partial_best_path->push_back(cur_state);
    while (cur_state != clat_.Start()) {
      StateId prev_state = forward_best_costs_and_preds_[cur_state].second;
      if (prev_state == fst::kNoStateId) {
        KALDI_WARN << "Failure in path cover algorithm for lattice.";
        return;
      }
      KALDI_ASSERT(cur_state != prev_state && "Lattice with cycles");
      partial_best_path->push_back(prev_state);
      cur_state = prev_state;
    }
    // Make sure the best path starts from the start state.
    std::reverse(partial_best_path->begin(), partial_best_path->end());
  }

  // Compute the best path from a state to a final state.
  void GetSecondPartialBestPath(StateId cur_state,
                                std::vector<StateId> *partial_best_path) {
    partial_best_path->clear();
    partial_best_path->push_back(cur_state);
    while (backward_best_costs_and_preds_[cur_state].second !=
           fst::kNoStateId) {
      StateId next_state = backward_best_costs_and_preds_[cur_state].second;
      if (next_state == fst::kNoStateId) {
        KALDI_WARN << "Failure in path cover algorithm for lattice.";
        return;
      }
      KALDI_ASSERT(cur_state != next_state && "Lattice with cycles");
      partial_best_path->push_back(next_state);
      cur_state = next_state;
    }
  }
  
  // Compute state sequences and costs of the set of paths.
  void ComputeStateSeqsAndCosts(std::vector<PathType> *paths_and_costs) {
    paths_and_costs->clear();
    for (StateId state = 0; state < clat_.NumStates(); state++) {
      for (fst::ArcIterator<CompactLattice> aiter(clat_, state);
           !aiter.Done(); aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        StatePair cur_arc_index = std::make_pair(state, arc.nextstate);
        // Check whether the best path including current arc has been covered.
        StateId prev_state = forward_best_costs_and_preds_[state].second;
        StatePair prev_arc_index = std::make_pair(prev_state, state);
        StateId next_state = backward_best_costs_and_preds_[state].second;
        if (arc_stats_[prev_arc_index] == true &&
            next_state == arc.nextstate) {
          arc_stats_[cur_arc_index] = true;
          continue; // Avoid generating repeated best paths.
        }
        // Get the state sequence of the best path including current arc.
        std::vector<StateId> first_partial_best_seq, second_partial_best_seq;
        GetFirstPartialBestPath(state, &first_partial_best_seq);
        GetSecondPartialBestPath(arc.nextstate, &second_partial_best_seq);
        first_partial_best_seq.insert(first_partial_best_seq.end(),
                                      second_partial_best_seq.begin(),
                                      second_partial_best_seq.end());
        arc_stats_[cur_arc_index] = true;
        // Compute the cost of the best path including current arc.
        double forward_cost = forward_best_costs_and_preds_[state].first,
               backward_cost = ConvertToCost(arc.weight) +
                 backward_best_costs_and_preds_[arc.nextstate].first,
               best_cost = forward_cost + backward_cost;
        std::pair<std::vector<StateId>, double> cur_path =
          std::make_pair(first_partial_best_seq, best_cost);
        paths_and_costs->push_back(cur_path);
      }
    }
  }
  
  // Convert the generated paths into CompactLattice format as output.
  void ConvertToCompactLattice(const std::vector<PathType> &paths_and_costs,
                               std::vector<CompactLattice> *cover_clats) {
    cover_clats->clear();
    for (int32 i = 0; i < paths_and_costs.size(); i++) {
      std::vector<StateId> path = paths_and_costs[i].first;
      CompactLattice cur_clat;
      for (StateId s = 0; static_cast<int32>(s) < path.size(); s++) {
        cur_clat.AddState();
        if (s == 0) cur_clat.SetStart(s);
        if (static_cast<int32>(s + 1) < path.size()) { // transition to next state.
          bool have_arc = false;
          CompactLatticeArc cur_arc;
          for (fst::ArcIterator<CompactLattice> aiter(clat_, path[s]);
               !aiter.Done(); aiter.Next()) {
            const CompactLatticeArc &arc = aiter.Value();
            if (arc.nextstate == path[s + 1]) {
              if (!have_arc ||
                  ConvertToCost(arc.weight) < ConvertToCost(cur_arc.weight)) {
                cur_arc = arc;
                have_arc = true;
              }
            }
          }
          KALDI_ASSERT(have_arc && "Code error.");
          cur_clat.AddArc(s, CompactLatticeArc(cur_arc.ilabel, cur_arc.olabel,
                                               cur_arc.weight, s + 1));
        } else { // final-weight.
          cur_clat.SetFinal(s, clat_.Final(path[s]));
        }
      }
      cover_clats->push_back(cur_clat);
    }
  }
  
  // Compute and sort the set of best paths that include each arc.
  void ComputePathCover(std::vector<PathType> *paths_and_costs) {
    // Compute forward and backward best (viterbi) costs and traceback states.
    CompactLatticeBestCostsAndTracebacks(clat_, &forward_best_costs_and_preds_,
                                         &backward_best_costs_and_preds_);
    ComputeStateSeqsAndCosts(paths_and_costs);
    // Sort generated best paths by their costs (from the best to the worst).
    std::sort(paths_and_costs->begin(), paths_and_costs->end(),
              [](const std::pair<std::vector<StateId>, double> &a,
                 const std::pair<std::vector<StateId>, double> &b) {
                return a.second < b.second;
              });
  }

 private:
  const CompactLattice &clat_;
  std::vector<std::pair<double, StateId> > forward_best_costs_and_preds_;
  std::vector<std::pair<double, StateId> > backward_best_costs_and_preds_;
  
  typedef std::pair<StateId, StateId> StatePair;
  typedef unordered_map<StatePair, bool, PairHasher<StateId> > MapType;
  MapType arc_stats_;
};

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;

    const char *usage =
        "Generate minimal paths that can cover every arc on a lattice, and each\n"
        "path must be the best path for at least one arc it includes. Output\n"
        "transcription, state sequence, and cost of each path in each lattice.\n"
        "This binary is mainly used for parallel lattice rescoring with a\n"
        "neural LM trained with PyTorch (or other tool). An example can be\n"
        "found in local/pytorchnn/run_nnlm.sh on Switchboard dataset.\n"
        "Usage: lattice-path-cover [options] <lattice-rspecifier> [ <transcriptions-wspecifier> ] [ <states-wspecifier>] [ <path-costs-wspecifier>]\n"
        " e.g.: lattice-path-cover --acoustic-scale=0.1 --word-symbol-table=data/lang/words.txt ark:1.lats ark,t:1.words, ark,t:1.states, ark,t:1.costs\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_scale = 1.0;
    std::string word_syms_filename;
    
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for "
                "acoustic likelihoods.");
    po.Register("lm-scale", &lm_scale, "Scaling factor for language model "
                "scores.");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for "
                "words [for debug output]");

    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        transcriptions_wspecifier = po.GetOptArg(2),
        states_wspecifier = po.GetOptArg(3),
        path_costs_wspecifier = po.GetOptArg(4);

    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    Int32VectorWriter transcriptions_writer(transcriptions_wspecifier);
    Int32VectorWriter states_writer(states_wspecifier);
    DoubleWriter path_costs_writer(path_costs_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    int32 n_done = 0, n_fail = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
      kaldi::TopSortCompactLatticeIfNeeded(&clat);
      kaldi::PathCoverComputer computer(clat);
      std::vector<std::pair<std::vector<CompactLatticeArc::StateId>, double> >
        cover_paths;
      computer.ComputePathCover(&cover_paths);
      std::vector<CompactLattice> cover_clats;
      computer.ConvertToCompactLattice(cover_paths, &cover_clats);
      for (int32 i = 0; i < cover_paths.size(); i++) {
        std::string cur_key = key + "-" + std::to_string(i + 1);
        std::vector<CompactLatticeArc::StateId> states_of_path =
          cover_paths[i].first;
        double path_cost = cover_paths[i].second;
        CompactLattice clat_path = cover_clats[i];
        Lattice path;
        ConvertLattice(clat_path, &path);
        if (path.Start() == fst::kNoStateId) {
          KALDI_WARN << "Path failed for key " << cur_key;
          n_fail++;
        } else {
          std::vector<int32> alignment;
          std::vector<int32> words;
          LatticeWeight weight;
          GetLinearSymbolSequence(path, &alignment, &words, &weight);
          if (transcriptions_wspecifier != "")
            transcriptions_writer.Write(cur_key, words);
          if (states_wspecifier != "")
            states_writer.Write(cur_key, states_of_path);
          if (path_costs_wspecifier != "")
            path_costs_writer.Write(cur_key, path_cost);
          if (word_syms != NULL) {
            std::cerr << cur_key << ' ';
            for (size_t i = 0; i < words.size(); i++) {
              std::string s = word_syms->Find(words[i]);
              if (s == "")
                KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
              std::cerr << s << ' ';
            }
            std::cerr << '\n';
          }
          n_done++;
        }
      } // done with one lattice
    } // done with all lattices

    KALDI_LOG << "Done " << n_done << " paths, failed for " << n_fail;

    delete word_syms;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

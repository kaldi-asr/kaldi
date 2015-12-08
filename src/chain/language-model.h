// chain/language-model.h

// Copyright      2015  Johns Hopkins University (Author: Daniel Povey)

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


#ifndef KALDI_CHAIN_LANGUAGE_MODEL_H_
#define KALDI_CHAIN_LANGUAGE_MODEL_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {


namespace chain {

// Options for phone language model estimation.  This is similar to an
// un-smoothed language model of a certain order (e.g. triphone).  We won't be
// actually decoding with this, we'll just use it as the 'denominator graph' in
// acoustic model estimation.  The reason for avoiding smoothing is to reduce
// the number of transitions in the language model, which will improve
// efficiency of training.

struct LanguageModelOptions {
  int32 ngram_order;  // you might want to tune this
  int32 num_lm_states;  // you also might want to tune this
  int32 no_prune_ngram_order;  // e.g. set this to 3 and it won't prune the
                               // trigram contexts (note: a trigram
                               // history-state has 2 known left phones)... this
                               // tends to make for a more compact graph (since
                               // the context FST anyway expands to trigram).

  LanguageModelOptions():
      ngram_order(5),
      num_lm_states(10000),
      no_prune_lm_order(3) { }

  void Register(OptionsItf *opts) {
    opts->Register("ngram-order", &ngram_order, "n-gram order for the phone "
                   "language model used for the 'denominator model'");
    opts->Register("num-lm-states", &num_lm_states, "Maximum number of language "
                   "model states allowed.  We do hard backoff to lower-order "
                   "n-gram to limit the num-states; pruning of states is based "
                   "on data-count.");
    opts->Register("no-prune-ngram-order", &no_prune_ngram_order, "n-gram order "
                   "below which the language model is not pruned (should "
                   "probably be set the same as your --context-width for phone "
                   "context in tree building, to make the graph as compact as "
                   "possible)");
  }
};

/**
   This LanguageModelEstimator class estimates an n-gram language model
   with a kind of 'hard' backoff that is intended to reduce the number of
   arcs in the final compiled FST.  Basically, we never back off to the lower-order
   n-gram state, but we sometimes do just say, "this state's count is too small
   so we won't have this state at all", and this LM state disappears and
   transitions to it go to the lower-order n-gram's state.

   This language model is implemented as a set of states, and transitions
   between these states; there is no concept of a backoff transition here.
   Because this maps very naturally to an FST, we output it as an FST.
 */
class LanguageModelEstimator {
 public:
  LanguageModelEstimator(LanguageModelOptions &opts): opts_(opts) {
    KALDI_ASSERT(opts.ngram_order >= 2);
  }

  // Adds counts for this sentence.  Basically does: for each n-gram in the
  // sentence, count[n-gram] += 1.  The only constraint on 'sentence' is that it
  // should contain no zeros.
  void AddCounts(const std::vector<int32> &sentence);

  // Estimates the LM and outputs it as an FST.  Note: there is
  // no concept here of backoff arcs.
  void Estimate(fst::StdVectorFst *fst) const;

 protected:
  struct LmState {
    // the phone history associated with this state (length can vary).
    std::vector<int32> history;
    // maps from
    std::map<int32, int32> phone_to_count;
    // total count of this state.
    int32 tot_count;
    // LM-state index of the backoff LM state (if it exists, else -1)...
    // provided for convenience.
    int32 backoff_lmstate_index;

    // this is only set after we decide on the FST state numbering (at the end).
    // If not set, it's -1.
    int32 fst_state;

    void AddCount(int32 phone, int32 count);

    // Log-likelihood of data in this case, summed, not averaged:
    // i.e. sum(phone in phones) count(phone) * log-prob(phone | this state).
    BaseFloat LogLike();
    // Add the contents of another LmState.
    void Add(const LmState &other);

    LmState(): tot_count(0), backoff_lmstate_index(-1), fst_state(-1) { }
    LmState(const LmState &other):
      history(other.history), phone_to_count(other.phone_to_count),
      tot_count(other.tot_count), backoff_lmstate_index(other.backoff_lmstate_index),
      fst_state(other.fst_state) { }
  };

  // maps from history to int32
  typedef unordered_map<std::vector<int32>, int32, VectorHasher<int32> > MapType;

  LanguageModelOptions opts_;

  MapType hist_to_lmstate_index_;
  std::vector<LmState> lm_states_;  // indexed by lmstate_index, the LmStates.

  // adds the counts for this ngram (called from AddCounts()).
  inline void IncrementCount(const std::vector<int32> &ngram);

  // Computes the cost, in log-likelihood, of backing off lm_state to its
  // backoff state, i.e. combining its counts with those of its backoff state.
  // As some special cases: if this state has a zero count, the cost is infinity
  // (no point backing off a state that doesn't exist yet), and if the backoff
  // state has a zero count but this state has a nonzero count, we set the cost
  // to 1e-15 * (count of this state).  Before the backoff states have any
  // counts, this encourages the lowest-count states to get backed-off first.
  BaseFloat ComputeBackoffCost(int32 lm_state);


  // For each history-state, makes sure that all backoff history states down to
  // the history state of length determined by no_prune_ngram_order exist.
  // (They won't have any counts).
  void AugmentHistories();

  inline static void AddCountToMap(const std::vector<int32> &key,
                                   int32 value,
                                   MapType *map);

  // copies elements of map to a vector.
  inline static void CopyMapToVector(
      const MapType &map,
      std::vector<std::pair<int32, std::vector<int32> > > *vec);
  // copies keys of a map to a set.
  inline static void CopyMapKeysToSet(
      const MapType &map,
      SetType *set);


  // Gets total count for each history (each history is a sequence of phones of
  // length n-1).
  void GetHistoryCounts(MapType *hist_counts) const;

  // Augment 'hist_counts' with counts for all backed-off history states
  // down to history states of length 1: basically,
  // for key in keys(*hist_counts) {
  //   count = (*hist_counts)[key];
  //   while (key.size() > 1) { key.shift(); (*hist_counts)[key] += count; }
  void AugmentHistoryCountsWithBackoff(MapType *hist_counts) const;

  // comparator object used in GetHistoryStates.  Used to sort history counts
  // from least to greatest.  If count is the same, treats longer histories as
  // having smaller counts (so they will be deleted first).
  struct HistoryCountCompare {
    // this is treated as an operator <.
    bool operator () (const std::pair<int32, std::vector<int32> > &a,
                      const std::pair<int32, std::vector<int32> > &b) {
      // we primarily compare on the data-count (the .first), but if the
      // data-count is the same, if a has a longer .second vector, which means
      // it's a higher order n-gram, we return true (interpreted as a < b),
      // meaning a is considered as having a lower count; this will ensure that if
      // counts are the same, higher-order states get deleted first, to ensure
      // that backoff states of existing states still exist.
      if (a.first < b.first) return true;
      else if (a.first > b.first) return false;
      else return a.second.size() > b.second.size();
    }
  };

  // Works out the set of history-states that will be included in the LM.
  void GetHistoryStates(SetType *history_states) const;


  // This function gets a map from a 'history' (i.e. a sequence of phones of
  // length >= 0 and <= opts.ngram_order - 1) to a history-state; each element of
  // 'hist_states' gets its own integer.  Returns the number of history-states,
  // which are numbered from zero.
  int32 GetHistoryToStateMap(const SetType &hist_states,
                             MapType *hist_to_state) const;

  // This function creates a map 'state_transitions' that maps the
  // pair (history-state, phone) to the next history state that we transition to
  // after seeing that phone.  If the phone is 0, we don't add an entry
  // (it becomes a final-prob).
  // Note: it also reads the raw counts from counts_.
  // Returns the initial state.
  int32 GetStateTransitions(
      const MapType &hist_to_state,
      PairMapType *transitions) const;


  // Given a 'hist_to_state' map, and a history vector (representing some words
  // of context), this function returns the state corresponding to a given
  // history vector.  This may involve backoff.  The vector 'hist' must be
  // nonempty.  If there is no such state, we'll throw an exception.
  static int32 GetStateForHist(const MapType &hist_to_state,
                               std::vector<int32> hist);


  // Creates the counts, in the format:
  //   num-count = (history-state, symbol) -> count,
  //   den-count = history-state -> count
  // where a particular LM probability will be written as
  //  num-count / den-count.
  void GetCounts(const MapType &hist_to_state,
                 int32 num_history_states,
                 PairMapType *num_counts,
                 std::vector<int32> *den_counts) const;

  void OutputToFst(
      int32 initial_state,
      const unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > &num_counts,
      const std::vector<int32> &den_counts,
      const unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > &transitions,
      fst::StdVectorFst *fst) const;

};



}  // namespace ctc
}  // namespace kaldi

#endif


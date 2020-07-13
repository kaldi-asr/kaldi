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
  int32 num_extra_lm_states;  // you also might want to tune this
  int32 no_prune_ngram_order;  // e.g. set this to 3 and it won't prune the
                               // trigram contexts (note: a trigram
                               // history-state has 2 known left phones)... this
                               // tends to make for a more compact graph (since
                               // the context FST anyway expands to trigram).

  LanguageModelOptions():
      ngram_order(4),
      num_extra_lm_states(1000),
      no_prune_ngram_order(3) { }

  void Register(OptionsItf *opts) {
    opts->Register("ngram-order", &ngram_order, "n-gram order for the phone "
                   "language model used for the 'denominator model'");
    opts->Register("num-extra-lm-states", &num_extra_lm_states, "Number of LM "
                   "states desired on top of the number determined by the "
                   "--no-prune-ngram-order option.");
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
  LanguageModelEstimator(LanguageModelOptions &opts): opts_(opts),
                                                      num_active_lm_states_(0) {
    KALDI_ASSERT(opts.ngram_order >= 1 && opts.no_prune_ngram_order >= 1);
  }

  // Adds counts for this sentence.  Basically does: for each n-gram in the
  // sentence, count[n-gram] += 1.  The only constraint on 'sentence' is that it
  // should contain no zeros.
  void AddCounts(const std::vector<int32> &sentence);

  // Estimates the LM and outputs it as an FST.  Note: there is
  // no concept here of backoff arcs.
  void Estimate(fst::StdVectorFst *fst);

 protected:
  struct LmState {
    // the phone history associated with this state (length can vary).
    std::vector<int32> history;
    // maps from
    std::map<int32, int32> phone_to_count;
    // total count of this state.  As we back off states to lower-order states
    // (and note that this is a hard backoff where we completely remove un-needed
    // states) this tot_count may become zero.
    int32 tot_count;

    // total count of this state plus all states that back off to this state.
    // only valid after SetParentCounts() is called.
    int32 tot_count_with_parents;

    // LM-state index of the backoff LM state (if it exists, else -1)...
    // provided for convenience.  The backoff state exist if and only
    // if history.size() >= no_prune_ngram_order
    int32 backoff_lmstate_index;

    // keeps track of the number of other LmStates 'other' for whom
    // (other.tot_count > 0 or other.num_parents > 0) and
    // other.backoff_lmstate_index is the index of this LM state.
    // This lets us know whether this state has a chance, in the future,
    // of getting a nonzero count, which in turn is used in the
    // BackoffAllowed() function.
    int32 num_parents;

    // this is only set after we decide on the FST state numbering (at the end).
    // If not set, it's -1.
    int32 fst_state;

    // True if backoff of this state is allowed (which implies it's in the queue).
    // Backoff of this state is allowed (i.e. we will consider removing this state)
    // if its history length is >= opts.no_prune_ngram_order, and it has nonzero
    // count, and
    bool backoff_allowed;

    void AddCount(int32 phone, int32 count);

    // Log-likelihood of data in this case, summed, not averaged:
    // i.e. sum(phone in phones) count(phone) * log-prob(phone | this state).
    BaseFloat LogLike() const;
    // Add the contents of another LmState.
    void Add(const LmState &other);
    // Clear all counts from this state.
    void Clear();
    LmState(): tot_count(0), tot_count_with_parents(0),  backoff_lmstate_index(-1),
               fst_state(-1), backoff_allowed(false) { }
    LmState(const LmState &other):
        history(other.history), phone_to_count(other.phone_to_count),
        tot_count(other.tot_count), tot_count_with_parents(other.tot_count_with_parents),
        backoff_lmstate_index(other.backoff_lmstate_index),
        fst_state(other.fst_state), backoff_allowed(other.backoff_allowed) { }
  };

  // maps from history to int32
  typedef unordered_map<std::vector<int32>, int32, VectorHasher<int32> > MapType;

  LanguageModelOptions opts_;

  MapType hist_to_lmstate_index_;
  std::vector<LmState> lm_states_;  // indexed by lmstate_index, the LmStates.

  // Keeps track of the number of lm states that have nonzero counts.
  int32 num_active_lm_states_;

  // The number of LM states that we would have due to the
  // no_prune_ngram_order_.  Equals the number of history-states of length
  // no_prune_ngram_order_ - 1.  Used to compute the total number of desired
  // state (by adding opts_.num_extra_lm_states).
  int32 num_basic_lm_states_;

  // Queue of pairs: (likelihood change [which is negative], lm_state_index).
  // We always pick the one with the highest (least negative) likelihood change
  // to merge.  Note: elements in the queue can get out of date, so it's
  // necessary to check that something is up-to-date (i.e. the likelihood change
  // is accurate) before backing off a state.
  // Note: after InitializeQueue() is called, any state that has nonzero count
  // and history-length >= no_prune_ngram_order, will be in the queue.
  //
  // This whole algorithm is slightly approximate (i.e. it may not always back
  // off the absolutely lowest-cost states), because we don't force
  // recomputation of all the costs each time we back something off.  Generally
  // speaking, these costs will only increase as we back off more states, so the
  // approximation is not such a big deal.
  std::priority_queue<std::pair<BaseFloat, int32> > queue_;


  // adds the counts for this ngram (called from AddCounts()).
  inline void IncrementCount(const std::vector<int32> &history,
                             int32 next_phone);


  // Computes whether backoff should be allowed for this lm_state.  (the caller
  // can set the backoff_allowed variable to match).  Backoff is allowed if the
  // history length is >= opts_.no_prune_ngram_order, and tot_count ==
  // tot_count_with_parents (i.e. there are no parents that are not yet backed
  // off), and the total count is nonzero, and all transitions from this state
  // involve backoff.  (i.e. backoff is disallowed if the the history-state
  // (this history-state + next-phone) exists.
  bool BackoffAllowed(int32 lm_state) const;

  // sets up tot_count_with_parents in all the lm-states
  void SetParentCounts();

  // Computes the change, in log-likelihood caused by backing off this lm state
  // to its backoff state, i.e. combining its counts with those of its backoff
  // state.  This lm state must have backoff_allowed set to true.  This function
  // returns what can be interpreted as a negated cost.  As a special case, if
  // the backoff state has a zero count but this state has a nonzero count, we
  // set the like-change to -1e-15 * (count of this state).  Before the backoff
  // states have any counts, this encourages the lowest-count states to get
  // backed-off first.
  BaseFloat BackoffLogLikelihoodChange(int32 lmstate_index) const;

  // Adds to the queue, all LmStates that have nonzero count and history-length is
  // >= no_prune_ngram_order.
  void InitializeQueue();

  // does the logic of pruning/backing-off states.
  void DoBackoff();

  // This function, will back off the counts of this lm_state to its
  // backoff state, and update num_active_lm_states_ as appropriate.
  // If the count of the backoff state was previously zero, and the backoff
  // state's history-length is >= no_prune_ngram_order, the backoff
  // state will get added to the queue.
  void BackOffState(int32 lm_state);

  // Check, that num_active_lm_states_ is accurate, and returns
  // the number of 'basic' LM-states (i.e. the number of lm-states whose history
  // is of length no_prune_ngram_order - 1).
  int32 CheckActiveStates() const;

  // Finds and returns an LM-state index for a history -- or -1 if it doesn't
  // exist.  No backoff is done.
  int32 FindLmStateIndexForHistory(const std::vector<int32> &hist) const;

  // Finds and returns an LM-state index for a history -- and creates one if
  // it doesn't exist -- and also creates any backoff states needed, down
  // to history-length no_prune_ngram_order - 1.
  int32 FindOrCreateLmStateIndexForHistory(const std::vector<int32> &hist);

  // Finds and returns the most specific LM-state index for a history or
  // backed-off versions of it, that exists and has nonzero count.  Will die if
  // there is no such history.  [e.g. if there is no unigram backoff state,
  // which generally speaking there won't be.]
  int32 FindNonzeroLmStateIndexForHistory(std::vector<int32> hist) const;

  // after all backoff has been done, assigns FST state indexes to all states
  // that exist and have nonzero count.  Returns the number of states.
  int32 AssignFstStates();

  // find the FST index of the initial-state, and returns it.
  int32 FindInitialFstState() const;

  void OutputToFst(
      int32 num_fst_states,
      fst::StdVectorFst *fst) const;

};



}  // namespace chain
}  // namespace kaldi

#endif


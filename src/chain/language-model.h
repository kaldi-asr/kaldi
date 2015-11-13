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

// Options for phone language model estimation.  In the simplest form this is
// an un-smoothed language model of a certain order (e.g. triphone).  We won't
// be actually decoding with this, we'll just use it as the 'denominator graph'
// in acoustic model estimation.   The reason for avoiding smoothing is to
// reduce the number of transitions in the language model, which will improve
// efficiency of training.

struct LanguageModelOptions {
  int32 ngram_order;
  std::string leftmost_context_questions_rxfilename;
  int32 num_extra_states;

  LanguageModelOptions():
      ngram_order(3), // you'll rarely want to change this.
      num_extra_states(250) { }

  void Register(OptionsItf *opts) {
    opts->Register("ngram-order", &ngram_order, "n-gram order for the phone "
                   "language model used for the 'denominator model'");
    opts->Register("leftmost-context-questions",
                   &leftmost_context_questions_rxfilename,
                   "In order to reduce the number of states in the compiled graph, "
                   "you can limit the number of questions for the left-most context "
                   "position; if so you should supply them here so we can merge "
                   "language-model states where appropriate.  Only makes sense "
                   "if the --ngram-order is the same as the --context-width for "
                   "tree building.");
    opts->Register("num-extra-states", &num_extra_states, "Only applicable if the "
                   "--leftmost-context-questions option is used.  Controls how "
                   "many language-model setates we can add that are 'more specific' "
                   "than dictated by the sets defined by the "
                   "--leftmost-context-questions.  This helps get a more accurate "
                   "language model, at the cost of some extra LM states.");
  }
};

/**
   This LanguageModelEstimator class estimates an unsmoothed n-gram language
   model (typically trigram); it's intended for use on phones.  It's intentional
   that we use an un-smoothed n-gram: this limits the number of transitions in
   the compiled denominator graph by not including unseen triphones.

   It also supports merging together some of the history-states, using the
   'leftmost-context-questions', in a way that coincides with how history-states
   can be merged in the tree; this keeps the decoding graph small.  Basically,
   if a set of phones cannot be distinguished by the leftmost-context-questions,
   we merge their history-states, e.g. if x b and y b  are history-states and
   x and y are in the same class, we make a single history-state for (x or y) b.
   However, the 'num-extra-states' option allows us to pick some of the highest
   count history-states of individual phones, and take them out of the 'shared'
   history states, to form more specific states.  This allows us to build
   a stronger language model despite that constraint.

   This language model is implemented as a set of states, and transitions
   between these states; there is no concept of a backoff transition here.
   Because this maps very naturally to an FST, we output it as an FST.
 */
class LanguageModelEstimator {
 public:
  LanguageModelEstimator(LanguageModelOptions &opts): opts_(opts),
                                                      max_phone_(-1) { }

  // Adds counts for this sentence.  Basically does: for each n-gram in the
  // sentence, count[n-gram] += 1.  The only constraint on 'sentence' is that it
  // should contain no zeros.
  void AddCounts(const std::vector<int32> &sentence);

  // Estimates the LM and outputs it as an FST.  Note: there is
  // no concept here of backoff arcs.
  void Estimate(fst::StdVectorFst *fst) const;

 protected:
  typedef unordered_map<std::vector<int32>, int32, VectorHasher<int32> > MapType;
  typedef unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > PairMapType;
  LanguageModelOptions opts_;
  MapType counts_;
  int32 max_phone_;  // highest-numbered phone seen.

  // does counts_[ngram]++.
  inline void IncrementCount(const std::vector<int32> &ngram);

  // used inside Estimate:

  // If leftmost_context_questions_rxfilename is non-empty, this function reads
  // the extra questions and works out the phone sets that are distinguishable
  // by these questions.  the 'phone_to_set' vector maps for each phone to a set
  // identifier which is an integer > 0.  It also maps from phone zero to zero,
  // as a special case.
  // Modifies max_phone_.
  // If leftmost_context_questions_rxfilename is empty, this function makes
  // 'phone_to_set' a map from each phone to itself.
  void ComputePhoneSets(std::vector<int32> *phone_to_set) const;

  // Gets total count for each history (each history is a sequence of
  // phones of length n-1).
  void GetHistoryCounts(MapType *hist_counts) const;


  // This function returns the count cutoff value that will give us the number
  // of 'special' histories (those which are not merged into the history-states
  // defined by the extra questions).  If --num-extra-states == 0 or
  // --leftmost-context-questions is not defined, it returns the maximum integer
  // representable in int32;
  // otherwise it sorts the history-state counts, and returns the value of the
  // --num-extra-states'th largest count.
  int32 GetHistoryCountCutoff(const MapType &hist_counts) const;


  // This function gets a map from a 'history' (i.e. a sequence of phones of
  // length n-1) to a history-state.  All histories with counts above the count
  // cutoff map uniquely to a state; for histories below that cutoff, histories
  // whose leftmost phone is in the same 'set' are merged into one.  Returns the
  // number of (integer) history-states.  history-states are zero-based.
  int32 GetHistoryToStateMap(const MapType &hist_counts,
                             int32 count_cutoff,
                             const std::vector<int32> &phone_to_set,
                             MapType *hist_to_state) const;

  // This function creates a map 'state_transitions' that maps the
  // pair (history-state, phone) to the next history state that we transition to
  // after seeing that phone.  If the phone is 0, we don't add an entry
  // (it becomes a final-prob).
  // Note: it also reads the raw counts from counts_.
  // Returns the initial state.
  int32 GetStateTransitions(
      const MapType &hist_to_state,
      unordered_map<std::pair<int32, int32>, int32,
                    PairHasher<int32> > *transitions) const;


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


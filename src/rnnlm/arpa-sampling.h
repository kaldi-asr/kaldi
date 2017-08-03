// arpa-sampling.h

// Copyright 2017  Ke Li
//           2017  Johns Hopkins University (author: Daniel Povey)

// See ../COPYING for clarification regarding multiple authors
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
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_RNNLM_ARPA_SAMPLING_H_
#define KALDI_RNNLM_ARPA_SAMPLING_H_

#include "fst/fstlib.h"
#include "util/common-utils.h"
#include "lm/arpa-file-parser.h"

namespace kaldi {

/**
   This class allows you to read an ARPA file and access it in a way that's
   specialized for some sampling algorithms we use during RNNLM training
   (we have to sample from a distribution that comes from an ARPA file).

   You'll normally construct this object using an 'options' object
   and a symbol table, and then call Read() to read from the file.
*/
class ArpaSampling : public ArpaFileParser {
 public:
  friend class ArpaSamplingTest;

  typedef std::vector<int32> HistType;
  // WeightedHistType represents a vector of pairs of a history and
  // its associated weight
  typedef std::vector<std::pair<HistType, BaseFloat> > WeightedHistType;

  // ARPA LM file is read by function "void Read(std::istream &is, bool binary)"
  // in ArpaFileParser. Only text mode is supported.
  ArpaSampling(ArpaParseOptions options, fst::SymbolTable* symbols):
      ArpaFileParser(options, symbols) { }

  // This function outputs the unigram distribution of all words represented
  // by integers from 0 to maximum symbol id
  // Note: there can be gaps of integers for words in the ARPA LM, we set the
  // probabilities of words that are not in the ARPA LM to be 0.0, e.g.,
  // symbol id 0 which represents epsilon has probability 0.0
  const std::vector<BaseFloat> &GetUnigramDistribution() const {
    return unigram_probs_;
  }

  // This function accepts a list of histories with associated weights and outputs
  // 1) the non_unigram_probs (maps the higher-than-unigram words to their
  // corresponding probabilities given the list of histories) and
  // 2) a scalar unigram_weight = sum of history_weight * backoff_weight of
  //   that history
  // The sum of the returned unigram prob plus the .second elements of
  // the output 'non_unigram_probs' will not necessarily be equal to 1.0, but
  // it will be equal to the total of the weights of histories in 'histories'.
  BaseFloat GetDistribution(const WeightedHistType &histories,
             std::unordered_map<int32, BaseFloat> *non_unigram_probs) const;


  // This is an alternative interface to GetDistribution() that outputs a list
  // of pairs (word-id, weight), that's sorted and unique on word-id, instead of
  // an unordered_map.
  BaseFloat GetDistribution(const WeightedHistType &histories,
            std::vector<std::pair<int32, BaseFloat> > *non_unigram_probs) const;

  // Return the n-gram order, e.g. 1 for a unigram LM, 2 for a bigram.
  int32 Order() const { return higher_order_probs_.size() + 1; }
 protected:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable();
  virtual void ConsumeNGram(const NGram& ngram);
  // In ReadComplete(), we change the format from 'with backoff' to
  // 'with addition', by subtracting lower-order contributions to the
  // probability.
  virtual void ReadComplete();

 private:

  /**
     This function adds backoff states to a weighted set of history-states.
     What this means is: any (history-state, weight) pair in 'histories' is
     interpreted as a suitably weighted sum over that history-state and the
     relevant backoff states, down to unigram.  The output 'histories_closure'
     adds in the backoff states, with the relevant weights, down to the unigram
     state.

     Before doing this, each history-state in 'histories' is backed off until
     we reach a history-state that actually exists.  Therefore the history
     states in 'histories_closure' will all exist.  The sum of the weights
     in 'histories_closure' will, at exit, be >= the sum of the weights
     in 'histories'; the total weight will normally be more than at the input.
     This can be viewed as a change in representation of the distribution, where
     the output representation in 'histories_closure' does not require any backoff
     or smoothing, because it's explicitly included in the sum.

        @param [in] histories  The input histories (most of these will be
                               histories of the highest ordered allowed, e.g.
                               of length 2 for a trigram LM).
        @param [out] histories_closure   The histories, backed off until we
                               reach a history-state that exists in the LM,
                               and then also including backoff states
                               (of higher order than unigram; the unigram
                               weight is output as 'unigram_weight'.
        @param [out] total_weight   This function outputs the total of
                               histories[*].second to this location; it's
                               used in debugging code.
        @param [out] unigram_weight  To this location we output the total unigram
                              backoff weight of all history-states in
                               'histories'.  If the empty/unigram history-state
                               was present in 'histories', its weight will also
                               be included in this sum.
   */
  void AddBackoffToHistoryStates(
      const WeightedHistType &histories,
      WeightedHistType *histories_closure,
      BaseFloat *total_weight,
      BaseFloat *unigram_weight) const;

  /**  For each history (except the empty, unigram history) there is a
       HistoryState that you can look up.  It stores actual probabilities,
       between 0 and 1, not log-probs.  The language model is stored in memory
       in a 'with addition' format, meaning that for a trigram (for example),
       you can't just read off the probability directly from the trigram state;
       you have to add it to the probability from lower-order states.
   */
  struct HistoryState {
    // 0.0 < backoff_prob < 1.0: the probability of backing off from this
    // history to the next lower-order history state, e.g. if this history-state
    // is for "of the" -> x, it would be the probability of backoff to "the" ->
    // x
    BaseFloat backoff_prob;

    // 'word_to_prob' is a map from word to probability of that word.  It
    // doesn't contain the total probability: unlike for the on-disk ARPA
    // format, to get the probability of the word you have to add the
    // contributions from all backoff orders.
    // the BaseFloat should normally be positive, but if you are using
    // the 'wrong' type of LM, there is a possibility it will be negative;
    // warnings will be printed in this case.
    std::unordered_map<int32, BaseFloat> word_to_prob;
    HistoryState(): backoff_prob(1.0) { }
  };


  // This function should only be called from ReadComplete(), because
  // it assumes that the probabilities are stored in a 'backoff' manner
  // as they are in ARPA files, and if you call this *after*
  // ReadComplete() has been called, it will give you the wrong answer.
  // It returns the probability (not the log-prob) of word 'word' with
  // history 'history', e.g. if 'word' is c and 'history' is [a, b]
  // it gives you the probability of a b -> c.
  // 'state' is the history state corresponding to 'history', if
  // history.size() > 0; it is only provided as an efficiency thing, to
  // avoid an unnecessary map lookup, and the function does not require

  //  @param [in] history      History in which to get the probability
  //                           of 'word'.  This history state is required
  //                           to exist.
  //  @param [in] state        If non-NULL, must be the history-state
  //                           corresponding to 'history'.  If NULL,
  //                           this function will work out the correct
  //                           history-state.
  //  @param [in] word         The word for which we want the probability.
  //  @return                  Returns the probability *not* the log-prob
  //                           of the word 'word' given history 'history'.
  BaseFloat GetProbWithBackoff(
      const std::vector<int32> &history,
      const HistoryState *state,
      int32 word) const;


  // Unigram probabilities, indexed by word-id.  These will sum to one.
  std::vector<BaseFloat> unigram_probs_;

  // Probabilities (not log-probs) of n-grams with higher order than one.  Note:
  // unlike the way the ARPA format is stored, we store these as if "with
  // addition", i.e.  to get the predicted probability of a trigram we'd have to
  // add in the bigram and unigram contributions.
  //
  // Indexed by n-gram order minus two, then by history, to a
  // HistoryState.
  std::vector<std::unordered_map<HistType, HistoryState,
                                 VectorHasher<int32> > > higher_order_probs_;
};
}  // end of namespace kaldi
#endif  // KALDI_RNNLM_ARPA_SAMPLING_H_

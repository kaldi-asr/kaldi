// sampling-lm-estimate.h

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_RNNLM_SAMPLING_LM_ESTIMATE_H_
#define KALDI_RNNLM_SAMPLING_LM_ESTIMATE_H_

#include "fst/fstlib.h"  // for symbol table
#include "util/common-utils.h"

namespace kaldi {
namespace rnnlm {

// Options struct for class SamplingLmEstimator.
struct SamplingLmEstimatorOptions {
  int32 vocab_size;          // highest-numbered word plus one; must be set.
  int32 ngram_order;
  BaseFloat discounting_constant;
  // search in the long comment below for more explanation of the following
  // factors.
  BaseFloat unigram_factor;
  BaseFloat backoff_factor;
  BaseFloat bos_factor;
  BaseFloat unigram_power;

  int32 bos_symbol;
  int32 eos_symbol;
  int32 brk_symbol;

  SamplingLmEstimatorOptions(): vocab_size(-1),
                                ngram_order(3),
                                discounting_constant(1.0),
                                unigram_factor(100.0),
                                backoff_factor(2.0),
                                bos_factor(5.0),
                                unigram_power(0.8),
                                bos_symbol(1),
                                eos_symbol(2),
                                brk_symbol(-1) { }

  void Register(OptionsItf *po) {
    po->Register("vocab-size", &vocab_size, "If set, must be set to the "
                 "highest-numbered vocabulary word plus one; otherwise "
                 "this is worked out from the symbol table.");
    po->Register("ngram-order", &ngram_order, "Order for the n-gram model "
                 "(must be >= 1), e.g. 3 means trigram");
    po->Register("discounting-constant", &discounting_constant, "Constant for "
                 "absolute discounting; should be in the range 0.8 to 1.0, and "
                 "smaller values give a larger language model");
    po->Register("unigram-factor", &unigram_factor, "Factor by which p(w|h) "
                 "for non-unigram history state h (with the backoff term "
                 "excluded) has to be greater than p(w|unigram-state) for "
                 "us to include it in the model.  Must be >0.0, will normally "
                 "be >1.0.");
    po->Register("backoff-factor", &backoff_factor, "Factor by which p(w|h) "
                 "for higher-than-bigram history state h (with the backoff term "
                 "excluded) has to be greater than p(w|backoff-state) for us "
                 "to include it in the model (in addition to the "
                 "--unigram-factor constraint).  Must be >0.0 and "
                 "< unigram-factor");
    po->Register("bos-factor", &bos_factor, "Factor by which p(w|h) "
                 "for h == the BOS history state (with the backoff term "
                 "excluded) has to be higher than p(w|unigram-state) "
                 "for us to include it in the model.  Must be >0.0 and "
                 "<= unigram-factor.");
    po->Register("bos-symbol", &bos_symbol,
                 "Integer id for the BOS word (<s>)");
    po->Register("eos-symbol", &eos_symbol,
                 "Integer id for the EOS word (</s>)");
    po->Register("brk-symbol", &brk_symbol,
                 "Integer id for the Break word (<brk>). Not needed but "
                 "included for ease of scripting");
    po->Register("unigram-power", &unigram_power, "Important configuration "
                 "value.  After all other stages of estimating the model, "
                 "the unigram probabilities are taken to this power and then "
                 "rescaled to sum to 1.0.  Note: it's traditional in "
                 "importance sampling to use this kind of power term.  "
                 "There are both theoretical and practical reasons why we want "
                 "to just apply this power to the unigram portion.  E.g. 0.75.");
  }
  void Check() const;
};


class SamplingLm;  // Forward declaration.

/**
   This class is responsible for creating a backoff n-gram language model of a
   type that's suitable for use in the importance sampling algorithm we use for
   RNNLM training.  It's the type of language model that could in principle be
   written in ARPA format, but it's created in a special way.  There are a few
   characteristics of the importance sampling algorithm that make it desirable
   to write a special purpose language model instead of using a generic language
   model toolkit.

   These are:

      - When we sample, we sample from a distribution that is the average of a
        fairly large number of history states N (e.g., N=128), that can be treated
        as independently chosen for practical purposes (except that sometimes
        they'll all be the BOS history, which is a special case).
      - The convergence of the sampling-based method won't be sensitive to small
        differences in the probabilities of the distribution we sample on.
      - It's important not to have too many words that are specifically predicted
        from a typical history-state, or it makes the sampling process slow.

   To give a feel for the kind of phenomenon we want to model: we want to model
   situations where a word is *much* more likely than it would otherwise be
   given a particular history-state: for example "san" -> "andreas / francisco"
   and similar things.  The rule of thumb will be that, for instance, if we
   sample from the average of N distributions, then a word has to be about N
   times more likely than it would be when predicted from the unigram
   distribution, for us to really care about it.  If not, once we divide by N
   and sum with the other distributions, it's not going to have very much impact
   on the result.  That argument is valid if the history-state appears only once
   in the average; and if the history-state concerned is common enough to be
   likely to occur more than once in the average, then it would contribute
   enough to the unigram distribution that we don't have to worry anyway.
   There are a couple of things that make this argument not quite true:
   (1) dataset-weighting, where the distributions don't get an equal weight--
   this can be corrected for by just making the factor difference in probability
   that we "care about" close to 1; and (2) the beginning of the minibatch (t=0),
   where all histories are the BOS history '<s>'.  We'll deal with (2) by
   reducing the factor that controls how many words we keep for the BOS history,
   so that we'll get more predicted words in that history-state.

   All the above is motivation.  First we describe the input to the
   LM-estimation process.  The input is sentences with attached corpus-weights,
   just as it is for RNNLM training (but we assume there is no *multiplicity* of
   the data from the different corpuses, as this doesn't interact well with
   smoothing; instead if we were planning to repeat the data k times during
   training, we'll just multiply its scaling factor by k when estimating the
   LM).  The sentences will be integerized by the time we get them, and we'll be
   aware of the integer values of certain 'special' symbols <s> and </s>, as
   well as the vocabulary size.

   Now for characteristics of the actual smoothing and interpolation algorithm.
   Firstly, it's not Kneser-Ney like-- the unigram distribution is estimated
   from unigram stats, the bigram distribution from bigram stats, the trigram
   from trigram stats-- no counts-of-counts or similarly motivated things.  The
   discounting method is just absolute discounting with a fixed discounting
   constant D (default: D = 0.9), which is within the typical range of these
   things if you were to actually estimate it from data.

   Suppose we're estimating a trigram history a b -> x, and we've already
   estimated the backoff bigram state a -> x.  We have a bunch of counts for
   different values of x.  Those counts all have floating-point values (they
   won't be all multiples of 1.0 because of the corpus weighting), but for each
   "x" we keep track of the largest corpus-weight for any individual count, and
   that's what we discount for the absolute discounting.  E.g. if the largest
   corpus-weight was 2.0, then we'd discount D * 2.0 from the count for "x".  We
   keep track of the total count that we've discounted, for this history-state
   and from all possible "x" values, and this determines the weight allocated
   for backoff to the lower-order state.  After doing this we examine each
   word in this state.  The basic rule for higher-than-unigram states is:
       If p(w|h) is greater than unigram_factor * p(w|unigram-state),
    [where e.g. unigram_factor=100.0]
  then we keep it; otherwise we completely discount it (and note that this
  affects how much probability mass goes to backoff).
  However, this rule would lead us, for trigram states, to keep too many
  words that had already been adequately modeled by the backoff (bigram) state.
  So for higher-order than bigrams states we modify the rule to:
       If p(w|h) is greater than unigram_factor * p(w|unigram-state)
    AND  If p(w|h) is greater than backoff_factor * p(w|backoff-state)
        [where e.g. backoff_factor=2.0]
  then we keep it, else we completely back it off.

  This language model is 'with interpolation' in the sense of Kneser-Ney with
  interpolation (although it's not Kneser-Ney).  What we mean is that for words
  that we model, the probability of those words given the state we back off to
  is included.  This is mostly done for reasons of convenience; it won't tend to
  affect the probabilities very much since we already keep only those words that
  are considerably more probable than they would be given the backoff state.

  For the BOS ('<s>') history-state, which, as mentioned, presents a problem
  because it doesn't occur independently but will occur for all members of
  the minibatch on t=0, we have a different factor which we call bos_factor
  (e.g. bos_factor=5.0), which controls how many words we keep for the BOS
  state.  This ensure that we don't prune away too many words for this particular
  left-context.
 */
class SamplingLmEstimator {
 public:
  ///  Constructor.  Retains a reference to 'config'.
  SamplingLmEstimator(const SamplingLmEstimatorOptions &config);

  /** Processes one line of the input, adding it to the stored stats.

    @param [in] corpus_weight   Weight attached to the corpus from which this
                         data came.  (Note: you shouldn't repeat sentences
                         when providing them to this class, although this is
                         allowed during the actual RNNLM training; instead, you
                         should make sure that the multiplicity that you use
                         in the RNNLM for this corpus is reflected in
                         'corpus_weight'.
    @param [in] sentence  The sentence we're processing.  It is not expected
                         to contain the BOS symbol, and should not be terminated
                         by the EOS symbol, although the EOS symbol is allowed
                         internally (where it can be used to separate a sequence
                         of sentences from a dialogue or other sequence of text,
                         if you want to do this).
  */
  void ProcessLine(BaseFloat corpus_weight, const std::vector<int32> &sentence);

  // Reads the lines from this input stream, calling ProcessLine() on each
  // one.  Lines will be of the format:
  // <weight> <possibly-empty-sequence-of-integers>
  // e.g.:
  // 1.0  2560 8991
  void Process(std::istream &is);

  // Estimates the language model (internal representation); includes
  // discounting.  Setting the 'will_write_arpa' option to true forces it to
  // retain certain n-grams that it would otherwise have pruned, because they
  // are required in the ARPA file format.
  void Estimate(bool will_write_arpa);

  // Prints in ARPA format to a stream.  this is non-const because of
  // implementation internals, but it does not modify *this.
  void PrintAsArpa(std::ostream &os,
                   const fst::SymbolTable &symbols);


  ~SamplingLmEstimator();

  friend class SamplingLm;
 protected:
  // a struct used in HistoryState.
  struct Count {
    int32 word;  // word for which this is a count.
    BaseFloat highest_count;  // highest individual count in 'count', will be
                              // equal to one of the dataset weighting factors.
    double count;  // total count for this word (this gets reduced when we
                   // do smoothing).

    inline bool operator < (const Count &other) const {
      return word < other.word;
    }
  };

  struct HistoryState {
    // total_count > 0.0 is the total count for this state; it should always
    // equal the total of backoff_count plus the 'count' members of 'counts'.
    // it won't be set up, though, until you call ComputeTotalCount().
    BaseFloat total_count;

    // backoff_count >= 0.0 is the amount of probability mass that we have
    // already discounted from this state.  it will be zero before Estimate()
    // is called.
    BaseFloat backoff_count;

    // A vector of the currently existing counts.  It will always be sorted and
    // uniqe on 'word'.  We store it as a vector instead of (for example) an
    // unordered_map from predicted-word, to save memory.
    std::vector<Count> counts;

    // A vector of new individual predicted-words (pairs (word-id,
    // corpus-weight)) that we have yet to insert into 'counts'.  This will only
    // be nonempty before Estimate() is called in the enclosing class.  A call
    // to ProcessNewCounts() will empty this vector.  The strategy is to wait
    // until new_counts.size() >= counts.size() and then merge these into
    // 'counts'.  This is a tradeoff between speed and memory constraints.
    std::vector<std::pair<int32, BaseFloat> > new_counts;

    // This will be set to true if this history-state is protected from being
    // removed, because a history state that backs off to this history state
    // exists.  Note: there is also a notion of an n-gram being protected,
    // which is for similar reasons (to preserve the integrity of the ARPA
    // file) but is logically distinct from preserving the history state;
    // we can preserve a state that has no n-grams.
    bool is_protected;

    // when called, will empty the 'new_counts' vector, inserting its counts
    // into 'counts'.  If 'release_memory=true', it will make sure that the
    // memory used in 'new_counts' is freed.
    void ProcessNewCounts(bool release_memory);

    // Adds one count.
    void AddCount(int32 word, BaseFloat corpus_weight);

    void ComputeTotalCount();  // sets up total_count.

    HistoryState(): total_count(0.0), backoff_count(0.0),
                    is_protected(false) { }
  };

  inline void AddCount(const std::vector<int32> &history,
                       int32 word, BaseFloat corpus_weight) {
    GetHistoryState(history, true)->AddCount(word, corpus_weight);
  }

  // Scales the unigram counts by taking them to the power 'power' and
  // renormalizing.
  void TakeUnigramCountsToPower(BaseFloat power);

  // Returns the number of n-grams of the order 1 <= o <= config_.ngram_order.
  // Required to produce an ARPA file.
  int32 NumNgrams(int32 o) const;

  // This function sorts (on word) the vector of Count provided, then
  // appropriately merges the counts that have the same word, summing the
  // 'count' field and taking the max of the 'highest_count' field.
  static void SortAndUniqCounts(std::vector<Count> *counts);

  // Removes elements from the vector 'counts' where the .count field is zero.
  static void RemoveZeroCounts(std::vector<Count> *counts);

  // For 1 <= o < ngram_order, this function computes the raw, un-pruned and
  // un-backed-off counts for n-grams of order o (i.e. with history of length o
  // - 1).  The reason we need this is: when we call Process() or ProcessLine(),
  // supposing config_.ngram_order = 3, only trigram counts are stored (ignoring
  // beginning-of-sentence effects), so this function is required to compute the
  // lower order counts from the higher-order counts.
  void ComputeRawCountsForOrder(int32 o);


  // for 2 <= o <= ngram_order, smooths the distribution.  This consists of
  // discounting from the 'count' elements of struct Count, and adding the
  // subtracted quantities to the backoff_count elements of struct HistoryState.
  void SmoothDistributionForOrder(int32 o);

  // for 2 <= o <= ngram_order, prunes n-grams from the model.  This consists of
  // removing some of the Counts from the history states' count vectors,
  // according to some criteria that are described in the comment at the top of
  // this class.  The 'count' elements of the removed Counts are added to the
  // 'backoff_count' of the states.
  void PruneNgramsForOrder(int32 o);

  // This prunes away states with zero counts for order 'o' (e.g. if o == 3 we
  // mean trigram states, with history-length of 2.  But we preserve states that
  // have zero counts ("protected" states) if there exist other states that back
  // off to them.  For o > 2 (if will_write_arpa is true), this function also
  // adds back n-grams of the one-lower orer which are required in the ARPA file
  // because they lead to this state.
  void PruneStatesForOrder(int32 o, bool will_write_arpa);


  /**
     Prunes counts from a history-state; this version is for states that are
     trigram or above (i.e. for history length >= 2).
        @param [in] history  The history (sequence of words) corresponding
                             to this history-state, e.g. [a] for the history
                             state corresponding to a left-context of "a".
                             May be any length >= 1.  This is only necessary
                             for a rather obscure reason that has to do
                             with how the ARPA format works (an n-gram
                             can't be pruned if it leads to a history-state
                             that exists, e.g. if the state a b->x exists,
                             we can't prune away the bigram a -> b.
        @param [in] backoff_states  The list of states that this state
                             backs off to, i.e. this state backs off
                             to backoff_states[0], that backs off to
                             backoff_states[1], and so on, but down
                             to the bigram only (not including the unigram,
                             which we store in unigram_probs__).  It will
                             therefore be empty if history.size() == 1.
                             This is provided for more efficient computation
                             of the probability given the backoff state.
        @param [in,out] state  The state that we are to prune.
  */
  void PruneHistoryStateAboveBigram(
      const std::vector<int32> &history,
      const std::vector<const HistoryState*> &backoff_states,
      HistoryState *state);

  /**
    Prunes a history-state; this version is for bigram states
    (history.length() == 1).
       @param [in] history  The history corresponding to this
                            state, must be a vector of length 1.
       @param [in,out] state  The state we're pruning.
  */
  void PruneHistoryStateBigram(const std::vector<int32> &history,
                               HistoryState *state);


  // Returns the probability for word 'word' given a history, where
  // if you list the history-state for that history, then the state
  // it backs off to, and so on down to bigram, the result is
  // in 'states'.  Also works for unigram if you supply the empty
  // vector, although could just look that up in unigram_probs_.
  BaseFloat GetProbForWord(int32 word,
                           const std::vector<const HistoryState*> &states) const;


  // Convenience function; returns true if a history-state exists for
  // the history that you get when appending 'word' to 'history'.  We
  // check this because we can't prune away n-grams of this type;
  // it's not compatible with the ARPA language model format.
  bool IsProtected(const std::vector<int32> &history, int32 word) const;


  // Convenience function used when printing as ARPA.  Let 'h' be
  // the result of appending 'word' to 'history'.  This function returns 0.0
  // if there is no history-state for history 'h', otherwise the backoff
  // probability (backoff_count / total_count) for that history-state.
  BaseFloat BackoffProb(const std::vector<int32> &history, int32 word) const;



  // For 1 <= o <= ngram_order, this function calls (if o ==
  // config_.ngram_order) ProcessNewCounts(); and then ComputeTotalCount(), on
  // history-states corresponding to n-grams of this order.
  void FinalizeRawCountsForOrder(int32 o);

  // Returns a pointer to an existing or newly created history-state for history
  // 'history', which must satisfy history.size() < config_.ngram_order.
  // This is done by lookup in 'history_states_'.
  // If add_if_absent is false, will die if the state was not present; if
  // true, it will add the state.
  HistoryState *GetHistoryState(const std::vector<int32> &history,
                                bool add_if_absent);

  // This function gets the probability distribution for unigram.
  // It reads from with the raw unigram counts which are to be found in
  // in history_states_[0], and writes to unigram_probs_.
  // It also modifies the HistoryState for unigram, but you won't
  // be reading from that after calling this.
  void ComputeUnigramDistribution();

  // used when printing ARPA; prints to stream 'os' the unigrams.
  void PrintNgramsUnigram(std::ostream &os,
                          const fst::SymbolTable &symbols) const;

  // used when printing ARPA; prints to stream 'os' the n-grams of order 'o',
  // for o > 1.  actually doesn't change *this but is const because
  // it calls GetHistoryState().
  void PrintNgramsAboveUnigram(std::ostream &os, int32 o,
                               const fst::SymbolTable &symbols);

  const SamplingLmEstimatorOptions &config_;

  // history_states_ is indexed by the length of the history
  // (which equals the n-gram order minus one), and then is
  // a map from a vector containing the history, to the HistoryState.
  // the history is in the order the words were seen in the data
  // (i.e., not reversed).
  std::vector<unordered_map<std::vector<int32>, HistoryState*,
                            VectorHasher<int32> > > history_states_;

  // This vector contains the unigram probabilities, normalized to sum to one.
  // (only after you call Estimate()).
  std::vector<BaseFloat> unigram_probs_;

};



}  // namespace rnnlm
}  // namespace kaldi
#endif  // KALDI_RNNLM_SAMPLING_LM_ESTIMATE_H_

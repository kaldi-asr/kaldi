// ctc/language-model.h

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


#ifndef KALDI_CTC_LANGUAGE_MODEL_H_
#define KALDI_CTC_LANGUAGE_MODEL_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

// CTC means Connectionist Temporal Classification, see the paper by Graves et
// al.
//
// What we are implementing is an extension of CTC that we're calling
// context-dependent CTC (CCTC).  It requires the estimation of an n-gram
// language model on phones.
//
// This header implements a language-model class that's suitable for this
// phone-level model.  The implementation is efficient for the case where
// the number of symbols is quite small, e.g. not more than a few hundred,
// so tabulating probabilities makes sense.
// We don't put too much effort in making this the best possible language
// model and adding bells and whistles.
// We're implementing the count cutoffs in the way that we feel makes the
// most sense; it probably won't exactly match the way it's done in, say,
// SRILM.  And the way the discounting is done is also not quite the same
// it's done in the original Kneser-Ney publication, as we do it using
// continuous rather discrete counts; and we fix the constants rather
// than estimating them.
// this is Kneser-Ney "with addition" instead of backoff (see A Bit of Progress
// in Language Modeling).
namespace ctc {


struct LanguageModelOptions {
  int32 ngram_order;
  int32 state_count_cutoff1;
  int32 state_count_cutoff2plus;
  BaseFloat discount1;  // Discounting factor for singletons in Kneser-Ney type
                        // scheme
  BaseFloat discount2plus;  // Discounting factor for things with >1 count in
                            // Kneser-Ney type scheme.
  LanguageModelOptions():
      ngram_order(3), // recommend only 1 or 2 or 3.
      state_count_cutoff1(0),
      state_count_cutoff2plus(200), // count cutoff for n-grams of order >= 3 (if used)
      discount1(0.8),
      discount2plus(1.3) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("ngram-order", &ngram_order, "n-gram order for the phone "
                   "language model used while training the CTC model");
    opts->Register("state-count-cutoff1", &state_count_cutoff1,
                   "Count cutoff for language-model history states of order 1 "
                   "(meaning one left word is known, i.e. bigram states)");
    opts->Register("state-count-cutoff2plus",
                   &state_count_cutoff2plus,
                   "Count cutoff for language-model history states of order >= 2 ");
    opts->Register("discount1", &discount1, "Discount constant for 1-counts");
    opts->Register("discount2plus", &discount2plus,
                   "Discount constant for 2-counts or greater");
  }
};

/**
   This LanguageModel class implements a slight variant of a Kneser-Ney smoothed
   language model "with addition" (i.e. we add the backoff prob to the direct
   prob; in "A Bit of Progress in Language Modeling" this was found to be better).
   The variation is that do it with continuous counts instead of counts-of-counts,
   so instead of incrementing a count-of-counts for the lower order, we add the
   discounted amount, so we always deal with "real counts" (albeit the counts are
   fractional).  Because this makes the method of estimating the discounting
   constants impractical, we just use fixed values for them.  [Also, the Kneser-Ney
   discounting process is extended to the zero-gram case, which we define as
   distributing the probability mass equally to all vocabulary items; we assume
   the vocab size is known in advance.].  We use index 0 for both BOS and EOS;
   since they can never validly appear in the same context, this leads to no
   confusion and actually simplifies the code as we don't have to test for
   symbols appearing where they are disallowed.

   This language model (like the ARPA format) ensures that if "a b c" is a valid
   history-state, then "a b -> c" must exist as an n-gram (hence "a b" must be a
   valid history-state).  This ends up mattering in the CTC code.
 */
class LanguageModel {
 public:
  LanguageModel(): vocab_size_(0), ngram_order_(0) { }

  int32 NgramOrder() const { return ngram_order_; }

  // Note: phone indexes are 1-based, so they range from 1 to vocab_size_.
  // with 0 for BOS and EOS.
  int32 VocabSize() const { return vocab_size_; }
  
  // Get the language-model probability [not log-prob] for this history-plus-phone.
  // zeros in the non-final position are interpreted as <s>.
  // zeros in the final position are interpreted as </s>.
  BaseFloat GetProb(const std::vector<int32> &ngram) const; 

  void Write(std::ostream &os, bool binary) const;
  
  void Read(std::istream &is, bool binary);
  
 protected:
  friend class LanguageModelEstimator;
  friend class LmHistoryStateMap;
  
  typedef unordered_map<std::vector<int32>, BaseFloat,
                        VectorHasher<int32> > MapType;
  typedef unordered_map<std::vector<int32>, std::pair<BaseFloat,BaseFloat>, VectorHasher<int32> > PairMapType;

  int32 vocab_size_;

  int32 ngram_order_;

  // map from n-grams of the highest order to probabilities.
  MapType highest_order_probs_;

  // map from all other ngrams to (n-gram probability, history-state backoff
  // weight).  Note: history-state backoff weights will be 1.0 for history-states
  // that don't exist.  If a history-state exists, a direct n-gram probability
  // must exist.  Note: for normal n-grams we don't do any kind of pruning that could remove it;
  // and for the history [0] (for BOS) the predicted n-gram [0] (for EOS) will always
  // exist.
  PairMapType other_probs_;
};


// Computes the perplexity of the language model on the sentences provided;
// they should not contain zeros (we'll add the BOS/EOS things internally).
BaseFloat ComputePerplexity(const LanguageModel &lm,
                            std::vector<std::vector<int32> > &sentences);

// This class allows you to map a language model history to an integer id which,
// with the predicted word, is sufficient to work out the probability; it
// also provides a way to work out whether a language model history is a prefix
// of a longer language model history.
// It's
// useful in the CCTC code.  Because this isn't something that would normally
// appear in the interface of a language model, we make it a separate class.  Lm
// stands for "language model".  Because the term lm_history_state is used a lot
// in the CCTC code and also history_state exists there and means something
// different, we felt it was necessary to include "lm" in these names.
class LmHistoryStateMap {
 public:
  // Returns the number of history states.  A history state is a zero-based
  // index, so they go from 0 to NumHistoryStates() - 1.
  // these will
  int32 NumLmHistoryStates() const { return lm_history_states_.size(); }

  const std::vector<int32>& GetHistoryForState(int32 lm_history_state) const;
  
  BaseFloat GetProb(const LanguageModel &lm, int32 lm_history_state,
                    int32 predicted_word) const;
  
  // Maps a history to an integer lm-history-state. 
  int32 GetLmHistoryState(const std::vector<int32> &hist) const;

  // Returns true if this history is an LM history state (equivalent to
  // checking that  GetLmHistoryState(hist) == hist
  bool IsLmHistoryState(const std::vector<int32> &hist) const {
    return GetHistoryForState(GetLmHistoryState(hist)) == hist;
  }
  
  // Initialize the history states.
  void Init(const LanguageModel &lm);

 private:
  typedef unordered_map<std::vector<int32>, int32, VectorHasher<int32> > IntMapType;
  std::vector<std::vector<int32> > lm_history_states_;
  IntMapType history_to_state_;

};



class LanguageModelEstimator {
 public:
  // note: vocabulary ranges from [1 .. vocab_size].  Index 0
  // is used internally for both BOS and EOS (beginning-of-sentence and
  // end-of-sentence symbols), but it should not appear
  // explicitly in the input sentences.
  LanguageModelEstimator(const LanguageModelOptions &opts,
                         int32 vocab_size);

  // Adds counts for this sentence.  Basically does: for each n-gram,
  // count[n-gram] += 1.
  void AddCounts(const std::vector<int32> &sentence);

  // Does the discounting.  Call after calling AddCounts() for all sentences,
  // and then call Output().
  void Discount();

  // Outputs to the LM.  Call this after Discount().
  void Output(LanguageModel *lm) const;
private:
  // Returns the probability for this word given this history; used inside
  // Output().  This includes not just the direct prob, but the additional prob
  // that comes via backoff, since this is Kneser-Ney "with addition".  (this is
  // what we need to store in the otuput language model).
  BaseFloat GetProb(const std::vector<int32> &ngram) const;

  // Gets the backoff probability for this state, i.e. the
  // probability mass assigned to backoff.
  BaseFloat GetBackoffProb(std::vector<int32> &hist) const;
  
  typedef unordered_map<std::vector<int32>, BaseFloat, VectorHasher<int32> > MapType;
  typedef unordered_map<std::vector<int32>, std::pair<BaseFloat, BaseFloat>, VectorHasher<int32> > PairMapType;
  typedef unordered_set<std::vector<int32>, VectorHasher<int32> > SetType;
      
  // applies discounting or to the counts for all stored n-grams of this order.
  // If order >= 2 we apply this continuous Kneser-Ney-like discounting; if
  // order == 1 we apply add-one smoothing.
  void DiscountForOrder(int32 order);

  // order must be >= 1 and < ngram_order.  This function finds all
  // history-states of order 'order' (i.e. containing 'order' words in the
  // history-state), such that (the total count for that history-state is less
  // than min_count, and the history-state is not listed in 'protected_states'),
  // and it completely discounts all n-grams in those history-states, adding
  // their count to the backoff state.  We apply pruning at the level of
  // history-states because this is the level at which added cost is incurred.
  // For history-states which were not removed by this procedure, this function
  // computes their backoff state by removing the first phone, and adds it to
  // "protected_backoff_states".  This will protect
  // that backoff state from pruning when and if we prune the one-smaller order.
  void ApplyHistoryStateCountCutoffForOrder(
      int32 order, BaseFloat min_count,
      const SetType &protected_states,
      SetType *protected_backoff_states);

  // This function does, conceptually, counts_[vec] += count.
  // It's called during training and during discounting.
  inline void AddCountForNgram(const std::vector<int32> &vec, BaseFloat count);

  // This function does, conceptually,
  // history_state_counts_[hist] += (tot_count, discounted_count).
  // It's called during discounting.
  inline void AddCountsForHistoryState(const std::vector<int32> &hist,
                                       BaseFloat tot_count,
                                       BaseFloat discounted_count);

  //  This function, conceptually, returns counts_[vec] (or 0 if not there).
  inline BaseFloat GetCountForNgram(const std::vector<int32> &vec) const;

  // This function, conceptually, returns history_state_counts_[vec] (or (0,0) if
  // not there).  It represents (total-count, discounted-count).
  inline std::pair<BaseFloat,BaseFloat> GetCountsForHistoryState(
      const std::vector<int32> &vec) const;


  
  // Returns a discounting-amount for this count.  The amount returned will be
  // between zero and discount2plus.  It's the amount we are to subtract
  // from the count while discounting.  This simple implementation doesn't
  // allow us to have different discounting amounts for different n-gram
  // orders.  Because the count is continous, the discount1 and discount2plus
  // value are interpreted as values to interpolate to when non-integer
  // counts are provided.
  BaseFloat GetDiscountAmount(BaseFloat count) const;

  
  // Outputs into "backoff_vec", which must be initially empty, all but the
  // first element of "vec".
  inline static void RemoveFront(const std::vector<int32> &vec,
                                 std::vector<int32> *backoff_vec);

  // This function does (*map)[vec] += count, while
  // ensuring that it does the right thing if the vector wasn't
  // a key of the map to start with.
  inline static void AddToMap(const std::vector<int32> &vec,
                              BaseFloat count,
                              MapType *map);

  inline static void AddPairToMap(const std::vector<int32> &vec,
                                  BaseFloat count1, BaseFloat count2,
                                  PairMapType *map);
  

  // data members:
    const LanguageModelOptions &opts_;
  // the allowed words go from 1 to vocab_size_.  0 is reserved for
  // epsilon.
  int32 vocab_size_;
  // counts_ stores the raw counts.  We don't make much attempt at
  // memory efficiency here since a phone-level language model is quite a small
  // thing.   It's indexed first by the n-gram order, then by the ngram itself;
  // this makes things easier when iterating in the discounting code.
  std::vector<MapType> counts_;

  // stores a map from a history-state to a pair (the total count for this
  // history-state; the count that has been removed from this history-state via
  // backoff).  Indexed first by the order of the history state (which equals
  // the vector length).
  std::vector<PairMapType> history_state_counts_;
    
};




}  // namespace ctc
}  // namespace kaldi

#endif


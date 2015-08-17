// ctc/language-model.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "ctc/language-model.h"

namespace kaldi {
namespace ctc {


void LanguageModel::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<LanguageModel>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "<VocabSize>");
  WriteBasicType(os, binary, vocab_size_);
  if (!binary) os << "\n";  
  WriteToken(os, binary, "<NgramOrder>");
  WriteBasicType(os, binary, ngram_order_);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<HighestOrderProbs>");
  int32 size = highest_order_probs_.size();
  WriteBasicType(os, binary, size);
  if (!binary) os << "\n";
  for (MapType::const_iterator iter = highest_order_probs_.begin();
       iter != highest_order_probs_.end(); ++iter) {
    KALDI_ASSERT(iter->first.size() == ngram_order_ && iter->second > 0.0);
    WriteIntegerVector(os, binary, iter->first);
    WriteBasicType(os, binary, iter->second);
    if (!binary) os << "\n";    
  }
  WriteToken(os, binary, "<OtherProbs>");
  size = other_probs_.size();
  WriteBasicType(os, binary, size);
  for (PairMapType::const_iterator iter = other_probs_.begin();
       iter != other_probs_.end(); ++iter) {
    WriteIntegerVector(os, binary, iter->first);
    WriteBasicType(os, binary, iter->second.first);
    WriteBasicType(os, binary, iter->second.second);
    if (!binary) os << "\n";    
  }
  WriteToken(os, binary, "</LanguageModel>");  
}

void LanguageModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<LanguageModel>");
  ExpectToken(is, binary, "<VocabSize>");
  ReadBasicType(is, binary, &vocab_size_);
  ExpectToken(is, binary, "<NgramOrder>");
  ReadBasicType(is, binary, &ngram_order_);
  ExpectToken(is, binary, "<HighestOrderProbs>");
  int32 size;
  ReadBasicType(is, binary, &size);
  highest_order_probs_.clear();
  highest_order_probs_.rehash(size);
  for (int32 i = 0; i < size; i++) {
    std::vector<int32> vec;
    ReadIntegerVector(is, binary, &vec);
    KALDI_ASSERT(vec.size() == static_cast<size_t>(ngram_order_));
    BaseFloat prob;
    ReadBasicType(is, binary, &prob);
    KALDI_ASSERT(prob > 0.0);
    highest_order_probs_[vec] = prob;
  }
  ExpectToken(is, binary, "<OtherProbs>");
  ReadBasicType(is, binary, &size);
  other_probs_.clear();
  other_probs_.rehash(size);
  for (int32 i = 0; i < size; i++) {
    std::vector<int32> vec;
    BaseFloat prob, backoff_prob;
    ReadIntegerVector(is, binary, &vec);
    ReadBasicType(is, binary, &prob);
    ReadBasicType(is, binary, &backoff_prob);
    KALDI_ASSERT(prob > 0.0 && backoff_prob > 0.0);
    other_probs_[vec] = std::pair<BaseFloat,BaseFloat>(prob,backoff_prob);
  }
  ExpectToken(is, binary, "</LanguageModel>");  
}


BaseFloat LanguageModel::GetProb(const std::vector<int32> &ngram) const {
  size_t size = ngram.size();
  if (size > static_cast<size_t>(ngram_order_)) {
    std::vector<int32> new_ngram;
    new_ngram.insert(new_ngram.begin(), ngram.begin() +
                     (ngram.size() - static_cast<size_t>(ngram_order_)),
                     ngram.end());
    return GetProb(new_ngram);
  } else if (size == static_cast<size_t>(ngram_order_)) {
    MapType::const_iterator iter = highest_order_probs_.find(ngram);
    if (iter != highest_order_probs_.end())
      return iter->second;
  } else {
    KALDI_ASSERT(size > 0);
    PairMapType::const_iterator iter = other_probs_.find(ngram);    
    if (iter != other_probs_.end())
      return iter->second.first;
  }
  // If we reached here then there was no direct probability.
  if (size == 1) {
    // There should be a direct probability for each ngram, do a range check.
    KALDI_ERR << "Found no probability for word " << ngram[0]
              << ", vocab size is " << vocab_size_;
  }
  std::vector<int32> hist_state,
      backoff_ngram;
  hist_state.insert(hist_state.begin(), ngram.begin(),
                    ngram.begin() + size - 1);
  backoff_ngram.insert(backoff_ngram.begin(), ngram.begin() + 1,
                       ngram.end());
  PairMapType::const_iterator hist_iter = other_probs_.find(hist_state);
  BaseFloat backoff_prob = ((hist_iter == other_probs_.end()) ? 1.0 :
                            hist_iter->second.second);
  return backoff_prob * GetProb(backoff_ngram);
}


void LmHistoryStateMap::Init(const LanguageModel &lm) {
  lm_history_states_.clear();
  lm_history_states_.reserve(lm.other_probs_.size() + 1);
  // Add the empty history-state; this is not present explicitly in the lm, but
  // it has to be considered as a valid history-state.
  lm_history_states_.push_back(std::vector<int32>());
  LanguageModel::PairMapType::const_iterator iter = lm.other_probs_.begin(),
      end = lm.other_probs_.end();
  for (; iter != end; ++iter) {
    BaseFloat backoff_prob = iter->second.second;
    if (backoff_prob != 1.0) {
      // If the backoff-prob for this sequence is not 1.0, it means it
      // exists as a history state.
      lm_history_states_.push_back(iter->first);
    }
  }
  // this is to make sure the order is deterministic, in case there
  // is randomness somewhere in the hashing code (unlikely but possible).
  std::sort(lm_history_states_.begin(), lm_history_states_.end());
  // there should be no repeats, or it would be an error in the stl hashing code.
  KALDI_ASSERT(IsSortedAndUniq(lm_history_states_));

  history_to_state_.clear();
  for (size_t i = 0; i < lm_history_states_.size(); i++)
    history_to_state_[lm_history_states_[i]] = static_cast<int32>(i);
   
}

const std::vector<int32>& LmHistoryStateMap::GetHistoryForState(
    int32 lm_history_state) const {
  KALDI_ASSERT(static_cast<size_t>(lm_history_state) < lm_history_states_.size());
  return lm_history_states_[lm_history_state];
}

BaseFloat LmHistoryStateMap::GetProb(const LanguageModel &lm,
                                     int32 lm_history_state,
                                     int32 predicted_word) const {
  std::vector<int32> vec = GetHistoryForState(lm_history_state);
  vec.push_back(predicted_word);
  return lm.GetProb(vec);
}

int32 LmHistoryStateMap::GetLmHistoryState(std::vector<int32> &hist) const {
  IntMapType::const_iterator iter = history_to_state_.find(hist);
  if (iter != history_to_state_.end())
    return iter->second;
  KALDI_ASSERT(!hist.empty());
  std::vector<int32> shorter_hist;
  shorter_hist.insert(shorter_hist.begin(), hist.begin() + 1,
                      hist.end());
  return GetLmHistoryState(shorter_hist);
}



LanguageModelEstimator::LanguageModelEstimator(const LanguageModelOptions &opts,
                                               int32 vocab_size):
    opts_(opts), vocab_size_(vocab_size), counts_(opts_.ngram_order + 1),
    history_state_counts_(opts_.ngram_order) {
  std::vector<int32> word_vec(1);
  // for all words, and EOS (0), add a zero count.  This is the easiest way to
  // ensure that when we eventually create the LM, all words will get given
  // explicit probabilities even if they have no count (it comes from
  // smoothing).
  for (int32 word = 0; word <= vocab_size; word++) {
    word_vec[0] = word;
    AddCountForNgram(word_vec, 0.0);
  }
}

void LanguageModelEstimator::AddCounts(std::vector<int32> &sentence) {
  int32 order = opts_.ngram_order, vocab_size = vocab_size_;
  { // Do a sanity check on the input.
    std::vector<int32>::const_iterator iter = sentence.begin(),
        end = sentence.end();
    for (; iter != end; ++iter) {
      KALDI_ASSERT(*iter > 0 && *iter <= vocab_size);
    }
  }  
  
  std::vector<int32> ngram;
  ngram.push_back(0); // <s>
  for (int32 i = 0; i + 1 < order && i < sentence.size(); i++) {
    ngram.push_back(sentence[i]);
    AddCountForNgram(ngram, 1.0);
  }
  for (int32 i = order; i < sentence.size(); i++) {
    ngram.erase(ngram.begin());  // remove the first element
    ngram.push_back(sentence[i]);
    AddCountForNgram(ngram, 1.0);
  }
  ngram.erase(ngram.begin()); // remove the first element
  ngram.push_back(0);  // Add a 0 for end-of-sentence marker
  AddCountForNgram(ngram, 1.0);
}

BaseFloat LanguageModelEstimator::GetDiscountAmount(BaseFloat count) const {
  KALDI_ASSERT(count >= 0.0);
  if (count >= 2.0) {
    return opts_.discount2plus;
  } else if (count <= 1.0) {
    // interpolate linearly between 0 and 1.
    return count * opts_.discount1;
  } else { // 1.0 < count < 2.0
    // interpolate linearly between discount1 at count = 1.0
    // and discount2 at count = 2.0.
    return opts_.discount1 + (count-1.0) * (opts_.discount2plus -
                                            opts_.discount1);
  }
}

void LanguageModelEstimator::Discount() {
  SetType protected_states;
  
  for (int32 order = opts_.ngram_order; order >= 1; order--) {
    // We do the normal discounting before applying the count cutoffs, because
    // inside here is where the total counts for history-states are computed.
    DiscountForOrder(order);
        
    if (order > 1) {
      // Before doing the regular discounting, completely discount any n-grams
      // that belong to language-model states whose counts are too small.
      BaseFloat count_cutoff =
          (order - 1 == 1 ? opts_.state_count_cutoff1 :
                            opts_.state_count_cutoff2plus);
      SetType next_protected_states;
      ApplyHistoryStateCountCutoffForOrder(order - 1, count_cutoff,
                                           protected_states,
                                           &next_protected_states);
      protected_states.swap(next_protected_states);
    }
  }
}

void LanguageModelEstimator::DiscountForOrder(int32 order) {
  KALDI_ASSERT(order >= 1 && order <= opts_.ngram_order);
  // Discount the counts for each n-gram of this order, adding
  // them to the backoff state, and recording the discounted
  // amounts for the history-state.
  MapType::iterator iter = counts_[order].begin(), end = counts_[order].end();
  for (; iter != end; ++iter) {
    const std::vector<int32> &ngram = iter->first;
    BaseFloat count = iter->second;
    BaseFloat discounted_amount = GetDiscountAmount(count);
    iter->second -= discounted_amount;
    std::vector<int32> history_state(ngram);
    history_state.pop_back();
    // record the amount we discounted (which will later be used in estimating
    // backoff weights), as well as incrementing the total count for the
    // history state.
    AddCountsForHistoryState(history_state, count, discounted_amount);
    if (order > 1) {
      std::vector<int32> backoff_ngram(ngram);
      backoff_ngram.erase(backoff_ngram.begin());  // Erase first element.
      AddCountForNgram(backoff_ngram, discounted_amount);
    }
  }
}

// inline static
void LanguageModelEstimator::RemoveFront(const std::vector<int32> &vec,
                                         std::vector<int32> *backoff_vec) {
  KALDI_PARANOID_ASSERT(!vec.empty() && backoff_vec->empty());
  backoff_vec->insert(backoff_vec->begin(),
                      vec.begin() + 1,
                      vec.end());
}


void LanguageModelEstimator::ApplyHistoryStateCountCutoffForOrder(
    int32 order, BaseFloat min_count,
    const SetType &protected_states,
    SetType *protected_backoff_states) {

  SetType deleted_history_states;
  PairMapType &history_state_counts = history_state_counts_[order];
  {
    PairMapType::iterator iter = history_state_counts.begin(),
        end = history_state_counts.end();
    for (; iter != end; ) {
      const std::vector<int32> &history_state = iter->first;
      BaseFloat tot_count = iter->second.first;
      if (tot_count < min_count && protected_states.count(history_state) == 0) {
        // we will delete this history-state.
        deleted_history_states.insert(history_state);
        // this erase function returns an iterator to the next position.
        iter = history_state_counts.erase(iter);
      } else {
        // we will not delete this history-state.  Mark the backoff
        // state of this state as un-deletable.
        std::vector<int32> backoff_history_state;
        RemoveFront(history_state, &backoff_history_state);
        protected_backoff_states->insert(backoff_history_state);
        // Also make sure the n-gram with the same vector as this history-state
        // is not deletable (i.e. for history state a b c, the n-gram a b -> c
        // must not be deleted, so we need to also keep the history-state "a b".
        std::vector<int32> other_protected_history_state(history_state);
        other_protected_history_state.pop_back();
        protected_backoff_states->insert(other_protected_history_state);
        ++iter;
      }
    }
  }
  if (!deleted_history_states.empty()) {
    // Erase the counts that correspond to deleted history states, assigning
    // those counts to the corresponding backed-off ngrams.
    MapType &counts = counts_[order+1];
    MapType::iterator iter = counts.begin(), end = counts.end();
    for (; iter != end; ) {
      std::vector<int32> history = iter->first;  
      history.pop_back();  // get history state by popping predicted word
      if (deleted_history_states.count(history) != 0) {  // we must delete it
        BaseFloat count = iter->second;
        std::vector<int32> backoff_ngram;
        RemoveFront(iter->first, &backoff_ngram);
        AddCountForNgram(backoff_ngram, count);
        // this erase function returns an iterator to the next position.
        iter = counts.erase(iter);  
      } else {
        ++iter;
      }
    }
  }
}

//inline static
void LanguageModelEstimator::AddToMap(const std::vector<int32> &vec,
                             BaseFloat count,
                             MapType *map) {
  MapType::iterator iter = map->find(vec);
  if (iter != map->end()) {
    iter->second += count;
  } else {
    (*map)[vec] = count;
  }
}

// inline static
void LanguageModelEstimator::AddPairToMap(const std::vector<int32> &vec,
                                          BaseFloat count1, BaseFloat count2,
                                          PairMapType *map) {
  PairMapType::iterator iter = map->find(vec);
  if (iter != map->end()) {
    iter->second.first += count1;
    iter->second.second += count2;
  } else {
    (*map)[vec] = std::pair<BaseFloat,BaseFloat>(count1, count2);
  }
}


void LanguageModelEstimator::AddCountForNgram(const std::vector<int32> &vec,
                                              BaseFloat count) {
  AddToMap(vec, count, &(counts_[vec.size()]));
}

// inline
BaseFloat LanguageModelEstimator::GetCountForNgram(const std::vector<int32> &vec) const {
  int32 size = vec.size();
  MapType::const_iterator iter = counts_[size].find(vec);
  if (iter != counts_[size].end())
    return iter->second;
  else
    return 0.0;
}

// inline
void LanguageModelEstimator::AddCountsForHistoryState(
    const std::vector<int32> &hist, BaseFloat tot_count,
    BaseFloat discounted_count) {    
  AddPairToMap(hist, tot_count, discounted_count,
               &(history_state_counts_[hist.size()]));
}

std::pair<BaseFloat,BaseFloat> LanguageModelEstimator::GetCountsForHistoryState(
    const std::vector<int32> &vec) const {
  int32 size = vec.size();
  PairMapType::const_iterator iter = history_state_counts_[size].find(vec);
  if (iter != history_state_counts_[size].end())
    return iter->second;
  else
    return std::pair<BaseFloat,BaseFloat>(0.0, 0.0);
}

BaseFloat LanguageModelEstimator::GetProb(
    const std::vector<int32> &ngram) const {
  size_t size = ngram.size();
  KALDI_ASSERT(size > 0);
  std::vector<int32> hist(ngram);
  hist.pop_back();
  std::pair<BaseFloat,BaseFloat> hist_counts = GetCountsForHistoryState(hist);
  BaseFloat direct_prob, backoff_prob;
  if (hist_counts.first == 0) {
    // History state does not exist -> pure backoff.
    direct_prob = 0.0;
    backoff_prob = 1.0;
  } else {
    BaseFloat ngram_count = GetCountForNgram(ngram),
        tot_count = hist_counts.first,
        backoff_count = hist_counts.second;
    direct_prob = ngram_count / tot_count;
    backoff_prob = backoff_count / tot_count;
    KALDI_PARANOID_ASSERT(direct_prob <= 1.0 && backoff_prob <= 1.0);
  }
  KALDI_ASSERT(backoff_prob > 0.0);
  // the way this code works, there is backoff even for 1-grams:
  // we back off to a zero-gram model where probability is distributed
  // evenly among the vocab items.
  if (size > 1) {
    std::vector<int32> backoff_ngram;
    RemoveFront(ngram, &backoff_ngram);
    return direct_prob + backoff_prob * GetProb(backoff_ngram);
  } else {
    // backoff is to zero-gram.  the zero-gram model distributes the probability
    // equally among all words, plus EOS (hence the + 1 in vocab_size_ + 1).
    return direct_prob + backoff_prob / (vocab_size_ + 1);
  }
}


void LanguageModelEstimator::Output(LanguageModel *lm) const {

  lm->vocab_size_ = vocab_size_;
  lm->ngram_order_ = opts_.ngram_order;
  lm->highest_order_probs_.clear();
  lm->other_probs_.clear();

  for (int32 order = 1; order < opts_.ngram_order; order++) {
    // For all n-grams except the highest order, add their probabilities to
    // other_probs_, and set the backoff weight to 1 (which is the default
    // unless we actually have that as a history state).  Note: for order 1, all
    // words are in the map; we added them with zero counts when we initialized
    // the LanguageModelEstimator object.
    MapType counts = counts_[order];
    MapType::const_iterator iter = counts.begin(), end = counts.end();
    for (; iter != end; ++iter) {
      const std::vector<int32> &ngram = iter->first;
      BaseFloat prob = GetProb(ngram);
      KALDI_ASSERT(prob > 0.0);
      lm->other_probs_[ngram] = std::pair<BaseFloat,BaseFloat>(prob, 1.0);
    }
  }
  {
    // For the highest-order n-grams, add them to the highest_order_probs_ map.
    MapType counts = counts_[opts_.ngram_order];
    MapType::const_iterator iter = counts.begin(), end = counts.end();
    for (; iter != end; ++iter) {
      const std::vector<int32> &ngram = iter->first;
      BaseFloat prob = GetProb(ngram);
      KALDI_ASSERT(prob > 0.0);
      lm->highest_order_probs_[ngram] = prob;
    }
  }
  // Now add the backoff probabilities for all history states
  // We can assert as we add these, that probs already exist for these
  // n-grams (except for the empty history state, which of course has
  // no n-gram, and which anyway has no backoff weight; note that there
  // is no symbol for <UNK> here, we assume a fixed known vocabulary size).
  for (int32 order = 1; order < opts_.ngram_order; order++) {
    const PairMapType &history_state_counts = history_state_counts_[order];
    PairMapType::const_iterator iter = history_state_counts.begin(),
        end = history_state_counts.end();
    for (; iter != end; ++iter) {
      const std::vector<int32> &hist = iter->first;
      BaseFloat tot_count = iter->second.first,
          backoff_count = iter->second.second;
      KALDI_ASSERT(tot_count > 0.0 && backoff_count > 0.0 &&
                   backoff_count < tot_count);
      BaseFloat backoff_prob = backoff_count / tot_count;
      PairMapType::iterator out_iter = lm->other_probs_.find(hist);
      if (out_iter == lm->other_probs_.end())
        KALDI_ERR << "We have backoff state for an n-gram that doesn't exist.";
      out_iter->second.second = backoff_prob;
    }
  }
}

BaseFloat ComputePerplexity(const LanguageModel &lm,
                            std::vector<std::vector<int32> > &sentences) {
  double tot_logprob = 0.0;
  int32 num_probs = 0, order = lm.NgramOrder();
  for (size_t i = 0; i < sentences.size(); i++) {
    std::vector<int32> vec;
    vec.push_back(0); // BOS
    std::vector<int32> sent(sentences[i]);
    sent.push_back(0); // EOS;
    for (size_t j = 0; j < sent.size(); j++) {
      if (vec.size() > order - 1)
        vec.erase(vec.begin());  // shift left.
      vec.push_back(sent[j]);
      BaseFloat prob = lm.GetProb(vec);
      KALDI_ASSERT(prob > 0.0 && prob <= 1.0);
      tot_logprob += log(prob);
      num_probs++;
    }
  }
  BaseFloat perplexity = exp(-tot_logprob / num_probs);
  return perplexity;
}


}  // namespace ctc
}  // namespace kaldi



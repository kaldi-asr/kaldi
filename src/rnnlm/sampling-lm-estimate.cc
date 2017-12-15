// sampling-lm-estimate.cc

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

#include <iomanip>
#include <numeric>
#include "rnnlm/sampling-lm-estimate.h"

namespace kaldi {
namespace rnnlm {


void SamplingLmEstimatorOptions::Check() const {
  KALDI_ASSERT(vocab_size > 2);
  KALDI_ASSERT(bos_symbol > 0 && bos_symbol < vocab_size);
  KALDI_ASSERT(eos_symbol > 0 && eos_symbol < vocab_size);
  KALDI_ASSERT(eos_symbol != bos_symbol);
  KALDI_ASSERT(ngram_order >= 1 &&
               discounting_constant > 0 && discounting_constant <= 1.0 &&
               unigram_factor > 0.0 && backoff_factor > 0.0 &&
               unigram_factor > backoff_factor &&
               bos_factor > 0.0 && bos_factor <= unigram_factor);
  KALDI_ASSERT(unigram_power > 0.2 && unigram_power <= 1.0);
}

SamplingLmEstimator::SamplingLmEstimator(
    const SamplingLmEstimatorOptions &config):
    config_(config) {
  config_.Check();
  history_states_.resize(config.ngram_order);
}


void SamplingLmEstimator::ProcessLine(BaseFloat corpus_weight,
                                       const std::vector<int32> &sentence) {
  KALDI_ASSERT(corpus_weight >= 0.0);
  int32 ngram_order = config_.ngram_order,
      sentence_length = sentence.size(),
      vocab_size = config_.vocab_size;
  std::vector<int32> history;
  history.push_back(config_.bos_symbol);
  int32 i;
  for (i = 0; i < sentence_length && i + 1 < ngram_order; i++) {
    int32 this_word = sentence[i];
    // note: 0 is reserved for <eps>.
    KALDI_ASSERT(this_word > 0 && this_word < vocab_size);
    AddCount(history, this_word, corpus_weight);
    history.push_back(this_word);
  }
  for (; i < sentence_length; i++) {
    history.erase(history.begin());
    int32 this_word = sentence[i];
    AddCount(history, this_word, corpus_weight);
    history.push_back(this_word);
  }
  if (history.size() >= static_cast<size_t>(ngram_order))
    history.erase(history.begin());
  AddCount(history, config_.eos_symbol, corpus_weight);

  // TODO: remove the following.
  KALDI_ASSERT(history.size() == std::min(ngram_order - 1,
                                          sentence_length + 1));
}


void SamplingLmEstimator::Process(std::istream &is) {
  int32 num_lines = 0;
  std::vector<int32> words;
  std::string line;
  while (getline(is, line)) {
    num_lines++;
    std::istringstream line_is(line);
    BaseFloat weight;
    line_is >> weight;
    words.clear();
    int32 word;
    while (line_is >> word) {
      words.push_back(word);
    }
    if (!line_is.eof()) {
      KALDI_ERR << "Could not interpret input: " << line;
    }
    this->ProcessLine(weight, words);
  }
  KALDI_LOG << "Processed " << num_lines << " lines of input.";
}


void SamplingLmEstimator::HistoryState::AddCount(int32 word,
                                                  BaseFloat corpus_weight) {
  new_counts.push_back(std::pair<int32, BaseFloat>(word, corpus_weight));
  if (new_counts.size() == new_counts.capacity() &&
      new_counts.size() >= counts.size()) {
    bool release_memory = false;
    ProcessNewCounts(release_memory);
  }
}

void SamplingLmEstimator::HistoryState::ComputeTotalCount() {
  double tmp_total_count = 0.0;
  std::vector<Count>::const_iterator iter = counts.begin(),
      end = counts.end();
  for (; iter != end; ++iter)
    tmp_total_count += iter->count;
  total_count = tmp_total_count;
}

// static
void SamplingLmEstimator::SortAndUniqCounts(std::vector<Count> *counts) {
  // sort the vector<Count> in counts; they get sorted on 'word'
  // so that Counts with the same word are next to each other.
  std::sort(counts->begin(), counts->end());

  {
    // This block merges counts in '*counts' that have the same word.
    // It is adapted from MergePairVectorSumming().  This code is quite
    // optimized and not easy to read but MergePairVectorSumming() is well
    // tested and the changes in the code are small.
    std::vector<Count>::iterator out_iter = counts->begin(),
        in_iter = counts->begin(), end_iter = counts->end();
    // special case: while there is nothing to be changed, skip over
    // initial input (avoids unnecessary copying).
    while (in_iter + 1 < end_iter && in_iter[0].word != in_iter[1].word) {
      in_iter++;
      out_iter++;
    }
    while (in_iter < end_iter) {
      // We reach this point only at the first element of
      // each stretch of identical words
      *out_iter = *in_iter;
      ++in_iter;
      while (in_iter < end_iter && in_iter->word == out_iter->word) {
        if (in_iter->highest_count > out_iter->highest_count)
          out_iter->highest_count = in_iter->highest_count;
        out_iter->count += in_iter->count;
        ++in_iter;
      }
      out_iter++;
    }
    counts->erase(out_iter, end_iter);
  }
}


// see header for what this does.
void SamplingLmEstimator::HistoryState::ProcessNewCounts(bool release_memory) {
  if (!new_counts.empty()) {
    // Merge all counts in 'new_counts' into 'counts'.
    // We could do what we're doing with 'merged_counts' below, with 'counts'
    // directly, but that might increase the memory held in this HistoryState so
    // we use a temporary vector in some cases.
    std::vector<Count> tmp;
    size_t orig_counts_size = counts.size(),
        merge_size = orig_counts_size + new_counts.size();
    // 'merge_location' is the vector we use to merge and sort the Counts, it
    // either points to 'tmp' or to 'this->counts'.
    std::vector<Count> *merge_location;
    if (merge_size > counts.capacity()) {
      merge_location = &tmp;
      tmp.reserve(merge_size);
      tmp.insert(tmp.end(), counts.begin(), counts.end());
    } else {
      merge_location = &counts;
    }

    { // this block converts each member of 'new_counts' into a single Count,
      // appending it to the vector pointed to by 'merge_location'.
      merge_location->resize(merge_size);
      std::vector<std::pair<int32, BaseFloat> >::const_iterator
          in_iter = new_counts.begin();
      std::vector<Count>::iterator out_iter =
          merge_location->begin() + orig_counts_size,
          out_end = merge_location->end();
      for (; out_iter != out_end; ++in_iter, ++out_iter) {
        int32 word = in_iter->first;
        BaseFloat count = in_iter->second;
        out_iter->word = word;
        out_iter->highest_count = count;
        out_iter->count = count;
      }
    }
    SortAndUniqCounts(merge_location);
    if (merge_location != &counts) // copy to 'counts' if we were using a temporary.
      counts = *merge_location;
  }

  if (release_memory) {
    // shallow swapping with a new temporary is a trick to release memory from a
    // std::vector.
    std::vector<std::pair<int32, BaseFloat> > new_counts_temp;
    new_counts.swap(new_counts_temp);
  } else {
    new_counts.clear();
  }
}

void SamplingLmEstimator::ComputeRawCountsForOrder(int32 o) {
  KALDI_ASSERT(o >= 1 && o < config_.ngram_order);

  // We first make a map from the backed-off history to a list of the
  // history-states that back off to it.  This will help us to do the backoff in
  // a relatively memory-efficient way.
  unordered_map<std::vector<int32>,
      std::vector<const HistoryState*>, VectorHasher<int32> > lower_to_higher_order;


  // Normally we'd iterate over history_states_[o-1] to access counts of order
  // o, but we are iterating over history-states for the order one higher than
  // o.
  unordered_map<std::vector<int32>,
      HistoryState*, VectorHasher<int32> >::iterator
    iter = history_states_[o].begin(),
    end = history_states_[o].end();
  for (; iter != end; ++iter) {
    const std::vector<int32> &history = iter->first;
    // remove the left-most (most distant) word from the history to back off.
    std::vector<int32> backed_off_history(history.begin() + 1,
                                          history.end());
    const HistoryState *higher_order_state = iter->second;
    lower_to_higher_order[backed_off_history].push_back(higher_order_state);
  }

  unordered_map<std::vector<int32>, std::vector<const HistoryState*>,
      VectorHasher<int32> >::const_iterator
      state_list_iter = lower_to_higher_order.begin(),
      state_list_end= lower_to_higher_order.end();
  for (; state_list_iter != state_list_end; ++state_list_iter) {
    const std::vector<int32> &history = state_list_iter->first;
    const std::vector<const HistoryState*> &higher_order_states =
        state_list_iter->second;
    HistoryState *this_state = GetHistoryState(history, true);
    std::vector<Count> merged_counts;
    size_t merged_counts_size = 0;
    for (size_t i = 0; i < higher_order_states.size(); i++)
      merged_counts_size += higher_order_states[i]->counts.size();
    merged_counts.reserve(merged_counts_size);
    for (size_t i = 0; i < higher_order_states.size(); i++)
      merged_counts.insert(merged_counts.end(),
                           higher_order_states[i]->counts.begin(),
                           higher_order_states[i]->counts.end());
    SortAndUniqCounts(&merged_counts);
    this_state->counts = merged_counts;
  }
}


SamplingLmEstimator::HistoryState*
SamplingLmEstimator::GetHistoryState(const std::vector<int32> &history,
                                      bool add_if_absent) {
  KALDI_ASSERT(static_cast<int32>(history.size()) < config_.ngram_order);
  // 'values' is a reference to a pointer to a HistoryState.
  // If 'history' did not previously exist as a key in the map, this will
  // be NULL.  This is a feature of the stl that's valid at least as of C++11,
  // whereby POD types that are the values in maps will be set to zero
  // if newly created.
  /// https://stackoverflow.com/questions/8943261/stdunordered-map-initialization
  // or search for unordered_map value-initialized
  HistoryState *&value = history_states_[history.size()][history];
  if (value == NULL) {
    if (add_if_absent) {
      value = new HistoryState();
    } else {
      KALDI_ERR << "Expected history-state to exist (code error).";
    }
  }
  return value;
}

void SamplingLmEstimator::FinalizeRawCountsForOrder(int32 o) {
  KALDI_ASSERT(o >= 1 && o <= config_.ngram_order &&
               static_cast<int32>(history_states_.size()) ==
               config_.ngram_order);
  unordered_map<std::vector<int32>,
      HistoryState*, VectorHasher<int32> >::iterator
    iter = history_states_[o - 1].begin(),
    end = history_states_[o - 1].end();
  for (; iter != end; ++iter) {
    if (o == config_.ngram_order) {
      bool release_memory = true;
      iter->second->ProcessNewCounts(release_memory);
    }
    iter->second->ComputeTotalCount();
  }
}

void SamplingLmEstimator::Estimate(bool will_write_arpa) {
  for (int32 o = config_.ngram_order; o >= 1; o--) {
    if (o < config_.ngram_order)
      ComputeRawCountsForOrder(o);
    FinalizeRawCountsForOrder(o);
  }
  // Now we have the raw counts of orders but we have not yet done backoff or
  // pruning.
  ComputeUnigramDistribution();

  for (int32 o = 2; o <= config_.ngram_order; o++) {
    SmoothDistributionForOrder(o);
    PruneNgramsForOrder(o);
  }
  for (int32 o = config_.ngram_order; o >= 2; o--)
    PruneStatesForOrder(o, will_write_arpa);

  TakeUnigramCountsToPower(config_.unigram_power);
}

void SamplingLmEstimator::SmoothDistributionForOrder(int32 o) {
  KALDI_ASSERT(o >= 2 && o <= config_.ngram_order);
  unordered_map<std::vector<int32>, HistoryState*,
      VectorHasher<int32> >::iterator
    iter = history_states_[o-1].begin(), end = history_states_[o-1].end();
  BaseFloat D = config_.discounting_constant;  // 0 < D < 1.
  for (; iter != end; ++iter) {
    HistoryState *state = iter->second;
    KALDI_ASSERT(state->total_count > 0.0 && state->backoff_count == 0.0);
    std::vector<Count>::iterator counts_iter = state->counts.begin(),
        counts_end = state->counts.end();
    double backoff_count_tot = 0.0;
    for (; counts_iter != counts_end; ++counts_iter) {
      // note: in the case without data weightings, 'highest_count' will always be
      // 1 and so removed_count will equal D.
      BaseFloat removed_count = D * counts_iter->highest_count;
      counts_iter->count -= removed_count;
      backoff_count_tot += removed_count;
    }
    state->backoff_count = backoff_count_tot;
  }
}


void SamplingLmEstimator::PruneNgramsForOrder(int32 o) {
  KALDI_ASSERT(o >= 2 && o <= config_.ngram_order);
  unordered_map<std::vector<int32>, HistoryState*,
      VectorHasher<int32> >::iterator
    iter = history_states_[o-1].begin(), end = history_states_[o-1].end();
  size_t orig_num_ngrams = 0, cur_num_ngrams = 0;

  for (; iter != end; ++iter) {
    HistoryState *state = iter->second;
    orig_num_ngrams += state->counts.size();
    const std::vector<int32> &history = iter->first;
    KALDI_ASSERT(history.size() == o - 1);
    if (o > 2) {
      std::vector<int32> backoff_history(history);
      std::vector<const HistoryState*> backoff_states;
      while (backoff_history.size() > 1) {
        backoff_history.erase(backoff_history.begin());
        const HistoryState *backoff_state = GetHistoryState(backoff_history,
                                                            false);
        backoff_states.push_back(backoff_state);
      }
      PruneHistoryStateAboveBigram(history, backoff_states, state);
    } else { // o == 2
      PruneHistoryStateBigram(history, state);
    }
    cur_num_ngrams += state->counts.size();
  }
  KALDI_LOG << "For n-gram order " << o << ", pruned from "
            << orig_num_ngrams << " to " << cur_num_ngrams << " ngrams.";
}


void SamplingLmEstimator::PruneStatesForOrder(int32 o, bool will_write_arpa) {
  KALDI_ASSERT(o >= 2 && o <= config_.ngram_order);
  unordered_map<std::vector<int32>, HistoryState*,
      VectorHasher<int32> >::iterator
    iter = history_states_[o-1].begin(), end = history_states_[o-1].end();
  size_t orig_num_states = history_states_[o-1].size(),
      num_restored_ngrams = 0;

  // 'states_to_delete' will contain histories whose states we want to delete
  // from history_states_ because all their words have been pruned away (and they
  // are not protected.
  std::unordered_set<std::vector<int32>, VectorHasher<int32> > states_to_delete;
  for (; iter != end; ++iter) {
    const std::vector<int32> &history = iter->first;
    HistoryState *state = iter->second;

    if (state->counts.empty() && !state->is_protected) {
      // we'll delete this history state.
      states_to_delete.insert(history);
    } else {
      // we'll keep this history state; mark any state that it backs off
      // to as protected from being deleted, or we'll have problems later on.
      if (history.size() > 1) {
        std::vector<int32> backoff_history(history.begin() + 1, history.end());
        // mark the state this state backs off to, as protected.
        GetHistoryState(backoff_history, false)->is_protected = true;

        if (will_write_arpa) {
          // Make sure that the n-gram corresponding to this history state
          // was not deleted; if it was, we have to put it back into the
          // model because it's required to exist in the ARPA model.
          std::vector<int32> prev_history(history.begin(), history.end() - 1);
          int32 last_word = history.back();
          // we're checking that the n-gram 'prev_history -> last_word' exists.
          HistoryState *prev_state = GetHistoryState(prev_history, false);
          Count c;
          c.word = last_word;
          std::vector<Count>::iterator
              iter = std::lower_bound(prev_state->counts.begin(),
                                      prev_state->counts.end(),
                                      c);
          if (iter == prev_state->counts.end() || iter->word != last_word) {
            // The n-gram leading to this state had been pruned away, which will
            // give us problems when writing the ARPA model (there would be
            // nowhere to write the backoff probability for this state).  We
            // insert it back.
            //
            // To compute how much of the total count would have been backed off
            // in the deleted ngram, We guess at the highest_count, assuming
            // it's probably close to 1.0.  This will make almost no difference
            // to anything, these ngrams will be quite rare.
            BaseFloat backoff_count =
                std::min<BaseFloat>(config_.discounting_constant,
                                    0.5 * state->total_count);
            // because of how we store the stats, the count for that (now-lost)
            // ngram would have been equal to state->total_count before pruning.
            // Assuming we pruned 'backoff_count' from it, its value after
            // pruning would have been roughly as follows:
            c.count = state->total_count - backoff_count;
            c.highest_count = -123.4;  // convenient for future debugging.
            prev_state->backoff_count -= c.count;  // claw it back.
            KALDI_ASSERT(prev_state->backoff_count > 0.0);
            prev_state->counts.insert(iter, c);
            num_restored_ngrams++;
          }
        }
      }
    }
  }

  { // remove from the map history states that we marked for deletion.
    std::unordered_set<std::vector<int32>, VectorHasher<int32> >::const_iterator
        iter = states_to_delete.begin(), end = states_to_delete.end();
    for (; iter != end; ++iter) {
      const std::vector<int32> &history = *iter;
      delete history_states_[o-1][history];
      history_states_[o-1].erase(history);
    }
  }
  size_t cur_num_states = history_states_[o-1].size();
  std::ostringstream message;
  message << "For n-gram order " << o << ", pruned from "
          << orig_num_states << " to " << cur_num_states << " states";
  if (num_restored_ngrams > 0) {
    message << ", and restored "  << num_restored_ngrams << " required n-grams.";
  }
  KALDI_LOG << message.str();
}


// Prunes a history-state; this version is for the bigram state
// whose left-context is the BOS symbol.
void SamplingLmEstimator::PruneHistoryStateBigram(
    const std::vector<int32> &history, HistoryState *state) {
  KALDI_ASSERT(history.size() == 1);
  BaseFloat total_count = state->total_count;
  bool is_bos_state = (history[0] == config_.bos_symbol);
  // e.g. factor = is_bos ? 5.0 : 50.0 by default, meaning we keep
  // more n-grams for the BOS state than for a typical bigram state;
  // this is because the BOS state tends to be seen non-independently
  // within the minibatch.
  BaseFloat factor = is_bos_state ? config_.bos_factor : config_.unigram_factor;
  KALDI_ASSERT(factor > 0.0);
  // 'factor' is the factor by which the probability given this
  // history state must be greater than the probability given the
  // unigram state.

  std::vector<Count>::iterator iter = state->counts.begin(),
      end = state->counts.end();
  double backoff_count = state->backoff_count;  // accumulate in double
  for (; iter != end; ++iter) {
    Count &count = *iter;
    BaseFloat unigram_prob = unigram_probs_[count.word],
        bigram_prob_no_backoff = count.count / total_count;
    // note: when computing bigram_prob_no_backoff when deciding which thing to
    // prune, we ignore the backoff term 'state->backoff_count * unigram_prob /
    // total_count'.  This prevents the need for iteration and keeps things
    // simple.
    if (bigram_prob_no_backoff <= unigram_prob * factor) {
      // Completely prune this count away.
      backoff_count += count.count;
      count.count = 0.0;
    }
  }
  state->backoff_count = backoff_count;
  RemoveZeroCounts(&(state->counts));
}


void SamplingLmEstimator::PruneHistoryStateAboveBigram(
      const std::vector<int32> &history,
      const std::vector<const HistoryState*> &backoff_states,
      HistoryState *state) {
  BaseFloat unigram_factor = config_.unigram_factor,
      backoff_factor = config_.backoff_factor;
  BaseFloat total_count = state->total_count;
  KALDI_ASSERT(unigram_factor > 0.0 && backoff_factor >  0.0 &&
               unigram_factor > backoff_factor);
  std::vector<Count>::iterator iter = state->counts.begin(),
      end = state->counts.end();
  double backoff_count = state->backoff_count;  // accumulate in double
  for (; iter != end; ++iter) {
    Count &count = *iter;
    BaseFloat current_prob_no_backoff = count.count / total_count,
        prob_given_backoff_state = GetProbForWord(count.word,
                                                  backoff_states),
        unigram_prob = unigram_probs_[count.word];
    if (!(current_prob_no_backoff > unigram_factor * unigram_prob &&
          current_prob_no_backoff > backoff_factor * prob_given_backoff_state)) {
      // Remove this word.  It's not probable enough to keep according to our
      // rules.
      backoff_count += count.count;
      count.count = 0.0;
    }
  }
  state->backoff_count = backoff_count;
  RemoveZeroCounts(&(state->counts));
}


BaseFloat SamplingLmEstimator::GetProbForWord(
    int32 word, const std::vector<const HistoryState*> &states) const {
  // compute the probability from lowest to highest order.
  KALDI_ASSERT(word > 0 && word < static_cast<int32>(unigram_probs_.size()));
  BaseFloat ans = unigram_probs_[word];
  for (size_t i = 0; i < states.size(); i++) {
    const HistoryState *state = states[i];
    ans *= state->backoff_count / state->total_count;
    Count c;
    c.word = word;
    // look up word in the state's counts.
    std::vector<Count>::const_iterator
        iter = std::lower_bound(state->counts.begin(),
                                state->counts.end(),
                                c);
    if (iter != state->counts.end() && iter->word == word) {
      // we found the word
      ans += iter->count / state->total_count;
    }
  }
  return ans;
}

bool SamplingLmEstimator::IsProtected(const std::vector<int32> &history,
                                       int32 word) const {
  // n-grams of the highest order can't be protected because there
  // would be no higher-order state.
  if (static_cast<int32>(history.size()) + 1 == config_.ngram_order)
    return false;
  std::vector<int32> new_history;
  new_history.reserve(history.size() + 1);
  new_history.insert(new_history.end(), history.begin(), history.end());
  new_history.push_back(word);
  return (history_states_[new_history.size()].count(new_history) != 0);
}

BaseFloat SamplingLmEstimator::BackoffProb(
    const std::vector<int32> &history, int32 word) const {
  // n-grams of the highest order won't have their own history state.
  if (static_cast<int32>(history.size()) + 1 == config_.ngram_order)
    return 0.0;
  std::vector<int32> new_history;
  new_history.reserve(history.size() + 1);
  new_history.insert(new_history.end(), history.begin(), history.end());
  new_history.push_back(word);

  unordered_map<std::vector<int32>, HistoryState*,
      VectorHasher<int32>>::const_iterator iter =
      history_states_[new_history.size()].find(new_history);
  if (iter != history_states_[new_history.size()].end()) {
    HistoryState *state = iter->second;
    return state->backoff_count / state->total_count;
  } else {
    return 0.0;
  }
}


// static
void SamplingLmEstimator::RemoveZeroCounts(
    std::vector<SamplingLmEstimator::Count> *counts) {
  std::vector<Count>::const_iterator input_iter = counts->begin(),
      end = counts->end();
  std::vector<Count>::iterator output_iter = counts->begin();
  // this while loop is an optimization to avoid copying where
  // source and destination are the same; it could be removed.
  while (input_iter != end && input_iter->count != 0.0) {
    ++input_iter;
    ++output_iter;
  }
  for (; input_iter != end; ++input_iter) {
    if (input_iter->count != 0.0) {
      *output_iter = *input_iter;
      ++output_iter;
    }
  }
  counts->resize(output_iter - counts->begin());
}


void SamplingLmEstimator::ComputeUnigramDistribution() {
  int32 vocab_size = config_.vocab_size;
  if (history_states_[0].size() != 1) {
    KALDI_ERR << "There are no counts (no data processed?)";
  }
  HistoryState *unigram_state = history_states_[0].begin()->second;
  KALDI_ASSERT(unigram_state->backoff_count == 0.0);

  double discounted_count = 0.0;
  { // this block works out 'unigram_state->backoff_count' which is the same as
    // 'discounted_count', and discounts the counts in 'unigram_state->counts'.
    BaseFloat D = config_.discounting_constant;
    std::vector<Count>::iterator iter = unigram_state->counts.begin(),
        end = unigram_state->counts.end();
    for(; iter != end; ++iter) {
      BaseFloat count_to_discount = D * iter->highest_count;
      iter->count -= count_to_discount;
      discounted_count += count_to_discount;
    }
    unigram_state->backoff_count = discounted_count;
  }

  BaseFloat total_count = unigram_state->total_count;
  // 'uniform_prob' is the probability that we add to each word in the vocabulary
  // even if it was not seen.  We divide discounted_count equally among all
  // words; the - 2 is to exclude <eps> and <s>, which are never predicted.
  BaseFloat uniform_prob = (discounted_count / total_count) / (vocab_size - 2);
  KALDI_ASSERT(total_count > 0.0 && uniform_prob > 0.0);
  unigram_probs_.clear();
  unigram_probs_.resize(vocab_size, uniform_prob);
  unigram_probs_[0] = 0.0;
  unigram_probs_[config_.bos_symbol] = 0.0;

  std::vector<Count>::iterator iter = unigram_state->counts.begin(),
      end = unigram_state->counts.end();
  for(; iter != end; ++iter) {
    BaseFloat this_prob = iter->count / total_count;
    // 'this_prob' is the non-smoothed part of the unigram probability.
    unigram_probs_[iter->word] += this_prob;
  }

  double sum = std::accumulate(unigram_probs_.begin(),
                               unigram_probs_.end(), 0.0);
  KALDI_ASSERT(fabs(sum - 1.0) < 0.01);
}

SamplingLmEstimator::~SamplingLmEstimator() {
  for (size_t i = 0; i < history_states_.size(); i++) {
    for (auto iter = history_states_[i].begin(), end = history_states_[i].end();
         iter != end; ++iter)
      delete iter->second;
  }
}


void SamplingLmEstimator::TakeUnigramCountsToPower(BaseFloat power) {
  if (power == 1.0) return;
  double sum = 0.0;
  for (std::vector<BaseFloat>::iterator iter = unigram_probs_.begin(),
           end = unigram_probs_.end(); iter != end; ++iter) {
    *iter = std::pow(*iter, power);
    sum += *iter;
  }
  BaseFloat scale = 1.0 / sum;
  for (std::vector<BaseFloat>::iterator iter = unigram_probs_.begin(),
           end = unigram_probs_.end(); iter != end; ++iter)
    *iter = *iter * scale;
}


int32 SamplingLmEstimator::NumNgrams(int32 o) const {
  KALDI_ASSERT(o >= 1 && o <= config_.ngram_order);
  if (o == 1) {
    // - 1 is for <eps>.  <s> does get printed in the ARPA file, with a
    // probability of -99.
    return config_.vocab_size - 1;
  } else {
    int32 ans = 0;
    unordered_map<std::vector<int32>, HistoryState*,
        VectorHasher<int32> >::const_iterator
        iter = history_states_[o-1].begin(),
        end =  history_states_[o-1].end();
    for (; iter != end; ++iter) {
      HistoryState *state = iter->second;
      ans += static_cast<int32>(state->counts.size());
    }
    return ans;
  }
}

void SamplingLmEstimator::PrintNgramsUnigram(
    std::ostream &os, const fst::SymbolTable &symbols) const {
  int32 vocab_size = config_.vocab_size,
      bos_symbol = config_.bos_symbol;
  std::vector<int32> unigram_history;
  for (int32 word = 1; word < vocab_size; word++) {
    std::string printed_word = symbols.Find(word);
    KALDI_ASSERT(!printed_word.empty() && "Mismatching symbol-table?");
    BaseFloat word_logprob = (word == bos_symbol ? -99.0 :
                              log10(unigram_probs_[word]));
    BaseFloat backoff_prob = BackoffProb(unigram_history, word);
    os << word_logprob << '\t' << printed_word;
    if (backoff_prob != 0.0)  os << '\t' << log10(backoff_prob) << '\n';
    else os << '\n';
  }
}


void SamplingLmEstimator::PrintNgramsAboveUnigram(
    std::ostream &os, int32 o, const fst::SymbolTable &symbols) {
  unordered_map<std::vector<int32>, HistoryState*,
      VectorHasher<int32> >::const_iterator
      hist_iter = history_states_[o-1].begin(),
      hist_end =  history_states_[o-1].end();
  for (; hist_iter != hist_end; ++hist_iter) {
    const std::vector<int32> &history = hist_iter->first;
    const HistoryState *state = hist_iter->second;

    // 'backoff_states' will list states that 'state' backs off to down to and
    // including bigram.  it's used when computing probabilities.  It will be
    // empty if o == 2.
    std::vector<const HistoryState*> backoff_states;
    { // this block sets up 'states'
      std::vector<int32> backoff_history(history);
      while (backoff_history.size() > 1) {
        backoff_history.erase(backoff_history.begin());
        const HistoryState *backoff_state = GetHistoryState(backoff_history,
                                                            false);
        backoff_states.push_back(backoff_state);
      }
    }

    std::string history_str;
    { // This block will set history_str to the sequence of history words for
      // this history state, separated by space; e.g. "on Tuesdays".
      std::ostringstream history_os;
      for (size_t i = 0; i < history.size(); i++) {
        std::string printed_word = symbols.Find(history[i]);
        KALDI_ASSERT(printed_word != "" && "mismatched symbol table?");
        history_os << printed_word;
        if (i + 1 < history.size()) history_os << ' ';
      }
      history_str = history_os.str();
    }
    std::vector<Count>::const_iterator count_iter = state->counts.begin(),
        count_end = state->counts.end();
    BaseFloat total_count = state->total_count,
        backoff_count = state->backoff_count;
    for (; count_iter != count_end; ++count_iter) {
      const Count &count = *count_iter;
      std::string printed_word = symbols.Find(count.word);
      KALDI_ASSERT(printed_word != "" && "mismatched symbol table?");
      BaseFloat word_prob =
          (count.count + backoff_count * GetProbForWord(count.word,
                                                        backoff_states)) /
          total_count,
          backoff_prob = BackoffProb(history, count.word);
      os << log10(word_prob) << '\t' << history_str << ' ' << printed_word;
      if (backoff_prob != 0.0)  os << '\t' << log10(backoff_prob) << '\n';
      else os << '\n';
    }
  }
}


void SamplingLmEstimator::PrintAsArpa(std::ostream &os,
                                      const fst::SymbolTable &symbols) {
  os << std::fixed << std::setprecision(3);  // print log-probs as -a.bcd
  os << "\\data\\\n";
  for (int32 o = 1; o <= config_.ngram_order; o++)
    os << "ngram " << o << "=" << NumNgrams(o) << "\n";

  for (int32 o = 1; o <= config_.ngram_order; o++) {
    os << '\n' << '\\' << o << "-grams:\n";
    if (o == 1) PrintNgramsUnigram(os, symbols);
    else PrintNgramsAboveUnigram(os, o, symbols);
  }
  os << "\n\\end\\\n";
}


}  // namespace rnnlm
}  // namespace kaldi

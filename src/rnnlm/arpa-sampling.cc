// arpa-sampling.cc

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

#include "rnnlm/arpa-sampling.h"

namespace kaldi {

// This function reads in each ngram line from an ARPA file
void ArpaSampling::ConsumeNGram(const NGram& ngram) {
  int32 cur_order = ngram.words.size(),
      max_order = Order();
  int32 word = ngram.words.back();  // word is the last word in a ngram term
  KALDI_ASSERT(cur_order > 0 && word > 0);

  if (cur_order == 1) {
    // unigram
    if (unigram_probs_.size() <= static_cast<size_t>(word))
      unigram_probs_.resize(static_cast<size_t>(word + 1), 0.0);
    KALDI_ASSERT(unigram_probs_[word] == 0.0);  // or repeated unigram.
    unigram_probs_[word] = Exp(ngram.logprob);
    if (ngram.backoff != 0.0)
      higher_order_probs_[cur_order - 1][ngram.words].backoff_prob =
        Exp(ngram.backoff);
  } else {
    HistType history(ngram.words.begin(), ngram.words.end() - 1);
    // Note: we'll later on change the probability, subtracting the
    // part that is due to backoff.  This change of format is
    // convenient for our application.
    higher_order_probs_[cur_order - 2][history].word_to_prob[word] =
        Exp(ngram.logprob);
    if (ngram.backoff != 0.0) {
      KALDI_ASSERT(cur_order != max_order);
      higher_order_probs_[cur_order - 1][ngram.words].backoff_prob =
          Exp(ngram.backoff);
    }
  }
}

void ArpaSampling::HeaderAvailable() {
  unigram_probs_.reserve(NgramCounts()[0] + 100);
  // e.g. for a trigram LM we store bigram and trigram
  // history states in probs_, while unigram_probs_ stores
  // the unigram probabilities.
  int32 ngram_order = NgramCounts().size();
  higher_order_probs_.resize(ngram_order - 1);
}

BaseFloat ArpaSampling::GetProbWithBackoff(
    const std::vector<int32> &history,
    const HistoryState *state,
    int32 word) const {
  if (state == NULL) {
    int32 order = history.size() + 1;
    if (order == 1) {
      KALDI_ASSERT(static_cast<size_t>(word) < unigram_probs_.size());
      return unigram_probs_[word];
    } else {
      std::unordered_map<HistType, HistoryState, VectorHasher<int32> >::const_iterator
          hist_iter = higher_order_probs_[order - 2].find(history);
      KALDI_ASSERT(hist_iter != higher_order_probs_[order - 2].end());
      // it's not optimally efficient to recurse here, but this is on a code
      // path that will rarely be taken in practice.
      return GetProbWithBackoff(history, &(hist_iter->second), word);
    }
  } else {
    std::unordered_map<int32, BaseFloat>::const_iterator iter =
        state->word_to_prob.find(word);
    if (iter == state->word_to_prob.end()) {
      std::vector<int32> backoff_history(history.begin() + 1,
                                         history.end());
      return state->backoff_prob *
          GetProbWithBackoff(backoff_history, NULL, word);
    } else {
      return iter->second;
    }
  }
}


void ArpaSampling::ReadComplete() {
  int32 max_order = Order();
  for (int32 order = max_order; order >= 2; order--) {
    std::unordered_map<HistType, HistoryState, VectorHasher<int32> >
        &this_map = higher_order_probs_[order - 2];
    std::unordered_map<HistType, HistoryState,
        VectorHasher<int32> >::iterator
        hist_iter = this_map.begin(), hist_end = this_map.end();
    for (; hist_iter != hist_end; ++hist_iter) {
      const HistType &history = hist_iter->first;
      HistoryState &history_state = hist_iter->second;
      BaseFloat backoff_prob = history_state.backoff_prob;
      HistoryState *backoff_state;
      HistType backoff_history(history.begin() + 1, history.end());
      if (order == 2) backoff_state = NULL;  // unigram has different format.
      else backoff_state = &(higher_order_probs_[order - 3][backoff_history]);

      std::unordered_map<int32, BaseFloat>::iterator
          word_iter = history_state.word_to_prob.begin(),
          word_end = history_state.word_to_prob.end();
      double total_prob_after_subtracting = 0.0;
      for (; word_iter != word_end; ++word_iter) {
        int32 word = word_iter->first;
        BaseFloat prob = word_iter->second;
        // OK, we want to subtract the backoff part.
        BaseFloat backoff_part_of_prob = backoff_prob *
            GetProbWithBackoff(backoff_history, backoff_state, word);
        if (backoff_part_of_prob > 1.01 * prob) {
          KALDI_WARN << "Backoff part of prob is larger than prob itself: "
                     << backoff_part_of_prob << " > " << prob
                     << ".  This may mean your language model was not "
                     << "Kneser-Ney 'with addition'.  We advise to use "
                     << "Kneser-Ney with addition or some other type of "
                     << "LM 'with addition'.";
        }
        // OK, this could now be negative.  This shouldn't matter
        BaseFloat new_prob = prob - backoff_part_of_prob;
        word_iter->second = new_prob;
        total_prob_after_subtracting += new_prob;
      }
      BaseFloat new_total = total_prob_after_subtracting + backoff_prob;
      if (fabs(new_total - 1.0) > 0.01)
        KALDI_WARN << "Expected LM-state to sum to one, got "
                   << new_total;
    }
  }
}

void ArpaSampling::AddBackoffToHistoryStates(
    const WeightedHistType &histories,
    WeightedHistType *histories_closure,
    BaseFloat *total_weight_out,
    BaseFloat *unigram_weight_out) const {
  // the implementation of this function is not as efficient as it could be,
  // but it should not dominate.
  std::vector<std::pair<HistType, BaseFloat> >::const_iterator
      histories_iter = histories.begin(), histories_end = histories.end();
  int32 max_order = Order();
  std::unordered_map<HistType, BaseFloat,
      VectorHasher<int32> > hist_to_weight_map;
  double total_weight = 0.0, total_unigram_weight = 0.0;
  for (; histories_iter != histories_end; ++histories_iter) {
    std::vector<int32> history = histories_iter->first;
    int32 cur_hist_len = history.size();
    BaseFloat weight = histories_iter->second;
    total_weight += weight;
    KALDI_ASSERT(history.size() <= max_order - 1 && weight > 0);

    // back off until the history exists or until we reached the unigram state.
    while (cur_hist_len > 0 &&
           higher_order_probs_[cur_hist_len - 1].count(history) == 0) {
      history.erase(history.begin(), history.begin() + 1);
      cur_hist_len--;
    }
    // OK, the history-state exists.
    while (cur_hist_len > 0) {
      hist_to_weight_map[history] += weight;
      std::unordered_map<HistType, HistoryState, VectorHasher<int32> >::const_iterator
          iter = higher_order_probs_[cur_hist_len - 1].find(history);
      KALDI_ASSERT(iter != higher_order_probs_[cur_hist_len - 1].end());
      weight *= iter->second.backoff_prob;
      history.erase(history.begin(), history.begin() + 1);
      cur_hist_len--;
    }
    // at this point, 'history' is empty and 'weight' is the unigram
    // backoff weight for this history state.
    total_unigram_weight += weight;
  }
  histories_closure->clear();
  histories_closure->resize(hist_to_weight_map.size());
  std::unordered_map<HistType, BaseFloat, VectorHasher<int32> >::iterator
      hist_to_weight_iter = hist_to_weight_map.begin(),
      hist_to_weight_end = hist_to_weight_map.end();
  size_t pos = 0;
  for (; hist_to_weight_iter != hist_to_weight_end; ++hist_to_weight_iter) {
    (*histories_closure)[pos].first = hist_to_weight_iter->first;
    (*histories_closure)[pos].second = hist_to_weight_iter->second;
    pos++;
  }
  *total_weight_out = total_weight;
  *unigram_weight_out = total_unigram_weight;
  KALDI_ASSERT(pos == hist_to_weight_map.size());
}


BaseFloat ArpaSampling::GetDistribution(
    const WeightedHistType &histories,
    std::vector<std::pair<int32, BaseFloat> > *non_unigram_probs_out) const {
  std::unordered_map<int32, BaseFloat> non_unigram_probs_temp;
  // Call the other version of GetDistribution().
  BaseFloat ans = GetDistribution(histories, &non_unigram_probs_temp);
  non_unigram_probs_out->clear();
  non_unigram_probs_out->reserve(non_unigram_probs_temp.size());
  non_unigram_probs_out->insert(non_unigram_probs_out->end(),
                                non_unigram_probs_temp.begin(),
                                non_unigram_probs_temp.end());
  std::sort(non_unigram_probs_out->begin(),
            non_unigram_probs_out->end());
  return ans;
}

BaseFloat ArpaSampling::GetDistribution(
    const WeightedHistType &histories,
    std::unordered_map<int32, BaseFloat> *non_unigram_probs) const {
  WeightedHistType histories_closure;
  BaseFloat total_weight, total_unigram_weight;
  AddBackoffToHistoryStates(histories, &histories_closure,
                            &total_weight, &total_unigram_weight);
  non_unigram_probs->clear();
  double total_weight_check = total_unigram_weight;
  WeightedHistType::const_iterator iter = histories_closure.begin(),
      end = histories_closure.end();
  for (; iter != end; ++iter) {
    const HistType &history = iter->first;
    BaseFloat hist_weight = iter->second;
    int32 order = history.size() + 1;
    KALDI_ASSERT(order > 1);  // unigram history is not included at this point.
    std::unordered_map<HistType, HistoryState,
        VectorHasher<int32> >::const_iterator it_hist =
           higher_order_probs_[order - 2].find(history);
    KALDI_ASSERT(it_hist != higher_order_probs_[order - 2].end());
    std::unordered_map<int32, BaseFloat>::const_iterator
        word_iter = it_hist->second.word_to_prob.begin(),
        word_end = it_hist->second.word_to_prob.end();
    for (; word_iter != word_end; ++word_iter) {
      int32 word = word_iter->first;
      BaseFloat prob = word_iter->second;
      // note: if 'word' was not in the map, it's as if it were zero, for C++
      // version >= C++11; search for unordered_map value initialization for
      // explanation
      (*non_unigram_probs)[word] += prob * hist_weight;
      total_weight_check += prob * hist_weight;
    }
  }
  // Check that 'total_weight' and 'total_weight_check' are
  // the same.  'total_weight' is the total of the of the .second
  // member of the input 'histories', and 'total_weight_check' is the
  // total weight of 'non_unigrm_probs' plus 'total_unigram_weight'.
  // Essentially this is a check that the distribution given
  // by the ARPA file (and as processed by us) sums to one for each
  // history state.  If this check fails, it could either be
  // a problem with this code, or an issue with the software that
  // created the ARPA file.
  if (fabs(total_weight - total_weight_check) >
      0.01 * total_weight) {
    static int32 num_times_warned = 0;
    if (num_times_warned < 10) {
      KALDI_WARN << "Total weight does not have expected value (problem in "
          "your ARPA file, or this code).  Won't warn >10 times.";
      num_times_warned++;
    }
  }
  KALDI_ASSERT(total_unigram_weight > 0.0);
  return total_unigram_weight;
}


}  // namespace kaldi

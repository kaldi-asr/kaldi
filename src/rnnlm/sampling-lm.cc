// sampling-lm.cc

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

#include "rnnlm/sampling-lm.h"

namespace kaldi {
namespace rnnlm {

// This function reads in each ngram line from an ARPA file
void SamplingLm::ConsumeNGram(const NGram& ngram) {
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
    // ngram.logprob has already been converted to log-base e at
    // this point.
    higher_order_probs_[cur_order - 2][history].words_and_probs.push_back(
        std::pair<int32, BaseFloat>(word, Exp(ngram.logprob)));
    if (ngram.backoff != 0.0) {
      KALDI_ASSERT(cur_order != max_order);
      higher_order_probs_[cur_order - 1][ngram.words].backoff_prob =
          Exp(ngram.backoff);
    }
  }
}

void SamplingLm::HeaderAvailable() {
  unigram_probs_.reserve(NgramCounts()[0] + 100);
  // e.g. for a trigram LM we store bigram and trigram
  // history states in probs_, while unigram_probs_ stores
  // the unigram probabilities.
  int32 ngram_order = NgramCounts().size();
  higher_order_probs_.resize(ngram_order - 1);
}

BaseFloat SamplingLm::GetProbWithBackoff(
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
    std::pair<int32, BaseFloat> p(word, 0.0);
    std::vector<std::pair<int32, BaseFloat> >::const_iterator iter =
        std::lower_bound(state->words_and_probs.begin(),
                         state->words_and_probs.end(), p);
    if (iter != state->words_and_probs.end() && iter->first == word) {
      // the probability for this word was given in this history state.  (note:
      // we assume that at the time this function is called, the entire
      // probability is present here, as it is in the ARPA format LM.  See
      // documentation for this function for more explanation.
      return iter->second;
    } else {
      // we have to back off.
      std::vector<int32> backoff_history(history.begin() + 1,
                                         history.end());
      return state->backoff_prob *
          GetProbWithBackoff(backoff_history, NULL, word);
    }
  }
}

void SamplingLm::EnsureHistoryStatesSorted() {
  for (size_t i = 0; i < higher_order_probs_.size(); i++) {
    std::unordered_map<HistType, HistoryState, VectorHasher<int32> >::iterator
        iter = higher_order_probs_[i].begin(),
        end = higher_order_probs_[i].end();
    for (; iter != end; ++iter)
      std::sort(iter->second.words_and_probs.begin(),
                iter->second.words_and_probs.end());
  }
}

void SamplingLm::ReadComplete() {
  EnsureHistoryStatesSorted();
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

      std::vector<std::pair<int32, BaseFloat> >::iterator
          word_iter = history_state.words_and_probs.begin(),
          word_end = history_state.words_and_probs.end();
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

void SamplingLm::AddBackoffToHistoryStates(
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
      history.erase(history.begin());
      cur_hist_len--;
    }
    // OK, the history-state exists.
    while (cur_hist_len > 0) {
      hist_to_weight_map[history] += weight;
      std::unordered_map<HistType, HistoryState, VectorHasher<int32> >::const_iterator
          iter = higher_order_probs_[cur_hist_len - 1].find(history);
      KALDI_ASSERT(iter != higher_order_probs_[cur_hist_len - 1].end());
      weight *= iter->second.backoff_prob;
      history.erase(history.begin());
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


BaseFloat SamplingLm::GetDistribution(
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

BaseFloat SamplingLm::GetDistribution(
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
    std::vector<std::pair<int32, BaseFloat> >::const_iterator
        word_iter = it_hist->second.words_and_probs.begin(),
        word_end = it_hist->second.words_and_probs.end();
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

SamplingLm::SamplingLm(const SamplingLmEstimator &estimator):
    ArpaFileParser(ArpaParseOptions(), NULL),
    unigram_probs_(estimator.unigram_probs_),
    higher_order_probs_(estimator.history_states_.size() - 1) {
  for (int32 o = 2;
       o <= static_cast<int32>(estimator.history_states_.size()); o++) {
    higher_order_probs_[o-2].reserve(estimator.history_states_[o-1].size());
    unordered_map<std::vector<int32>, SamplingLmEstimator::HistoryState*,
                  VectorHasher<int32> >::const_iterator
        iter = estimator.history_states_[o-1].begin(),
        end =  estimator.history_states_[o-1].end();
    for (; iter != end; ++iter) {
      const std::vector<int32> &history = iter->first;
      const SamplingLmEstimator::HistoryState &src_state = *(iter->second);
      // the next statement adds a history state to the map.
      HistoryState &dest_state = higher_order_probs_[o-2][history];
      BaseFloat inv_total_count = BaseFloat(1.0) / src_state.total_count;
      dest_state.backoff_prob = src_state.backoff_count * inv_total_count;
      dest_state.words_and_probs.resize(src_state.counts.size());
      std::vector<SamplingLmEstimator::Count>::const_iterator
          src_iter = src_state.counts.begin(),
          src_end = src_state.counts.end();
      std::vector<std::pair<int32, BaseFloat> >::iterator
          dest_iter = dest_state.words_and_probs.begin();
      for (; src_iter != src_end; ++src_iter, ++dest_iter) {
        dest_iter->first = src_iter->word;
        dest_iter->second = inv_total_count * src_iter->count;
      }
    }
  }
}

void SamplingLm::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SamplingLm>");
  WriteToken(os, binary, "<Order>");
  int32 order = higher_order_probs_.size() + 1;
  WriteBasicType(os, binary, order);
  WriteToken(os, binary, "<VocabSize>");
  int32 vocab_size = unigram_probs_.size();
  WriteBasicType(os, binary, vocab_size);
  KALDI_ASSERT(!unigram_probs_.empty());
  // we have read and write functions in class Vector, so use that.
  SubVector<BaseFloat> probs(const_cast<BaseFloat*>(&(unigram_probs_[0])),
                             static_cast<int32>(unigram_probs_.size()));
  probs.Write(os, binary);
  for (int32 o = 2; o <= order; o++) {
    WriteToken(os, binary, "<StatesOfOrder>");
    WriteBasicType(os, binary, o);
    WriteToken(os, binary, "<NumStates>");
    int32 num_states = higher_order_probs_[o-2].size();
    WriteBasicType(os, binary, num_states);

    unordered_map<std::vector<int32>, HistoryState,
                  VectorHasher<int32> >::const_iterator
        iter = higher_order_probs_[o-2].begin(),
        end = higher_order_probs_[o-2].end();
    for (; iter != end; ++iter ){
      const std::vector<int32> &history = iter->first;
      const HistoryState &state = iter->second;
      WriteIntegerVector(os, binary, history);
      WriteBasicType(os, binary, state.backoff_prob);
      int32 num_words = state.words_and_probs.size();
      WriteBasicType(os, binary, num_words);
      for (int32 i = 0; i < num_words; i++) {
        WriteBasicType(os, binary, state.words_and_probs[i].first);
        WriteBasicType(os, binary, state.words_and_probs[i].second);
      }
      if (!binary) os << std::endl;
    }
  }
  WriteToken(os, binary, "</SamplingLm>");
}


void SamplingLm::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<SamplingLm>");
  ExpectToken(is, binary, "<Order>");
  int32 order;
  ReadBasicType(is, binary, &order);
  KALDI_ASSERT(order >= 1 && order < 100);
  higher_order_probs_.resize(order - 1);
  ExpectToken(is, binary, "<VocabSize>");
  int32 vocab_size;
  ReadBasicType(is, binary, &vocab_size);
  unigram_probs_.resize(vocab_size);
  // we have read and write functions in class Vector, so use that.
  SubVector<BaseFloat> probs(&(unigram_probs_[0]), vocab_size);
  probs.Read(is, binary);
  for (int32 o = 2; o <= order; o++) {
    ExpectToken(is, binary, "<StatesOfOrder>");
    int32 o2;
    ReadBasicType(is, binary, &o2);
    KALDI_ASSERT(o2 == o);
    int32 num_states;
    ExpectToken(is, binary, "<NumStates>");
    ReadBasicType(is, binary, &num_states);
    higher_order_probs_[o-2].reserve(num_states);
    for  (int32 s = 0; s < num_states; s++) {
      std::vector<int32> history;
      ReadIntegerVector(is, binary, &history);
      HistoryState &state = higher_order_probs_[o-2][history];
      ReadBasicType(is, binary, &(state.backoff_prob));
      int32 num_words;
      ReadBasicType(is, binary, &num_words);
      KALDI_ASSERT(num_words >= 0);
      state.words_and_probs.resize(num_words);
      for (int32 i = 0; i < num_words; i++) {
        ReadBasicType(is, binary, &(state.words_and_probs[i].first));
        ReadBasicType(is, binary, &(state.words_and_probs[i].second));
      }
    }
  }
  ExpectToken(is, binary, "</SamplingLm>");
}

// TODO: delete if unused.
void SamplingLm::Swap(SamplingLm *other) {
  unigram_probs_.swap(other->unigram_probs_);
  higher_order_probs_.swap(other->higher_order_probs_);
}

}  // namespace rnnlm
}  // namespace kaldi

// arpa-sampling.cc

#include "rnnlm/arpa-sampling.h"

namespace kaldi {

// This function reads in each ngram line from an ARPA file
void ArpaSampling::ConsumeNGram(const NGram& ngram) {
  int32 cur_order = ngram.words.size();
  int32 word = ngram.words.back();  // word is the last word in a ngram term
  HistType history(ngram.words.begin(), ngram.words.end() - 1);
  KALDI_ASSERT(history.size() == cur_order - 1);

  // log probability of a ngram term (natural log)
  BaseFloat log_prob = ngram.logprob;
  // backoff log probability of a ngram term (natural log)
  BaseFloat backoff_weight = ngram.backoff;
  std::pair <BaseFloat, BaseFloat> probs_pair;
  probs_pair = std::make_pair(log_prob, backoff_weight);
  probs_[cur_order - 1][history].insert({word, probs_pair});

  // total number of words equals the maximum symbol id plus one
  if (num_words_ < word + 1) {
    num_words_ = word + 1;
  }
}

void ArpaSampling::HeaderAvailable() {
  ngram_counts_ = NgramCounts();
  ngram_order_ = NgramCounts().size();
  probs_.resize(ngram_order_);
}

BaseFloat ArpaSampling::GetLogprob(int32 word, const HistType &history) const {
  BaseFloat prob = 0.0;
  KALDI_ASSERT(history.size() < ngram_order_);
  // Ngram order should be history size plus one. Since here the ngram order is
  // zero indexed, it equals history size.
  int32 order = history.size();
  NgramType::const_iterator it = probs_[order].find(history);
  if (it != probs_[order].end()) {
    MapType::const_iterator it2 = it->second.find(word);
    if (it2 != it->second.end()) {
      prob += it2->second.first;
      return prob;
    }
  }
  // If 1) history or 2) word given that history doesn't exist,
  // backoff to the previous order
  order = order - 1;
  if (order >= 0) {
    HistType h(history.begin() + 1, history.end());
    prob += GetLogprob(word, h);
    HistType h_new(history.begin(), history.end() - 1);
    prob += GetBackoffLogprob(history.back(), h_new);
  }
  return prob;
}

BaseFloat ArpaSampling::GetBackoffLogprob(int32 word,
                                          const HistType &history) const {
  BaseFloat bow = 0.0;
  KALDI_ASSERT(history.size() >= 0);
  int32 order = history.size();
  NgramType::const_iterator it = probs_[order].find(history);
  if (it != probs_[order].end()) {
    MapType::const_iterator it2 = it->second.find(word);
    if (it2 != it->second.end()) {
      bow = it2->second.second;
    }
  }
  return bow;
}

void ArpaSampling::GetUnigramDistribution(
                   std::vector<BaseFloat> *unigram_probs) const {
  unigram_probs->clear();
  unigram_probs->resize(num_words_);
  // zero is reserved for epislon.
  for (int32 i = 0; i < num_words_; ++i) {
    HistType h;  // empty history
    NgramType::const_iterator it = probs_[0].find(h);
    if (it != probs_[0].end()) {
      MapType::const_iterator it2 = it->second.find(i);
      if (it2 != it->second.end()) {
        (*unigram_probs)[i] = Exp(it2->second.first);
      }
    }
  }
}

BaseFloat ArpaSampling::GetDistribution(const WeightedHistType &histories,
                           std::unordered_map<int32, BaseFloat> *pdf_w) const {
  pdf_w->clear();
  BaseFloat unigram_weight = 0.0;
  BaseFloat total_weights = 0;
  BaseFloat prob = 0;
  WeightedHistType::const_iterator it = histories.begin();
  for (; it != histories.end(); ++it) {
    const HistType &h = it->first;
    int32 order = h.size();
    NgramType::const_iterator it_hist = probs_[order].find(h);
    if (it_hist != probs_[order].end()) {
      for (MapType::const_iterator it_word = it_hist->second.begin();
          it_word != it_hist->second.end(); ++it_word) {
        int32 word = it_word->first;
        if (order > 0) {
          HistType h1(h.begin(), h.end() - 1);
          HistType h2(h.begin() + 1, h.end());
          prob = it->second * (Exp(it_word->second.first) -
                 Exp(GetBackoffLogprob(h.back(), h1) + GetLogprob(word, h2)));
          (*pdf_w)[word] += prob;
        }
      }
    }
    total_weights += it->second;
    BaseFloat backoff = 0.0;
    HistType h1(h.begin(), h.end() - 1);
    backoff = GetBackoffLogprob(h.back(), h1);
    unigram_weight += it->second * Exp(backoff);
  }
  // If total input weights is zero, then each input weight is zero. In this
  // case, the output is the unigram distribution and unigram weight should be 1
  if (total_weights == 0) {
    unigram_weight = 1.0;
  }
  return unigram_weight;
}
}  // end of kaldi

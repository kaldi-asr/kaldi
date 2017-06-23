// arpa-sampling.cc

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <string>

#include "arpa-sampling.h"

namespace kaldi {

// This function reads in each ngram line from an ARPA file
void ArpaSampling::ConsumeNGram(const NGram& ngram) {
  int32 cur_order = ngram.words.size();
  int32 word = ngram.words.back(); // word is the last word in a ngram term 
  HistType history(ngram.words.begin(), ngram.words.begin() + cur_order - 1);
  KALDI_ASSERT(history.size() == cur_order - 1);

  // log probability of a ngram term (natural log)
  BaseFloat log_prob = ngram.logprob;
  // backoff log probability of a ngram term (natural log)
  BaseFloat backoff_weight = ngram.backoff;
  std::pair <BaseFloat, BaseFloat> probs_pair;
  probs_pair = std::make_pair(log_prob, backoff_weight);
  probs_[cur_order - 1][history].insert({word, probs_pair});
 
  // get the total number of words from an ARPA file
  if (cur_order == 1) {
    num_words_++;
  }
}

void ArpaSampling::HeaderAvailable() {
  ngram_counts_ = NgramCounts();
  ngram_order_ = NgramCounts().size(); 
  probs_.resize(ngram_order_);
}

BaseFloat ArpaSampling::GetProb(int32 order, int32 word, const HistType &history) {
  BaseFloat prob = 0.0;
  NgramType::const_iterator it = probs_[order - 1].find(history);
  if (it != probs_[order - 1].end() &&
      probs_[order-1][history].find(word) != probs_[order-1][history].end()) {
    prob += probs_[order-1][history][word].first;
  } else { // backoff to the previous order
    order--;
    if (order >= 1) {
      HistType::const_iterator first = history.begin() + 1;
      HistType::const_iterator last = history.end();
      HistType h(first, last);
      prob += GetProb(order, word, h);
      int32 word_new = history.back();
      HistType::const_iterator last_new = history.end() - 1;
      HistType h_new(history.begin(), last_new);
      prob += GetBackoffWeight(order, word_new, h_new);
    }
  }
  return prob;
}

BaseFloat ArpaSampling::GetBackoffWeight(int32 order, int32 word, 
    const HistType &history) {
  BaseFloat bow = 0.0;
  KALDI_ASSERT(order >= 1);
  NgramType::const_iterator it = probs_[order - 1].find(history);
  if (it != probs_[order - 1].end()) {
    WordToProbsMap::const_iterator it2 = probs_[order - 1][history].find(word);
    if (it2 != probs_[order - 1][history].end()) {
      bow = it2->second.second;
    }
  }
  return bow;
}

void ArpaSampling::GetUnigramDistribution(std::vector<BaseFloat> *unigram_probs) {
  (*unigram_probs).clear();
  (*unigram_probs).resize(num_words_ + 1);
  for (int32 i = 0; i < num_words_ + 1; ++i) {
    HistType h; // empty history
    WordToProbsMap::const_iterator it = probs_[0][h].find(i);
    if (it != probs_[0][h].end()) {
      (*unigram_probs)[i] = Exp(it->second.first);
    }    
  }
}

void ArpaSampling::ComputeHistoriesWeights(
    const std::vector<std::pair<HistType, BaseFloat> > &histories,
    HistWeightsType *hists_weights) {
  (*hists_weights).clear();
  for (std::vector<std::pair<HistType, BaseFloat> >::const_iterator
      it = histories.begin(); it != histories.end(); ++it) {
    HistType history((*it).first);
    KALDI_ASSERT(history.size() <= ngram_order_);
    (*hists_weights)[history] += (*it).second;
  }
}

BaseFloat ArpaSampling::GetDistribution(const std::vector<std::pair<HistType, 
    BaseFloat> > &histories, std::unordered_map<int32, BaseFloat> *pdf_w) {
  pdf_w->clear();
  HistWeightsType hists_weights;
  ComputeHistoriesWeights(histories, &hists_weights); 
  BaseFloat unigram_weight = 0.0;
  BaseFloat total_weights = 0;
  BaseFloat probsum = 0;
  BaseFloat prob = 0;
  for (HistWeightsType::const_iterator it = hists_weights.begin(); 
      it != hists_weights.end(); ++it) {
    HistType h(it->first);
    int32 order = h.size();
    NgramType::const_iterator it_hist = probs_[order].find(h);
    if (it_hist != probs_[order].end()) {
      for(WordToProbsMap::const_iterator it_word = probs_[order][h].begin(); 
          it_word != probs_[order][h].end(); ++it_word) {
        int32 word = it_word->first;
        if (order > 0) {
          HistType::iterator last = h.end() - 1;
          HistType::iterator first = h.begin() + 1;
          HistType h1(h.begin(), last);
          HistType h2(first, h.end());
          prob = it->second * (Exp(probs_[order][h][word].first) - 
                 Exp(GetBackoffWeight(order, h.back(), h1) + GetProb(order, word, h2)));
          (*pdf_w)[word] += prob;
          probsum += prob;
        }
      }
    }
    total_weights += it->second;
    BaseFloat backoff = 0.0;
    HistType::iterator last = h.end() - 1;
    HistType h1(h.begin(), last);
    backoff = GetBackoffWeight(order, h.back(), h1);
    unigram_weight += it->second * Exp(backoff);
  }
  // If total input weights is zero, then each input weight is zero. In this case
  // the output is the unigram distribution and unigram weight should be 1 
  if (total_weights == 0) {
    unigram_weight = 1.0;
  }
  return unigram_weight;
}

}  // end of kaldi

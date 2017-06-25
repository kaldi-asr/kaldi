// arpa_sampling.h

// Copyright 2017  Ke Li

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

class ArpaSampling : public ArpaFileParser {
 public:
  friend class ArpaSamplingTest;

  // HistType represents a history
  typedef std::vector<int32> HistType;
  // WeightedHistType represents a vector of pairs of a history and
  // its associated weight
  typedef std::vector<std::pair<HistType, BaseFloat> > WeightedHistType;

  // ARPA LM file is read by function "void Read(std::istream &is, bool binary)"
  // in ArpaFileParser. Only text mode is supported.
  ArpaSampling(ArpaParseOptions options, fst::SymbolTable* symbols)
     : ArpaFileParser(options, symbols) {
       ngram_order_ = 0;
       num_words_ = 0;
  }

  // This function computes the unigram distribution of all words represented
  // by integers from 0 to maximum symbol id
  // Note: there can be gaps of integers for words in the ARPA LM, we set the
  // probabilities of words that are not in the ARPA LM to be 0.0, e.g.,
  // symbol id 0 which represents epsilon has probability 0.0
  void GetUnigramDistribution(std::vector<BaseFloat> *unigram_probs) const;

  // This function accepts a list of histories with associated weights and outputs
  // 1) the non_unigram_probs (maps the higher-than-unigram words to their
  // corresponding probabilities given the list of histories) and
  // 2) a scalar unigram_weight = sum of history_weight * backoff_weight of
  // that history
  BaseFloat GetDistribution(const WeightedHistType &histories,
      std::unordered_map<int32, BaseFloat> *non_unigram_probs) const;

 protected:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable();
  virtual void ConsumeNGram(const NGram& ngram);
  virtual void ReadComplete() {}

 private:
  // MapType represents the words and their probabilities given
  // an arbitrary history
  typedef std::unordered_map<int32, std::pair<BaseFloat, BaseFloat> > MapType;

  // NgramType represents the map between a history and the map of
  // existing words and the associated probabilities given this history
  typedef std::unordered_map<HistType, MapType, VectorHasher<int32> > NgramType;

  // This function returns the log probability of a ngram, [history word],
  // from the read-in ARPA file if it exists. If the ngram does not exist,
  // it backs off to a lower order until the update ngram is found.
  BaseFloat GetLogprob(int32 word, const HistType& history) const;

  // This function returns the back-off log probability of a ngram
  // from the read-in ARPA file
  BaseFloat GetBackoffLogprob(int32 word, const HistType& history) const;

  // Highest N-gram order of the read-in ARPA LM
  int32 ngram_order_;

  // Highest symbol id present in the read-in ARPA LM puls one
  int32 num_words_;

  // Number of ngrams for each ngram order, indexed by ngram order minus one
  std::vector<int32> ngram_counts_;

  // Log probabilities of NgramType
  std::vector<NgramType> probs_;
};
}  // end of namespace kaldi
#endif  // KALDI_RNNLM_ARPA_SAMPLING_H_

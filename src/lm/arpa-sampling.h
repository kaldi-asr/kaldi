// arpa_sampling.h

// Copyright     2016  Ke Li

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

#ifndef ARPA_SAMPLING_H_
#define ARPA_SAMPLING_H_

#include <algorithm>
#include <map>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "fst/fstlib.h"
#include "util/common-utils.h"
#include "arpa-file-parser.h"

namespace kaldi {

class ArpaSampling : public ArpaFileParser {
 public:
  friend class ArpaSamplingTest;

  // HistType represents a history
  typedef std::vector<int32> HistType;
  // WordToProbsMap represents the words and their probabilities given
  // an arbitrary history
  typedef std::unordered_map<int32, std::pair<BaseFloat, BaseFloat> > WordToProbsMap; 
  // NgramType represents the map between a history and the map of existing words
  // and their probabilities given this history
  typedef std::unordered_map<HistType, WordToProbsMap, VectorHasher<int32> > NgramType;
  // HistWeightsType represents the map between histories (with weights) and
  // their computed weights
  typedef std::unordered_map<HistType, BaseFloat, VectorHasher<int32> > HistWeightsType;

  // constructor
  ArpaSampling(ArpaParseOptions options, fst::SymbolTable* symbols)
     : ArpaFileParser(options, symbols) { 
       ngram_order_ = 0;
       num_words_ = 0;
       bos_symbol_ = "<s>";
       eos_symbol_ = "</s>";
       unk_symbol_ = "<unk>";
  }
  
  // This function computes the unigram distribution of all vocab words which are
  // represented by integers from 1 to num_words (0 is reserved for epsilon)
  // Probabilities of integers are not represented by any valid word is 0, e.g.
  // 0 has probability 0.0
  void GetUnigramDistribution(std::vector<BaseFloat> *unigram_probs);

  // This function reads in a list of histories with input weights and returns
  // 1) the higher-than-unigram words and their corresponding probabilities 
  // given the list of histories and 
  // 2) a scalar alpha = sum of w_i * product of all backoff weights back to unigram 
  BaseFloat GetOutputWordsAndAlpha(const std::vector<std::pair<HistType, BaseFloat> > 
      &histories, std::unordered_map<int32, BaseFloat> *non_unigram_probs);
  
 protected:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable(); 
  virtual void ConsumeNGram(const NGram& ngram);
  virtual void ReadComplete() {}

 private:
  // This function returns the log probability of a LM state, [history word],
  // from the read-in ARPA file if it exists. If the LM state does not exist,
  // it backs off to a lower order until the updated LM state found.
  // Note: a LM state is a ngram term in a ARPA language model
  BaseFloat GetProb(int32 order, int32 word, const HistType& history);

  // This function returns the back-off log probability of a LM state
  // from the read-in ARPA file 
  BaseFloat GetBackoffWeight(int32 order, int32 word, const HistType& history);

  // This function returns the computed weights of histories with input weights
  void ComputeHistoriesWeights(const std::vector<std::pair<HistType, BaseFloat> > 
      &histories, HistWeightsType *hists_weights);

  // highest N-gram order of the read-in ARPA LM
  int32 ngram_order_;
  
  // total number of words of the read-in ARPA LM 
  int32 num_words_;

  // Begining of sentence symbol
  std::string bos_symbol_;

  // End of sentence symbol
  std::string eos_symbol_;

  // Unkown word symbol
  std::string unk_symbol_;

  // Counts of each LM state 
  std::vector<int32> ngram_counts_;

  // LM state probabilities
  std::vector<NgramType> probs_;

  // Histories' weights
  HistWeightsType hists_weights_;
  
};

}  // end of namespace kaldi

#endif  // ARPA_SAMPLING_H_

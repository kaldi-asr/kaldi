// rnnlm/sampling-lm-test.cc

// Copyright 2017  Ke Li
//           2017  Johns Hopkins University (author: Daniel Povey)

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

#include "rnnlm/sampling-lm.h"

namespace kaldi {
namespace rnnlm {

class SamplingLmTest {
 public:
  typedef SamplingLm::HistType HistType;
  typedef SamplingLm::WeightedHistType WeightedHistType;

  explicit SamplingLmTest(SamplingLm *arpa) {
    arpa_ = arpa;
  }
  // This function reads in a list of histories and their weights from a file
  // only text form is supported
  void ReadHistories(std::istream &is, bool binary,
      WeightedHistType *histories);

  void TestUnigramDistribution();

  void TestGetDistribution(const WeightedHistType &histories);
 private:
  // This SamplingLm object is used to get accesses to private and protected
  // members in SamplingLm class
  SamplingLm *arpa_;
};

void SamplingLmTest::ReadHistories(std::istream &is, bool binary,
    WeightedHistType *histories) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for ArpaFileParser";
  }
  const fst::SymbolTable* sym = arpa_->Symbols();
  std::string line;
  KALDI_LOG << "Start reading histories from file...";
  // int32 ngram_order = arpa_->ngram_order_;
  int32 ngram_order = arpa_->Order();
  while (getline(is, line)) {
    std::istringstream is(line);
    std::istream_iterator<std::string> begin(is), end;
    std::vector<std::string> tokens(begin, end);
    HistType history;
    int32 word;
    BaseFloat hist_weight = 0;
    for (int32 i = 0; i < tokens.size() - 1; ++i) {
      word = sym->Find(tokens[i]);
      if (word == -1) { // fst::kNoSymbol
        KALDI_ERR << "Found history contains word that is not in Arpa LM";
      }
      history.push_back(word);
    }
    HistType h1;
    if (history.size() >= ngram_order) {
      HistType h(history.end() - ngram_order + 1, history.end());
      h1 = h;
    }
    if (!ConvertStringToReal(tokens.back(), &hist_weight)) {
      KALDI_ERR << arpa_->LineReference() << ": invalid history weight '"
        << tokens.back() << "'";
    }
    KALDI_ASSERT(hist_weight >= 0);
    std::pair<HistType, BaseFloat> hist_pair;
    if (history.size() >= ngram_order) {
      hist_pair = std::make_pair(h1, hist_weight);
    } else {
      hist_pair = std::make_pair(history, hist_weight);
    }
    (*histories).push_back(hist_pair);
  }
  KALDI_LOG << "Successfully reading histories from file.";
}

void SamplingLmTest::TestUnigramDistribution() {
  std::vector<BaseFloat> unigram_probs;
  unigram_probs = arpa_->GetUnigramDistribution();
  // Check 0 (epsilon) has probability 0.0
  KALDI_ASSERT(unigram_probs[0] == 0.0);
  // Assert the sum of unigram probs of all words is 1.0
  BaseFloat probsum = 0.0;
  for (int32 i = 0; i < unigram_probs.size(); ++i) {
    probsum += unigram_probs[i];
  }
  KALDI_ASSERT(ApproxEqual(probsum, 1.0));
}

void SamplingLmTest::TestGetDistribution(const WeightedHistType &histories) {
  // get total input weights of histories
  BaseFloat total_weights = 0.0;
  WeightedHistType::const_iterator it = histories.begin();
  for (; it != histories.end(); ++it) {
    total_weights += it->second;
  }
  BaseFloat unigram_weight = 0.0;
  BaseFloat non_unigram_probsum = 0.0;
  std::vector<std::pair<int32, BaseFloat> > pdf;
  unigram_weight = arpa_->GetDistribution(histories, &pdf);
  for (int32 i = 0; i < pdf.size(); ++i) {
    non_unigram_probsum += pdf[i].second;
  }
  // assert unigram weight plus total non_unigram probs equals
  // the total input histories' weights
  KALDI_ASSERT(ApproxEqual(unigram_weight + non_unigram_probsum, total_weights));
}

}  // namespace rnnlm
}  // namespace kaldi

int main(int argc, char **argv) {
  using namespace kaldi;
  using namespace kaldi::rnnlm;

  const char *usage = "";
  ParseOptions po(usage);
  po.Read(argc, argv);
  std::string arpa_file = "test/0.1k_3gram_unpruned.arpa",
    history_file = "test/hists";

  ArpaParseOptions options;
  fst::SymbolTable symbols;

  enum {
    kEps = 0,
    kBos, kEos, kUnk
  };

  // Use spaces on special symbols, so we rather fail than read them by mistake.
  symbols.AddSymbol(" <eps>", kEps);
  options.bos_symbol = symbols.AddSymbol("<s>", kBos);
  options.eos_symbol = symbols.AddSymbol("</s>", kEos);
  options.unk_symbol = symbols.AddSymbol("<unk>", kUnk);
  options.oov_handling = ArpaParseOptions::kAddToSymbols;
  SamplingLm arpa(options, &symbols);

  bool binary;
  Input k1(arpa_file);
  arpa.Read(k1.Stream());

  SamplingLmTest mdl(&arpa);
  mdl.TestUnigramDistribution();

  Input k2(history_file, &binary);
  SamplingLmTest::WeightedHistType histories;
  mdl.ReadHistories(k2.Stream(), binary, &histories);
  mdl.TestGetDistribution(histories);
  KALDI_LOG << "Tests for SamplingLm class succeed.";
  return 0;
}

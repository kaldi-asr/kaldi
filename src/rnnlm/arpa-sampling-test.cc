// lm/arpa-sampling-test.cc

#include "rnnlm/arpa-sampling.h"

namespace kaldi {

class ArpaForSamplingTest {
 public:
  typedef ArpaForSampling::HistType HistType;
  typedef ArpaForSampling::MapType MapType;
  typedef ArpaForSampling::NgramType NgramType;
  typedef ArpaForSampling::WeightedHistType WeightedHistType;

  explicit ArpaForSamplingTest(ArpaForSampling *arpa) {
    arpa_ = arpa;
  }
  // This function reads in a list of histories and their weights from a file
  // only text form is supported
  void ReadHistories(std::istream &is, bool binary,
      WeightedHistType *histories);

  // This function tests the correctness of the read-in ARPA LM
  void TestReadingModel();

  // This function tests the generated unigram distribution
  void TestUnigramDistribution();

  // Test non-unigram words and alpha
  void TestGetDistribution(const WeightedHistType &histories);
 private:
  // This ArpaForSampling object is used to get access to private and protected
  // members in ArpaForSampling class
  ArpaForSampling *arpa_;
};

void ArpaForSamplingTest::ReadHistories(std::istream &is, bool binary,
    WeightedHistType *histories) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for ArpaFileParser";
  }
  const fst::SymbolTable* sym = arpa_->Symbols();
  std::string line;
  KALDI_LOG << "Start reading histories from file...";
  // int32 ngram_order = arpa_->ngram_order_;
  int32 ngram_order = arpa_->ngram_order_;
  while (getline(is, line)) {
    std::istringstream is(line);
    std::istream_iterator<std::string> begin(is), end;
    std::vector<std::string> tokens(begin, end);
    HistType history;
    int32 word;
    BaseFloat hist_weight = 0;
    for (int32 i = 0; i < tokens.size() - 1; ++i) {
      word = sym->Find(tokens[i]);
      if (word == fst::SymbolTable::kNoSymbol) {
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

// Test the read-in ARPA LM
void ArpaForSamplingTest::TestReadingModel() {
  KALDI_LOG << "Testing model reading part..."<< std::endl;
  KALDI_LOG << "Maximum symbol index is: " << arpa_->num_words_ - 1;
  KALDI_LOG << "Ngram_order is: " << arpa_->ngram_order_;
  KALDI_ASSERT(arpa_->probs_.size() == arpa_->ngram_counts_.size());
  for (int32 i = 0; i < arpa_->ngram_order_; ++i) {
    int32 size_ngrams = 0;
    KALDI_LOG << "Test: for order " << (i + 1);
    KALDI_LOG << "Expected number of " << (i + 1) << "-grams: "
              << arpa_->ngram_counts_[i];
    NgramType::const_iterator it1 = arpa_->probs_[i].begin();
    for (; it1 != arpa_->probs_[i].end(); ++it1) {
      HistType h(it1->first);
      for (MapType::const_iterator it2 = arpa_->probs_[i][h].begin();
          it2 != arpa_->probs_[i][h].end(); ++it2) {
        size_ngrams++;
      }
    }
    KALDI_LOG << "Read in number of " << (i + 1) << "-grams: " << size_ngrams;
  }
  // Assert the sum of unigram probs is 1.0
  BaseFloat prob_sum = 0.0;
  int32 max_symbol_id = 0;
  NgramType::const_iterator it1 = arpa_->probs_[0].begin();
  for (; it1 != arpa_->probs_[0].end(); ++it1) {
    HistType h(it1->first);
    for (MapType::const_iterator it2 = arpa_->probs_[0][h].begin();
        it2 != arpa_->probs_[0][h].end(); ++it2) {
      prob_sum += 1.0 * Exp(it2->second.first);
      max_symbol_id++;
    }
  }
  KALDI_ASSERT(max_symbol_id + 1 == arpa_->num_words_);
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));

  // Assert sum of bigram probs of all words given an arbitrary history is 1.0
  prob_sum = 0.0;
  it1 = arpa_->probs_[1].begin();
  HistType h(it1->first);
  HistType h_empty;
  for (MapType::const_iterator it = arpa_->probs_[0][h_empty].begin();
      it != arpa_->probs_[0][h_empty].end(); it++) {
    int32 word = it->first;
    MapType::const_iterator it2 = arpa_->probs_[1][h].find(word);
    if (it2 != arpa_->probs_[1][h].end()) {
      prob_sum += 1.0 * Exp(it2->second.first);
    } else {
      prob_sum += Exp(arpa_->GetLogprob(word, h));
    }
  }
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
}

// Test the generated unigram distribution
void ArpaForSamplingTest::TestUnigramDistribution() {
  std::vector<BaseFloat> unigram_probs;
  arpa_->GetUnigramDistribution(&unigram_probs);
  // Check 0 (epsilon) has probability 0.0
  KALDI_ASSERT(unigram_probs[0] == 0.0);
  // Assert the sum of unigram probs of all words is 1.0
  BaseFloat probsum = 0.0;
  for (int32 i = 0; i < unigram_probs.size(); ++i) {
    probsum += unigram_probs[i];
  }
  KALDI_ASSERT(ApproxEqual(probsum, 1.0));
}

void ArpaForSamplingTest::TestGetDistribution(const WeightedHistType &histories) {
  // get total input weights of histories
  BaseFloat total_weights = 0.0;
  WeightedHistType::const_iterator it = histories.begin();
  for (; it != histories.end(); ++it) {
    total_weights += it->second;
  }
  BaseFloat unigram_weight = 0.0;
  BaseFloat non_unigram_probsum = 0.0;
  std::unordered_map<int32, BaseFloat> pdf;
  unigram_weight = arpa_->GetDistribution(histories, &pdf);
  KALDI_ASSERT(pdf.size() <= arpa_->num_words_);
  for (std::unordered_map<int32, BaseFloat>::const_iterator it = pdf.begin();
      it != pdf.end(); it++) {
    non_unigram_probsum += it->second;
  }
  // assert unigram weight plus non_unigram probs' sum equals
  // the total input histories' weights
  KALDI_ASSERT(ApproxEqual(unigram_weight + non_unigram_probsum, total_weights));
}

}  // namespace kaldi

int main(int argc, char **argv) {
  using namespace kaldi;

  const char *usage = "";
  ParseOptions po(usage);
  po.Read(argc, argv);
  std::string arpa_file = po.GetArg(1), history_file = po.GetArg(2);

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
  ArpaForSampling arpa(options, &symbols);

  bool binary;
  Input k1(arpa_file, &binary);
  arpa.Read(k1.Stream(), binary);

  ArpaForSamplingTest mdl(&arpa);
  mdl.TestReadingModel();
  mdl.TestUnigramDistribution();

  Input k2(history_file, &binary);
  ArpaForSamplingTest::WeightedHistType histories;
  mdl.ReadHistories(k2.Stream(), binary, &histories);
  mdl.TestGetDistribution(histories);
  return 0;
}

// rnnlm/arpa-sampling-test.cc

#include "rnnlm/arpa-sampling.h"

namespace kaldi {

class ArpaSamplingTest {
 public:
  typedef ArpaSampling::HistType HistType;
  typedef ArpaSampling::WeightedHistType WeightedHistType;

  explicit ArpaSamplingTest(ArpaSampling *arpa) {
    arpa_ = arpa;
  }
  // This function reads in a list of histories and their weights from a file
  // only text form is supported
  void ReadHistories(std::istream &is, bool binary,
      WeightedHistType *histories);

  void TestUnigramDistribution();

  void TestHigherOrderProbs();

  void TestGetDistribution(const WeightedHistType &histories);
 private:
  // This ArpaSampling object is used to get accesses to private and protected
  // members in ArpaSampling class
  ArpaSampling *arpa_;
};

void ArpaSamplingTest::ReadHistories(std::istream &is, bool binary,
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

void ArpaSamplingTest::TestUnigramDistribution() {
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

void ArpaSamplingTest::TestGetDistribution(const WeightedHistType &histories) {
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

void ArpaSamplingTest::TestHigherOrderProbs() {
  int32 size = arpa_->higher_order_probs_.size();
  KALDI_ASSERT(size == arpa_->Order() - 1);

  // Assert sum of bigram probs of all words given an arbitrary history is 1.0
  BaseFloat prob_sum = 0.0;
  std::unordered_map<HistType, ArpaSampling::HistoryState,
                                 VectorHasher<int32> >::const_iterator it1;
  it1 = arpa_->higher_order_probs_[0].begin();
  HistType h(it1->first);
  KALDI_ASSERT(h.size() == 1);
  for (int32 i = 0; i < arpa_->unigram_probs_.size(); ++i) {
    int32 word = i;
    unordered_map<int32, BaseFloat>::const_iterator it =
      arpa_->higher_order_probs_[0][h].word_to_prob.find(word);
    if (it != arpa_->higher_order_probs_[0][h].word_to_prob.end()) {
      prob_sum += it->second;
      BaseFloat probs = arpa_->higher_order_probs_[0][h].backoff_prob;
      probs *= arpa_->unigram_probs_[word];
      prob_sum += probs;
    } else {
      BaseFloat probs = arpa_->higher_order_probs_[0][h].backoff_prob;
      probs *= arpa_->unigram_probs_[word];
      prob_sum += probs;
    }
  }
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
}
}  // namespace kaldi

int main(int argc, char **argv) {
  using namespace kaldi;

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
  ArpaSampling arpa(options, &symbols);

  bool binary;
  Input k1(arpa_file, &binary);
  arpa.Read(k1.Stream(), binary);

  ArpaSamplingTest mdl(&arpa);
  mdl.TestUnigramDistribution();
  mdl.TestHigherOrderProbs();

  Input k2(history_file, &binary);
  ArpaSamplingTest::WeightedHistType histories;
  mdl.ReadHistories(k2.Stream(), binary, &histories);
  mdl.TestGetDistribution(histories);
  KALDI_LOG << "Tests for ArpaSampling class succeed.";
  return 0;
}

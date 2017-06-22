// lm/arpa-sampling-test.cc

#include <math.h>
#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "arpa-sampling.h"

namespace kaldi {

class ArpaSamplingTest {
 public:
  typedef ArpaSampling::HistType HistType;
  typedef ArpaSampling::WordToProbsMap WordToProbsMap;
  typedef ArpaSampling::NgramType NgramType;
  typedef ArpaSampling::HistWeightsType HistWeightsType;
  
  ArpaSamplingTest (ArpaSampling *arpa) {
    arpa_ = arpa;
  } 
  // This function reads in a list of histories and their weights from a file
  // only text form is supported
  void ReadHistories(std::istream &is, bool binary, 
      std::vector<std::pair<HistType, BaseFloat> > *histories);
  
  // This function tests the correctness of the read-in ARPA LM 
  void TestReadingModel();

  // This function tests the generated unigram distribution 
  void TestUnigramDistribution();

  // Test non-unigram words and alpha
  // TODO: need to test alpha in a proper way 
  void TestNonUnigramWordsAndAlpha(const
      std::vector<std::pair<HistType, BaseFloat> > &histories);
 private:
  // This ArpaSampling object is used to get accesses to private and protected
  // members in ArpaSampling class
  ArpaSampling *arpa_;
};

void ArpaSamplingTest::ReadHistories(std::istream &is, bool binary,
    std::vector<std::pair<HistType, BaseFloat> > *histories) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for ArpaFileParser";
  }
  const fst::SymbolTable* sym = arpa_->Symbols();
  std::string line;
  KALDI_LOG << "Start reading histories from file...";
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
        word = sym->Find(arpa_->unk_symbol_);
      }
      history.push_back(word);
    }
    HistType h1;
    if (history.size() >= ngram_order) {
      HistType h(history.end() - ngram_order + 1, history.end());
      h1 = h;
    }
    #define PARSE_ERR (KALDI_ERR << arpa_->LineReference() << ": ")
    if (!ConvertStringToReal(tokens.back(), &hist_weight)) {
      PARSE_ERR << "invalid history weight '" << tokens.back() << "'";
    }
    #undef PARSE_ERR
    KALDI_ASSERT(hist_weight >=0);
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
void ArpaSamplingTest::TestReadingModel() {
  KALDI_LOG << "Testing model reading part..."<< std::endl;
  KALDI_LOG << "Vocab size is: " << arpa_->num_words_;
  KALDI_LOG << "Ngram_order is: " << arpa_->ngram_order_;
  KALDI_ASSERT(arpa_->probs_.size() == arpa_->ngram_counts_.size());
  for (int32 i = 0; i < arpa_->ngram_order_; ++i) {
    int32 size_ngrams = 0;
    KALDI_LOG << "Test: for order " << (i + 1);
    KALDI_LOG << "Expected number of " << (i + 1) << "-grams: " << arpa_->ngram_counts_[i];
    for (NgramType::const_iterator it1 = arpa_->probs_[i].begin(); it1 != arpa_->probs_[i].end(); ++it1) {
      HistType h(it1->first);
      for (WordToProbsMap::const_iterator it2 = arpa_->probs_[i][h].begin(); 
          it2 != arpa_->probs_[i][h].end(); ++it2) {
        size_ngrams++; // number of words given
      }
    }
    KALDI_LOG << "Read in number of " << (i + 1) << "-grams: " << size_ngrams;
  }
  // Assert the sum of unigram probs is 1.0
  BaseFloat prob_sum = 0.0;
  int32 count = 0;
  for (NgramType::const_iterator it1 = arpa_->probs_[0].begin(); it1 != arpa_->probs_[0].end(); ++it1) {
    HistType h(it1->first);
    for (WordToProbsMap::const_iterator it2 = arpa_->probs_[0][h].begin(); 
        it2 != arpa_->probs_[0][h].end(); ++it2) {
      prob_sum += 1.0 * Exp(it2->second.first);
      count++;
    }
  }
  KALDI_ASSERT(count == arpa_->num_words_);
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
  
  // Assert the sum of bigram probs of all words given an arbitrary history is 1.0
  prob_sum = 0.0;
  NgramType::const_iterator it1 = arpa_->probs_[1].begin();
  HistType h(it1->first);
  HistType h_empty; 
  for (WordToProbsMap::const_iterator it = arpa_->probs_[0][h_empty].begin();
      it != arpa_->probs_[0][h_empty].end(); it++) {
    int32 word = it->first;
    WordToProbsMap::const_iterator it2 = arpa_->probs_[1][h].find(word);
    if (it2 != arpa_->probs_[1][h].end()) {
      prob_sum += 1.0 * Exp(it2->second.first);
    } else {
      prob_sum += Exp(arpa_->GetProb(2, word, h));
    }
  }
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
}

// Test the generated unigram distribution
void ArpaSamplingTest::TestUnigramDistribution() {
  KALDI_LOG << "Testing the generated unigram distribution";
  int32 num_words = arpa_->num_words_;
  std::vector<BaseFloat> unigram_probs(num_words, 0.0);
  arpa_->GetUnigramDistribution(&unigram_probs);
  // Check 0 (epsilon) has probability 0.0
  KALDI_ASSERT(unigram_probs[0] == 0.0);
  // Assert the sum of unigram probs of all words is 1.0
  BaseFloat prob_sum = 0.0;
  for (int32 i = 0; i < unigram_probs.size(); ++i) {
    prob_sum += unigram_probs[i];
  }
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
}

void ArpaSamplingTest::TestNonUnigramWordsAndAlpha(
    const std::vector<std::pair<HistType, BaseFloat> > &histories) {
  BaseFloat alpha = 0.0;
  BaseFloat sum = 0.0;
  std::unordered_map<int32, BaseFloat> pdf;
  alpha = arpa_->GetOutputWordsAndAlpha(histories, &pdf);
  for (std::unordered_map<int32, BaseFloat>::const_iterator it = pdf.begin();
      it != pdf.end(); it++) {
    sum += it->second;
  }
  KALDI_LOG << "alpha: " << alpha;
  KALDI_LOG << "prob_sum of non-unigram words: " << sum;
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
  ArpaSampling arpa(options, &symbols);
  
  bool binary;
  Input k1(arpa_file, &binary);
  arpa.Read(k1.Stream(), binary);

  ArpaSamplingTest mdl(&arpa); 
  mdl.TestReadingModel();
  mdl.TestUnigramDistribution();

  Input k2(history_file, &binary);
  std::vector<std::pair<ArpaSamplingTest::HistType, BaseFloat> > histories;
  mdl.ReadHistories(k2.Stream(), binary, &histories);
  mdl.TestNonUnigramWordsAndAlpha(histories);
  return 0;
}

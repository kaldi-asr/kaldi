// lm/arpa-sampling-test.cc

#include "arpa-sampling.h"

#include <math.h>
#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

namespace kaldi {

void ArpaSampling::ReadHistories(std::istream &is, bool binary,
    std::vector<std::pair<HistType, BaseFloat> > *histories) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for ArpaFileParser";
  }
  const fst::SymbolTable* sym = Symbols();
  std::string line;
  KALDI_LOG << "Start reading histories from file...";
  while (getline(is, line)) {
    std::istringstream is(line);
    std::istream_iterator<std::string> begin(is), end;
    std::vector<std::string> tokens(begin, end);
    HistType history;
    int32 word;
    BaseFloat hist_weight = 0;
    for (int32 i = 0; i < tokens.size() - 1; i++) {
      word = sym->Find(tokens[i]);
      if (word == fst::SymbolTable::kNoSymbol) {
        word = sym->Find(unk_symbol_);
      }
      history.push_back(word);
    }
    HistType h1;
    if (history.size() >= ngram_order_) {
      HistType h(history.end() - ngram_order_ + 1, history.end());
      h1 = h;
    }
    #define PARSE_ERR (KALDI_ERR << LineReference() << ": ")
    if (!ConvertStringToReal(tokens.back(), &hist_weight)) {
      PARSE_ERR << "invalid history weight '" << tokens.back() << "'";
    }
    #undef PARSE_ERR
    // hist_weight = std::stof(tokens.back());
    KALDI_ASSERT(hist_weight >=0);
    std::pair<HistType, BaseFloat> hist_pair;
    if (history.size() >= ngram_order_) {
      hist_pair = std::make_pair(h1, hist_weight);
    } else {
      hist_pair = std::make_pair(history, hist_weight);
    }
    (*histories).push_back(hist_pair);
  }
  KALDI_LOG << "Successfully reading histories from file.";
}

// Test the read-in ARPA LM 
void ArpaSampling::TestReadingModel() {
  KALDI_LOG << "Testing model reading part..."<< std::endl;
  KALDI_LOG << "Vocab size is: " << num_words_;
  KALDI_LOG << "Ngram_order is: " << ngram_order_;
  KALDI_ASSERT(probs_.size() == ngram_counts_.size());
  for (int32 i = 0; i < ngram_order_; i++) {
    int32 size_ngrams = 0;
    KALDI_LOG << "Test: for order " << (i + 1);
    KALDI_LOG << "Expected number of " << (i + 1) << "-grams: " << ngram_counts_[i];
    for (NgramType::const_iterator it1 = probs_[i].begin(); it1 != probs_[i].end(); ++it1) {
      HistType h(it1->first);
      for (WordToProbsMap::const_iterator it2 = probs_[i][h].begin(); 
          it2 != probs_[i][h].end(); ++it2) {
        size_ngrams++; // number of words given
      }
    }
    KALDI_LOG << "Read in number of " << (i + 1) << "-grams: " << size_ngrams;
  }
  // Assert the sum of unigram probs is 1.0
  BaseFloat prob_sum = 0.0;
  int32 count = 0;
  for (NgramType::const_iterator it1 = probs_[0].begin(); it1 != probs_[0].end();++it1) {
    HistType h(it1->first);
    for (WordToProbsMap::const_iterator it2 = probs_[0][h].begin(); 
        it2 != probs_[0][h].end(); ++it2) {
      prob_sum += 1.0 * exp(it2->second.first);
      count++;
    }
  }
  KALDI_ASSERT(count == num_words_);
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
  
  // Assert the sum of bigram probs of all words given an arbitrary history is 1.0
  prob_sum = 0.0;
  NgramType::const_iterator it1 = probs_[1].begin();
  HistType h(it1->first);
  HistType h_empty; 
  for (WordToProbsMap::const_iterator it = probs_[0][h_empty].begin();
      it != probs_[0][h_empty].end(); it++) {
    int32 word = it->first;
    WordToProbsMap::const_iterator it2 = probs_[1][h].find(word);
    if (it2 != probs_[1][h].end()) {
      prob_sum += 1.0 * exp(it2->second.first);
    } else {
      prob_sum += exp(GetProb(2, word, h));
    }
  }
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
}

// Test the generated unigram distribution
void ArpaSampling::TestUnigramDistribution() {
  KALDI_LOG << "Testing the generated unigram distribution";
  int32 num_words = GetNumWords();
  std::vector<BaseFloat> unigram_probs(num_words, 0.0);
  GetUnigramDistribution(&unigram_probs);
  // Check 0 (epsilon) has probability 0.0
  KALDI_ASSERT(unigram_probs[0] == 0.0);
  // Assert the sum of unigram probs of all words is 1.0
  BaseFloat prob_sum = 0.0;
  for (int32 i = 0; i < unigram_probs.size(); i++) {
    prob_sum += unigram_probs[i];
  }
  KALDI_ASSERT(ApproxEqual(prob_sum, 1.0));
}

} // namespace kaldi

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
  ArpaSampling mdl(options, &symbols);
  
  bool binary;
  Input k1(arpa_file, &binary);
  mdl.Read(k1.Stream(), binary);
  mdl.TestReadingModel();
  mdl.TestUnigramDistribution();

  Input k2(history_file, &binary);
  std::vector<std::pair<ArpaSampling::HistType, BaseFloat> > histories;
  mdl.ReadHistories(k2.Stream(), binary, &histories);
  BaseFloat alpha = 0.0;
  unordered_map<int32, BaseFloat> pdf;
  alpha = mdl.GetOutputWordsAndAlpha(histories, &pdf);
  return 0;
}

// rnnlm/rnnlm-test-utils.cc

// Copyright 2017  Daniel Povey
//           2017  Hossein Hadian
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

#include <numeric>
#include "rnnlm/rnnlm-test-utils.h"

namespace kaldi {
namespace rnnlm {

void GetForbiddenSymbols(std::set<std::string> *forbidden_symbols) {
  *forbidden_symbols = {"<eps>", "<s>", "<brk>", "</s>"};
}

///  Reads all the lines from a text file and appends
///  them to the "sentences" vector.
void ReadAllLines(const std::string &filename,
                  std::vector<std::vector<std::string> > *sentences) {
  std::ifstream is(filename.c_str());
  std::string line;
  while (std::getline(is, line)) {
    std::vector<std::string> split_line;
    SplitStringToVector(line, "\t\r\n ", true, &split_line);
    sentences->push_back(split_line);
  }
  if (sentences->size() < 1)
    KALDI_ERR << "No line could be read from the file.";
}

void GetTestSentences(const std::set<std::string> &forbidden_symbols,
                      std::vector<std::vector<std::string> > *sentences) {
  sentences->clear();
  ReadAllLines("arpa-sampling-test.cc", sentences);
  ReadAllLines("rnnlm-example-test.cc", sentences);
  ReadAllLines("rnnlm-example.cc", sentences);
  ReadAllLines("rnnlm-example-utils.cc", sentences);

  // find and escape forbidden symbols
  for (int i = 0; i < sentences->size(); i++)
    for (int j = 0; j < (*sentences)[i].size(); j++)
      if (forbidden_symbols.find((*sentences)[i][j]) != forbidden_symbols.end())
        (*sentences)[i][j] = "\\" + (*sentences)[i][j];
}

fst::SymbolTable *GetSymbolTable(
    const std::vector<std::vector<std::string> > &sentences) {
  fst::SymbolTable* table = new fst::SymbolTable();
  table->AddSymbol("<eps>", 0);
  table->AddSymbol("<s>", 1);
  table->AddSymbol("</s>", 2);
  table->AddSymbol("<brk>", 3);
  for (int i = 0; i < sentences.size(); i++)
    for (int j = 0; j < sentences[i].size(); j++)
      table->AddSymbol(sentences[i][j]);
  return table;
}

void ConvertToInteger(
    const std::vector<std::vector<std::string> > &string_sentences,
    const fst::SymbolTable &symbol_table,
    std::vector<std::vector<int32> > *int_sentences) {
  int_sentences->resize(string_sentences.size());
  for (int i = 0; i < string_sentences.size(); i++) {
    (*int_sentences)[i].resize(string_sentences[i].size());
    for (int j = 0; j < string_sentences[i].size(); j++) {
      kaldi::int64 key = symbol_table.Find(string_sentences[i][j]);
      KALDI_ASSERT(key != fst::SymbolTable::kNoSymbol);
      (*int_sentences)[i][j] = static_cast<int32>(key);
    }
  }
}

/** Simple implementation of Interpolated Kneser-Ney smoothing,
    see the formulas in "A Bit of Progress in Language Modeling"

    Note that we won't follow the procedure of SRILM implementation(collect
    counts, then modify counts, and then discount to get probs). Instead,
    we accumulate the number of context occurrences directly during we pass
    through the training text. We translate the perl code in the appendix of
    the paper and extend it to arbitrary ngram order. Also, as in SRILM,
    we use the original(unmodified) count for the ngrams starting with <s>.
*/
class InterpolatedKneserNeyLM {
 public:
  struct LMState {
    int32 numerator;
    int32 denominator;
    int32 non_zero_count;
    BaseFloat prob;
    BaseFloat bow;

    LMState() : numerator(0), denominator(0), non_zero_count(0),
                prob(0.0), bow(0.0) {};
  };
  typedef unordered_map<std::vector<int32>, LMState, VectorHasher<int32> > Ngrams;

  /** Constructor.
       @param [in] ngram_order  The n-gram order of the language model to
                                be estimated.
       @param [in] discount   Fixed value of discount, i.e. the D in formula.
   */
  InterpolatedKneserNeyLM(int32 ngram_order, double discount) {
    ngram_order_ = ngram_order;
    discount_ = discount;
    ngrams_.resize(ngram_order + 1); // ngrams_[0] unused
  }

  void FillWords(const std::vector<int32> &sentence,
                 int32 pos, int32 order,
                 int32 bos_symbol, int32 eos_symbol,
                 std::vector<int32> *words) {
    KALDI_ASSERT(pos >= -1 && pos <= sentence.size());

    words->resize(order);
    for (int32 k = 0; k < order; k++, pos++) {
      if (pos < 0) {
        (*words)[k] = bos_symbol;
      } else if (pos >= sentence.size()) {
        (*words)[k] = eos_symbol;
      } else {
        (*words)[k] = sentence[pos];
      }
    }
  }
  /* Collect the ngram counts from corpus. */
  void CollectCounts(const std::vector<std::vector<int32> > &sentences,
                     int32 bos_symbol, int32 eos_symbol) {
    std::vector<int32> words;
    std::vector<int32> subwords;

    for (int32 i = 0; i < sentences.size(); i++) {
      for (int32 j = 0; j < sentences[i].size() + 1; j++) {
        int32 max_order = j - ngram_order_ + 1;
        if (max_order < -1) {
          max_order = -max_order;
        } else {
          max_order = ngram_order_;
        }

        // in the following for-loop, only the max_order ngrams will
        // get their actual counts. And the max_order ngrams are the
        // ngram with ngram_order_ or the ngrams starting with <s>.
        for (int32 order = max_order; order >= 1; order--) {
          FillWords(sentences[i], j - order + 1, order,
                    bos_symbol, eos_symbol, &words);
          // accumulate numerator
          LMState& state = ngrams_[order][words];
          state.numerator++;

          if (order == 1) {
            unigram_denominator_++;
          } else {
            // accumulate denominator for context
            subwords.assign(words.begin(), words.end() - 1);
            state = ngrams_[order - 1][subwords];
            state.denominator++;
            if (state.numerator > 1) { // not first insertion
              // accumulate for context
              state.non_zero_count++;

              // accumulate numerator for low order gram
              subwords.assign(words.begin() + 1, words.end());
              state = ngrams_[order - 1][subwords];
              state.numerator++;

              // for lower order ngram, we only need occurrence, so if it is
              // already in the map, we skip it
              break;
            }
          }
        }
      }
    }
  }

  /* Compute ngram probs and bows with the counts. */
  void EstimateProbAndBow() {
    for (int32 order = 1; order <= ngram_order_; order++) {
      Ngrams::iterator it = ngrams_[order].begin();
      for (; it != ngrams_[order].end(); it++) {
        LMState& state = it->second;
        if (order == 1) {
          // here, we assume all words in the vocabulary are appeared in the
          // training text, or we have to do discount. Since we won't get
          // the symbol table until WriteToARPA(), we don't know the size
          // of vocabulary and can't work out the discount.
          state.prob = 1.0 * state.numerator / unigram_denominator_;
        } else {
          std::vector<int32> subwords;
          Ngrams::const_iterator context, lower_order;

          subwords.assign(it->first.begin(), it->first.end() - 1);
          context = ngrams_[order - 1].find(subwords);
          KALDI_ASSERT(context != ngrams_[order - 1].end());
          state.prob = (state.numerator - discount_)
                       / context->second.denominator;

          // interpolate lower orders
          for (int32 o = 1; o < order; o++) {
            subwords.assign(it->first.begin() + o - 1, it->first.end() - 1);
            context = ngrams_[order - o].find(subwords);
            KALDI_ASSERT(context != ngrams_[order - o].end());

            subwords.assign(it->first.begin() + o, it->first.end());
            lower_order = ngrams_[order - o].find(subwords);
            KALDI_ASSERT(lower_order != ngrams_[order - o].end());

            state.prob += context->second.bow * lower_order->second.prob;
          }
        }

        state.bow = state.non_zero_count * discount_ / state.denominator;
      }
    }
  }

  /** Estimate the language model with corpus in sentences.

      @param [in] sentences   The sentences of input data.  These will contain
                              just the actual words, not the BOS or EOS symbols.
      @param [in] bos_symbol  The integer id of the beginning-of-sentence
                              symbol
      @param [in] eos_symbol  The integer id of the end-of-sentence
                             symbol
   */
  void Estimate(const std::vector<std::vector<int32> > &sentences,
                int32 bos_symbol, int32 eos_symbol) {
    CollectCounts(sentences, bos_symbol, eos_symbol);
    EstimateProbAndBow();
  }

  static BaseFloat ProbToLobProb(BaseFloat prob) {
    if (prob == 0.0) {
      return -99.0;
    } else {
      return log10(prob);
    }
  }

  static void WriteNgram(const std::vector<int32> words,
                  BaseFloat prob, BaseFloat bow,
                  const fst::SymbolTable &symbol_table, std::ostream &os) {
    os << ProbToLobProb(prob) << "\t";
    for (int32 i = 0; i < words.size() - 1; i++) {
      os << symbol_table.Find(words[i]) << " ";
    }
    os << symbol_table.Find(words[words.size() - 1]);
    os << ProbToLobProb(bow) << "\n";
  }

  /** Write to the ostream with ARPA format. Throws on error.
       @param [in] symbol_table  The OpenFst symbol table. It's needed
                                 because the ARPA file we write out
                                 is in text form.
       @param [out] os       The stream to which this function will write
                             the language model in ARPA format.
   */
  void WriteToARPA(const fst::SymbolTable &symbol_table,
                   std::ostream &os) const {
    // we write out only the words appeared in training text, instead of all
    // words in symbol_table, since there would be some special symbols in
    // symbol_table and our unigram distribution are calculated without
    // considering the (maybe exist) extra words.
    os << "\\\\data\\\\\n";
    for (int32 order = 1; order <= ngram_order_; order++) {
      os << "ngram " << order << "=" << ngrams_[order].size();
    }

    for (int32 order = 1; order <= ngram_order_; order++) {
      os << "\n\\" << order << "-grams:\n";
      Ngrams::const_iterator it = ngrams_[order].begin();
      for (; it != ngrams_[order].end(); it++) {
        WriteNgram(it->first, it->second.prob, it->second.bow,
                   symbol_table, os);
      }
    }

    os << "\n\\\\end\\\\\n";
  }

 private:

  // Ngram order
  int32 ngram_order_;

  // Fix value of discount
  double discount_;

  // ngrams for each order
  std::vector<Ngrams> ngrams_;

  // denominator for unigrams
  int32 unigram_denominator_;
};

void EstimateAndWriteLanguageModel(
    int32 ngram_order,
    const fst::SymbolTable &symbol_table,
    const std::vector<std::vector<int32> > &sentences,
    int32 bos_symbol, int32 eos_symbol,
    std::ostream &os) {
  InterpolatedKneserNeyLM lm(ngram_order, 0.6);
  lm.Estimate(sentences, bos_symbol, eos_symbol);
  lm.WriteToARPA(symbol_table, os);
}

}  // namespace rnnlm
}  // namespace kaldi

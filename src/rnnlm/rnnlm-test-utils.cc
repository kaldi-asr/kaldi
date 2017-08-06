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
*/
class InterpolatedKneserNeyLM {
 public:
  struct LMState {
    int32 count;
    BaseFloat prob;
    BaseFloat bow;
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
    ngrams_.resize(ngram_order);
  }

  /* Collect the ngram counts from corpus. */
  void CollectCounts(const std::vector<std::vector<int32> > &sentences,
                int32 bos_symbol, int32 eos_symbol) {
    std::vector<int32> ngram(ngram_order_);

    for (int32 i = 0; i < sentences.size(); i++) {
      for (int32 j = 0; j < sentences[i].size() + 1; j++) {
        int32 max_order = j - ngram_order_ + 1;
        if (max_order < -1) {
          max_order = -max_order;
        } else {
          max_order = ngram_order_;
        }

        for (int32 order = 1; order <= max_order; order++) {
          for (int32 k = 1; k <= order; k++) {
            int32 pos = j - order + k;
            if (pos < 0) {
              ngram[k - 1] = bos_symbol;
            } else if (pos >= sentences[i].size()) {
              ngram[k - 1] = eos_symbol;
            } else {
              ngram[k - 1] = sentences[i][pos];
            }
          }

          Ngrams::iterator it = ngrams_[order].find(ngram);
          if (it == ngrams_[order].end()) {
            ngrams_[order].insert(make_pair(ngram, LMState{1, 0.0, 0.0}));
          } else {
            ++(it->second.count);
          }
        }
      }
    }
  }

  /* Modify the ngram counts with Kneser-Ney smoothing. */
  void EstimateProbAndBow() {
  }

  /** Estimate the language model with corpus in sentences.
      We won't follow the procedure of SRILM implementation(collect counts,
      then modeify counts, and then discount and get probs). Instead,
      we translate the perl code in the appendix of the paper. Also, we don't
      treat the ngram starting with non-events(<s>) as a special case.

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

  /** Write to the ostream with ARPA format. Throws on error.
       @param [in] symbol_table  The OpenFst symbol table. It's needed.
                                 because the ARPA file we write out
                                 is in text form.
       @param [out] os       The stream to which this function will write
                             the language model in ARPA format.
   */

  void WriteToARPA(const fst::SymbolTable &symbol_table,
                   std::ostream &os) const;

 private:

  // Ngram order
  int32 ngram_order_;

  // Fix value of discount
  double discount_;

  // ngrams for each order
  std::vector<Ngrams> ngrams_;
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

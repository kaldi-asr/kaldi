// lm/arpa-file-parser.cc

// Copyright 2014  Guoguo Chen
// Copyright 2016  Smart Action Company LLC (kkm)

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

#include <sstream>

#include <fst/fstlib.h>

#include "base/kaldi-error.h"
#include "base/kaldi-math.h"
#include "lm/arpa-file-parser.h"
#include "util/text-utils.h"

namespace kaldi {

ArpaFileParser::ArpaFileParser(ArpaParseOptions options, fst::SymbolTable* symbols)
    : options_(options), symbols_(symbols), line_number_(0) {
}

ArpaFileParser::~ArpaFileParser() {
}

void ArpaFileParser::Read(std::istream &is, bool binary) {
  if (binary) {
    KALDI_ERR << "binary-mode reading is not implemented for ArpaFileParser";
  }

  // Argument sanity checks.
  if (options_.bos_symbol <= 0 || options_.eos_symbol <= 0 ||
      options_.bos_symbol == options_.eos_symbol)
    KALDI_ERR << "BOS and EOS symbols are required, must not be epsilons, and "
              << "differ from each other. Given:"
              << " BOS=" << options_.bos_symbol
              << " EOS=" << options_.eos_symbol;
  if (symbols_ != NULL &&
      options_.oov_handling == ArpaParseOptions::kReplaceWithUnk &&
      (options_.unk_symbol <= 0 ||
       options_.unk_symbol == options_.bos_symbol ||
       options_.unk_symbol == options_.eos_symbol))
    KALDI_ERR << "When symbol table is given and OOV mode is kReplaceWithUnk, "
              << "UNK symbol is required, must not be epsilon, and "
              << "differ from both BOS and EOS symbols. Given:"
              << " UNK=" << options_.unk_symbol
              << " BOS=" << options_.bos_symbol
              << " EOS=" << options_.eos_symbol;
  if (symbols_ != NULL && symbols_->Find(options_.bos_symbol).empty())
    KALDI_ERR << "BOS symbol must exist in symbol table";
  if (symbols_ != NULL && symbols_->Find(options_.eos_symbol).empty())
    KALDI_ERR << "EOS symbol must exist in symbol table";
  if (symbols_ != NULL && options_.unk_symbol > 0 &&
      symbols_->Find(options_.unk_symbol).empty())
    KALDI_ERR << "UNK symbol must exist in symbol table";

  ngram_counts_.clear();
  line_number_ = 0;

#define PARSE_ERR (KALDI_ERR << "in line " << line_number_ << ": ")

  // Give derived class an opportunity to prepare its state.
  ReadStarted();

  std::string line;

  // Processes "\data\" section.
  bool keyword_found = false;
  while (++line_number_, getline(is, line) && !is.eof()) {
    if (line.empty()) continue;

    // The section keywords starts with backslash. We terminate the while loop
    // if a new section is found.
    if (line[0] == '\\') {
      if (!keyword_found && line == "\\data\\") {
        KALDI_LOG << "Reading \\data\\ section.";
        keyword_found = true;
        continue;
      }
      break;
    }

    if (!keyword_found) continue;

    // Enters "\data\" section, and looks for patterns like "ngram 1=1000",
    // which means there are 1000 unigrams.
    std::size_t equal_symbol_pos = line.find("=");
    if (equal_symbol_pos != std::string::npos)
      line.replace(equal_symbol_pos, 1, " = ");  // Inserts spaces around "="
    std::vector<std::string> col;
    SplitStringToVector(line, " \t", true, &col);
    if (col.size() == 4 && col[0] == "ngram" && col[2] == "=") {
      int32 order, ngram_count = 0;
      if (!ConvertStringToInteger(col[1], &order) ||
          !ConvertStringToInteger(col[3], &ngram_count)) {
        PARSE_ERR << "Cannot parse ngram count '" << line << "'.";
      }
      if (ngram_counts_.size() <= order) {
        ngram_counts_.resize(order);
      }
      ngram_counts_[order - 1] = ngram_count;
    } else {
      KALDI_WARN << "Uninterpretable line in \\data\\ section: " << line;
    }
  }

  if (ngram_counts_.size() == 0)
    PARSE_ERR << "\\data\\ section missing or empty.";

  // Signal that grammar order and n-gram counts are known.
  HeaderAvailable();

  NGram ngram;
  ngram.words.reserve(ngram_counts_.size());

  // Processes "\N-grams:" section.
  for (int32 cur_order = 1; cur_order <= ngram_counts_.size(); ++cur_order) {
    // Skips n-grams with zero count.
    if (ngram_counts_[cur_order - 1] == 0) {
      KALDI_WARN << "Zero ngram count in ngram order " << cur_order
                 << "(look for 'ngram " << cur_order << "=0' in the \\data\\ "
                 << " section). There is possibly a problem with the file.";
      continue;
    }

    // Must be looking at a \k-grams: directive at this point.
    std::ostringstream keyword;
    keyword << "\\" << cur_order << "-grams:";
    if (line != keyword.str()) {
      PARSE_ERR << "Invalid directive '" << line << "', "
                << "expecting '" << keyword.str() << "'.";
    }
    KALDI_LOG << "Reading " << line << " section.";

    int32 ngram_count = 0;
    while (++line_number_, getline(is, line) && !is.eof()) {
      if (line.empty()) continue;
      if (line[0] == '\\') break;

      std::vector<std::string> col;
      SplitStringToVector(line, " \t", true, &col);

      if (col.size() < 1 + cur_order ||
          col.size() > 2 + cur_order ||
          (cur_order == ngram_counts_.size() && col.size() != 1 + cur_order)) {
        PARSE_ERR << "Invalid n-gram line '"  << line << "'";
      }
      ++ngram_count;

      // Parse out n-gram logprob and, if present, backoff weight.
      if (!ConvertStringToReal(col[0], &ngram.logprob)) {
        PARSE_ERR << "Invalid n-gram logprob '" << col[0] << "'.";
      }
      ngram.backoff = 0.0;
      if (col.size() > cur_order + 1) {
        if (!ConvertStringToReal(col[cur_order + 1], &ngram.backoff))
          PARSE_ERR << "Invalid backoff weight '" << col[cur_order + 1] << "'.";
      }
      // Convert to natural log unless the option is set not to.
      if (!options_.use_log10) {
        ngram.logprob *= M_LN10;
        ngram.backoff *= M_LN10;
      }

      ngram.words.resize(cur_order);
      bool skip_ngram = false;
      for (int32 index = 0; !skip_ngram && index < cur_order; ++index) {
        int32 word;
        if (symbols_) {
          // Symbol table provided, so symbol labels are expected.
          if (options_.oov_handling == ArpaParseOptions::kAddToSymbols) {
            word = symbols_->AddSymbol(col[1 + index]);
          } else {
            word = symbols_->Find(col[1 + index]);
            if (word == fst::SymbolTable::kNoSymbol) {
              switch(options_.oov_handling) {
                case ArpaParseOptions::kReplaceWithUnk:
                  word = options_.unk_symbol;
                  break;
                case ArpaParseOptions::kSkipNGram:
                  skip_ngram = true;
                  break;
                default:
                  PARSE_ERR << "Word '"  << col[1 + index]
                            << "' not in symbol table.";
              }
            }
          }
        } else {
          // Symbols not provided, LM file should contain integers.
          if (!ConvertStringToInteger(col[1 + index], &word) || word < 0) {
            PARSE_ERR << "invalid symbol '" << col[1 + index] << "'";
          }
        }
        // Whichever way we got it, an epsilon is invalid.
        if (word == 0) {
          PARSE_ERR << "Epsilon symbol '" << col[1 + index]
                    << "' is illegal in ARPA LM.";
        }
        ngram.words[index] = word;
      }
      if (!skip_ngram) {
        ConsumeNGram(ngram);
      }
    }
    if (ngram_count > ngram_counts_[cur_order - 1]) {
      PARSE_ERR << "Header said there would be " << ngram_counts_[cur_order]
                << " n-grams of order " << cur_order << ", but we saw "
                << ngram_count;
    }
  }

  if (line != "\\end\\") {
    PARSE_ERR << "Invalid or unexpected directive line '" << line << "', "
              << "expected \\end\\.";
  }

  ReadComplete();

#undef PARSE_ERR
}

}  // namespace kaldi

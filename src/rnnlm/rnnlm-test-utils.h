// rnnlm-test-utils.h

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_RNNLM_TEST_UTILS_H_
#define KALDI_RNNLM_TEST_UTILS_H_

#include "fst/fstlib.h"
#include "util/common-utils.h"
#include "lm/arpa-file-parser.h"

namespace kaldi {
namespace rnnlm {


/// Creates a set of std::string containing
/// the forbidden symbols "<eps>", "<s>", "<brk>", "</s>"
void GetForbiddenSymbols(std::set<std::string> *forbidden_symbols);


/// Reads source files in the current directory to create a dataset
/// suitable for LM training and testing.
///  @param [in] forbidden_symbols  This contains symbols which will be escaped
///                     with backslash whenever they appear as symbols in
///                     'sentences', to avoid creating sequences which contain
///                     special values such as <s> and </s>.
///  @param [out] sentences  At exit, will contain a list of sentences; each
///                      sentence is a sequence (possibly empty) of words,
///                      and each word is nonempty and free of whitespace,
///                      and different from the symbols listed in
///                      'forbidden_symbols'.
void GetTestSentences(const std::set<std::string> &forbidden_symbols,
                      std::vector<std::vector<std::string> > *sentences);



/// Returns a symbol table that maps <eps> -> 0, <s> -> 1,
/// </s> -> 2, <brk> -> 3, and any symbols that appear in 'sentences' to
/// other values.
fst::SymbolTable *GetSymbolTable(
    const std::vector<std::vector<std::string> > &sentences);


/// Converts the data in 'string_sentences' into integers via the
/// symbol table 'symbol_table', and writes to 'int_sentences'.
/// All words must be covered in the symbol table.a
void ConvertToInteger(
    const std::vector<std::vector<std::string> > &string_sentences,
    const fst::SymbolTable &symbol_table,
    std::vector<std::vector<int32> > *int_sentences);

/**
   This function estimates a backoff n-gram language model from the data in
   'sentences' and writes it to the stream 'os' in ARPA format.  This function
   is only used for testing so it doesn't have to be very efficient.  As it
   happens, this function estimates a Kneser-Ney language model (with a fixed
   value of D), but the calling code doesn't rely on this.

     @param [in] ngram_order  The n-gram order of the language model to
                              be written, e.g. 1.
     @param [in] symbol_table  The OpenFst symbol table, which will include
                              all the symbols in 'sentences' as well as
                              bos_symbol and eos_symbol.  It's needed.
                              because the ARPA file we write out
                              is in text form.
     @param [in] sentences   The sentences of input data.  These will contain
                             just the actual words, not the BOS or EOS symbols.
     @param [in] bos_symbol  The integer id of the beginning-of-sentence
                             symbol
     @param [in] eos_symbol  The integer id of the end-of-sentence
                             symbol
     @param [out] os         The stream to which this function will write
                             the language model in ARPA format.
 */
void EstimateAndWriteLanguageModel(
    int32 ngram_order,
    const fst::SymbolTable &symbol_table,
    const std::vector<std::vector<int32> > &sentences,
    int32 bos_symbol, int32 eos_symbol,
    std::ostream &os);





}  // namespace rnnlm
}  // namespace kaldi
#endif  // KALDI_RNNLM_ARPA_SAMPLING_H_

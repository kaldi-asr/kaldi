// idlaktxp/mod-pronounce.h

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#ifndef KALDI_IDLAKTXP_MOD_PRONOUNCE_H_
#define KALDI_IDLAKTXP_MOD_PRONOUNCE_H_

// This file defines the basic txp module which incrementally parses
// either text, tox (token oriented xml) tokens, or spurts (phrases)
// containing tox tokens.

#include <string>
#include "pugixml.hpp"

#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpmodule.h"
#include "idlaktxp/txpnrules.h"
#include "idlaktxp/txplexicon.h"
#include "idlaktxp/txplts.h"

namespace kaldi {

/// Convert tokens into pronunications based on lexicons and
/// lts rules. Currently only one lexicon is supported. A user lexicon
/// and ability to add bilingual lexicons may be added. /ref idlaktxp_pron
class TxpPronounce : public TxpModule {
 public:
  explicit TxpPronounce();
  ~TxpPronounce();
  bool Init(const TxpParseOptions &opts);
  bool Process(pugi::xml_document* input);

 private:
  /// Checks lexicon and lts to determine pronuciations
  /// and appends it to the lex lookup structure
  void AppendPron(const char* entry, const std::string &word,
                  TxpLexiconLkp* lexlkp);
  /// A normalisation rules object is required to allow the default
  /// pronunciation of symbol and digit characters
  TxpNRules nrules_;
  /// A pronuciation lexicon object
  TxpLexicon lex_;
  /// A cart based letter to sound rul object
  TxpLts lts_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_MOD_PRONOUNCE_H_

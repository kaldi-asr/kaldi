// idlaktxp/mod-tokenise.h

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

#ifndef KALDI_IDLAKTXP_MOD_TOKENISE_H_
#define KALDI_IDLAKTXP_MOD_TOKENISE_H_

// This file defines the basic txp module which incrementally parses
// either text, tox (token oriented xml) tokens, or spurts (phrases)
// containing tox tokens.

#include <string>
#include "idlaktxp/txpmodule.h"
#include "idlaktxp/txpnrules.h"

namespace kaldi {

/// Tokenise input text into tokens and whitespace
/// \ref idlaktxp_token
class TxpTokenise : public TxpModule {
 public:
  explicit TxpTokenise();
  ~TxpTokenise() {}
  bool Init(const TxpParseOptions &opts);
  bool Process(pugi::xml_document* input);

 private:
  /// Analyses the characters and sets flags giving case, foriegn
  /// charcater info
  int32 SetPuncCaseInfo(std::string *tkin, pugi::xml_node *tk);
  /// A normalisation rule database used to decide case etc.
  /// Currently this data will be loaded muliple times across
  /// multiple modules
  TxpNRules nrules_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_MOD_TOKENISE_H_

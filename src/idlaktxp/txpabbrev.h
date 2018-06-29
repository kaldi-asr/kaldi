// idlaktxp/txpabbrev.h

// Copyright 2018 CereProc Ltd.  (Author: Matthew Aylett)

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

#ifndef KALDI_IDLAKTXP_TXPABBREV_H_
#define KALDI_IDLAKTXP_TXPABBREV_H_

// This file controls pre normalisation abbreviation expansion

#include <utility>
#include <vector>
#include <string>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"
#include "idlaktxp/txputf8.h"

namespace kaldi {

struct TxpAbbrevInfo;

/// List of abbreviations in token size order
typedef std::vector<TxpAbbrevInfo * > AbbrevVector;

/// Holds mapping between punctuation characters and what strength
/// (in terms of break index) and time (seconds). Also holds whether this
/// is valid when the punctuation is before or after a token
class TxpAbbrev: public TxpXmlData {
 public:
  explicit TxpAbbrev() {}
  ~TxpAbbrev();
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "abbrev", name);
  }
  bool Parse(const std::string &tpdb);
  TxpAbbrevInfo * LookupAbbrev(const char * tk,
                               const char * prepunc,
                               const char * pstpunc);
  TxpAbbrevInfo * LookupAbbrev(const char * tk);
  int32 CheckPrePunc(const char * prepunc, TxpAbbrevInfo * abb);
  int32 CheckPstPunc(const char * pstpunc, TxpAbbrevInfo * abb);
 private:
  void StartElement(const char* name, const char** atts);
  void EndElement(const char *);
  void CharHandler(const char* data, int32 len);
  /// vector of abbreviations to expand before normalisation
  AbbrevVector abbreviations_;
  /// Parser status currently operating on this abbreviation
  TxpAbbrevInfo * current_abbrev_;
  /// Parser status in this lex tag
  std::string current_lexentry_;
};

/// Structure which holds abbreviation information
struct TxpAbbrevInfo {
 public:
  TxpAbbrevInfo() {}
  ~TxpAbbrevInfo() {}
  void AddExpansionToken(const char * expansion,
                         int expansion_len,
                         const char * lexentry);
  std::string prepunc;
  std::string token;
  std::string pstpunc;
  StringVector expansions;
  StringVector lexentries;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPABBREV_H_

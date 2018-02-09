// idlaktxp/txplexicon.h

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

#ifndef KALDI_IDLAKTXP_TXPLEXICON_H_
#define KALDI_IDLAKTXP_TXPLEXICON_H_

// This file defines the lexicon class to hold pronunciation dictionaries

#include <map>
#include <string>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"

namespace kaldi {

struct TxpLexiconEntry;
struct TxpLexiconLkp;

/// Custom comparison function to order lex items by default followed by
/// other entries.
struct TxpLexCompare {
 public:
   bool operator()(const std::string x, const std::string y) const {
    std::size_t p1, p2;
    p1 = x.find(":");
    p2 = y.find(":");
    if (!x.substr(0, p1).compare(y.substr(0, p2))) {
      /// if the same word always return default entries first
      if (x.find(":default") != std::string::npos &&
          y.find(":default") == std::string::npos) {
        return true;
      } else if (x.find(":default") == std::string::npos &&
               y.find(":default") != std::string::npos) {
        return false;
      /// otherwise in string entry order
      } else {
        return x.compare(y) < 0;
      }
    } else {
      /// otherwise in word order
      return (x.substr(0, p1).compare(y.substr(0, p2)) < 0);
    }
  }
};

/// Generic string to string lookup with custom comparison function
typedef std::multimap<std::string, std::string, TxpLexCompare> LookupLex;


/// Hold pronunciations for words by entry.
/// There must be at least one default pronunciation for every word
class TxpLexicon: public TxpXmlData {
 public:
  explicit TxpLexicon() : inlex_(false) {}
  ~TxpLexicon() {}
  void Init(const TxpParseOptions &opts, const std::string &name) {
    TxpXmlData::Init(opts, "lexicon", name);
  }
  /// Fill the lexicon lookup structure with the correct pronunciation
  int GetPron(const std::string &word,
              const std::string &entry,
              TxpLexiconLkp* lkp);

 private:
  void StartElement(const char* name, const char** atts);
  void EndElement(const char*);
  void CharHandler(const char* data, const int32 len);
  /// Holds <entry>:<word> and default:<word> pronunciation lookups
  LookupLex lookup_;
  /// Holds parser status in lex item
  bool inlex_;
  /// Hold current entry value during parse
  std::string entry_;
  /// Holds current default value during parse
  std::string isdefault_;
  /// Holds current pronunciation during parse
  std::string pron_;
  /// Holds current word during parse
  std::string word_;
};

/// This structure is used to find the pronunciation of a word
struct TxpLexiconLkp {
  TxpLexiconLkp() : pron(""), lts(false) {}
  void Reset() {
    pron = "";
    lts = false;
    altprons.clear();
  }
  /// Pronunciation in space delimited phones
  std::string pron;
  /// Set to true is letter to sound rules used to determine pronuncuation
  bool lts;
  /// List of all pronunciations of this word
  StringVector altprons;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPLEXICON_H_

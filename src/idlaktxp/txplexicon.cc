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

#include "idlaktxp/txplexicon.h"

namespace kaldi {

void TxpLexicon::StartElement(const char* name, const char** atts) {
  if (!strcmp(name, "lex")) {
    inlex_ = true;
    SetAttribute("entry", atts, &entry_);
    SetAttribute("pron", atts, &pron_);
    SetAttribute("default", atts, &isdefault_);
    word_ = "";
  }
}

void TxpLexicon::CharHandler(const char* data, const int32 len) {
  if (inlex_) {
    word_ = word_ + std::string(data, len);
  }
}

void TxpLexicon:: EndElement(const char* name) {
  LookupMap::iterator it;
  if (!strcmp(name, "lex")) {
    inlex_ = false;
    // if word as delimiter char : in it we can't add it
    if (word_.find(":") != std::string::npos) {
      KALDI_ERR << "Lexicon error with forbidden ':' character: " << word_;
      return;
    }
    // Only one default for each word is valid
    if (isdefault_ == "true") {
      it =  lookup_.find(word_ + std::string(":default"));
      if (it == lookup_.end()) {
        lookup_.insert(LookupItem(word_ + std::string(":default"), pron_));
      } else {
        KALDI_ERR << "Lexicon error muliple default entries for word: "
                  << word_;
        return;
      }
    }
    // If entry types repeat only the first will normally be accessed
    lookup_.insert(LookupItem(word_ + std::string(":") + entry_, pron_));
  }
}

int TxpLexicon::GetPron(const std::string &word,
                        const std::string &entry,
                        TxpLexiconLkp* lkp) {
  LookupMap::iterator it;
  std::size_t pos;
  if (!entry.empty())
    it = lookup_.find(word + std::string(":") + entry);
  else
    it = lookup_.find(word + std::string(":default"));
  if (it != lookup_.end()) {
    lkp->pron += it->second;
    // Extract other pronunciations
    it = lookup_.find(word + std::string(":default"));
    ++it;
    pos = (it->first).find(":");
    while ((it->first).substr(0, pos) == word) {
      lkp->altprons.push_back(it->second);
      ++it;
      if (it == lookup_.end()) break;
      pos = (it->first).find(":");
    }
    return true;
  } else {
    return false;
  }
}

}  // namespace kaldi

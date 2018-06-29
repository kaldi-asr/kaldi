// idlaktxp/txpabbrev.cc

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

// This file controls pre normalisation abbreviation expansion

#include "idlaktxp/txpabbrev.h"

namespace kaldi {

TxpAbbrev::~TxpAbbrev() {
  int32 i;
  for (i = 0; i < abbreviations_.size(); i++) {
    delete(abbreviations_[i]);
  }
}
TxpAbbrevInfo * TxpAbbrev::LookupAbbrev(const char * tk,
                                        const char * prepunc,
                                        const char * pstpunc) {
  for (AbbrevVector::const_iterator it = abbreviations_.begin();
       it != abbreviations_.end();
       ++it) {
    if (!strcmp(tk, (*it)->token.c_str())) {
      if (CheckPrePunc(prepunc, (*it)) != NO_INDEX &&
          (CheckPstPunc(pstpunc, (*it)) != NO_INDEX))
        return (*it);
    }
  }
  return NULL;
}

TxpAbbrevInfo * TxpAbbrev::LookupAbbrev(const char * tk) {
  for (AbbrevVector::const_iterator it = abbreviations_.begin();
       it != abbreviations_.end();
       ++it) {
    if (!strcmp(tk, (*it)->token.c_str())) {
      if ((*it)->prepunc.empty() && (*it)->pstpunc.empty())
        return (*it);
    }
  }
  return NULL;
}

int32 TxpAbbrev::CheckPrePunc(const char * prepunc, TxpAbbrevInfo * abb) {
  // no prepunc specified
  if (abb->prepunc.empty()) return 0;
  // prepunc matches
  if (!strcmp(prepunc + strlen(prepunc) - abb->prepunc.size(),
              abb->prepunc.c_str()))
    return abb->prepunc.size();
  // prepunc does not match
  return NO_INDEX;
}

int32 TxpAbbrev::CheckPstPunc(const char * pstpunc, TxpAbbrevInfo * abb) {
  // no pstpunc specified
  if (abb->pstpunc.empty()) return 0;
  // pstpunc matches
  if (!strncmp(pstpunc, abb->pstpunc.c_str(), abb->pstpunc.size()))
    return abb->pstpunc.size();
  // prepunc does not match
  return NO_INDEX;
}

void TxpAbbrevInfo::AddExpansionToken(const char * expansion,
                                      int expansion_len,
                                      const char * lexentry) {
  lexentries.push_back(std::string(lexentry));
  expansions.push_back(std::string(expansion, expansion_len));
}

bool TxpAbbrev::Parse(const std::string &tpdb) {
  bool r;
  r = TxpXmlData::Parse(tpdb);
  if (!r)
    KALDI_WARN << "Error reading abbreviation file: " << tpdb;
  return r;
}

void TxpAbbrev::StartElement(const char* name, const char** atts) {
  if (!strcmp(name, "lex")) {
    SetAttribute("entry", atts, &current_lexentry_);
  } else if (!strcmp(name, "abb")) {
    // create abbinfo
    current_abbrev_ = new TxpAbbrevInfo();
    SetAttribute("pstpunc", atts, &(current_abbrev_->pstpunc));
    SetAttribute("token", atts, &(current_abbrev_->token));
    SetAttribute("prepunc", atts, &(current_abbrev_->prepunc));
  }
}

void TxpAbbrev::EndElement(const char* name) {
  if (!strcmp(name, "lex")) {
    current_lexentry_ = "";
  } else if (!strcmp(name, "abb")) {
    abbreviations_.push_back(current_abbrev_);
  }
}

void TxpAbbrev::CharHandler(const char* data, int32 len) {
  const char * p = NULL, * word = NULL;
  int32 i = 0, wordlen = 0;
  p = data;
  if (!p) return;
  for (i = 0; i < len; p++, i++) {
    if (!*p) break;
    if (*p == ' ' || *p == '\n') {
      if (word) {
        current_abbrev_->AddExpansionToken(word, wordlen,
                                           current_lexentry_.c_str());
        word = NULL;
        wordlen = 0;
      }
    } else {
      if (!word) word = p;
      wordlen++;
    }
  }
  if (word) current_abbrev_->AddExpansionToken(word, wordlen,
                                               current_lexentry_.c_str());
}

}  // namespace kaldi

// idlaktxp/txppos.cc

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

// This is a very simple part of speech tagger. It could be improved by taking
// note of breaks caused by punctuation

#include "idlaktxp/txppos.h"

namespace kaldi {

bool TxpPos::Parse(const std::string &tpdb) {
  bool r;
  r = TxpXmlData::Parse(tpdb);
  if (!r)
    KALDI_WARN << "Error reading part of speech tagger file: " << tpdb;
  return r;
}

const char* TxpPos::GetPos(const char* ptag, const char* word) {
  RgxVector::iterator it;
  LookupMap::iterator lkp;
  const char* current_tag;
  current_tag =  most_common_.c_str();
  // Regex tagger
  for (it = rgxtagger_.begin(); it != rgxtagger_.end(); ++it) {
    if ((*it).pos == TXPPOSRGX_POS_WORD &&
        !strcmp((*it).pattern.c_str(), word)) {
      current_tag = (*it).tag.c_str();
      break;
    } else if ((*it).pos == TXPPOSRGX_POS_PREFIX &&
             !strncmp((*it).pattern.c_str(), word, (*it).pattern.length())) {
      current_tag = (*it).tag.c_str();
      break;
    } else if ((*it).pos == TXPPOSRGX_POS_SUFFIX &&
             (*it).pattern.length() < strlen(word) &&
             !strcmp((*it).pattern.c_str(),
                     word + (strlen(word) - (*it).pattern.length()))) {
      current_tag = (*it).tag.c_str();
      break;
    }
  }
  // Pattern tagger: unigram
  lkp = patterntagger_.find(std::string(word));
  if (lkp != patterntagger_.end()) {
    current_tag = (lkp->second).c_str();
  }
  // bigram
  if (*ptag) {
    lkp = patterntagger_.find(std::string(ptag) + "_" + std::string(word));
    if (lkp != patterntagger_.end()) {
      current_tag = lkp->second.c_str();
    }
  }
  return current_tag;
}

void TxpPos::StartElement(const char* name, const char** atts) {
  std::string pos;
  std::string tag;
  std::string ptag;
  TxpPosRgx rgx;
  if (!strcmp(name, "most_common")) {
    SetAttribute("tag", atts, &most_common_);
  }
  if (!strcmp(name, "r")) {
    SetAttribute("pos", atts, &pos);
    if (pos == "WORD") rgx.pos = TXPPOSRGX_POS_WORD;
    else if (pos == "PREFIX") rgx.pos = TXPPOSRGX_POS_PREFIX;
    else if (pos == "SUFFIX") rgx.pos = TXPPOSRGX_POS_SUFFIX;
    SetAttribute("pat", atts, &(rgx.pattern));
    SetAttribute("tag", atts, &(rgx.tag));
    rgxtagger_.push_back(rgx);
  } else if (!strcmp(name, "w")) {
    SetAttribute("name", atts, &word_);
  } else if (!strcmp(name, "p")) {
    SetAttribute("tag", atts, &tag);
    SetAttribute("ptag", atts, &ptag);
    if (ptag.empty())
      patterntagger_.insert(LookupItem(word_, tag));
    else
      patterntagger_.insert(LookupItem(word_, tag + "_" + word_));
  }
}

bool TxpPosSet::Parse(const std::string &tpdb) {
  bool r;
  r = TxpXmlData::Parse(tpdb);
  if (!r)
    KALDI_WARN << "Error reading part of speech set file: " << tpdb;
  return r;
}

void TxpPosSet::StartElement(const char* name, const char** atts) {
  std::string tagname;
  std::string set;
  if (!strcmp(name, "tag")) {
    SetAttribute("name", atts, &tagname);
    SetAttribute("set", atts, &set);
    posset_.insert(LookupItem(tagname, set));
  }
}

const char* TxpPosSet::GetPosSet(const char* pos) {
  LookupMap::iterator lkp;
  lkp = posset_.find(std::string(pos));
  if (lkp != posset_.end()) {
    return (lkp->second).c_str();
  }
  return NULL;
}

}  // namespace kaldi

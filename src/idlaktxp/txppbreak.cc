// idlaktxp/txppbreak.cc

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

// This file hold information that relates punctuation to pause insertion

#include "idlaktxp/txppbreak.h"

namespace kaldi {

bool TxpPbreak::Parse(const std::string &tpdb) {
  bool r;
  r = TxpXmlData::Parse(tpdb);
  if (!r)
    KALDI_WARN << "Error reading phrase break file: " << tpdb;
  return r;
}

bool TxpPbreak::GetPbreak(const char* punc,
                          enum TXPPBREAK_POS pos,
                          TxpPbreakInfo &info) {
  PbreakMap::iterator lkp;
  const char* p;
  TxpUtf8 utf8;
  int32 clen;
  bool found = false;
  p = punc;
  while (*p) {
    clen = utf8.Clen(p);
    if (pos == TXPPBREAK_POS_PRE) {
      lkp = pre_pbreak_.find(std::string(p, clen));
      if (lkp != pre_pbreak_.end()) found = true;
    } else {
      lkp = pst_pbreak_.find(std::string(p, clen));
      if (lkp != pst_pbreak_.end()) found = true;
    }
    if (found && lkp->second.time > info.time) info.time = lkp->second.time;
    if (found && lkp->second.type > info.type) info.type = lkp->second.type;
    p += clen;
  }
  return found;
}

void TxpPbreak::GetWhitespaceBreaks(const char* ws, int32 col,
                                    bool hzone,
                                    int32 hzone_start, int32 hzone_end,
                                    bool* newline, bool* newline2) {
  const char* p = ws;
  TxpUtf8 utf8;
  int32 clen;
  int32 nls = 0;
  bool wspstnl = false;
  *newline = false;
  *newline2 = false;
  while (*p) {
    clen = utf8.Clen(p);
    if (!strncmp(p, "\r", clen)) {
      if (!strncmp(p + 1, "\n", 1)) p++;
      nls++;
    } else if (!strncmp(p, "\n", clen)) {
      if (!strncmp(p + 1, "\r", 1)) p++;
      nls++;
    }
    if (nls && !strncmp(p, " ", 1)) wspstnl = true;
    p += clen;
  }
  if (nls > 1) *newline2 = true;
  if (nls == 1) {
    if (!wspstnl && hzone && col > hzone_start && col < hzone_end)
      *newline = false;
    else
      *newline = true;
  }
}

const TxpPbreakInfo* TxpPbreak::GetPbreakPst(const char* punc) {
  PbreakMap::iterator lkp;
  lkp = pst_pbreak_.find(std::string(punc));
  if (lkp !=  pst_pbreak_.end()) return &(lkp->second);
  return NULL;
}

const TxpPbreakInfo* TxpPbreak::GetPbreakPre(const char* punc) {
  PbreakMap::iterator lkp;
  lkp = pre_pbreak_.find(std::string(punc));
  if (lkp !=  pre_pbreak_.end()) return &(lkp->second);
  return NULL;
}

void TxpPbreak::StartElement(const char* name, const char** atts) {
  std::string punc;
  std::string pos;
  std::string time;
  std::string type;
  TxpPbreakInfo pbreak;
  if (!strcmp(name, "break")) {
    SetAttribute("name", atts, &punc);
    SetAttribute("pos", atts, &pos);
    SetAttribute("time", atts, &time);
    SetAttribute("type", atts, &type);
    pbreak.time = atof(time.c_str());
    pbreak.type = atoi(type.c_str());
    if (pos == "pstpunc") {
      pst_pbreak_.insert(PbreakItem(punc, pbreak));
    } else {
      pre_pbreak_.insert(PbreakItem(punc, pbreak));
    }
  }
}


}  // namespace kaldi

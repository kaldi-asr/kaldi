// idlaktxp/txpsylmax.cc

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

// This file contains functions which carry out maximal onset
// syllabification

#include "idlaktxp/txpsylmax.h"

namespace kaldi {

bool TxpSylmax::Parse(const std::string &tpdb) {
  bool r;
  r = TxpXmlData::Parse(tpdb);
  if (!r)
    KALDI_WARN << "Error reading phrase break file: " << tpdb;
  return r;
}

void TxpSylmax::Maxonset(PhoneVector *pronptr) {
  std::string pat;
  int32 nlen, olen, pos, i;
  PhoneVector &pron = *pronptr;
  StringSet::iterator it;
  // move forwards trying to find the nucleus
  for (pos = 0; pos < pron.size(); pos++) {
    // try largest nuclei first
    nlen = FindNucleus(pron, pos);
    if (nlen) {
      olen = FindOnset(pron, pos);
      if (olen) {
        if (pos - olen - 1 >= 0) pron[pos - olen - 1].codab = true;
        for (i = pos - olen; i < pos; i++)
          pron[i].type = TXPSYLMAX_TYPE_ONSET;
      }
      if (pos > 0) pron[pos - 1].codab = true;
      // mark nucleus
      for (i = pos; i < pos + nlen; i++)
        pron[i].type = TXPSYLMAX_TYPE_NUCLEUS;
      pron[pos + nlen - 1].codab = true;
      if (nlen > 1) pos += nlen - 1;
    }
  }
  AddSylBound(&pron);
}

void TxpSylmax::Writespron(PhoneVector *pronptr, std::string *sylpron) {
  int32 pos, newwrdb = -1, i;
  const char* bound;
  PhoneVector &pron = *pronptr;
  sylpron->clear();
  // if wrdb is at a cross word point move it back to the end of
  // the previous syllable
  for (pos = 0; pos < pron.size(); pos++) {
    if (pron[pos].wrdb == true) {
      if (pron[pos].cross_word == false) {
        newwrdb = pos;
        break;
      } else {
        // move word boundary back
        newwrdb = pos;
        while (newwrdb >=0 && !pron[newwrdb].sylb) {
          newwrdb--;
        }
        if (newwrdb >=0) pron[newwrdb].wrdb = true;
        pron[pos].wrdb = false;
        break;
      }
    }
  }
  // copy coda of previous word if held back
  // for cross word syllabification
  // if newwrdb == -1 there is no pron for this token anymore
  for (i = 0; i <= newwrdb; i++) {
    if (pron[i].sylb) bound = "|";
    else if (pron[i].codab)
      bound = "+";
    else
      bound = "_";
    *sylpron += pron[i].name + pron[i].stress + bound;
  }
  // remove processed phones from the front of the array
  pron.erase(pron.begin(), pron.begin() + newwrdb + 1);
}

// returns size of coda if cross word
int32 TxpSylmax::AddSylBound(PhoneVector *pronptr) {
  PhoneVector &pron = *pronptr;
  enum TXPSYLMAX_TYPE ltype = pron[0].type;
  bool codab = pron[0].codab;
  int i, coda;
  for (i = 1; i < pron.size(); i++) {
    if (codab &&
        !((ltype == TXPSYLMAX_TYPE_ONSET &&
           pron[i].type == TXPSYLMAX_TYPE_NUCLEUS) ||
          (ltype == TXPSYLMAX_TYPE_NUCLEUS &&
           pron[i].type == TXPSYLMAX_TYPE_CODA)))
      pron[i - 1].sylb = true;
    ltype = pron[i].type;
    codab = pron[i].codab;
  }
  // Returns position of the start of the last coda in the word
  // if cross word syllabification
  if (pron[pron.size() - 1].cross_word) {
    for (i = i - 1; i >= 0; i--) {
      if (pron[i].type != TXPSYLMAX_TYPE_CODA) {
        coda = i + 1;
        break;
      } else if (i == 0 && pron[i].type == TXPSYLMAX_TYPE_CODA) {
        coda = 1;
      }
    }
    return coda;
  } else {
    pron[pron.size() - 1].sylb = true;
  }
  return 0;
}

int32 TxpSylmax::FindNucleus(const PhoneVector &pron, int32 pos) {
  int len;
  std::string pat;
  StringSet::iterator it;
  for (len = max_nucleus_; len > 0; len--) {
    if (GetPhoneNucleusPattern(pron, pos, len, stress_, &pat)) {
      it = nuclei_.find(pat);
      if (it != nuclei_.end())
        return len;
    }
  }
  return 0;
}

int32 TxpSylmax::FindOnset(const PhoneVector &pron, int32 pos) {
  int len;
  std::string pat;
  StringSet::iterator it;
  for (len = max_onset_; len > 0; len--) {
    if (GetPhoneOnsetPattern(pron, pos, len, &pat)) {
      it = onsets_.find(pat);
      if (it != onsets_.end())
        return len;
    }
  }
  return 0;
}

// convert pronunciation into a vector
// TODO(matthew): generalise to read in sylpron
int32 TxpSylmax::GetPhoneVector(const char* pron,
                                PhoneVector *phonevectorptr) {
  PhoneVector &phonevector = *phonevectorptr;
  std::vector<std::string> array;
  const char *p, *s;
  int32 i, len;
  p = pron;
  i = 0;
  if (!p) return 0;
  while (*p) {
    s = strstr(p, " ");
    if (s) {
      len = static_cast<int32>(s - p);
      InsertPhone(phonevectorptr, p, len, false);
      p = s;
      p++;
    } else {
      len = strlen(p);
      InsertPhone(phonevectorptr, p, len, true);
      p += len;
      }
    i++;
  }
  return phonevector.size();
}

// TODO(matthew): generalise to read in sylpron
void TxpSylmax::InsertPhone(PhoneVector *phonevectorptr,
                            const char *p, int32 len, bool wrdb) {
  PhoneVector &phonevector = *phonevectorptr;
  TxpSylItem item;
  item.Clear();
  item.wrdb = wrdb;
  if (p[len - 1] >= '0' && p[len - 1] <= '9') {
    item.name = std::string(p, len -1);
    item.stress = std::string(p + len - 1, 1);
  } else if (p[len - 1] == '+') {
    item.name = std::string(p, len -1);
        item.cross_word = true;
  } else {
    item.name = std::string(p, len);
  }
  phonevector.push_back(item);
}

bool TxpSylmax::GetPhoneNucleusPattern(const PhoneVector &pron,
                                       int32 pos, int32 len,
                                       bool with_stress, std::string *pat) {
  PhoneVector::const_iterator it;
  int32 i;
  pat->clear();
  for (i = 0, it = pron.begin() + pos; i < len; ++it, ++i) {
    if (it == pron.end()) {
      pat->clear();
      return false;
    }
    if (!pat->empty()) *pat += " ";
    // Remove stress marks if required (forward only for nucleus)
    if (!with_stress)
      *pat += it->name;
    else
      *pat += it->name + it->stress;
  }
  return true;
}

bool TxpSylmax::GetPhoneOnsetPattern(const PhoneVector &pron,
                                     int32 pos, int32 len,
                                     std::string *pat) {
  PhoneVector::const_reverse_iterator it;
  int32 i;
  pat->clear();
  for (i = 0, it = pron.rbegin() + (pron.size() - pos); i < len; ++it, ++i) {
    if (it == pron.rend()) {
      pat->clear();
      return false;
    }
    // word boundary blocks onset pattern
    if (it->wrdb && !it->cross_word) {
      pat->clear();
      return false;
    }
    if (!pat->empty()) *pat = " " + *pat;
    *pat = it->name + *pat;
  }
  return true;
}

bool TxpSylmax::IsSyllabic(const char *phone) {
  StringSet::iterator it = syllabic_.find(std::string(phone));
  return (it != onsets_.end());
}


void TxpSylmax::StartElement(const char *name, const char ** atts) {
  std::string item;
  PhoneVector pv;
  int32 size;
  if (!strcmp("nuclei", name)) {
    SetAttribute("stress", atts, &item);
    stress_ = false;
    if (item == "true" || item == "True" || item == "TRUE")
      stress_ = true;
  } else if (!strcmp("s", name)) {
    SetAttribute("name", atts, &item);
    syllabic_.insert(item);
  } else if (!strcmp("o", name)) {
    SetAttribute("pat", atts, &item);
    onsets_.insert(item);
    size = GetPhoneVector(item.c_str(), &pv);
    if (size > max_onset_) max_onset_ = size;
  } else if (!strcmp("n", name)) {
    SetAttribute("pat", atts, &item);
    nuclei_.insert(item);
    size = GetPhoneVector(item.c_str(), &pv);
    if (size > max_nucleus_) max_nucleus_ = size;
  }
}

}  // namespace kaldi

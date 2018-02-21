// idlaktxp/txplts.h

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

#include "idlaktxp/txplts.h"
#include "idlaktxp/txputf8.h"

namespace kaldi {

static int32 pos2int(const std::string &pos);
static bool ApplyQuestion(const char* word, const int32 pos, const char* utfchar);
static const char* ApplyTree(const TxpLtsTree &tree,
                              const char* word,
                              const int32 pos);

void TxpLts::StartElement(const char* name, const char** atts) {
  std::string terminal;
  LtsMap::iterator it;
  if (!strcmp(name, "tree")) {
    SetAttribute("ltr", atts, &ltr_);
    SetAttribute("terminal", atts, &terminal);
    TxpLtsTree tree;
    tree.root = atoi(terminal.c_str());
    ltslkp_.insert(LtsItem(ltr_, tree));
  } else if (!strcmp(name, "node")) {
    TxpLtsNode node;
    std::string pos;
    std::string yes;
    std::string no;
    SetAttribute("pos", atts, &pos);
    SetAttribute("posval", atts, &(node.posval));
    SetAttribute("yes", atts, &yes);
    SetAttribute("no", atts, &no);
    SetAttribute("val", atts, &(node.val));
    if (!pos.empty()) {
      // non terminal node
      node.yes = atoi(yes.c_str());
      node.no = atoi(no.c_str());
      node.pos = pos2int(pos);
    }
    it = ltslkp_.find(ltr_);
    (it->second).nodes.push_back(node);
  }
}

int TxpLts::GetPron(const std::string &word, TxpLexiconLkp* lkp) {
  TxpUtf8 utf8;
  int32 clen, pos = 0;
  const char *p, *phone, *stress;
  std::string ltr;
  LtsMap::iterator it;
  TxpLtsTree stress_tree;

  // get stress lookup
  it = ltslkp_.find(std::string("0"));
  if (it == ltslkp_.end()) {
      KALDI_WARN << "No stress lookup tree: name='0'";
  } else {
    stress_tree = it->second;
  }
  // process each letter
  p = word.c_str();
  while (*p) {
    clen = utf8.Clen(p);
    ltr = std::string(p, clen);
    it = ltslkp_.find(ltr);
    if (it != ltslkp_.end()) {
      phone = ApplyTree(it->second, word.c_str(), pos);
      const char * orig = phone;
      // If not a null result
      if (strcmp("0", phone)) {
        const char *sep = strchr(phone, '_');
        if (!lkp->pron.empty()) lkp->pron += " ";
        // Split the phone on "_" as LTS can return multiphone
        while (sep) {
          lkp->pron += std::string(phone, sep - phone);
          lkp->pron += " ";
          phone = sep + 1;
          sep = strchr(phone, '_');
        }

        // Check is syllabic and if so get stress
        if (phone[strlen(phone) - 1] == '0') {
          stress = ApplyTree(stress_tree, word.c_str(), pos);
          lkp->pron += std::string(phone, strlen(phone) - 1);
          lkp->pron += stress;
	      free((void *)stress);
        } else {
          lkp->pron += phone;
        }
      }
      free((void *)orig);
    } else {
      // HACKY: Remove annoying warning for English possessive form
      if (ltr.compare("'"))
        KALDI_WARN << "Letter not in LTS tree: " << ltr;
    }
    p += clen;
    pos++;
  }
  lkp->lts = true;
  return true;
}

static int32 pos2int(const std::string &pos) {
  int32 loc = 0, idx = 0;
  while ((loc = pos.find("n.", loc)) != std::string::npos) {
    idx += 1;
    loc += 2;
  }
  loc = 0;
  while ((loc = pos.find("p.", loc)) != std::string::npos) {
    idx -= 1;
    loc += 2;
  }
  return idx;
}

static bool ApplyQuestion(const char* word, const int32 pos, const char* utfchar) {
  TxpUtf8 utf8;
  int32 clen = 0, idx = 0;
  const char* p = word;
  // position before word start
  if (pos < 0 && !strcmp(utfchar, "#")) return true;
  while (*p) {
    clen = utf8.Clen(p);
    if (idx == pos) break;
    p += clen;
    idx++;
  }
  // position after word end
  if (*p &&  !strcmp(utfchar, "#")) return true;
  // matches utf8char at pos
  if (!strncmp(p, utfchar, clen)) return true;
  return false;
}

static const char* ApplyTree(const TxpLtsTree &tree,
                              const char* word,
			     const int32 pos) {
  int32 currentnode = tree.root;
  TxpLtsNode node;
  while (1) {
    node = tree.nodes[currentnode];
    // check for non terminal node
    if (!node.val.empty()) {
      break;
    } else {
      if (ApplyQuestion(word, pos + node.pos, node.posval.c_str())) {
        currentnode = node.yes;
      } else {
        currentnode = node.no;
      }
    }
  }
  return strdup(node.val.c_str());
}

}  // namespace kaldi

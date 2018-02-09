// idlaktxp/mod-tokenise.cc

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

#include "idlaktxp/mod-tokenise.h"

namespace kaldi {

TxpTokenise::TxpTokenise() : TxpModule("tokenise") {}

bool TxpTokenise::Init(const TxpParseOptions &opts) {
  opts_ = &opts;
  tpdb_ = opts.GetTpdb();
  nrules_.Init(opts, std::string(opts_->GetValue(GetName().c_str(), "arch")));
  return nrules_.Parse(tpdb_);
}

bool TxpTokenise::Process(pugi::xml_document* input) {
  const char* p;
  int32 n = 0;
  int32 offset, col = 0;
  std::string token, wspace, tmp;
  pugi::xml_node tkroot, tk, ws, ntxt;
  // all text nodes are tokenised
  pugi::xpath_node_set text =
      input->document_element().select_nodes("//text()");
  // iterate through text nodes
  text.sort();
  for (pugi::xpath_node_set::const_iterator it = text.begin();
       it != text.end();
       ++it) {
    pugi::xml_node node = (*it).node();
    p = node.value();
    while (*p) {
      // break off tokens and spacing and add as an element
      p = nrules_.ConsumeToken(p, &token, &wspace);
      if (token.length()) {
        col += token.length();
        tkroot = tk = node.parent().insert_child_before("tk", node);
        ntxt = tk.append_child(pugi::node_pcdata);
        ntxt.set_value(token.c_str());
        n += 1;
        nrules_.ReplaceUtf8Punc(token, &tmp);
        SetPuncCaseInfo(&tmp, &tk);
      }
      if (wspace.length()) {
        if (token.length())
          ws = tkroot.append_child("ws");
        else
          ws = node.parent().insert_child_before("ws", node);
        ntxt = ws.append_child(pugi::node_pcdata);
        ntxt.set_value(wspace.c_str());
        // Set column position for carriage returns and newlines
        // Used to decide on soft hyphenation and phrasing
        offset = wspace.find("\n");
        if (offset == std::string::npos) offset = wspace.find("\r");
        if (offset != std::string::npos) {
          std::stringstream sstream;
          sstream << col + offset;
          ws.append_attribute("col").set_value(sstream.str().c_str());
          col = wspace.length() - offset;
        } else {
          col += wspace.length();
        }
      }
    }
    // Delete original text node
    node.parent().remove_child(node);
  }
  return true;
}

// Analyses the characters and sets flags giving case, foriegn character info
int32 TxpTokenise::SetPuncCaseInfo(std::string* tkin, pugi::xml_node* tk) {
  const char* p;
  TxpCaseInfo caseinfo;
  pugi::xml_node node;
  int32 n = 0;
  std::string token;
  std::string prepunc;
  std::string pstpunc;
  p = tkin->c_str();
  while (*p) {
    p = nrules_.ConsumePunc(p, &prepunc, &token, &pstpunc);
    if (n) {
      *tk = tk->parent().insert_child_after("tk", *tk);
    }
    if (prepunc.length()) {
      tk->append_attribute("prepunc");
      tk->attribute("prepunc").set_value(prepunc.c_str());
    }
    if (token.length()) {
      tk->append_attribute("norm");
      nrules_.NormCaseCharacter(&token, caseinfo);
      tk->attribute("norm").set_value(token.c_str());
    }
    if (pstpunc.length()) {
      tk->append_attribute("pstpunc");
      tk->attribute("pstpunc").set_value(pstpunc.c_str());
    }
    if (caseinfo.lowercase) tk->append_attribute("lc").set_value("true");
    if (caseinfo.uppercase) tk->append_attribute("uc").set_value("true");
    if (caseinfo.foreign) tk->append_attribute("foreign").set_value("true");
    if (caseinfo.symbols) tk->append_attribute("symbols").set_value("true");
    if (caseinfo.capitalised) tk->append_attribute("caps").set_value("true");
    n++;
  }
  return true;
}

}  // namespace kaldi

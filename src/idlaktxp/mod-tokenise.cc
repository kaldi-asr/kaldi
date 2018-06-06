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
  abbrev_.Init(opts, std::string(GetOptValue("arch")));
  return nrules_.Parse(tpdb_) && abbrev_.Parse(tpdb_);
}

bool TxpTokenise::Process(pugi::xml_document* input) {
  const char* p;
  int32 n = 0;
  int32 offset, col = 0;
  std::string token, wspace, tmp;
  pugi::xml_node tkroot, tk, tkcopy, ws, ntxt, lex;
  TxpAbbrevInfo * abbrev_info;
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
        /// check for full token matches without partial punctuation
        /// i.e. :-) but not (US) 
        abbrev_info = abbrev_.LookupAbbrev(token.c_str());
        if (abbrev_info) {
          for(int32 i = 0; i < abbrev_info->expansions.size(); i++) {
            if (!i) {
              tk.append_attribute("norm");
              tk.attribute("norm").set_value(abbrev_info->expansions[0].c_str());
              if (!abbrev_info->lexentries[0].empty()) {
                lex = tk.parent().insert_child_after("lex", tk);
                lex.append_attribute("type").set_value(abbrev_info->lexentries[0].c_str());
                tkcopy = lex.append_copy(tk);
                tk.parent().remove_child(tk);
                tk = tkcopy;
              }
            }
            else {
              if  (!abbrev_info->lexentries[i].empty()) {
                lex = tk.parent().insert_child_after("lex", tk);
                lex.append_attribute("type").set_value(abbrev_info->lexentries[i].c_str());
                tk = lex.append_child("tk");
                tk.append_attribute("norm").set_value(abbrev_info->expansions[i].c_str());
                tk = lex; //so we add new tokens after the lex tag not inside it
              }
              else {
                tk = tk.parent().insert_child_after("tk", tk);
                tk.append_attribute("norm").set_value(abbrev_info->expansions[i].c_str());
              }
            }
          }
        }
        // if no full match of an abbreviation unpack punctuation
        else {
          SetPuncCaseInfo(&tmp, &tk);
        }
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
  pugi::xml_node node, lex, tkcopy;
  int32 n = 0;
  std::string token;
  std::string prepunc;
  std::string pstpunc;
  TxpAbbrevInfo * abbrev_info;
  p = tkin->c_str();
  while (*p) {
    p = nrules_.ConsumePunc(p, &prepunc, &token, &pstpunc);
    if (n) {
      *tk = tk->parent().insert_child_after("tk", *tk);
    }
    // check to see if there is an abbreviation that matches the token and
    // completely or partially matches the punctuation
    abbrev_info = abbrev_.LookupAbbrev(token.c_str(), prepunc.c_str(), pstpunc.c_str());
    if (abbrev_info) {
      // trim punctuation as appropriate
      prepunc = prepunc.substr(0, prepunc.size() - abbrev_.CheckPrePunc(prepunc.c_str(), abbrev_info));
      pstpunc = pstpunc.substr(abbrev_.CheckPstPunc(pstpunc.c_str(), abbrev_info),
                                                    pstpunc.size() - abbrev_.CheckPstPunc(pstpunc.c_str(),
                                                                                          abbrev_info));
      for(int32 i = 0; i < abbrev_info->expansions.size(); i++) {
        if (!i) {
          tk->append_attribute("norm");
          tk->attribute("norm").set_value(abbrev_info->expansions[0].c_str());
          if (!abbrev_info->lexentries[0].empty()) {
            lex = tk->parent().insert_child_after("lex", *tk);
            lex.append_attribute("type").set_value(abbrev_info->lexentries[0].c_str());
            tkcopy = lex.append_copy(*tk);
            tk->parent().remove_child(*tk);
            *tk = tkcopy;
          }
          if (prepunc.length()) {
            tk->append_attribute("prepunc");
            tk->attribute("prepunc").set_value(prepunc.c_str());
          }
        }
        else {
          if  (!abbrev_info->lexentries[i].empty()) {
            lex = tk->parent().insert_child_after("lex", *tk);
            lex.append_attribute("type").set_value(abbrev_info->lexentries[i].c_str());
            *tk = lex.append_child("tk");
            tk->append_attribute("norm").set_value(abbrev_info->expansions[i].c_str());
            *tk = lex;
          }
          else {
            *tk = tk->parent().insert_child_after("tk", *tk);
            tk->append_attribute("norm").set_value(abbrev_info->expansions[i].c_str());
          }
        }
      }
      if (pstpunc.length()) {
        tk->append_attribute("pstpunc");
        tk->attribute("pstpunc").set_value(pstpunc.c_str());
      }
    }
    else {
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
    }
    n++;
  }
  return true;
}

}  // namespace kaldi

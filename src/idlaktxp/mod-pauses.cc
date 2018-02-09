// idlaktxp/mod-pauses.cc

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

#include "idlaktxp/txpmodule.h"
#include "idlaktxp/mod-pauses.h"

namespace kaldi {

static void _insert_break_after(pugi::xml_node* tk,
                                const TxpPbreakInfo* pbreak);
static void _insert_break_before(pugi::xml_node* tk,
                                 const TxpPbreakInfo* pbreak);

TxpPauses::TxpPauses() : TxpModule("pauses") {}

bool TxpPauses::Init(const TxpParseOptions &opts) {
  opts_ = &opts;
  tpdb_ = opts.GetTpdb();
  pbreak_.Init(opts, std::string(GetOptValue("arch")));
  hzone_ = GetOptValueBool("hzone");
  hzone_start_ = atoi(GetOptValue("hzone-start"));
  hzone_end_ = atoi(GetOptValue("hzone-end"));
  return pbreak_.Parse(tpdb_);
}

TxpPauses::~TxpPauses() {
}

// If fileids are present in the document it breaks the
// document up by them
bool TxpPauses::Process(pugi::xml_document* input) {
  pugi::xpath_node_set files;
  files = input->document_element().select_nodes("//fileid");
  if (files.size()) {
    for (pugi::xpath_node_set::const_iterator it = files.begin();
         it != files.end();
         ++it) {
      pugi::xml_node file = (*it).node();
      ProcessFile(&file);
    }
  } else {
    pugi::xml_node document_element = input->document_element();
    ProcessFile(&document_element);
  }
  return true;
}

// Adds breaks as required and ensures all documents/fileids
// have a break initial and final
bool TxpPauses::ProcessFile(pugi::xml_node* file) {
  pugi::xml_node *breakitem = NULL, *ptk = NULL, pretk;
  const TxpPbreakInfo* pbreak;
  TxpPbreakInfo pbreakpunc;
  bool prebreak, pstbreak, newline = false, newline2 = false;
  pugi::xpath_node_set files;
  pugi::xpath_node_set tks =
      file->select_nodes(".//tk|.//break|.//ws");
  tks.sort();
  for (pugi::xpath_node_set::const_iterator it = tks.begin();
       it != tks.end();
       ++it) {
    pugi::xml_node tk = (*it).node();
    if (!strcmp(tk.name(), "break")) {
      if (!tk.attribute("type")) tk.append_attribute("type");
      if (!tk.attribute("time")) tk.append_attribute("time");
      if (!tk.attribute("strength").empty()) {
        pbreak = pbreak_.GetPbreakPst(tk.attribute("strength").value());
        if (pbreak) {
          tk.attribute("type").set_value(pbreak->type);
          tk.attribute("time").set_value(pbreak->time);
        }
      }
      // break tags override punctuation so keep a record of what happened
      // before tk
      if (!*tk.attribute("type").value()) {
        tk.attribute("type").set_value(pbreak_.get_default_type());
      }
      if (!*tk.attribute("time").value()) {
        tk.attribute("time").set_value(pbreak_.get_default_time());
      }
      breakitem = &tk;
      // tk.print(std::cout);
    } else if (!strcmp(tk.name(), "ws")) {
      pbreak_.GetWhitespaceBreaks(tk.first_child().value(),
                                  tk.attribute("col").as_int(0),
                                  hzone_, hzone_start_, hzone_end_,
                                  &newline, &newline2);
    } else if (!strcmp(tk.name(), "tk")) {
      // std::cout << "!" << tk.attribute("norm").value() << "\n";
      // if (ptk) std::cout << "P:" << ptk->attribute("norm").value() << " - ";
      // std::cout << "L:" << tk.attribute("norm").value() << std::endl;
      pbreakpunc.Clear();
      // Only add breaks if no break is present
      if (!breakitem) {
        // first token
        if (!ptk) {
          if (pbreak_.GetPbreak(tk.attribute("prepunc").value(),
                                TXPPBREAK_POS_PRE, pbreakpunc)) {
            pbreak = &pbreakpunc;
          } else {
            // Insert default document start break
            pbreak = pbreak_.GetPbreakPre("DEFAULT");
          }
          _insert_break_before(&tk, pbreak);
        } else {
          // Between tokens
          // break caused by punctuation
          // std::cout << "X:" << tk.attribute("prepunc").value() << std::endl;
          pstbreak = pbreak_.GetPbreak(ptk->attribute("pstpunc").value(),
                                       TXPPBREAK_POS_PST, pbreakpunc);
          prebreak = pbreak_.GetPbreak(tk.attribute("prepunc").value(),
                                       TXPPBREAK_POS_PRE, pbreakpunc);
          pbreak = &pbreakpunc;
          if (!prebreak && !pstbreak) {
            // Break caused by whitespace
            if (newline2) {
            pbreak = pbreak_.GetPbreakPst("newlineX2");
            } else if (newline) {
              pbreak = pbreak_.GetPbreakPst("newlineX2");
            } else {
              pbreak = NULL;
            }
          }
          _insert_break_after(ptk, pbreak);
        }
      }
      pretk = (*it).node();
      ptk = &pretk;
      breakitem = NULL;
    }
  }
  // last item
  if (!breakitem && ptk) {
    if (pbreak_.GetPbreak(ptk->attribute("pstpunc").value(),
                          TXPPBREAK_POS_PST, pbreakpunc)) {
      pbreak = &pbreakpunc;
    } else {
      // Insert default document end break
      pbreak = pbreak_.GetPbreakPst("DEFAULT");
    }
    _insert_break_after(ptk, pbreak);
  }
  return true;
}

static void _insert_break_after(pugi::xml_node* tk,
                                const TxpPbreakInfo* pbreak) {
  pugi::xml_node breakitem;
  if (!pbreak) return;
  // std::cout << tk.attribute("norm").value() << std::endl;
  breakitem = tk->parent().insert_child_after("break", *tk);
  breakitem.append_attribute("type").set_value(pbreak->type);
  breakitem.append_attribute("time").set_value(pbreak->time);
}

static void _insert_break_before(pugi::xml_node* tk,
                                 const TxpPbreakInfo* pbreak) {
  pugi::xml_node breakitem;
  if (!pbreak) return;
  // std::cout << tk.attribute("norm").value() << std::endl;
  breakitem = tk->parent().insert_child_before("break", *tk);
  breakitem.append_attribute("type").set_value(pbreak->type);
  breakitem.append_attribute("time").set_value(pbreak->time);
}

}  // namespace kaldi

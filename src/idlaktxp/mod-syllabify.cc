// idlaktxp/mod-syllabify.cc

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

#include "idlaktxp/mod-syllabify.h"

namespace kaldi {

const char* txpsyllabletype[] = {"onset", "nucleus", "coda"};

static void _add_sylxml(const char* spron, pugi::xml_node* node);

TxpSyllabify::TxpSyllabify() : TxpModule("syllabify") {}

TxpSyllabify::~TxpSyllabify() {
}

bool TxpSyllabify::Init(const TxpParseOptions &opts) {
  opts_ = &opts;
  tpdb_ = opts.GetTpdb();
  sylmax_.Init(opts, std::string(GetOptValue("arch")));
  return sylmax_.Parse(tpdb_);
}

bool TxpSyllabify::Process(pugi::xml_document* input) {
  std::string sylpron;
  PhoneVector pvector;
  pugi::xpath_node_set spts = input->document_element().select_nodes("//spt");
  spts.sort();
  for (pugi::xpath_node_set::const_iterator it = spts.begin();
       it != spts.end();
       ++it) {
    pugi::xml_node pre_node;
    pugi::xml_node spt = (*it).node();
    pugi::xpath_node_set tks = spt.select_nodes("descendant::tk");
    tks.sort();
    for (pugi::xpath_node_set::const_iterator it2 = tks.begin();
         it2 != tks.end();
         ++it2) {
      pugi::xml_node node = (*it2).node();
      if (!node.attribute("pron").empty() && strlen(node.attribute("pron").value()) > 0) {
        sylmax_.GetPhoneVector(node.attribute("pron").value(), &pvector);
        if (it2 == tks.end() - 1) pvector[pvector.size() - 1].cross_word = false;
        sylmax_.Maxonset(&pvector);
        if (it2 != tks.begin()) {
          sylmax_.Writespron(&pvector, &sylpron);
          pre_node.append_attribute("spron").set_value(sylpron.c_str());
          // add syllabic xml structure
          _add_sylxml(sylpron.c_str(), &pre_node);
        }
        pre_node = node;
      }
      else {
          node.append_attribute("spron").set_value("");
      }
    }
    if (!pre_node.empty()) {
      sylmax_.Writespron(&pvector, &sylpron);
      pre_node.append_attribute("spron").set_value(sylpron.c_str());
      // add syllabic xml structure
      _add_sylxml(sylpron.c_str(), &pre_node);
    }
  }
  return true;
}

static void _add_sylxml(const char* spron, pugi::xml_node* node) {
    // p is an interative  pointer to the syllabified string
    // phone points to the next phone, plen its string length
    // syl points to the syllable, slen its string length
    // stress points to the nucleus type (in English 0 - no stress
    // 2 - secondary stress, 1 - stressed)
    // type keeps track of syllable subtype 0 - onset, 1 - nucleus
    // 2 - coda
    const char *p, *phone = NULL, *syl = NULL, *stress = "";
    int32 plen = 0, slen = 0, type = 0, sylid = 1, phonid = 1;
    // buffers used to create null terminated strings
    std::string s;
    std::string s2;
    // used to add xml structure
    pugi::xml_node sylnode, phonenode;
    sylnode = node->append_child("syl");
    for (p = spron; *p; p++) {
      // Syllable marker
      if (*p == '|' || *p == '_' || *p == '+') {
        // output phone and phone syllable type
        if (phone) {
          s = std::string(phone, plen);
          phonenode = sylnode.append_child("phon");
          phonenode.append_attribute("val").set_value(s.c_str());
          phonenode.append_attribute("type").set_value(txpsyllabletype[type]);
          phonenode.append_attribute("phonid").set_value(phonid);
          phonid++;
        }
        // if (phone) std::cout << s.c_str() << ' ' << type << '\n';
        // reset phone
        phone = NULL;
        plen = 0;
        // update type
        if (*p == '+') {
          type++;
          slen++;
          // std::cout << *p << " s" << slen << "\n";
        } else if (*p == '|') {
          // end of syllable
          // output syllable
          if (syl) {
              s = std::string(syl, slen);
              s2 = std::string(stress, 1);
              sylnode.append_attribute("val").set_value(s.c_str());
              sylnode.append_attribute("stress").set_value(s2.c_str());
              sylnode.append_attribute("sylid").set_value(sylid);
              sylnode.append_attribute("nophons").set_value(phonid - 1);
          }
        // if (syl) std::cout << "H" << s.c_str() << ' ' << *stress << '\n';
        // reset syllable, stress and type
        if (*(p + 1)) sylnode = node->append_child("syl");
          syl = NULL;
          slen = 0;
          sylid++;
          stress = "";
          type = 0;
          phonid = 1;
        } else {
          // between phone marker '_'
          slen++;
          // std::cout << *p << " s" << slen << "\n";
        }
      } else if (*p >= '0' && *p <= '9') {
        // phone type (stress) marker
        // set to nucleus type
        type = 1;
        // output phone without stress marker
        if (phone) {
          s = std::string(phone, plen);
          phonenode = sylnode.append_child("phon");
          phonenode.append_attribute("val").set_value(s.c_str());
          phonenode.append_attribute("type").set_value(txpsyllabletype[type]);
          phonenode.append_attribute("phonid").set_value(phonid);
          phonid++;
        }
        // if (phone) std::cout << s.c_str() << ' ' << type << '\n';
        // reset phone
        phone = NULL;
        plen = 0;
        // update stress value
        stress = p;
        slen++;
        // std::cout << *p << " n" << slen << "\n";
      } else {
        // other i.e. phone symbol - increment
        if (!phone) phone = p;
        plen++;
        if (!syl) syl = p;
        slen++;
        // std::cout << *p << " e" << slen << "\n";
      }
    }
    // output any phone/syllable left over (in case syllable
    // not properly terminated
    if (phone) {
      s = std::string(phone, plen);
      phonenode = sylnode.append_child("phon");
      phonenode.append_attribute("val").set_value(s.c_str());
      phonenode.append_attribute("type").set_value(txpsyllabletype[type]);
      phonenode.append_attribute("phonid").set_value(phonid);
      phonid++;
    }
    // if (phone) std::cout << s.c_str() << ' ' << type << '\n';
    if (syl) {
      s = std::string(syl, slen);
      s2 = std::string(stress, 1);
    }
    // if (syl) std::cout << s.c_str() << ' ' << *stress << '\n';
    if (syl) {
      sylnode = node->append_child("syl");
      sylnode.append_attribute("val").set_value(s.c_str());
      sylnode.append_attribute("stress").set_value(s2.c_str());
      sylnode.append_attribute("sylid").set_value(sylid);
      sylnode.append_attribute("nophons").set_value(phonid - 1);
      sylid++;
    }
    sylnode.parent().append_attribute("nosyl").set_value(sylid - 1);
  }
}  // namespace kaldi

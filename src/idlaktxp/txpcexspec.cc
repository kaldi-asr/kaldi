// idlaktxp/txpcexspec.cc

// Copyright 2013 CereProc Ltd.  (Author: Matthew Aylett)

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

#include "idlaktxp/txpcexspec.h"
#include "idlaktxp/cexfunctions.h"

namespace kaldi {

// parse file into Cexspec class adding feature specification and
// feature functions to architecture
void TxpCexspec::StartElement(const char* name, const char** atts) {
  std::string att, att2;
  StringSet* set;
  TxpCexspecFeat feat;
  int32 featidx;

  // add features and other definitions
  // top level behaviour
  if (!strcmp(name, "cex")) {
    SetAttribute("maxfieldlen", atts, &att);
    if (!att.empty()) {
      cexspec_maxfieldlen_ = atoi(att.c_str());
    }
    SetAttribute("pausehandling", atts, &att);
    if (!att.empty()) {
      if (!strcmp(att.c_str(), "UTT")) {
        pauhand_ = CEXSPECPAU_HANDLER_UTTERANCE;
      }
    }
    // sets for string based feature functions
  } else if (!strcmp(name, "set")) {
    SetAttribute("name", atts, &att);
    if (att.empty()) {
      // throw an error
      KALDI_WARN << "Badly formed cex set: "
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    curset_ = att;
    SetAttribute("null", atts, &att);
    if (!att2.empty()) {
      att = CEXSPEC_NULL;
    }
    sets_.insert(LookupMapSetItem(curset_, StringSet()));
    setnull_.insert(LookupItem(curset_, att));
    // features
  } else if (!strcmp(name, "item")) {
    SetAttribute("name", atts, &att);
    if (att.empty()) {
      // throw an error
      KALDI_WARN << "Badly formed cex set item: "
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    set = &(sets_.find(curset_)->second);
    set->insert(att);
  } else if (!strcmp(name, "feat")) {
    SetAttribute("name", atts, &att);
    feat.name = att;
    if (att.empty()) {
      // throw an error
      KALDI_WARN << "Badly formed cex set item: "
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    featidx = GetFeatureIndex(att);
    if (featidx == NO_INDEX) {
      curfunc_ = "";
      // throw an error
      KALDI_WARN << "Missing feature in architecture: " << att
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    // valid function try to add rest of the features specification
    curfunc_ = att;
    SetAttribute("htsname", atts, &att);
    feat.htsname = att;
    SetAttribute("desc", atts, &att);
    feat.desc = att;
    SetAttribute("delim", atts, &att);
    if (att.empty() && cexspecfeats_.size()) {
      // throw an error
      KALDI_WARN << "Missing delimiter for feature architecture: " << curfunc_
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    feat.delim = att;
    feat.func = CEXFUNC[featidx];
    feat.pause_context = false;
    SetAttribute("pauctx", atts, &att);
    if (att == "true" || att == "True" || att == "TRUE") {
      feat.pause_context = true;
    }
    SetAttribute("pauctx", atts, &att2);
    feat.type = CEXFUNCTYPE[featidx];
    if (feat.type == CEXSPEC_TYPE_STR) {
      SetAttribute("set", atts, &att);
      if (att.empty()) {
        // throw an error
        KALDI_WARN << "Missing set name for string feature architecture: "
                   << curfunc_
                   << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                   << " Col:" << XML_GetCurrentColumnNumber(parser_);
        return;
      }
      // check set has been added
      if (sets_.find(att) == sets_.end()) {
        // throw an error
        KALDI_WARN << "Missing set for string feature architecture: "
                   << curfunc_
                   << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                   << " Col:" << XML_GetCurrentColumnNumber(parser_)
                   << " (must define before function)";
        return;
      }
      feat.set = att;
      feat.nullvalue = setnull_.find(att)->second;
    } else if (feat.type == CEXSPEC_TYPE_INT) {
      SetAttribute("min", atts, &att);
      if (!att.empty()) feat.min = atoi(att.c_str());
      SetAttribute("max", atts, &att);
      if (att.empty()) {
        // throw an error
        KALDI_WARN << "Missing maximum value for integer feature architecture: "
                   << curfunc_
                   << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                   << " Col:" << XML_GetCurrentColumnNumber(parser_);
        return;
      }
      feat.max = atoi(att.c_str());
    }
    // if we have got to here the feature is valid add it to the architecture
    cexspecfeatlkp_.insert(LookupIntItem(curfunc_, cexspecfeats_.size()));
    cexspecfeats_.push_back(feat);
  } else if (!strcmp(name, "mapping")) {
    SetAttribute("fromstr", atts, &att);
    SetAttribute("tostr", atts, &att2);
    if (att.empty() || att2.empty()) {
      // throw an error
      KALDI_WARN << "bad mapping item: " << curfunc_
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    if (!curfunc_.empty()) {
      LookupInt::iterator iter;
      iter = cexspecfeatlkp_.find(curfunc_);
      if (iter != cexspecfeatlkp_.end()) {
        cexspecfeats_[iter->second].mapping.insert(LookupItem(att, att2));
      }
    }
  }
}

/// return maximum width in bytes of feature string
int32 TxpCexspec::MaxFeatureSize() {
  int32 maxsize = 0;
  TxpCexspecFeatVector::iterator iter;
  // iterate through features and add max size and delimiters
  for (iter = cexspecfeats_.begin(); iter != cexspecfeats_.end(); ++iter) {
    maxsize += CEXSPEC_MAXFIELDLEN;
    maxsize += (*iter).delim.size();
  }
  return maxsize;
}

void TxpCexspec::GetFunctionSpec(pugi::xml_node * header) {
  TxpCexspecFeatVector::iterator iter;
  pugi::xml_node function;
  // iterate through features and add max size and delimiters
  for (iter = cexspecfeats_.begin(); iter != cexspecfeats_.end(); ++iter) {
    function = header->append_child("cexfunction");
    function.append_attribute("name").set_value((*iter).name.c_str());
    function.append_attribute("delim").set_value((*iter).delim.c_str());
    function.append_attribute("isinteger").set_value((*iter).type);
  }  
}

// Add pause tk, syl, phon to break tags and information
// on pause type
// This makes interation for feature extraction more
// homogeneous
int32 TxpCexspec::AddPauseNodes(pugi::xml_document* doc) {
  bool uttbreak, endbreak, internalbreak;
  pugi::xml_node node, childnode;
  pugi::xpath_node_set nodes =
      doc->document_element().select_nodes("//phon|//break");
  nodes.sort();
  for (pugi::xpath_node_set::const_iterator it = nodes.begin();
       it != nodes.end();
       ++it) {
    node = it->node();
    if (!strcmp(node.name(), "break")) {
      // determine break type
      if (!strcmp(node.attribute("type").value(), "4")) {
        uttbreak = true;
      } else {
        uttbreak = false;
      }
      if (it == nodes.begin()) {
        endbreak = false;
        internalbreak = false;
      } else if (it + 1 == nodes.end()) {
        endbreak = true;
        internalbreak = false;
      } else if (!strcmp((it - 1)->node().name(), "break")) {
        endbreak = false;
        internalbreak = true;
      } else {
        endbreak = true;
        internalbreak = true;
      }
      if (pauhand_ == CEXSPECPAU_HANDLER_UTTERANCE && !uttbreak &&
          internalbreak && !endbreak) continue;
      childnode = node.append_child("tk");
      childnode.append_attribute("pron").set_value("pau");
      childnode = childnode.append_child("syl");
      childnode.append_attribute("val").set_value("pau");
      childnode = childnode.append_child("phon");
      childnode.append_attribute("val").set_value("pau");
    }
  }
  return true;
}

// call the feature functions
bool TxpCexspec::ExtractFeatures(const TxpCexspecContext &context, std::string* buf) {
  TxpCexspecFeatVector::iterator iter;
  struct TxpCexspecFeat feat;
  bool rval = true;
  // iterate through features inserting nulls when required
  for (iter = cexspecfeats_.begin(); iter != cexspecfeats_.end(); ++iter) {
    feat = *iter;
    if (!feat.func(this, &feat, &context, buf))
      rval = false;
  }
  return rval;
}

/// utility to return index of a feature function
int32 TxpCexspec::GetFeatureIndex(const std::string &name) {
  for (int i = 0; i < CEX_NO_FEATURES; i++) {
    if (!strcmp(CEXFUNCLBL[i], name.c_str())) return i;
  }
  return NO_INDEX;
}

bool TxpCexspec::AppendValue(const TxpCexspecFeat &feat, bool error,
                      const char* s, std::string* buf) const {
  const StringSet* set;
  StringSet::iterator i;
  set = &(sets_.find(feat.set)->second);
  if (set->find(std::string(s)) == set->end()) {
    AppendError(feat, buf);
    return false;
  }
  buf->append(feat.delim);
  buf->append(Mapping(feat, s));
  return true;
}

bool TxpCexspec::AppendValue(const TxpCexspecFeat &feat, bool error,
                      int32 i, std::string* buf) const {
  std::stringstream stream;
  if (feat.type != CEXSPEC_TYPE_INT) {
    AppendError(feat, buf);
    return false;
  }
  if (i < feat.min) i = feat.min;
  if (i > feat.max) i = feat.max;
  stream << i;
  buf->append(feat.delim);
  buf->append(Mapping(feat, stream.str()));
  return true;
}

bool TxpCexspec::AppendNull(const TxpCexspecFeat &feat, std::string* buf) const {
  buf->append(feat.delim);
  buf->append(Mapping(feat, feat.nullvalue));
  return true;
}

bool TxpCexspec::AppendError(const TxpCexspecFeat &feat, std::string* buf) const {
  buf->append(feat.delim);
  buf->append(CEXSPEC_ERROR);
  return true;
}

const std::string TxpCexspec::Mapping(const TxpCexspecFeat &feat,
                                 const std::string &instr) const {
  LookupMap::const_iterator i;
  i = feat.mapping.find(instr);
  if (i != feat.mapping.end()) {
    return i->second;
  } else {
    return instr;
  }
}

TxpCexspecModels::~TxpCexspecModels() {
  int32 i;
  for (i = 0; i < models_.size(); i++) {
    delete models_[i];
  }
}

void TxpCexspecModels::Init(TxpCexspec* cexspec) {
  buflen_ = cexspec->MaxFeatureSize() + 1;
}

void TxpCexspecModels::Clear() {
  int32 i;
  for (i = 0; i < models_.size(); i++) {
    delete models_[i];
  }
  models_.clear();
}

char* TxpCexspecModels::Append() {
  char* buf = new char[buflen_];
  memset(buf, 0, buflen_);
  models_.push_back(buf);
  return buf;
}

TxpCexspecContext::TxpCexspecContext(const pugi::xml_document &doc,
                       enum CEXSPECPAU_HANDLER pauhand) : pauhand_(pauhand) {
  phones_ = doc.document_element().select_nodes("//phon");
  syllables_ = doc.document_element().select_nodes("//syl");
  words_ = doc.document_element().select_nodes("//tk");
  spurts_ = doc.document_element().select_nodes("//spt");
  utterances_ = doc.document_element().select_nodes("//utt");
  phones_.sort();
  syllables_.sort();
  words_.sort();
  spurts_.sort();
  utterances_.sort();
  current_phone_ = phones_.begin();
  current_syllable_ = syllables_.begin();
  current_word_ = words_.begin();
  current_spurt_ = spurts_.begin();
  current_utterance_ = utterances_.begin();
}

bool TxpCexspecContext::Next() {
  pugi::xml_node node, phon, empty;

  // std::cout << (current_phone_->node()).attribute("val").value() << ":"
  //           << (current_syllable_->node()).attribute("val").value() << ":"
  //           << (current_word_->node()).attribute("norm").value() << ":"
  //           << (current_spurt_->node()).attribute("phraseid").value() << ":"
  //           << (current_utterance_->node()).attribute("uttid").value() << " "
  //           << (current_phone_->node()).attribute("type").value() << "\n";
  // iterate over phone/break items
  current_phone_++;
  if (current_phone_ == phones_.end()) return false;
  phon = current_phone_->node();
  // update other iterators as required
  // dummy pau items already added for tk and syl levels
  node = GetContextUp(phon, "syl");
  while (node != current_syllable_->node()) current_syllable_++;
  node = GetContextUp(phon, "tk");
  while (node != current_word_->node()) current_word_++;
  node = GetContextUp(phon, "spt");
  while (node != current_spurt_->node()) current_spurt_++;
  node = GetContextUp(phon, "utt");
  while (node != current_utterance_->node()) current_utterance_++;

  return true;
}

// look up from the node until we find the correct current context node
pugi::xml_node TxpCexspecContext::GetContextUp(const pugi::xml_node &node,
                                          const char* name) const {
  pugi::xml_node parent;
  pugi::xml_node empty;
  parent = node.parent();
  while (!parent.empty()) {
    if (!strcmp(parent.name(), name)) return parent;
    parent = parent.parent();
  }
  return empty;
}

// return phon back or forwards from current phone
pugi::xml_node TxpCexspecContext::GetPhone(const int32 idx,
                                           const bool pause_context) const {
  int32 i;
  int32 pau_found = 0;
  pugi::xml_node empty;
  if (idx >= 0) {
    for (i = 0; i < idx; i++) {
      if ((current_phone_ + i) == phones_.end()) return empty;
      if (!strcmp("pau", (current_phone_ + i)->node().attribute("val").value()))
        pau_found++;
    }
  } else {
    for (i = 0; i > idx; i--) {
      if ((current_phone_ + i) == phones_.begin()) return empty;
      if (!strcmp("pau", (current_phone_ + i)->node().attribute("val").value()))
        pau_found++;
    }
  }
  if ((current_phone_ + i) == phones_.end()) return empty;
  if (pau_found == 2 && !pause_context) return empty;
  return (current_phone_ + i)->node();
}

// return parent syllable back or forwards from current phone
pugi::xml_node TxpCexspecContext::GetSyllable(const int32 idx,
                                              const bool pause_context) const {
  int32 i;
  // int32 pau_found = 0;
  pugi::xml_node empty;
  // Get the parent syllable node of the current phone.
  if (idx >= 0) {
    for (i = 0; i < idx; i++) {
      if ((current_syllable_ + i) == syllables_.end()) return empty;
      // if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        // pau_found++;
    }
  } else {
    for (i = 0; i > idx; i--) {
      if ((current_syllable_ + i) == syllables_.begin()) return empty;
      // if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        // pau_found++;
    }
  }
  if ((current_syllable_ + i) == syllables_.end()) return empty;
  // if (pau_found == 2 && !pauctx) return empty;
  return (current_syllable_ + i)->node();
}

// return parent token back or forwards from current phone
pugi::xml_node TxpCexspecContext::GetWord(const int32 idx,
                                          const bool pause_context) const {
  int32 i;
  // int32 pau_found = 0;
  pugi::xml_node empty;
  // Get the parent token node of the current phone.
  if (idx >= 0) {
    for (i = 0; i < idx; i++) {
      if ((current_word_ + i) == words_.end()) return empty;
      // if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        // pau_found++;
    }
  } else {
    for (i = 0; i > idx; i--) {
      if ((current_word_ + i) == words_.begin()) return empty;
      // if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        // pau_found++;
    }
  }
  if ((current_word_ + i) == words_.end()) return empty;
  // if (pau_found == 2 && !pauctx) return empty;
  return (current_word_ + i)->node();
}

// return parent spurt back or forwards from current phone
pugi::xml_node TxpCexspecContext::GetSpurt(const int32 idx,
                                           const bool pause_context) const {
  int32 i;
  // int32 pau_found = 0;
  pugi::xml_node empty;
  // Get the parent token node of the current phone.
  if (idx >= 0) {
    for (i = 0; i < idx; i++) {
      if ((current_spurt_ + i) == spurts_.end()) return empty;
      // if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        // pau_found++;
    }
  } else {
    for (i = 0; i > idx; i--) {
      if ((current_spurt_ + i) == spurts_.begin()) return empty;
      // if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        // pau_found++;
    }
  }
  if ((current_spurt_ + i) == spurts_.end()) return empty;
  // if (pau_found == 2 && !pauctx) return empty;
  return (current_spurt_ + i)->node();
}



}  // namespace kaldi

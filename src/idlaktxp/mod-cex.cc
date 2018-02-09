// idlaktxp/mod-cex.cc

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

#include "idlaktxp/mod-cex.h"

namespace kaldi {

TxpCex::TxpCex() : TxpModule("cex") {}

TxpCex::~TxpCex() {
}

bool TxpCex::Init(const TxpParseOptions &opts) {
  opts_ = &opts;
  tpdb_ = opts.GetTpdb();
  cexspec_.Init(opts, std::string(GetOptValue("arch")));
  return cexspec_.Parse(tpdb_);
}

bool TxpCex::Process(pugi::xml_document* input) {
  std::string* model = new std::string();
  std::string cexfunctions;
  pugi::xml_node header = GetHeader(input);
  cexspec_.GetFunctionSpec(&header);
  cexspec_.AddPauseNodes(input);
  pugi::xpath_node_set tks =
      input->document_element().select_nodes("//phon");
  tks.sort();
  TxpCexspecContext context(*input,  cexspec_.GetPauseHandling());
  kaldi::int32 i = 0;
  for (pugi::xpath_node_set::const_iterator it = tks.begin();
       it != tks.end();
       ++it, i++, context.Next()) {
    pugi::xml_node phon = (*it).node();
    model->clear();
    cexspec_.ExtractFeatures(context, model);
    phon.text() = model->c_str();
  }
  delete model;
  // should set to error status
  return true;
}

bool TxpCex::IsSptPauseHandling() {
  if (cexspec_.GetPauseHandling() == CEXSPECPAU_HANDLER_SPURT) return true;
  return false;
}

}  // namespace kaldi

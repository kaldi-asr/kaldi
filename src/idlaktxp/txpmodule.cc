// idlaktxp/toxmodule.cc

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

namespace kaldi {

const char * TxpModule::GetOptValue(const char *key) {
  return opts_->GetValue(name_.c_str(), key);
}

// True/true/TRUE -> true, False, false, FALSE, anything else -> false
bool TxpModule::GetOptValueBool(const char *key) {
  const std::string val = opts_->GetValue(name_.c_str(), key);
  if ((val == "True") || (val == "true") || (val == "TRUE"))
    return true;
  return false;
}

pugi::xml_node TxpModule::GetHeader(pugi::xml_document* input) {
  // check we have a txpheader node for definition information
  pugi::xml_node txpheader =
      input->document_element().child("txpheader");
  if (!txpheader) txpheader =
                      input->document_element().prepend_child("txpheader");
  // remove a module entry if it is already present
  txpheader.remove_child(name_.c_str());
  return txpheader.append_child(name_.c_str());
}

}  // namespace kaldi

// idlaktxp/txpmodule.h

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

#ifndef KALDI_IDLAKTXP_TXPMODULE_H_
#define KALDI_IDLAKTXP_TXPMODULE_H_

// This file defines the basic txp module which incrementally parses
// either text, tox (token oriented xml) tokens, or spurts (phrases)
// containing tox tokens.

#include <string>
#include "pugixml.hpp"

#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpparse-options.h"

namespace kaldi {

/// Base class for all modules
///
/// Takes puji XML input and modifies it for puji XML output
/// Has a configuration section which can be accessed using utility
/// functions
class TxpModule {
 public:
  /// Construct module, also loads tpdb modules in specific instances
  explicit TxpModule(const std::string &name) : opts_(NULL), name_(name) {}
  virtual ~TxpModule() {}
  /// Load voice data
  virtual bool Init(const TxpParseOptions &opts) {return true;}
  /// Process the XML, modifying the XML to reflect linguistic
  /// information
  virtual bool Process(pugi::xml_document* input) {return true;}
  /// Return a configuration value for this module as a string (rendundant)
  const std::string GetConfigValue(const std::string &key);
  /// Return an option value for this module as a string
  const char * GetOptValue(const char *key);
  /// Return a boolean configuration
  /// True/true/TRUE -> true, False, false, FALSE, anything else -> false
  bool GetConfigValueBool(const std::string &key);
  /// Return a boolean configuration
  /// True/true/TRUE -> true, False, false, FALSE, anything else -> false
  bool GetOptValueBool(const char *key);
  /// Get the name of the module
  const std::string&  GetName() const {return name_;}
  /// Gets or adds a module header to txpheader
  pugi::xml_node GetHeader(pugi::xml_document* input);

 protected:
  /// Options structure for the module
  const TxpParseOptions * opts_;
  /// Directory for text processing database (tpdb) files
  /// used by txpxmldata objectys in the module
  std::string tpdb_;

 private:
  /// Name of the module
  std::string name_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPMODULE_H_

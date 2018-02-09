// idlaktxp/txpparse-options.h

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

// Speech synthesis centred around a voice database and all binaries need to
// load options from this database as well as having defaults, user options
// specified in --config=<optionfile> and on the command line.
// Therefore this is a subclassed version of parse options which allows
// construction with the tpdb pathname as well as the standard adding of
// options.

#ifndef KALDI_IDLAKTXP_TXPPARSE_OPTIONS_H_
#define KALDI_IDLAKTXP_TXPPARSE_OPTIONS_H_

#include <string>
#include <map>
#include <utility>
#include "util/parse-options.h"
#include "idlaktxp/idlak-common.h"

namespace kaldi {

/// string to string pointer lookup
typedef std::map<std::string, std::string*> LookupMapPtr;
/// string/string pointer pair
typedef std::pair<std::string, std::string*> LookupItemPtr;

/// Class derived from standard ParseOptions in order to allow txp system
/// to load a voice specific set of configurations switches from
/// <tpdb path>/default.conf and set txp options to default values before
/// overiding with --config=<file> and command line swithes.
/// txp variables are all std::string and are stored in this class
/// for txp modules to access as required.
class TxpParseOptions : public ParseOptions {
 public:
  /// Set defaults and register switches in txpoptions_
  explicit TxpParseOptions(const char* usage);
  /// Delete added switches in txpoptions_
  ~TxpParseOptions();
  /// Read in tpdb path, set tpdb, load default.conf from voice
  int Read(int argc, const char *const *argv);
  /// Return txp switch value used by txp modules
  const char* GetValue(const char* module, const char* key) const;
  /// Return path to tpdb
  const char* GetTpdb() const;
  /// Override path to tpdb
  void SetTpdb(const std::string &tpdb) {tpdb_ = tpdb;}
 private:
  /// Path to tpdb (text processing database) used by the voice
  std::string tpdb_;
  /// holds txp specific switch settings
  LookupMapPtr txpoptions_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPPARSE_OPTIONS_H_

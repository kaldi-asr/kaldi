// idlaktxp/txpparse-options.cc

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

#include "idlaktxp/txpparse-options.h"
#include "util/text-utils.h"
#include "base/kaldi-common.h"

namespace kaldi {

/// This is the full default configuration for txp. Only keys that
/// have a default value here can be specified in module setup
/// \todo currently nonsense values and modules
const std::string txpconfigdefault =
    "--general-lang=en\n"
    "--general-region=us\n"
    "--general-acc=ga\n"
    "--general-spk=\n"
    "--tokenise-arch=default\n"
    "--tokenise-processing-mode=lax\n"
    "--pauses-arch=default\n"
    "--pauses-hzone=True\n"
    "--pauses-hzone-start=60\n"
    "--pauses-hzone-end=100\n"
    "--postag-arch=default\n"
    "--phrasing-max-phrase-length=30\n"
    "--phrasing-by-utterance=True\n"
    "--phrasing-max-utterance-length=10\n"
    "--phrasing-phrase-length-window=10\n"
    "--normalise-trace=0\n"
    "--normalise-active=True\n"
    "--pronounce-arch=default\n"
    "--pronounce-novowel-spell=True\n"
    "--syllabify-arch=default\n"
    "--syllabify-slang=\n"
    "--cex-arch=default\n";

TxpParseOptions::TxpParseOptions(const char *usage)
    : ParseOptions(usage) {
  std::istringstream is(txpconfigdefault);
  std::string line, key, value, *stringptr;
  LookupMapItem item;
  RegisterStandard("tpdb", &tpdb_,
                   "Text processing database (directory XML language/speaker files)"); //NOLINT
  while (std::getline(is, line)) {
    // trim out the comments
    size_t pos;
    if ((pos = line.find_first_of('#')) != std::string::npos) {
      line.erase(pos);
    }
    // skip empty lines
    Trim(&line);
    if (line.length() == 0) continue;

    // parse option
    bool has_equal_sign;
    SplitLongArg(line, &key, &value, &has_equal_sign);
    NormalizeArgName(&key);
    Trim(&value);
    stringptr = new std::string(value);
    txpoptions_.insert(LookupItemPtr(key, stringptr));
    Register(key.c_str(), stringptr,
             "Idlak Text Processing Option");
  }
}

TxpParseOptions::~TxpParseOptions() {
  for (LookupMapPtr::iterator iter = txpoptions_.begin();
      iter != txpoptions_.end(); iter++) {
    delete iter->second;
  }
}

int TxpParseOptions::Read(int argc, const char *const *argv) {
  std::string key, value;
  int i;
  // first pass: look for tpdb parameter
  for (i = 1; i < argc; i++) {
    if (std::strncmp(argv[i], "--", 2) == 0) {
      bool has_equal_sign;
      SplitLongArg(argv[i], &key, &value, &has_equal_sign);
      NormalizeArgName(&key);
      Trim(&value);
      if (key.compare("tpdb") == 0) {
        // check for tpdb configuration file
        std::ifstream is((value + "/default.conf").c_str(), std::ifstream::in);
        if (is.good()) {
          is.close();
          ReadConfigFile(value + "/default.conf");
        }
      }
      if (key.compare("help") == 0) {
        PrintUsage();
        exit(0);
      }
    }
  }
  return ParseOptions::Read(argc, argv);
}

// key --<key> is treated as --general-<key> 
const char* TxpParseOptions::GetValue(const char* module, const char* key) const {
  LookupMapPtr::const_iterator lookup;
  std::string optkey(module);
  optkey = optkey + "-" + key;
  lookup = txpoptions_.find(optkey);
  if (lookup == txpoptions_.end()) {
    // if general section also look for field without section
    // description
    if (!strcmp(module, "general")) {
      optkey = key;
      lookup = txpoptions_.find(optkey);
      if (lookup == txpoptions_.end()) return NULL;
      else return (lookup->second)->c_str();   
    }
  }
  return (lookup->second)->c_str();
}
const char* TxpParseOptions::GetTpdb() const {
  return tpdb_.c_str();
}

}  // namespace kaldi

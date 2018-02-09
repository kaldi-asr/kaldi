// idlaktxp/txppcre.cc

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

#include "idlaktxp/txppcre.h"

namespace kaldi {

// Convert string into a pcre regular expression
const pcre* TxpPcre::Compile(const char* rgx) {
  const char* error;
  int erroffset;
  return pcre_compile(rgx, PCRE_UTF8, &error, &erroffset, NULL);
}

// Apply a regular expression to some input
bool TxpPcre::Execute(const pcre* rgx, const std::string &input) {
  input_ = input.c_str();
  n_ = pcre_exec(rgx, NULL, input_, input.length(), 0, PCRE_NO_UTF8_CHECK,
                 ovector_,
                 KALDI_TXPPCRE_MAXMATCH * 3);
  if (n_ >= 0) return true;
  return false;
}

// Apply a regular expression to a string and return remaining unmatched string
const char* TxpPcre::Consume(const pcre* rgx, const char* input, int32 len) {
  input_ = input;
  n_ = pcre_exec(rgx, NULL, input, len, 0, PCRE_NO_UTF8_CHECK,
                 ovector_,
                 KALDI_TXPPCRE_MAXMATCH * 3);
  if (n_ >= 0) {
    return input + ovector_[1];
  }
  return NULL;
}

const char* TxpPcre::Consume(const pcre* rgx, const char* input) {
  return Consume(rgx, input, strlen(input));
}

// Copy matched element n of input into tgt
bool TxpPcre::SetMatch(int32 n, std::string* tgt) {
  if (n >= 0 && n < n_) {
    *tgt = std::string(input_ + ovector_[(n + 1) * 2],
                      ovector_[((n + 1) * 2) + 1] - ovector_[(n + 1) * 2]);
    return true;
  }
  return false;
}

}  // namespace kaldi

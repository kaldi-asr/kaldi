// idlaktxp/txppcre.h

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

#ifndef KALDI_IDLAKTXP_TXPPCRE_H_
#define KALDI_IDLAKTXP_TXPPCRE_H_

// This file wraps pcre into a class which holds match information

#include <pcre.h>
#include <string>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"

#define KALDI_TXPPCRE_MAXMATCH 10

namespace kaldi {

/// \addtogroup idlak_utils
/// @{

/// Simple C++ wrapper around pcre
class TxpPcre {
 public:
  /// Convert string into a pcre regular expression
  const pcre* Compile(const char* rgx);
  /// Apply a regular expression to some input
  bool Execute(const pcre* rgx, const std::string &input);
  /// Apply a regular expression to a string and return remaining unmatched
  /// string
  const char* Consume(const pcre* rgx, const char* input, int32 len);
  const char* Consume(const pcre* rgx, const char* input);
  /// Copy matched element of input into tgt
  bool SetMatch(int32 match, std::string *tgt);
  /// Return number of matches
  int32 Matches() {return n_ - 1;}
 private:
  /// Hold pointer to last matched string
  const char* input_;
  /// Hold integer offsets to matches in input
  int32 ovector_[KALDI_TXPPCRE_MAXMATCH * 3];
  /// Number of matches in regex
  int32 n_;
};

/// @} end of \addtogroup idlak_utils

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPPCRE_H_

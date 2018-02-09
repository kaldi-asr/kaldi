// idlaktxp/toxmodules.h

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

#ifndef KALDI_IDLAKTXP_IDLAK_COMMON_H_
#define KALDI_IDLAKTXP_IDLAK_COMMON_H_

// This file defines items used across the whole of idlak

#include <map>
#include <set>
#include <vector>
#include <utility>
#include <string>

namespace kaldi {

#define NO_INDEX -1

/// Generic string to string lookup
typedef std::map<std::string, std::string> LookupMap;
/// Generic string/string pair
typedef std::pair<std::string, std::string> LookupItem;
/// Generic string to string map lookup
typedef std::map<std::string, LookupMap> LookupMapMap;
/// Generic string/lookupmap pair
typedef std::pair<std::string, LookupMap> LookupMapItem;
/// Generic string to vector index
typedef std::map<std::string, int32> LookupInt;
/// Generic string/vector index pair
typedef std::pair<std::string, int32> LookupIntItem;
/// Generic string array
typedef std::vector<std::string> StringVector;
/// Generic string set
typedef std::set<std::string> StringSet;
/// Generic vector of char*
typedef std::vector<char* > CharPtrVector;
}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_IDLAK_COMMON_H__

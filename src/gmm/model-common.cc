// gmm/model-common.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"

namespace kaldi {
GmmFlagsType StringToGmmFlags(std::string str) {
  GmmFlagsType flags = 0;
  for (const char *c = str.c_str(); *c != '\0'; c++) {
    switch (*c) {
      case 'm': flags |= kGmmMeans; break;
      case 'v': flags |= kGmmVariances; break;
      case 'w': flags |= kGmmWeights; break;
      case 't': flags |= kGmmTransitions; break;
      case 'a': flags |= kGmmAll; break;
      default: KALDI_ERR << "Invalid element " << CharToString(*c)
                         << " of GmmFlagsType option string "
                         << str;
    }
  }
  return flags;
}

std::string GmmFlagsToString(GmmFlagsType flags) {
  std::string ans;
  if (flags & kGmmMeans) ans += "m";
  if (flags & kGmmVariances) ans += "v";
  if (flags & kGmmWeights) ans += "w";
  if (flags & kGmmTransitions) ans += "t";
  return ans;
}

GmmFlagsType AugmentGmmFlags(GmmFlagsType flags) {
KALDI_ASSERT((flags & ~kGmmAll) == 0);  // make sure only valid flags are present.
  if (flags & kGmmVariances) flags |= kGmmMeans;
  if (flags & kGmmMeans) flags |= kGmmWeights;
  if (!(flags & kGmmWeights)) {
    KALDI_WARN << "Adding in kGmmWeights (\"w\") to empty flags.";
    flags |= kGmmWeights; // Just add this in regardless:
    // if user wants no stats, this will stop programs from crashing due to dim mismatches.
  }
  return flags;
}

SgmmUpdateFlagsType StringToSgmmUpdateFlags(std::string str) {
  SgmmUpdateFlagsType flags = 0;
  for (const char *c = str.c_str(); *c != '\0'; c++) {
    switch (*c) {
      case 'v': flags |= kSgmmPhoneVectors; break;
      case 'M': flags |= kSgmmPhoneProjections; break;
      case 'w': flags |= kSgmmWeightProjections; break;
      case 'S': flags |= kSgmmCovarianceMatrix; break;
      case 'c': flags |= kSgmmSubstateWeights; break;
      case 'N': flags |= kSgmmSpeakerProjections; break;
      case 't': flags |= kSgmmTransitions; break;
      case 'a': flags |= kSgmmAll; break;
      default: KALDI_ERR << "Invalid element " << CharToString(*c)
                         << " of SgmmUpdateFlagsType option string "
                         << str;
    }
  }
  return flags;
}


SgmmUpdateFlagsType StringToSgmmWriteFlags(std::string str) {
  SgmmWriteFlagsType flags = 0;
  for (const char *c = str.c_str(); *c != '\0'; c++) {
    switch (*c) {
      case 'g': flags |= kSgmmGlobalParams; break;
      case 's': flags |= kSgmmStateParams; break;
      case 'n': flags |= kSgmmNormalizers; break;
      case 'u': flags |= kSgmmBackgroundGmms; break;
      case 'a': flags |= kSgmmAll; break;
      default: KALDI_ERR << "Invalid element " << CharToString(*c)
                         << " of SgmmWriteFlagsType option string "
                         << str;
    }
  }
  return flags;
}




}  // End namespace kaldi

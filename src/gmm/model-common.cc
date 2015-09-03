// gmm/model-common.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
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
#include <queue>
#include <numeric>

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
      case 'w': flags |= kSgmmPhoneWeightProjections; break;
      case 'S': flags |= kSgmmCovarianceMatrix; break;
      case 'c': flags |= kSgmmSubstateWeights; break;
      case 'N': flags |= kSgmmSpeakerProjections; break;
      case 't': flags |= kSgmmTransitions; break;
      case 'u': flags |= kSgmmSpeakerWeightProjections; break;
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

struct CountStats {
  CountStats(int32 p, int32 n, BaseFloat occ)
      : pdf_index(p), num_components(n), occupancy(occ) {}
  int32 pdf_index;
  int32 num_components;
  BaseFloat occupancy;
  bool operator < (const CountStats &other) const {
    return occupancy/(num_components+1.0e-10) <
        other.occupancy/(other.num_components+1.0e-10);
  }
};


void GetSplitTargets(const Vector<BaseFloat> &state_occs,
                     int32 target_components,
                     BaseFloat power,
                     BaseFloat min_count,
                     std::vector<int32> *targets) {
  std::priority_queue<CountStats> split_queue;
  int32 num_pdfs = state_occs.Dim();
  
  for (int32 pdf_index = 0; pdf_index < num_pdfs; pdf_index++) {
    BaseFloat occ = pow(state_occs(pdf_index), power);
    // initialize with one Gaussian per PDF, to put a floor
    // of 1 on the #Gauss
    split_queue.push(CountStats(pdf_index, 1, occ));
  }
  
  for (int32 num_gauss = num_pdfs; num_gauss < target_components;) {
    CountStats state_to_split = split_queue.top();
    if (state_to_split.occupancy == 0) {
      KALDI_WARN << "Could not split up to " << target_components
                 << " due to min-count = " << min_count
                 << " (or no counts at all)\n";
      break;
    }
    split_queue.pop();
    BaseFloat orig_occ = state_occs(state_to_split.pdf_index);
    if ((state_to_split.num_components+1) * min_count >= orig_occ) {
      state_to_split.occupancy = 0; // min-count active -> disallow splitting
      // this state any more by setting occupancy = 0.
    } else {
      state_to_split.num_components++;
      num_gauss++;
    }
    split_queue.push(state_to_split);
  }
  targets->resize(num_pdfs);  
  while (!split_queue.empty()) {
    int32 pdf_index = split_queue.top().pdf_index;
    int32 pdf_tgt_comp = split_queue.top().num_components;
    (*targets)[pdf_index] = pdf_tgt_comp;
    split_queue.pop();
  }
}

}  // End namespace kaldi

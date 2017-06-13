// hmm/tree-accu.h

// Copyright 2009-2011 Microsoft Corporation
//           2013-2015 Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_HMM_TREE_ACCU_H_
#define KALDI_HMM_TREE_ACCU_H_

#include <cctype>  // For isspace.
#include <limits>
#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "tree/clusterable-classes.h"
#include "tree/build-tree-questions.h" // needed for this typedef:
// typedef std::vector<std::pair<EventVector, Clusterable*> > BuildTreeStatsType;

namespace kaldi {

/// \ingroup tree_group_top
/// @{

struct AccumulateTreeStatsOptions {
  BaseFloat var_floor;
  std::string ci_phones_str;
  std::string phone_map_rxfilename;
  bool collapse_pdf_classes;
  int context_width;
  int central_position;
  AccumulateTreeStatsOptions(): var_floor(0.01), context_width(3),
                                central_position(1) { }


  void Register(OptionsItf *opts) {
    opts->Register("var-floor", &var_floor, "Variance floor for tree "
                   "clustering.");
    opts->Register("ci-phones", &ci_phones_str, "Colon-separated list of "
                   "integer indices of context-independent phones (after "
                   "mapping, if --phone-map option is used).");
    opts->Register("context-width", &context_width, "Context window size.");
    opts->Register("central-position", &central_position, "Central "
                   "context-window position (zero-based)");
    opts->Register("phone-map", &phone_map_rxfilename,
                   "File name containing old->new phone mapping (each line is: "
                   "old-integer-id new-integer-id)");
  }
};

// This class is a binary representation of AccumulateTreeStatsOptions.
struct AccumulateTreeStatsInfo {
  BaseFloat var_floor;
  std::vector<int32> ci_phones;  // sorted, uniq vector of context-independent
                                 // phones.
  std::vector<int32> phone_map;  // if nonempty, maps old phones to new phones.
  int32 context_width;
  int32 central_position;
  AccumulateTreeStatsInfo(const AccumulateTreeStatsOptions &opts);
};

/// Accumulates the stats needed for training context-dependency trees (in the
/// "normal" way).  It adds to 'stats' the stats obtained from this file.  Any
/// new GaussClusterable* pointers in "stats" will be allocated with "new".

void AccumulateTreeStats(const TransitionModel &trans_model,
                         const AccumulateTreeStatsInfo &info,
                         const std::vector<int32> &alignment,
                         const Matrix<BaseFloat> &features,
                         std::map<EventType, GaussClusterable*> *stats);



/*** Read a mapping from one phone set to another.  The phone map file has lines
 of the form <old-phone> <new-phone>, where both entries are integers, usually
 nonzero (but this is not enforced).  This program will crash if the input is
 invalid, e.g. there are multiple inconsistent entries for the same old phone.
 The output vector "phone_map" will be indexed by old-phone and will contain
 the corresponding new-phone, or -1 for any entry that was not defined. */

void ReadPhoneMap(std::string phone_map_rxfilename,
                  std::vector<int32> *phone_map);



/// @}

}  // end namespace kaldi.

#endif

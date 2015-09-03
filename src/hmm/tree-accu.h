// hmm/tree-accu.h

// Copyright 2009-2011 Microsoft Corporation
//                2013 Johns Hopkins University (author: Daniel Povey)

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


/// Accumulates the stats needed for training context-dependency trees (in the
/// "normal" way).  It adds to 'stats' the stats obtained from this file.  Any
/// new GaussClusterable* pointers in "stats" will be allocated with "new".

void AccumulateTreeStats(const TransitionModel &trans_model,
                         BaseFloat var_floor,
                         int N,  // context window size.
                         int P,  // central position.
                         const std::vector<int32> &ci_phones,  // sorted
                         const std::vector<int32> &alignment,
                         const Matrix<BaseFloat> &features,
                         const std::vector<int32> *phone_map, // or NULL
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

// idlakfeat/hmm-utils-idlak.h

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

#ifndef KALDI_IDLAK_HMM_UTILS_H_
#define KALDI_IDLAK_HMM_UTILS_H_

#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

/// \ingroup hmm_group
/// @{

/**
 * ConvertFullCtxAlignment converts an alignment that was created using one
 * model, to another model.  They must use a compatible topology (so we
 * know the state alignments of the new model).
 * It returns false if it could not be split to phones (probably
 * because the alignment was partial), but for other kinds of
 * error that are more likely a coding error, it will throw
 * an exception.
 */
bool ConvertFullCtxAlignment(const TransitionModel &old_trans_model,
                             const TransitionModel &new_trans_model,
                             const ContextDependencyInterface &new_ctx_dep,
                             const std::vector<int32> &tid_ali,
                             const std::vector< std::vector <int32> > &full_ali,
                             std::vector<int32> *new_tid_ali);

/**
 * GetPhoneWindows takes an alignment in terms of transition-ids and works out
 * the phonetic context windows for a given context-width and central position.
 * It also returns the list of phones seen in the alignment if the 'phones'
 * argument is not NULL. If the 'per_frame' option is true, then the phonetic
 * context windows (and phones) are repeated for the number of frames they span.
 */
bool GetPhoneWindows(const TransitionModel &trans_model,
                     const std::vector<int32> &alignment,
                     int32 context_width,
                     int32 central_pos,
                     bool per_frame,
                     std::vector< std::vector<int32> > *phone_windows,
                     std::vector<int32> *phones);

/// @} end "addtogroup hmm_group"

} // end namespace kaldi


#endif

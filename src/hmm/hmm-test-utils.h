// hmm/hmm-test-utils.h

// Copyright 2009-2011   Microsoft Corporation
//                2015   Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_HMM_HMM_TEST_UTILS_H_
#define KALDI_HMM_HMM_TEST_UTILS_H_

#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

// Here we put a convenience function for generating a TransitionModel object --
// useful in test code.  We may put other testing-related things here in time.

// This function returns a randomly generated TransitionModel object.
// If 'ctx_dep' is not NULL, it outputs to *ctx_dep a pointer to the
// tree that was used to generate the transition model.
TransitionModel *GenRandTransitionModel(ContextDependency **ctx_dep);

/// This function returns a HmmTopology object giving a normal 3-state topology,
/// covering all phones in the list "phones".  This is mainly of use in testing
/// code.
HmmTopology GetDefaultTopology(const std::vector<int32> &phones);


/// This method of generating an arbitrary HmmTopology object allows you to
/// specify the number of pdf-classes for each phone separately.
/// 'num_pdf_classes' is indexed by the phone-index (so the length will be
/// longer than the length of the 'phones' vector, which for example lacks the
/// zero index and may have gaps).
HmmTopology GenRandTopology(const std::vector<int32> &phones,
                            const std::vector<int32> &num_pdf_classes);

/// This version of GenRandTopology() generates the phone list and number of pdf
/// classes randomly.
HmmTopology GenRandTopology();

/// This function generates a random path through the HMM for the given
/// phone.  The 'path' output is a list of pairs (HMM-state, transition-index)
/// in which any nonemitting states will have been removed.  This is
/// used in other test code.
/// the 'reorder' option is as described in the documentation; if true, the
/// self-loops from a state are reordered to come after the forward-transition.
void GeneratePathThroughHmm(const HmmTopology &topology,
                            bool reorder,
                            int32 phone,
                            std::vector<std::pair<int32, int32> > *path);


/// For use in test code, this function generates an alignment (a sequence of
/// transition-ids) corresponding to a given phone sequence.
void GenerateRandomAlignment(const ContextDependencyInterface &ctx_dep,
                             const TransitionModel &trans_model,
                             bool reorder,
                             const std::vector<int32> &phone_sequence,
                             std::vector<int32> *alignment);




}  // namespace kaldi

#endif

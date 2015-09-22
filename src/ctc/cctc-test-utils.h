// ctc/ctc-test-utils.h

// Copyright      2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CTC_CCTC_TEST_UTILS_H_
#define KALDI_CTC_CCTC_TEST_UTILS_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "ctc/cctc-transition-model.h"
#include "ctc/cctc-graph.h"
#include "ctc/language-model.h"
#include "ctc/cctc-supervision.h"
#include "ctc/cctc-training.h"
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"


namespace kaldi {
namespace ctc {
// This file contains various things that are only needed in testing code.

/// This function generates 'fake' data for testing language models (it's actually
/// obtained from the characters in a file of C++ source code).  It outputs
/// to 'vocab_size' a vocabulary size between 64 and 127, and it outputs to
/// 'data' and 'validation_data' two sequences of 'sentences'.  Each sentence is
/// a sequence of 'words' (actually in the CTC setup thes will really represent
/// phones, but in language modeling terminology they are words).  The words
/// range from 1 to vocab_size.  Sentences are not guaranteed to be nonemtpy.
void GenerateLanguageModelingData(
    int32 *vocab_size,
    std::vector<std::vector<int32> > *data,
    std::vector<std::vector<int32> > *validation_data);

/// This function, modified from GenRandContextDependency(), generates a random
/// context-dependency tree that only has left-context, and ensures that all
/// pdf-classes are numbered zero (as required for the CCTC code), as opposed
/// to the normal range [0, 1, 2] for traditional 3-state HMMs.
ContextDependency *GenRandContextDependencySpecial(
    const std::vector<int32> &phone_ids);


/// Randomly generates an object of type CctcTransitionModel.
void GenerateCctcTransitionModel(CctcTransitionModel *trans_model);


}  // namespace ctc
}  // namespace kaldi

#endif


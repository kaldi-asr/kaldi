// hmm/simple-hmm-utils.h

// Copyright 2009-2011  Microsoft Corporation
//                2016  Vimal Manohar (Johns Hopkins University)

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

#ifndef KALDI_HMM_SIMPLE_HMM_UTILS_H_
#define KALDI_HMM_SIMPLE_HMM_UTILS_H_

#include "hmm/hmm-utils.h"
#include "simplehmm/simple-hmm.h"
#include "fst/fstlib.h"

namespace kaldi {

fst::VectorFst<fst::StdArc>* GetHTransducer(
    const SimpleHmm &model, 
    BaseFloat transition_scale = 1.0, BaseFloat self_loop_scale = 1.0);

/**
  * Converts the SimpleHmm into H tranducer; result owned by caller.
  * Caution: our version of
  * the H transducer does not include self-loops; you have to add those later.
  * See \ref hmm_graph_get_h_transducer.  The H transducer has on the
  * input transition-ids.
  * The output side contains the one-indexed mappings of pdf_ids, typically
  * just pdf_id + 1.
  */
fst::VectorFst<fst::StdArc>*
GetSimpleHmmAsFst (const SimpleHmm &model, 
                   BaseFloat transition_scale = 1.0, 
                   BaseFloat self_loop_scale = 1.0);


}  // end namespace kaldi

#endif

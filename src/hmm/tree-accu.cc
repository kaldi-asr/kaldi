// hmm/tree-accu.cc

// Copyright 2009-2011 Microsoft Corporation

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
#include "util/kaldi-io.h"
#include "hmm/tree-accu.h"
#include "hmm/hmm-utils.h"

namespace kaldi {


void AccumulateTreeStats(const TransitionModel &trans_model,
                         BaseFloat var_floor,
                         int N,  // context window size.
                         int P,  // central position.
                         const std::vector<int32> &ci_phones,
                         const std::vector<int32> &alignment,
                         const Matrix<BaseFloat> &features,
                         std::map<EventType, GaussClusterable*> *stats) {

  KALDI_ASSERT(IsSortedAndUniq(ci_phones));
  std::vector<std::vector<int32> > split_alignment;
  bool ans = SplitToPhones(trans_model, alignment, &split_alignment);
  if (!ans) {
    KALDI_WARN << "AccumulateTreeStats: alignment appears to be bad, not using it\n";
    return;
  }
  int cur_pos = 0;
  int dim = features.NumCols();
  KALDI_ASSERT(features.NumRows() == static_cast<int32>(alignment.size()));
  for (int i = -N; i < static_cast<int>(split_alignment.size()); i++) {
    // consider window starting at i, only if i+P is within
    // list of phones.
    if (i + P >= 0 && i + P < static_cast<int>(split_alignment.size())) {
      int32 central_phone = trans_model.TransitionIdToPhone(split_alignment[i+P][0]);
      bool is_ctx_dep = ! std::binary_search(ci_phones.begin(),
                                             ci_phones.end(),
                                             central_phone);
      EventType evec;
      for (int j = 0; j < N; j++) {
        int phone;
        if (i + j >= 0 && i + j < static_cast<int>(split_alignment.size()))
          phone = trans_model.TransitionIdToPhone(split_alignment[i+j][0]);
        else
          phone = 0;  // ContextDependency class uses 0 to mean "out of window";
        // we also set the phone arbitrarily to 0

        // Don't add stuff to the event that we don't "allow" to be asked, due
        // to the central phone being context-independent: check "is_ctx_dep".
        // Why not just set the value to zero in this
        // case?  It's for safety.  By omitting the key from the event, we
        // ensure that there is no way a question can ever be asked that might
        // give an inconsistent answer in tree-training versus graph-building.
        // [setting it to zero would have the same effect given the "normal"
        // recipe but might be less robust to changes in tree-building recipe].
        if (is_ctx_dep || j == P)
          evec.push_back(std::make_pair<EventKeyType, EventValueType>(j, phone));
      }
      for (int j = 0; j < static_cast<int>(split_alignment[i+P].size());j++) {
        // for central phone of this window...
        EventType evec_more(evec);
        int32 pdf_class = trans_model.TransitionIdToPdfClass(split_alignment[i+P][j]);
        // pdf_class will normally by 0, 1 or 2 for 3-state HMM.
        std::pair<EventKeyType, EventValueType> pr(kPdfClass, pdf_class);
        evec_more.push_back(pr);
        std::sort(evec_more.begin(), evec_more.end());  // these must be sorted!
        if (stats->count(evec_more) == 0)
          (*stats)[evec_more] = new GaussClusterable(dim, var_floor);
        
        BaseFloat weight = 1.0;
        (*stats)[evec_more]->AddStats(features.Row(cur_pos), weight);
        cur_pos++;
      }
    }
  }
  KALDI_ASSERT(cur_pos == static_cast<int>(alignment.size()));
}





}  // end namespace kaldi

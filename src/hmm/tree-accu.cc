// hmm/tree-accu.cc

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
#include "util/kaldi-io.h"
#include "hmm/tree-accu.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

static int32 MapPhone(const std::vector<int32> *phone_map,
                      int32 phone) {
  if (phone == 0 || phone_map == NULL) return phone;
  else if (phone < 0 || phone >= phone_map->size()) {
    KALDI_ERR << "Out-of-range phone " << phone << " bad --phone-map option?";
  }
  return (*phone_map)[phone];
}


void AccumulateTreeStats(const TransitionModel &trans_model,
                         BaseFloat var_floor,
                         int N,  // context window size.
                         int P,  // central position.
                         const std::vector<int32> &ci_phones,
                         const std::vector<int32> &alignment,
                         const Matrix<BaseFloat> &features,
                         const std::vector<int32> *phone_map,
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
      int32 central_phone =
          MapPhone(phone_map,
                   trans_model.TransitionIdToPhone(split_alignment[i+P][0]));
      bool is_ctx_dep = ! std::binary_search(ci_phones.begin(),
                                             ci_phones.end(),
                                             central_phone);
      EventType evec;
      for (int j = 0; j < N; j++) {
        int phone;
        if (i + j >= 0 && i + j < static_cast<int>(split_alignment.size()))
          phone =
              MapPhone(phone_map,
                       trans_model.TransitionIdToPhone(split_alignment[i+j][0]));
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
          evec.push_back(std::make_pair(static_cast<EventKeyType>(j), static_cast<EventValueType>(phone)));
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


void ReadPhoneMap(std::string phone_map_rxfilename,
                  std::vector<int32> *phone_map) {
  phone_map->clear();
  // phone map file has format e.g.:
  // 1 1
  // 2 1
  // 3 2
  // 4 2
  std::vector<std::vector<int32> > vec;  // vector of vectors, each with two elements
  // (if file has right format). first is old phone, second is new phone
  if (!ReadIntegerVectorVectorSimple(phone_map_rxfilename, &vec))
    KALDI_ERR << "Error reading phone map from " <<
        PrintableRxfilename(phone_map_rxfilename);
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].size() != 2 || vec[i][0]<=0 || vec[i][1]<=0 ||
       (vec[i][0]<static_cast<int32>(phone_map->size()) &&
        (*phone_map)[vec[i][0]] != -1))
      KALDI_ERR << "Error reading phone map from "
                 <<   PrintableRxfilename(phone_map_rxfilename)
                 << " (bad line " << i << ")";
    if (vec[i][0]>=static_cast<int32>(phone_map->size()))
      phone_map->resize(vec[i][0]+1, -1);
    KALDI_ASSERT((*phone_map)[vec[i][0]] == -1);
    (*phone_map)[vec[i][0]] = vec[i][1];
  }
  if (phone_map->empty()) {
    KALDI_ERR << "Read empty phone map from "
              << PrintableRxfilename(phone_map_rxfilename);
  }
}



}  // end namespace kaldi

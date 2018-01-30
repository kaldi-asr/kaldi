// hmm/hmm-utils.cc

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

#include <vector>

#include "idlakfeat/hmm-utils-idlak.h"
#include "hmm/hmm-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-lib.h"

namespace kaldi {

// Note: Most of the code in GetPhoneWindows also appears in ConvertAlignment
bool GetPhoneWindows(const TransitionModel &trans_model,
                     const std::vector<int32> &alignment,
                     int32 context_width,
                     int32 central_pos,
                     bool per_frame,
                     std::vector< std::vector<int32> > *phone_windows,
                     std::vector<int32> *phones) {
  KALDI_ASSERT(phone_windows != NULL);
  phone_windows->clear();

  std::vector< std::vector<int32> > split;  // split into phones

  if (!SplitToPhones(trans_model, alignment, &split))
    return false;

  int32 num_phones = split.size();  // Number of distinct phones in alignment
  std::vector<int32> tmp_phones(num_phones);
  for (int32 i = 0; i < num_phones; i++) {
    KALDI_ASSERT(!split[i].empty());
    tmp_phones[i] = trans_model.TransitionIdToPhone(split[i][0]);
  }

  for (int32 win_start = -context_width;
      win_start < static_cast<int32>(num_phones+context_width);
      win_start++) {  // start of a context window.
    int32 current_phone_pos = win_start + central_pos;
    if (current_phone_pos >= 0 && current_phone_pos  < num_phones) {
      std::vector<int32> phone_window(context_width, 0);
      for (int32 offset = 0; offset < context_width; offset++)
        if ((win_start+offset) >= 0 && (win_start+offset) < num_phones)
          phone_window[offset] = tmp_phones[win_start+offset];
      if (per_frame) {
        int32 num_frames = split[current_phone_pos].size();
        for (int32 i = 0; i < num_frames; ++i)
          phone_windows->push_back(phone_window);
      } else {
        phone_windows->push_back(phone_window);
      }
    }
  }

  if (per_frame) {
    KALDI_ASSERT(phone_windows->size() == alignment.size());
  } else {
    KALDI_ASSERT(phone_windows->size() == num_phones);
  }

  if (phones != NULL) {
    if (per_frame) {
      phones->clear();
      for (int32 i = 0; i < num_phones; ++i) {
        int32 num_frames = split[i].size();
        for (int32 f = 0; f < num_frames; ++f)
          phones->push_back(tmp_phones[i]);
      }
      KALDI_ASSERT(phones->size() == alignment.size());
    } else {
      (*phones) = tmp_phones;
    }
  }
  return true;
}


bool ConvertFullCtxAlignment(const TransitionModel &old_trans_model,
                             const TransitionModel &new_trans_model,
                             const ContextDependencyInterface &new_ctx_dep,
                             const std::vector<int32> &tid_ali,
                             const std::vector< std::vector <int32> > &full_ali,
                             std::vector<int32> *new_tid_ali) {
  KALDI_ASSERT(new_tid_ali != NULL);
  KALDI_ASSERT(tid_ali.size() == full_ali.size() &&
               "Regular and full-context alignments must be of same size.");
  new_tid_ali->clear();
  int32 num_frames = tid_ali.size(),
      ctx_width = new_ctx_dep.ContextWidth(),
      central_pos = new_ctx_dep.CentralPosition();

  for (int32 i = 0; i < num_frames; ++i) {
    KALDI_ASSERT(full_ali[i].size() == ctx_width);
    int32 old_tid = tid_ali[i],
        phone = old_trans_model.TransitionIdToPhone(old_tid),
        old_tstate = old_trans_model.TransitionIdToTransitionState(old_tid);
    int32 forward_pdf_class = 
        old_trans_model.TransitionStateToForwardPdfClass(old_tstate),
        self_loop_pdf_class =
        old_trans_model.TransitionStateToSelfLoopPdfClass(old_tstate);
    int32 hmm_state = old_trans_model.TransitionIdToHmmState(old_tid),
        trans_idx = old_trans_model.TransitionIdToTransitionIndex(old_tid);
    KALDI_ASSERT(full_ali[i][central_pos] == phone &&
                 "Mismatched regular and full-context alignments.");
    KALDI_ASSERT((*full_ali[i].end()) == forward_pdf_class &&
                 "Mismatched regular and full-context alignments.");
    std::vector<int32> fullctx(full_ali[i].begin(), full_ali[i].end()-1);
    int32 new_forward_pdf, new_self_loop_pdf;
    if (!new_ctx_dep.Compute(fullctx, forward_pdf_class, &new_forward_pdf) 
        || !new_ctx_dep.Compute(fullctx, self_loop_pdf_class, &new_self_loop_pdf)) {
      std::ostringstream ctx_ss;
      WriteIntegerVector(ctx_ss, false, full_ali[i]);
      KALDI_ERR << "Decision tree did not produce an answer for: pdf-class = "
                << forward_pdf_class << " context window = " << ctx_ss.str();
    }
    int32 new_trans_state = 
        new_trans_model.TupleToTransitionState(phone,
                                               hmm_state,
                                               new_forward_pdf,
                                               new_self_loop_pdf);
    int32 new_tid = new_trans_model.PairToTransitionId(new_trans_state,
                                                       trans_idx);
    new_tid_ali->push_back(new_tid);
  }

  KALDI_ASSERT(new_tid_ali->size() == tid_ali.size());
  return true;
}


} // namespace kaldi

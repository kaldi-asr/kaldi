// hmm/simple-hmm-utils.cc

// Copyright 2009-2011  Microsoft Corporation
//                2016  Vimal Manohar

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

#include "simplehmm/simple-hmm-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-lib.h"

namespace kaldi {

fst::VectorFst<fst::StdArc>* GetHTransducer(
    const SimpleHmm &model,
    BaseFloat transition_scale, BaseFloat self_loop_scale) {
  using namespace fst;
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  VectorFst<Arc> *fst = GetSimpleHmmAsFst(model, transition_scale, 
                                          self_loop_scale);
  
  for (StateIterator<MutableFst<Arc> > siter(*fst);
       !siter.Done(); siter.Next()) {
    Arc::StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel == 0) {
        KALDI_ASSERT(arc.olabel == 0);
        continue;
      }

      KALDI_ASSERT(arc.ilabel == arc.olabel && 
                   arc.ilabel <= model.NumTransitionIds());

      arc.olabel = model.TransitionIdToPdf(arc.ilabel) + 1;
      aiter.SetValue(arc);
    }
  }

  return fst;
}

fst::VectorFst<fst::StdArc> *GetSimpleHmmAsFst(
    const SimpleHmm &model, 
    BaseFloat transition_scale, BaseFloat self_loop_scale) {
  using namespace fst;
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  KALDI_ASSERT(model.NumPdfs() > 0);
  const HmmTopology &topo = model.GetTopo();
  // This special Hmm has only one phone
  const HmmTopology::TopologyEntry &entry  = topo.TopologyForPhone(1);

  VectorFst<StdArc> *ans = new VectorFst<StdArc>;

  // Create a mini-FST with a superfinal state [in case we have emitting
  // final-states, which we usually will.]

  std::vector<StateId> state_ids;
  for (size_t i = 0; i < entry.size(); i++)
    state_ids.push_back(ans->AddState());
  KALDI_ASSERT(state_ids.size() > 1);  // Or invalid topology entry.
  ans->SetStart(state_ids[0]);
  StateId final_state = state_ids.back();
  ans->SetFinal(final_state, Weight::One());

  for (int32 hmm_state = 0;
       hmm_state < static_cast<int32>(entry.size());
       hmm_state++) {
    int32 pdf_class = entry[hmm_state].forward_pdf_class;
    int32 self_loop_pdf_class = entry[hmm_state].self_loop_pdf_class;
    KALDI_ASSERT(self_loop_pdf_class == pdf_class);

    if (pdf_class != kNoPdf) {
      KALDI_ASSERT(pdf_class < model.NumPdfs());
    }

    int32 trans_idx;
    for (trans_idx = 0;
         trans_idx < static_cast<int32>(entry[hmm_state].transitions.size());
         trans_idx++) {
      BaseFloat log_prob;
      Label label;
      int32 dest_state = entry[hmm_state].transitions[trans_idx].first;

      if (pdf_class == kNoPdf) {
        // no pdf, hence non-estimated probability.  very unusual case.  [would
        // not happen with normal topology] .  There is no transition-state
        // involved in this case.
        KALDI_ASSERT(hmm_state != dest_state);
        log_prob = transition_scale
                   * Log(entry[hmm_state].transitions[trans_idx].second);
        label = 0;
      } else {  // normal probability.
        int32 trans_state =
            model.TupleToTransitionState(1, hmm_state, pdf_class, pdf_class);
        int32 trans_id =
            model.PairToTransitionId(trans_state, trans_idx);
          
        log_prob = model.GetTransitionLogProb(trans_id);

        if (hmm_state == dest_state) 
          log_prob *= self_loop_scale;
        else 
          log_prob *= transition_scale;
        // log_prob is a negative number (or zero)...
        label = trans_id;
      }
      ans->AddArc(state_ids[hmm_state],
                  Arc(label, label, Weight(-log_prob),
                  state_ids[dest_state]));
    }
  }

  fst::RemoveEpsLocal(ans);  // this is safe and will not blow up.
  // Now apply probability scale.
  // We waited till after the possible weight-pushing steps,
  // because weight-pushing needs "real" weights in order to work.
  // ApplyProbabilityScale(config.transition_scale, ans);
  return ans;
}

}  // end namespace kaldi

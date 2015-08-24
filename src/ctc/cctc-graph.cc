// cctc/cctc-graph.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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


#include "ctc/cctc-graph.h"

namespace kaldi {
namespace ctc {


CctcDeterministicOnDemandFst::CctcDeterministicOnDemandFst(
    const CctcTransitionModel &trans_model,
    BaseFloat phone_language_model_weight):
    trans_model_(trans_model),
    phone_language_model_weight_(phone_language_model_weight) { }


// Note: we've put the testing code for this into cctc-transition-model-test.cc.
bool CctcDeterministicOnDemandFst::GetArc(StateId s, Label ilabel, fst::StdArc* oarc) {
  int32 history_state = s, phone_or_blank = ilabel - 1;
  oarc->nextstate = trans_model_.GetNextHistoryState(history_state,
                                                     phone_or_blank);
  oarc->ilabel = ilabel;
  oarc->olabel = trans_model_.GetGraphLabel(history_state, phone_or_blank);
  if (phone_language_model_weight_ == 0.0) {
    oarc->weight = Weight::One();
  } else {
    oarc->weight = Weight(phone_language_model_weight_ *
                          -log(trans_model_.GetLmProb(history_state,
                                                      phone_or_blank)));
  }
  // If the ilabel was not in the correct range (from one to num-phones + 1)
  // we would have crashed in one of the calls to the CctcTransitionModel
  // object.
  return true;
}

void ShiftPhonesAndAddBlanks(fst::StdVectorFst *fst) {
  typedef fst::MutableArcIterator<fst::StdVectorFst > IterType;

  fst::StdArc self_loop_arc;
  self_loop_arc.ilabel = 1;  // blank plus one.
  self_loop_arc.olabel = 0;  // epsilon
  self_loop_arc.weight = fst::StdArc::Weight::One();
  
  int32 num_states = fst->NumStates();
  for (int32 state = 0; state < num_states; state++) {
    for (IterType aiter(fst, state); !aiter.Done(); aiter.Next()) {
      fst::StdArc arc(aiter.Value());
      if (arc.ilabel != 0) {
        arc.ilabel++;
        aiter.SetValue(arc);       
      }
    }
    self_loop_arc.nextstate = state;
    fst->AddArc(state, self_loop_arc);
  }
}

void CreateCctcDecodingFst(
    const CctcTransitionModel &trans_model,
    BaseFloat phone_language_model_weight,
    const fst::StdVectorFst &phone_and_blank_fst,
    fst::StdVectorFst *decoding_fst) {
  CctcDeterministicOnDemandFst cctc_fst(trans_model,
                                        phone_language_model_weight);
  // the next line does:
  // *decoding_fst = Compose(Inverse(cctc_fst), phone_and_blank_fst);
  ComposeDeterministicOnDemandInverse(phone_and_blank_fst, &cctc_fst,
                                      decoding_fst);
}




}  // namespace ctc
}  // namespace kaldi

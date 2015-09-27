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
#include "lat/lattice-functions.h"  // for PruneLattice
#include "lat/minimize-lattice.h"   // for minimization
#include "lat/push-lattice.h"       // for minimization

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


// This function, not declared in the header, is used inside
// DeterminizeLatticePhonePrunedCctc.  It is a CCTC version of
// DeterminizeLatticeInsertPhones(), which is defined in
// ../lat/determinize-lattice-pruned.cc.
template<class Weight>
typename fst::ArcTpl<Weight>::Label DeterminizeLatticeInsertPhonesCctc(
    const CctcTransitionModel &trans_model,
    fst::MutableFst<fst::ArcTpl<Weight> > *fst) {
  using namespace fst;
  // Define some types.
  typedef ArcTpl<Weight> Arc;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  // Work out the first phone symbol. This is more related to the phone
  // insertion function, so we put it here and make it the returning value of
  // DeterminizeLatticeInsertPhones(). 
  Label first_phone_label = fst::HighestNumberedInputSymbol(*fst) + 1;

  // Insert phones here.
  for (StateIterator<MutableFst<Arc> > siter(*fst);
       !siter.Done(); siter.Next()) {
    StateId state = siter.Value();
    if (state == fst->Start())
      continue;
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();

      // Note: the words are on the input symbol side and transition-id's are on
      // the output symbol side.
      int32 phone;
      if ((arc.olabel != 0) &&
          (phone = trans_model.GraphLabelToPhone(arc.olabel)) > 0) {
        if (arc.ilabel == 0) {
          // If there is no word on the arc, insert the phone directly.
          arc.ilabel = first_phone_label + phone;
        } else {
          // Otherwise, add an additional arc.
          StateId additional_state = fst->AddState();
          StateId next_state = arc.nextstate;
          arc.nextstate = additional_state;
          fst->AddArc(additional_state,
                      Arc(first_phone_label + phone, 0,
                          Weight::One(), next_state));
        }
      }
      aiter.SetValue(arc);
    }
  }
  return first_phone_label;
}



// this function, not declared in the header, is a 'CCTC' version of
// DeterminizeLatticePhonePrunedFirstPass(), as defined in
// ../lat/determinize-lattice-pruned.cc.  It's only called from
// DeterminizeLatticePhonePrunedCctc().
template<class Weight, class IntType>
bool DeterminizeLatticePhonePrunedFirstPassCctc(
    const CctcTransitionModel &trans_model,
    double beam,
    fst::MutableFst<fst::ArcTpl<Weight> > *fst,
    const fst::DeterminizeLatticePrunedOptions &opts) {
  using namespace fst;
  // First, insert the phones.
  typename ArcTpl<Weight>::Label first_phone_label =
      DeterminizeLatticeInsertPhonesCctc(trans_model, fst);
  TopSort(fst);
  
  // Second, do determinization with phone inserted.
  bool ans = DeterminizeLatticePruned<Weight>(*fst, beam, fst, opts);

  // Finally, remove the inserted phones.
  // We don't need a special 'CCTC' version of this function.
  DeterminizeLatticeDeletePhones(first_phone_label, fst);
  TopSort(fst);

  return ans;
}


// this function, not declared in the header, is a 'CCTC' version of
// DeterminizeLatticePhonePruned(), as defined in ../lat/determinize-lattice-pruned.cc.
// It's only called from DeterminizeLatticePhonePrunedWrapperCctc().
template<class Weight, class IntType>
bool DeterminizeLatticePhonePrunedCctc(
    const CctcTransitionModel &trans_model,
    fst::MutableFst<fst::ArcTpl<Weight> > *ifst,
    double beam,
    fst::MutableFst<fst::ArcTpl<fst::CompactLatticeWeightTpl<Weight, IntType> > >
      *ofst,
    fst::DeterminizeLatticePhonePrunedOptions opts) {
  using namespace fst;
  // Returning status.
  bool ans = true;

  // Make sure at least one of opts.phone_determinize and opts.word_determinize
  // is not false, otherwise calling this function doesn't make any sense.
  if ((opts.phone_determinize || opts.word_determinize) == false) {
    KALDI_WARN << "Both --phone-determinize and --word-determinize are set to "
               << "false, copying lattice without determinization.";
    // We are expecting the words on the input side.
    ConvertLattice<Weight, IntType>(*ifst, ofst, false);
    return ans;
  }

  // Determinization options.
  DeterminizeLatticePrunedOptions det_opts;
  det_opts.delta = opts.delta;
  det_opts.max_mem = opts.max_mem;

  // If --phone-determinize is true, do the determinization on phone + word
  // lattices.
  if (opts.phone_determinize) {
    KALDI_VLOG(1) << "Doing first pass of determinization on phone + word "
                  << "lattices."; 
    ans = DeterminizeLatticePhonePrunedFirstPassCctc<Weight, IntType>(
        trans_model, beam, ifst, det_opts) && ans;

    // If --word-determinize is false, we've finished the job and return here.
    if (!opts.word_determinize) {
      // We are expecting the words on the input side.
      ConvertLattice<Weight, IntType>(*ifst, ofst, false);
      return ans;
    }
  }

  // If --word-determinize is true, do the determinization on word lattices.
  if (opts.word_determinize) {
    KALDI_VLOG(1) << "Doing second pass of determinization on word lattices.";
    ans = DeterminizeLatticePruned<Weight, IntType>(
        *ifst, beam, ofst, det_opts) && ans;
  }

  // If --minimize is true, push and minimize after determinization.
  if (opts.minimize) {
    KALDI_VLOG(1) << "Pushing and minimizing on word lattices.";
    ans = fst::PushCompactLatticeStrings<Weight, IntType>(ofst) && ans;
    ans = fst::PushCompactLatticeWeights<Weight, IntType>(ofst) && ans;
    ans = fst::MinimizeCompactLattice<Weight, IntType>(ofst) && ans;
  }

  return ans;
}


bool DeterminizeLatticePhonePrunedWrapperCctc(
    const CctcTransitionModel &trans_model,
    fst::MutableFst<kaldi::LatticeArc> *ifst,
    double beam,
    fst::MutableFst<kaldi::CompactLatticeArc> *ofst,
    fst::DeterminizeLatticePhonePrunedOptions opts) {
  using namespace fst;
  bool ans = true;
  Invert(ifst);
  if (ifst->Properties(fst::kTopSorted, true) == 0) {
    if (!TopSort(ifst)) {
      // Cannot topologically sort the lattice -- determinization will fail.
      KALDI_ERR << "Topological sorting of state-level lattice failed (probably"
                << " your lexicon has empty words or your LM has epsilon cycles"
                << ").";
    }
  }
  ILabelCompare<kaldi::LatticeArc> ilabel_comp;
  ArcSort(ifst, ilabel_comp);
  ans = DeterminizeLatticePhonePrunedCctc<kaldi::LatticeWeight, kaldi::int32>(
      trans_model, ifst, beam, ofst, opts);
  Connect(ofst);
  return ans;
}



}  // namespace ctc
}  // namespace kaldi

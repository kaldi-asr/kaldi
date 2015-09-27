// ctc/cctc-graph.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CTC_CCTC_GRAPH_H_
#define KALDI_CTC_CCTC_GRAPH_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"
#include "ctc/cctc-transition-model.h"
#include "fstext/deterministic-fst.h"
#include "lat/determinize-lattice-pruned.h"

namespace kaldi {
namespace ctc {

  
/**  This function adds one to all the phones to the FST and adds self-loops
     for the the optional blank symbols to all of its states.

     @param [in,out]          The FST we modify.  At entry, the symbols on its input side
                              must be phones in the range [1, num_phones]
                              the output-side symbols will be left as they are.
                              If this represents a decoding graph you'll probably
                              want to have determinized this with disambiguation symbols
                              in place, then removed the disambiguation symbols and minimized
                              it.

                              What this function does is to add 1 to all nonzero
                              input symbols on arcs (to convert phones to
                              phones-plus-one), then at each state of the
                              modified FST, add a self-loop with a 1
                              (blank-plus-one) on the input and 0 (epsilon) on
                              the output.
*/
void ShiftPhonesAndAddBlanks(fst::StdVectorFst *fst);

/**  This function creates an FST that we can use for decoding with CTC.
     Internally it composes on the left with an object of type
     CctcDeterministicOnDemandFst.

     @param [in] trans_model  The CCTC transition model that defines the graph
                              labels.
     @param [in] phone_language_model_weight   This parameter should usually be set
                              to zero, but if you set it to nonzero then to each
                              real-phone arc in the output FST, a cost equal to
                              phone_language_model_weight times the corresponding
                              -log(phone-language-model-prob) will be added to the
                              arc.  In decoding we won't be adding in any phone LM probs because
                              the spirit of the framework is to use them only in training;
                              and in testing, use the proper language model instead; but
                              if you want to add them with a small weight (e.g. 0.1 or 0.2)
                              you can set this parameter accordingly.
     @param [in] phone_and_blank_fst   The input FST, with (phones and blanks) plus one
                              its input side, and anything you want (e.g. words) on its
                              output side.  This might be the output of AddBlanksToFst().

     @param [out] decoding_fst The output FST (will be larger than the input FST
                              due to the effects of left context).  The symbols
                              on the input side are (one-based) graph-labels, as
                              defined by class CctcTransitionModel.  The symbols
                              on the output side will be whatever they were
                              before, e.g. words.
*/
void CreateCctcDecodingFst(const CctcTransitionModel &trans_model,
                           BaseFloat phone_language_model_weight,
                           const fst::StdVectorFst &phone_and_blank_fst,
                           fst::StdVectorFst *decoding_fst);


/**
   This class wraps a CctcTransitionModel as a DeterministicOnDemandFst;
   it is used by function CreateCctcDecodingFst to create a decoding-graph
   suitable for use with a CCTC model.
   You compose with this (actually with the inverse of this, on the left;
   see function ComposeDeterministicOnDemandInverse()) to convert a graph labeled with
   phones-plus-one on its input to a graph with cctc "graph-labels" on its input.
 */
class CctcDeterministicOnDemandFst:
      public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Initialize this object.  phone_language_model_weight may be configurable on
  // the command line.  It's a weight on the phone-language-model log-probs, and
  // its range would normally be between zero and one, but the default value
  // should be zero.
  CctcDeterministicOnDemandFst(const CctcTransitionModel &trans_model,
                               BaseFloat phone_language_model_weight);
  
  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return trans_model_.InitialHistoryState(); }

  // Note: in the CCTC framework we don't really model final-probs in any non-trivial
  // way, since the probabibility of the end of the phone sequence is one if we saw
  // the end of the acoustic sequence, and zero otherwise.
  virtual Weight Final(StateId s) { return Weight::One(); }
  
  // The ilabel is a (phone-or-blank) plus one.  The state-id is the
  // history-state in the trans_model_.  The interface of GetArc requires ilabel
  // to be nonzero (not epsilon).
  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);
  
 private:
  const CctcTransitionModel &trans_model_;
  BaseFloat phone_language_model_weight_;
};


/** This is a Cctc version of the function DeterminizeLatticePhonePrunedWrapper,
    declared in ../lat/determinize-lattice-pruned.h.  It can be used
    as a top-level interface to all the determinization code.  It's destructive
    of its input.
*/
bool DeterminizeLatticePhonePrunedWrapperCctc(
    const CctcTransitionModel &trans_model,
    fst::MutableFst<kaldi::LatticeArc> *ifst,
    double prune,
    fst::MutableFst<kaldi::CompactLatticeArc> *ofst,
    fst::DeterminizeLatticePhonePrunedOptions opts);


}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_GRAPH_H_

// decoder/grammar-fst.cc

// Copyright   2018  Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/grammar-fst.h"
#include "fstext/grammar-context-fst.h"

namespace fst {


void GrammarFstConfig::Check() const {
  if (nonterm_phones_offset <= 0)
    KALDI_ERR << "--nonterm-phones-offset must be set to a positive value.";
}

GrammarFst::GrammarFst(
    const GrammarFstConfig &config,
    const ConstFst<StdArc> &top_fst,
    const std::vector<std::pair<Label, const ConstFst<StdArc> *> > &ifsts):
    config_(config),
    top_fst_(&top_fst),
    ifsts_(ifsts) {
  config.Check();
  encoding_multiple_ = GetEncodingMultiple(config_.nonterm_phones_offset);
}


// This class contains the implementation of the function
// PrepareForGrammarFst(), which is declared in grammar-fst.h.
class GrammarFstPreparer {
 public:
  using FST = VectorFst<StdArc>;
  using Arc = StdArc;
  using StateId = Arc::StateId;
  using Label = Arc::Label;

  GrammarFstPreparer(int32 nonterm_phones_offset,
                     VectorFst<StdArc> *fst):
      nonterm_phones_offset_(nonterm_phones_offset),
      fst_(fst),
      num_new_states_(0) { }

  void Prepare() {
    if (fst->Start() == kNoStateId) {
      KALDI_ERR << "FST has no states.";
    }
    if (fst->Properties(fst::kILabelSorted, true) == 0) {
      // Make sure the FST is sorted on ilabel.
      fst::ILabelCompare<StdArc> ilabel_comp;
      fst::ArcSort(std_lm_fst, ilabel_comp);
    }
    for (StateId s = 0; s < fst_->NumStates(); s++) {
      if (IsSpecialState(s)) {
        if (TransitionsToMultipleInstances(s)) {
          InsertEpsilonsForState(s);
          // now all olabels from s will be epsilon, so no need
          // to add final-prob to it.
        } else {
          MakeSureStateHasFinalProb(s);
        }
      }
    }
  }

 private:

  bool IsSpecialState(StateId s) const {
    bool ans = false;
    for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel >= kNontermBigNumber) // 1 million
        return true;
    }
    return false;
  }

  // Returns true if the state has arcs out of it that will transition to
  // multiple FST instance (for example: one to this same FST, and one to the
  // FST for a user-defined nonterminal).  This function also does some other
  // checks that these nonterminals have the expected structure, and will crash
  // if the checks fail.
  bool TransitionsToMultipleInstances(StateId s) const {
    // dest_nonterminals will encode something about which other FSTs this FST might
    // transition to.  It will contain:
    //   0 if this state has any transition leaving it that is to this same FST
    //     (i.e. an epsilon or a normal transition-id)
    //   #nontermXXX if this state has an ilabel which would be decoded as
    //       the pair (#nontermXXX, p1).  Here, #nontermXXX is either an
    //       inbuilt nonterminal like #nonterm_begin, #nonterm_end or #nonterm_reenter,
    //       or a user-defined nonterminal like #nonterm:foo.
    std::set<int32> dest_nonterminals;
    // normally we'll have encoding_multiple = 1000, big_number = 1000000.
    int32 encoding_multiple = GetEncodingMultiple(config_.nonterm_phones_offset),
        big_number = kNontermBigNumber,
        nonterm_phones_offset = config_.nonterm_phones_offset;
    for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      int32 nonterminal;
      if (arc.ilabel < big_number) {
        nonterminal = 0;
      } else {
        nonterminal = (arc.ilabel - big_number) / encoding_multiple;
        if (nonterminal <= nonterm_phones_offset) {
          KALDI_ERR << "Problem decoding nonterminal symbol "
              "(wrong --nonterm-phones-offset option?), ilabel="
                    << arc.ilabel;
        }
        if (nonterminal >= nonterm_phones_offset + kNontermUserDefined) {
          // This is a user-defined symbol.  Check that the destination state of
          // this arc has arcs with kNontermReenter on them.  We'll separately
          // check that such states don't have other types of arcs coming from
          // them (search for kNontermReenter below), so it's sufficient to
          // check the first arc.
          ArcIterator<FST> next_aiter(*fst_, arc.nextstate);
          if (next_aiter.Done())
            KALDI_ERR << "Destination state of a user-defined nonterminal "
                "has no arcs leaving it.";
          const Arc &next_arc = next_aiter.Value();
          int32 next_nonterminal = (next_arc.ilabel - big_number) /
              encoding_multiple;
          if (next_nonterminal != nonterm_phones_offset + kNontermReenter) {
            KALDI_ERR << "Expected arcs with user-defined nonterminals to be "
                "followed by arcs with kNontermReenter.";
          }
        }
      }
      dest_nonterminals.insert(nonterminal);
    }
    if (dest_nonterminals.size() > 1) {
      // OK, it looks like it will have transitions to multiple FST instances in
      // the actual graph.  Just do some checking, that there is nothing
      // unexpected in there.
      for (std::set<int32>::const_iterator iter = dest_nonterminals.begin();
           iter != dest_nonterminals.end(); ++iter) {
        int32 nonterminal = *iter;
        if (nonterminal == nonterm_phones_offset + kNontermBegin ||
            nonterminal == nonterm_phones_offset + kNontermReenter)
          // we don't expect any state to have symbols (kNontermBegin:p1) on arcs
          // coming out of and also other types of symbol.  The same goes for
          // kNontermReenter.


    }
  }


  int32 nonterm_phones_offset_;
  VectorFst<StdArc> *fst_;
      int32 num_new_states_;
};


void PrepareForGrammarFst(int32 nonterm_phones_offset,
                          VectorFst<StdArc> *fst) {

  fst::ILabelCompare<LatticeArc> ilabel_comp;

}



} // end namespace fst

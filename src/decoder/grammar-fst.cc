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
  InitNonterminalMap();
  InitEntryPoints();
}

void GrammarFst::DecodeSymbol(Label label,
                              int32 *nonterminal_symbol,
                              int32 *left_context_phone) {
  // encoding_multiple will normally equal 1000, and
  // kNontermBigNumber is 1000000.
  int32 big_number = kNontermBigNumber,
      encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_);
  // The following assertion should be optimized out as the condition is
  // statically known.
  KALDI_ASSERT(static_cast<int32>(kBigNumber) %
               static_cast<int32>(kMediumNumber) == 0);

  *nonterminal_symbol = (label - (int32)kNontermBigNumber) / encoding_multiple;
  *left_context_phone = label %  encoding_multiple;
  if (*nonterm_symbol <= nonterm_phones_offset_ ||
      *left_context_phone == 0 ||
      *left_context_phone > nonterm_phones_offset_)
    KALDI_ERR << "Decoding invalid label " << label
              << ": code error or invalid --nonterm-phones-offset?";

}

void GrammarFst::InitNonterminalMap() {
  nonterminal_map_.clear();
  for (size_t i = 0; i < ifsts_.size(); i++) {
    int32 nonterminal = ifsts_[i].first;
    if (nonterminal_map_.count(nonterminal))
      KALDI_ERR << "Nonterminal symbol " << nonterminal_symbol
                << " is paired with two FSTs.";
    if (nonterminal < GetPhoneSymbolFor(kNontermUserDefined))
      KALDI_ERR << "Nonterminal symbol " << nonterminal_symbol
                << " in input pairs, is expected to be >= "
                << GetPhoneSymbolFor(kNontermUserDefined);
    nonterminal_map_[nonterminal] = static_cast<int32>(i);
  }
}


void InitEntryPoints() {
  entry_points_.resize(ifsts_.size());
  for (size_t i = 0; i < ifsts_.size(); i++) {
    const ConstFst &fst = *(ifsts_[i].second);
    InitEntryOrReentryPoints(fst, fst.Start(),
                             GetPhoneSymblFor(kNontermBegin),
                             &(entry_points_[i]));
  }
}


void GrammarFst::InitEntryOrReentryPoints(
    const ConstFst<StdArc> &fst,
    int32 entry_state,
    int32 expected_nonterminal_symbol,
    std::unordered_map<int32, BaseStateId> *phone_to_state) {
  phone_to_state->clear();
  ArcIterator<ConstFst<StdArc> > aiter(fst, entry_state);
  for (; !aiter.Done(); aiter.Next) {
    const Arc &arc = aiter.Value();
    int32 nonterminal, left_context_phone;
    if (arc.ilabel <= (int32)kNontermBigNumber) {
      if (entry_state == fst.Start()) {
        KALDI_ERR << "There is something wrong with the graph; did you forget to "
            "add #nonterm_begin and #nonterm_end to the non-top-level FSTs "
            "before compiling?";
      } else {
        KALDI_ERR << "There is something wrong with the graph; re-entry state is "
            "not as anticipated.";
      }
    }
    DecodeSymbol(arc.ilabel, &nonterminal, &left_context_phone);
    if (nonterminal != expected_nonterminal_symbol) {
      KALDI_ERR << "Expected arcs from this state to have nonterminal-symbol "
                << expected_nonterminal_symbol << ", but got "
                << nonterminal;
    }
    std::pair<int32, BaseStateId> p(left_context_phone, arc.nextstate);
    if (!phone_to_state->insert(p).second) {
      // if it was not successfully inserted, it means there were two arcs with
      // the same left-context phone, which does not make sense; that's an
      // error, likely a code error.
      KALDI_ERR << "Two arcs had the same left-context phone.";
    }
  }
}

ExpandedState *GrammarFst::ExpandState(int32 instance_id,
                                       BaseStateId state_id) {
  int3 big_number = kNontermBigNumber;
  const ConstFst<StdArc> &fst = instances_[instance_id];
  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);
  if (aiter.Done() || aiter.Value().ilabel < big_number) {
    return NULL;   // There was actually no need to expand this state; it was a
                   // normal final state.  We previously ensured that each state
                   // has only 'one type of arc' coming out of it, so checking
                   // the first arc is sufficient.
  }
  ExpandedState *ans = new ExpandedState;

  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber;
  nonterminal = (arc.ilabel - big_number) / encoding_multiple;
  if (nonterminal == GetPhoneSymbolFor(kNontermBegin) ||
      nonterminal == GetPhoneSymbolFor(kNontermReenter)) {
    KALDI_ERR << "Encountered unexpected type of nonterminal while "
        "expanding state.";
  } else if (nonterminal == GetPhoneSymbolFor(kNontermEnd)) {
    if (instance_id == 0)
      KALDI_ERR << "Did not expect #nonterm_end symbol in FST-instance 0.";
    const FstInstance &instance = instances_[instance_id];
    int32 return_instance_id = instance.return_instance;
    ans->dest_fst_instance = return_instance_id;
    for (; !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.value();
      int32 this_nonterminal, left_context_phone;
      DecodeSymbol(arc.ilabel, &this_nonterminal, &left_context_phone);
      KALDI_ASSERT(this_nonterminal == nonterminal &&
                   ">1 nonterminals from a state; did you use "
                   "PrepareForGrammarFst()?");
      std::unordered_map<int32, BaseStateId> &reentry_iter =
          instances_[instance_id].reentry_points.find(left_context_phone);
      if (reentry_iter == instance.reentry_points.end()) {
        KALDI_ERR << "FST with index " << ifst_index
                  << " returns with left-context " << left_context_phone
                  << " but parent FST "
                  << instances_[return_instance_id].ifst_index
                  << " does not support that left-context at state "
                  << instance.return_state;
      }
      arc.ilabel = 0; // we'll make this an epsilon arc in the expanded FST.
      arc.olabel = ... ;
    }

  }

}

// This class contains the implementation of the function
// PrepareForGrammarFst(), which is declared in grammar-fst.h.
class GrammarFstPreparer {
 public:
  using FST = VectorFst<StdArc>;
  using Arc = StdArc;
  using StateId = Arc::StateId;
  using Label = Arc::Label;
  using Weight = Arc::Weight;

  GrammarFstPreparer(int32 nonterm_phones_offset,
                     VectorFst<StdArc> *fst):
      nonterm_phones_offset_(nonterm_phones_offset),
      fst_(fst),
      num_new_states_(0) { }

  void Prepare() {
    if (fst_->Start() == kNoStateId) {
      KALDI_ERR << "FST has no states.";
    }
    if (fst_->Properties(kILabelSorted, true) == 0) {
      // Make sure the FST is sorted on ilabel.
      ILabelCompare<StdArc> ilabel_comp;
      ArcSort(fst_, ilabel_comp);
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
    KALDI_VLOG(2) << "Added " << num_new_states_ << " new states while "
        "preparing for grammar FST.";
  }

 private:

  // Returns true if state 's' has at least one arc coming out of it with
  // a special nonterminal-related symbol on it.
  bool IsSpecialState(StateId s) const {
    for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel >= kNontermBigNumber) // 1 million
        return true;
    }
    return false;
  }

  // If state s doesn't have any final-prob, add a final-prob with cost
  // 1024.0.
  void MakeSureStateHasFinalProb(StateId s) {
    if (fst_->Final(s) == Weight::Zero()) {
      fst_->SetFinal(s, Weight(1024.0));
    }
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
    int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
        big_number = kNontermBigNumber;
    for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      int32 nonterminal;
      if (arc.ilabel < big_number) {
        nonterminal = 0;
      } else {
        nonterminal = (arc.ilabel - big_number) / encoding_multiple;
        if (nonterminal <= nonterm_phones_offset_) {
          KALDI_ERR << "Problem decoding nonterminal symbol "
              "(wrong --nonterm-phones-offset option?), ilabel="
                    << arc.ilabel;
        }
        if (nonterminal >= nonterm_phones_offset_ + kNontermUserDefined) {
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
          if (next_nonterminal != nonterm_phones_offset_ + kNontermReenter) {
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
        if (nonterminal == nonterm_phones_offset_ + kNontermBegin ||
            nonterminal == nonterm_phones_offset_ + kNontermReenter)
          // we don't expect any state which has symbols (kNontermBegin:p1) on
          // arcs coming out of it to also have other types of symbol.  The same
          // goes for kNontermReenter.
          KALDI_ERR << "We do not expect states with arcs of type "
              "kNontermBegin/kNontermReenter coming out of them, to also have "
              "other types of arc.";
      }
      return true;  // This state transitions to multiple FST instances.
    }
    return false;
  }

  // Fix olabels...
  //  On arcs with #nonterm_enter or #nonterm_reenter (these arcs will
  //  not actually be push them forward
  //

  // Returns true if any arc leaving this state has nonzero olabels.
  // Supp
  bool HasOlabels(StateId s) const {
  }

  // For each non-self-loop arc out of state s, replace that arc with an epsilon
  // arc with unit weight, to a new state.  That new state will have the
  // original arc.
  void InsertEpsilonsForState(StateId s) {
    fst::MutableArcIterator<FST> iter(fst_, s);
    for (; !iter.Done(); iter.Next()) {
      Arc arc = iter.Value();
      if (arc.nextstate != s) {
        StateId new_state = fst_->AddState();
        num_new_states_++;
        fst_->AddArc(new_state, arc);
        // now make this arc be an epsilon arc that goes to 'new_state'.
        arc.ilabel = 0;
        arc.olabel = 0;
        arc.weight = Weight::One();
        arc.nextstate = new_state;
        iter.SetValue(arc);
      }
    }
  }

  int32 nonterm_phones_offset_;
  VectorFst<StdArc> *fst_;
  int32 num_new_states_;
};


void PrepareForGrammarFst(int32 nonterm_phones_offset,
                          VectorFst<StdArc> *fst) {
  GrammarFstPreparer p(nonterm_phones_offset, fst);
  p.Prepare();
}




} // end namespace fst

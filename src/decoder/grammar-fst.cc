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
  int32 nonterm_phones_offset = config_.nonterm_phones_offset,
      encoding_multiple = GetEncodingMultiple(nonterm_phones_offset);
  // The following assertion should be optimized out as the condition is
  // statically known.
  KALDI_ASSERT(static_cast<int32>(kNontermBigNumber) %
               static_cast<int32>(kNontermMediumNumber) == 0);

  *nonterminal_symbol = (label - (int32)kNontermBigNumber) / encoding_multiple;
  *left_context_phone = label %  encoding_multiple;
  if (*nonterminal_symbol <= nonterm_phones_offset ||
      *left_context_phone == 0 ||
      *left_context_phone > nonterm_phones_offset)
    KALDI_ERR << "Decoding invalid label " << label
              << ": code error or invalid --nonterm-phones-offset?";

}

void GrammarFst::InitNonterminalMap() {
  nonterminal_map_.clear();
  for (size_t i = 0; i < ifsts_.size(); i++) {
    int32 nonterminal = ifsts_[i].first;
    if (nonterminal_map_.count(nonterminal))
      KALDI_ERR << "Nonterminal symbol " << nonterminal
                << " is paired with two FSTs.";
    if (nonterminal < GetPhoneSymbolFor(kNontermUserDefined))
      KALDI_ERR << "Nonterminal symbol " << nonterminal
                << " in input pairs, is expected to be >= "
                << GetPhoneSymbolFor(kNontermUserDefined);
    nonterminal_map_[nonterminal] = static_cast<int32>(i);
  }
}


void GrammarFst::InitEntryPoints() {
  entry_points_.resize(ifsts_.size());
  for (size_t i = 0; i < ifsts_.size(); i++) {
    const ConstFst<StdArc> &fst = *(ifsts_[i].second);
    InitEntryOrReentryPoints(fst, fst.Start(),
                             GetPhoneSymbolFor(kNontermBegin),
                             &(entry_points_[i]));
  }
}


void GrammarFst::InitEntryOrReentryPoints(
    const ConstFst<StdArc> &fst,
    int32 entry_state,
    int32 expected_nonterminal_symbol,
    std::unordered_map<int32, int32> *phone_to_arc) {
  phone_to_arc->clear();
  ArcIterator<ConstFst<StdArc> > aiter(fst, entry_state);
  int32 arc_index = 0;
  for (; !aiter.Done(); aiter.Next(), ++arc_index) {
    const StdArc &arc = aiter.Value();
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
    std::pair<int32, int32> p(left_context_phone, arc_index);
    if (!phone_to_arc->insert(p).second) {
      // if it was not successfully inserted, it means there were two arcs with
      // the same left-context phone, which does not make sense; that's an
      // error, likely a code error.
      KALDI_ERR << "Two arcs had the same left-context phone.";
    }
  }
}

GrammarFst::ExpandedState *GrammarFst::ExpandState(int32 instance_id,
                                       BaseStateId state_id) {
  int32 big_number = kNontermBigNumber;
  const ConstFst<StdArc> &fst = *(instances_[instance_id].fst);
  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);
  if (aiter.Done() || aiter.Value().ilabel < big_number) {
    return NULL;   // There was actually no need to expand this state; it was a
                   // normal final state.  We previously ensured that each state
                   // has only 'one type of arc' coming out of it, so checking
                   // the first arc is sufficient.
  }
  ExpandedState *ans = new ExpandedState;

  const StdArc &arc = aiter.Value();
  int32 encoding_multiple = GetEncodingMultiple(config_.nonterm_phones_offset),
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

    const FstInstance &return_instance = instances_[return_instance_id];
    const ConstFst<StdArc> &return_fst = *(return_instance.fst);
    // return_aiter is the iterator in the state we return to.
    ArcIterator<ConstFst<StdArc> > return_aiter(return_fst,
                                                instance.return_state);

    for (; !aiter.Done(); aiter.Next()) {
      const StdArc &leaving_arc = aiter.Value();
      int32 this_nonterminal, left_context_phone;
      DecodeSymbol(leaving_arc.ilabel, &this_nonterminal,
                   &left_context_phone);
      KALDI_ASSERT(this_nonterminal == nonterminal &&
                   ">1 nonterminals from a state; did you use "
                   "PrepareForGrammarFst()?");
      std::unordered_map<int32, BaseStateId>::const_iterator reentry_iter =
          instances_[instance_id].reentry_points.find(left_context_phone);
      if (reentry_iter == instance.reentry_points.end()) {
        KALDI_ERR << "FST with index " << instance.ifst_index
                  << " returns with left-context " << left_context_phone
                  << " but parent FST "
                  << instances_[return_instance_id].ifst_index
                  << " does not support that left-context at state "
                  << instance.return_state;
      }
      size_t return_arc_index = reentry_iter->second;
      return_aiter.Seek(return_arc_index);
      const StdArc &arriving_arc = return_aiter.Value();
      // 'arc' will combine the information on 'leaving_arc' and 'arriving_arc',
      // except that the ilabel will be set to zero.
      if (leaving_arc.olabel != 0) {
        // If the following fails it would maybe indicate you hadn't called
        // PrepareForGrammarFst(), or there was an error in that, because
        // we made sure the leaving arc does not have an olabel.  Search
        // in that code for 'olabel_problem' for more details.
        KALDI_ERR << "Leaving arc has zero olabel.";
      }
      StdArc arc;
      arc.ilabel = 0;
      // We checked that one of those two olabels is zero; their sum is
      // whichever is nonzero.
      arc.olabel = arriving_arc.olabel;
      arc.nextstate = arriving_arc.nextstate;
      arc.weight = Plus(leaving_arc.weight, arriving_arc.weight);
      ans->arcs.push_back(arc);
    }
  } else { // TODO
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
      num_new_states_(0),
      simple_final_state_(kNoStateId) { }

  void Prepare() {
    if (fst_->Start() == kNoStateId) {
      KALDI_ERR << "FST has no states.";
    }
    if (fst_->Properties(kILabelSorted, true) == 0) {
      // Make sure the FST is sorted on ilabel, if it wasn't already; the
      // decoder code requires epsilons to precede non-epsilons.
      ILabelCompare<StdArc> ilabel_comp;
      ArcSort(fst_, ilabel_comp);
    }
    for (StateId s = 0; s < fst_->NumStates(); s++) {
      if (IsSpecialState(s)) {
        bool transitions_to_multiple_instances,
            has_olabel_problem;
        CheckProperties(s, &transitions_to_multiple_instances,
                        &has_olabel_problem);
        if (transitions_to_multiple_instances ||
            has_olabel_problem) {
          InsertEpsilonsForState(s);
          // now all olabels on arcs leaving from s will be input-epsilon, so it
          // won't be treated as a 'special' state any more.
        } else {
          // OK, state s is a special state.
          FixArcsToFinalStates(s);
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
  bool IsSpecialState(StateId s) const;

  // If state s doesn't have any final-prob, add a final-prob with cost
  // 1024.0.
  void MakeSureStateHasFinalProb(StateId s) {
    if (fst_->Final(s) == Weight::Zero()) {
      fst_->SetFinal(s, Weight(1024.0));
    }
  }

  /**
     This function does some checking for 'special states', that they have
     certain expected properties, and also detects certain problematic
     conditions that we need to fix.

     @param [in] s  The state whose properties we are checking.  It is
                    expected that state 's' have at least one arc leaving it
                    with a special nonterminal symbol on it.
     @param [out] transitions_to_multiple_instances   Will be set to true
                   if state s has arcs out of it that will transition to
                   multiple FST instance (for example: one to this same FST, and
                   one to the FST for a user-defined nonterminal).  This is
                   the same as checking that all the labels correspond to
                   the same symbol #nontermXXX (or no such symbol, but in that
                   case this function wouldn't have been called.
     @param [out] has_olabel_problem   True if this state has at least one
                   arc with an ilabel corresponding to a user-defined
                   nonterminal or #nonterm_exit, and an olabel that is
                   nonzero.  In this case we will insert an input-epsilon
                   arc to fix the problem, basically using the same code as to
                   fix the 'transitions_to_multiple_instances' problem.
                   It's maybe not 100% ideal (it would have been better to push
                   the label backward), but in the big scheme of things this is
                   a very insignificant issue.

   This function also does some other checks that things are as expected,
   and will crash if those checks fail.
  */

  void CheckProperties(StateId s,
                       bool *transitions_to_multiple_instances,
                       bool *has_olabel_problem) const;

  // Fixes any final-prob-related problems with this state.  The problem we
  // aim to fix is that there may be arcs with nonterminal symbol
  // #nonterm_end which transition to a state with non-unit final prob.
  // This function assimilates that final-prob into the arc leaving from this state,
  // and makes the arc transition to a new state with unit final-prob.
  //
  // The purpose of this is to keep state-expansion (runtime) code simple.
  //
  // It would have been more efficient to do this in CheckProperties(), but
  // doing it this way is clearer; and the time taken here should actually be
  // tiny.
  void FixArcsToFinalStates(StateId s);

  // For each non-self-loop arc out of state s, replace that arc with an
  // epsilon-input arc to a newly created state, with unit weight and the
  // original olabel of the arc; and then an arc from that newly created state
  // to the original 'nextstate' of the arc, with the arc's weight and ilabel on
  // it.
  void InsertEpsilonsForState(StateId s);

  inline int32 GetPhoneSymbolFor(enum NonterminalValues n) const {
    return nonterm_phones_offset_ + static_cast<int32>(n);
  }

  int32 nonterm_phones_offset_;
  VectorFst<StdArc> *fst_;
  int32 num_new_states_;
  // If needed we may add a 'simple final state' to fst_, which has unit
  // final-prob.  This is used when we ensure that states with kNontermExit on
  // them transition to a state with unit final-prob, so we don't need to
  // look at the final-prob when expanding states.
  StateId simple_final_state_;
};

bool GrammarFstPreparer::IsSpecialState(StateId s) const {
  for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    if (arc.ilabel >= kNontermBigNumber) // 1 million
      return true;
  }
  return false;
}

void GrammarFstPreparer::CheckProperties(StateId s,
                                         bool *transitions_to_multiple_instances,
                                         bool *has_olabel_problem) const {
  *transitions_to_multiple_instances = false;
  *has_olabel_problem = false;

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
      if (nonterminal >= GetPhoneSymbolFor(kNontermUserDefined)) {
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
        if (arc.olabel != 0)
          *has_olabel_problem = true;
      }
      if (nonterminal == GetPhoneSymbolFor(kNontermBegin) &&
          s != fst_->Start()) {
        KALDI_ERR << "#nonterm_begin symbol is not on the first arc. "
            "Perhaps you didn't do fstdeterminizestar while compil?";
      }
      if (nonterminal == GetPhoneSymbolFor(kNontermEnd)) {
        if (fst_->NumArcs(arc.nextstate) != 0) {
          KALDI_ERR << "Arc with kNontermEnd is not the final arc.";
        }
        if (arc.olabel != 0)
          *has_olabel_problem = true;
      }
    }
    dest_nonterminals.insert(nonterminal);
  }
  if (dest_nonterminals.size() > 1) {
    // This state has arcs leading to multiple FST instances.
    *transitions_to_multiple_instances = true;
    // Do some checking, to see that there is nothing really unexpected in
    // there.
    for (std::set<int32>::const_iterator iter = dest_nonterminals.begin();
         iter != dest_nonterminals.end(); ++iter) {
      int32 nonterminal = *iter;
      if (nonterminal == nonterm_phones_offset_ + kNontermBegin ||
          nonterminal == nonterm_phones_offset_ + kNontermReenter)
        // we don't expect any state which has symbols like (kNontermBegin:p1)
        // on arcs coming out of it, to also have other types of symbol.  The
        // same goes for kNontermReenter.
        KALDI_ERR << "We do not expect states with arcs of type "
            "kNontermBegin/kNontermReenter coming out of them, to also have "
            "other types of arc.";
    }
  }
}

void GrammarFstPreparer::FixArcsToFinalStates(StateId s) {
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber;
  for (MutableArcIterator<FST> aiter(fst_, s ); !aiter.Done(); aiter.Next()) {
    Arc arc = aiter.Value();
    if (arc.ilabel < big_number)
      continue;
    int32 nonterminal = (arc.ilabel - big_number) / encoding_multiple;
    if (nonterminal ==  GetPhoneSymbolFor(kNontermEnd) &&
        fst_->Final(arc.nextstate) != Weight::One()) {
      KALDI_ASSERT(fst_->NumArcs(arc.nextstate) == 0 &&
                   fst_->Final(arc.nextstate) != Weight::Zero());
      if (simple_final_state_ == kNoStateId) {
        simple_final_state_ = fst_->AddState();
        fst_->SetFinal(fst_->AddState(), Weight::One());
      }
      arc.weight = Times(arc.weight, fst_->Final(arc.nextstate));
      arc.nextstate = simple_final_state_;
      aiter.SetValue(arc);
    }
  }
}

void GrammarFstPreparer::InsertEpsilonsForState(StateId s) {
  fst::MutableArcIterator<FST> iter(fst_, s);
  for (; !iter.Done(); iter.Next()) {
    Arc arc = iter.Value();
    if (arc.nextstate != s) {
      StateId new_state = fst_->AddState();
      num_new_states_++;
      int32 olabel = arc.olabel;
      arc.olabel = 0;
      fst_->AddArc(new_state, arc);
      // now make this arc be an epsilon arc that goes to 'new_state',
      // with unit weight but the original olabel of the arc.
      arc.ilabel = 0;
      arc.olabel = olabel;
      arc.weight = Weight::One();
      arc.nextstate = new_state;
      iter.SetValue(arc);
    }
  }
}


void PrepareForGrammarFst(int32 nonterm_phones_offset,
                          VectorFst<StdArc> *fst) {
  GrammarFstPreparer p(nonterm_phones_offset, fst);
  p.Prepare();
}




} // end namespace fst

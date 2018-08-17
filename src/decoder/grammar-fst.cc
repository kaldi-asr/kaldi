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


GrammarFst::GrammarFst(
    int32 nonterm_phones_offset,
    const ConstFst<StdArc> &top_fst,
    const std::vector<std::pair<Label, const ConstFst<StdArc> *> > &ifsts):
    nonterm_phones_offset_(nonterm_phones_offset),
    top_fst_(&top_fst),
    ifsts_(ifsts) {
  KALDI_ASSERT(nonterm_phones_offset_ > 1);
  InitNonterminalMap();
  entry_arcs_.resize(ifsts_.size());
  if (!ifsts_.empty()) {
    // We call this mostly so that if something is wrong with the input FSTs,
    // it will be detected soon
    InitEntryArcs(0);
  }
}

GrammarFst::~GrammarFst() {
  for (size_t i = 0; i < instances_.size(); i++) {
    FstInstance &instance = instances_[i];
    std::unordered_map<BaseStateId, ExpandedState*>::const_iterator
        iter = instance.expanded_states.begin(),
        end = instance.expanded_states.end();
    for (; iter != end; ++iter) {
      ExpandedState *e = iter->second;
      delete e;
    }
  }
}


void GrammarFst::DecodeSymbol(Label label,
                              int32 *nonterminal_symbol,
                              int32 *left_context_phone) {
  // encoding_multiple will normally equal 1000 (but may be a multiple of 1000
  // if there are a lot of phones); kNontermBigNumber is 1000000.
  int32 big_number = static_cast<int32>(kNontermBigNumber),
      nonterm_phones_offset = nonterm_phones_offset_,
      encoding_multiple = GetEncodingMultiple(nonterm_phones_offset);
  // The following assertion should be optimized out as the condition is
  // statically known.
  KALDI_ASSERT(big_number % static_cast<int32>(kNontermMediumNumber) == 0);

  *nonterminal_symbol = (label - big_number) / encoding_multiple;
  *left_context_phone = label % encoding_multiple;
  if (*nonterminal_symbol <= nonterm_phones_offset ||
      *left_context_phone == 0 || *left_context_phone >
      nonterm_phones_offset + static_cast<int32>(kNontermBos))
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
                << " in input pairs, was expected to be >= "
                << GetPhoneSymbolFor(kNontermUserDefined);
    nonterminal_map_[nonterminal] = static_cast<int32>(i);
  }
}


void GrammarFst::InitEntryArcs(int32 i) {
  KALDI_ASSERT(static_cast<size_t>(i) < ifsts_.size());
  const ConstFst<StdArc> &fst = *(ifsts_[i].second);
  InitEntryOrReentryArcs(fst, fst.Start(),
                         GetPhoneSymbolFor(kNontermBegin),
                         &(entry_arcs_[i]));
}

void GrammarFst::InitInstances() {
  KALDI_ASSERT(instances_.empty());
  instances_.resize(1);
  instances_[0].ifst_index = -1;
  instances_[0].fst = top_fst_;
  instances_[0].parent_instance = -1;
  instances_[0].parent_state = -1;
}

void GrammarFst::InitEntryOrReentryArcs(
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
      // If it was not successfully inserted in the phone_to_arc map, it means
      // there were two arcs with the same left-context phone, which does not
      // make sense; that's an error, likely a code error (or an error when the
      // input FSTs were generated).
      KALDI_ERR << "Two arcs had the same left-context phone.";
    }
  }
}

GrammarFst::ExpandedState *GrammarFst::ExpandState(
    int32 instance_id, BaseStateId state_id) {
  int32 big_number = kNontermBigNumber;
  const ConstFst<StdArc> &fst = *(instances_[instance_id].fst);
  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);
  KALDI_ASSERT(!aiter.Done() && aiter.Value().ilabel > big_number &&
               "Something is not right; did you call PrepareForGrammarFst()?");

  const StdArc &arc = aiter.Value();
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      nonterminal = (arc.ilabel - big_number) / encoding_multiple;
  if (nonterminal == GetPhoneSymbolFor(kNontermBegin) ||
      nonterminal == GetPhoneSymbolFor(kNontermReenter)) {
    KALDI_ERR << "Encountered unexpected type of nonterminal while "
        "expanding state.";
  } else if (nonterminal == GetPhoneSymbolFor(kNontermEnd)) {
    return ExpandStateEnd(instance_id, state_id);
  } else if (nonterminal >= GetPhoneSymbolFor(kNontermUserDefined)) {
    return ExpandStateUserDefined(instance_id, state_id);
  } else {
    KALDI_ERR << "Encountered unexpected type of nonterminal "
              << nonterminal << " while expanding state.";
  }
  return NULL;  // Suppress compiler warning
}


// static inline
void GrammarFst::CombineArcs(const StdArc &leaving_arc,
                             const StdArc &arriving_arc,
                             StdArc *arc) {
  // The following assertion shouldn't fail; we ensured this in
  // PrepareForGrammarFst(), search for 'olabel_problem'.
  KALDI_ASSERT(leaving_arc.olabel == 0);
  // 'leaving_arc' leaves one fst, and 'arriving_arcs', conceptually arrives in
  // another.  This code merges the information of the two arcs to make a
  // cross-FST arc.  The ilabel information is discarded as it was only intended
  // for the consumption of the GrammarFST code.
  arc->ilabel = 0;
  arc->olabel = arriving_arc.olabel;
  arc->weight = Times(leaving_arc.weight, arriving_arc.weight);
  arc->nextstate = arriving_arc.nextstate;
}

GrammarFst::ExpandedState *GrammarFst::ExpandStateEnd(
    int32 instance_id, BaseStateId state_id) {
  if (instance_id == 0)
    KALDI_ERR << "Did not expect #nonterm_end symbol in FST-instance 0.";
  const FstInstance &instance = instances_[instance_id];
  int32 parent_instance_id = instance.parent_instance;
  const ConstFst<StdArc> &fst = *(instance.fst);
  const FstInstance &parent_instance = instances_[parent_instance_id];
  const ConstFst<StdArc> &parent_fst = *(parent_instance.fst);

  ExpandedState *ans = new ExpandedState;
  ans->dest_fst_instance = parent_instance_id;

  // parent_aiter is the arc-iterator in the state we return to.  We'll Seek()
  // to a different position 'parent_aiter' for each arc leaving this state.
  // (actually we expect just one arc to leave this state).
  ArcIterator<ConstFst<StdArc> > parent_aiter(parent_fst,
                                              instance.parent_state);

  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);

  for (; !aiter.Done(); aiter.Next()) {
    const StdArc &leaving_arc = aiter.Value();
    int32 this_nonterminal, left_context_phone;
    DecodeSymbol(leaving_arc.ilabel, &this_nonterminal,
                 &left_context_phone);
    KALDI_ASSERT(this_nonterminal == GetPhoneSymbolFor(kNontermEnd) &&
                 ">1 nonterminals from a state; did you use "
                 "PrepareForGrammarFst()?");
    std::unordered_map<int32, int32>::const_iterator reentry_iter =
        instances_[instance_id].reentry_arcs.find(left_context_phone),
        reentry_end = instances_[instance_id].reentry_arcs.end();
    if (reentry_iter == reentry_end) {
      KALDI_ERR << "FST with index " << instance.ifst_index
                << " ends with left-context-phone " << left_context_phone
                << " but parent FST does not support that left-context "
          "at the return point.";
    }
    size_t parent_arc_index = static_cast<size_t>(reentry_iter->second);
    parent_aiter.Seek(parent_arc_index);
    const StdArc &arriving_arc = parent_aiter.Value();
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
    CombineArcs(leaving_arc, arriving_arc, &arc);
    ans->arcs.push_back(arc);
  }
  return ans;
}

int32 GrammarFst::GetChildInstanceId(int32 instance_id, int32 nonterminal,
                                     int32 state) {
  int64 encoded_pair = (static_cast<int64>(nonterminal) << 32) + state;
  // 'new_instance_id' is the instance-id we'd assign if we had to create a new one.
  // We try to add it at once, to avoid having to do an extra map lookup in case
  // it wasn't there and we did need to add it.
  int32 child_instance_id = instances_.size();
  {
    std::pair<int64, int32> p(encoded_pair, child_instance_id);
    std::pair<std::unordered_map<int64, int32>::const_iterator, bool> ans =
        instances_[instance_id].child_instances.insert(p);
    if (!ans.second) {
      // The pair was not inserted, which means the key 'encoded_pair' did exist in the
      // map.  Return the value in the map.
      child_instance_id = ans.first->second;
      return child_instance_id;
    }
  }
  // If we reached this point, we did successfully insert 'child_instance_id' into
  // the map, because the key didn't exist.  That means we have to actually create
  // the instance.
  instances_.resize(child_instance_id + 1);
  const FstInstance &parent_instance = instances_[instance_id];
  FstInstance &child_instance = instances_[child_instance_id];

  // Work out the ifst_index for this nonterminal.
  std::unordered_map<int32, int32>::const_iterator iter =
      nonterminal_map_.find(nonterminal);
  if (iter == nonterminal_map_.end()) {
    KALDI_ERR << "Nonterminal " << nonterminal << " was requested, but "
        "there is no FST for it.";
  }
  int32 ifst_index = iter->second;
  child_instance.ifst_index = ifst_index;
  child_instance.fst = ifsts_[ifst_index].second;
  child_instance.parent_instance = instance_id;
  child_instance.parent_state = state;
  InitEntryOrReentryArcs(*(parent_instance.fst), state,
                         GetPhoneSymbolFor(kNontermReenter),
                         &(child_instance.reentry_arcs));
  return child_instance_id;
}

GrammarFst::ExpandedState *GrammarFst::ExpandStateUserDefined(
    int32 instance_id, BaseStateId state_id) {
  const ConstFst<StdArc> &fst = *(instances_[instance_id].fst);
  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);

  ExpandedState *ans = new ExpandedState;
  int32 dest_fst_instance = -1;  // We'll set it in the loop.
                                 // and->dest_fst_instance will be set to this.

  for (; !aiter.Done(); aiter.Next()) {
    const StdArc &leaving_arc = aiter.Value();
    int32 nonterminal, left_context_phone;
    DecodeSymbol(leaving_arc.ilabel, &nonterminal,
                 &left_context_phone);
    int32 child_instance_id = GetChildInstanceId(instance_id,
                                                 nonterminal,
                                                 leaving_arc.nextstate);
    if (dest_fst_instance < 0) {
      dest_fst_instance = child_instance_id;
    } else if (dest_fst_instance != child_instance_id) {
      KALDI_ERR << "Same state leaves to different FST instances "
          "(Did you use PrepareForGrammarFst()?)";
    }
    const FstInstance &child_instance = instances_[child_instance_id];
    const ConstFst<StdArc> &child_fst = *(child_instance.fst);
    int32 child_ifst_index = child_instance.ifst_index;
    std::unordered_map<int32, int32> &entry_arcs = entry_arcs_[child_ifst_index];
    if (entry_arcs.empty())
      InitEntryArcs(child_ifst_index);
    // Get the arc-index for the arc leaving the start-state of child FST that
    // corresponds to this phonetic context.
    std::unordered_map<int32, int32>::const_iterator entry_iter =
        entry_arcs.find(left_context_phone);
    if (entry_iter == entry_arcs.end()) {
      KALDI_ERR << "FST for nonterminal " << nonterminal
                << " does not have an entry point for left-context-phone "
                << left_context_phone;
    }
    int32 arc_index = entry_iter->second;
    ArcIterator<ConstFst<StdArc> > child_aiter(child_fst, child_fst.Start());
    child_aiter.Seek(arc_index);
    const StdArc &arriving_arc = child_aiter.Value();
    StdArc arc;
    CombineArcs(leaving_arc, arriving_arc, &arc);
    ans->arcs.push_back(arc);
  }
  ans->dest_fst_instance = dest_fst_instance;
  return ans;
}

void GrammarFst::Write(std::ostream &os) const {
  bool binary = true;
  int32 format = 1,
      num_ifsts = ifsts_.size();
  WriteBasicType(os, binary, format);
  WriteBasicType(os, binary, num_ifsts);
  WriteBasicType(os, binary, nonterm_phones_offset_);

  FstWriteOptions wopts("unknown");
  top_fst_->Write(os.Stream(), wopts);
  for (int32 i = 0; i < num_ifsts; i++) {
    int32 nonterminal = ifsts_[i].first;
    WriteBasicType(os, binary, nonterminal);
    ifsts_[i]->Write(os.Stream(), wopts);
  }
}

void GrammarFst::Read(std::istream &is) {
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
      fst_(fst), num_new_states_(0), simple_final_state_(kNoStateId) { }

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
            has_olabel_problem || fst_->Final(s) != Weight::Zero()) {
          InsertEpsilonsForState(s);
          // now all olabels on arcs leaving from state s will be input-epsilon,
          // so it won't be treated as a 'special' state any more.
        } else {
          // OK, state s is a special state.
          FixArcsToFinalStates(s);
          MaybeAddFinalProbToState(s);
        }
      }
    }
    if (simple_final_state_ != kNoStateId)
      num_new_states_++;
    KALDI_VLOG(2) << "Added " << num_new_states_ << " new states while "
        "preparing for grammar FST.";
  }

 private:

  // Returns true if state 's' has at least one arc coming out of it with a
  // special nonterminal-related ilabel on it (i.e. an ilabel >= 1 million).
  bool IsSpecialState(StateId s) const;

  // This function verifies that state s does not currently have any
  // final-prob (crashes if that fails); then, if the arcs leaving s have
  // nonterminal symbols kNontermEnd or user-defined nonterminals (>=
  // kNontermUserDefined), it adds a final-prob with cost given by
  // KALDI_GRAMMAR_FST_SPECIAL_WEIGHT to the state.
  //
  // State s is required to be a 'special state', i.e. have special symbols on
  // arcs leaving it, and the function assumes (since it will already
  // have been checked) that the arcs leaving s, if there are more than
  // one, all correspond to the same nonterminal symbol.
  void MaybeAddFinalProbToState(StateId s);

  /**
     This function does some checking for 'special states', that they have
     certain expected properties, and also detects certain problematic
     conditions that we need to fix.

     @param [in] s  The state whose properties we are checking.  It is
                   expected that state 's' has at least one arc leaving it
                   with a special nonterminal symbol on it.
     @param [out] transitions_to_multiple_instances   Will be set to true
                   if state s has arcs out of it that will transition to
                   multiple FST instances (for example: one to this same FST, and
                   one to the FST for a user-defined nonterminal).  If the arcs
                   leaving this state do not all have ilabels corresponding to
                   a single symbol #nontermXXX, then we set this to true;
                   also, if this state has arcs with a user-defined nonterminal
                   on their label and their destination-states are not all the
                   same, we also set this to true.  (This should not happen, but
                   we check for it anyway).
     @param [out] has_olabel_problem   True if this state has at least one
                   arc with an ilabel corresponding to a user-defined
                   nonterminal or #nonterm_exit, and an olabel that is
                   nonzero.  In this case we will insert an input-epsilon
                   arc to fix the problem, basically using the same code as to
                   fix the 'transitions_to_multiple_instances' problem.
                   It's maybe not 100% ideal (it would have been better to push
                   the label backward), but in the big scheme of things this is
                   a very insignificant issue; it may not even happen.

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
  // by making the arc transition to a new state with unit final-prob, and
  // incorporating the original final-prob into the arc's weight.
  //
  // The purpose of this is to keep the GrammarFst code simple.
  //
  // It would have been more efficient to do this in CheckProperties(), but
  // doing it this way is clearer; and the time taken here should actually be
  // tiny.
  void FixArcsToFinalStates(StateId s);

  // For each non-self-loop arc out of state s, replace that arc with an
  // epsilon-input arc to a newly created state, with the arc's original weight
  // and the original olabel of the arc; and then an arc from that newly created
  // state to the original 'nextstate' of the arc, with the arc's ilabel on it,
  // and unit weight.  If this arc had a final-prob, make it an epsilon arc from
  // this state to the state 'simple_final_state_' (which has unit final-prob);
  // we create that state if needed.
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

  // The set 'dest_nonterminals' will encode something related to the set of
  // other FSTs this FST might transition to.  It will contain:
  //   0 if this state has any transition leaving it that is to this same FST
  //     (i.e. an epsilon or a normal transition-id)
  //   #nontermXXX if this state has an arc leaving it with an ilabel which
  //       would be decoded as the pair (#nontermXXX, p1).  Here, #nontermXXX is
  //       either an inbuilt nonterminal like #nonterm_begin, #nonterm_end or
  //       #nonterm_reenter, or a user-defined nonterminal like #nonterm:foo.
  std::set<int32> dest_nonterminals;
  // normally we'll have encoding_multiple = 1000, big_number = 1000000.
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber;

  // If we encounter arcs with user-defined nonterminals on them (unlikely), we
  // need to make sure that their destination-states are all the same.  If not,
  // they would transition to different FST instances, so we'd have to setq
  // 'transitions_to_multiple_instances' to true.  'destination_state' is used
  // to check this.
  StateId destination_state = kNoStateId;

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
        // This is a user-defined symbol.
        if (destination_state != kNoStateId &&
            arc.nextstate != destination_state) {
          // The code is not expected to come here.
          *transitions_to_multiple_instances = true;
        }
        destination_state = arc.nextstate;

        // Check that the destination state of this arc has arcs with
        // kNontermReenter on them.  We'll separately check that such states
        // don't have other types of arcs leaving them (search for
        // kNontermReenter below), so it's sufficient to check the first arc.
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
        KALDI_ERR << "#nonterm_begin symbol is present but this is not the "
            "first arc.  Did you do fstdeterminizestar while compiling?";
      }
      if (nonterminal == GetPhoneSymbolFor(kNontermEnd)) {
        if (fst_->NumArcs(arc.nextstate) != 0 ||
            fst_->Final(arc.nextstate) == Weight::Zero()) {
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
    if (nonterminal ==  GetPhoneSymbolFor(kNontermEnd)) {
      KALDI_ASSERT(fst_->NumArcs(arc.nextstate) == 0 &&
                   fst_->Final(arc.nextstate) != Weight::Zero());
      if (fst_->Final(arc.nextstate) == Weight::One())
        continue;  // There is no problem to fix.
      if (simple_final_state_ == kNoStateId) {
        simple_final_state_ = fst_->AddState();
        fst_->SetFinal(simple_final_state_, Weight::One());
      }
      arc.weight = Times(arc.weight, fst_->Final(arc.nextstate));
      arc.nextstate = simple_final_state_;
      aiter.SetValue(arc);
    }
  }
}

void GrammarFstPreparer::MaybeAddFinalProbToState(StateId s) {
  if (fst_->Final(s) != Weight::Zero()) {
    // Something went wrong and it will require some debugging.  In Prepare(),
    // if we detected that the special state had a nonzero final-prob, we
    // would have inserted epsilons to remove it, so there may be a bug in
    // this class's code.
    KALDI_ERR << "State already final-prob.";
  }
  ArcIterator<FST> aiter(*fst_, s );
  KALDI_ASSERT(!aiter.Done());
  const Arc &arc = aiter.Value();
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber,
      nonterminal = (arc.ilabel - big_number) / encoding_multiple;
  KALDI_ASSERT(nonterminal >= GetPhoneSymbolFor(kNontermBegin));
  if (nonterminal == GetPhoneSymbolFor(kNontermEnd) ||
      nonterminal >= GetPhoneSymbolFor(kNontermUserDefined)) {
    fst_->SetFinal(s, Weight(KALDI_GRAMMAR_FST_SPECIAL_WEIGHT));
  }
}


void GrammarFstPreparer::InsertEpsilonsForState(StateId s) {
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber;
  fst::MutableArcIterator<FST> iter(fst_, s);
  for (; !iter.Done(); iter.Next()) {
    Arc arc = iter.Value();
    {
      // Do a sanity check.  We shouldn't be inserting epsilons for certain
      // types of state, so check that it's not one of those.
      int32 nonterminal = (arc.ilabel - big_number) / encoding_multiple;
      if (nonterminal == GetPhoneSymbolFor(kNontermBegin) ||
          nonterminal == GetPhoneSymbolFor(kNontermReenter)) {
        KALDI_ERR << "Something went wrong; did not expect to insert epsilons "
            "for this type of state.";
      }
    }
    if (arc.nextstate != s) {
      StateId new_state = fst_->AddState();
      num_new_states_++;
      Arc new_arc(arc);
      new_arc.olabel = 0;
      new_arc.weight = Weight::One();
      fst_->AddArc(new_state, new_arc);
      // now make this arc be an epsilon arc that goes to 'new_state',
      // with unit weight but the original olabel of the arc.
      arc.ilabel = 0;
      arc.nextstate = new_state;
      iter.SetValue(arc);
    } else {  // Self-loop
      KALDI_ASSERT(arc.ilabel < big_number &&
                   "Self-loop with special label found.");
    }
  }
  if (fst_->Final(s) != Weight::Zero()) {
    if (fst_->Final(s).Value() == KALDI_GRAMMAR_FST_SPECIAL_WEIGHT) {
      // TODO: find a way to detect if it was a coincidence, or not make it an
      // error, because in principle a user-defined grammar could contain this
      // special cost.
      KALDI_ERR << "It looks like you are calling PrepareForGrammarFst twice.";
    }
    if (simple_final_state_ == kNoStateId) {
      simple_final_state_ = fst_->AddState();
      fst_->SetFinal(simple_final_state_, Weight::One());
    }
    Arc arc;
    arc.ilabel = 0;
    arc.olabel = 0;
    arc.nextstate = simple_final_state_;
    arc.weight = fst_->Final(s);
    fst_->AddArc(s, arc);
    fst_->SetFinal(s, Weight::Zero());
  }
}


void PrepareForGrammarFst(int32 nonterm_phones_offset,
                          VectorFst<StdArc> *fst) {
  GrammarFstPreparer p(nonterm_phones_offset, fst);
  p.Prepare();
}




} // end namespace fst

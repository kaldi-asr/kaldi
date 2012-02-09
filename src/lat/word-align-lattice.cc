// lat/word-align-lattice.cc

// Copyright 2011   Microsoft Corporation

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


#include "lat/word-align-lattice.h"
#include "hmm/transition-model.h"
#include "util/stl-utils.h"

namespace kaldi {

class LatticeWordAligner {
 public:
  typedef CompactLatticeArc::StateId StateId;
  typedef CompactLatticeArc::Label Label;
  
  class ComputationState { /// The state of the computation in which,
   public:
    /// along a single path in the lattice, we work out the word
    /// boundaries and output aligned arcs.
    
    /// Advance the computation state by adding the symbols and weights
    /// from this arc.
    void Advance(const CompactLatticeArc &arc) {
      const std::vector<int32> &string = arc.weight.String();
      transition_ids_.insert(transition_ids_.end(),
                             string.begin(), string.end());
      if (arc.ilabel != 0) // note: arc.ilabel==arc.oabel (acceptor)
        word_labels_.push_back(arc.ilabel);
      weight_ = Times(weight_, arc.weight.Weight());      
    }

    /// If it can output a whole word, it will do so, will put it in arc_out,
    /// and return true; else it will return false.  If it detects an error
    /// condition and *error = false, it will set *error to true and print
    /// a warning.  In this case it may or may not [output an arc and return true],
    /// depending on what we think is most likely the right thing to do.  Of
    /// course once *error is set, something has gone wrong so don't trust
    /// the output too fully.
    /// Note: the "next_state" of the arc will not be set, you have to do that
    /// yourself.
    bool OutputArc(const WordBoundaryInfo &info,
                   const TransitionModel &tmodel,
                   CompactLatticeArc *arc_out,
                   bool *error) {
      // order of this ||-expression doesn't matter for
      // function behavior, only for efficiency, since the
      // cases are disjoint.
      return OutputNormalWordArc(info, tmodel, arc_out, error) ||
          OutputSilenceArc(info, tmodel, arc_out, error) ||
          OutputOnePhoneWordArc(info, tmodel, arc_out, error);
    }

    bool OutputSilenceArc(const WordBoundaryInfo &info,
                          const TransitionModel &tmodel,
                          CompactLatticeArc *arc_out,
                          bool *error); 
    bool OutputOnePhoneWordArc(const WordBoundaryInfo &info,
                               const TransitionModel &tmodel,
                               CompactLatticeArc *arc_out,
                               bool *error); 
    bool OutputNormalWordArc(const WordBoundaryInfo &info,
                             const TransitionModel &tmodel,
                             CompactLatticeArc *arc_out,
                             bool *error);
    
    bool IsEmpty() { return (transition_ids_.empty() && word_labels_.empty()); }
    
    /// FinalWeight() will return "weight" if both transition_ids
    /// and word_labels are empty, otherwise it will return
    /// Weight::Zero().
    LatticeWeight FinalWeight() { return (IsEmpty() ? weight_ : LatticeWeight::Zero()); }

    /// This function may be called when you reach the end of
    /// the lattice and this structure hasn't voluntarily
    /// output words using "OutputArc".  If IsEmpty() == false,
    /// then you can call this function and it will output
    /// an arc.  The only
    /// non-error state in which this happens, is when a word
    /// (or silence) has ended, but we don't know that it's
    /// ended because we haven't seen the first transition-id
    /// from the next word.  Otherwise (error state), the output
    /// will consist of partial words, and this will only
    /// happen for lattices that were somehow broken, i.e.
    /// had not reached the final state.
    void OutputArcForce(const WordBoundaryInfo &info,
                        const TransitionModel &tmodel,
                        CompactLatticeArc *arc_out,
                        bool *error);
  
    size_t Hash() const {
      VectorHasher<int32> vh;
      return vh(transition_ids_) + 90647 * vh(word_labels_);
      // 90647 is an arbitrary largish prime number.
      // We don't bother including the weight in the hash--
      // we don't really expect duplicates with the same vectors
      // but different weights, and anyway, this is only an
      // efficiency issue.
    }

    // Just need an arbitrary complete order.
    bool operator == (const ComputationState &other) const {
      return (transition_ids_ == other.transition_ids_
              && word_labels_ == other.word_labels_
              && weight_ == other.weight_);
    }
    
    ComputationState(): weight_(LatticeWeight::One()) { } // initial state.
    ComputationState(const ComputationState &other):
        transition_ids_(other.transition_ids_), word_labels_(other.word_labels_),
        weight_(other.weight_) { }
   private:
    std::vector<int32> transition_ids_;
    std::vector<int32> word_labels_;
    LatticeWeight weight_; // contains two floats.
  };


  struct Tuple {
    Tuple(StateId input_state, ComputationState comp_state):
        input_state(input_state), comp_state(comp_state) {}
    StateId input_state;
    ComputationState comp_state;
  };

  struct TupleHash {
    size_t operator() (const Tuple &state) const {
      return state.input_state + 102763 * state.comp_state.Hash();
      // 102763 is just an arbitrary prime number 
    }
  };
  struct TupleEqual {
    bool operator () (const Tuple &state1, const Tuple &state2) const {
      // treat this like operator ==
      return (state1.input_state == state2.input_state
              && state1.comp_state == state2.comp_state);
    }
  };
  
  typedef unordered_map<Tuple, StateId, TupleHash, TupleEqual> MapType;

  StateId GetStateForTuple(const Tuple &tuple, bool add_to_queue) {
    MapType::iterator iter = map_.find(tuple);
    if (iter == map_.end()) { // not in map.
      StateId output_state = lat_out_->AddState();
      map_[tuple] = output_state;
      if (add_to_queue)
        queue_.push_back(std::make_pair(tuple, output_state));
      return output_state;
    } else {
      return iter->second;
    }
  }
  
  void ProcessFinal(Tuple tuple, StateId output_state) {
    // ProcessFinal is only called if the input_state has
    // final-prob of One().  [else it should be zero.  This
    // is because we called CreateSuperFinal().]
    
    if (tuple.comp_state.IsEmpty()) { // computation state doesn't have
      // anything pending.
      std::vector<int32> empty_vec;
      CompactLatticeWeight cw(tuple.comp_state.FinalWeight(), empty_vec);
      lat_out_->SetFinal(output_state, Plus(lat_out_->Final(output_state), cw));      
    } else {
      // computation state has something pending, i.e. input or
      // output symbols that need to be flushed out.  Note: OutputArc() would
      // have returned false or we wouldn't have been called, so we have to
      // force it out.
      CompactLatticeArc lat_arc;
      tuple.comp_state.OutputArcForce(info_, tmodel_, &lat_arc, &error_);
      lat_arc.nextstate = GetStateForTuple(tuple, true); // true == add to queue.
      // The final-prob stuff will get called again from ProcessQueueElement().
      // Note: because we did CreateSuperFinal(), this final-state on the input
      // lattice will have no output arcs (and unit final-prob), so there will be
      // no complications with processing the arcs from this state (there won't
      // be any).
      KALDI_ASSERT(output_state != lat_arc.nextstate);
      lat_out_->AddArc(output_state, lat_arc);
    }
  }

  
  void ProcessQueueElement() {
    KALDI_ASSERT(!queue_.empty());
    Tuple tuple = queue_.back().first;
    StateId output_state = queue_.back().second;
    queue_.pop_back();

    // First thing is-- we see whether the computation-state has something
    // pending that it wants to output.  In this case we don't do
    // anything further.  This is a chosen behavior similar to the
    // epsilon-sequencing rules encoded by the filters in
    // composition.
    CompactLatticeArc lat_arc;
    Tuple tuple2(tuple); //temp
    if (tuple.comp_state.OutputArc(info_, tmodel_, &lat_arc, &error_)) {
      // note: this function changes the tuple (when it returns true).
      lat_arc.nextstate = GetStateForTuple(tuple, true); // true == add to queue,
      // if not already present.
      KALDI_ASSERT(output_state != lat_arc.nextstate);
      lat_out_->AddArc(output_state, lat_arc);
    } else {
      // when there's nothing to output, we'll process arcs from the input-state.
      // note: it would in a sense be valid to do both (i.e. process the stuff
      // above, and also these), but this is a bit like the epsilon-sequencing
      // stuff in composition: we avoid duplicate arcs by doing it this way.

      if (lat_.Final(tuple.input_state) != CompactLatticeWeight::Zero()) {
        KALDI_ASSERT(lat_.Final(tuple.input_state) == CompactLatticeWeight::One());
        // ... since we did CreateSuperFinal.
        ProcessFinal(tuple, output_state);
      }
      // Now process the arcs.  Note: final-state shouldn't have any arcs.
      for(fst::ArcIterator<CompactLattice> aiter(lat_, tuple.input_state);
          !aiter.Done(); aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        Tuple next_tuple(tuple);
        next_tuple.comp_state.Advance(arc);
        next_tuple.input_state = arc.nextstate;
        StateId next_output_state = GetStateForTuple(next_tuple, true); // true == add to queue,
        // if not already present.
        // We add an epsilon arc here (as the input and output happens
        // separately)... the epsilons will get removed later.
        KALDI_ASSERT(next_output_state != output_state);
        lat_out_->AddArc(output_state,
                         CompactLatticeArc(0, 0,
                                           CompactLatticeWeight::One(),
                                           next_output_state));
      }
    }
  }
  
  LatticeWordAligner(const CompactLattice &lat,
                     const TransitionModel &tmodel,
                     const WordBoundaryInfo &info,
                     CompactLattice *lat_out):
      lat_(lat), tmodel_(tmodel), info_in_(info), info_(info), lat_out_(lat_out),
      error_(false) {
    fst::CreateSuperFinal(&lat_); // Creates a super-final state, so the
    // only final-probs are One().
    
    // Inside this class, we don't want to use zero for the silence
    // or partial-word labels, as this will interfere with the RmEpsilon
    // stage, where we don't want the arcs corresponding to silence or
    // partial words to be removed-- only the arcs with nothing at all
    // on them.
    if (info_.partial_word_label == 0 || info_.silence_label == 0) {
      int32 unused_label = 1 + HighestNumberedOutputSymbol(lat);
      KALDI_ASSERT(unused_label > 0);
      if (info_.partial_word_label == 0)
        info_.partial_word_label = unused_label;
      if (info_.silence_label == 0)
        info_.silence_label = unused_label;
    }
  }

  // Removes epsilons; also removes unreachable states...
  // not sure if these would exist if original was connected.
  // This also replaces the temporary symbols for the silence
  // and partial-words, with epsilons, if we wanted epsilons.
  void RemoveEpsilonsFromLattice() {
    // Remove epsilon arcs from output lattice.
    RmEpsilon(lat_out_, true); // true = connect.
    std::vector<int32> syms_to_remove;
    if (info_in_.partial_word_label == 0)
      syms_to_remove.push_back(info_.partial_word_label);
    if (info_in_.silence_label == 0)
      syms_to_remove.push_back(info_.silence_label);
    if (!syms_to_remove.empty()) {
      RemoveSomeInputSymbols(syms_to_remove, lat_out_);
      Project(lat_out_, fst::PROJECT_INPUT);      
    }
  }
  
  bool AlignLattice() {
    lat_out_->DeleteStates();
    if (lat_.Start() == fst::kNoStateId) {
      KALDI_WARN << "Trying to word-align empty lattice.";
      return false;
    }
    ComputationState initial_comp_state;
    Tuple initial_tuple(lat_.Start(), initial_comp_state);
    StateId start_state = GetStateForTuple(initial_tuple, true); // True = add this to queue.
    lat_out_->SetStart(start_state);
    
    while (!queue_.empty())
      ProcessQueueElement();

    RemoveEpsilonsFromLattice();
    
    return !error_;
  }
  
  CompactLattice lat_;
  const TransitionModel &tmodel_;
  const WordBoundaryInfo &info_in_;
  WordBoundaryInfo info_;
  CompactLattice *lat_out_;

  std::vector<std::pair<Tuple, StateId> > queue_;
  MapType map_; // map from tuples to StateId.
  bool error_;
  
};

bool LatticeWordAligner::ComputationState::OutputSilenceArc(
    const WordBoundaryInfo &info, const TransitionModel &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {
  if (transition_ids_.empty()) return false;
  int32 phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
  if (!std::binary_search(info.silence_phones.begin(),
                          info.silence_phones.end(),
                          phone)) return false;
  // we assume the start of transition_ids_ is the start of the phone [silence];
  // this is a precondition.
  size_t len = transition_ids_.size(), i;
  // Keep going till we reach a "final" transition-id; note, if
  // reorder==true, we have to go a bit further after this.
  for (i = 1; i < len; i++) {
    int32 tid = transition_ids_[i];
    int32 this_phone = tmodel.TransitionIdToPhone(tid);
    if (this_phone != phone && ! *error) { // error condition: should have reached final transition-id first.
      *error = true;
      KALDI_WARN << "Phone changed before final transition-id found "
          "[broken lattice or mismatched model or wrong --reorder option?]";
    }
    if (tmodel.IsFinal(tid))
      break;
  }
  if (i == len) return false; // fell off loop.
  i++; // go past the one for which IsFinal returned true.
  if (info.reorder) // we have to consume the following self-loop transition-ids.
    while (i < len && tmodel.IsSelfLoop(transition_ids_[i])) i++;
  if (i == len) return false; // we don't know if it ends here... so can't output arc.

  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != phone
      && ! *error) { // another check.
    KALDI_WARN << "Phone changed unexpectedly in lattice "
        "[broken lattice or mismatched model?]";
  }
  // interpret i as the number of transition-ids to consume.
  std::vector<int32> tids_out(transition_ids_.begin(), transition_ids_.begin()+i);
  
  // consumed transition ids from our internal state.
  *arc_out = CompactLatticeArc(info.silence_label, info.silence_label,
                               CompactLatticeWeight(weight_, tids_out), fst::kNoStateId);
  transition_ids_.erase(transition_ids_.begin(), transition_ids_.begin()+i); // delete these
  weight_ = LatticeWeight::One(); // we just output the weight.
  return true;
}


bool LatticeWordAligner::ComputationState::OutputOnePhoneWordArc(
    const WordBoundaryInfo &info, const TransitionModel &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {
  if (transition_ids_.empty()) return false;
  if (word_labels_.empty()) return false;
  int32 phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
  if (!std::binary_search(info.wbegin_and_end_phones.begin(),
                          info.wbegin_and_end_phones.end(),
                          phone)) return false;
  // we assume the start of transition_ids_ is the start of the phone.
  // this is a precondition.
  size_t len = transition_ids_.size(), i;
  for (i = 1; i < len; i++) {
    int32 tid = transition_ids_[i];
    int32 this_phone = tmodel.TransitionIdToPhone(tid);
    if (this_phone != phone && ! *error) { // error condition: should have reached final transition-id first.
      KALDI_WARN << "Phone changed before final transition-id found "
          "[broken lattice or mismatched model or wrong --reorder option?]";
      // just continue, ignoring this-- we'll probably output something...
    }
    if (tmodel.IsFinal(tid))
      break;
  }
  if (i == len) return false; // fell off loop.
  i++; // go past the one for which IsFinal returned true.
  if (info.reorder) // we have to consume the following self-loop transition-ids.
    while (i < len && tmodel.IsSelfLoop(transition_ids_[i])) i++;
  if (i == len) return false; // we don't know if it ends here... so can't output arc.

  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != phone
      && ! *error) { // another check.
    KALDI_WARN << "Phone changed unexpectedly in lattice "
        "[broken lattice or mismatched model?]";
    *error = true;
  }
  
  // interpret i as the number of transition-ids to consume.
  std::vector<int32> tids_out(transition_ids_.begin(), transition_ids_.begin()+i);
  
  // consumed transition ids from our internal state.
  int32 word = word_labels_[0];
  *arc_out = CompactLatticeArc(word, word,
                               CompactLatticeWeight(weight_, tids_out), fst::kNoStateId);
  transition_ids_.erase(transition_ids_.begin(), transition_ids_.begin()+i); // delete these
  word_labels_.erase(word_labels_.begin(), word_labels_.begin()+1); // remove the word we output.
  weight_ = LatticeWeight::One(); // we just output the weight.
  return true;
}


/// This function tries to see if it can output a normal word arc--
/// one with at least two phones in it.
bool LatticeWordAligner::ComputationState::OutputNormalWordArc(
    const WordBoundaryInfo &info, const TransitionModel &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {
  if (transition_ids_.empty()) return false;
  if (word_labels_.empty()) return false;
  int32 begin_phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
  if (!std::binary_search(info.wbegin_phones.begin(),
                          info.wbegin_phones.end(),
                          begin_phone)) return false;
  // we assume the start of transition_ids_ is the start of the phone.
  // this is a precondition.
  size_t len = transition_ids_.size(), i;

  // Eat up the transition-ids of this word-begin phone until we get to the
  // "final" transition-id.  [there may be self-loops following this though,
  // if reorder==true]
  for (i = 0; i < len && !tmodel.IsFinal(transition_ids_[i]); i++);
  if (i == len) return false;
  i++; // Skip over this final-transition.
  if (info.reorder) // Skip over any reordered self-loops for this final-transition
    for (; i < len && tmodel.IsSelfLoop(transition_ids_[i]); i++);
  if (i == len) return false;
  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != begin_phone
      && ! *error) { // another check.
    KALDI_WARN << "Phone changed unexpectedly in lattice "
        "[broken lattice or mismatched model?]";
    *error = true;
  }
  // Now keep going till we hit a word-ending phone.
  // Note: we don't expect anything except word-internal phones
  // here, but we'll just print a warning if we get something
  // else.
  for (; i < len; i++) {
    int32 this_phone = tmodel.TransitionIdToPhone(transition_ids_[i]);
    if (std::binary_search(info.wend_phones.begin(),
                           info.wend_phones.end(),
                           this_phone)) break; // we hit a word-end phone.
    if (!std::binary_search(info.winternal_phones.begin(),
                            info.winternal_phones.end(),
                            this_phone) && !*error) {
      KALDI_WARN << "Unexpected phone " << this_phone
                 << " found inside a word.";
      *error = true;
    }
  }
  if (i == len) return false;

  // OK, we hit a word-ending phone.  Continue till we get to
  // a "final-transition".

  // this variable just used for checks.
  int32 final_phone = tmodel.TransitionIdToPhone(transition_ids_[i]); 
  for (; i < len; i++) {
    int32 this_phone = tmodel.TransitionIdToPhone(transition_ids_[i]);
    if (this_phone != final_phone && ! *error) {
      *error = true;
      KALDI_WARN << "Phone changed before final transition-id found "
          "[broken lattice or mismatched model or wrong --reorder option?]";
    }
    if (tmodel.IsFinal(transition_ids_[i])) break;
  }
  if (i == len) return false;
  i++;
  // We got to the final-transition of the final phone;
  // if reorder==true, continue eating up the self-loop.
  if (info.reorder == true)
    while (i < len && tmodel.IsSelfLoop(transition_ids_[i])) i++;
  if (i == len) return false;
  if (tmodel.TransitionIdToPhone(transition_ids_[i-1]) != final_phone
      && ! *error) {
    *error = true;
    KALDI_WARN << "Phone changed while following final self-loop "
        "[broken lattice or mismatched model or wrong --reorder option?]";
  }

  // OK, we're ready to output the word.
  // Interpret i as the number of transition-ids to consume.
  std::vector<int32> tids_out(transition_ids_.begin(), transition_ids_.begin()+i);
  
  // consumed transition ids from our internal state.
  int32 word = word_labels_[0];
  *arc_out = CompactLatticeArc(word, word,
                               CompactLatticeWeight(weight_, tids_out),
                               fst::kNoStateId);
  transition_ids_.erase(transition_ids_.begin(), transition_ids_.begin()+i); // delete these
  word_labels_.erase(word_labels_.begin(), word_labels_.begin()+1); // remove the word we output.
  weight_ = LatticeWeight::One(); // we just output the weight.
  return true;
}

// returns true if this vector of transition-ids could be a valid
// word (note: doesn't do exhaustive checks, just sanity checks).
static bool IsPlausibleWord(const WordBoundaryInfo &info,
                            const TransitionModel &tmodel,
                            const std::vector<int32> &transition_ids) {
  if (transition_ids.empty()) return false;
  int32 first_phone = tmodel.TransitionIdToPhone(transition_ids.front()),
      last_phone = tmodel.TransitionIdToPhone(transition_ids.back());
  if ( (std::binary_search(info.wbegin_and_end_phones.begin(),
                           info.wbegin_and_end_phones.end(),
                           first_phone) && first_phone == last_phone)
       ||
       (std::binary_search(info.wbegin_phones.begin(),
                           info.wbegin_phones.end(), first_phone) &&
        std::binary_search(info.wend_phones.begin(),
                           info.wend_phones.end(), last_phone))) {
    if (! info.reorder) {
      return (tmodel.IsFinal(transition_ids.back()));
    } else {
      return (tmodel.IsFinal(transition_ids.back())
              || tmodel.IsSelfLoop(transition_ids.back()));
    }
  } else return false;
}

    
void LatticeWordAligner::ComputationState::OutputArcForce(
    const WordBoundaryInfo &info, const TransitionModel &tmodel,
    CompactLatticeArc *arc_out,  bool *error) {

  KALDI_ASSERT(!IsEmpty());
  if (!word_labels_.empty()
      && !transition_ids_.empty()) { // We have at least one word to
    // output, and some transition-ids.  We assume that the normal OutputArc was called
    // and failed, so this means we didn't see the end of that
    // word. 
    int32 word = word_labels_[0];
    if (! *error && !IsPlausibleWord(info, tmodel, transition_ids_)) {
      *error = true;
      KALDI_WARN << "Invalid word at end of lattice [partial lattice, forced out?]";
    }
    CompactLatticeWeight cw(weight_, transition_ids_);
    *arc_out = CompactLatticeArc(word, word, cw, fst::kNoStateId);
    weight_ = LatticeWeight::One();
    transition_ids_.clear();
    word_labels_.erase(word_labels_.begin(), word_labels_.begin()+1);
  } else if (!word_labels_.empty() && transition_ids_.empty()) {
    // We won't create arcs with these word labels on, as most likely
    // this will cause errors down the road.  This is an error
    // condition anyway, in some sense.
    if (! *error) {
      *error = true;
      KALDI_WARN << "Discarding word-ids at the end of a sentence, "
          "that don't have alignments.";
    }
    CompactLatticeWeight cw(weight_, transition_ids_);
    // This creates an epsilon arc with a weight on it, but
    // no transition-ids since the vector is empty.
    // The word labels are discarded.
    *arc_out = CompactLatticeArc(0, 0, cw, fst::kNoStateId);
    weight_ = LatticeWeight::One();
    word_labels_.clear();
  } else if (!transition_ids_.empty() && word_labels_.empty()) {
    // Transition-ids but no word label-- either silence or
    // partial word.
    int32 first_phone = tmodel.TransitionIdToPhone(transition_ids_[0]);
    if (std::binary_search(info.silence_phones.begin(),
                           info.silence_phones.end(),
                           first_phone)) { // first phone is silence...
      if (!first_phone == tmodel.TransitionIdToPhone(transition_ids_.back())
          && ! *error) {
        *error = true;
        // Phone changed-- this is a code error, because the regular OutputArc
        // should have output an arc (a silence arc) if that phone finished.
        // So we make it fatal.
        KALDI_ERR << "Broken silence arc at end of utterance (the phone "
            "changed); code error";
      }
      CompactLatticeWeight cw(weight_, transition_ids_);
      *arc_out = CompactLatticeArc(info.silence_label, info.silence_label,
                                   cw, fst::kNoStateId);
    } else {
      // Not silence phone -- treat as partial word (with no word label).
      // This is in itself an error condition, i.e. the lattice was maybe
      // forced out.
      if (! *error) {
        *error = true;
        KALDI_WARN << "Partial word detected at end of utterance";
      }
      CompactLatticeWeight cw(weight_, transition_ids_);
      *arc_out = CompactLatticeArc(info.partial_word_label, info.partial_word_label,
                                   cw, fst::kNoStateId);
    }
    transition_ids_.clear();
    weight_ = LatticeWeight::One();
  } else {
    KALDI_ERR << "Code error, word-aligning lattice"; // this shouldn't
    // be able to happen; we don't call this function of they're both empty.
  }
}

static bool AreDisjoint(std::vector<int32> vec1,
                        std::vector<int32> vec2) {
  Uniq(&vec1); // remove dups...
  Uniq(&vec2);
  vec2.insert(vec2.end(), vec1.begin(), vec1.end());
  size_t size = vec2.size();
  SortAndUniq(&vec2);
  return (vec2.size() == size);
}  

WordBoundaryInfo::WordBoundaryInfo(const WordBoundaryInfoOpts &opts) {
  if (!kaldi::SplitStringToIntegers(opts.wbegin_phones, ":",
                                    false,
                                    &wbegin_phones)
      || wbegin_phones.empty())
    KALDI_ERR << "Invalid argument to --wbegin-phones option: "
              << opts.wbegin_phones;
  if (!kaldi::SplitStringToIntegers(opts.wend_phones, ":",
                                    false,
                                    &wend_phones)
      || wend_phones.empty())
    KALDI_ERR << "Invalid argument to --wend-phones option: "
              << opts.wend_phones;
  if (!kaldi::SplitStringToIntegers(opts.wbegin_and_end_phones, ":",
                                    false,
                                    &wbegin_and_end_phones)
      || wbegin_and_end_phones.empty())
    KALDI_ERR << "Invalid argument to --wbegin-and-end-phones option: "
              << opts.wbegin_and_end_phones;
  if (!kaldi::SplitStringToIntegers(opts.winternal_phones, ":",
                                    false,
                                    &winternal_phones)
      || winternal_phones.empty())      
    KALDI_ERR << "Invalid argument to --winternal-phones option: "
              << opts.winternal_phones;
  if (!kaldi::SplitStringToIntegers(opts.silence_phones, ":",
                                    false,
                                    &silence_phones)) // we let this one be empty.
    KALDI_ERR << "Invalid argument to --silence-phones option: "
              << opts.silence_phones;

  std::vector<std::vector<int32>* > all_vecs;
  all_vecs.push_back(&wbegin_phones);
  all_vecs.push_back(&wend_phones);
  all_vecs.push_back(&wbegin_and_end_phones);
  all_vecs.push_back(&winternal_phones);
  all_vecs.push_back(&silence_phones);  
  
  for(size_t i = 0; i < all_vecs.size(); i++) {
    std::sort(all_vecs[i]->begin(), all_vecs[i]->end());
    if (!IsSortedAndUniq(*all_vecs[i]))
      KALDI_ERR << "Expecting all options such as --wbegin-phones "
          "to have no repetitions";
  }
  
  for(size_t i = 0; i < all_vecs.size(); i++)
    for (size_t j = i+1; j < all_vecs.size(); j++)
      if (!AreDisjoint(*(all_vecs[i]), *(all_vecs[j])))
        KALDI_ERR << "Expecting the phones in options such as "
            "--wbegin-phones to be disjoint from each other. "
            "Make sure this is what you mean.";
  
  reorder = opts.reorder;
  silence_label = opts.silence_label;
  partial_word_label = opts.partial_word_label;
  if (opts.silence_may_be_word_internal) {
    winternal_phones.insert(winternal_phones.end(),
                            silence_phones.begin(), silence_phones.end());
    SortAndUniq(&winternal_phones);
  }
  if (opts.silence_has_olabels) { // output labels will be in lattice for silence.
    // (because it appeared on output side of WFST, e.g. it was in your ARPA).
    // Note: in this case you can't have optional silence in your lexicon.
    // Treat the silence phones as word begin-and-end phones.
    wbegin_and_end_phones.insert(wbegin_and_end_phones.end(),
                                 silence_phones.begin(), silence_phones.end());
    SortAndUniq(&wbegin_and_end_phones);
    silence_phones.clear();
  }
}
  
  
bool WordAlignLattice(const CompactLattice &lat,
                      const TransitionModel &tmodel,
                      const WordBoundaryInfo &info,
                      CompactLattice *lat_out) {
  LatticeWordAligner aligner(lat, tmodel, info, lat_out);
  return aligner.AlignLattice();
}



class WordAlignedLatticeTester {
 public:
  WordAlignedLatticeTester(const CompactLattice &lat,
                           const TransitionModel &tmodel,
                           const WordBoundaryInfo &info,
                           const CompactLattice &aligned_lat,
                           bool was_ok):
      lat_(lat), tmodel_(tmodel), info_(info), aligned_lat_(aligned_lat),
      was_ok_(was_ok) {}
  
  void Test() {
    // First test that each aligned arc is valid.
    typedef CompactLattice::StateId StateId ;
    for (StateId s = 0; s < aligned_lat_.NumStates(); s++) {
      for (fst::ArcIterator<CompactLattice> iter(aligned_lat_, s);
           !iter.Done();
           iter.Next()) {
        TestArc(iter.Value());
      }
      if (aligned_lat_.Final(s) != CompactLatticeWeight::Zero()) {
        TestFinal(aligned_lat_.Final(s));
      }
    }
    if(was_ok_)
      TestEquivalent();
  }
 private:
  void TestArc(const CompactLatticeArc &arc) {
    if (! (TestArcSilence(arc, was_ok_) || TestArcNormalWord(arc) || TestArcOnePhoneWord(arc)
           || TestArcEmpty(arc) || (!was_ok_ && TestArcPartialWord(arc))))
      KALDI_ERR << "Invalid arc in aligned CompactLattice: "
                << arc.ilabel << " " << arc.olabel << " " << arc.nextstate
                << " " << arc.weight;
  }
  bool TestArcEmpty(const CompactLatticeArc &arc) {
    if (arc.ilabel != 0) return false; // Check there is no label.  Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    return tids.empty();
  }
  bool TestArcSilence(const CompactLatticeArc &arc, bool was_ok) {
    // This only applies when silence doesn't have word labels.
    if (arc.ilabel !=  info_.silence_label) return false; // Check the label is
    // the silence label. Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    int32 first_phone = tmodel_.TransitionIdToPhone(tids.front());
    if (!std::binary_search(info_.silence_phones.begin(),
                            info_.silence_phones.end(),
                            first_phone)) return false;
    for (size_t i = 0; i < tids.size(); i++)
      if (tmodel_.TransitionIdToPhone(tids[i]) != first_phone) return false;
      
    if (!info_.reorder) return tmodel_.IsFinal(tids.back());
    else {
      for (size_t i = 0; i < tids.size(); i++) {
        if(tmodel_.IsFinal(tids[i])) { // got the "final" transition, which is
          // reordered to actually not be final.  Make sure that all the
          // rest of the transition ids are the self-loop of that same
          // transition-state.
          for(size_t j = i+1; j < tids.size(); j++) {
            if (!(tmodel_.TransitionIdToTransitionState(tids[j])
                  == tmodel_.TransitionIdToTransitionState(tids[i]))) return false;
          }
          return true;
        }
      }
      if (!was_ok_)
        return false; // fell off loop.  No final-state present.
      else
        return true; // OK for no final-state to be present if
      // this is a partial lattice.
    }
  }

  bool TestArcOnePhoneWord(const CompactLatticeArc &arc) {
    if (arc.ilabel == 0) return false; // Check there's a label.  Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    int32 first_phone = tmodel_.TransitionIdToPhone(tids.front());
    if (!std::binary_search(info_.wbegin_and_end_phones.begin(),
                            info_.wbegin_and_end_phones.end(),
                            first_phone)) return false;
    for (size_t i = 0; i < tids.size(); i++)
      if (tmodel_.TransitionIdToPhone(tids[i]) != first_phone) return false;
      
    if (!info_.reorder) return tmodel_.IsFinal(tids.back());
    else {
      for (size_t i = 0; i < tids.size(); i++) {
        if(tmodel_.IsFinal(tids[i])) { // got the "final" transition, which is
          // reordered to actually not be final.  Make sure that all the
          // rest of the transition ids are the self-loop of that same
          // transition-state.
          for(size_t j = i+1; j < tids.size(); j++) {
            if (!(tmodel_.TransitionIdToTransitionState(tids[j]))
                == tmodel_.TransitionIdToTransitionState(tids[i])) return false;
          }
          return true;
        }
      }
      return false; // fell off loop.  No final-state present.
    }
  }

  bool TestArcNormalWord(const CompactLatticeArc &arc) {
    if (arc.ilabel == 0) return false; // Check there's a label.  Note, ilabel==olabel.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    int32 first_phone = tmodel_.TransitionIdToPhone(tids.front());
    if (!std::binary_search(info_.wbegin_phones.begin(),
                            info_.wbegin_phones.end(),
                            first_phone)) return false;
    size_t i;
    { // first phone.
      int num_final = 0;
      for (i = 0; i < tids.size(); i++) {
        if (tmodel_.IsFinal(tids[i])) num_final++;
        if (tmodel_.TransitionIdToPhone(tids[i]) != first_phone) break;
      }
      if (num_final != 1) return false; // Something went wrong-- perhaps we
      // got two beginning phones in a row.
    }
    { // middle phones.  Skip over them.
      while (i < tids.size() &&
             std::binary_search(info_.winternal_phones.begin(),
                                info_.winternal_phones.end(),
                                tmodel_.TransitionIdToPhone(tids[i])))
        i++;
    }
    if (i == tids.size()) return false; // No final phone.
    int32 final_phone = tmodel_.TransitionIdToPhone(tids[i]);
    if (!std::binary_search(info_.wend_phones.begin(),
                            info_.wend_phones.end(),
                            final_phone)) return false; // not word-ending.
    for (size_t j = i; j < tids.size(); j++) // make sure only this final phone till end.
      if (tmodel_.TransitionIdToPhone(tids[j]) != final_phone)
        return false; // Other phones after final phone.
    for (size_t j = i; j < tids.size(); j++) {
      if (tmodel_.IsFinal(tids[j])) { // Found "final transition"..   Note:
        // may be "reordered" with its self loops.
        if (!info_.reorder) return (j+1 == tids.size());
        else {
          // Make sure the only thing that follows this is self-loops
          // of the final transition-state.
          for (size_t k=j+1; k<tids.size(); k++)
            if (tmodel_.TransitionIdToTransitionState(tids[k])
                != tmodel_.TransitionIdToTransitionState(tids[j])
                || !tmodel_.IsSelfLoop(tids[k])) return false;
          return true;
        }
      }
    }
    return false; // Found no final phone.
  }

  bool TestArcPartialWord(const CompactLatticeArc &arc) {
    if (arc.ilabel != info_.partial_word_label) return false; // label should
    // be the partial-word label.
    const std::vector<int32> &tids = arc.weight.String();
    if (tids.empty()) return false;
    return true; // We're pretty liberal when it comes to partial words here.
  }
  
  void TestFinal(const CompactLatticeWeight &w) {
    if (!w.String().empty())
      KALDI_ERR << "Expect to have no strings on final-weights of lattices.";
  }
  void TestEquivalent() {
    CompactLattice aligned_lat(aligned_lat_);
    if (info_.silence_label != 0) { // remove silence labels.
      std::vector<int32> to_remove;
      to_remove.push_back(info_.silence_label);
      RemoveSomeInputSymbols(to_remove, &aligned_lat);
      Project(&aligned_lat, fst::PROJECT_INPUT);
    }

    if (!RandEquivalent(lat_, aligned_lat, 5/*paths*/, 1.0e+10/*delta*/, rand()/*seed*/,
                        200/*path length (max?)*/))
      KALDI_ERR << "Equivalence test failed (testing word-alignment of lattices.)";
  }
  
  const CompactLattice &lat_;
  const TransitionModel &tmodel_;
  const WordBoundaryInfo &info_;
  const CompactLattice &aligned_lat_;
  bool was_ok_;
};
  
  


void TestWordAlignedLattice(const CompactLattice &lat,
                            const TransitionModel &tmodel,
                            const WordBoundaryInfo &info,
                            const CompactLattice &aligned_lat,
                            bool was_ok) {
  WordAlignedLatticeTester t(lat, tmodel, info, aligned_lat, was_ok);
  t.Test();
}






}  // namespace kaldi

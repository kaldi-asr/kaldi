// fstext/compose-trim-inl.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_FSTEXT_COMPOSE_TRIM_INL_H_
#define KALDI_FSTEXT_COMPOSE_TRIM_INL_H_

// Do not include this file directly.  It is included by ComposeTrim.h

namespace fst {

/** ComposeTrimmerOptions is an options class used by ComposeTrim.  The user
    should not need to use this class directly.
*/
struct ComposeTrimmerOptions {
  bool connect;
  bool index_first_fst;
  bool output_first_fst;
  explicit ComposeTrimmerOptions(bool connect_in = true, bool index_fst1_in = true,
                                 bool output_fst1_in = true):
      connect(connect_in), index_first_fst(index_fst1_in), output_first_fst(output_fst1_in) {}
};

// Note that there is no filter in this type of composition.  We can tolerate duplicate
//   arcs because the composed FST is only needed for trimming.
template < class A,
         class M = Matcher<Fst<A> > >
class ComposeTrimmer {
 public:
  typedef A Arc;
  typedef typename A::Label Label;
  typedef typename A::Weight Weight;
  typedef typename A::StateId StateId;
  typedef size_t ArcId;  // offset into arc list.
  typedef typename A::StateId ComposedStateId;

  struct ComposedArc {
    // Information we store for a composed arc.

    Arc output_arc;  // Original arc in the fst that we are outputting (a subset of).
    StateId nextstate;
    ComposedArc(const Arc &output_arc_in, StateId s): output_arc(output_arc_in), nextstate(s) {}
  };

  struct ComposedState {  // Information we store for a composed state.
    bool is_final;
    bool is_coaccessible;
    StateId s1;  // state in fst1_
    StateId s2;  // state in fst2_
    // There is no filter-state as we do not use a filter.
    vector<ComposedArc> arcs;
    ComposedState(StateId s1_in, StateId s2_in): is_final(false), is_coaccessible(false),
                                                 s1(s1_in), s2(s2_in)  {}
  };

  ComposedStateId Initial() {
    // Returns initial state, which is always 0, if FST is nonempty-- creates if necessary.
    if (states_.size() == 0) {
      StateId s1 = fst1_->Start(), s2 = fst2_->Start();
      if (s1 == kNoStateId || s2 == kNoStateId)
        return kNoStateId;
      else
        states_.push_back(new ComposedState(s1, s2));
    }
    return 0;
  }

  ComposedStateId FindStateId(StateId s1, StateId s2) {
    typedef typename std::map<StateId, ComposedStateId>::iterator IterType;
    assert(s1>=0 && s2>=0);
    StateId s1_idx = s1, s2_idx = s2;
    if (! opts_.index_first_fst) std::swap(s1_idx, s2_idx);

    if (static_cast<StateId>(hash_.size()) <= s1_idx) {
      hash_.resize(s1_idx+1, 0);
    }
    if (hash_[s1_idx] == NULL) hash_[s1_idx] = new std::map<StateId, ComposedStateId> ();
    std::map<StateId, ComposedStateId> &m = *(hash_[s1_idx]);

    IterType iter = m.find(s2_idx);
    if (iter == m.end()) {  // no such state -> add it.
      ComposedStateId nextstate = (ComposedStateId) states_.size();
      states_.push_back(new ComposedState(s1, s2));
      Q_.push_back(nextstate);
      m[s2_idx] = nextstate;
      return nextstate;
    } else {
      return iter->second;
    }
  }


  void AddArc(ComposedStateId s, const Arc &arc1,  const Arc &arc2) {
    ComposedStateId d = FindStateId(arc1.nextstate, arc2.nextstate);
    states_[s]->arcs.push_back(ComposedArc(opts_.output_first_fst ? arc1:arc2, d));
  }

  inline void ExpandBy2(ComposedStateId s) {  // Visit each arc in fst2.
    ComposedState &state = *(states_[s]);
    StateId s1 = state.s1, s2 = state.s2;

    // for each arc in fst2_, find a match in fst1_.
    matcher1_->SetState(s1);
    // First the loop label on fst1 (matches epsilons on fst2_).
    Arc loop2(kNoLabel, 0, Weight::One(), s2);  // self-loop in fst2 with \epsilon_L as label.
    if (matcher1_->Find(kNoLabel)) {
      for (; !matcher1_->Done(); matcher1_->Next()) {
        const Arc &arc1 = matcher1_->Value();  // will be eps-output transition  in fst1.
        if (arc1.weight != Weight::Zero()) {
          AddArc(s, arc1, loop2);
        }
      }
    }
    // Now for each real arc in fst2.
    for (ArcIterator<Fst<Arc> > iter2(*fst2_, s2); !iter2.Done(); iter2.Next()) {
      const Arc &arc2 = iter2.Value();
      if (matcher1_->Find(arc2.ilabel)) {
        for (; !matcher1_->Done(); matcher1_->Next()) {
          const Arc &arc1 = matcher1_->Value();  // will be eps-output transition  in fst1.
          if (arc1.weight != Weight::Zero()) {
            AddArc(s, arc1, arc2);
          }
        }
      }
    }
  }

  inline void ExpandBy1(ComposedStateId s) {
    // Visit each arc in fst1.
    // This is a mirror-image of ExpandBy2.
    ComposedState &state = *(states_[s]);
    StateId s1 = state.s1, s2 = state.s2;
    // Match the other way round: for each arc in fst1_ (including loop arc),
    // find match in fst2_.
    matcher2_->SetState(s2);
    // First processing epsilons on fst2_.
    Arc loop1(0, kNoLabel, Weight::One(), s1);  // self-loop in fst1 with \epsilon_L as label.
    if (matcher2_->Find(kNoLabel)) {
      for (; !matcher2_->Done(); matcher2_->Next()) {
        const Arc &arc2 = matcher2_->Value();  // will be eps-output transition  in fst1.
        if (arc2.weight != Weight::Zero()) {
          AddArc(s, loop1, arc2);
        }
      }
    }
    // Now for each real arc in fst1....
    for (ArcIterator<Fst<Arc> > iter1(*fst1_, s1); !iter1.Done(); iter1.Next()) {
      const Arc &arc1 = iter1.Value();
      if (matcher2_->Find(arc1.olabel)) {
        for (; !matcher2_->Done(); matcher2_->Next()) {
          const Arc &arc2 = matcher2_->Value();  // will be eps-output transition  in fst1.
          if (arc2.weight != Weight::Zero()) {
            AddArc(s, arc1, arc2);
          }
        }
      }
    }
  }


  void Expand(ComposedStateId s) {
    ComposedState &state = *(states_[s]);
    StateId s1 = state.s1, s2 = state.s2;

    if (match_type_ == MATCH_OUTPUT ||
       (match_type_ == MATCH_BOTH && fst1_->NumArcs(s1) > fst2_->NumArcs(s2))) {
      ExpandBy2(s);
    } else {
      ExpandBy1(s);
    }
    if (fst1_->Final(s1) != Weight::Zero()  && fst2_->Final(s2) != Weight::Zero())
      states_[s]->is_final = true;
  }

  void FindCoaccessible() {
    // Marks co-accessible states.
    vector<vector<ComposedStateId> > input_trans(states_.size());  // indexed by ComposedStateId.
    // input_trans[s] is a list of all composed states with a transition to s.  Duplicates removed.
    // set up input_trans.
    {
      size_t sz = states_.size(), s;
      for (s = 0;s < sz;s++) {
        vector<ComposedArc> &arcs = states_[s]->arcs;
        typename vector<ComposedArc>::iterator iter = arcs.begin(), end = arcs.end();
        for (; iter != end; ++iter) {
          input_trans[iter->nextstate].push_back(s);
        }
      }
      // Now de-duplicate list of input arcs.
      for (s = 0;s < sz;s++) {
        vector<ComposedStateId> &ids = input_trans[s];
        std::sort(ids.begin(), ids.end());
        typename vector<ComposedStateId>::iterator iter
            = std::unique(ids.begin(), ids.end());
        ids.erase(iter, ids.end());
      }
    }

    vector<ComposedStateId> StateQueue;

    {  // Initialize queue to final states.
      ComposedStateId sz = states_.size();
      for (ComposedStateId s = 0;s < sz;s++) {
        states_[s]->is_coaccessible = states_[s]->is_final;
        if (states_[s]->is_coaccessible) {
          StateQueue.push_back(s);
        }
      }
    }
    while (StateQueue.size() != 0) {  // Main loop.
      ComposedStateId s = StateQueue.back();
      StateQueue.pop_back();
      vector<ComposedStateId> &vec = input_trans[s];
      typename vector<ComposedStateId>::iterator iter = vec.begin(), end = vec.end();
      for (; iter != end ; ++iter) {
        ComposedStateId prev_s = *iter;
        if (! states_[prev_s]->is_coaccessible) {
          states_[prev_s]->is_coaccessible = true;
          StateQueue.push_back(prev_s);
        }
      }
    }
  }

  void Compose() {
    assert(Q_.empty());
    ComposedStateId initial = Initial();
    if (initial != kNoStateId)
      Q_.push_back(initial);
    while (!Q_.empty()) {
      ComposedStateId s = Q_.back();
      Q_.pop_back();
      Expand(s);
    }
  }

  void AllocateStates(const vector<pair<StateId, ComposedStateId> > &pairs,
                      vector<StateId> &mapping) {
    // called
    // For each unique state-id s1 that appears as a "first" elment of "pairs", assign
    // a new integer s1' to it and put the mapping s1 -> s1' in "mapping".
    // For each one, allocates a new state in ofst_.
    // Also sets initial state.
    assert(mapping.size() == 0);
    typename vector<pair<StateId, ComposedStateId> >::const_iterator iter = pairs.begin(), end = pairs.end();
    for (; iter != end; ++iter) {
      StateId s1 = iter->first;
      if (static_cast<StateId>(mapping.size()) <= s1) mapping.resize(s1+1, kNoStateId);
      if (mapping[s1] == kNoStateId) {  // not yet allocated.
        mapping[s1] = ofst_->AddState();
      }
    }
    StateId initial = fst1_->Start();
    assert(!(initial >= static_cast<StateId>(mapping.size()) ||
             mapping[initial] == kNoStateId) &&
           "Output FST should not be empty-- problem in algorithm");
    ofst_->SetStart(mapping[initial]);
  }

  void SwapFsts() {  // called within Output(), when it's the trimmed version of fst2_ we want to modify.
    // Output desired is fst2_, so swap everything and then act like we wanted fst1_.
    std::swap(fst2_, fst1_);  // swap pointers.
    size_t sz = states_.size();
    for (size_t i = 0;i < sz;i++) {
      ComposedState &state = *states_[i];
      std::swap(state.s1, state.s2);
    }
  }

  // Compare class for comparing arcs-- defines total order ("<" operator).
  // Ilabel is compared first-- useful to maintain sorting in ilabel for right fst.
  class ArcSortCompareIlabelFirst {
   public:
    bool operator() (Arc arc1, Arc arc2) const {
      if (arc1.ilabel < arc2.ilabel) return true;
      else if (arc1.ilabel > arc2.ilabel) return false;
      else if (arc1.olabel < arc2.olabel) return true;
      else if (arc1.olabel > arc2.olabel) return false;
      else if (arc1.nextstate < arc2.nextstate) return true;
      else if (arc1.nextstate > arc2.nextstate) return false;
      else return (arc1.weight.Value() < arc2.weight.Value());
    }
  };

    // Compare class for comparing arcs-- defines total order ("<" operator).
  // Olabel is compared first-- useful to maintain sorting in olabel for left fst
  class ArcSortCompareOlabelFirst {
   public:
    bool operator() (Arc arc1, Arc arc2) const {
      if (arc1.olabel < arc2.olabel) return true;
      else if (arc1.olabel > arc2.olabel) return false;
      else if (arc1.ilabel < arc2.ilabel) return true;
      else if (arc1.ilabel > arc2.ilabel) return false;
      else if (arc1.nextstate < arc2.nextstate) return true;
      else if (arc1.nextstate > arc2.nextstate) return false;
      else return (arc1.weight.Value() < arc2.weight.Value());
    }
  };

  // Predicate class for comparing arcs.
  class ArcEqual {
   public:
    bool operator() (Arc arc1, Arc arc2) const {
      return (arc1.ilabel == arc2.ilabel &&
              arc1.olabel == arc2.olabel &&
              arc1.nextstate == arc2.nextstate &&
              arc1.weight == arc2.weight);
    }
  };


  void ProcessOutputState(StateId s1, const vector<ComposedStateId> &composed_states,
                          const vector<StateId> &mapping, vector<Arc> &temp_space) {
    // composed_states is the subset of (co-accessible) composed states that
    // share this "original" state.
    vector<Arc> &all_arcs(temp_space);
    all_arcs.clear();  // memory will still be there.

    bool was_final = false;
    for (size_t idx = 0;idx < composed_states.size();idx++) {
      ComposedStateId s = composed_states[idx];
      ComposedState &state = *states_[s];
      if (state.is_final)  was_final = true;
      typename vector<ComposedArc>::iterator aiter = state.arcs.begin(), aend = state.arcs.end();
      for (; aiter!= aend; ++aiter) {
        ComposedArc &carc = *aiter;
        Arc &arc = carc.output_arc;  // arc in the fst we are outputting a subset of.
        if (arc.ilabel != kNoLabel && arc.olabel != kNoLabel // Not the loop arc...
           && states_[carc.nextstate]->is_coaccessible) {  // note-- important this is carc not arc.
          all_arcs.push_back(arc);
        }
      }
    }

    {  // Now de-duplicate all_arcs.
      // The reason for sorting is to make duplicates appear next to each other, but
      // we choose different sorting function depending whether these arcs are from
      // the left or right FST, for convenience so the output can be used in composition--
      // this maintains the appropriate sorted order in the output.
      if (opts_.output_first_fst)
        std::sort(all_arcs.begin(), all_arcs.end(), ArcSortCompareOlabelFirst());
      else
        std::sort(all_arcs.begin(), all_arcs.end(), ArcSortCompareIlabelFirst());
      typename vector<Arc>::iterator iter
          = std::unique(all_arcs.begin(), all_arcs.end(), ArcEqual());
      all_arcs.erase(iter, all_arcs.end());  // remove duplicates.
    }

    StateId s1_dash = mapping[s1];

    typename vector<Arc>::iterator iter = all_arcs.begin(), end = all_arcs.end();
    for (; iter!= end; ++iter) {
      assert(s1_dash != kNoStateId);
      Arc &arc = *iter;
      arc.nextstate = mapping[arc.nextstate];
      assert(arc.nextstate != kNoStateId);
      ofst_->AddArc(s1_dash, arc);
    }
    if (was_final) {
      assert(fst1_->Final(s1) != Weight::Zero());
      ofst_->SetFinal(s1_dash, fst1_->Final(s1));
    }
  }

  void Output() {

    {  // First free up memory that is not needed, namely in hash_.
      for (size_t i = 0;i < hash_.size();i++) if (hash_[i]) delete hash_[i];
      { vector<std::map<StateId, ComposedStateId>* > tmp; tmp.swap(hash_); }
    }

    if (! opts_.output_first_fst)
      SwapFsts();

    assert(ofst_->NumStates() == 0);
    ofst_->SetInputSymbols(fst1_->InputSymbols());
    ofst_->SetOutputSymbols(fst1_->OutputSymbols());
    if (states_.size() == 0) return;  // Nothing to do-- composition is empty fst.


    // From now we assume we want to output to ofst_, an FST that's equivalent to
    // fst1_, except it only contains the states and arcs and final-weights
    // that are accessed after composition (and trimming).

    // First step is: we want to process things in order according to the
    // state-id in fst1_.  This helps us avoid using too much storage.
    // To do this we first construct a set of pairs of (state-1, composed-state).
    size_t sz = states_.size();
    vector<pair<StateId, ComposedStateId> > pairs(sz);
    size_t idx = 0;
    for (size_t i = 0;i < sz;i++) {
      if (states_[i]->is_coaccessible)
        pairs[idx++] = pair<StateId, ComposedStateId>(states_[i]->s1, (ComposedStateId) i);
    }
    if (idx == 0) { return; }  // No states are coaccessible -> return.  Output is empty.
    pairs.resize(idx);
    std::sort(pairs.begin(), pairs.end());  // gets them in order by s1.
    // Now find contiguous regions by s1.

    vector<StateId> mapping;
    AllocateStates(pairs, mapping);
    // now mapping is from s1 to "new s1"

    typename vector<pair<StateId, ComposedStateId> >::iterator iter = pairs.begin(), end = pairs.end();
    vector<Arc> temp_space;
    while (iter != end) {  // Note, we are only iterating over coaccessible states.
      StateId s1 = iter->first;
      vector<ComposedStateId> composed_states;
      while (iter != end && iter->first == s1) {
        composed_states.push_back(iter->second);
        ++iter;
      }
      ProcessOutputState(s1, composed_states, mapping, temp_space);  // adds arcs and final-state info to ofst_.
    }
    // we maintained sorted order of arcs as we want, thanks to the stuff with
    // ArcSortCompareOlabelFirst vs.  ArcSortCompareIlabelFirst above.
    if (opts_.output_first_fst)
      ofst_->SetProperties(kOLabelSorted, kOLabelSorted|kNotOLabelSorted);
    else
      ofst_->SetProperties(kILabelSorted, kILabelSorted|kNotILabelSorted);
  }


  ComposeTrimmer(const Fst<A> &fst1, const Fst<A> &fst2,
                 MutableFst<A> *ofst,
                 ComposeTrimmerOptions opts = ComposeTrimmerOptions()):
      fst1_(&fst1), fst2_(&fst2), ofst_(ofst), matcher1_(new M(*fst1_, MATCH_OUTPUT)),
      matcher2_(new M(*fst2_, MATCH_INPUT)), opts_(opts)
  {  // pointers to fst1_, fst2_, ofst_ are not owned-- no point since everythying
    // happens in the initializer.

    // Work out the "match type"-- i.e. whether one FST, or both, is sorted
    // in the appropriate way for fast search during composition.
    MatchType type1 = matcher1_->Type(false);
    MatchType type2 = matcher2_->Type(false);
    if (type1 == MATCH_OUTPUT && type2  == MATCH_INPUT) {
      match_type_ = MATCH_BOTH;
    } else if (type1 == MATCH_OUTPUT) {
      match_type_ = MATCH_OUTPUT;
    } else if (type2 == MATCH_INPUT) {
      match_type_ = MATCH_INPUT;
    } else if (matcher1_->Type(true) == MATCH_OUTPUT) {
      match_type_ = MATCH_OUTPUT;
    } else if (matcher2_->Type(true) == MATCH_INPUT) {
      match_type_ = MATCH_INPUT;
    } else {
      LOG(FATAL) << "ComposeTrim: 1st argument cannot match on output labels "
                 << "and 2nd argument cannot match on input labels (sort?).";
    }

    Compose();
    if (opts.connect) FindCoaccessible();
    else for (size_t s = 0;s<states_.size();s++) states_[s]->is_coaccessible = true;
    Output();
  }



  ~ComposeTrimmer() {
    delete matcher1_;
    delete matcher2_;
    for (size_t i = 0;i < states_.size();i++) delete states_[i];
    for (size_t i = 0;i < hash_.size();i++) if (hash_[i]) delete hash_[i];
  }


 private:
  vector<std::map<StateId, ComposedStateId>* > hash_;  // hash_[state1][state2] = composed_state.
  vector<ComposedState*> states_;
  vector<ComposedStateId> Q_;
  const Fst<A> *fst1_;  // pointer not owned.
  const Fst<A> *fst2_;  // pointer not owned.
  MutableFst<A> *ofst_;  // Trimmed version of fst1_;  Pointer not owned.
  M *matcher1_;  // M is type of matcher.
  M *matcher2_;
  ComposeTrimmerOptions opts_;
  MatchType match_type_;  // MATCH_OUTPUT, MATCH_INPUT, or MATCH_BOTH-- depending which way(s) fst1_ and fst2_ are sorted.
  DISALLOW_COPY_AND_ASSIGN(ComposeTrimmer);
};




template<class Arc>
void ComposeTrimLeft(const Fst<Arc> &fst1, const Fst<Arc> &fst2, bool connect, MutableFst<Arc> *ofst1) {
  ComposeTrimmerOptions opts;  // All default.
  opts.connect = connect;
  opts.output_first_fst = true;
  ComposeTrimmer<Arc, SortedMatcher<Fst<Arc> > > trimmer(fst1, fst2, ofst1, opts);  // All the work gets done in the initializer.
}


template<class Arc>
void ComposeTrimRight(const Fst<Arc> &fst1, const Fst<Arc> &fst2, bool connect, MutableFst<Arc> *ofst2) {
  ComposeTrimmerOptions opts;  // All default.
  opts.connect = connect;
  opts.output_first_fst = false;  // outputs to fst2 instead.
  ComposeTrimmer<Arc, SortedMatcher<Fst<Arc> > > trimmer(fst1, fst2, ofst2, opts);  // All the work gets done in the initializer.
}


}

#endif

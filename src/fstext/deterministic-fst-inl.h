// fstext/deterministic-fst-inl.h

// Copyright 2011 Gilles Boulianne

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

#ifndef KALDI_FSTEXT_DETERMINISTIC_FST_INL_H_
#define KALDI_FSTEXT_DETERMINISTIC_FST_INL_H_
#include "base/kaldi-common.h"
#include "fstext/fstext-utils.h"

// Do not include this file directly.  It is included by deterministic-fst.h.

// NOTES: normally only the composition case should use the CacheImpl. In the
//        single FST case we could just return values from the FST without
//        recording anything to the cache. However the cache is used anyway
//        to make it clearer what will be needed for the composition implementation.


namespace fst {

/// \addtogroup deterministic_fst_group
/// @{

// in addition to CacheImpl, there is a cache for GetArc() results
template<class Arc>
bool DeterministicOnDemandFstImpl<Arc>::HasArc(StateId s, Label l) {
  cache_calls_++;
  StateLabelPair sl = make_pair(s,l);
  scm_it_ = state_cache_map_.find(sl);
  if (scm_it_ != state_cache_map_.end()){
    cache_hits_++;
    return true;
  } else {
    return false;
  }
}

template<class Arc>
void DeterministicOnDemandFstImpl<Arc>::AddSingleArc(StateId s, Label l, Arc& a) {
  StateLabelPair sl = make_pair(s,l);
  state_cache_map_[sl] = a;
}

template<class Arc>
void DeterministicOnDemandFstImpl<Arc>::SetArc(StateId s, Label l) {
  // add an invalid arc with a kNoStateId nextstate
  Arc a(0,0,0,kNoStateId);
  DeterministicOnDemandFstImpl<Arc>::AddSingleArc(s,l,a);
}

template<class Arc>
bool DeterministicOnDemandFstImpl<Arc>::GetCachedArc(StateId s, Label l, Arc *oarc) {
  // assume HasArc() was just called and return true, thus scm_it_ is properly set
  *oarc = scm_it_->second;
  return (oarc->nextstate != kNoStateId);  // return true if valid arc
}

template<class Arc>
typename DeterministicOnDemandFstImpl<Arc>::StateId DeterministicOnDemandFstImpl<Arc>::Start() {
  if (! CacheImpl<Arc>::HasStart()) {
    StateId s = 0;
    if (fst2_) {
      // composition
      StateId s1 = fst1_->Start(), s2 = fst2_->Start();
      if (s1 == kNoStateId || s2 == kNoStateId) return kNoStateId;
      if (composedState_.size()==0) AddComposedState(s1,s2);
    } else {
      // single FST case
      s = fst1_->Start();
    }
    assert(s == 0);
    SetStart(s);
    return s;
  }
  return CacheImpl<Arc>::Start();
}

// Constructor from single FST
template<class Arc>
  DeterministicOnDemandFstImpl<Arc>::DeterministicOnDemandFstImpl(const Fst<Arc>&fst): 
      cache_calls_(0), cache_hits_(0), fst1_(&fst), fst2_(NULL),
      scm_it_(state_cache_map_.end()) { // initialize scm_it_ to keep older compilers happy.
    if (!fst1_->Properties(kILabelSorted, true))
      KALDI_ERR << "DeterministicOnDemandFst: error: input FST must be input label sorted";
    if (!fst1_->Properties(kIDeterministic, true))
      KALDI_ERR << "DeterministicOnDemandFst: error: input Fst must be input-label deterministic.";
  SetType("deterministic");
}

// Constructor from 2 DODF
template<class Arc>
DeterministicOnDemandFstImpl<Arc>::DeterministicOnDemandFstImpl(const Fst<Arc> &fst1,const Fst<Arc> &fst2):
    cache_calls_(0), cache_hits_(0), fst1_(&fst1), fst2_(&fst2),
    scm_it_(state_cache_map_.end())  { // initialize iterator to keep older compilers happy
  if (!fst1_->Properties(kILabelSorted, true) || !fst2_->Properties(kILabelSorted, true))
	KALDI_ERR << "DeterministicOnDemandFst: error: input FST's must be input label sorted";
  if (!fst1_->Properties(kIDeterministic, true) || !fst2_->Properties(kIDeterministic, true))
	KALDI_ERR << "DeterministicOnDemandFst: error: input FST's must be input-label deterministic";
  SetType("deterministic");
}

template<class Arc>
DeterministicOnDemandFstImpl<Arc>::DeterministicOnDemandFstImpl(const DeterministicOnDemandFstImpl<Arc> &other):
    scm_it_(state_cache_map_.end()) // initialize iterator to keep older compilers happy.
{
  /* to be implemented */
  KALDI_ERR << "DeterministicOnDemandFst copying not yet supported.";
}

template<class Arc>
typename DeterministicOnDemandFstImpl<Arc>::Weight
DeterministicOnDemandFstImpl<Arc>::Final(StateId s) {
  if (!HasFinal(s)) {  
    // Work out final-state weight.
    Weight w;
    if (fst2_){
      StatePair sp = composedState_[s];
      w = Times(GetFinalFromNonDetFst(fst1_, sp.first),
                GetFinalFromNonDetFst(fst2_, sp.second));
    } else {
      w = GetFinalFromNonDetFst(fst1_, s);
    }
    SetFinal(s, w);
    return w;
  }
  return CacheImpl<Arc>::Final(s);
}

template<class Arc>
size_t DeterministicOnDemandFstImpl<Arc>::NumArcs(StateId s) {
  if (!HasArcs(s))
    Expand(s);
  return CacheImpl<Arc>::NumArcs(s);
}

template<class Arc>
size_t DeterministicOnDemandFstImpl<Arc>::NumOutputEpsilons(StateId s) {
  if (!HasArcs(s))
    Expand(s);
  return CacheImpl<Arc>::NumOutputEpsilons(s);
}

template<class Arc>
void DeterministicOnDemandFstImpl<Arc>::InitArcIterator(StateId s, ArcIteratorData<Arc> *data) {
  if (!HasArcs(s))
    Expand(s);
  CacheImpl<Arc>::InitArcIterator(s, data);
}


template<class Arc>
typename Arc::Weight DeterministicOnDemandFstImpl<Arc>::GetFinalFromNonDetFst(
    const Fst<Arc>* fst, StateId s) {
  Weight w = fst->Final(s);
  if (w != Weight::Zero()) return w;
  SortedMatcher<Fst<Arc> > sm(*fst, MATCH_INPUT, 1); // enable matching epsilons as well
  sm.SetState(s);
  if (sm.Find(kNoLabel)) {   // kNoLabel will match epsilons, but not
    // the implicit self-loop; see matcher.h.
    const Arc &arc = sm.Value();
    return Times(arc.weight, GetFinalFromNonDetFst(fst, arc.nextstate));
  }
  return Weight::Zero();
}

// helper method for GetArc()
template<class Arc>
bool DeterministicOnDemandFstImpl<Arc>::GetArcFromNonDetFst(
    const Fst<Arc>* fst, StateId s, Label ilabel, Arc *oarc,
    typename Arc::Weight iweight) {
  
  // use a SortedMatcher
  // fst should already have been tested for correct "sortedness"
  SortedMatcher<Fst<Arc> > sm(*fst, MATCH_INPUT, 1);
  sm.SetState(s);
  if (sm.Find(ilabel)) {
    const Arc &arc = sm.Value();
    *oarc = Arc(ilabel, arc.olabel, Times(arc.weight, iweight), arc.nextstate);
    return true;
  }
  // didn't find matching ilabel, look for an epsilon transition and take it
  // Note: "kNoLabel" matches any "non-consuming" transitions including epsilons;
  // we use this not 0 because 0 would match the implicit self-loop and we don't
  // want this.
  if (sm.Find(kNoLabel)) {
    const Arc &arc = sm.Value();
    return DeterministicOnDemandFstImpl<Arc>::GetArcFromNonDetFst(
        fst, arc.nextstate, ilabel, oarc, Times(arc.weight,iweight));

  }    
  return false;  // otherwise, no match is possible
}

// This function is specific to DeterministicOnDemandFst.
template<class Arc>
bool DeterministicOnDemandFstImpl<Arc>::GetArc(StateId s, Label ilabel, Arc *oarc) {

  // Check first if we already have the result in cache
  if (HasArc(s, ilabel)) {
    return(GetCachedArc(s, ilabel, oarc));
  } 
  // otherwise compute the arc and set cache info
  if (fst2_) {
    // composition case
    StatePair sp = composedState_[s];
    Arc arc1, arc2;
    bool r1 = GetArcFromNonDetFst(fst1_, sp.first, ilabel, &arc1); 
    if (!r1) {
      SetArc(s, ilabel); // set without arc added
      return false;
    }
    if (arc1.olabel == 0) {
      // we don't move in fst2_
      *oarc = Arc(ilabel, 0, arc1.weight, AddComposedState(arc1.nextstate,sp.second));
      AddSingleArc(s, ilabel, *oarc); // add 1 arc and set
      return true;
    }
    bool r2 = GetArcFromNonDetFst(fst2_, sp.second, arc1.olabel, &arc2);
    if (!r2) {
      SetArc(s,ilabel); // set without arc added
      return false;
    }
    // we have matching arcs from both FSTs, compute composed arc
    *oarc = Arc(ilabel, arc2.olabel,
				Times(arc1.weight, arc2.weight), 
				AddComposedState(arc1.nextstate, arc2.nextstate));
    AddSingleArc(s, ilabel, *oarc); // add 1 arc and set
    return true;
  } else {
    // single FST case: not cached
    return GetArcFromNonDetFst(fst1_, s, ilabel, oarc);
  }
}

// Note that Expand is not called if we do the composition using
// GetArc(), which is the normal case.
template<class Arc>
void DeterministicOnDemandFstImpl<Arc>::Expand(StateId s) {  // expands arcs only [not final state weight].

  if (fst2_) {
    KALDI_ERR << "DeterministicOnDemandFstImpl<Arc>::Expand() not implemented for the composition case.";
  }
  // Add all the arcs to the cache. Just for clarity, not efficiency.
  for (ArcIterator<Fst<Arc> > aiter(*fst1_, s); !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    AddArc(s, arc);
  }  
  SetArcs(s);  // mark the arcs as "done". [so HasArcs returns true].
}

template<class Arc>
DeterministicOnDemandFst<Arc>::DeterministicOnDemandFst(const DeterministicOnDemandFst<Arc> &fst, bool reset) {
  if (reset) {
    impl_ = new DeterministicOnDemandFstImpl<Arc>(*(fst.impl_));
    // Copy constructor of DeterministicOnDemandFstImpl.
    // Main use of calling with reset = true is to free up memory
    // (e.g. then you could delete original one).
  } else {
    impl_ = fst.impl_;
    impl_->IncrRefCount();
  }
}

///

} // end namespace fst


#endif

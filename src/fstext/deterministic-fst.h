// fstext/deterministic-fst.h

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
//
// This file includes material from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: riley@google.com (Michael Riley)

#ifndef KALDI_FSTEXT_DETERMINISTIC_FST_H_
#define KALDI_FSTEXT_DETERMINISTIC_FST_H_

/* This header defines the DeterministicOnDemand interface,
   which is an FST with a special interface that allows
   only a single arc with a non-epsilon input symbol
   out of each state.
*/

#include <algorithm>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;

#include <string>
#include <utility>
#include <vector>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fst/slist.h>

#include "util/const-integer-set.h"

namespace fst {

/// \addtogroup deterministic_fst_group "Classes and functions related to on-demand deterministic FST's"
/// @{

/*
   DeterministicOnDemandFstImpl inherits from CacheImpl, which handles caching of states.
*/


template < class Arc> 
class DeterministicOnDemandFstImpl : public CacheImpl<Arc> {
 public:

  // Inherit the stuff about setting "type", properties and symbol-tables, from
  // FstImpl, which we inherit from (in a long chain) via CacheImpl<Arc>.
  using FstImpl<Arc>::SetType;
  using FstImpl<Arc>::SetProperties;
  using FstImpl<Arc>::Properties;
  using FstImpl<Arc>::SetInputSymbols;
  using FstImpl<Arc>::SetOutputSymbols;

  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  // Construct from an existing Fst, treating it deterministically as in a "backoff language model"
  DeterministicOnDemandFstImpl(const Fst<Arc>&fst);  // defined in -inl.h

  // On-demand composition between two DeterministicOnDemandFst's
  // We need a DODF on the left so that the result is also DODF 
  DeterministicOnDemandFstImpl(const Fst<Arc> &fst1, const Fst<Arc> &fst2);   // defined in -inl.h

  // Copy constructor
  DeterministicOnDemandFstImpl(const DeterministicOnDemandFstImpl<Arc> &other);  // defined in -inl.h

  ~DeterministicOnDemandFstImpl() {}

  StateId Start();           // defined in -inl.h

  Weight Final(StateId s);   // defined in -inl.h

  // This expands the state.
  size_t NumArcs(StateId s); // defined in -inl.h

  // This expands the state.
  size_t NumInputEpsilons(StateId s) { return 0; }

  size_t NumOutputEpsilons(StateId s); // defined in -inl.h

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data);  // defined in -inl.h

  // This function is specific to DeterministicOnDemandFst.  It's not part of
  // the Fst interface.  If there is an arc from state s with label ilabel,
  // it returns true and assigns the arc to *arc. Otherwise it returns false.
  bool GetArc(StateId s, Label ilabel, Arc *oarc); // defined in -inl.h

  // Note that Expand is not normally called.
  void Expand(StateId s);               // defined in -inl.h

  unsigned int cache_calls_, cache_hits_;

private:
   
  // there are 2 cases:
  //    - there is a single underlying FST (then fst2_ is NULL)
  //    - the composition case, where there are 2 underlying FSTs (then fst2_ is not NULL)
  const Fst<Arc> *fst1_;
  const Fst<Arc> *fst2_;
  
  // private helper method for GetArc()
  // Used to get first matching arc from one of the underlying non-deterministic FST's
  // treating it "deterministically":
  // It returns the first arc found with input ilabel out of state s.
  // If there is none, but there is an arc with input epsilon, this arc is followed
  // (recursively and cumulating the weights) until an arc with input ilabel is found.
  // If called with input ilabel equal to epsilon, treats it as any other label
  // (i.e. matches it only with epsilon labels).
  static bool GetArcFromNonDetFst(const Fst<Arc> *fst, StateId s, Label ilabel,
                                  Arc *oarc, Weight iweight = Weight::One());
  
  // private helper method for GetFinal().  If current state is final returns it;
  // else follows epsilons recursively till it finds a final state and returns the
  // first one it finds (or Zero() if none found).
  Weight GetFinalFromNonDetFst(const Fst<Arc> *fst, StateId s);
  
  // state management for composition
  typedef std::pair<StateId, StateId> StatePair;

  class StatePairEqual {
   public:
    bool operator()(const StatePair &x, const StatePair &y) const {
      return x.first == y.first && x.second == y.second;
    }
  };

  class StatePairKey{
   public:
    size_t operator()(const StatePair &x) const {
      return static_cast<size_t>(x.first*kPrime+x.second);
	}
  private:
    static const int kPrime = 7853;
  };

  typedef unordered_map<StatePair, StateId, StatePairKey, StatePairEqual> StateMap;  // map to composed StateId

  StateMap state_map_;   // map from state in fst1 and fst2 to composed state
  std::vector<StatePair> composed_state_;               // indexed by composed StateId 

  // add composed state to internal data
  StateId AddComposedState(StateId s1, StateId s2) {
    StatePair sp = make_pair(s1, s2);
    typename StateMap::iterator mit = state_map_.find(sp);
    StateId cs;
    if (mit == state_map_.end()) {
      // new, add it
      cs = composed_state_.size();
      composed_state_.push_back(sp);
      state_map_[sp] = cs;
      // cerr << "Adding composed state ("<<s1<<","<<s2<<") = "<<cs<<endl;
    } else {
      cs = (*mit).second;
    }
    return cs;
  }

  // cache for GetArc() results
  typedef std::pair<StateId, Label> StateLabelPair;
  
  typedef unordered_map<StateLabelPair, Arc, StatePairKey > StateLabelMap;

  StateLabelMap state_cache_map_;

  typename StateLabelMap::iterator scm_it_;
  bool HasArc(StateId s, Label l);
  void AddSingleArc(StateId s, Label l, Arc &a);
  void SetArc(StateId s, Label l);
  bool GetCachedArc(StateId s, Label l, Arc *oarc);
    
};


/*
   Actual FST for DeterministicOnDemandFst.  
   Most of the work gets done in DeterministicOnDemandFstImpl.

   A DeterministicOnDemandFst is a transducer that has only non-epsilon input symbols,
   and a single arc per state for a given input symbol.  It is an on-demand FST.  
   However, it does not create itself in the usual way by expanding states by 
   enumerating all their arcs.It is only possible to see if a state has an arc out
   for a particular symbol.

*/

template <class Arc>
class DeterministicOnDemandFst : public Fst<Arc> {
 public:
  friend class ArcIterator< DeterministicOnDemandFst<Arc> >;
  friend class StateIterator< DeterministicOnDemandFst<Arc> >;
  friend class CacheArcIterator< DeterministicOnDemandFst<Arc> >;

  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef CacheState<Arc> State;

  DeterministicOnDemandFst(const Fst<Arc> &fst): impl_(new DeterministicOnDemandFstImpl<Arc>(fst)) {}

  DeterministicOnDemandFst(const DeterministicOnDemandFst<Arc> &fst, bool reset = false); // defined in -inl.h

  DeterministicOnDemandFst(const Fst<Arc> &fst1, const Fst<Arc> &fst2) :
   	impl_(new DeterministicOnDemandFstImpl<Arc>(fst1,fst2)) {}   // no options yet

  virtual ~DeterministicOnDemandFst() { if (!impl_->DecrRefCount()) delete impl_;  }

  virtual StateId Start() const { return impl_->Start(); }

  virtual Weight Final(StateId s) const { return impl_->Final(s); }

  StateId NumStates() const { return impl_->NumStates(); }

  virtual bool GetArc(StateId s, Label olabel, Arc *oarc) const {
    return impl_->GetArc(s, olabel, oarc);
  }

  unsigned int CacheCalls() {return impl_->cache_calls_;}
  unsigned int CacheHits() {return impl_->cache_hits_;}

  size_t NumArcs(StateId s) const { return impl_->NumArcs(s); }

  size_t NumInputEpsilons(StateId s) const {
    return impl_->NumInputEpsilons(s);
  }

  size_t NumOutputEpsilons(StateId s) const {
    return impl_->NumOutputEpsilons(s);
  }

  virtual uint64 Properties(uint64 mask, bool test) const {
    if (test) {
      uint64 known, test = TestProperties(*this, mask, &known);
      impl_->SetProperties(test, known);
      return test & mask;
    } else {
      return impl_->Properties(mask);
    }
  }

  virtual const string& Type() const { return impl_->Type(); }

  virtual DeterministicOnDemandFst<Arc>  *Copy(bool reset = false) const {
    return new DeterministicOnDemandFst<Arc>(*this, reset);
  }

  virtual const SymbolTable* InputSymbols() const {
    return impl_->InputSymbols();
  }

  virtual const SymbolTable* OutputSymbols() const {
    return impl_->OutputSymbols();
  }

  virtual inline void InitStateIterator(StateIteratorData<Arc> *data) const;

  virtual void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const {
    impl_->InitArcIterator(s, data);
  }

  friend class CacheStateIterator<DeterministicOnDemandFst<Arc> >;  // so it can see impl_.
 private:
  // visible to friends:
  DeterministicOnDemandFstImpl<Arc> *GetImpl() const { return impl_; }

  DeterministicOnDemandFstImpl<Arc> *impl_; 
  void operator = (const DeterministicOnDemandFstImpl<Arc> &fst);  // disallow
};

// Specialization for DeterministicOnDemandFst, of StateIterator.
// Just directs to use the one from CacheFst.
template<class A>
class StateIterator< DeterministicOnDemandFst<A> >
    : public CacheStateIterator< DeterministicOnDemandFst<A> > {
 public:
  explicit StateIterator(const DeterministicOnDemandFst<A> &fst)
    : CacheStateIterator< DeterministicOnDemandFst<A> >(fst, fst.GetImpl()) {}
};


// Specialization for DeterministicOnDemandFst, of ArcIterator.
// Just directs to use the one from CacheFst.
template <class A>
class ArcIterator< DeterministicOnDemandFst<A> >
    : public CacheArcIterator< DeterministicOnDemandFst<A> > {
 public:
  typedef typename A::StateId StateId;

  ArcIterator(const DeterministicOnDemandFst<A> &fst, StateId s)
    : CacheArcIterator< DeterministicOnDemandFst<A> >(fst.GetImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) // arcs not already computed.
      fst.GetImpl()->Expand(s);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(ArcIterator);
};

template <class A> inline
void DeterministicOnDemandFst<A>::InitStateIterator(StateIteratorData<A> *data) const {
  data->base = new StateIterator< DeterministicOnDemandFst<A> >(*this);
}


/// @}
}  // namespace fst

#include "deterministic-fst-inl.h"

#endif

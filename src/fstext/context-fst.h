// fstext/context-fst.h

// Copyright 2009-2011  Microsoft Corporation

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
//
// This file includes material from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
// See ../../COPYING for clarification regarding multiple authors
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



#ifndef KALDI_FSTEXT_CONTEXT_FST_H_
#define KALDI_FSTEXT_CONTEXT_FST_H_

/* This header defines a context FST "C" which transduces from symbols representing phone
   context windows (e.g. "a, b, c") to individual phone, e.g. "a".  The context
   FST is an on-demand FST.  It has its own matcher type that makes it particularly
   efficient to compose with.
*/

#include <algorithm>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fst/slist.h>

#include "util/const-integer-set.h"

namespace fst {

/// \addtogroup context_fst_group "Classes and functions related to context expansion"
/// @{

/*
   ContextFstImpl inherits from CacheImpl, which handles caching of states.
*/


template < class Arc,
         class LabelT = int32> // make the vector<Label> things actually vector<int32> for
                             // easier compatibility with Kaldi code.
class ContextFstImpl : public CacheImpl<Arc> {
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

  typedef unordered_map<vector<LabelT>,
                        StateId, kaldi::VectorHasher<LabelT> > VectorToStateType;
  typedef unordered_map<vector<LabelT>,
                        Label, kaldi::VectorHasher<LabelT> > VectorToLabelType;

  typedef typename VectorToStateType::const_iterator VectorToStateIter;
  typedef typename VectorToLabelType::const_iterator VectorToLabelIter;

  ContextFstImpl(Label subsequential_symbol,  // epsilon not allowed.
                 const vector<LabelT> &phones,
                 const vector<LabelT> &disambig_syms,
                 int32 N,  // size of ctx window
                 int32 P);

  ContextFstImpl(const ContextFstImpl &other);

  ~ContextFstImpl() { }

  const vector<vector<LabelT> > &ILabelInfo() { return ilabel_info_; }

  StateId Start();

  Weight Final(StateId s);

  // Warning!  Not fully tested for correctness.  Does not really matter, the
  // way this function is being used so far.
  size_t NumArcs(StateId s);

  size_t NumInputEpsilons(StateId s);

  size_t NumOutputEpsilons(StateId s) { return 0; }

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data);

  // This function is specific to ContextFst.  It's not part of
  // the Fst interface.  It attempts to create an arc out of state s,
  // with output label "olabel" [it works out the input label from
  // the value of "olabel".  It returns true if it is able to create
  // an arc, and false otherwise.
  bool CreateArc(StateId s, Label olabel, Arc *oarc);

  // Note that Expand is not called if we do the composition using
  // ContextMatcher<Arc>.
  // This function expands arcs only [not final state weight].
  void Expand(StateId s);

 private:
  //! Finds state-id corresponding to this vector of phones.  Inserts it if necessary.
  StateId FindState(const vector<LabelT> &seq);

  //! Finds the label index corresponding to this context-window of phones.
  //! Inserts it if necessary.
  Label FindLabel(const vector<LabelT> &label_info);

  // Ask whether symbol on output side is disambiguation symbol.
  bool IsDisambigSymbol(Label lab) {  return (disambig_syms_.count(lab) != 0); }

  // Ask whether symbol on input side is disambiguation symbol.
  bool IsPhoneSymbol(Label lab) {  return (phone_syms_.count(lab) != 0); }

  inline void CreateDisambigArc(StateId s, Label olabel, Arc *oarc);  // called from CreateArc.

  inline bool CreatePhoneOrEpsArc(StateId src, StateId dst, Label olabel,
                                  const vector<LabelT> &phone_seq, Arc *oarc);

  // maps from vector<LabelT> to StateId.
  VectorToStateType state_map_;
  vector<vector<LabelT> > state_seqs_;

  // maps from vector<LabelT> to Label
  VectorToLabelType ilabel_map_;
  vector<vector<LabelT> > ilabel_info_;

  // Stuff we were provided at input (but changed to more convenient form):
  kaldi::ConstIntegerSet<Label> phone_syms_;
  kaldi::ConstIntegerSet<Label> disambig_syms_;
  Label subsequential_symbol_;
  int N_;
  int P_;
  int pseudo_eps_symbol_;  // This is the symbol we put on epsilon arcs at the start
  // of the graph.  If we have "real" disambiguation symbols AND N > P+1, then this cannot be
  // epsilon or there is a danger of non-determinizable output.  It's because in this case,
  // the disambiguation symbols are shifted left w.r.t. the phones, and there becomes
  // and ambiguity if a disambiguation symbol appears at the start of a sequence onthe
  // input of CLG, whether it was at the very start of the input of LG, or just after, say,
  // the first real phone.  What we do if we need pseudo_eps_symbol_ to be not epsilon,
  // we create a special symbol with symbol-id 1 and sequence representation (ilabels entry)
  // [ 0 ] .  In the printed form we call this #-1.
  std::string separator_;
};


/*
   Actual FST for ContextFst.  Most of the work gets done in ContextFstImpl.

   A ContextFst is a transducer from symbols representing phones-in-context,
   to phones.  It is an on-demand FST.  However, it does not create itself in the usual
   way by expanding states by enumerating all their arcs.  This is possible to enable
   iterating over arcs, but it is not recommended.  Instead, we define a special
   Matcher class that knows how to request the specific arc corresponding to a particular
   output label.

   This class requires a list of all the phones and disambiguation
   symbols, plus the subsequential symbol.  This is required to be able to
   enumerate all output symbols (if we want to access it in an inefficient way), and
   also to distinguish between phones and disambiguation symbols.
*/

template <class Arc,
          class LabelT = int32> // make the vector<LabelT> things actually vector<int32> for
                              // easier compatibility with Kaldi code.
class ContextFst : public Fst<Arc> {
 public:
  friend class ArcIterator< ContextFst<Arc> >;
  friend class StateIterator< ContextFst<Arc> >;
  // We have to supply the default template argument below to work around a
  // Visual Studio bug.
  friend class CacheArcIterator< ContextFst<Arc>,
                                 DefaultCacheStateAllocator<CacheState<Arc> >  >;

  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef CacheState<Arc> State;


  /// See \ref graph_context for more details.
  ContextFst(Label subsequential_symbol,  // epsilon not allowed.
             const vector<LabelT>& phones,  // symbols on output side of fst.
             const vector<LabelT>& disambig_syms,  // symbols on output side of fst.
             int32 N,  // Size of context window
             int32 P):  // Pos of "central" phone in ctx window, from 0..N-1.
      impl_ (new ContextFstImpl<Arc, LabelT>(subsequential_symbol, phones, disambig_syms, N, P))
  { assert(std::numeric_limits<LabelT>::is_signed); }

  ContextFst(const ContextFst<Arc, LabelT> &fst, bool reset = false);

  virtual ~ContextFst() { if (!impl_->DecrRefCount()) delete impl_;  }

  virtual StateId Start() const { return impl_->Start(); }

  virtual Weight Final(StateId s) const { return impl_->Final(s); }

  StateId NumStates() const { return impl_->NumStates(); }

  // This function is used in ContextMatcher.
  // Semantically this is not really const, as it causes states to be
  // added to the state table in impl_, and the input vocabulary to be
  // expanded, but C++ lets us make this const, and compose.h
  // requires it (because it provides const fst's to the Matcher
  // object.
  bool CreateArc(StateId s, Label olabel, Arc *oarc) const {
    return impl_->CreateArc(s, olabel, oarc);
  }

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

  // Careful: the output of ILabelInfo depends on what has been visited.
  const vector<vector<LabelT> > &ILabelInfo() { return impl_->ILabelInfo(); }

  virtual const string& Type() const { return impl_->Type(); }

  virtual ContextFst<Arc>  *Copy(bool reset = false) const {
    return new ContextFst<Arc>(*this, reset);
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

  friend class CacheStateIterator<ContextFst<Arc> >;  // so it can see impl_.
 private:
  ContextFstImpl<Arc, LabelT> *impl_;  // protected so CacheStateIterator
  // Makes visible to friends.
  ContextFstImpl<Arc, LabelT> *GetImpl() const { return impl_; }
 // would be: ImplToFst<ContextFstImpl<Arc, LabelT> >::GetImpl(); 
 // but need to convert to using the ImplToFst stuff.

  void operator = (const ContextFstImpl<Arc> &fst);  // disallow
};

/// Useful utility function for writing these vectors to disk.
/// writes as int32 for binary compatibility since I will typically
/// be "int".
template<class I>
void WriteILabelInfo(std::ostream &os, bool binary,
                     const vector<vector<I> > &info);

/// Useful utility function for reading these vectors from disk.
/// writes as int32 (see WriteILabelInfo above).
template<class I>
void ReadILabelInfo(std::istream &is, bool binary,
                    vector<vector<I> > *info);


/// The following function is mainly of use for printing and debugging.
template<class I>
SymbolTable *CreateILabelInfoSymbolTable(const vector<vector<I> > &info,
                                         const SymbolTable &phones_symtab,
                                         std::string separator,
                                         std::string disambig_prefix);  // e.g. separator = "/", disambig_prefix = "#"



// Specialization for ContextFst, of StateIterator.
// Just directs to use the one from CacheFst.
template<class A>
class StateIterator< ContextFst<A> >
    : public CacheStateIterator< ContextFst<A> > {
 public:
  explicit StateIterator(const ContextFst<A> &fst)
    : CacheStateIterator< ContextFst<A> >(fst, fst.GetImpl()) {}
};


// Specialization for ContextFst, of ArcIterator.
// Just directs to use the one from CacheFst.
template <class A>
class ArcIterator< ContextFst<A> >
    : public CacheArcIterator< ContextFst<A> > {
 public:
  typedef typename A::StateId StateId;

  ArcIterator(const ContextFst<A> &fst, StateId s)
    : CacheArcIterator< ContextFst<A> >(fst.GetImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) // arcs not already computed.
      fst.GetImpl()->Expand(s);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(ArcIterator);
};

template <class A, class I> inline
void ContextFst<A, I>::InitStateIterator(StateIteratorData<A> *data) const {
  data->base = new StateIterator< ContextFst<A> >(*this);
}



// ContextMatcher is a matcher type that is specialized to compose a ContextFst
// on the left, with an arbitrary FST on the right.  It does so by, rather than
// using arc iterators (which would force a call to Expand in ContextFstImpl, which
// would expand all the states), uses the CreateArc function of ContextFst.  This
// function is specific to the ContextFst type.  ContextMatcher queries the
// type of the FST using FstType(), and verifies that the left hand FST is a context
// FST, and then uses a static cast to ContextFst.  [We can't make it a template
// argument, as the template for ComposeFstOptions is only templated on a single FST
// class, and uses the same Matcher type for both the left and right].

template <class Arc, class LabelT>
class ContextMatcher : public MatcherBase<Arc> {  // CAREFUL: templated on arc, not on FST like normal Matcher.
 public:
  typedef Fst<Arc> FST;  // basic FST type that we get passed
  // because this is used in composition, typically one side will be a ContextFst,
  // and one will be some other type which we just treat as Fst<Arc>
  typedef ContextFst<Arc, LabelT> ContextF;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  // If input is a context FST, and match_type == MATCH_OUTPUT,
  // then we do matching by creating the arcs on-demand.
  // Otherwise we do not match.

  ContextMatcher(const FST &fst, MatchType match_type)
      : fst_(fst.Copy()),
        match_label_(kNoLabel),
        s_ (kNoStateId),
        ready_(false) {
    if (match_type == MATCH_OUTPUT && fst.Type() == (string)"context") {
      match_type_ = MATCH_OUTPUT;
    } else {
      match_type_ = MATCH_NONE;
    }
  }

  ContextMatcher(const ContextMatcher<Arc, LabelT> &matcher, bool safe)
      : fst_(matcher.fst_->Copy(safe)),
        match_type_(matcher.match_type_),
        match_label_(kNoLabel),
        s_ (kNoStateId),
        ready_(false) {}

  virtual ~ContextMatcher() {
    delete fst_;
  }

  virtual const FST &GetFst() const { return *fst_; }

  virtual ContextMatcher<Arc, LabelT> *Copy(bool safe = false) const {
    return new ContextMatcher<Arc, LabelT>(*this, safe);
  }

  virtual MatchType Type(bool test) const {
    return match_type_;
  }

  void SetState(StateId s) {
    if (match_type_ == MATCH_NONE)
      LOG(FATAL) << "ContextMatcher: bad match type";
    s_ = s;
  }

  bool Find(Label match_label);

  bool Done() const {
    return !ready_;
  }

  const Arc& Value() const {
    assert(ready_);
    return arc_;
  }

  void Next() {  // we only ever get one arc so just set ready_ to false.
    assert(ready_);
    ready_ = false;
  }


  virtual uint64 Properties(uint64 props) const { return props; } // simple matcher that does
  // not change its FST, so properties are properties of FST it is applied to

 private:
  virtual void SetState_(StateId s) { SetState(s); }
  virtual bool Find_(Label label) { return Find(label); }
  virtual bool Done_() const { return Done(); }
  virtual const Arc& Value_() const { return Value(); }
  virtual void Next_() { Next(); }

  const FST *fst_;
  MatchType match_type_;          // Type of match to perform
  Label match_label_;             // Current label to be matched
  StateId s_;                     // Current state.
  Arc arc_;                       // Current arc.
  bool ready_;                     // True if arc is waiting to be output.
  bool current_loop_;             // Current arc is the implicit loop

  void operator = (const SortedMatcher<FST> &);  // Disallow
};



/* This is a specialization of Compose, where the left argument is of
   type ContextFst.
   For clarity we distinguish it with a different name.
   It uses the special matcher which should be more efficient than
   a normal matcher.
   The fst ifst2 must have the subsequential loop (if not a left-context-only
   system)
*/
template<class Arc, class LabelT>
void ComposeContextFst(const ContextFst<Arc, LabelT> &ifst1, const Fst<Arc> &ifst2,
                       MutableFst<Arc> *ofst,
                       const ComposeOptions &opts = ComposeOptions()) {
  ComposeFstOptions<Arc, ContextMatcher<Arc, LabelT> > nopts;
  nopts.gc_limit = 0;  // Cache only the most recent state for fastest copy.
  *ofst = ComposeFst<Arc>(ifst1, ifst2, nopts);
  if (opts.connect)
    Connect(ofst);
}

/**
   Used in the command-line tool fstcomposecontext.  It creates a context FST and
   composes it on the left with "ifst" to make "ofst".  It outputs the label
   information to ilabels_out.  "ifst" is mutable because we need to add the
   subsequential loop.
 */
inline void ComposeContext(vector<int32> &disambig_syms,
                           int N, int P,
                           VectorFst<StdArc> *ifst,
                           VectorFst<StdArc> *ofst,
                           vector<vector<int32> > *ilabels_out);


/**
  Modifies an FST so that it transuces the same paths, but the input side of the
  paths can all have the subsequential symbol '$' appended to them any number of
  times (we could easily specify the number of times, but accepting any number of
  repetitions is just more convenient).  The actual way we do this is for each
  final state, we add a transition with weight equal to the final-weight of that
  state, with input-symbol '$' and output-symbols \<eps\>, and ending in a new
  super-final state that has unit final-probability and a unit-weight self-loop
  with '$' on its input and \<eps\> on its output.  The reason we don't just
  add a loop to each final-state has to do with preserving stochasticity
  (see \ref fst_algo_stochastic).  We keep the final-probability in all the
  original final-states rather than setting them to zero, so the resulting FST
  can accept zero '$' symbols at the end (in case we had no right context).
*/
template<class Arc>
void AddSubsequentialLoop(typename Arc::Label subseq_symbol,
                          MutableFst<Arc> *fst);

/// @}
}  // namespace fst

#include "context-fst-inl.h"

#endif

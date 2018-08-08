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

/* This header defines a context FST "C" (the "C" in "HCLG") which transduces
   from symbols representing phone context windows (e.g. "a, b, c") to
   individual phones, e.g. "a".  Search for "hbka.pdf" ("Speech Recognition
   with Weighted Finite State Transducers") by M. Mohri, for more context.
*/

#include <unordered_map>
using std::unordered_map;

#include <algorithm>
#include <string>
#include <vector>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "util/const-integer-set.h"
#include "fstext/deterministic-fst.h"

namespace fst {

/// \addtogroup context_fst_group "Classes and functions related to context expansion"
/// @{

namespace internal {

/*
   ContextFstImpl inherits from CacheImpl, which handles caching of states.
*/

template <class Arc,
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
  typedef DefaultCacheStore<Arc> Store;
  typedef typename Store::State State;
  typedef unordered_map<vector<LabelT>,
                        StateId, kaldi::VectorHasher<LabelT> > VectorToStateType;
  typedef unordered_map<vector<LabelT>,
                        Label, kaldi::VectorHasher<LabelT> > VectorToLabelType;

  typedef typename VectorToStateType::const_iterator VectorToStateIter;
  typedef typename VectorToLabelType::const_iterator VectorToLabelIter;

  ContextFstImpl(Label subsequential_symbol,  // epsilon not allowed.
                 const vector<LabelT> &phones,
                 const vector<LabelT> &disambig_syms,
                 int32 context_width,  // size of ctx window
                 int32 central_position);

  ContextFstImpl(const ContextFstImpl &other);

  ~ContextFstImpl() { }

  // See \ref tree_ilabel
  // "http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel" for more
  // information about the ilabel_info.
  const vector<vector<LabelT> > &ILabelInfo() const { return ilabel_info_; }

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
  // an ambiguity if a disambiguation symbol appears at the start of a sequence on the
  // input of CLG, whether it was at the very start of the input of LG, or just after, say,
  // the first real phone.  What we do if we need pseudo_eps_symbol_ to be not epsilon,
  // we create a special symbol with symbol-id 1 and sequence representation (ilabels entry)
  // [ 0 ] .  In the printed form we call this #-1.
};

}  // namespace internal

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
class ContextFst : public ImplToFst<internal::ContextFstImpl<Arc, LabelT>> {
 public:
  friend class ArcIterator<ContextFst<Arc>>;
  friend class StateIterator<ContextFst<Arc>>;

  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef DefaultCacheStore<Arc> Store;
  typedef typename Store::State State;
  typedef internal::ContextFstImpl<Arc, LabelT> Impl;

  /// See \ref graph_context for more details.
  ContextFst(Label subsequential_symbol,  // epsilon not allowed.
             const vector<LabelT>& phones,  // symbols on output side of fst.
             const vector<LabelT>& disambig_syms,  // symbols on output side of fst.
             int32 N,  // Size of context window
             int32 P)  // Pos of "central" phone in ctx window, from 0..N-1.
      : ImplToFst<Impl>(std::make_shared<Impl>(
            subsequential_symbol, phones, disambig_syms, N, P)) {
    assert(std::numeric_limits<LabelT>::is_signed);
  }

  ContextFst(const ContextFst<Arc, LabelT> &fst, bool safe = false)
      : ImplToFst<Impl>(fst, safe) {}

  ContextFst<Arc, LabelT> *Copy(bool safe = false) const override {
    return new ContextFst<Arc, LabelT>(*this, safe);
  }

  inline void InitStateIterator(StateIteratorData<Arc> *data) const override;

  void InitArcIterator(StateId s, ArcIteratorData<Arc> *data) const override {
    GetMutableImpl()->InitArcIterator(s, data);
  }

  // This function is used in ContextMatcher.
  // Semantically this is not really const, as it causes states to be
  // added to the state table in impl_, and the input vocabulary to be
  // expanded, but C++ lets us make this const, and compose.h
  // requires it (because it provides const fst's to the Matcher object.
  bool CreateArc(StateId s, Label olabel, Arc *oarc) const {
    return GetMutableImpl()->CreateArc(s, olabel, oarc);
  }

  // Careful: the output of ILabelInfo depends on what has been visited.
  const vector<vector<LabelT> > &ILabelInfo() const {
    return GetImpl()->ILabelInfo();
  }

 private:
  using ImplToFst<Impl>::GetImpl;
  using ImplToFst<Impl>::GetMutableImpl;

  ContextFst &operator=(const ContextFst &fst) = delete;
};




/// Useful utility function for writing these vectors to disk.
/// writes as int32 for binary compatibility since it will typically be "int".
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
    : CacheStateIterator< ContextFst<A> >(fst, fst.GetMutableImpl()) {}
};


// Specialization for ContextFst, of ArcIterator.
// Just directs to use the one from CacheFst.
template <class A>
class ArcIterator< ContextFst<A> >
    : public CacheArcIterator< ContextFst<A> > {
 public:
  typedef typename A::StateId StateId;

  ArcIterator(const ContextFst<A> &fst, StateId s)
    : CacheArcIterator< ContextFst<A> >(fst.GetMutableImpl(), s) {
    if (!fst.GetImpl()->HasArcs(s)) // arcs not already computed.
      fst.GetMutableImpl()->Expand(s);
  }
};

template <class Arc, class I> inline
void ContextFst<Arc, I>::InitStateIterator(StateIteratorData<Arc> *data) const {
  data->base = new StateIterator< ContextFst<Arc> >(*this);
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
inline void ComposeContext(const vector<int32> &disambig_syms,
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


/*
   InverseContextFst represents the inverse of the context FST "C" (the "C" in
   "HCLG") which transduces from symbols representing phone context windows
   (e.g. "a, b, c") to individual phones, e.g. "a".  So InverseContextFst
   transduces from phones to symbols representing phone context windows.  The
   point is that the inverse is deterministic, so the DeterministicOnDemandFst
   interface is applicable, which turns out to be a convenient way to implement
   this.

   This doesn't implement the full Fst interface, it implements the
   DeterministicOnDemandFst interface which is much simpler and which is
   sufficient for what we need to do with this.

   Search for "hbka.pdf" ("Speech Recognition with Weighted Finite State
   Transducers") by M. Mohri, for more context.
*/

class InverseContextFst: public DeterministicOnDemandFst<StdArc> {
public:
  typedef StdArc Arc;
  typedef typename StdArc::StateId StateId;
  typedef typename StdArc::Weight Weight;
  typedef typename StdArc::Label Label;

  /// See \ref graph_context for more details.
  InverseContextFst(Label subsequential_symbol,  // epsilon not allowed.
                    const vector<int32>& phones,  // symbols on output side of fst.
                    const vector<int32>& disambig_syms,  // symbols on output side of fst.
                    int32 N,  // Size of context window
                    int32 P);  // Pos of "central" phone in ctx window, from 0..N-1.


  virtual StateId Start() { return 0; }

  virtual Weight Final(StateId s);

  /// Note: ilabel must not be epsilon.
  virtual bool GetArc(StateId s, Label ilabel, Arc *arc);

  ~InverseContextFst() { }

private:

  /// Returns the state-id corresponding to this vector of phones; creates the
  /// state it if necessary.  Requires seq.size() == context_width_ - 1.
  StateId FindState(const vector<int32> &seq);

  /// Finds the label index corresponding to this context-window of phones
  /// (likely of width context_width_).  Inserts it into the
  /// ilabel_info_/ilabel_map_ tables if necessary.
  Label FindLabel(const vector<int32> &label_info);

  inline bool IsDisambigSymbol(Label lab) { return (disambig_syms_.count(lab) != 0); }

  inline bool IsPhoneSymbol(Label lab) { return (phone_syms_.count(lab) != 0); }

  /// Create disambiguation-symbol self-loop arc; where 'ilabel' must correspond to
  /// a disambiguation symbol.  Called from CreateArc().
  inline void CreateDisambigArc(StateId s, Label ilabel, Arc *arc);

  /// Creates an arc, this function is to be called only when 'ilabel'
  /// corresponds to a phone.  Called from CreateArc().  The olabel may end be
  /// epsilon, instead of a phone-in-context, if the system has right context
  /// and we are very near the beginning of the phone sequence.
  inline void CreatePhoneOrEpsArc(StateId src, StateId dst, Label ilabel,
                                  const vector<int32> &phone_seq, Arc *arc);


  /// If phone_seq is nonempty then this function it left by one and appends
  /// 'label' to it, otherwise it does nothing.  We expect (but do not check)
  /// that phone_seq->size() == context_width_ - 1.
  inline void ShiftSequenceLeft(Label label, std::vector<int32> *phone_seq);

  /// This utility function does something equivalent to the following 3 steps:
  ///   *full_phone_sequence =  seq;
  ///  full_phone_sequence->append(label)
  ///  Replace any values equal to 'subsequential_symbol_' in
  /// full_phone_sequence with zero (this is to avoid having to keep track of
  /// the value of 'subsequential_symbol_' outside of this program).
  /// This function assumes that seq.size() == context_width_ - 1, and also that
  /// 'subsequential_symbol_' does not appear in positions 0 through
  /// central_position_ of 'seq'.
  inline void GetFullPhoneSequence(const std::vector<int32> &seq, Label label,
                                   std::vector<int32> *full_phone_sequence);

  // Map type to map from vectors of int32 (representing phonetic contexts,
  // which will be of dimension context_width - 1) to StateId (corresponding to
  // the state index in this FST).
  typedef unordered_map<vector<int32>, StateId,
                        kaldi::VectorHasher<int32> > VectorToStateMap;

  // Map type to map from vectors of int32 (representing ilabel-info,
  // see http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel) to
  // Label (the output label in this FST).
  typedef unordered_map<vector<int32>, Label,
                        kaldi::VectorHasher<int32> > VectorToLabelMap;


  // Sometimes called N, context_width_ this is the width of the
  // phonetic context, e.g. 3 for triphone, 2 for biphone, one for monophone.
  // It is a user-specified value.
  int32 context_width_;

  // Sometimes called P, central_position_ is is the (zero-based) "central
  // position" in the context window, meaning the phone that is "in" a certain
  // context.  The most widely used values of (context-width, central-position)
  // are: (3,1) for triphone, (1,0) for monophone, and (2, 1) for left biphone.
  // This is also specified by the user.  As an example, in the left-biphone
  // [ 5, 6 ], we view it as "the phone numbered 6 with the phone numbered 5 as
  // its left-context".
  int32 central_position_;

  // The following three variables were also passed in by the caller:

  // 'phone_syms_' are a set of phone-ids, typically 1, 2, .. num_phones.
  kaldi::ConstIntegerSet<Label> phone_syms_;

  // disambig_syms_ is the set of integer ids of the disambiguation symbols,
  // usually represented in text form as #0, #1, #2, etc.  These are inserted
  // into the grammar (for #0) and the lexicon (for #1, #2, ...) in order to
  // make the composed FSTs determinizable.  They are treated "specially" by the
  // context FST in that they are not part of the context, they are just "passed
  // through" via self-loops.  See the Mohri chapter mrentioned above for more
  // information.
  kaldi::ConstIntegerSet<Label> disambig_syms_;

  // subsequential_symbol_, represented as "$" in the Mohri chapter mentioned
  // above, is something which terminates phonetic sequences to force out the
  // last phones-in-context.  In our implementation it's added to det(LG) as a
  // self-loop on final states before composing with C.
  // (c.f. AddSubsequentialLoop()).
  Label subsequential_symbol_;


  // pseudo_eps_symbol_, which in printed form we refer to as "#-1", is a symbol that
  // appears on the ilabels of the context transducer C, i.e. the olabels of this
  // FST which is C's inverse.  It is a symbol we introduce to solve a special problem
  // in systems with right-context (context_width_ > central_position_ + 1) that
  // use disambiguation symbols.  It exists to prevent CLG from being nondeterminizable.
  //
  // The issue is that, in this case, the disambiguation symbols are shifted
  // left w.r.t. the phones, and there becomes an ambiguity, if a disambiguation
  // symbol appears at the start of a sequence on the input of CLG, about
  // whether it was at the very start of the input of LG, or just after, say,
  // the first real phone.  This can lead to determinization failure under
  // certain circumstances.  What we do if we need pseudo_eps_symbol_ to be not
  // epsilon, we create a special symbol with symbol-id 1 and sequence
  // representation (ilabels entry) [ 0 ] .
  int32 pseudo_eps_symbol_;

  // maps from vector<int32>, representing phonetic contexts of length
  // context_width-1, to StateId.  (The states of the "C" fst correspond to
  // phonetic contexts, but we only create them as and when they are needed).
  VectorToStateMap state_map_;

  // The inverse of 'state_map_': gives us the phonetic context corresponding to
  // each state-id.
  vector<vector<int32> > state_seqs_;



};


/// @}
}  // namespace fst

#include "context-fst-inl.h"

#endif  // KALDI_FSTEXT_CONTEXT_FST_H_

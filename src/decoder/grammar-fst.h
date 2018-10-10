// decoder/grammar-fst.h

// Copyright    2018  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_GRAMMAR_FST_H_
#define KALDI_DECODER_GRAMMAR_FST_H_

/**
   For an extended explanation of the framework of which grammar-fsts are a
   part, please see \ref grammar (i.e. ../doc/grammar.dox).

   This header implements a special FST type which we use in that framework;
   it is a lightweight wrapper which stitches together several FSTs and makes
   them look, to the decoder code, like a single FST.  It is a bit like
   OpenFst's Replace() function, but with support for left-biphone context.
 */



#include "fst/fstlib.h"
#include "fstext/grammar-context-fst.h"

namespace fst {


// GrammarFstArc is an FST Arc type which differs from the normal StdArc type by
// having the state-id be 64 bits, enough to store two indexes: the higher 32
// bits for the FST-instance index, and the lower 32 bits for the state within
// that FST-instance.
// Obviously this leads to very high-numbered state indexes, which might be
// a problem in some circumstances, but the decoder code doesn't store arrays
// indexed by state, it uses hashes, so this isn't a problem.
struct GrammarFstArc {
  typedef fst::TropicalWeight Weight;
  typedef int Label;  // OpenFst's StdArc uses int; this is for compatibility.
  typedef int64 StateId;

  Label ilabel;
  Label olabel;
  Weight weight;
  StateId nextstate;

  GrammarFstArc() {}

  GrammarFstArc(Label ilabel, Label olabel, Weight weight, StateId nextstate)
      : ilabel(ilabel),
        olabel(olabel),
        weight(std::move(weight)),
        nextstate(nextstate) {}
};

#define KALDI_GRAMMAR_FST_SPECIAL_WEIGHT 4096.0

class GrammarFst;

// Declare that we'll be overriding class ArcIterator for class GrammarFst.
// This wouldn't work if we were fully using the OpenFst framework,
// e.g. if we had GrammarFst inherit from class Fst.
template<> class ArcIterator<GrammarFst>;


/**
   GrammarFst is an FST that is 'stitched together' from multiple FSTs, that can
   recursively incorporate each other.  (This is limited to left-biphone
   phonetic context). This class does not inherit from fst::Fst and does not
   support its full interface-- only the parts that are necessary for the
   decoder to work when templated on it.

   The basic interface is inspired by OpenFst's 'ReplaceFst' (see its
   replace.h), except that this handles left-biphone phonetic context, which
   requires, essentially, having multiple exit-points and entry-points for
   sub-FSTs that represent nonterminals in the grammar; and multiple return
   points whenever we invoke a nonterminal.  For more information
   see \ref grammar (i.e. ../doc/grammar.dox).

   Caution: this class is not thread safe, i.e. you shouldn't access the same
   GrammarFst from multiple threads.  We can fix this later if needed.
 */
class GrammarFst {
 public:
  typedef GrammarFstArc Arc;
  typedef TropicalWeight Weight;

  // StateId is actually int64.  The high-order 32 bits are interpreted as an
  // instance_id, i.e. and index into the instances_ vector; the low-order 32
  // bits are the state index in the FST instance.
  typedef Arc::StateId StateId;

  // The StateId of the individual FST instances (int, currently).
  typedef StdArc::StateId BaseStateId;

  typedef Arc::Label Label;


  /**
     Constructor.  This constructor is very lightweight; the only immediate work
     it does is to iterate over the arcs in the start states of the provided
     FSTs in order to set up the appropriate entry points.

     For simplicity (to avoid templates), we limit the input FSTs to be of type
     ConstFst<StdArc>; this limitation could be removed later if needed.  You
     can always construct a ConstFst<StdArc> if you have another StdArc-based
     FST type.  If the FST was read from disk, it may already be of type
     ConstFst, and dynamic_cast might be sufficient to convert the type.

     @param [in] nonterm_phones_offset   The integer id of the symbol
             "#nonterm_bos" in phones.txt.
     @param [in] top_fst    The top-level FST of the grammar, which will
              usually invoke the fsts in 'ifsts'.  The fsts in 'ifsts' may
              also invoke each other recursively.  Even left-recursion is
              allowed, although if it is with zero cost, it may blow up when you
              decode.  When an FST invokes another, the invocation point will
              have sequences of two special symbols which would be decoded as:
                  (#nonterm:foo,p1) (#nonterm_reenter,p2)
              where p1 and p2 (which may be real phones or #nonterm_bos)
              represent the phonetic left-context that we enter, and leave, the
              sub-graph with respectively.
     @param [in] ifsts   ifsts is a list of pairs (nonterminal-symbol,
              the HCLG.fst corresponding to that symbol).  The nonterminal
              symbols must be among the user-specified nonterminals in
              phones.txt, i.e. the things with names like "#nonterm:foo" and
              "#nonterm:bar" in phones.txt.  Also no nonterminal may appear more
              than once in 'fsts'.  ifsts may be empty, even though that doesn't
              make much sense.  This function does not take ownership of
              these pointers (i.e. it will not delete them when it is destroyed).
    */
  GrammarFst(
      int32 nonterm_phones_offset,
      const ConstFst<StdArc> &top_fst,
      const std::vector<std::pair<int32, const ConstFst<StdArc> *> > &ifsts);

  ///  This constructor should only be used prior to calling Read().
  GrammarFst(): top_fst_(NULL) { }

  // This Write function allows you to dump a GrammarFst to disk as a single
  // object.  It only supports binary mode, but the option is allowed for
  // compatibility with other Kaldi read/write functions (it will crash if
  // binary == false).
  void Write(std::ostream &os, bool binary) const;

  // Reads the format that Write() outputs.  Will crash if binary == false.
  void Read(std::istream &os, bool binary);

  StateId Start() const {
    // the top 32 bits of the 64-bit state-id will be zero, because the
    // top FST instance has instance-id = 0.
    return static_cast<StateId>(top_fst_->Start());
  }

  Weight Final(StateId s) const {
    // If the fst-id (top 32 bits of s) is nonzero, this state is not final,
    // because we need to return to the top-level FST before we can be final.
    if (s != static_cast<StateId>(static_cast<int32>(s))) {
      return Weight::Zero();
    } else {
      BaseStateId base_state = static_cast<BaseStateId>(s);
      Weight ans = top_fst_->Final(base_state);
      if (ans.Value() == KALDI_GRAMMAR_FST_SPECIAL_WEIGHT) {
        return Weight::Zero();
      } else {
        return ans;
      }
    }
  }

  // This is called in LatticeFasterDecoder.  As an implementation shortcut, if
  // the state is an expanded state, we return 1, meaning 'yes, there are input
  // epsilons'; the calling code doesn't actually care about the exact number.
  inline size_t NumInputEpsilons(StateId s) const {
    // Compare with the constructor of ArcIterator.
    int32 instance_id = s >> 32;
    BaseStateId base_state = static_cast<int32>(s);
    const GrammarFst::FstInstance &instance = instances_[instance_id];
    const ConstFst<StdArc> *base_fst = instance.fst;
    if (base_fst->Final(base_state).Value() != KALDI_GRAMMAR_FST_SPECIAL_WEIGHT) {
      return base_fst->NumInputEpsilons(base_state);
    } else {
      return 1;
    }
  }

  inline std::string Type() const { return "grammar"; }

  ~GrammarFst();
 private:

  struct ExpandedState;

  friend class ArcIterator<GrammarFst>;

  // sets up nonterminal_map_.
  void InitNonterminalMap();

  // sets up entry_arcs_[i].  We do this only on demand, as each one is
  // accessed, so that if there are a lot of nonterminals, this object doesn't
  // too much work when it is initialized.  Each call to this function only
  // takes time O(number of left-context phones), which is quite small, but we'd
  // like to avoid that if possible.
  void InitEntryArcs(int32 i);

  // sets up instances_ with the top-level instance.
  void InitInstances();

  // Does the initialization tasks after nonterm_phones_offset_,
  // top_fsts_ and ifsts_ have been set up
  void Init();

  // clears everything.
  void Destroy();

  /*
    This utility function sets up a map from "left-context phone", meaning
    either a phone index or the index of the symbol #nonterm_bos, to
    an arc-index leaving a particular state in an FST (i.e. an index
    that we could use to Seek() to the matching arc).

      @param [in]  fst  The FST we are looking for state-indexes for
      @param [in]  entry_state  The state in the FST-- must have arcs with
                 ilabels decodable as (nonterminal_symbol, left_context_phone).
                 Will either be the start state (if 'nonterminal_symbol'
                 corresponds to #nonterm_begin), or an internal state
                 (if 'nonterminal_symbol' corresponds to #nonterm_reenter).
                 The arc-indexes of those arcs will be the values
                 we set in 'phone_to_arc'
      @param [in]  nonterminal_symbol  The index in phones.txt of the
                 nonterminal symbol we expect to be encoded in the ilabels
                 of the arcs leaving 'entry_state'.  Will either correspond
                 to #nonterm_begin or #nonterm_reenter.
      @param [out] phone_to_arc  We output the map from left_context_phone
                 to the arc-index (i.e. the index we'd have to Seek() to
                 in an arc-iterator set up for the state 'entry_state).
   */
  void InitEntryOrReentryArcs(
      const ConstFst<StdArc> &fst,
      int32 entry_state,
      int32 nonterminal_symbol,
      std::unordered_map<int32, int32> *phone_to_arc);


  inline int32 GetPhoneSymbolFor(enum NonterminalValues n) {
    return nonterm_phones_offset_ + static_cast<int32>(n);
  }
  /**
     Decodes an ilabel into a pair (nonterminal, left_context_phone).  Crashes
     if something went wrong or ilabel did not represent that (e.g. was less
     than kNontermBigNumber).

       @param [in] the ilabel to be decoded.  Note: the type 'Label' will in practice be int.
       @param [out] The nonterminal part of the ilabel after decoding.
                   Will be a value greater than nonterm_phones_offset_.
       @param [out] The left-context-phone part of the ilabel after decoding.
                    Will either be a phone index, or the symbol corresponding
                    to #nonterm_bos (meaning no left-context as we are at
                    the beginning of the sequence).
   */
  void DecodeSymbol(Label label,
                    int32 *nonterminal_symbol,
                    int32 *left_context_phone);


  // This function creates and returns an ExpandedState corresponding to a
  // particular state-id in the FstInstance for this instance_id.  It is called
  // when we have determined that an ExpandedState needs to be created and that
  // it is not currently present.  It creates and returns it; the calling code
  // needs to add it to the expanded_states map for its FST instance.
  ExpandedState *ExpandState(int32 instance_id, BaseStateId state_id);

  // Called from ExpandState() when the nonterminal type on the arcs is
  // #nonterm_end, this implements ExpandState() for that case.
  ExpandedState *ExpandStateEnd(int32 instance_id, BaseStateId state_id);

  // Called from ExpandState() when the nonterminal type on the arcs is a
  // user-defined nonterminal, this implements ExpandState() for that case.
  ExpandedState *ExpandStateUserDefined(int32 instance_id, BaseStateId state_id);

  // Called from ExpandStateUserDefined(), this function attempts to look up the
  // pair (nonterminal, state) in the map
  // instances_[instance_id].child_instances.  If it exists (because this
  // return-state has been expanded before), it returns the value it found;
  // otherwise it creates the child-instance and returns its newly created
  // instance-id.
  inline int32 GetChildInstanceId(int32 instance_id, int32 nonterminal,
                                  int32 state);

  /**
    Called while expanding states, this function combines information from two
    arcs: one leaving one sub-fst and one arriving in another sub-fst.

      @param [in] leaving_arc  The arc leaving the first FST; must have
                     zero olabel.  The ilabel will have a nonterminal symbol
                     like #nonterm:foo or #nonterm_end on it, encoded with a
                     phonetic context, but we ignore the ilabel.
      @param [in] arriving_arc  The arc arriving in the second FST.
                    It will have an ilabel consisted of either #nonterm_begin
                    or #nonterm_enter combined with a left-context phone,
                    but we ignore the ilabel.
      @param [in] cost_correction  A correction term that we add to the
                    cost of the arcs.  This basically cancels out the
                    "1/num_options" part of the weight that we added in L.fst
                    when we put in all the phonetic-context options.  We
                    did that to keep the FST stochastic, so that if we ever
                    pushed the weights, it wouldn't lead to weird effects.
                    This takes out that correction term... things will
                    still sum to one in the appropriate way, because in fact
                    when we cross these FST boundaries we only take one
                    specific phonetic context, rather than all possibilities.
      @param [out] arc  The arc that we output.  Will have:
                   - weight equal to the product of the input arcs' weights,
                      times a weight constructed from 'cost_correction'.
                   - olabel equal to arriving_arc.olabel (leaving_arc's olabel
                     will be zero).
                   - ilabel equal to zero (we discard both ilabels, they are
                     not transition-ids but special symbols).
                   - nextstate equal to the nextstate of arriving_arc.
  */
  static inline void CombineArcs(const StdArc &leaving_arc,
                                 const StdArc &arriving_arc,
                                 float cost_correction,
                                 StdArc *arc);

  /** Called from the ArcIterator constructor when we encounter an FST state with
      nonzero final-prob, this function first looks up this state_id in
      'expanded_states' member of the corresponding FstInstance, and returns it
      if already present; otherwise it populates the 'expanded_states' map with
      something for this state_id and returns the value.
  */
  inline ExpandedState *GetExpandedState(int32 instance_id,
                                         BaseStateId state_id) {
    std::unordered_map<BaseStateId, ExpandedState*> &expanded_states =
        instances_[instance_id].expanded_states;

    std::unordered_map<BaseStateId, ExpandedState*>::iterator iter =
        expanded_states.find(state_id);
    if (iter != expanded_states.end()) {
      return iter->second;
    } else {
      ExpandedState *ans = ExpandState(instance_id, state_id);
      // Don't use the reference 'expanded_states'; it could have been
      // invalidated.
      instances_[instance_id].expanded_states[state_id] = ans;
      return ans;
    }
  }

  /**
     Represents an expanded state in an FstInstance.  We expand states whenever
     we encounter states with a final-cost equal to
     KALDI_GRAMMAR_FST_SPECIAL_WEIGHT (4096.0).  The function
     PrepareGrammarFst() makes sure to add this special final-cost on states
     that have special arcs leaving them. */
  struct ExpandedState {
    // The final-prob for expanded states is always zero; to avoid
    // corner cases, we ensure this via adding epsilon arcs where
    // needed.

    // fst-instance index of destination state (we will have ensured previously
    // that this is the same for all outgoing arcs).
    int32 dest_fst_instance;

    // List of arcs out of this state, where the 'nextstate' element will be the
    // lower-order 32 bits of the destination state and the higher order bits
    // will be given by 'dest_fst_instance'.  We do it this way, instead of
    // constructing a vector<Arc>, in order to simplify the ArcIterator code and
    // avoid unnecessary branches in loops over arcs.
    // We guarantee that this 'arcs' array will always be nonempty; this
    // is to avoid certain hassles on Windows with automated bounds-checking.
    std::vector<StdArc> arcs;
  };


  // An FstInstance is a copy of an FST.  The instance numbered zero is for
  // top_fst_, and (to state it approximately) whenever any FST instance invokes
  // another FST a new instance will be generated on demand.
  struct FstInstance {
    // ifst_index is the index into the ifsts_ vector that corresponds to this
    // FST instance, or -1 if this is the top-level instance.
    int32 ifst_index;

    // Pointer to the FST corresponding to this instance: it will equal top_fst_
    // if ifst_index == -1, or ifsts_[ifst_index].second otherwise.
    const ConstFst<StdArc> *fst;

    // 'expanded_states', which will be populated on demand as states in this
    // FST instance are accessed, will only contain entries for states in this
    // FST that the final-prob's value equal to
    // KALDI_GRAMMAR_FST_SPECIAL_WEIGHT.  (That final-prob value is used as a
    // kind of signal to this code that the state needs expansion).
    std::unordered_map<BaseStateId, ExpandedState*> expanded_states;

    // 'child_instances', which is populated on demand as states in this FST
    // instance are accessed, is logically a map from pair (nonterminal_index,
    // return_state) to instance_id.  When we encounter an arc in our FST with a
    // user-defined nonterminal indexed 'nonterminal_index' on its ilabel, and
    // with 'return_state' as its nextstate, we look up that pair
    // (nonterminal_index, return_state) in this map to see whether there already
    // exists an FST instance for that.  If it exists then the transition goes to
    // that FST instance; if not, then we create a new one.  The 'return_state'
    // that's part of the key in this map would be the same as the 'parent_state'
    // in that child FST instance, and of course the 'parent_instance' in
    // that child FST instance would be the instance_id of this instance.
    //
    // In most cases each return_state would only have a single
    // nonterminal_index, making the 'nonterminal_index' in the key *usually*
    // redundant, but in principle it could happen that two user-defined
    // nonterminals might share the same return-state.
    std::unordered_map<int64, int32> child_instances;

    // The instance-id of the FST we return to when we are done with this one
    // (or -1 if this is the top-level FstInstance so there is nowhere to
    // return).
    int32 parent_instance;

    // The state in the FST of 'parent_instance' at which we expanded this FST
    // instance, and to which we return (actually we return to the next-states
    // of arcs out of 'parent_state').
    int32 parent_state;

    // 'parent_reentry_arcs' is a map from left-context-phone (i.e. either a
    // phone index or #nonterm_bos), to an arc-index, which we could use to
    // Seek() in an arc-iterator for state parent_state in the FST-instance
    // 'parent_instance'.  It's set up when we create this FST instance.  (The
    // arcs used to enter this instance are not located here, they can be
    // located in entry_arcs_[instance_id]).  We make use of reentry_arcs when
    // we expand states in this FST that have #nonterm_end on their arcs,
    // leading to final-states, which signal a return to the parent
    // FST-instance.
    std::unordered_map<int32, int32> parent_reentry_arcs;
  };

  // The integer id of the symbol #nonterm_bos in phones.txt.
  int32 nonterm_phones_offset_;

  // The top-level FST passed in by the user; contains the start state and
  // final-states, and may invoke FSTs in 'ifsts_' (which can also invoke
  // each other recursively).
  const ConstFst<StdArc> *top_fst_;

  // A list of pairs (nonterm, fst), where 'nonterm' is a user-defined
  // nonterminal symbol as numbered in phones.txt (e.g. #nonterm:foo), and
  // 'fst' is the corresponding FST.
  std::vector<std::pair<int32, const ConstFst<StdArc> *> > ifsts_;

  // Maps from the user-defined nonterminals like #nonterm:foo as numbered
  // in phones.txt, to the corresponding index into 'ifsts_', i.e. the ifst_index.
  std::unordered_map<int32, int32> nonterminal_map_;

  // entry_arcs_ will have the same dimension as ifsts_.  Each entry_arcs_[i]
  // is a map from left-context phone (i.e. either a phone-index or
  // #nonterm_bos) to the corresponding arc-index leaving the start-state in
  // the FST 'ifsts_[i].second'.
  // We populate this only on demand as each one is needed (except for the
  // first one, which we populate immediately as a kind of sanity check).
  // Doing it on-demand prevents this object's initialization from being
  // nontrivial in the case where there are a lot of nonterminals.
  std::vector<std::unordered_map<int32, int32> > entry_arcs_;

  // The FST instances.  Initially it is a vector with just one element
  // representing top_fst_, and it will be populated with more elements on
  // demand.  An instance_id refers to an index into this vector.
  std::vector<FstInstance> instances_;

  // A list of FSTs that are to be deleted when this object is destroyed.  This
  // will only be nonempty if we have read this object from the disk using
  // Read().
  std::vector<const ConstFst<StdArc> *> fsts_to_delete_;
};


/**
   This is the overridden template for class ArcIterator for GrammarFst.  This
   is only used in the decoder, and the GrammarFst is not a "real" FST (it just
   has a very similar-looking interface), so we don't need to implement all the
   functionality that the regular ArcIterator has.
 */
template <>
class ArcIterator<GrammarFst> {
 public:
  using Arc = typename GrammarFst::Arc;
  using BaseArc = StdArc;
  using StateId = typename Arc::StateId;  // int64
  using BaseStateId = typename StdArc::StateId;  // int
  using ExpandedState = GrammarFst::ExpandedState;

  // Caution: uses const_cast to evade const rules on GrammarFst.  This is for
  // compatibility with how things work in OpenFst.
  inline ArcIterator(const GrammarFst &fst_in, StateId s) {
    GrammarFst &fst = const_cast<GrammarFst&>(fst_in);
    // 'instance_id' is the high order bits of the state.
    int32 instance_id = s >> 32;
    // 'base_state' is low order bits of the state.  It's important to
    // explicitly say int32 below, not BaseStateId == int, which might on some
    // compilers be a 64-bit type.
    BaseStateId base_state = static_cast<int32>(s);
    const GrammarFst::FstInstance &instance = fst.instances_[instance_id];
    const ConstFst<StdArc> *base_fst = instance.fst;
    if (base_fst->Final(base_state).Value() != KALDI_GRAMMAR_FST_SPECIAL_WEIGHT) {
      // A normal state
      dest_instance_ = instance_id;
      base_fst->InitArcIterator(s, &data_);
      i_ = 0;
    } else {
      // A special state
      ExpandedState *expanded_state = fst.GetExpandedState(instance_id,
                                                           base_state);
      dest_instance_ = expanded_state->dest_fst_instance;
      // it's ok to leave the other members of data_ uninitialized, as they will
      // never be interrogated.
      data_.arcs = &(expanded_state->arcs[0]);
      data_.narcs = expanded_state->arcs.size();
      i_ = 0;
    }
    // Ideally we want to call CopyArcToTemp() now, but we rely on the fact that
    // the calling code needs to call Done() before accessing Value(); we call
    // CopyArcToTemp() from Done().  Of course this is slightly against the
    // semantics of Done(), but it's more efficient to have Done() call
    // CopyArcToTemp() than this function or Next(), as Done() already has to
    // test that the arc-iterator has not reached the end.
  }

  inline bool Done() {
    if (i_ < data_.narcs) {
      CopyArcToTemp();
      return false;
    } else {
      return true;
    }
  }

  inline void Next() {
    i_++;
    // Note: logically, at this point we should do:
    // if (i_ < data_.size)
    //  CopyArcToTemp();
    // Instead we move this CopyArcToTemp() invocation into Done(), which we
    // know will always be called after Next() and before Value(), because the
    // user has no other way of knowing whether the iterator is still valid.
    // This is for efficiency.
  }

  inline const Arc &Value() const { return arc_; }

 private:

  inline void CopyArcToTemp() {
    const StdArc &src = data_.arcs[i_];
    arc_.ilabel = src.ilabel;
    arc_.olabel = src.olabel;
    arc_.weight = src.weight;
    arc_.nextstate = (static_cast<int64>(dest_instance_) << 32) |
        src.nextstate;
  }

  // The members of 'data_' that we use are:
  //  const Arc *arcs;
  //  size_t narcs;
  ArcIteratorData<StdArc> data_;


  int32 dest_instance_;  // The index of the FstInstance that we transition to from
                         // this state.
  size_t i_;  // i_ is the index into the 'arcs' pointer.

  Arc arc_;  // 'Arc' is the current arc in the GrammarFst, that this iterator
             // is pointing to.  It will be a copy of data_.arcs[i], except with
             // the 'nextstate' modified to encode dest_instance_ in the higher
             // order bits.  Making a copy is of course unnecessary for the most
             // part, but Value() needs to return a reference; we rely on the
             // compiler to optimize out any unnecessary moves of data.
};

/**
   This function copies a GrammarFst to a VectorFst (intended mostly for testing
   and comparison purposes).  GrammarFst doesn't actually inherit from class
   Fst, so we can't just construct an FST from the GrammarFst.

   grammar_fst gets expanded by this call, and although we could make it a const
   reference (because the ArcIterator does actually use const_cast), we make it
   a non-const pointer to emphasize that this call does change grammar_fst.
 */
void CopyToVectorFst(GrammarFst *grammar_fst,
                     VectorFst<StdArc> *vector_fst);

/**
   This function prepares 'ifst' for use in GrammarFst: it ensures that it has
   the expected properties, changing it slightly as needed.  'ifst' is expected
   to be a fully compiled HCLG graph that is intended to be used in GrammarFst.
   The user will most likely want to copy it to the ConstFst type after calling
   this function.

   The following describes what this function does, and the reasons why
   it has to do these things:

     - To keep the ArcIterator code simple (to avoid branches in loops), even
       for expanded states we store the destination fst-instance index
       separately per state, not per arc.  This requires that any transitions
       across FST boundaries from a single FST must be to a single destination
       FST (for a given source state).  We fix this problem by introducing
       epsilon arcs and new states whenever we find a state that would cause a
       problem for the above.
     - In order to signal to the GrammarFst code that a particular state has
       cross-FST-boundary transitions, we set the final-prob to a nonzero value
       on that state.  Specifically, we use a weight with Value() == 4096.0.
       When the GrammarFst code sees that value it knows that it was not a
       'real' final-prob.  Prior to doing this we ensure, by adding epsilon
       transitions as needed, that the state did not previously have a
       final-prob.
     - For arcs that are final arcs in an FST that represents a nonterminal
       (these arcs would have #nonterm_exit on them), we ensure that the
       states that they transition to have unit final-prob (i.e. final-prob
       equal to One()), by incorporating any final-prob into the arc itself.
       This avoids the GrammarFst code having to inspect those final-probs
       when expanding states.

     @param [in] nonterm_phones_offset   The integer id of
                the symbols #nonterm_bos in the phones.txt file.
     @param [in,out] fst  The FST to be (slightly) modified.
 */
void PrepareForGrammarFst(int32 nonterm_phones_offset,
                          VectorFst<StdArc> *fst);


} // end namespace fst


#endif

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
   For an extended explanation of the framework of which grammar-fsts
   are a part, please see the comment in ***TBD****.

   This header implements a special FST type which we use in that
   framework.

 */



#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include "itf/decodable-itf.h"
#include "itf/options-itf.h"
#include "util/stl-utils.h"
#include "fstext/grammar-context-fst.h"

namespace fst {


class GrammarFstConfig {
 public:
  // This config class currently only has one member, but we may later add
  // others that relate to entry and exit olabels for the nonterminals, to make
  // it possible to recover the structure of invoking the nonterminals from the
  // decoding output.

  int32 nonterm_phones_offset;

  GrammarFstConfig(): nonterm_phones_offset(-1)  {}

  void Register(kaldi::OptionsItf *po) {
    po->Register("nonterm-phones-offset", &nonterm_phones_offset,
                 "The integer id of the symbol #nonterm_bos in phones.txt");
  }
  void Check() const;
};


// GrammarFstArc is an FST Arc type which differs from the expected StdArc type
// by having the state-id be 64 bits, enough to store two indexes (the higher 32
// bits for the FST-instance index, and the lower 32 bits for the state within
// that FST-instance).
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

class GrammarFst;

// Declare that we'll be overriding class ArcIterator for class GrammarFst.
// This wouldn't work if we were fully using the OpenFst framework,
// e.g. inheriting from class Fst.
template<> class ArcIterator<GrammarFst>;



/**
   GrammarFst is an FST that is 'stitched together' from multiple FSTs, that can
   recursively incorporate each other.  (This is limited to left-biphone
   phonetic context). This class does not inherit from fst::Fst and does not
   support its full interface-- only the parts that are necessary for the
   decoder.

   The basic interface is inspired by OpenFst's 'ReplaceFst' (see its
   replace.h), except that this handles left-biphone phonetic context, which
   requires, essentially, having multiple exit-points and entry-points when we
   dive into a sub-FST.  See ***TBD*** for higher-level documentation.


   Now we describe how the GrammarFst stitches together the pieces in
   'top_fst' and 'ifsts'.  The GrammarFst will treat *certain* ilabels in these
   FSTs specially.  The ilabels treated specially will be those over 1000000.
   If we find an ilabel over 1 million we can decode it as:
        (nonterm_symbol, phone_or_nonterm_eps) = decode_ilabel(ilabel)
   where:
        'nonterm_symbol' is the id of a symbol in phones.txt of the form
      #nonterm_entry, #nonterm_exit, #nonterm_return or #nonterm:foo (where
      foo is an example which stands in for any user-defined nonterminal such
      as #nonterm:name).
        'phone_or_nonterms_eps' is the id of either a phone in phones.txt, or
      the symbol #nonterm_eps which stands in for the undefined phonetic
      left-context we encounter at the start of a phone sequence.
   The decoding process is documented where the function DecodeIlabel() is
   declared.

   The following describes the FST that the GrammarFst will be equivalent to
   (up to state-numbering):
     .. TODO..


 */
class GrammarFst {
 public:
  typedef GrammarFstArc Arc;
  typedef TropicalWeight Weight;

  // StateId is actually int64.  The high-order 32 bits are indexes into the
  // instances_ vector; the low-order 32 bits are the state index in the FST
  // instance.
  typedef Arc::StateId StateId;

  // The StateId of the individual FST instances.
  typedef StdArc::StateId BaseStateId;

  typedef Arc::Label Label;



  /**
     Constructor (note: the constructor is pretty lightweight, it only
     immediately examines the start states of the provided FSTs in order to
     set up the appropriate entry points).

     For simplicity (to avoid templates) and because we want to use some of the
     class-specific functionality (like the Arcs() function), we limit the input
     FSTs to be of type ConstFst<StdArc>.  You can always construct a
     ConstFst<StdArc> if you have another StdArc-based FST type.  If the FST was
     read from disk, it may already be of type ConstFst, and dynamic_cast might
     be sufficient to convert the type.

     @param [in] config  Configuration object, containing, most importantly,
              nonterm_phones_offset which is the integer id of the symbol
             "#nonterm_bos" in phones.txt.
     @param [in] top_fst   top_fst is the top-level FST of the grammar,
              which will usually invoke the fsts in 'ifsts'.
              The fsts in 'ifsts' may also invoke each other
              recursively.  Even left-recursion is allowed,
              although if it is with zero cost, it will blow
              up when you decode.  When an FST invokes
              another, it will be with sequences of special symbols
              which would be decoded as:
                  (#nonterm:foo,p1) (#nonterm_reenter,p2)
              where p1 and p2 (which may also be #nonterm:eps) represent
              the phonetic left-context that we enter, and leave, the sub-graph
              with.
     @param [in] ifsts   ifsts is a list of pairs (nonterminal-symbol,
              the HCLG fst corresponding to that symbol).  The nonterminal
              symbols must be among the user-specified nonterminals in
              phones.txt, i.e. the things with names like "#nonterm:foo"
              and "#nonterm:bar" in phones.txt.  Also they must not be
              repeated.  ifsts may be empty, even though that doesn't
              make much sense.
    */
  GrammarFst(
      const GrammarFstConfig &config,
      const ConstFst<StdArc> &top_fst,
      const std::vector<std::pair<int32, const ConstFst<StdArc> *> > &ifsts);


  ///  This constructor should only be used prior to calling Read().
  GrammarFst() { }

  // This Write function uses Kaldi-type mechanisms,  and doesn't make use of the
  // OpenFst headers; OpenFst programs won't be able to read this format.
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &os, bool binary);

  StateId Start() const {
    // the top 32 bits of the 64-bit state-id will be zero.
    return static_cast<int64>(top_fst_->Start());
  }

  Weight Final(StateId s) const {
    // If the fst-id (top 32 bits of s) is nonzero, this state is not final,
    // because we need to return to the top-level FST before we can be final.
    if (s != static_cast<int64>(static_cast<int32>(s)))
      return Weight::Zero();
    else
      return top_fst_->Final(static_cast<BaseStateId>(s));
  }

  inline std::string Type() const { return "grammar"; }

 private:

  struct ExpandedState;

  friend class ArcIterator<GrammarFst>;

  // sets up nonterminal_map_.
  void InitNonterminalMap();

  // sets up entry_points_.
  void InitEntryPoints();


  /*
    This utility funciton sets up a map from "left-context phone", meaning
    either a phone index or the index of the symbol #nonterm_bos, to
    a state-index in an FST.

      @param [in]  fst  The FST we are looking for state-indexes for
      @param [in]  entry_state  The state in the FST that arcs with
                 labels decodable as (nonterminal_symbol, left_context_phone)
                 leave.  Will either be the start state (if 'nonterminal_symbol'
                 corresponds to #nonterm_begin), or an internal state
                 (if 'nonterminal_symbol' corresponds to #nonterm_reenter).
                 The destination-states of those arcs will be the values
                 we set in 'phone_to_state'
      @param [in]  nonterminal_symbol  The index in phones.txt of the
                 nonterminal symbol we expect to be encoded in the ilabels
                 of the arcs leaving 'entry_state'.  Will either correspond
                 to #nonterm_begin or #nonterm_reenter.
      @param [out] phone_to_arc  We output the map from left_context_phone
                 to the arc-index (i.e. the index we'd have to Seek() to
                 in an arc-iterator set up for the state 'entry_state).
   */
  void InitEntryOrReentryPoints(
      const ConstFst<StdArc> &fst,
      int32 entry_state,
      int32 nonterminal_symbol,
      std::unordered_map<int32, int32> *phone_to_arc);


  inline int32 GetPhoneSymbolFor(enum NonterminalValues n) {
    return config_.nonterm_phones_offset + static_cast<int32>(n);
  }
  /**
     Decodes an ilabel in to a pair of (nonterminal, left_context_phone).  Crashes
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
  // it is not currently present.  It adds it to the expanded_states map for
  // this FST instance, and returns it.  Note: it is possible for it to add NULL
  // to that map, and return NULL, in the case where this was a genuine
  // final-state which did not need expansion.
  ExpandedState *ExpandState(int32 instance_id, BaseStateId state_id);

  // Called from the ArcIterator constructor when we encounter an FST state with
  // nonzero final-prob, this function first looks up this state_id in
  // 'expanded_states' member of the corresponding FstInstance, and returns it
  // if already present; otherwise it populates the 'expanded_states' map with
  // something for this state_id and returns the value.
  //
  // It is possible for the ExpandedState this function returns to be NULL; this is
  // what happens for we encounter states in the top-level FST that were
  // genuinely final (as opposed to states that we artificially set a final-prob
  // on to trigger this state-expansion code.
  inline ExpandedState *GetExpandedState(int32 instance_id,
                                         BaseStateId state_id) {
    FstInstance &instance = instances_[instance_id];
    std::unordered_map<BaseStateId, ExpandedState*> &expanded_states =
        instance.expanded_states;

    std::unordered_map<BaseStateId, ExpandedState*>::iterator iter =
        expanded_states.find(state_id);
    if (iter != expanded_states.end())
      return iter->second;
    else
      return ExpandState(instance_id, state_id);
  }

  // Configuration object; contains nonterm_phones_offset.
  GrammarFstConfig config_;

  // The top-level FST passed in by the user; contains the start state and
  // final-states, and may invoke FSTs in 'ifsts_' (which can also invoke
  // each other recursively).
  const Fst<StdArc> *top_fst_;

  // A list of pairs (nonterm, fst), where 'nonterm' is a user-defined
  // nonterminal symbol as numbered in phones.txt (e.g. #nonterm:foo), and
  // 'fst' is the corresponding FST.
  std::vector<std::pair<int32, const ConstFst<StdArc> *> > ifsts_;

  // Maps from the user-defined nonterminals like #nonterm:foo as defined
  // in phones.txt, to the corresponding index into 'ifsts_'.
  std::unordered_map<int32, int32> nonterminal_map_;

  // entry_points_, which will have the same dimension as ifst_, is a map from
  // left-context phone (i.e. either a phone-index or #nonterm_bos) to the
  // corresponding arc-index leaving the start-state in this FST.
  std::vector<std::unordered_map<int32, int32> > entry_points_;

  // Represents an expanded state in an FstInstance.  We expand states whenever
  // we encounter states with a nonzero final-prob.  Note: nonzero final-probs
  // function as a signal that says "trye to expand this state"; the user is
  // expected to have made sure that all states that need to be expanded
  // actually have a nonzero final-prob.
  struct ExpandedState {
    // The final-prob for expanded states is always zero; to avoid
    // corner cases, we ensure this via adding epsilon arcs where
    // needed.

    // fst-instance index of destination state (we will have ensured previously
    // that this is the same for all outgoing arcs).
    int32 dest_fst_instance;

    // List of arcs out of this state, where the 'nextstate' element will be the
    // lower-order 32 bits of the destination state and the higher order bits
    // will be given by 'dest_fst_instance_'.  We do it this way, instead of
    // constructing a vector<Arc>, in order to simplify the ArcIterator code and
    // avoid unnecessarybranches in loops over arcs.
    // We guarantee that this 'arcs' array will always be nonempty; this
    // is to avoid certain hassles on Windows with automated bounds-checking.
    std::vector<StdArc> arcs;
  };


  // An FstInstance represents an instance of a sub-FST.
  struct FstInstance {
    // ifst_index is the index into the ifsts_ vector that corresponds to this
    // FST instance, or -1 if this is the top-level instance.
    int32 ifst_index;

    // Pointer to the FST corresponding to this instance: top_fst_ if
    // ifst_index == -1, or ifsts_[ifst_index].second otherwise.
    const ConstFst<StdArc> *fst;

    // 'expanded_states', which will be populated on demand as states are
    // accessed, will only contain entries for states in this FST that have
    // nonzero final-prob.  (The final-prob is used as a kind of signal to this
    // code that the state needs expansion).
    //
    // In cases where states actually needed to be expanded because they
    // contained transitions to other sub-FSTs, the value in the map will
    // be non-NULL; in cases where we only tried to expand the state  because
    // it happened to have a final-prob (but didn't really need expansion),
    // we'll store NULL and the calling code will know to treat it as
    // a normal state.
    std::unordered_map<BaseStateId, ExpandedState*> expanded_states;


    // The instance-id of the FST we return to when we are done with this one
    // (or -1 if this is the top-level FstInstance so there is nowhere to
    // return).
    int32 return_instance;

    // The state in the FST of 'return_instance' at which we expanded this; the
    // values in 'reentry_points' are the 'nextstates' of arcs leaving this
    // state.
    int32 return_state;

    // return_points has similar semantics to entry_points_, but it refers to
    // the state index in the FST to which we will return from this FST.
    // It's indexed by phone index (or #nonterm_eps).  We make use of this
    // when we expand states in this FST that have nonzero final-prob.
    std::unordered_map<int32, int32> reentry_points;
  };

  std::vector<FstInstance> instances_;
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
  using BaseStateId = typename StdArc::StateId;
  using ExpandedState = GrammarFst::ExpandedState;

  inline ArcIterator(const GrammarFst &fst_in, StateId s) {
    // Caution: uses const_cast to evade const rules on GrammarFst.
    // This is for compatibility with how things work in OpenFst.
    GrammarFst &fst = const_cast<GrammarFst&>(fst_in);
    int32 instance_id = s >> 32;  // high order bits
    BaseStateId base_state = static_cast<int32>(s);  // low order bits.
    const GrammarFst::FstInstance &instance = fst.instances_[instance_id];
    const ConstFst<StdArc> *base_fst = instance.fst;
    ExpandedState *expanded_state;
    if (fst.Final(base_state) == TropicalWeight::Zero() ||
        (expanded_state = fst.GetExpandedState(instance_id, s)) == NULL) {
      // If this state doesn't cross FST boundaries then we just iterate over
      // the arcs in the underlying fst.
      base_fst->InitArcIterator(s, &data_);
      i_ = 0;
      dest_instance_ = instance_id;
    } else {
      dest_instance_ = expanded_state->dest_fst_instance;
      data_.arcs = &(expanded_state->arcs[0]);
      data_.narcs = expanded_state->arcs.size();
    }
    // In principle we might want to call CopyArcToTemp() now, but
    // we rely on the fact that the calling code needs to call Done()
    // before accessing Value(); Done() calls CopyArcToTemp().
    // Of course this is slightly against the semantics of Done(), but
    // it's more efficient to have Done() call CopyArcToTemp() than
    // this function or Next().
  }

  inline bool Done() {
    if (i_ < data_.narcs) {
      CopyArcToTemp();
      return true;
    } else {
      return false;
    }
  }

  inline void Next() {
    i_++;
    // Note: logically, at this point we would do:
    // if (i_ < data_.size)
    //  CopyArcToTemp();
    // Instead we move this CopyArcToTemp() invocation into Done(), which we
    // know will always be called after Next() and before Value(), because the
    // user has no other way of knowing whether the iterator is still valid.
    // The reason for moving it that way is that I believe it will give the
    // compiler an easier time to optimize the code.
  }

  // Each time we do Next(), we re-copy the ilabel, olabel and weight of the arc
  // of the user-provided FST to 'arc_', and compute its nexstate from
  // dest_instance_ and the original arc's nextstate.  This is of course
  // potentially wasteful, as we copy the arc to a temporary.  The hope is that
  // the compiler will realize what is happening and will optimize most of the
  // waste out.  If not, this can be revisited later.
  inline const Arc &Value() const { return arc_; }

 private:

  inline void CopyArcToTemp() {
    const StdArc &src = data_.arcs[i_];
    arc_.ilabel = src.ilabel;
    arc_.olabel = src.olabel;
    arc_.weight = src.weight;
    arc_.nextstate = (static_cast<int64>(dest_instance_) << 32) +
        src.nextstate;
  }

  // The members of 'data_' that we use are:
  //  const Arc *arcs;             // O.w. arcs pointer
  //  size_t narcs;                // ... and arc count.
  ArcIteratorData<StdArc> data_;


  int32 dest_instance_;  // The index of the FstInstance that transitions from this
                         // state go to.
  size_t i_;  // i_ is the index into the 'arcs' pointer.

  Arc arc_;  // 'Arc' is the current arc in the GrammarFst, that we point to.
             // It will be the same as data_.arcs[i], except with the
             // 'nextstate' modified to encode the dest_instance_ in the
             // higher order bits.
};


/**
   This function prepares 'ifst' for use in GrammarFst: it ensures that it has
   the expected properties, changing it slightly as needed.  'ifst' is expected
   to be a fully compiled HCLG graph that is intended to be used in GrammarFst.
   The user will most likely want to copy it to the ConstFst type after calling
   this function.  Prior to doing the 'special fixes', if 'fst' was not
   ilabel-sorted it ilabel-sorts it.  We'd normally do this prior to decoding
   with an FST.  You can't ilabel-sort after calling this function, though, as
   that would put some arcs in the wrong order.

   The following describes what this function does, and the reasons why
   it has to:

     - To keep the ArcIterator code simple, even for expanded states we store
       the destination fst-instance index separately per state.  This requires
       that any transitions across FST boundaries from a single FST must be to a
       single destination FST (per source state).  We can fix this problem by
       introducing epsilon arcs and new states whenever we find a state that
       would cause a problem for the above.
     - We need to ensure that, after a state has been expanded by the GrammarFst
       code, epsilon arcs still precede non-epsilon arcs, which a requirement of
       our decoders.  (The issue is that there are certain "special" arcs, which
       cross FST boundaries, that the GrammarFst code will turn into epsilons).
     - In order to signal to the GrammarFst code that a state has
       cross-FST-boundary transitions, we set final-probs on some states.  We
       set these to a weight with Value() == 1024.0.  When the GrammarFst code
       sees that value it knows that it was not a 'real' final-prob.


     @param [in] nonterm_phones_offset   The integer id of
                the symbols #nonterm_bos in the phones.txt file.
     @param [in,out] fst  The FST to be (slightly) modified.

 */
void PrepareForGrammarFst(int32 nonterm_phones_offset,
                          VectorFst<StdArc> *fst);


} // end namespace fst


#endif

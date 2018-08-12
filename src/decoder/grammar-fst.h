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

namespace fst {


class GrammarFstConfig {
  // This config class currently only has one member, but we may later add
  // others that relate to entry and exit olabels for the nonterminals, to make
  // it possible to recover the structure of invoking the nonterminals from the
  // decoding output.

  int32 nonterm_phones_offset;

  GrammarFstConfig(): nonterm_phones_offset(-1)  {}

  void Register(OptionsItf *po) {
    po->Register("nonterm-phones-offset", &nonterm_phones_offset,
                 "The integer id of the symbol #nonterm_bos in phones.txt");
  }
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

  static const string &Type() {
    static const string *const type = "grammar";
    return *type;
  }
};

class GrammarFst;

// Declare that we'll be overriding class ArcIterator for class GrammarFst.
// This wouldn't work if we were fully using the OpenFst framework,
// e.g. inheriting from class Fst.
template<> class ArcIterator<GrammarFst>;



/**
   GrammarFst is an FST that is 'stitched together' from multiple FSTs, that can
   recursively incorporate each other.  (This is limited to left-biphone
   phonetic context).  Note: this class does not inherit from fst::Fst and
   does not support its full interface-- only the parts that are necessary
   for the decoder.

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

  // The StateId of the individual FSTs instances.
  typedef StdArc::StateId BaseStateId;



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
              repeated.
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
    return static_cast<int64>(top_fst_.Start());
  }

  Weight Final(StateId s) const {
    // If the fst-id (top 32 bits of s) is nonzero, this state is not final,
    // because we need to return to the top-level FST before we can be final.
    if (s != static_cast<int64>(static_cast<int32>(s)))
      return Weight::Zero();
    else
      return top_fst_.Final(static_cast<BaseStateId>(s));
  }


 private:

  friend class ArcIterator<GrammarFst>;

  // This function, which is to be called after top_fsts_, ifsts_ and
  // nonterm_phones_offset_ have been set up, initializes the following derived
  // variables:
  //  encoding_multiple_
  //  nonterminal_map_
  //  entry_points_
  void Init();
  void CreateEntryMap();


  // This function creates and returns an ExpandedState corresponding to a
  // particular state-id in the FstInstance for this instance_id.  It is called
  // when we have determined that an ExpandedState (possibly NULL) needs to be
  // created and that it is not currently present.  It adds it to the
  // expanded_states map and returns it.
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
    const FstInstance &instance = instances_[instance_id];
    std::unordered_map<BaseStateId, ExpandedState*> &expanded_states =
        instance.expanded_states;

    std::unordered_map<BaseStateId, ExpandedState*>::iterator iter =
        expanded_states.find(state_id);
    if (iter != expanded_states.end())
      return *iter;
    else
      return ExpandState(instance_id, state_id);
  }

  // Configuration object; contains nonterm_phones_offset.
  GrammarFstConfig config_;

  // The top-level FST passed in by the user; contains the start state and
  // final-states, and may invoke FSTs in 'ifsts_' (which can also invoke
  // each other recursively).
  const Fst<StdArc> &top_fst_;

  // A list of pairs (nonterm, fst), where 'nonterm' is a user-defined
  // nonterminal symbol as numbered in phones.txt (e.g. #nonterm:foo), and
  // 'fst' is the corresponding FST.
  std::vector<std::pair<int32, const Fst<StdArc> *> > ifsts_;


  // encoding_multiple_ is the smallest multiple of 1000 that is greater than
  // config_.nonterm_phones_offset.  It's a value that we use to encode and decode
  // ilabels on HCLG.
  int32 encoding_multiple_;

  // Maps from the user-defined nonterminals like #nonterm:foo as defined
  // in phones.txt, to the corresponding index into 'ifsts_'.
  std::unordered_map<Label, int32> nonterminal_map_;

  // entry_points_, which will have the same dimension as ifst_, is a map from
  // left-context phone (i.e. either a phone-index or #nonterm_bos) to the
  // corresponding start-state in this FST.
  std::vector<std::unordered_map<Label, BaseStateId> > entry_points_;

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
    // We need to guarantee that this 'arcs' array will always be nonempty; this
    // is to avoid certain hassles on Windows with automated bounds-checking.
    std::vector<StdArc> arcs;
  };


  // An FstInstance represents an instance of a sub-FST.
  struct FstInstance {
    // ifst_index is the index into the ifsts_ vector that corresponds to this
    // FST instance.
    int32 ifst_index;

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


    // entry_points, which is only set up for instances other than instance 0
    // (the top-level instance), is a map from phonetic left-context (either a
    // phone id or the symbol #nonterm_eps representing no left-context, using
    // the integer mapping defined in phones.txt), to the state in the FST that
    // we start at if that is the left-context.  Will be kNoStateId for
    // left-contexts which this FST does not make available.
    std::vector<BaseStateId> entry_points;

    // The instance-id of the FST we return to when we are done with this one
    // (or -1 if this is the top-level FstInstance so there is nowhere to
    // return).
    int32 return_instance;

    // return_points has similar semantics to entry_points, but it refers to
    // the state index in the FST to which we will return from this FST.
    // It's indexed by phone index (or #nonterm_eps).  We make use of this
    // when we expand states in this FST that have nonzero final-prob.
    std::vector<BaseStateId> return_points;

  };


  std::vector<FstInstance> instances_;




};


/**
   This is the overridden template for class ArcIterator for GrammarFst.
   This is only used in the decoder, so we don't need to implement all
   the functionality that the regular ArcIterator has.
 */
template <>
class ArcIterator<GrammarFst> {
  using Arc = typename GrammarFst::Arc;
  using BaseArc = StdArc;
  using StateId = typename Arc::StateId;  // int64
  using BaseStateId = typename StdArc::StateId;

  ArcIterator(const GrammarFst &fst, StateId s) {
    int32 instance_id = s >> 32;  // high order bits
    BaseStateId base_state = static_cast<int32>(s);  // low order bits.
    const GrammarFst::FstInstance &instance = fst.instances_[instance_id];
    const ConstFst<StdArc> *base_fst = instance.fst;
    if (fst->Final(base_state) == TropicalWeight::Zero()) {
      // If this state is non-final then we just iterate over the
      // arcs in the underlying fst.
      arcs_ = base_fst->GetImpl()->Arcs(base_state);
      end_arcs_ = arcs_ + base_fst->GetImpl()->NumArcs(base_state);
    } else {
      instance.

  }

 private:

  StdArc *arcs_;  // Current pointer to arcs in base FST (in usual case)
  StdArc *arcs_end_;  // end of arcs array.

  Arc arc_;  //


};




} // end namespace fst


#endif

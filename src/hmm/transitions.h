// hmm/transitions.h

// Copyright 2009-2012  Microsoft Corporation
//                2015  Guoguo Chen
//                2019  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_HMM_TRANSITIONS_H_
#define KALDI_HMM_TRANSITIONS_H_

#include "base/kaldi-common.h"
#include "util/const-integer-set.h"
#include "fst/fst-decl.h" // forward declarations.
#include "hmm/topology.h"
#include "itf/options-itf.h"
#include "itf/context-dep-itf.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {


// The class Transitions handles various integer mappings.
// It used to be the home for the trainable transitions, but these
// no longer exist.  This class can be initialized from the
// tree and the topology.
//
// The topology of an individual phone is as defined in topology.h.
//
//  This class basically defines the concept of a "transition-id",
//  which is a construct that we use in compiled decoding graphs
//  to make it easy to look up the 'pdf-id' (think of this as the
//  distribution or neural net output column associated with this
//  state) and also figure out which phone we are in and which
//  arc in that phone.
//
//  In the original Kaldi, this object contained trainable transition
//  probabilities, but these have been removed to simplify things.
//
//  A transition-id maps to a 4-tuple as follows:
//       (pdf-id, phone, topo-state, arc-index)
//  where 'topo-state' is the state index in the fst::StdFst
//  for the topology, and 'arc-index' is the index of
//  the arc leaving that state (zero for the first-listed one,
//  one for the second, etc.)


// List of the various types of quantity referred to here and what they mean:
//           phone:  a phone index (1, 2, 3 ...)
//       topo-state:  a state index in the phone-topology FST (see topology.h)
//       arc-index:  The index of the arc leaving this topo-state:
//                   0 for the first-listed one, 1 for the second.  Will be used
//                   to Seek() in the ArcIterator.
//          pdf-id:  A number output by the Compute() function of ContextDependency (it
//                   indexes pdf's, either forward or self-loop).  Zero-based.
//                   In DNN-based systems this would be the column index of
//                   the neural net output.
// (*)self-loop-pdf-id:  The pdf-id associated with the self-loop of this state,
//                   if there is one (we do not allow >1), or -1 if there is no
//                   self-loop.  This will be the same as 'pdf-id' if this transition
//                   *is* the self-loop.  It might seem odd that we require this
//                   to get the transition-id for a non-self-loop arc; the reason
//                   why it's necessary is that we initially create the graph
//                   without self-loops (for efficiency) and we need to be able
//                   to look up the corresponding self-loop transition-id to
//                   add self-loops to the graph.
//
//   transition-id:  The numbers that we put on the decoding-graph arcs.
//                   Each transition-id is associated with a 4-tuple
//                   (pdf-id, phone, topo-state, arc-index).
//


class Transitions {

 public:
  /// Initialize the object.  This is deterministic, so initializing
  /// from the same objects will give you an equivalent numbering.
  /// The class keeps a copy of the Topology object, but not
  /// the ContextDependency object.
  Transitions(const ContextDependencyInterface &ctx_dep,
              const Topology &topo);


  /// Constructor that takes no arguments: typically used prior to calling Read.
  Transitions(): num_pdfs_(0) { }

  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

  // This struct is the information associated with one transition-id.
  // You can work out the transition-id from the first 5 fields.
  struct TransitionIdInfo {
    int32 phone;      // The phone
    int32 topo_state; // The state in the topology FST for this phone
    int32 arc_index;  // The arc-index leaving this state
    int32 pdf_id;     // The pdf-id associated with this arc (obtained from the
                      // tree and phonetic-context information, etc.)

    int32 self_loop_pdf_id;  // The pdf-id associated with the self-loop
                             // transition (if any) leaving the *destination*
                             // state of this arc, or -1 if that state has no
                             // self-loop.  Search for (*) above for
                             // explanation.

    // The remaining fields are 'derived information' that are worked out
    // from the information above and from the phone topology, and placed
    // here for convenience.

    // is_self_loop is true if this is a self-loop (a transition to the same
    // state).  We often need to know this, so it's convenient to have this
    // information here.
    bool is_self_loop;
    // is_initial is true if this is a transition leaving the
    // initial state.
    // you transition through the HMM (we check that the topology has no
    // other transitions to the first HMM-state).
    bool is_initial;

    // is_final is true if this is a transition entering a final
    // state.  This is used together with is_initial (and boundary
    // information) to locate phone boundaries, e.g. for lattice
    // word alignment: an 'is_final' transition-id followed by an
    // 'is_initial' transition-id marks a phone boundary, which
    // we know because we do not allow the start-state in
    // topologies to be final.
    bool is_final;

    // transition_cost is the cost (negative log-prob) of this transition).
    BaseFloat transition_cost;
    // The transition-id associated with the self-loop of the *destination* of
    // this arc, if there is one, or 0 if there is no such self-loop.
    int32 self_loop_transition_id;


    bool operator < (const TransitionIdInfo &other) const {
      if (phone < other.phone) return true;
      else if (phone > other.phone) return false;
      else if (topo_state < other.topo_state) return true;
      else if (topo_state > other.topo_state) return false;
      else if (pdf_id < other.pdf_id) return true;
      else if (pdf_id > other.pdf_id) return false;
      else return (self_loop_pdf_id < other.self_loop_pdf_id);
    }
    // TODO.  operator == can compare all members. Also compare derived members?
    bool operator == (const TransitionIdInfo &other) const {
      return phone == other.phone && topo_state == other.topo_state &&
          pdf_id == other.pdf_id && self_loop_pdf_id == other.self_loop_pdf_id;
    }
  };


  /// return reference to HMM-topology object.
  const Topology &GetTopo() const { return topo_; }

  const TransitionIdInfo &InfoForTransitionId(int32 transition_id) const;

  inline int32 TransitionIdToPdfFast(int32 trans_id) const;

  /// This allows you to look up a transition-id.  It returns 0 if nothing
  /// was found.
  int32 TupleToTransitionId(int32 phone, int32 topo_state, int32 arc_index,
                            int32 pdf_id, int32 self_loop_pdf_id) const;


  /// Returns the total number of transition-ids (note, these are one-based).
  inline int32 NumTransitionIds() const { return info_.size() - 1; }

  // NumPdfs() returns the number of pdfs (pdf-ids) in the tree,
  // as returned by ctx_dep.NumPdfs() for the tree passed to the constructor.
  int32 NumPdfs() const { return num_pdfs_; }

  /// Returns a sorted, unique list of phones.
  const std::vector<int32> &GetPhones() const { return topo_.GetPhones(); }


  /// Print will print the transition model in a human-readable way, for purposes of human
  /// inspection.  The "occs" are optional (they are indexed by pdf-id).
  void Print(std::ostream &os,
             const std::vector<std::string> &phone_names,
             const Vector<double> *occs = NULL);

  /// returns true if this is identical to 'other'
  bool operator == (const Transitions &other);

 private:

  // Called from constructor.  initializes info_ (at least, the first 5
  // fields); you then have to call ComputeDerived() to initalize teh rest.
  void ComputeInfo(const ContextDependencyInterface &ctx_dep);

  void ComputeDerived();  // Called from constructor and Read function.

  void Check() const;


  Topology topo_;

  /// Information about transition-ids, indexed by transition-id.
  /// the tuples are in sorted order which allows us to do the reverse mapping from
  /// tuple to transition id.
  std::vector<TransitionIdInfo> info_;


  /// Accessing pdf_ids_[i] allows us to look up info_[i].pdf_id in a way that
  /// is more friendly to memory caches than accessing info_; this is done in
  /// the inner loops of decoders so it makes sense to optimize for it.
  std::vector<int32> pdf_ids_;

  /// This is a copy of the NumPdfs() returned by the tree when we constructed
  /// this object.  Note: pdf-ids are zero-based.
  int32 num_pdfs_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(Transitions);
};

inline int32 Transitions::TransitionIdToPdfFast(int32 trans_id) const {
  // Note: it's a little dangerous to assert this only in paranoid mode.
  // However, this function is called in the inner loop of decoders and
  // the assertion likely takes a significant amount of time.  We make
  // sure that past the end of thd id2pdf_id_ array there are big
  // numbers, which will make the calling code more likely to segfault
  // (rather than silently die) if this is called for out-of-range values.
  KALDI_PARANOID_ASSERT(
      static_cast<size_t>(trans_id) < pdf_ids_.size() &&
      "Likely graph/model mismatch (graph built from wrong model?)");
  return pdf_ids_[trans_id];
}

/// Works out which pdfs might correspond to the given phones.  Will return true
/// if these pdfs correspond *just* to these phones, false if these pdfs are also
/// used by other phones.
/// @param trans_model [in] Transition-model used to work out this information
/// @param phones [in] A sorted, uniq vector that represents a set of phones
/// @param pdfs [out] Will be set to a sorted, uniq list of pdf-ids that correspond
///                   to one of this set of phones.
/// @return  Returns true if all of the pdfs output to "pdfs" correspond to phones from
///          just this set (false if they may be shared with phones outside this set).
bool GetPdfsForPhones(const Transitions &trans_model,
                      const std::vector<int32> &phones,
                      std::vector<int32> *pdfs);

/// Works out which phones might correspond to the given pdfs. Similar to the
/// above GetPdfsForPhones(, ,)
bool GetPhonesForPdfs(const Transitions &trans_model,
                      const std::vector<int32> &pdfs,
                      std::vector<int32> *phones);
/// @}


} // end namespace kaldi


#endif

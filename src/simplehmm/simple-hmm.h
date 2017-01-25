// hmm/simple-hmm.h

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Guoguo Chen)
//                2016  Vimal Manohar (Johns Hopkins University)

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

#ifndef KALDI_HMM_SIMPLE_HMM_H
#define KALDI_HMM_SIMPLE_HMM_H

#include "base/kaldi-common.h"
#include "util/const-integer-set.h"
#include "fst/fst-decl.h" // forward declarations.
#include "hmm/hmm-topology.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace simple_hmm {

/// \addtogroup hmm_group
/// @{

// The class SimpleHmm is a repository for the transition probabilities.
// The model is exactly like a single phone. It has a HMM topology defined in
// hmm-topology.h.  Each HMM-state has a number of
// transitions (and final-probs) out of it.  Each emitting HMM-state defined in
// the HmmTopology class has an associated class-id.
// The transition model associates the
// transition probs with the (HMM-state, class-id).  We associate with
// each such pair a transition-state.  Each
// transition-state has a number of associated probabilities to estimate;
// this depends on the number of transitions/final-probs in the topology for
// that HMM-state.  Each probability has an associated transition-index.
// We associate with each (transition-state, transition-index) a unique transition-id.
// Each individual probability estimated by the transition-model is asociated with a
// transition-id.
//
// List of the various types of quantity referred to here and what they mean:
//       HMM-state:  a number (0, 1, 2...) that indexes TopologyEntry (see hmm-topology.h)
// transition-state:  the states for which we estimate transition probabilities for transitions
//                    out of them.  In some topologies, will map one-to-one with pdf-ids.
//                    One-based, since it appears on FSTs.
// transition-index:  identifier of a transition (or final-prob) in the HMM.  Indexes the
//                    "transitions" vector in HmmTopology::HmmState.  [if it is out of range,
//                    equal to transitions.size(), it refers to the final-prob.]
//                    Zero-based.
//   transition-id:   identifier of a unique parameter of the
//   SimpleHmm.
//                    Associated with a (transition-state, transition-index) pair.
//                    One-based, since it appears on FSTs.
//
// List of the possible mappings SimpleHmm can do:
//                   (HMM-state, class-id) -> transition-state
//                   (transition-state, transition-index) -> transition-id
//  Reverse mappings:
//                        transition-id -> transition-state
//                        transition-id -> transition-index
//                     transition-state -> HMM-state
//                     transition-state -> class-id
//
// The main things the SimpleHmm object can do are:
//    Get initialized (need HmmTopology objects).
//    Read/write.
//    Update [given a vector of counts indexed by transition-id].
//    Do the various integer mappings mentioned above.
//    Get the probability (or log-probability) associated with a particular transition-id.


struct MleSimpleHmmUpdateConfig {
  BaseFloat floor;
  BaseFloat mincount;
  MleSimpleHmmUpdateConfig(BaseFloat floor = 0.01,
                           BaseFloat mincount = 5.0):
      floor(floor), mincount(mincount) { }

  void Register (OptionsItf *opts) {
    opts->Register("transition-floor", &floor,
                   "Floor for transition probabilities");
    opts->Register("transition-min-count", &mincount,
                   "Minimum count required to update transitions from a state");
  }
};

struct MapSimpleHmmUpdateConfig {
  BaseFloat tau;
  MapSimpleHmmUpdateConfig(): tau(5.0) { }

  void Register (OptionsItf *opts) {
    opts->Register("transition-tau", &tau, "Tau value for MAP estimation of transition "
                   "probabilities.");
  }
};

class SimpleHmm {

 public:
  /// Initialize the object [e.g. at the start of training].
  /// The class keeps a copy of the HmmTopology object.
  SimpleHmm(const HmmTopology &hmm_topo);

  /// Constructor that takes no arguments: typically used prior to calling Read.
  SimpleHmm() { }

  void Read(std::istream &is, bool binary);  // note, no symbol table: topo object always read/written w/o symbols.
  void Write(std::ostream &os, bool binary) const;


  /// return reference to HMM-topology object.
  const HmmTopology &GetTopo() const { return topo_; }

  /// \name Integer mapping functions
  /// @{

  int32 HmmStateToTransitionState(int32 hmm_state) const;
  int32 PairToTransitionId(int32 trans_state, int32 trans_index) const;
  int32 TransitionIdToTransitionState(int32 trans_id) const;
  int32 TransitionIdToTransitionIndex(int32 trans_id) const;
  int32 TransitionStateToHmmState(int32 trans_state) const;
  int32 TransitionStateToPdfClass(int32 trans_state) const;
  // returns the self-loop transition-id, or zero if
  // this state doesn't have a self-loop.
  int32 SelfLoopOf(int32 trans_state) const;

  int32 TransitionIdToPdfClass(int32 trans_id) const;
  int32 TransitionIdToHmmState(int32 trans_id) const;

  /// @}

  bool IsFinal(int32 trans_id) const;  // returns true if this trans_id goes to the final state
  // (which is bound to be nonemitting).
  bool IsSelfLoop(int32 trans_id) const;  // return true if this trans_id corresponds to a self-loop.

  /// Returns the total number of transition-ids (note, these are one-based).
  inline int32 NumTransitionIds() const { return id2state_.size()-1; }

  /// Returns the number of transition-indices for a particular transition-state.
  /// Note: "Indices" is the plural of "index".   Index is not the same as "id",
  /// here.  A transition-index is a zero-based offset into the transitions
  /// out of a particular transition state.
  int32 NumTransitionIndices(int32 trans_state) const;

  /// Returns the total number of transition-states (note, these are one-based).
  int32 NumTransitionStates() const { return states_.size(); }

  // NumPdfs() in the model.
  int32 NumPdfs() const { return num_pdfs_; }

  // Transition-parameter-getting functions:
  BaseFloat GetTransitionProb(int32 trans_id) const;
  BaseFloat GetTransitionLogProb(int32 trans_id) const;

  // The following functions are more specialized functions for getting
  // transition probabilities, that are provided for convenience.

  /// Returns the log-probability of a particular non-self-loop transition
  /// after subtracting the probability mass of the self-loop and renormalizing;
  /// will crash if called on a self-loop.  Specifically:
  /// for non-self-loops it returns the log of (that prob divided by (1 minus
  /// self-loop-prob-for-that-state)).
  BaseFloat GetTransitionLogProbIgnoringSelfLoops(int32 trans_id) const;

  /// Returns the log-prob of the non-self-loop probability
  /// mass for this transition state. (you can get the self-loop prob, if a self-loop
  /// exists, by calling GetTransitionLogProb(SelfLoopOf(trans_state)).
  BaseFloat GetNonSelfLoopLogProb(int32 trans_state) const;

  /// Does Maximum Likelihood estimation.  The stats are counts/weights, indexed
  /// by transition-id.  This was previously called Update().
  void MleUpdate(const Vector<double> &stats,
                 const MleSimpleHmmUpdateConfig &cfg,
                 BaseFloat *objf_impr_out,
                 BaseFloat *count_out);

  /// Does Maximum A Posteriori (MAP) estimation.  The stats are counts/weights,
  /// indexed by transition-id.
  void MapUpdate(const Vector<double> &stats,
                 const MapSimpleHmmUpdateConfig &cfg,
                 BaseFloat *objf_impr_out,
                 BaseFloat *count_out);

  /// Print will print the simple HMM in a human-readable way, 
  /// for purposes of human
  /// inspection.  
  /// The "occs" are optional (they are indexed by pdf-classes).
  void Print(std::ostream &os,
             const Vector<double> *occs = NULL);


  void InitStats(Vector<double> *stats) const { stats->Resize(NumTransitionIds()+1); }

  void Accumulate(BaseFloat prob, int32 trans_id, Vector<double> *stats) const {
    KALDI_ASSERT(trans_id <= NumTransitionIds());
    (*stats)(trans_id) += prob;
    // This is trivial and doesn't require class members, but leaves us more open
    // to design changes than doing it manually.
  }

  /// returns true if all the integer class members are identical (but does not
  /// compare the transition probabilities.
  bool Compatible(const SimpleHmm &other) const;

 private:
  void MleUpdateShared(const Vector<double> &stats,
                       const MleSimpleHmmUpdateConfig &cfg,
                       BaseFloat *objf_impr_out, BaseFloat *count_out);
  void MapUpdateShared(const Vector<double> &stats,
                       const MapSimpleHmmUpdateConfig &cfg,
                       BaseFloat *objf_impr_out, BaseFloat *count_out);
  
  // called from constructor and Read(): initializes states_
  void Initialize();  
  // called from constructor and Read(): computes state2id_ and id2state_
  void ComputeDerived();  
  // computes quantities derived from log-probs (currently just
  // non_self_loop_log_probs_; called whenever log-probs change.
  void ComputeDerivedOfProbs();  
  void InitializeProbs();  // called from constructor.
  void Check() const;

  HmmTopology topo_;

  /// States indexed by transition state minus one;
  /// the states are in sorted order which allows us to do the reverse mapping
  /// from state to transition state
  std::vector<int32> states_;

  /// Gives the first transition_id of each transition-state; indexed by
  /// the transition-state.  Array indexed 1..num-transition-states+1
  /// (the last one is needed so we can know the num-transitions of the last
  /// transition-state.
  std::vector<int32> state2id_;

  /// For each transition-id, the corresponding transition
  /// state (indexed by transition-id).
  std::vector<int32> id2state_;

  /// For each transition-id, the corresponding log-prob.
  /// Indexed by transition-id.
  Vector<BaseFloat> log_probs_;

  /// For each transition-state, the log of (1 - self-loop-prob).  Indexed by
  /// transition-state.
  Vector<BaseFloat> non_self_loop_log_probs_;

  /// This is equal to the one + highest-numbered pdf class.
  int32 num_pdfs_;


  DISALLOW_COPY_AND_ASSIGN(SimpleHmm);

};

/// @}


}  // end namespace simple_hmm
}  // end namespace kaldi


#endif

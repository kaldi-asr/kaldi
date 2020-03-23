// chain/chain-den-graph.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_DEN_GRAPH_H_
#define KALDI_CHAIN_CHAIN_DEN_GRAPH_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "chain/chain-datastruct.h"
#include "hmm/transition-model.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace chain {


/**  This class is responsible for storing the FST that we use as the
     'anti-model' or 'denominator-model', that models all possible phone
     sequences (or most possible phone sequences, depending how we built it)..
     It stores the FST in a format where we can access both the transitions out
     of each state, and the transitions into each state.

     This class supports both GPU and non-GPU operation, but is optimized for
     GPU.
 */
class DenominatorGraph {
 public:

  // the number of states in the HMM.
  int32 NumStates() const;

  // the number of PDFs (the labels on the transitions are numbered from 0 to
  // NumPdfs() - 1).
  int32 NumPdfs() const { return num_pdfs_; }

  DenominatorGraph();

  // Initialize from epsilon-free acceptor FST with pdf-ids plus one as the
  // labels.  'num_pdfs' is only needed for checking.
  DenominatorGraph(const fst::StdVectorFst &fst,
                   int32 num_pdfs);

  // returns the pointer to the forward-transitions array, indexed by hmm-state,
  // which will be on the GPU if we're using a GPU.
  const Int32Pair *ForwardTransitions() const;

  // returns the pointer to the backward-transitions array, indexed by
  // hmm-state, which will be on the GPU if we're using a GPU.
  const Int32Pair *BackwardTransitions() const;

  // returns the array to the actual transitions (this is indexed by the ranges
  // returned from the ForwardTransitions and BackwardTransitions arrays).  The
  // memory will be GPU memory if we are using a GPU.
  const DenominatorGraphTransition *Transitions() const;

  // returns the initial-probs of the HMM-states... note, these initial-probs
  // don't mean initial at the start of the file, because we usually train on
  // pieces of a file.  They are approximate initial-probs obtained by running
  // the HMM for a fixed number of time-steps (e.g. 100) and averaging the
  // posteriors over those time-steps.  The exact values won't be very critical.
  // Note: we renormalize each HMM-state to sum to one before doing this.
  const CuVector<BaseFloat> &InitialProbs() const;

  // This function outputs a modified version of the FST that was used to
  // build this object, that has an initial-state with epsilon transitions to
  // each state, with weight determined by initial_probs_; and has each original
  // state being final with probability one (note: we remove epsilons).  This is
  // used in computing the 'penalty_logprob' of the Supervision objects, to
  // ensure that the objective function is never positive, which makes it more
  // easily interpretable.  'ifst' must be the same FST that was provided to the
  // constructor of this object.  [note: ifst and ofst may be the same object.]
  // This function ensures that 'ofst' is ilabel sorted (which will be useful in
  // composition).
  void GetNormalizationFst(const fst::StdVectorFst &ifst,
                           fst::StdVectorFst *ofst);

  // This function is only used in testing code.
  void ScaleInitialProbs(BaseFloat s) { initial_probs_.Scale(s); }

  // Use default copy constructor and assignment operator.
 private:
  // functions called from the constructor
  void SetTransitions(const fst::StdVectorFst &fst, int32 num_pfds);

  // work out the initial-probs.  Note, there are no final-probs; we treat all
  // states as final with probability one [we have a justification for this..
  // assuming it's roughly a well-normalized HMM, this makes sense; note that we
  // train on chunks, so the beginning and end of a chunk appear at arbitrary
  // points in the sequence.  At both beginning and end of the chunk, we limit
  // ourselves to only those pdf-ids that were allowed in the numerator
  // sequence.
  void SetInitialProbs(const fst::StdVectorFst &fst);

  // forward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions out of this state.
  CuArray<Int32Pair> forward_transitions_;
  // backward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions into this state.
  CuArray<Int32Pair> backward_transitions_;
  // This stores the actual transitions.
  CuArray<DenominatorGraphTransition> transitions_;

  // The initial-probability of all states, used on the first frame of a
  // sequence [although we also apply the constraint that on the first frame,
  // only pdf-ids that were active on the 1st frame of the numerator, are
  // active.  Because in general sequences won't start at the start of files, we
  // make this a generic probability distribution close to the limiting
  // distribution of the HMM.  This isn't too critical.
  CuVector<BaseFloat> initial_probs_;

  int32 num_pdfs_;
};


// Function that does acceptor minimization without weight pushing...
// this is useful when constructing the denominator graph.
void MinimizeAcceptorNoPush(fst::StdVectorFst *fst);

// Utility function used while building the graph.  Converts
// transition-ids to pdf-ids plus one.  Assumes 'fst'
// is an acceptor, but does not check this (only looks at its
// ilabels).
void MapFstToPdfIdsPlusOne(const TransitionModel &trans_model,
                           fst::StdVectorFst *fst);

// Starting from an acceptor on phones that represents some kind of compiled
// language model (with no disambiguation symbols), this function creates the
// denominator-graph.  Note: there is similar code in chain-supervision.cc, when
// creating the supervision graph.
void CreateDenominatorFst(const ContextDependency &ctx_dep,
                          const TransitionModel &trans_model,
                          const fst::StdVectorFst &phone_lm,
                          fst::StdVectorFst *den_graph);


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_DEN_GRAPH_H_

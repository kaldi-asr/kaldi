// chain/den-graph.h

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


#ifndef KALDI_CHAIN_DEN_GRAPH_H_
#define KALDI_CHAIN_DEN_GRAPH_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "chain/chain-datastruct.h"
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

  int32 NumStates();

  DenominatorGraph(const fst::StdVectorFst> &fst);

  // returns the pointer to the forward-transitions array, indexed by hmm-state,
  // which will be on the GPU if we're using a GPU.
  const Int32Pair *ForwardTransitions() const;

  // returns the pointer to the backward-transitions array, indexed by
  // hmm-state, which will be on the GPU if we're using a GPU.
  const Int32Pair *BackwardTransitions() const;

  // returns the array to the actual transitions (this is indexed by the ranges
  // returned from the ForwardTransitions and BackwardTransitions arrays).  The
  // memory will be GPU memory if we are using a GPU.
  const GraphTransition *Transitions() const;

  // returns the initial-probs of the HMM-states... note, these initial-probs
  // don't mean initial at the start of the file, because we usually train
  // on pieces of a file.  They are approximate initial-probs obtained
  // by running the HMM for a fixed number of iters from a flat start.  The
  // exact values won't be very critical.
  const CuVector<BaseFloat> &InitialProbs() const;

  // returns the index of the HMM-state that has the highest value in
  // InitialProbs (and which we believe will always be reachable from most other
  // states... later on we may check this more carefully [TODO]).
  // It's used in getting the 'arbitrary_scale' value to keep the alphas
  // in a good dynamic range.
  int32 SpecialHmmState() const { return special_hmm_state_; }
 private:
  // functions called from the constructor
  void SetTransitions(const fst::StdVectorFst> &fst);

  // work out the initial-probs and the 'special state'
  // Note, there are no final-probs; we treat all states as final
  // with probability one [we have a justification for this..
  // assuming it's roughly a well-normalized HMM, this makes sense;
  // note that we train on chunks, so the beginning and end of a chunk
  // appear at arbitrary points in the sequence.
  // At both beginning and end of the chunk, we limit ourselves to
  // only those pdf-ids that were allowed in the numerator sequence.
  void SetInitialProbs(const fst::StdVectorFst> &fst);

  // forward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions out of this state.
  CuArray<Int32Pair> forward_transitions_;
  // backward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions out of this state.
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

  // The index of a somewhat arbitrarily chosen HMM-state that we
  // use for adjusting the alpha probabilities.  It needs to be
  // one that is reachable from all states (i.e. not a special
  // state that's only reachable at sentence-start).  We choose
  // whichever one has the greatest initial-prob.  It's set
  // in SetInitialProbs().
  int32 special_hmm_state_;
};




}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_HMM_H_

// chain/chain-generic-numerator.h

// Copyright       2017  Hossein Hadian


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


#ifndef KALDI_CHAIN_CHAIN_GENERIC_NUMERATOR_H_
#define KALDI_CHAIN_CHAIN_GENERIC_NUMERATOR_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-supervision.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-array.h"
#include "chain/chain-datastruct.h"

namespace kaldi {
namespace chain {

/* This extended comment explains how end-to-end (i.e. flat-start) chain
   training is done and how it is mainly different from regular chain training.

   The key differnece with regular chain is that the end-to-end supervision FST
   (i.e. numerator graph) can have loops and more than one final state (we
   call it 'Generic' numerator in the code). This is because we do not
   have any alignments so we can't split the utterances and we can't remove
   the self-loops.
   Of course, the end-to-end FST still has to be epsilon-free and have pdf_id+1
   on its input and output labels, just like the regular supervision FST.
   The end-to-end supervision (which contains the generic numerator FST's) is
   created using TrainingGraphToSupervision from a training FST (i.e. an FST
   created using compile-train-graphs). It is stored in the same struct as
   regular supervision (i.e. chain::Supervision) but this function
   sets the 'e2e' flag to true. Also the generic  numerator FSTs
   are stored in 'e2e_fsts' instead of 'fst'.

   The TrainingGraphToSupervision function is called in nnet3-chain-e2e-get-egs
   binary to create end-to-end chain egs. The only difference between a regular
   and end-to-end chain example is the supervision as explained above.

   class GenericNumeratorComputation is responsible for doing Forward-Backward
   on a generic FST (i.e. the kind of FST we use in end-to-end chain
   training). It is the same as DenominatorComputation with 2 differences:
   [1] it runs on CPU
   [2] it does not use leakyHMM

   When the 'e2e' flag of a supervision is set, the ComputeChainObjfAndDeriv
   function in chain-training.cc uses GenericNumeratorComputation (instead
   of NumeratorCompuation) to compute the numerator derivatives.
 */


// This class is responsible for the forward-backward of the
// end-to-end 'supervision' (numerator) FST. This kind of FST can
// have self-loops.
// Note: An end-to-end supervision is the same as a regular supervision
// (class chain::Supervision) except the 'e2e' flag is set to true
// and the numerator FSTs are stored in 'e2e_fsts' instead of 'fst'

class GenericNumeratorComputation {

 public:

  /// Initializes the object.
  GenericNumeratorComputation(const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output);

  // Does the forward computation.  Returns the total log-prob multiplied
  // by supervision_.weight.
  BaseFloat Forward();

  // Does the backward computation and (efficiently) adds the derivative of the
  // nnet output w.r.t. the (log-prob times supervision_.weight times
  // deriv_weight) to 'nnet_output_deriv'.
  bool Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);

 private:

  // Defining this constant as an enum is easier.  it controls a memory/speed
  // tradeoff, determining how many frames' worth of the transposed derivative
  // we store at a time.  It's not very critical; the only disadvantage from
  // setting it small is that we have to invoke an AddMat kernel more times.
  enum { kMaxDerivTimeSteps = 8 };

  // sets up the alpha for frame t = 0.
  void AlphaFirstFrame();

  // the alpha computation for some 0 < t <= num_time_steps_.
  void AlphaGeneralFrame(int32 t);

  BaseFloat ComputeTotLogLike();

  // sets up the beta for frame t = num_time_steps_.
  void BetaLastFrame();

  // the beta computation for 0 <= beta < num_time_steps_.
  void BetaGeneralFrame(int32 t);

  // some checking that we can do if debug mode is activated, or on frame zero.
  // Sets ok_ to false if a bad problem is detected.
  void BetaGeneralFrameDebug(int32 t);


  const Supervision &supervision_;

  // the transposed neural net output.
  Matrix<BaseFloat> exp_nnet_output_transposed_;

  // in_transitions_ lists all the incoming transitions for
  // each state of each numerator graph
  // out_transitions_ does the same but for the outgoing transitions
  std::vector<std::vector<std::vector<DenominatorGraphTransition> > >
  in_transitions_, out_transitions_;

  // final probs for each state of each numerator graph
  Matrix<double> final_probs_; // indexed by seq, state

  // an offset subtracted from the logprobs of transitions out of the first
  // state of each graph to help reduce numerical problems. Note the
  // generic forward-backward computations cannot be done in log-space.
  Vector<BaseFloat> offsets_;

  // maximum number of states among all the numerator graphs
  // (it is used as a stride in alpha_ and beta_)
  int32 max_num_hmm_states_;

  // the derivs w.r.t. the nnet outputs (transposed)
  // (the dimensions and functionality is the same as in
  // DenominatorComputation)
  Matrix<BaseFloat> nnet_output_deriv_transposed_;

  // forward and backward probs matrices. These have the
  // same dimension and functionality as alpha_ and beta_
  // in DenominatorComputation except here we don't use beta
  // sums (becasue we don't use leakyHMM). However, we use
  // alpha sums to help avoid numerical issues.
  Matrix<double> alpha_;
  Matrix<double> beta_;

  // vector of total probs (i.e. for all the sequences)
  // (it's exactly like 'tot_probe_' in DenominatorComputation)
  Vector<double> tot_prob_;

  bool ok_;
};

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_GENERIC_NUMERATOR_H_

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
#define BFloat double

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


// This class is responsible for the forward-backward of the
// end-to-end 'supervision' (numerator) FST. This kind of FST can
// have self-loops.

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

  enum { kMaxDerivTimeSteps = 4 };
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


  //  const ChainTrainingOptions &opts_;
  const Supervision &supervision_;

  // the transposed neural net output.
  Matrix<BaseFloat> exp_nnet_output_transposed_;

  // in_transitions_ lists all the incoming transitions for
  // each state of each numerator graph
  // out_transitions_ does the same but for the outgoing transitions
  std::vector<std::vector<std::vector<DenominatorGraphTransition> > >
  in_transitions_, out_transitions_;

  // final probs for each state of each numerator graph
  Matrix<BFloat> final_probs_; // indexed by seq, state

  // an offset subtracted from the logprobs of transitions out of the first
  // state of each graph to help reduce numerical problems. Note the
  // generic forward-backward computations cannot be done in log-space.
  Vector<BaseFloat> offsets_;

  // maximum number of states among all the numerator graphs
  int32 max_num_hmm_states_;

  // the derivs w.r.t. the nnet outputs (transposed)
  Matrix<BaseFloat> nnet_output_deriv_transposed_;

  // forward and backward probs matrices
  Matrix<BFloat> alpha_;
  Matrix<BFloat> beta_;

  // vector of total probs (i.e. for all the sequences)
  Vector<BFloat> tot_prob_;

  bool ok_;
};

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_GENERIC_NUMERATOR_H_

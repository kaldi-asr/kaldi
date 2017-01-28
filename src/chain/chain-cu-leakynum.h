// chain/chain-cu-numerator.h

// Copyright       2015  Hossein Hadian


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


#ifndef KALDI_CHAIN_CHAIN_CU_LEAKYNUM_H_
#define KALDI_CHAIN_CHAIN_CU_LEAKYNUM_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-array.h"
#include "chain/chain-num-graph.h"
#include "chain/chain-training.h"

namespace kaldi {
namespace chain {


// This class is responsible for the forward-backward of the 'supervision'
// (numerator) FST.
//
// note: the supervision.weight is ignored by this class, you have to apply
// it externally.


class CuLeakyNumeratorComputation {
 public:


  CuLeakyNumeratorComputation(const ChainTrainingOptions &opts,
                              const NumeratorGraph &num_graph,
                              const DenominatorGraph &den_graph,
                              const CuMatrixBase<BaseFloat> &nnet_output);

  // Does the forward computation.  Returns the total log-prob multiplied
  // by supervision_.weight.
  BaseFloat Forward();

  // Does the backward computation and (efficiently) adds the derivative of the
  // nnet output w.r.t. the (log-prob times supervision_.weight times
  // deriv_weight) to 'nnet_output_deriv'.
  bool Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);

  inline int32 TotalNumStates() {return num_graph_.MaxNumStates()
                                        + den_graph_.NumStates(); }
 private:

  enum { kMaxDerivTimeSteps = 4 };

  // sets up the alpha for frame t = 0 for num and den states.
  void AlphaFirstFrame();

  // the alpha computation for some 0 < t <= num_time_steps_ for num states.
  void AlphaNumFrame(int32 t);

  // the alpha computation for some 0 < t <= num_time_steps_ for den states.
  void AlphaDenFrame(int32 t);
  
  // computes alpha sum (i.e. c_t which is for over/under flow) and alpha prime
  // which is used in alpha computation for numerator states
  // it also addes alpha prime to the current alpha (refer to the algorithm)
  void AlphaSumAndPrime(int32 t);

  // computes alpha hat and adds it to alpha for den states
  void AlphaHat(int32 t);


  BaseFloat ComputeTotLogLike();

  // sets up the beta for t = num_time_steps_ for den and num states.
  void BetaLastFrame();

  // the beta computation for 0 <= beta < num_time_steps_ for num states.
  void BetaNumFrame(int32 t);

  // the beta computation for 0 <= beta < num_time_steps_ for den states.
  void BetaDenFrame(int32 t);

  // computes beta prime
  // which is used in beta computation for numerator states
  // it also addes beta prime to the current beta (refer to the algorithm)
  void BetaPrime(int32 t);

  // computes beta hat and adds it to beta for den states
  void BetaHat(int32 t);

  // some checking that we can do if debug mode is activated, or on frame zero.
  // Sets ok_ to false if a bad problem is detected.
  void BetaGeneralFrameDebug(int32 t);


  const ChainTrainingOptions &opts_;
  NumeratorGraph num_graph_;
  DenominatorGraph den_graph_;

  // number of separate frame sequences
  int32 num_sequences_;

  // number of frames per sequence.  nnet_output_.NumRows() equals
  // num_sequences_ * frames_per_sequence.
  int32 frames_per_sequence_;

  // the exp transpsed of the neural net output.
  CuMatrix<BaseFloat> exp_nnet_output_transposed_;

  // the derivs w.r.t. the nnet outputs (transposed)
  CuMatrix<BaseFloat> nnet_output_deriv_transposed_;

  CuMatrix<BaseFloat> alpha_num_;

  CuMatrix<BaseFloat> alpha_den_;

  CuMatrix<BaseFloat> beta_num_;

  CuMatrix<BaseFloat> beta_den_;

  CuVector<BaseFloat> tot_prob_;

  BaseFloat leak_eta_;
  CuVector<BaseFloat> unleak_etas_;
  BaseFloat num_transitions_scale_; // scale applied to all num to num transitions (except when we are computing hats or primes) 
  BaseFloat den_transitions_scale_;

  // the log of tot_prob_.
  CuVector<BaseFloat> tot_log_prob_;

  bool ok_;
};


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_CU_LEAKYNUM_H_


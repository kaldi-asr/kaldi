// chain/chain-denominator.h

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


#ifndef KALDI_CHAIN_CHAIN_DENOMINATOR_H_
#define KALDI_CHAIN_CHAIN_DENOMINATOR_H_

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

namespace kaldi {
namespace chain {

// This does forward-backward in parallel on a number of sequences, using a
// single HMM.

class DenominatorComputation {
 public:
  /*
    Constructor.  'nnet_output' is the raw nnet output (which we'll treat as
    pseudo-log-likelihoods).

    @param [in] graph  The HMM that we use for the denominator (like a decoding graph,
                       with pdf-ids on the transitions).

    'nnet_output' is the


  */


  DenominatorComputation(const DenominatorGraph &graph,
                         const CuMatrixBase<BaseFloat> &nnet_output,
                         const std::vector<std::vector<int32 > > &initial_pdf_ids,
                         const std::vector<std::vector<int32 > > &final_pdf_ids);



  // Does the forward computation, and returns the total negated log-like summed
  // over all sequences.
  BaseFloat Forward();

  void Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);


private:
  const CudaHmm &hmm_;
  const Supervision &supervision_;
  const CuMatrixBase<BaseFloat> &nnet_output_;


  // sets up the alpha for frame t = 0.
  void AlphaFirstFrame();
  // the alpha computation for some 0 < t <= num_time_steps_.
  void AlphaGeneralFrame(int32 t);

  // done after all the alphas, this function computes and returns the total
  // log-likelihood summed over all the sequences, and sets tot_prob_ (if we're
  // doing correction) log_correction_term_.  Note, this won't be scaled by
  // 'deriv_scale' (which of course we haven't seen by the time this is called,
  // from the Forward() computation).
  BaseFloat ComputeTotLogLike();

  // backward computation without rearrangement.
  void BackwardInternal();

  void BetaLastFrame();
  // beta computation for 0 <= beta < num_time_steps_.
  void BetaGeneralFrame(int32 t);

  // the transpose of the nnet output (more efficient).
  CuMatrix<BaseFloat> nnet_output_transposed_;

  // the derivs w.r.t. the nnet output-derivs.
  CuMatrix<BaseFloat> nnet_output_deriv_tranposed_;

  // the alpha probabilities; dimension is num-time-steps + 1 by (num-hmm-states
  // * num-sequences).  Note, they are not logs.
  CuMatrix<BaseFloat> alpha_;

  // the beta probabilities (rolling buffer); dimension is 2 * (num-hmm-states *
  // num-sequences).  Note: for efficiency and simplification, these are actually
  // the beta / tot_prob_.
  CuMatrix<BaseFloat> beta_;

  // the total probability for each sequence, excluding the product of
  // correction terms.  we multiply on each frame by 1/alpha of hmm-state 0 of
  // the previous frame; the products
  CuVector<BaseFloat> tot_prob_;
  // the log of tot_prob_.
  CuVector<BaseFloat> tot_log_prob_;

  // [once we start using correction terms, this will be:] the log of the total
  // correction term for each sequence, which is the product of the alpha_[special hmm state] over
  // all the frames.
  CuVector<BaseFloat> log_correction_term_;
};



}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_DENOMINATOR_H_


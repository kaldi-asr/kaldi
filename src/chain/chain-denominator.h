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
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-array.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-training.h"

namespace kaldi {
namespace chain {


/*
  This extended comment describes how we implement forward-backward without log
  and without overflow, and also the leaky-HMM idea.

  We'll start by establishing the notation for conventional forward-backward,
  then add the 'arbitrary-scale' concept that prevents overflow, and then
  add the 'leaky-hmm' concept.

  All this is done in parallel over multiple sequences, but the computations
  are independent over the separate sequences, so we won't introduce any notation
  or index for the sequence; we'll just explain it for one sequences.

  Suppose we have I hmm-states, numbered i = 0 ... I-1 (we'll use i and j for
  hmm-state indexes).  Let foll(i) give a list of arcs leaving state i, and
  pred(i) give a list of arcs entering state i, and we'll use notation like:
    for (j, p, n) in foll(i):
  for iterating over those arcs, where in this case j is the destination-state,
  p is the transition-probability of the arc and n is the pdf-id index.
  We can then look up the emission probability as x(t, n) for some frame
  0 <= t < T.

  ** Version 1 of the computation (naive version) **

  * Forward computation (version 1)

  In the forward computation we're computing alpha(i, t) for 0 <= t <= T):
    - For the first frame, set alpha(0, i) = init(i), where init(i) is the
      initial-probabilitiy from state i.  # in our framework these are obtained
      #  by running the HMM for a while and getting an averaged occupation
      # probability, and using this as an initial-prob, since the boundaries of
      # chunks don't really correspond to utterance boundaries in general.]
    - For t = 1 ... T:
        for i = 0 ... I-1:
           alpha(t, i) = 0
           for (j, p, n) in pred(i):  # note: j is preceding-state.
              alpha(t, i) += x(t-1, n) * alpha(t-1, j) * p.

    - total-prob = \sum_i alpha(T, i).  # note, we take the final-probs of all states
                                        # to be 1.0.

  * Backward computation (version 1)

  And now for the backward computation.  Contrary to tradition, we include the
  inverse of the total-prob as a factor in the betas.  This is both more
  convenient (it simplifies the way we obtain posteriors), and makes the
  algorithm more generalizable as all the beta quantities can be interpreted as
  the partial derivative of the logprob with respect to their corresponding
  alpha.

  In forward backward notation, gamma is normally used for state-level
  occupation probabilities, but what we care about here is pdf-id-level
  occupation probabilities (i.e. the partial derivative of the log-likelihood
  w.r.t. the logs of the x(t, n) quantities), so we use gamma for that.

    - for the final frame:
       for each i, beta(T, i) = 1 / total-prob.
    - for t = T-1 ... 0:
        for i = 0 ... I-1:
           beta(t, i) = 0
           for (j, p, n) in foll(i):  # note: j is following-state.
              beta(t, i) += x(t, n) * beta(t+1, j) * p.
              gamma(t, n) += alpha(t, i) * x(t, n) * beta(t+1, j) * p.

  ** Version 2 of the computation (renormalized version) **

  Version 1 of the algorithm is susceptible to numeric underflow and overflow,
  due to the limited range of IEEE floating-point exponents.
  Define tot-alpha(t) = \sum_i alpha(t, i).  Then the renormalized version of
  the computation is as above, except whenever the quantity x(t, n) appears,
  we replace it with x(t, n) / alpha(t).  In the algorithm we refer to
  1.0 / tot-alpha(t) as 'arbitrary_scale', because mathematically we can use any
  value here as long as we are consistent and the value only varies with t
  and not with n; we'll always get the same posteriors (gamma).

  When the algorithm outputs log(total-prob) as the total log-probability
  of the HMM, we have to instead return the expression:
    log(total-prob) + \sum_{t=0}^{T-1} tot-alpha(t).
  to correct for the scaling of the x values.

  The algorithm is still vulnerable to overflow in the beta computation because
  it's possible that the dominant path could have a very tiny alpha.  However,
  once we introduce the leaky-HMM idea (below), this problem will disappear.

  ** Version 3 of the computation (leaky-HMM version) **

  The leaky-HMM idea is intended to improve generalization by allowing paths
  other than those explicitly allowed by the FST we compiled.  Another way to
  look at it is as a way of hedging our bets about where we split the utterance,
  so it's as we're marginalizing over different splits of the utterance.  You
  could also think of it as a modification of the FST so that there is an
  epsilon transition from each state to a newly added state, with probability
  one, and then an epsilon transition from the newly added state to each state
  with probability leaky-hmm-prob * init(i) [except we need a mechanism so that
  no more than two epsilon transitions can be taken per frame- this would involve
  creating two copies of the states]

  Recall that we mentioned that init(i) is the initial-probability of
  HMM-state i, but these are obtained in such a way that they can be treated
  as priors, or average occupation-probabilities.

  Anyway, the way we formulate leaky-hmm is as follows:

  * Forward computation (version 3)

  Let leaky-hmm-prob be a constant defined by the user, with 0.1 being a typical
  value.  It defines how much probability we give to the 'leaky' transitions.

  - For frame 0, set alpha(0, i) = init(i).
  - For 0 <= t <= T, define tot-alpha(t) = \sum_i alpha(t, i).
  - For 0 <= t <= T, define alpha'(t, i) = alpha(t, i) + tot-alpha(t) * leaky-hmm-prob * init(i).

  - For 1 <= t <= T, the computation of alpha(t, i) is as before except we use
      the previous frame's alpha' instead of alpha.  That is:
           alpha(t, i) = 0
           for (j, p, n) in pred(i):  # note: j is preceding-state.
              alpha(t, i) += alpha'(t-1, j) * p * x(t-1, n) / tot-alpha(t-1)

  - total-prob = \sum_i alpha'(T, i)

  The corrected log-prob that we return from the algorithm will be
   (total-prob + \sum_{t=0}^{T-1} tot-alpha(t)).

  * Backward computation (version 3)

  The backward computation is as follows.  It is fairly straightforward to
  derive if you think of it as an instance of backprop where beta, tot-beta and
  beta' are the partial derivatives of the output log-prob w.r.t. the
  corresponding alpha, tot-alpha and alpha' quantities.  Note, tot-beta is not
  really the sum of the betas as its name might suggest, it's just the
  derivative w.r.t. tot-alpha.

   - beta'(T, i) = 1 / total-prob.
   - for 0 <= t <= T, define tot-beta(t) = leaky-hmm-prob * \sum_i init(i) * beta'(t, i)
   - for 0 <= t <= T, define beta(t, i) = beta'(t, i) + tot-beta(t).
   - for 0 <= t < T, we compute beta'(t, i) and update gamma(t, n) as follows:
        for 0 <= i < I:
           beta'(t, i) = 0
           for (j, p, n) in foll(i):  # note: j is following-state.
              beta'(t, i) += beta(t+1, j) * p * x(t, n) / tot-alpha(t)
              gamma(t, n) += alpha'(t, i) * beta(t+1, j) * p *  x(t, n) / tot-alpha(t)

   Note: in the code, the tot-alpha and tot-beta quantities go in the same
   memory location that the corresponding alpha and beta for state I would go.

 */


// This does forward-backward in parallel on a number of sequences, using a
// single HMM.
class DenominatorComputation {
 public:
  /*
    Constructor.  'nnet_output' is the raw nnet output (which we'll treat as
    pseudo-log-likelihoods).

    @param [in] opts  The options.
    @param [in] graph  The HMM that we use for the denominator (like a decoding graph,
                       with pdf-ids on the transitions).
    @param [in] num_sequences The number of separate time sequences (all of the same length)
                       that we are working with.  Must divide nnet_output.NumRows().
    @param [in] nnet_output  The output of the neural network for this minibatch.
                       The rows must be ordered as (first frame of all sequences)
                       (second frame of all sequences), etc.
  */
  DenominatorComputation(const ChainTrainingOptions &opts,
                         const DenominatorGraph &den_graph,
                         int32 num_sequences,
                         const CuMatrixBase<BaseFloat> &nnet_output);

  // Does the forward computation, and returns the total negated log-like summed
  // over all sequences.  You will have to scale this by any supervision
  // weighting factor, manually.
  BaseFloat Forward();

  // this adds deriv_weight times (the derivative of the log-prob w.r.t. the
  // nnet output), to 'nnet_output_deriv'.
  // returns true if everything seemed OK, false if a failure was detected.
  bool Backward(BaseFloat deriv_weight,
                CuMatrixBase<BaseFloat> *nnet_output_deriv);

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
  // does the 'alpha-dash' computation for time t.  this relates to
  // 'leaky hmm'.
  void AlphaDash(int32 t);

  // done after all the alphas, this function computes and returns the total
  // log-likelihood summed over all the sequences, and sets tot_prob_ (if we're
  // doing correction) log_correction_term_.  Note, this won't be scaled by
  // 'deriv_scale' (which of course we haven't seen by the time this is called,
  // from the Forward() computation).
  BaseFloat ComputeTotLogLike();

  void BetaDashLastFrame();
  // beta computation for 0 <= beta < num_time_steps_.
  void BetaDashGeneralFrame(int32 t);
  // compute the beta quantity from the beta-dash quantity (relates to leaky hmm).
  void Beta(int32 t);

  // some checking that we can do if debug mode is activated, or on frame zero.
  // Sets ok_ to false if a bad problem is detected.
  void BetaGeneralFrameDebug(int32 t);

  const ChainTrainingOptions &opts_;
  const DenominatorGraph &den_graph_;

  // number of separate frame sequences
  int32 num_sequences_;
  // number of frames per sequence.  nnet_output_.NumRows() equals
  // num_sequences_ * frames_per_sequence.
  int32 frames_per_sequence_;

  // The transpose of the exp() of the nnet output (the transpose is more
  // convenient for memory locality, and the exp() avoids us having to
  // exponentiate in the forward-backward).
  //
  // The row-index is the pdf-id; and the column index equals (frame_index *
  // num_sequences + sequence_index).
  CuMatrix<BaseFloat> exp_nnet_output_transposed_;

  // the derivs w.r.t. the nnet outputs (transposed)
  CuMatrix<BaseFloat> nnet_output_deriv_transposed_;

  // the (temporarily) alpha and (more permanently) alpha-dash probabilities;
  // dimension is (frames_per_sequence + 1) by (num-hmm-states * num-sequences +
  // num_sequences).  Note, they are not logs.  The last 'num_sequences'
  // columns, where the alpha for the state indexed 'num_hmm_states' would live,
  // are for the alpha-sums, which relates to leaky HMM.
  CuMatrix<BaseFloat> alpha_;

  // the beta (also beta-dash) probabilities (rolling buffer); dimension is 2 *
  // (num-hmm-states * num-sequences + num_sequences).  [the last
  // 'num_sequences' columns are for the beta-sums, which relates to leaky HMM.]
  // Note: for efficiency and to simplify the equations, these are actually the
  // beta / tot_prob_.
  CuMatrix<BaseFloat> beta_;

  // the total probability for each sequence, excluding the product of
  // correction terms.  [the correction terms refer to the fact that we multiply
  // on each frame by 1/alpha of hmm-state 0 of the previous frame.].
  // After the correction terms the total probability is fairly close to 1,
  // which is why we can store it as non-log.
  CuVector<BaseFloat> tot_prob_;

  // the log of tot_prob_.
  CuVector<BaseFloat> tot_log_prob_;

  // the log of the total correction term for each sequence, which is the
  // product of the alpha-sums [used in the leaky-hmm computation] over all the
  // frames.  The 'correction terms' are terms that we divide the alphas and
  // betas by in order to keep them in a good dynamic range.  The product of
  // them must be included in the total likelihood.
  CuVector<BaseFloat> log_correction_term_;

  bool ok_;
};



}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_DENOMINATOR_H_


// chain/chain-kernels.cu

// Copyright  2015  Johns Hopkins University (author: Daniel Povey)


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


#include <cfloat>
#include "chain/chain-kernels-ansi.h"



template <typename Real>
__device__ inline void atomic_add(Real* address, Real value) {
  Real old = value;
  Real ret = atomicExch(address, 0.0f);
  Real new_old = ret + old;
  while ((old = atomicExch(address, new_old)) != 0.0f) {
    new_old = atomicExch(address, 0.0f);
    new_old += old;
  }
}


// one iteration of the forward computation in the 'tombstone' CTC HMM computation.
// The grid y determines which HMM-state we handle.  [put this in the grid because
// HMM-states don't all take the same amount of time in the backwards direction, and it's
// better for scheduling to have them at the outer level.]
// The block x and grid x determine which sequence (0 ... num_sequences - 1) we handle;
// note that num_sequences == the number of elements in the minibatch, and we
// insist they all have the same number of time steps.
// note: 'probs' is indexed by sequence-index + (pdf-index * prob_stride).
__global__
static void _cuda_chain_hmm_forward(const Int32Pair *backward_transitions,
                                    const DenominatorGraphTransition *transitions,
                                    int32_cuda t, int32_cuda num_sequences,
                                    int32_cuda special_hmm_state,
                                    const BaseFloat *probs,
                                    int32_cuda prob_stride,
                                    const BaseFloat *prev_alpha,
                                    BaseFloat *this_alpha) {
  // 'state_info', indexed by hmm-state, consists of [start, end] indexes into
  // the 'transition_info' array.  The state_info supplied to this function consists of
  // indexes for transitions *into* this state.
  // 'probs' has dimension num-output-indexes by num_sequences; its stride is 'prob_stride'.
  // 'prev_alpha' and 'this_alpha', which are extracted from a larger matrix,
  // both have dimension num-history-states by num-sequences.

  // s is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // h is the hmm-state index.
  int32_cuda s = threadIdx.x + blockIdx.x * blockDim.x,
      h  = blockIdx.y;
  if (s >= num_sequences)
    return;

  double this_tot_alpha = 0.0;
  const DenominatorGraphTransition
      *trans_iter = transitions + backward_transitions[h].first,
      *trans_end = transitions + backward_transitions[h].second;
  for (; trans_iter != trans_end; ++trans_iter) {
    BaseFloat transition_prob = trans_iter->transition_prob;
    int32_cuda pdf_id = trans_iter->pdf_id,
        prev_hmm_state = trans_iter->hmm_state;
    BaseFloat pseudo_loglike = probs[pdf_id * prob_stride + s],
        this_prev_alpha = prev_alpha[prev_hmm_state * num_sequences + s];
    this_tot_alpha += this_prev_alpha * transition_prob * pseudo_loglike;
  }
  // Let arbitrary_scale be the inverse of the alpha value for the
  // hmm-state indexed special_hmm_state_ on the previous frame (for this
  // sequence); we multiply this into all the transition-probabilities
  // from the previous frame to this frame, in both the forward and
  // backward passes, in order to keep the alphas in a good numeric range.
  // This won't affect the posteriors, but when computing the total
  // likelihood we'll need to compensate for it later on.
  BaseFloat arbitrary_scale =
      1.0 / prev_alpha[special_hmm_state * num_sequences + s];
  this_alpha[h * num_sequences + s] = this_tot_alpha * arbitrary_scale;
}


__global__
static void _cuda_chain_hmm_backward(const Int32Pair *forward_transitions,
                                     const DenominatorGraphTransition *transitions,
                                     int32_cuda t, int32_cuda num_sequences,
                                     int32_cuda special_hmm_state,
                                     const BaseFloat *probs, int32_cuda prob_stride,
                                     const BaseFloat *this_alpha, const BaseFloat *next_beta,
                                     BaseFloat *this_beta, BaseFloat *log_prob_deriv) {
  // 'state_info', indexed by hmm-state, consists of [start, end] indexes into
  // the 'transition_info' array.  The state_info supplied to this function consists of
  // indexes for transitions *out of* this state.
  // 'probs' has dimension num-output-indexes by num_sequences, and contains just
  //  the probs for this time index.  Its stride is prob_stride.
  // 'this_alpha', 'next_beta' and 'this_beta' all have dimension
  // num-history-states by num-sequences.
  // The beta probs are normalized in such a way (by multiplying by 1/(total-data-prob))
  // that to get occupation counts we don't need to multiply by 1/total-data-prob.
  // deriv_scale is a factor (e.g. -1.0 or -0.99) that we multiply these derivs by
  // while accumulating them.

  // s is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // h is the hmm-state index.
  int32_cuda s = threadIdx.x + blockIdx.x * blockDim.x,
      h  = blockIdx.y;
  if (s >= num_sequences)
    return;

  BaseFloat this_alpha_prob = this_alpha[h * num_sequences + s],
      inv_arbitrary_scale =
      this_alpha[special_hmm_state * num_sequences + s];
  double tot_variable_factor = 0.0;
  BaseFloat occupation_factor = this_alpha_prob / inv_arbitrary_scale;
  const DenominatorGraphTransition
      *trans_iter = transitions + forward_transitions[h].first,
      *trans_end = transitions + forward_transitions[h].second;
  for (; trans_iter != trans_end; ++trans_iter) {
    BaseFloat transition_prob = trans_iter->transition_prob;
    int32_cuda pdf_id = trans_iter->pdf_id,
        next_hmm_state = trans_iter->hmm_state;
    BaseFloat variable_factor = transition_prob *
        next_beta[next_hmm_state * num_sequences + s] *
        probs[pdf_id * prob_stride + s];
    tot_variable_factor += variable_factor;
    BaseFloat occupation_prob = variable_factor * occupation_factor;
    atomic_add(log_prob_deriv + (pdf_id * prob_stride + s), occupation_prob);
  }
  this_beta[h * num_sequences + s] = tot_variable_factor / inv_arbitrary_scale;
}


void cuda_chain_hmm_forward(dim3 Gr, dim3 Bl,
                            const Int32Pair *backward_transitions,
                            const DenominatorGraphTransition *transitions,
                            int32_cuda t, int32_cuda num_sequences,
                            int32_cuda special_hmm_state,
                            const BaseFloat *probs, int32_cuda prob_stride,
                            const BaseFloat *prev_alpha,
                            BaseFloat *this_alpha) {
  _cuda_chain_hmm_forward<<<Gr,Bl>>>(backward_transitions, transitions, t,
                                     num_sequences, special_hmm_state,
                                     probs, prob_stride, prev_alpha, this_alpha);
}

void cuda_chain_hmm_backward(dim3 Gr, dim3 Bl,
                             const Int32Pair *forward_transitions,
                             const DenominatorGraphTransition *transitions,
                             int32_cuda t, int32_cuda num_sequences,
                             int32_cuda special_hmm_state,
                             const BaseFloat *probs, int32_cuda prob_stride,
                             const BaseFloat *this_alpha, const BaseFloat *next_beta,
                             BaseFloat *this_beta,
                             BaseFloat *log_prob_deriv) {
  _cuda_chain_hmm_backward<<<Gr,Bl>>>(forward_transitions, transitions,
                                      t, num_sequences, special_hmm_state,
                                      probs, prob_stride, this_alpha, next_beta,
                                      this_beta, log_prob_deriv);
}


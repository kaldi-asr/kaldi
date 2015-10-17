// ctc/ctc-kernels.cu

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
#include "ctc/cctc-kernels-ansi.h"



// note, we don't need to know the y or z dims as they are guaranteed
// to be within range, since they are taken from the block index.

template <typename Real>
__global__
static void _cuda_rearrange_3d_tensor(int32_cuda xdim,
                                      int32_cuda xstride_in,
                                      int32_cuda ystride_in,
                                      int32_cuda zstride_in,
                                      int32_cuda xstride_out,
                                      int32_cuda ystride_out,
                                      int32_cuda zstride_out,
                                      const Real *src,
                                      Real *dst) {
  // threads only vary in x.
  int32_cuda x = threadIdx.x + blockIdx.x * blockDim.x,
      y = blockIdx.y,
      z = blockIdx.z;
  if (x >= xdim) return;
  dest[x * xstride_out + y * ystride_out + z * zstride_out] =
      src[x * xstride_in + y * ystride_in + z * zstride_in];
}


template <typename Real>
__device__ inline void atomicAdd(Real* address, Real value) {
  Real old = value;
  Real ret = atomicExch(address, 0.0f);
  Real new_old = ret + old;
  while ((old = atomicExch(address, new_old)) != 0.0f) {
    new_old = atomicExch(address, 0.0f);
    new_old += old;
  }
}


// one iteration of the forward computation in the 'tombstone' CTC HMM computation.
// The grid x and y determine which HMM-state we handle.  [put this in the grid because
// HMM-states don't all take the same amount of time in the backwards direction, and it's
// better for scheduling to have them at the outer level.]
// The block x and y determine which sequence (0 ... num_sequences - 1) we handle;
// note that num_sequences == the number of elements in the minibatch, and we
// insist they all have the same number of time steps.
__global__
static void _cuda_ctc_hmm_forward(Int32Pair *state_info,
                                  CtcHmmTransition *transition_info,
                                  int32_cuda t, int32_cuda num_sequences,
                                  const BaseFloat *num_probs, int32 num_stride,
                                  const BaseFloat *den_probs, int32 den_stride,
                                  const BaseFloat *prev_alpha, int32 alpha_stride,
                                  BaseFloat *this_alpha) {
  // 'state_info', indexed by hmm-state, consists of [start, end] indexes into
  // the 'transition_info' array.  The state_info supplied to this function consists of
  // indexes for transitions *into* this state.
  // num_probs has dimension num-output-indexes by num_sequences, and contains just
  // num-probs for this time index.
  // den_probs has dimension num-history-states by num_sequences, and contains just
  // den-probs for this time index.
  // 'prev_alpha' and 'this_alpha', which will likely be extracted from a larger
  // matrix, both have dimension num-history-states by num-sequences.

  // sequence_index is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // note: dimension of alpha array
  int32_cuda sequence_index = threadIdx.x + blockIdx.x * blockDim.x,
      num_sequences = num_dim.num_cols,
      hmm_state = blockIdx.y;
  if (sequence_index >= num_sequences)
    return;

  // 'inv_transition_scale' is a constant that we divide all
  // transition-probs by on this frame.  Mathematically we could use any value
  // here and it wouldn't affect the result, but we use this value in order
  // to keep things in a good numerical range.
  BaseFloat inv_transition_scale = prev_alpha[0];

  // there no need to say "if (hmm_state < num_hmm_states) return;" or
  // even to know the number of HMM states, because CUDA does not pad
  // the grid dimension, only the block dimension (num-threads).
  int32_cuda trans_index = state_info[hmm_state].first,
      trans_end = state_info[hmm_state].second;
  BaseFloat this_tot_alpha = 0.0;

  for (; trans_index != trans_end; trans_index++) {
    float transition_prob = transition_info[t].transition_prob;
    int32 num_index = transition_info[t].num_index,
        den_index = transition_info[t].den_index,
        src_hmm_state = transition_info[t].hmm_state;
    this_tot_alpha += transition_prob *
        prev_alpha[src_hmm_state * alpha_stride + sequence_index] *
        num_probs[num_index * num_dim.stride + sequence_index] /
        den_probs[den_index * den_dim.stride + sequence_index];
  }
  this_alpha[hmm_state * alpha_stride + sequence_index] =
      this_tot_alpha / inv_transition_scale;
}


__global__
static void _cuda_ctc_hmm_backward(Int32Pair *state_info,
                                   CtcHmmTransition *transition_info,
                                   int32_cuda t, int32_cuda num_sequences,
                                   const BaseFloat *num_probs, int32 num_stride,
                                   const BaseFloat *den_probs, int32 den_stride,
                                   const BaseFloat *this_alpha,
                                   const BaseFloat *next_beta, BaseFloat *this_beta,
                                   BaseFloat deriv_scale,
                                   BaseFloat *num_logprob_derivs, BaseFloat *den_logprob_derivs) {
  // 'state_info', indexed by hmm-state, consists of [start, end] indexes into
  // the 'transition_info' array.  The state_info supplied to this function consists of
  // indexes for transitions *out of* this state.
  // num_probs has dimension num-output-indexes * num_sequences, and contains just
  //   num-probs for this time index.  It varies first on the output-index, and then on
  //   the sequence-index.
  // den_probs has dimension num-history-states * num_sequences, and contains just
  // den-probs for this time index  varies first on the history-state and next
  // 'this_alpha', 'next_beta' and 'this_beta' all have dimension
  // num-history-states by num-sequences.
  // The beta probs are normalized in such a way (by multiplying by 1/(total-data-prob))
  // that to get occupation counts we don't need to multiply by 1/total-data-prob.
  // deriv_scale is a factor (e.g. -1.0 or -0.99) that we multiply these derivs by
  // while accumulating them.

  // sequence_index is the index of the sequence within the minibatch,
  // from 0 .. num-egs-in-this-minibatch - 1.
  // note: dimension of alpha array
  int32_cuda sequence_index = threadIdx.x + blockIdx.x * blockDim.x,
      num_sequences = num_dim.num_cols,
      hmm_state = blockIdx.y;
  if (sequence_index >= num_sequences)
    return;

  // 'inv_transition_scale' is a constant that we divide all
  // transition-probs by on this frame.  Mathematically we could use any value
  // here and it wouldn't affect the result, but we use this value in order
  // to keep things in a good numerical range.  Must be identical to the one
  // used in the forward pass.
  BaseFloat inv_transition_scale = this_alpha[sequence_index];

  // there no need to say "if (hmm_state < num_hmm_states) return;" or
  // even to know the number of HMM states, because CUDA does not pad
  // the grid dimension, only the block dimension (num-threads).
  int32_cuda trans_index = state_info[hmm_state].first,
      trans_end = state_info[hmm_state].second;

  BaseFloat tot_variable_factor = 0.0,
      common_factor = 1.0 / (den_probs[hmm_state * den_stride + sequence_index] *
                             inv_transition_scale),
      occupation_factor = common_factor * deriv_scale *
        this_alpha[hmm_state * alpha_stride + sequence_index];


  for (; trans_index != trans_end; trans_index++) {
    float transition_prob = transition_info[t].transition_prob;
    int32 num_index = transition_info[t].num_index,
        dest_hmm_state = transition_info[t].hmm_state;

    BaseFloat variable_factor = transition_prob *
        next_beta[dest_hmm_state * alpha_stride + sequence_index] *
        num_probs[num_index * num_dim.stride + sequence_index];
    // accumulate part of this beta prob.
    tot_variable_factor += variable_factor;

    // 'transition_occupation_prob' is basically an alpha * beta type of
    // quantity for this transition; it would be between 0 and 1, were it not
    // for deriv_scale.  (We don't have any 1/tot-prob to worry about here,
    // because we multiplied that into the beta at the end).
    BaseFloat scaled_transition_occupation_prob = occupation_factor * variable_factor;
    atomicAdd(num_logprob_derivs + (num_index * num_dim.stride) + sequence_index,
              scaled_transition_occupation_prob);
  }

  BaseFloat my_beta = common_factor * tot_variable_factor;
  this_beta[hmm_state * den_stride + sequence_index] = my_beta;

  // 'scaled_state_occupation_prob' is the state occupation probability
  // (between 0 and 1) multiplied by deriv_scale.
  BaseFloat scaled_state_occupation_prob = occupation_factor * my_beta;

  // set rather than add.  we'll make the interface reflect this; it saves a
  // load.q
  den_logprob_derivs[hmm_state * den_stride + sequence_index] =
      scaled_state_occupation_prob;
}



void cudaF_ctc_hmm_forward(dim3 Gr, dim3 Bl,
                           const CtcHmmHeader *hmm, int32_cuda t,
                           int32_cuda num_time_steps, int32_cuda num_sequences,
                           const float *num_probs,
                           const float *den_probs,
                           float *alpha, MatrixDim alpha_dim) {
  _cuda_ctc_hmm_forward<<<Gr,Bl>>>(hmm, t, max_t, num_sequqnces,
                                   num_probs, num_dim,
                                   den_probs, den_dim,
                                   alpha, alpha_dim);
}


void cudaF_ctc_hmm_backward(dim3 Gr, dim3 Bl,
                           const CtcHmmHeader *hmm, int32_cuda t,
                           int32_cuda num_time_steps, int32_cuda num_sequences,
                           const float *num_probs,
                           const float *den_probs,
                           float *alpha, MatrixDim alpha_dim) {
  _cuda_ctc_hmm_backward<<<Gr,Bl>>>(hmm, t, max_t, num_sequqnces,
                                    num_probs, num_dim,
                                    den_probs, den_dim,
                                    alpha, alpha_dim);
}



void cudaF_rearrange_3d_tensor(dim3 Gr, dim3 Bl,
                               int32_cuda xdim,
                               int32_cuda xstride_in,
                               int32_cuda ystride_in,
                               int32_cuda zstride_in,
                               int32_cuda xstride_out,
                               int32_cuda ystride_out,
                               int32_cuda zstride_out,
                               float *dst) {
  _cuda_rearrange_3d_tensor<<<Gr,Bl>>>(xdim, xstride_in, ystride_in, zstride_in,
                                       xstride_out, ystride_out, zstride_out);
}
void cudaF_rearrange_3d_tensor(dim3 Gr, dim3 Bl,
                               int32_cuda xdim,
                               int32_cuda xstride_in,
                               int32_cuda ystride_in,
                               int32_cuda zstride_in,
                               int32_cuda xstride_out,
                               int32_cuda ystride_out,
                               int32_cuda zstride_out,
                               float *dst) {
  _cuda_rearrange_3d_tensor<<<Gr,Bl>>>(xdim, xstride_in, ystride_in, zstride_in,
                                       xstride_out, ystride_out, zstride_out);
}

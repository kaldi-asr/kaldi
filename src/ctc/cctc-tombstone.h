// ctc/cctc-tombstone.h

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


#ifndef KALDI_CTC_CCTC_TOMBSTONE_H_
#define KALDI_CTC_CCTC_TOMBSTONE_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "ctc/language-model.h"
#include "ctc/cctc-transition-model.h"
#include "ctc/cctc-supervision.h"
#include "ctc/cctc-datastruct.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace ctc {

// This function is used inside RearrangeNnetOutput and
// RearrangeNnetOutputReverse; it copies 3d tensors.
// note: we expect MatrixIndexT to equal int32.
// It does: for x=0..xdim-1, for y=0..ydim-1, for z=0..zdim-1,
//  dst[x*dst_xstride + y*dst_ystride+z*dst_zstride] =
//  src[x*src_xstride + y*src_ystride+z*src_zstride];
template <typename Real>
void Tensor3dCopy(int32 xdim, int32 ydim, int32 zdim,
                  int32 src_xstride, int32 src_ystride, int32 src_zstride,
                  int32 dst_xstride, int32 dst_ystride, int32 dst_zstride,
                  const Real *src, Real *dst);

// might also need this:
//template <typename Real>
//void Tensor3dAdd(int32 xdim, int32 ydim, int32 zdim,
//                 int32 src_xstride, int32 src_ystride, int32 src_zstride,
//                 int32 dst_xstride, int32 dst_ystride, int32 dst_zstride,
//                 const Real *src, Real *dst);


/**
 Rearranges neural net output from an input with
     num-rows = num-sequences * num-time-steps  [arranged first by sequence-index]
     num-cols = numerator-dim.
 to an output 'nnet_output_rearranged' which has:
     num-rows = num-time-steps
     num-cols = numerator-dim * num-sequences [arranged first by nnet-output-index].

 The num-time-steps, num-sequences and numerator-dim are inferred from the
 dimensions of the matrices.  Note that this same function is used for the
 denominator indexes where we have num-history-states instead of
 numerator-dim, but the interface is the same.  */
void RearrangeNnetOutput(const CuMatrixBase<BaseFloat> &nnet_output,
                         CuMatrixBase<BaseFloat> *nnet_output_rearranged);

/**
   This function does the opposite rearrangement to the one done in
   RearrangeNnetOutput.
*/
void RearrangeNnetOutputReverse(const CuMatrixBase<BaseFloat> &nnet_output_rearranged,
                                CuMatrixBase<BaseFloat> *nnet_output);


/**  This class is responsible for storing the transition-model, interpreted as
     a HMM, in a convenient form in GPU memory if we are using the GPU (or on
     CPU memory if not).  It's intended for the 'negative' forward-backward of
     tombstone CCTC.
 */
class CctcHmm {
 public:
  // The HMM states are the same as the history-states in the
  // CctcTransitionModel.
  int32 NumHmmStates();

  CctcHmm(const CctcTransitionModel &trans_mdl);

  // returns the pointer to the forward-transitions array, indexed by hmm-state,
  // which will be on the GPU if we're using a GPU.
  const Int32Pair *ForwardTransitions() const;

  // returns the pointer to the backward-transitions array, indexed by
  // hmm-state, which will be on the GPU if we're using a GPU.
  const Int32Pair *BackwardTransitions() const;


  // returns the array to the actual transitions (this is indexed by ranges
  // returned from the ForwardTransitions and BackwardTransitions arrays).
  // The memory will be GPU memory if we are using a GPU.
  const CctcHmmTransition *Transitions() const;

  // returns the initial-probs of the HMM-states... note, these initial-probs
  // don't mean initial at the start of the file, because we usually train
  // on pieces of a file.  They are approximate initial-probs obtained
  // by running the HMM for a fixed number of iters from a flat start.  The
  // exact values won't be very critical.
  const CuVector<BaseFloat> &InitialProbs() const;

 private:
  // functions called from the constructor
  void SetTransitions(const CctcTransitionModel &trans_mdl);
  void SetInitialProbs(const CctcTransitionModel &trans_mdl);

  // forward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions out of this state.
  CuArray<Int32Pair> forward_transitions_;
  // backward_transitions_ is an array, indexed by hmm-state index,
  // of start and end indexes into the transition_ array, which
  // give us the set of transitions out of this state.
  CuArray<Int32Pair> backward_transitions_;
  // This stores the actual transitions.
  CuArray<CctcHmmTransition> transitions_;

  // The initial-probability of all states, used on the first frame of a
  // sequence.  Because in general sequences won't start at the start of files,
  // we make this a generic probability distribution close to the limiting
  // distribution of the HMM.  This isn't too critical.
  CuVector<BaseFloat> initial_probs_;
};




// This header relates to the 'tombstone' extension of CTC, and contains
// utilities for efficient forward-backward over the entire model
// (all word sequences), which becomes a negated term in the objective
// function (like the denominator lattice in MMI training).

// This class is supposed to be initialized just once and then used repeatedly,
// as it needs to do some startup work.  This class supports both CPU and GPU
// versions of the computation, and it uses the GPU is you have initialized the
// device (however, the CPU one will be very slow).
class CctcNegativeComputation {
 public:
  // note: num_sequences is the number of egs that have been combined into a
  // single eg (which must all be of the same size), which will be the number of
  // distinct values of 'n' in the output indexes.  All must have the same
  // number of frames, and we assume that we're sorted first on n and then on t,
  // since that's the way the positive computation requires them to be.
  CctcNegativeComputation(const CctcTransitionModel &trans_model,
                          const CuMatrix<BaseFloat> &cu_weights,
                          const CctcHmm &hmm,
                          const CuMatrixBase<BaseFloat> &exp_nnet_output,
                          const CuMatrixBase<BaseFloat> &denominators,
                          int32 num_sequences);
  // Does the forward computation, and returns the total negated log-like summed
  // over all sequences.
  BaseFloat Forward();

  // It *sets* nnet_output_deriv and denominators_deriv.  These
  // are the derivatives w.r.t. the likelihood, which gets negated in the
  // overall objective function.
  // Note: 'nnet_output_deriv' is the deriv w.r.t. the log of
  // 'exp_nnet_output_deriv', i.e the log of the numerators.
  // Note, you may want to scale them (e.g. by -1) afterward; that is the
  // responsibility of calling code.
  void Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv,
                CuMatrixBase<BaseFloat> *denominators_deriv);


private:
  const CctcTransitionModel &trans_model_;
  const CuMatrix<BaseFloat> &cu_weights_;
  const CctcHmm &hmm_;
  const CuMatrixBase<BaseFloat> &exp_nnet_output_;
  const CuMatrixBase<BaseFloat> &denominators_;
  int32 num_sequences_;
  int32 num_time_steps_;
  int32 numerator_dim_;  // == trans_model_.NumTreeIndexes() +
                         // trans_model_.NumBlankIndexes();
  int32 num_hmm_states_;  // == trans_model_.NumHistoryStates().


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

  // the numerator-probs (== part of exp-nnet-output) rearranged.  dimension
  // is num-time-steps by (partial-nnet-output-dim * num-sequences.)
  // partial-nnet-output-dim is trans_model_.NumTreeIndexes() plus
  // trans_model_.NumHistoryStates().
  CuMatrix<BaseFloat> numerators_rearranged_;
  // the denominator probs rearranged.  dimension is
  // num-time-steps by (num-hmm-states * num-sequences).
  CuMatrix<BaseFloat> denominators_rearranged_;

  // the derivs w.r.t. the log of the numerator-probs, rearranged.
  CuMatrix<BaseFloat> log_numerator_derivs_rearranged_;
  // the derivs w.r.t. the denominator-probs, rearranged.
  CuMatrix<BaseFloat> denominator_derivs_rearranged_;

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
  // correction term for each sequence, which is the product of the alpha_0 over
  // all the frames.
  CuVector<BaseFloat> log_correction_term_;

};


}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_TRAINING_H_


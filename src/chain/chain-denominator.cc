// chain/chain-denominator.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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


#include "chain/chain-denominator.h"
#include "chain/chain-kernels-ansi.h"

namespace kaldi {
namespace chain {

DenominatorComputation::DenominatorComputation(
    const ChainTrainingOptions &opts,
    const DenominatorGraph &den_graph,
    int32 num_sequences,
    const CuMatrixBase<BaseFloat> &nnet_output):
    opts_(opts),
    den_graph_(den_graph),
    num_sequences_(num_sequences),
    frames_per_sequence_(nnet_output.NumRows() / num_sequences_),
    nnet_output_deriv_transposed_(
        nnet_output.NumCols(),
        std::min<int32>(nnet_output.NumRows(),
                        static_cast<int32>(kMaxDerivTimeSteps) *
                        num_sequences_)),
    alpha_(frames_per_sequence_ + 1,
           den_graph_.NumStates() * num_sequences_ + num_sequences_,
           kUndefined),
    beta_(2, den_graph_.NumStates() * num_sequences_ + num_sequences_,
          kUndefined),
    tot_prob_(num_sequences_, kUndefined),
    tot_log_prob_(num_sequences_, kUndefined),
    log_correction_term_(num_sequences_, kUndefined),
    ok_(true) {
  KALDI_ASSERT(opts_.leaky_hmm_coefficient > 0.0 &&
               opts_.leaky_hmm_coefficient < 1.0);
  // make sure the alpha sums and beta sums are zeroed.
  alpha_.ColRange(den_graph_.NumStates() * num_sequences_,
                  num_sequences_).SetZero();
  beta_.ColRange(den_graph_.NumStates() * num_sequences_,
                 num_sequences_).SetZero();

  KALDI_ASSERT(nnet_output.NumRows() % num_sequences == 0);
  // the kStrideEqualNumCols argument means we'll allocate a contiguous block of
  // memory for this; it is added to ensure that the same block of memory
  // (cached in the allocator) can be used for xent_output_deriv when allocated
  // from chain-training.cc.
  exp_nnet_output_transposed_.Resize(nnet_output.NumCols(),
                                     nnet_output.NumRows(),
                                     kUndefined, kStrideEqualNumCols);
  exp_nnet_output_transposed_.CopyFromMat(nnet_output, kTrans);
  // We limit the nnet output to the range [-30,30] before doing the exp;
  // this avoids NaNs appearing in the forward-backward computation, which
  // is not done in log space.
  exp_nnet_output_transposed_.ApplyExpLimited(-30.0, 30.0);
}


void DenominatorComputation::AlphaFirstFrame() {
  // dim == num_hmm_states_ * num_sequences_.
  BaseFloat *first_frame_alpha = alpha_.RowData(0);
  // create a 'fake matrix' - view this row as a matrix.
  // initializer takes [pointer, num-rows, num-cols, stride].
  CuSubMatrix<BaseFloat> alpha_mat(first_frame_alpha,
                                   den_graph_.NumStates(),
                                   num_sequences_,
                                   num_sequences_);
  // TODO (possible): It would be more efficient here if we implemented a
  // CopyColsFromVec function in class CuMatrix.
  alpha_mat.SetZero();
  alpha_mat.AddVecToCols(1.0, den_graph_.InitialProbs(), 0.0);
}


// the alpha computation for some 0 < t <= num_time_steps_.
void DenominatorComputation::AlphaGeneralFrame(int32 t) {
  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BaseFloat *this_alpha = alpha_.RowData(t);
  const BaseFloat *prev_alpha_dash = alpha_.RowData(t - 1);
  const Int32Pair *backward_transitions = den_graph_.BackwardTransitions();
  const DenominatorGraphTransition *transitions = den_graph_.Transitions();
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
      num_hmm_states = den_graph_.NumStates(),
      num_sequences = num_sequences_;

  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               (t-1) * num_sequences_, num_sequences_);
  const BaseFloat *prob_data = probs.Data();

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);

    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      cuda_chain_hmm_forward(dimGrid, dimBlock,
                             backward_transitions, transitions,
                             num_sequences, den_graph_.NumStates(),
                             prob_data, probs.Stride(), prev_alpha_dash,
                             this_alpha);
      CU_SAFE_CALL(cudaGetLastError());
      if (dimGrid.y == num_hmm_states) {
        break;  // this is the normal case.
      } else {
        // We reach this code only in the unusual case where num_hmm_states >
        // 65535.  We can compute the alphas for the remaining HMM states by
        // moving some of the array pointers and making the call again.
        backward_transitions += dimGrid.y;
        this_alpha += dimGrid.y * num_sequences;
        num_hmm_states -= dimGrid.y;
        dimGrid.y = num_hmm_states;
      }
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    int32 prob_stride = probs.Stride();
    for (int32 h = 0; h < num_hmm_states; h++) {
      for (int32 s = 0; s < num_sequences; s++) {
        double this_tot_alpha = 0.0;
        const DenominatorGraphTransition
            *trans_iter = transitions + backward_transitions[h].first,
            *trans_end = transitions + backward_transitions[h].second;
        for (; trans_iter != trans_end; ++trans_iter) {
          BaseFloat transition_prob = trans_iter->transition_prob;
          int32 pdf_id = trans_iter->pdf_id,
              prev_hmm_state = trans_iter->hmm_state;
          BaseFloat prob = prob_data[pdf_id * prob_stride + s],
              this_prev_alpha = prev_alpha_dash[prev_hmm_state * num_sequences + s];
          this_tot_alpha += this_prev_alpha * transition_prob * prob;
        }
        // Let arbitrary_scale be the inverse of the alpha-sum value that we
        // store in the same place we'd store the alpha for the state numbered
        // 'num_hmm_states'. We multiply this into all the
        // transition-probabilities from the previous frame to this frame, in
        // both the forward and backward passes, in order to keep the alphas in
        // a good numeric range.  This won't affect the posteriors, but when
        // computing the total likelihood we'll need to compensate for it later
        // on.
        BaseFloat arbitrary_scale =
            1.0 / prev_alpha_dash[num_hmm_states * num_sequences + s];
        KALDI_ASSERT(this_tot_alpha - this_tot_alpha == 0);
        this_alpha[h * num_sequences + s] = this_tot_alpha * arbitrary_scale;
      }
    }
  }
}

void DenominatorComputation::AlphaDash(int32 t) {
  BaseFloat *this_alpha = alpha_.RowData(t);

  // create a 'fake matrix' for the regular alphas- view this row as a matrix.
  // initializer takes [pointer, num-rows, num-cols, stride].
  CuSubMatrix<BaseFloat> alpha_mat(this_alpha,
                                   den_graph_.NumStates(),
                                   num_sequences_,
                                   num_sequences_);

  // the alpha-dash is the sum of alpha over all states.
  CuSubVector<BaseFloat> alpha_sum_vec(this_alpha +
                                       den_graph_.NumStates() * num_sequences_,
                                       num_sequences_);
  alpha_sum_vec.AddRowSumMat(1.0, alpha_mat, 0.0);

  alpha_mat.AddVecVec(opts_.leaky_hmm_coefficient,
                      den_graph_.InitialProbs(),
                      alpha_sum_vec);
  // it's now alpha-dash.
}

// compute beta from beta-dash.
void DenominatorComputation::Beta(int32 t) {
  BaseFloat *this_beta_dash = beta_.RowData(t % 2);
  // create a 'fake matrix' for the regular beta-dash (which is
  // the counterpart of alpha-dash)- view this row as a matrix.
  // initializer takes [pointer, num-rows, num-cols, stride].
  CuSubMatrix<BaseFloat> beta_dash_mat(this_beta_dash,
                                       den_graph_.NumStates(),
                                       num_sequences_,
                                       num_sequences_);
  // making the t index implicit, the beta-dash-sum for each sequence is the sum
  // over all states i of beta_i * opts_.leaky_hmm_coefficient * initial_prob_i.
  CuSubVector<BaseFloat> beta_dash_sum_vec(
      this_beta_dash + den_graph_.NumStates() * num_sequences_,
      num_sequences_);
  beta_dash_sum_vec.AddMatVec(opts_.leaky_hmm_coefficient, beta_dash_mat,
                              kTrans, den_graph_.InitialProbs(), 0.0);
  // we are computing beta in place.  After the following, beta-dash-mat
  // will contain the actual beta (i.e. the counterpart of alpha),
  // not the beta-dash.
  beta_dash_mat.AddVecToRows(1.0, beta_dash_sum_vec);
}

BaseFloat DenominatorComputation::Forward() {
  AlphaFirstFrame();
  AlphaDash(0);
  for (int32 t = 1; t <= frames_per_sequence_; t++) {
    AlphaGeneralFrame(t);
    AlphaDash(t);
  }
  return ComputeTotLogLike();
}

BaseFloat DenominatorComputation::ComputeTotLogLike() {
  tot_prob_.Resize(num_sequences_);
  // View the last alpha-dash as a matrix of size num-hmm-states by num-sequences.
  CuSubMatrix<BaseFloat> last_alpha_dash(
      alpha_.RowData(frames_per_sequence_),
      den_graph_.NumStates(),
      num_sequences_,
      num_sequences_);

  tot_prob_.AddRowSumMat(1.0, last_alpha_dash, 0.0);
  // we should probably add an ApplyLog() function that takes a vector argument.
  tot_log_prob_ = tot_prob_;
  tot_log_prob_.ApplyLog();
  BaseFloat tot_log_prob = tot_log_prob_.Sum();

  // We now have to add something for the arbitrary scaling factor.  [note: the
  // purpose of the arbitrary scaling factors was to keep things in a good
  // floating-point range]
  // The inverses of all the tot-alpha quantities, for t = 0
  // ... frames_per_sequence_ - 1, were included as the 'arbitrary factors' in
  // the transition-probs, so we need to multiply them all together (not
  // inversed) and add them as a correction term to the total log-likes.
  // These tot-alpha quantities were stored in the same place that we would
  // have stored the HMM-state numbered 'num_hmm_states'.
  int32 num_hmm_states = den_graph_.NumStates();
  CuSubMatrix<BaseFloat> inv_arbitrary_scales(
      alpha_, 0, frames_per_sequence_,
      num_sequences_ * num_hmm_states, num_sequences_);
  CuMatrix<BaseFloat> log_inv_arbitrary_scales(
      inv_arbitrary_scales);
  log_inv_arbitrary_scales.ApplyLog();
  BaseFloat log_inv_arbitrary_scales_product =
      log_inv_arbitrary_scales.Sum();
  return tot_log_prob + log_inv_arbitrary_scales_product;
}



bool DenominatorComputation::Backward(
    BaseFloat deriv_weight,
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  BetaDashLastFrame();
  Beta(frames_per_sequence_);
  for (int32 t = frames_per_sequence_ - 1; t >= 0; t--) {
    BetaDashGeneralFrame(t);
    if (GetVerboseLevel() >= 1 || t == 0)
      BetaGeneralFrameDebug(t);
    Beta(t);
    if (t % kMaxDerivTimeSteps == 0) {
      // commit the derivative stored in nnet_output_deriv_transposed_ by adding
      // its transpose to the appropriate sub-matrix of 'nnet_output_deriv'.
      int32 chunk_frames = std::min<int32>(static_cast<int32>(kMaxDerivTimeSteps),
                                           frames_per_sequence_ - t),
                num_pdfs = exp_nnet_output_transposed_.NumRows();
      CuSubMatrix<BaseFloat> transposed_deriv_part(
          nnet_output_deriv_transposed_,
          0, num_pdfs,
          0, chunk_frames * num_sequences_);
      CuSubMatrix<BaseFloat> output_deriv_part(
          *nnet_output_deriv,
          t * num_sequences_, chunk_frames * num_sequences_,
          0, num_pdfs);
      output_deriv_part.AddMat(deriv_weight, transposed_deriv_part, kTrans);
      if (t != 0)
        transposed_deriv_part.SetZero();
    }
  }
  return ok_;
}

void DenominatorComputation::BetaDashLastFrame() {
  // sets up the beta-dash quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.

  int32 t = frames_per_sequence_;
  BaseFloat *last_frame_beta_dash = beta_.RowData(t % 2);

  // create a 'fake matrix' - view this row as a matrix.
  CuSubMatrix<BaseFloat> beta_dash_mat(last_frame_beta_dash,
                                       den_graph_.NumStates(),
                                       num_sequences_,
                                       num_sequences_);
  CuVector<BaseFloat> inv_tot_prob(tot_prob_);
  inv_tot_prob.InvertElements();
  // the beta values at the end of the file only vary with the sequence-index,
  // not with the HMM-index.  We treat all states as having a final-prob of one.
  beta_dash_mat.CopyRowsFromVec(inv_tot_prob);
}

void DenominatorComputation::BetaDashGeneralFrame(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  // t_wrapped gives us the time-index we use when indexing
  // nnet_output_deriv_transposed_; to save memory we limit the size of the
  // matrix, storing only chunks of frames at a time, and we add it to the
  // non-transposed output whenever we finish a chunk.
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);
  const BaseFloat *this_alpha_dash = alpha_.RowData(t),
      *next_beta = beta_.RowData((t + 1) % 2);
  BaseFloat *this_beta_dash = beta_.RowData(t % 2);
  const Int32Pair *forward_transitions = den_graph_.ForwardTransitions();
  const DenominatorGraphTransition *transitions = den_graph_.Transitions();
  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               t * num_sequences_, num_sequences_),
      log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                     t_wrapped * num_sequences_, num_sequences_);

  int32 num_hmm_states = den_graph_.NumStates(),
      num_sequences = num_sequences_;

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);
    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      cuda_chain_hmm_backward(dimGrid, dimBlock, forward_transitions, transitions,
                              num_sequences, num_hmm_states,
                              probs.Data(), probs.Stride(),
                              this_alpha_dash, next_beta, this_beta_dash,
                              log_prob_deriv.Data(), log_prob_deriv.Stride());
      CU_SAFE_CALL(cudaGetLastError());
      if (dimGrid.y == num_hmm_states) {
        break;  // this is the normal case.
      } else {
        // We reach this code only in the unusual case where num_hmm_states >
        // 65535.  We can compute the betas (and log-prob derivatives) for the
        // remaining HMM states by moving some of the array pointers and making
        // the call again.
        forward_transitions += dimGrid.y;
        this_alpha_dash += dimGrid.y * num_sequences;
        this_beta_dash += dimGrid.y * num_sequences;
        num_hmm_states -= dimGrid.y;
        dimGrid.y = num_hmm_states;
      }
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    int32 prob_stride = probs.Stride(),
         deriv_stride = log_prob_deriv.Stride();
    const BaseFloat *prob_data = probs.Data();
    BaseFloat *log_prob_deriv_data = log_prob_deriv.Data();
    for (int32 h = 0; h < num_hmm_states; h++) {
      for (int32 s = 0; s < num_sequences; s++) {
        BaseFloat this_alpha_dash_prob = this_alpha_dash[h * num_sequences + s],
            inv_arbitrary_scale =
            this_alpha_dash[num_hmm_states * num_sequences + s];
        double tot_variable_factor = 0.0;
        BaseFloat occupation_factor = this_alpha_dash_prob /
            inv_arbitrary_scale;
        const DenominatorGraphTransition
            *trans_iter = transitions + forward_transitions[h].first,
            *trans_end = transitions + forward_transitions[h].second;
        for (; trans_iter != trans_end; ++trans_iter) {
          BaseFloat transition_prob = trans_iter->transition_prob;
          int32 pdf_id = trans_iter->pdf_id,
              next_hmm_state = trans_iter->hmm_state;
          BaseFloat variable_factor = transition_prob *
              next_beta[next_hmm_state * num_sequences + s] *
              prob_data[pdf_id * prob_stride + s];
          tot_variable_factor += variable_factor;
          BaseFloat occupation_prob = variable_factor * occupation_factor;
          log_prob_deriv_data[pdf_id * deriv_stride + s] += occupation_prob;
        }
        this_beta_dash[h * num_sequences + s] =
            tot_variable_factor / inv_arbitrary_scale;
      }
    }
  }
}

void DenominatorComputation::BetaGeneralFrameDebug(int32 t) {
  BaseFloat num_hmm_states = den_graph_.NumStates(),
      alpha_beta_size = num_hmm_states * num_sequences_;
  CuSubVector<BaseFloat> this_alpha_dash(alpha_.RowData(t), alpha_beta_size),
      this_beta_dash(beta_.RowData(t % 2), alpha_beta_size);
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps),
      num_pdfs = exp_nnet_output_transposed_.NumRows();
  CuSubMatrix<BaseFloat> this_log_prob_deriv(
      nnet_output_deriv_transposed_, 0, num_pdfs,
      t_wrapped * num_sequences_, num_sequences_);
  BaseFloat alpha_beta_product = VecVec(this_alpha_dash,
                                        this_beta_dash),
      this_log_prob_deriv_sum = this_log_prob_deriv.Sum();
  if (!ApproxEqual(alpha_beta_product, num_sequences_)) {
    KALDI_WARN << "On time " << t << ", alpha-beta product "
               << alpha_beta_product << " != " << num_sequences_
               << " alpha-dash-sum = " << this_alpha_dash.Sum()
               << ", beta-dash-sum = " << this_beta_dash.Sum();
    if (fabs(alpha_beta_product - num_sequences_) > 2.0) {
      KALDI_WARN << "Excessive error detected, will abandon this minibatch";
      ok_ = false;
    }
  }
  // use higher tolerance, since we are using randomized pruning for the
  // log-prob derivatives.
  if (!ApproxEqual(this_log_prob_deriv_sum,
                   num_sequences_, 0.01)) {
    KALDI_WARN << "On time " << t << ", log-prob-deriv sum "
               << this_log_prob_deriv_sum << " != " << num_sequences_;
    if (fabs(this_log_prob_deriv_sum - num_sequences_) > 2.0) {
      KALDI_WARN << "Excessive error detected, will abandon this minibatch";
      ok_ = false;
    }
  }
}


}  // namespace chain
}  // namespace kaldi

// chain/chain-cu-numerator.cc

// Copyright      2015   Hossein Hadian

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


#include "chain/chain-cu-numerator.h"
#include "chain/chain-kernels-ansi.h"

namespace kaldi {
namespace chain {

CuNumeratorComputation::CuNumeratorComputation(
                                    const ChainTrainingOptions &opts,
                                    const NumeratorGraph &num_graph,
                                    const CuMatrixBase<BaseFloat> &nnet_output):
    opts_(opts),
    num_graph_(num_graph),
    num_sequences_(num_graph.NumSequences()),
    frames_per_sequence_(nnet_output.NumRows() / num_sequences_),
    exp_nnet_output_transposed_(nnet_output, kTrans),
    nnet_output_deriv_transposed_(
        exp_nnet_output_transposed_.NumRows(),
        exp_nnet_output_transposed_.NumCols()),
    alpha_(frames_per_sequence_ + 1,
           num_graph_.MaxNumStates() * num_sequences_ + num_sequences_,
           kSetZero),
    // we actually do not need beta for state num_graph_.MaxNumStates(),
    // so no "+ num_sequences_"
    beta_(2, num_graph_.MaxNumStates() * num_sequences_,
          kSetZero),
    tot_prob_(num_sequences_, kUndefined),
    tot_log_prob_(num_sequences_, kUndefined),
    ok_(true) {
  //KALDI_ASSERT(opts_.leaky_hmm_coefficient > 0.0 &&
  //             opts_.leaky_hmm_coefficient < 1.0);

  KALDI_ASSERT(nnet_output.NumRows() % num_sequences_ == 0);
  exp_nnet_output_transposed_.ApplyExp();
}


void CuNumeratorComputation::AlphaFirstFrame() {
  // select alpha for time 0
  BaseFloat *first_frame_alpha = alpha_.RowData(0);
  // now make a view of the first num_sequences elements (i.e. alpha_0(0)
  // for all sequences)
  // initializer takes [pointer, length].
  CuSubVector<BaseFloat> alpha_hmm_state0(first_frame_alpha, num_sequences_);
  // set alpha_0(0) for all sequences to 1.0 and leave the rest to be 0.0.
  // i.e. the only start state is state 0.
  alpha_hmm_state0.Set(1.0);

  // Now compute alpha-sums for t==0 which is obviously 1.0 for each sequence
  CuSubVector<BaseFloat> alpha_sum_vec(
                                     first_frame_alpha +
                                     num_graph_.MaxNumStates() * num_sequences_,
                                     num_sequences_);
  alpha_sum_vec.Set(1.0);
}


// the alpha computation for some 0 < t <= num_time_steps_.
void CuNumeratorComputation::AlphaGeneralFrame(int32 t) {
  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BaseFloat *this_alpha = alpha_.RowData(t);
  const BaseFloat *prev_alpha = alpha_.RowData(t - 1);
  const Int32Pair *backward_transitions = num_graph_.BackwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
      max_num_hmm_states = num_graph_.MaxNumStates(),
      num_sequences = num_sequences_;

  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               (t - 1) * num_sequences_, num_sequences_);
  const BaseFloat *prob_data = probs.Data();

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), max_num_hmm_states, 1);

    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      cuda_chain_num_hmm_forward(dimGrid, dimBlock,
                             backward_transitions, transitions,
                             num_sequences, num_graph_.NumStates(),
                             max_num_hmm_states,
                             prob_data, probs.Stride(), prev_alpha,
                             this_alpha);
      CU_SAFE_CALL(cudaGetLastError());
      if (dimGrid.y == max_num_hmm_states) {
        break;  // this is the normal case.
      } else {
        KALDI_ERR << "Not supported yet.\n";
      }
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    int32 prob_stride = probs.Stride();
    for (int32 s = 0; s < num_sequences; s++) {
      for (int32 h = 0; h < num_graph_.NumStates()[s]; h++) {
        double this_tot_alpha = 0.0;
        const DenominatorGraphTransition
            *trans_iter = transitions +
              backward_transitions[s*max_num_hmm_states+h].first,
            *trans_end = transitions +
              backward_transitions[s*max_num_hmm_states+h].second;
        for (; trans_iter != trans_end; ++trans_iter) {
          BaseFloat transition_prob = trans_iter->transition_prob;
          int32 pdf_id = trans_iter->pdf_id,
                prev_hmm_state = trans_iter->hmm_state;
          BaseFloat prob = prob_data[pdf_id * prob_stride + s],
              this_prev_alpha = prev_alpha[prev_hmm_state * num_sequences + s];
          this_tot_alpha += this_prev_alpha * transition_prob * prob;
        }
        // Let arbitrary_scale be the inverse of the alpha-sum value that we
        // store in the same place we'd store the alpha for the state numbered
        // 'max_num_hmm_states'. We multiply this into all the
        // transition-probabilities from the previous frame to this frame, in
        // both the forward and backward passes, in order to keep the alphas in
        // a good numeric range.  This won't affect the posteriors, but when
        // computing the total likelihood we'll need to compensate for it later
        // on.
        BaseFloat arbitrary_scale =
            1.0 / prev_alpha[max_num_hmm_states * num_sequences + s];
        KALDI_ASSERT(this_tot_alpha - this_tot_alpha == 0);
        this_alpha[h * num_sequences + s] = this_tot_alpha * arbitrary_scale;
      }
    }
  }

  // Now compute alpha-sums for frame t:
  CuSubMatrix<BaseFloat> alpha_mat(this_alpha,
                                   num_graph_.MaxNumStates(),
                                   num_sequences_,
                                   num_sequences_);
  CuSubVector<BaseFloat> alpha_sum_vec(this_alpha +
                                     num_graph_.MaxNumStates() * num_sequences_,
                                     num_sequences_);
  alpha_sum_vec.AddRowSumMat(1.0, alpha_mat, 0.0);
}

BaseFloat CuNumeratorComputation::Forward() {
  AlphaFirstFrame();
  for (int32 t = 1; t <= frames_per_sequence_; t++) {
    AlphaGeneralFrame(t);
  }
  return ComputeTotLogLike();
}

BaseFloat CuNumeratorComputation::ComputeTotLogLike() {
  tot_prob_.Resize(num_sequences_);
  // View the last alpha as a matrix of size num-hmm-states by num-sequences.
  CuSubMatrix<BaseFloat> last_alpha(
      alpha_.RowData(frames_per_sequence_),
      num_graph_.MaxNumStates(),
      num_sequences_,
      num_sequences_);

  tot_prob_.AddRowSumMat(1.0, last_alpha, 0.0);
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
  // have stored the HMM-state numbered 'max_num_hmm_states'.
  int32 max_num_hmm_states = num_graph_.MaxNumStates();
  CuSubMatrix<BaseFloat> inv_arbitrary_scales(
      alpha_, 0, frames_per_sequence_,
      num_sequences_ * max_num_hmm_states, num_sequences_);
  CuMatrix<BaseFloat> log_inv_arbitrary_scales(
      inv_arbitrary_scales);
  log_inv_arbitrary_scales.ApplyLog();
  BaseFloat log_inv_arbitrary_scales_product =
      log_inv_arbitrary_scales.Sum();
  return tot_log_prob + log_inv_arbitrary_scales_product;
}


bool CuNumeratorComputation::Backward(
    BaseFloat deriv_weight,
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  BetaLastFrame();
  for (int32 t = frames_per_sequence_ - 1; t >= 0; t--) {
    BetaGeneralFrame(t);
    if (GetVerboseLevel() >= 1 || t == 0)
      BetaGeneralFrameDebug(t);
  }
  nnet_output_deriv->AddMat(
                           deriv_weight, nnet_output_deriv_transposed_, kTrans);
  return ok_;
}

void CuNumeratorComputation::BetaLastFrame() {
  // sets up the beta quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.

  int32 t = frames_per_sequence_;
  BaseFloat *last_frame_beta = beta_.RowData(t % 2);

  // create a 'fake matrix' - view this row as a matrix.
  CuSubMatrix<BaseFloat> beta_mat(last_frame_beta,
                                  num_graph_.MaxNumStates(),
                                  num_sequences_,
                                  num_sequences_);

  // There is only 1 final state in each sequence's HMM, and its prob is 1.0
  // Please refer to chain-supervision.h,cc for more info
  // since final state indexes are different for each sequence, we set them in
  // a for loop.
  int32 *num_states_cpu = new int32[num_graph_.NumSequences()];
  num_graph_.CopyNumStatesToCpu(num_states_cpu);
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    int32 final_state = num_states_cpu[seq] - 1;
    beta_mat(final_state, seq) = 1.0 / tot_prob_(seq);
  }
  delete num_states_cpu;
}

void CuNumeratorComputation::BetaGeneralFrame(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  const BaseFloat *this_alpha = alpha_.RowData(t),
                  *next_beta = beta_.RowData((t + 1) % 2);
  BaseFloat *this_beta = beta_.RowData(t % 2);
  const Int32Pair *forward_transitions = num_graph_.ForwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               t * num_sequences_, num_sequences_),
      log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                     t * num_sequences_, num_sequences_);

  int32 max_num_hmm_states = num_graph_.MaxNumStates(),
        num_sequences = num_sequences_;

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), max_num_hmm_states, 1);
    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      cuda_chain_num_hmm_backward(dimGrid, dimBlock, forward_transitions,
                              transitions,
                              num_sequences, num_graph_.NumStates(),
                              max_num_hmm_states,
                              probs.Data(), probs.Stride(),
                              this_alpha, next_beta, this_beta,
                              log_prob_deriv.Data(), log_prob_deriv.Stride());
      CU_SAFE_CALL(cudaGetLastError());
      if (dimGrid.y == max_num_hmm_states) {
        break;  // this is the normal case.
      } else {
        KALDI_ERR << "Not supported yet.\n";
      }
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    int32 prob_stride = probs.Stride(),
          deriv_stride = log_prob_deriv.Stride();
    const BaseFloat *prob_data = probs.Data();
    BaseFloat *log_prob_deriv_data = log_prob_deriv.Data();
    for (int32 s = 0; s < num_sequences; s++) {
      for (int32 h = 0; h < num_graph_.NumStates()[s]; h++) {
        BaseFloat this_alpha_prob = this_alpha[h * num_sequences + s],
            inv_arbitrary_scale =
            this_alpha[max_num_hmm_states * num_sequences + s];
        double tot_variable_factor = 0.0;
        BaseFloat occupation_factor = this_alpha_prob /
            inv_arbitrary_scale;
        const DenominatorGraphTransition
            *trans_iter = transitions +
              forward_transitions[s*max_num_hmm_states + h].first,
            *trans_end = transitions +
              forward_transitions[s*max_num_hmm_states + h].second;
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
        this_beta[h * num_sequences + s] =
            tot_variable_factor / inv_arbitrary_scale;
      }
    }
  }
}

void CuNumeratorComputation::BetaGeneralFrameDebug(int32 t) {
  BaseFloat max_num_hmm_states = num_graph_.MaxNumStates(),
      alpha_beta_size = max_num_hmm_states * num_sequences_;
  CuSubVector<BaseFloat> this_alpha(alpha_.RowData(t), alpha_beta_size),
      this_beta(beta_.RowData(t % 2), alpha_beta_size);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  CuSubMatrix<BaseFloat> this_log_prob_deriv(
      nnet_output_deriv_transposed_, 0, num_pdfs,
      t * num_sequences_, num_sequences_);
  BaseFloat alpha_beta_product = VecVec(this_alpha,
                                        this_beta),
      this_log_prob_deriv_sum = this_log_prob_deriv.Sum();
  if (!ApproxEqual(alpha_beta_product, num_sequences_)) {
    KALDI_WARN << "On time " << t << ", alpha-beta product "
               << alpha_beta_product << " != " << num_sequences_
               << " alpha-sum = " << this_alpha.Sum()
               << ", beta-sum = " << this_beta.Sum();
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

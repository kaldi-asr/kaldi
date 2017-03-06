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


#include "chain/chain-cu-leakynum.h"
#include "chain/chain-kernels-ansi.h"

namespace kaldi {
namespace chain {

CuLeakyNumeratorComputation::CuLeakyNumeratorComputation(
                                    const ChainTrainingOptions &opts,
                                    const NumeratorGraph &num_graph,
                                    const DenominatorGraph &den_graph,
                                    const CuMatrixBase<BaseFloat> &nnet_output):
    opts_(opts),
    num_graph_(num_graph),
    den_graph_(den_graph),
    num_sequences_(num_graph.NumSequences()),
    frames_per_sequence_(nnet_output.NumRows() / num_sequences_),
    exp_nnet_output_transposed_(nnet_output, kTrans),
    nnet_output_deriv_transposed_(
        exp_nnet_output_transposed_.NumRows(),
        std::min<int32>(exp_nnet_output_transposed_.NumCols(),
                        static_cast<int32>(kMaxDerivTimeSteps) *
                        num_sequences_)),

    alpha_num_(frames_per_sequence_ + 1,
           (num_graph.MaxNumStates() + 3) * num_sequences_,
           kSetZero),
    // +3 breakdown: location num_graph.MaxNumStates() is for alpha-sums
    //               location num_graph.MaxNumStates()+1 is for alpha-hats
    //               location num_graph.MaxNumStates()+2 is for alpha-primes


    alpha_den_(frames_per_sequence_ + 1,
           (den_graph.NumStates() + 1) * num_sequences_,
           kSetZero),
    // location den_graph.NumStates() is for alpha-sums (mirror the above ones
    // -- this is to avoid changing the kernels)


    beta_num_(2, (num_graph.MaxNumStates() + 1) * num_sequences_,
          kSetZero),
    // location num_graph.MaxNumStates() is for beta-primes and beta-hats


    beta_den_(2, den_graph.NumStates() * num_sequences_,
          kSetZero),
    tot_prob_(num_sequences_, kUndefined),
    tot_log_prob_(num_sequences_, kUndefined),
    ok_(true) {
  KALDI_ASSERT(opts_.leakynum_leak_prob >= 0.0 &&
               opts_.leakynum_leak_prob < 1.0);
  KALDI_ASSERT(opts_.leakynum_unleak_prob >= 0.0 &&
               opts_.leakynum_unleak_prob < 1.0);

  KALDI_ASSERT(nnet_output.NumRows() % num_sequences_ == 0);
  exp_nnet_output_transposed_.ApplyExp();



  BaseFloat tot_leak_den = 0.0;
  if (opts_.leakynum_use_priors)
    tot_leak_den = den_graph_.InitialProbs().Sum(); // must be 1.0
  else
    tot_leak_den = den_graph_.InitialProbs().Dim();
  leak_eta_ = opts_.leakynum_leak_prob / tot_leak_den / (1.0 - opts_.leakynum_leak_prob);


  Vector<BaseFloat> unleak_etas(num_sequences_);
  for (int32 seq = 0; seq < num_sequences_; seq++)
    unleak_etas(seq) = opts_.leakynum_unleak_prob / num_graph_.GetTotWeightSum(seq) / (1.0 - opts_.leakynum_leak_prob);
  unleak_etas_ = unleak_etas;

  num_transitions_scale_ = 1.0 - opts_.leakynum_leak_prob;
  den_transitions_scale_ = 1.0 - opts_.leakynum_unleak_prob;

  num_graph_.ScaleTransitions(num_transitions_scale_);
  den_graph_.ScaleTransitions(den_transitions_scale_ * opts_.leakynum_extra_den_scale);
  // TODO: if scale first transitions is enabled we should scale the offset
}


void CuLeakyNumeratorComputation::AlphaFirstFrame() {
  // select alpha for time 0
  BaseFloat *first_frame_alpha_num = alpha_num_.RowData(0);
  // now make a view of the first num_sequences elements (i.e. alpha_0(0)
  // for all sequences)
  // initializer takes [pointer, length].
  CuSubVector<BaseFloat> alpha_numhmm_state0(first_frame_alpha_num, num_sequences_);
  // set alpha_num_0(0) for all sequences to 1.0 and leave the rest to be 0.0.
  // i.e. the only start state is state 0 of num graph.
  // note: alpha_den is already 0
  alpha_numhmm_state0.Set(1.0);
}

void CuLeakyNumeratorComputation::AlphaSumAndPrime(int32 t) {
  BaseFloat *this_alpha_num = alpha_num_.RowData(t);
  BaseFloat *this_alpha_den = alpha_den_.RowData(t);

  CuSubMatrix<BaseFloat> alpha_mat_num(this_alpha_num,
                                   num_graph_.MaxNumStates(),
                                   num_sequences_,
                                   num_sequences_);

  CuSubMatrix<BaseFloat> alpha_mat_den(this_alpha_den,
                                   den_graph_.NumStates(),
                                   num_sequences_,
                                   num_sequences_);

  // the alpha-sum is the sum of alpha over all states of den and num graphs.
  // location of storing alpha-sum: at the end of alpha_num_
  // and also a copy at the end of alpha_den_. This is to avoid any changes
  // in the cuda kernels.
  CuSubVector<BaseFloat> alpha_sum_vec(this_alpha_num +
                                       num_graph_.MaxNumStates()
                                       * num_sequences_,
                                       num_sequences_);

  /// note: alpha prime is the sum of alphas for all den states
  CuSubVector<BaseFloat> alpha_prime(this_alpha_num +
                                     (num_graph_.MaxNumStates() + 2)
                                     * num_sequences_,
                                     num_sequences_);

  CuSubVector<BaseFloat> alpha_sum_vec_loc2(this_alpha_den +
                                       den_graph_.NumStates()
                                       * num_sequences_,
                                       num_sequences_);

  /// first sum up den states
  alpha_sum_vec.AddRowSumMat(1.0, alpha_mat_den, 0.0);
  alpha_prime.CopyFromVec(alpha_sum_vec);
  alpha_prime.MulElements(unleak_etas_);

  /// now sum num states too
  alpha_sum_vec.AddRowSumMat(1.0, alpha_mat_num, 1.0);
  alpha_sum_vec_loc2.CopyFromVec(alpha_sum_vec);
}

// the alpha computation for some 0 < t <= num_time_steps_ for numerator states
// the only difference from normal formulas is that alpha(t-1) is first
// added with alpha-prime
void CuLeakyNumeratorComputation::AlphaNumFrame(int32 t) {
  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BaseFloat *this_alpha_num = alpha_num_.RowData(t);
  const BaseFloat *prev_alpha_num = alpha_num_.RowData(t - 1);
  const Int32Pair *backward_transitions = num_graph_.BackwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
                   max_num_hmm_states = num_graph_.MaxNumStates();

  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               (t - 1) * num_sequences_, num_sequences_);
  const BaseFloat *prob_data = probs.Data();

  CuSubVector<BaseFloat> prev_alpha_prime(prev_alpha_num +
                                     (max_num_hmm_states + 2)
                                     * num_sequences_,
                                     num_sequences_);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences_), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences_, dimBlock.x), max_num_hmm_states, 1);

    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      cuda_chain_num_hmm_forward(dimGrid, dimBlock,
                             backward_transitions, transitions,
                             num_sequences_, num_graph_.NumStates(),
                             max_num_hmm_states,
                             prob_data, probs.Stride(), prev_alpha_num,
                             this_alpha_num);
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
    for (int32 s = 0; s < num_sequences_; s++) {
      BaseFloat inv_arbitrary_scale =
          prev_alpha_num[max_num_hmm_states * num_sequences_ + s];
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
              this_prev_alpha_num = prev_alpha_num[prev_hmm_state * num_sequences_ + s];
          this_tot_alpha += (this_prev_alpha_num + prev_alpha_prime(s)) / inv_arbitrary_scale * transition_prob * prob;
          if (this_tot_alpha - this_tot_alpha != 0) {
            KALDI_LOG << "t: " << t << ", seq: " << s << ", h: " << h
            << ", prev-alpha: " << this_prev_alpha_num
            << ", prob: " << prob
            << ", inv_arbitrary_scale: " << inv_arbitrary_scale;
            KALDI_ERR << "Alpha-num failure.";
          }
        }
        KALDI_ASSERT(this_tot_alpha - this_tot_alpha == 0);
        this_alpha_num[h * num_sequences_ + s] = this_tot_alpha ;//* arbitrary_scale; //#SCC#
      }
    }
  }

}

// the alpha computation for some 0 < t <= num_time_steps_ for
// denominator states. This code is exactly like the one in chain-denominator.cc
void CuLeakyNumeratorComputation::AlphaDenFrame(int32 t) {
  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BaseFloat *this_alpha = alpha_den_.RowData(t);
  const BaseFloat *prev_alpha = alpha_den_.RowData(t - 1);
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
    Timer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);

    while (1) {
      if (dimGrid.y > 65535)  // the hardware doesn't allow more than this.
        dimGrid.y = 65535;
      cuda_chain_hmm_forward(dimGrid, dimBlock,
                             backward_transitions, transitions,
                             num_sequences, den_graph_.NumStates(),
                             prob_data, probs.Stride(), prev_alpha,
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
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
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
              this_prev_alpha = prev_alpha[prev_hmm_state * num_sequences + s];
          this_tot_alpha += this_prev_alpha * transition_prob * prob;
        }
        BaseFloat arbitrary_scale =
            1.0 / prev_alpha[num_hmm_states * num_sequences + s];
        KALDI_ASSERT(this_tot_alpha - this_tot_alpha == 0);
        this_alpha[h * num_sequences + s] = this_tot_alpha * arbitrary_scale;
      }
    }
  }
}

// alpha-hat is a sum over all transitions of the num graph -- check the formulas
// after computing alpha-hat
// (which is stored at location num_graph_.MaxNumStates() + 1), the alpha hats
// are added to alphas for den states.

void CuLeakyNumeratorComputation::AlphaHat(int32 t) {

  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BaseFloat *prev_alpha_num = alpha_num_.RowData(t - 1);
  const Int32Pair *backward_transitions = num_graph_.BackwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
      max_num_hmm_states = num_graph_.MaxNumStates();

  CuSubVector<BaseFloat> prev_alpha_hats(prev_alpha_num +
                                         (max_num_hmm_states + 1) * num_sequences_,
                                         num_sequences_);

  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               (t - 1) * num_sequences_, num_sequences_);
  const BaseFloat *prob_data = probs.Data();

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(n_blocks(num_sequences_, CU1DBLOCK));
    cuda_chain_leakynum_alpha_hat(dimGrid, dimBlock, backward_transitions,
                                 transitions,
                                 num_sequences_, num_graph_.NumStates(),
                                 max_num_hmm_states,
                                 probs.Data(), probs.Stride(),
                                 prev_alpha_num);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    int32 prob_stride = probs.Stride();
    for (int32 s = 0; s < num_sequences_; s++) {
      double prev_alpha_hat = 0.0;
      // iterate over all transitions of num graph
      const DenominatorGraphTransition
        *trans_iter = transitions +
            backward_transitions[s*num_graph_.MaxNumStates()+0].first,
        *trans_end = transitions +
            backward_transitions[s*num_graph_.MaxNumStates()+num_graph_.NumStates()[s]-1].second;
      for (; trans_iter != trans_end; ++trans_iter) {
        BaseFloat transition_prob = trans_iter->transition_prob;
        int32 pdf_id = trans_iter->pdf_id,
              prev_hmm_state = trans_iter->hmm_state;
        BaseFloat prob = prob_data[pdf_id * prob_stride + s],
          this_prev_alpha_num = prev_alpha_num[prev_hmm_state * num_sequences_ + s];
        prev_alpha_hat += this_prev_alpha_num * transition_prob * prob;
      }
      BaseFloat arbitrary_scale =
          1.0 / prev_alpha_num[max_num_hmm_states * num_sequences_ + s];
      KALDI_ASSERT(prev_alpha_hat - prev_alpha_hat == 0);
      prev_alpha_hats(s) = prev_alpha_hat * arbitrary_scale;
    }
  }

  BaseFloat *this_alpha_den = alpha_den_.RowData(t);
  CuSubMatrix<BaseFloat> alpha_mat_den(this_alpha_den,
                                       den_graph_.NumStates(),
                                       num_sequences_,
                                       num_sequences_);
  if (opts_.leakynum_use_priors == 1)
    alpha_mat_den.AddVecVec(leak_eta_,
                            den_graph_.InitialProbs(),
                            prev_alpha_hats);
  else
    alpha_mat_den.AddVecToRows(leak_eta_,
                               prev_alpha_hats, 1.0);
}

BaseFloat CuLeakyNumeratorComputation::Forward() {
  AlphaFirstFrame();
  for (int32 t = 1; t <= frames_per_sequence_; t++) {
    AlphaSumAndPrime(t - 1);  // compute alpha-sum(t-1) and alpha-prime_{t-1} (to be added to alpha_{t-1} in alpha_t computation)
    AlphaNumFrame(t);  // compute alpha_t for num states (add alphaprime_{t-1} (scaled by unleak_eta) to alpha_{t-1}'s)
    AlphaDenFrame(t);  // standard compute alpha_t for den states
    AlphaHat(t);  // compute alphahat_t and add it (scaled by leak_eta) to alpha_t for den states
  }
  return ComputeTotLogLike();
}

BaseFloat CuLeakyNumeratorComputation::ComputeTotLogLike() {
  tot_prob_.Resize(num_sequences_);
  // View the last numerator alpha as a matrix of size num-hmm-states by num-sequences.

  CuSubMatrix<BaseFloat> last_alpha_num(
      alpha_num_.RowData(frames_per_sequence_),
      num_graph_.MaxNumStates(),
      num_sequences_,
      num_sequences_);

  int32 num_states_cpu[num_sequences_];
  num_graph_.CopyNumStatesToCpu(num_states_cpu);
  /// tot_prob_(seq) for each seq is the num-alpha(T) for the final state of that graph
  /// that's because the only valid paths are the ones that end up in the final
  /// state of the numerator graph.

  // void Lookup(const std::vector<Int32Pair> &indexes, Real *output) const
  std::vector<Int32Pair> final_state_indexes(num_sequences_);
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    final_state_indexes[seq].first = num_states_cpu[seq] - 1;
    final_state_indexes[seq].second = seq;
  }
  last_alpha_num.Lookup(final_state_indexes, tot_prob_.Data());

  tot_log_prob_ = tot_prob_;
  tot_log_prob_.ApplyLog();
  if (num_graph_.AreFirstTransitionsScaled())
    tot_log_prob_.AddVec(-1.0, num_graph_.FirstTransitionOffsets());
  BaseFloat tot_log_prob = tot_log_prob_.Sum();

  int32 max_num_hmm_states = num_graph_.MaxNumStates();
  CuSubMatrix<BaseFloat> inv_arbitrary_scales(
      alpha_num_, 0, frames_per_sequence_,
      num_sequences_ * max_num_hmm_states, num_sequences_);
  CuMatrix<BaseFloat> log_inv_arbitrary_scales(
      inv_arbitrary_scales);
  log_inv_arbitrary_scales.ApplyLog();
  BaseFloat log_inv_arbitrary_scales_product =
      log_inv_arbitrary_scales.Sum();
  return (tot_log_prob + log_inv_arbitrary_scales_product) * num_graph_.GetSupervisionWeight();
}


bool CuLeakyNumeratorComputation::Backward(
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  BetaLastFrame();
  for (int32 t = frames_per_sequence_ - 1; t >= 0; t--) {
    BetaPrime(t + 1); // Compute Bprime_{t+1}
    BetaNumFrame(t); // backward computation for denominator: B_t (the next betas are increased by Bprime_{t+1})
    BetaDenFrame(t); // standard backward computation for numerator B_t
    BetaHat(t);  // B^{den}_t(i) <-- B_t(i) + Bhat_{t+1}
    if (GetVerboseLevel() >= 1 || t == 0)
      BetaGeneralFrameDebug(t);
    if (t % kMaxDerivTimeSteps == 0) {
      // commit the derivative stored in exp_nnet_output_transposed_ by adding
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
      output_deriv_part.AddMat(num_graph_.GetSupervisionWeight(),
                               transposed_deriv_part, kTrans);
      if (t != 0)
        transposed_deriv_part.SetZero();
    }
  }
  return ok_;
}

void CuLeakyNumeratorComputation::BetaLastFrame() {
  // sets up the beta quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.

  int32 t = frames_per_sequence_;
  BaseFloat *last_frame_beta_num = beta_num_.RowData(t % 2);

  // create a 'fake matrix' - view this row as a matrix.
  CuSubMatrix<BaseFloat> beta_mat_num(last_frame_beta_num,
                                      num_graph_.MaxNumStates(),
                                      num_sequences_,
                                      num_sequences_);

  // There is only 1 final state in each sequence's HMM, and its prob is 1.0
  // Please refer to chain-supervision.h,cc for more info
  // since final state indexes are different for each sequence, we set them in
  // a for loop.
  int32 *num_states_cpu = new int32[num_graph_.NumSequences()];
  num_graph_.CopyNumStatesToCpu(num_states_cpu);  //TODO(hhadian) this might be really slow -- check it
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    int32 final_state = num_states_cpu[seq] - 1;
    beta_mat_num(final_state, seq) = 1.0 / tot_prob_(seq);
  }
  delete num_states_cpu;
}

/// Computes beta-prime for time t
void CuLeakyNumeratorComputation::BetaPrime(int32 t) {
  BaseFloat *this_beta_den = beta_den_.RowData(t % 2);
  CuSubMatrix<BaseFloat> beta_den_mat(this_beta_den,
                                       den_graph_.NumStates(),
                                       num_sequences_,
                                       num_sequences_);

  BaseFloat *this_beta_num = beta_num_.RowData(t % 2);

  CuSubVector<BaseFloat> beta_prime(this_beta_num +
                                   num_graph_.MaxNumStates() * num_sequences_,
                                   num_sequences_);
  if (opts_.leakynum_use_priors == 1)
    beta_prime.AddMatVec(leak_eta_, beta_den_mat,  // IxB --> trans: BxI
                         kTrans, den_graph_.InitialProbs(), 0.0);
  else
    beta_prime.AddRowSumMat(leak_eta_, beta_den_mat, 0.0);
}

/// Computes beta for numerator states.
/// Similar to standard formulas but with the next betas are fist modified a little

void CuLeakyNumeratorComputation::BetaNumFrame(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);
  const BaseFloat *this_alpha_num = alpha_num_.RowData(t),
                  *next_beta_num = beta_num_.RowData((t + 1) % 2);
  BaseFloat *this_beta_num = beta_num_.RowData(t % 2);
  const Int32Pair *forward_transitions = num_graph_.ForwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               t * num_sequences_, num_sequences_),
      log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                     t_wrapped * num_sequences_, num_sequences_);

  int32 max_num_hmm_states = num_graph_.MaxNumStates(),
        num_sequences = num_sequences_;

  CuSubVector<BaseFloat> next_beta_prime(next_beta_num +
                                         num_graph_.MaxNumStates() * num_sequences_,
                                         num_sequences_);

  CuSubVector<BaseFloat> alpha_prime(this_alpha_num +
                                     (num_graph_.MaxNumStates() + 2)
                                     * num_sequences_,
                                     num_sequences_);
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
                              this_alpha_num, next_beta_num, this_beta_num,
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
        BaseFloat this_alpha_prob = this_alpha_num[h * num_sequences + s],
            inv_arbitrary_scale =
            this_alpha_num[max_num_hmm_states * num_sequences + s];
        double tot_variable_factor = 0.0;
        const DenominatorGraphTransition
            *trans_iter = transitions +
              forward_transitions[s*max_num_hmm_states + h].first,
            *trans_end = transitions +
              forward_transitions[s*max_num_hmm_states + h].second;
        for (; trans_iter != trans_end; ++trans_iter) {
          int32 pdf_id = trans_iter->pdf_id,
              next_hmm_state = trans_iter->hmm_state;
          BaseFloat transition_prob = trans_iter->transition_prob,
                    next_beta_prob = next_beta_num[next_hmm_state * num_sequences + s];
          BaseFloat shared_factor = transition_prob * prob_data[pdf_id * prob_stride + s] / inv_arbitrary_scale;
          BaseFloat variable_factor1 = next_beta_prob * shared_factor;
          BaseFloat variable_factor2 = (next_beta_prob + next_beta_prime(s)) *
                                       shared_factor;
          tot_variable_factor += variable_factor2;
          BaseFloat occupation_prob = variable_factor1 * this_alpha_prob;
          log_prob_deriv_data[pdf_id * deriv_stride + s] += occupation_prob;

          /// handle Gamma Leaky Transitions:
          log_prob_deriv_data[pdf_id * deriv_stride + s] += shared_factor * (
             (this_alpha_prob * next_beta_prime(s))    +    (next_beta_prob * alpha_prime(s))  );

          if (tot_variable_factor - tot_variable_factor != 0) {
            KALDI_LOG << "t: " << t << ", seq: " << s << ", h: " << h << ", pdf_id: " << pdf_id
            << ", var-factor: " << variable_factor1
            << ", ocup-prob: " << occupation_prob
            << ", next-beta: " << next_beta_num[next_hmm_state * num_sequences + s]
            << ", this_alpha_prob: " << this_alpha_prob
            << ", prob: " << prob_data[pdf_id * prob_stride + s]
            << ", inv arbitrary_scale: " << inv_arbitrary_scale;
            KALDI_ERR << "Beta failure.";
          }

        }
        this_beta_num[h * num_sequences + s] =
            tot_variable_factor; /// inv_arbitrary_scale; #SCC#
      }
    }
  }
}

/// This code is exactly like BetaGeneralFrame in chain-denominator.cc
void CuLeakyNumeratorComputation::BetaDenFrame(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  // t_wrapped gives us the time-index we use when indexing
  // nnet_output_deriv_transposed_; to save memory we limit the size of the
  // matrix, storing only chunks of frames at a time, and we add it to the
  // non-transposed output whenever we finish a chunk.
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);
  const BaseFloat *this_alpha_dash = alpha_den_.RowData(t),
      *next_beta = beta_den_.RowData((t + 1) % 2);
  BaseFloat *this_beta_dash = beta_den_.RowData(t % 2);
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
    Timer tim;
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
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
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

void CuLeakyNumeratorComputation::BetaHat(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();

  const BaseFloat *this_alpha = alpha_num_.RowData(t);
  BaseFloat *next_beta = beta_num_.RowData((t + 1) % 2);
  const Int32Pair *forward_transitions = num_graph_.ForwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();

  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               t * num_sequences_, num_sequences_);
  int32 max_num_hmm_states = num_graph_.MaxNumStates(),
        num_sequences = num_sequences_;

  CuSubVector<BaseFloat> beta_hats(next_beta +
                                   max_num_hmm_states * num_sequences_,
                                   num_sequences_);
  // we store the beta hats at the end of next beta so that we don't need to
  // pass a separate parameter to the cuda kernel.
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(n_blocks(num_sequences, CU1DBLOCK));
    cuda_chain_leakynum_beta_hat(dimGrid, dimBlock, forward_transitions,
                                 transitions,
                                 num_sequences, num_graph_.NumStates(),
                                 max_num_hmm_states,
                                 probs.Data(), probs.Stride(),
                                 this_alpha, next_beta);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    int32 prob_stride = probs.Stride();
    const BaseFloat *prob_data = probs.Data();
    for (int32 s = 0; s < num_sequences; s++) {
      BaseFloat arbitrary_scale =
          1.0 / this_alpha[max_num_hmm_states * num_sequences + s];
      double next_beta_hat = 0.0;
      // iterate over all transitions of num graph
      const DenominatorGraphTransition
        *trans_iter = transitions +
            forward_transitions[s*max_num_hmm_states+0].first,
        *trans_end = transitions +
            forward_transitions[s*max_num_hmm_states+num_graph_.NumStates()[s]-1].second;
      for (; trans_iter != trans_end; ++trans_iter) {
        BaseFloat transition_prob = trans_iter->transition_prob;
        int32 pdf_id = trans_iter->pdf_id,
              to_hmm_state = trans_iter->hmm_state;
        BaseFloat obs_prob = prob_data[pdf_id * prob_stride + s],
                  next_beta_s = next_beta[to_hmm_state * num_sequences + s];
        next_beta_hat += next_beta_s * transition_prob * obs_prob * arbitrary_scale;
      }
      KALDI_ASSERT(next_beta_hat - next_beta_hat == 0);
      beta_hats(s) = next_beta_hat;
    }
  }

  BaseFloat *this_beta_den = beta_den_.RowData(t % 2);
  CuSubMatrix<BaseFloat> beta_den_mat(this_beta_den,
                                      den_graph_.NumStates(),
                                      num_sequences_,
                                      num_sequences_);
  beta_hats.MulElements(unleak_etas_);
  beta_den_mat.AddVecToRows(1.0, beta_hats, 1.0);
}


void CuLeakyNumeratorComputation::BetaGeneralFrameDebug(int32 t) {
  BaseFloat num_hmm_states_num = num_graph_.MaxNumStates(),
            alpha_beta_size_num = num_hmm_states_num * num_sequences_;
  CuSubVector<BaseFloat> this_alpha_num(alpha_num_.RowData(t), alpha_beta_size_num),
                         this_beta_num(beta_num_.RowData(t % 2), alpha_beta_size_num);

  BaseFloat num_hmm_states_den = den_graph_.NumStates(),
            alpha_beta_size_den = num_hmm_states_den * num_sequences_;
  CuSubVector<BaseFloat> this_alpha_den(alpha_den_.RowData(t), alpha_beta_size_den),
                         this_beta_den(beta_den_.RowData(t % 2), alpha_beta_size_den);

  BaseFloat alpha_beta_product_num = VecVec(this_alpha_num, this_beta_num);
  BaseFloat alpha_beta_product_den = VecVec(this_alpha_den, this_beta_den);
  BaseFloat alpha_beta_product = alpha_beta_product_num + alpha_beta_product_den;

  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps),
        num_pdfs = exp_nnet_output_transposed_.NumRows();
  CuSubMatrix<BaseFloat> this_log_prob_deriv(
      nnet_output_deriv_transposed_, 0, num_pdfs,
      t_wrapped * num_sequences_, num_sequences_);
  BaseFloat this_log_prob_deriv_sum = this_log_prob_deriv.Sum();

  if (!ApproxEqual(alpha_beta_product, num_sequences_)) {
    KALDI_WARN << "On time " << t << ", alpha-beta product "
               << alpha_beta_product << " != " << num_sequences_
               << " alpha-sum-num = " << this_alpha_num.Sum()
               << ", alpha_beta_product_num = " << alpha_beta_product_num
               << ", alpha_beta_product_den = " << alpha_beta_product_den
               << ", beta-sum-num = " << this_beta_num.Sum();
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

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
           num_graph.MaxNumStates() * num_sequences_ + num_sequences_ * 2,
           kSetZero),
    alpha_den_(frames_per_sequence_ + 1,
           den_graph.NumStates() * num_sequences_ + num_sequences_,
           kSetZero),
    // "+ num_sequences_ * 2" is for the alpha-sums and alpha-prime/hat
    beta_num_(2, num_graph.MaxNumStates() * num_sequences_ + num_sequences_,
          kSetZero),
    beta_den_(2, den_graph.NumStates() * num_sequences_,
          kSetZero),
    tot_prob_(num_sequences_, kUndefined),
    tot_log_prob_(num_sequences_, kUndefined),
    ok_(true) {
  KALDI_ASSERT(opts_.num_leak_coefficient > 0.0 &&
               opts_.num_leak_coefficient < 1.0);

  KALDI_ASSERT(nnet_output.NumRows() % num_sequences_ == 0);
  exp_nnet_output_transposed_.ApplyExp();
}


void CuLeakyNumeratorComputation::AlphaFirstFrame() {
  // select alpha for time 0
  BaseFloat *first_frame_alpha = alpha_num_.RowData(0);
  // now make a view of the first num_sequences elements (i.e. alpha_0(0)
  // for all sequences)
  // initializer takes [pointer, length].
  CuSubVector<BaseFloat> alpha_hmm_state0(first_frame_alpha, num_sequences_);
  // set alpha_num_0(0) for all sequences to 1.0 and leave the rest to be 0.0.
  // i.e. the only start state is state 0 of num graph.
  // note: alpha_den is already 0
  alpha_hmm_state0.Set(1.0);
}

void CuLeakyNumeratorComputation::AlphaSumAndPrime(int32 t) {
  BaseFloat *this_alpha_num = alpha_num_.RowData(t);
  BaseFloat *this_alpha_den = alpha_den_.RowData(t);

  // create a 'fake matrix' for the regular alphas- view this row as a matrix.
  // initializer takes [pointer, num-rows, num-cols, stride].

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
  // and also at the end of alpha_den_. This is to avoid any changes in the
  // cuda kernels.
  CuSubVector<BaseFloat> alpha_sum_vec(this_alpha_num +
                                       num_graph_.MaxNumStates()
                                       * num_sequences_,
                                       num_sequences_);

  CuSubVector<BaseFloat> alpha_sum_vec_loc2(this_alpha_den +
                                       den_graph_.NumStates()
                                       * num_sequences_,
                                       num_sequences_);

  alpha_sum_vec.AddRowSumMat(1.0, alpha_mat_den, 0.0);

  /// before adding alphas of num state, we should add this to alpha(t-1)
  /// note: alpha prime is the sum of alphas for all den states
  /// add alpha-prime(t-1) to alpha(t-1) so that they can be used in the
  /// *standard* way to compute next alpha (i.e. apha(t))
  alpha_mat_num.AddVecToRows(1.0, alpha_sum_vec);
  
  /// now add num states too
  alpha_sum_vec.AddRowSumMat(1.0, alpha_mat_num, 1.0);
  alpha_sum_vec_loc2.CopyFromVec(alpha_sum_vec);
}

// the alpha computation for some 0 < t <= num_time_steps_.
void CuLeakyNumeratorComputation::AlphaNumFrame(int32 t) {
  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BaseFloat *this_alpha = alpha_num_.RowData(t);
  const BaseFloat *prev_alpha = alpha_num_.RowData(t - 1);
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
      BaseFloat inv_arbitrary_scale =
          prev_alpha[max_num_hmm_states * num_sequences + s];  //#SCC#
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
          this_tot_alpha += this_prev_alpha / inv_arbitrary_scale * transition_prob * prob; //#SCC#
          
          if (this_tot_alpha - this_tot_alpha != 0) {
            KALDI_LOG << "t: " << t << ", seq: " << s << ", h: " << h
            << ", prev-alpha: " << this_prev_alpha 
            << ", prob: " << prob
            << ", inv_arbitrary_scale: " << inv_arbitrary_scale;
            KALDI_ERR << "Alpha-num failure.";
          }
          
          
        }
        // Let arbitrary_scale be the inverse of the alpha-sum value that we
        // store in the same place we'd store the alpha for the state numbered
        // 'max_num_hmm_states'. We multiply this into all the
        // transition-probabilities from the previous frame to this frame, in
        // both the forward and backward passes, in order to keep the alphas in
        // a good numeric range.  This won't affect the posteriors, but when
        // computing the total likelihood we'll need to compensate for it later
        // on.
        KALDI_ASSERT(this_tot_alpha - this_tot_alpha == 0);
        this_alpha[h * num_sequences + s] = this_tot_alpha ;//* arbitrary_scale; //#SCC#
      }
    }
  }

}

// the alpha computation for some 0 < t <= num_time_steps_.
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

void CuLeakyNumeratorComputation::AlphaHat(int32 t) {

  KALDI_ASSERT(t > 0 && t <= frames_per_sequence_);
  BaseFloat *prev_alpha = alpha_num_.RowData(t - 1);
  const Int32Pair *backward_transitions = num_graph_.BackwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
      max_num_hmm_states = num_graph_.MaxNumStates(),
      num_sequences = num_sequences_;
  //CuVector<BaseFloat> prev_alpha_hats(num_sequences_, kUndefined);
  CuSubVector<BaseFloat> prev_alpha_hats(prev_alpha +
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
    dim3 dimGrid(n_blocks(num_sequences, CU1DBLOCK));
    cuda_chain_leakynum_alpha_hat(dimGrid, dimBlock, backward_transitions,
                                 transitions,
                                 num_sequences, num_graph_.NumStates(),
                                 max_num_hmm_states,
                                 probs.Data(), probs.Stride(),
                                 prev_alpha);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    int32 prob_stride = probs.Stride();
    for (int32 s = 0; s < num_sequences; s++) {
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
          this_prev_alpha = prev_alpha[prev_hmm_state * num_sequences + s];
        prev_alpha_hat += this_prev_alpha * transition_prob * prob;
      }
      BaseFloat arbitrary_scale =
          1.0 / prev_alpha[max_num_hmm_states * num_sequences + s];
      KALDI_ASSERT(prev_alpha_hat - prev_alpha_hat == 0);
      prev_alpha_hats(s) = prev_alpha_hat * arbitrary_scale;
    }
  }

  BaseFloat *this_alpha_den = alpha_den_.RowData(t);
  CuSubMatrix<BaseFloat> alpha_mat_den(this_alpha_den,
                                   den_graph_.NumStates(),
                                   num_sequences_,
                                   num_sequences_);
  alpha_mat_den.AddVecVec(opts_.num_leak_coefficient,
                      den_graph_.InitialProbs(),
                      prev_alpha_hats);
}

BaseFloat CuLeakyNumeratorComputation::Forward() {
  AlphaFirstFrame();
  for (int32 t = 1; t <= frames_per_sequence_; t++) {
    AlphaSumAndPrime(t - 1);
    AlphaNumFrame(t);
    AlphaDenFrame(t);
    AlphaHat(t);
  }
  return ComputeTotLogLike();
}

BaseFloat CuLeakyNumeratorComputation::ComputeTotLogLike() {
  tot_prob_.Resize(num_sequences_);
  // View the last alpha as a matrix of size num-hmm-states by num-sequences.


  CuSubMatrix<BaseFloat> last_alpha(
      alpha_num_.RowData(frames_per_sequence_),
      num_graph_.MaxNumStates(),
      num_sequences_,
      num_sequences_);


  int32 num_states_cpu[num_sequences_];
  num_graph_.CopyNumStatesToCpu(num_states_cpu);
  ///  tot_prob_(seq) for each seq is the num-alpha(T) for the final state of that graph
  //void Lookup(const std::vector<Int32Pair> &indexes, Real *output) const
  std::vector<Int32Pair> final_state_indexes(num_sequences_);
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    final_state_indexes[seq].first = num_states_cpu[seq] - 1;
    final_state_indexes[seq].second = seq;
  }
  last_alpha.Lookup(final_state_indexes, tot_prob_.Data());

  // we should probably add an ApplyLog() function that takes a vector argument.
  tot_log_prob_ = tot_prob_;
  tot_log_prob_.ApplyLog();
  if (num_graph_.AreFirstTransitionsScaled())
    tot_log_prob_.AddVec(1.0, num_graph_.FirstTransitionOffsets());
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
      alpha_num_, 0, frames_per_sequence_,
      num_sequences_ * max_num_hmm_states, num_sequences_);
  CuMatrix<BaseFloat> log_inv_arbitrary_scales(
      inv_arbitrary_scales);
  log_inv_arbitrary_scales.ApplyLog();
  BaseFloat log_inv_arbitrary_scales_product =
      log_inv_arbitrary_scales.Sum();
  return tot_log_prob + log_inv_arbitrary_scales_product;
}


bool CuLeakyNumeratorComputation::Backward(
    BaseFloat deriv_weight,
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  BetaLastFrame();
  for (int32 t = frames_per_sequence_ - 1; t >= 0; t--) {
    BetaPrime(t + 1);
    BetaNumFrame(t);
    BetaDenFrame(t);
    BetaHat(t);
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
      output_deriv_part.AddMat(deriv_weight, transposed_deriv_part, kTrans);
      if (t != 0)
        transposed_deriv_part.SetZero();
    }
  }
//  nnet_output_deriv->AddMat(
//                           deriv_weight, nnet_output_deriv_transposed_, kTrans);
  return ok_;
}

void CuLeakyNumeratorComputation::BetaLastFrame() {
  // sets up the beta quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.

  int32 t = frames_per_sequence_;
  BaseFloat *last_frame_beta = beta_num_.RowData(t % 2);

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
  num_graph_.CopyNumStatesToCpu(num_states_cpu);  //TODO(hhadian) this might be really slow -- check it
  for (int32 seq = 0; seq < num_sequences_; seq++) {
    int32 final_state = num_states_cpu[seq] - 1;
    beta_mat(final_state, seq) = 1.0 / tot_prob_(seq);
  }
  delete num_states_cpu;

//  std::cout << "Beta at time T:\n";
//  beta_mat.Write(std::cout, false); 
}

// compute beta from beta-dash.
void CuLeakyNumeratorComputation::BetaPrime(int32 t) {
  BaseFloat *this_beta_den = beta_den_.RowData(t % 2);
  // create a 'fake matrix' for the regular beta-dash (which is
  // the counterpart of alpha-dash)- view this row as a matrix.
  // initializer takes [pointer, num-rows, num-cols, stride].
  CuSubMatrix<BaseFloat> beta_den_mat(this_beta_den,
                                       den_graph_.NumStates(),
                                       num_sequences_,
                                       num_sequences_);

  CuVector<BaseFloat> beta_den_weighted_sum_vec(num_sequences_, kUndefined); //TODO(hh): no need for kSetZero? #CAOK#
  beta_den_weighted_sum_vec.AddMatVec(opts_.num_leak_coefficient, beta_den_mat,
                              kTrans, den_graph_.InitialProbs(), 0.0);

  BaseFloat *this_beta_num = beta_num_.RowData(t % 2);
  CuSubMatrix<BaseFloat> beta_num_mat(this_beta_num,
                                       num_graph_.MaxNumStates(),
                                       num_sequences_,
                                       num_sequences_);
  beta_num_mat.AddVecToRows(1.0, beta_den_weighted_sum_vec, 1.0);

//  std::cout << "Beta primes for time " << t << ":\n";
//  beta_den_weighted_sum_vec.Write(std::cout, false);
}


void CuLeakyNumeratorComputation::BetaNumFrame(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  // t_wrapped gives us the time-index we use when indexing
  // nnet_output_deriv_transposed_; to save memory we limit the size of the
  // matrix, storing only chunks of frames at a time, and we add it to the
  // non-transposed output whenever we finish a chunk.
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);
  const BaseFloat *this_alpha = alpha_num_.RowData(t),
                  *next_beta = beta_num_.RowData((t + 1) % 2);
  BaseFloat *this_beta = beta_num_.RowData(t % 2);
  const Int32Pair *forward_transitions = num_graph_.ForwardTransitions();
  const DenominatorGraphTransition *transitions = num_graph_.Transitions();
  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               t * num_sequences_, num_sequences_),
      log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                     t_wrapped * num_sequences_, num_sequences_);

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
              prob_data[pdf_id * prob_stride + s] / inv_arbitrary_scale;  // #SCC#
          tot_variable_factor += variable_factor;
          BaseFloat occupation_prob = variable_factor * this_alpha_prob; //occupation_factor; #SCC#
          log_prob_deriv_data[pdf_id * deriv_stride + s] += occupation_prob;


          if (tot_variable_factor - tot_variable_factor != 0) {
            KALDI_LOG << "t: " << t << ", seq: " << s << ", h: " << h << ", pdf_id: " << pdf_id
            << ", var-factor: " << variable_factor 
            << ", ocup-prob: " << occupation_prob
            << ", next-beta: " << next_beta[next_hmm_state * num_sequences + s]
            << ", this_alpha_prob: " << this_alpha_prob
            << ", prob: " << prob_data[pdf_id * prob_stride + s]
            << ", inv arbitrary_scale: " << inv_arbitrary_scale;
            KALDI_ERR << "Beta failure.";
          }

        }
        this_beta[h * num_sequences + s] =
            tot_variable_factor ; /// inv_arbitrary_scale; #SCC#
      }
    }
  }
}

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
  // t_wrapped gives us the time-index we use when indexing
  // nnet_output_deriv_transposed_; to save memory we limit the size of the
  // matrix, storing only chunks of frames at a time, and we add it to the
  // non-transposed output whenever we finish a chunk.

  const BaseFloat *this_alpha = alpha_num_.RowData(t);
  BaseFloat *next_beta = beta_num_.RowData((t + 1) % 2);
  //#SS# CuVector<BaseFloat> beta_hats(num_sequences_, kUndefined);
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
      // this_beta[max_num_hmm_states * num_sequences + s] = beta_hat; // * arbitrary_scale;
    }
  }

  BaseFloat *this_beta_den = beta_den_.RowData(t % 2); 
  CuSubMatrix<BaseFloat> beta_den_mat(this_beta_den,
                                      den_graph_.NumStates(),
                                      num_sequences_,
                                      num_sequences_);
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

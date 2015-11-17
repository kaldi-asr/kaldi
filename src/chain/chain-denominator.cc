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
    const CuMatrixBase<BaseFloat> &nnet_output,
    const std::vector<std::vector<int32 > > &initial_pdf_ids,
    const std::vector<std::vector<int32 > > &final_pdf_ids):
    opts_(opts),
    den_graph_(den_graph),
    initial_pdf_ids_(initial_pdf_ids),
    final_pdf_ids_(final_pdf_ids),
    num_sequences_(num_sequences),
    frames_per_sequence_(nnet_output.NumRows() / num_sequences_),
    exp_nnet_output_transposed_(nnet_output, kTrans),
    nnet_output_deriv_transposed_(exp_nnet_output_transposed_.NumRows(),
                                  exp_nnet_output_transposed_.NumCols()),
    alpha_(frames_per_sequence_ + 1, den_graph_.NumStates() * num_sequences_,
           kUndefined),
    beta_(2, den_graph_.NumStates() * num_sequences_, kUndefined),
    tot_prob_(num_sequences_, kUndefined),
    tot_log_prob_(num_sequences_, kUndefined),
    log_correction_term_(num_sequences_, kUndefined) {
  KALDI_ASSERT(nnet_output.NumRows() % num_sequences == 0);
  exp_nnet_output_transposed_.ApplyExp();
  ModifyInitialAndFinalOutputs();
}


void DenominatorComputation::ModifyInitialAndFinalOutputs() {
  if (initial_pdf_ids_.empty() && final_pdf_ids_.empty())
    return;
  KALDI_ASSERT(initial_pdf_ids_.size() == num_sequences_ &&
               final_pdf_ids_.size() == num_sequences_);
  std::vector<MatrixElement<BaseFloat> > elements;
  elements.reserve(num_sequences_ * 10);  // just a guess.
  int32 num_sequences = num_sequences_,
      frames_per_sequence = frames_per_sequence_;
  // note: conceptually we are adding the negative of this penalty to all
  // initial/final pdf-ids which are *not* listed for their respective sequence;
  // in practice we add this penalty to those listed, and then subtract twice
  // this penalty from the overall log-likelihood; this is more efficient.
  BaseFloat penalty = opts_.pdf_boundary_penalty;
  for (int32 seq = 0; seq < num_sequences; seq++) {

    for (int32 i = 0; i < 2; i++) {
      const std::vector<int32> &vec = (i == 0 ?
                                       initial_pdf_ids_[seq] :
                                       final_pdf_ids_[seq]);
      std::vector<int32>::const_iterator iter = vec.begin(),
          end = vec.end();
      int32 frame_index = (i == 0 ? 0 : frames_per_sequence - 1);
      for (; iter != end; ++iter) {
        int32 pdf_id = *iter;
        MatrixElement<BaseFloat> elem;
        elem.row = pdf_id;
        elem.column = frame_index * num_sequences + seq;
        elem.weight = penalty;
        elements.push_back(elem);
      }
    }
  }
  exp_nnet_output_transposed_.AddElements(1.0, elements);
  exp_nnet_output_transposed_.ApplyExp();
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
  const BaseFloat *prev_alpha = alpha_.RowData(t - 1);
  const Int32Pair *backward_transitions = den_graph_.BackwardTransitions();
  const DenominatorGraphTransition *transitions = den_graph_.Transitions();
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows(),
      num_hmm_states = den_graph_.NumStates(),
      num_sequences = num_sequences_,
      special_hmm_state = den_graph_.SpecialHmmState();

  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               (t-1) * num_sequences_, num_sequences_);
  const BaseFloat *prob_data = probs.Data();

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);

    cuda_chain_hmm_forward(dimGrid, dimBlock, backward_transitions, transitions, t,
                         num_sequences, special_hmm_state, prob_data,
                         probs.Stride(), prev_alpha, this_alpha);

    CU_SAFE_CALL(cudaGetLastError());
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
        // Let arbitrary_scale be the inverse of the alpha value for the
        // hmm-state indexed special_hmm_state_ on the previous frame (for this
        // sequence); we multiply this into all the transition-probabilities
        // from the previous frame to this frame, in both the forward and
        // backward passes, in order to keep the alphas in a good numeric range.
        // This won't affect the posteriors, but when computing the total
        // likelihood we'll need to compensate for it later on.
        BaseFloat arbitrary_scale =
            1.0 / prev_alpha[special_hmm_state * num_sequences + s];
        KALDI_ASSERT(this_tot_alpha - this_tot_alpha == 0);
        this_alpha[h * num_sequences + s] = this_tot_alpha * arbitrary_scale;
      }
    }
  }
}

BaseFloat DenominatorComputation::Forward() {
  AlphaFirstFrame();
  for (int32 t = 1; t <= frames_per_sequence_; t++)
    AlphaGeneralFrame(t);
  return ComputeTotLogLike();
}

BaseFloat DenominatorComputation::ComputeTotLogLike() {
  tot_prob_.Resize(num_sequences_);
  // View the last alpha as a matrix of size num-time-steps by num-sequences.
  CuSubMatrix<BaseFloat> last_alpha(alpha_.RowData(frames_per_sequence_),
                                    den_graph_.NumStates(),
                                    num_sequences_,
                                    num_sequences_);

  tot_prob_.AddRowSumMat(1.0, last_alpha, 0.0);
  // we should probably add an ApplyLog() function that takes a vector argument.
  tot_log_prob_ = tot_prob_;
  tot_log_prob_.ApplyLog();
  BaseFloat tot_log_prob = tot_log_prob_.Sum();

  // We now have to add something for the arbitrary scaling factor.  the
  // inverses of all the alphas for hmm-states numbered zero, for t = 0
  // ... frames_per_sequence_ - 1, were included as the 'arbitrary factors' in the
  // transition-probs, so we need to multiply them all together (not inversed)
  // and add them as a correction term to the total log-likes.  Note: the
  // purpose of the arbitrary scaling factors was to keep things in a good
  // floating-point range.
  CuSubMatrix<BaseFloat> inv_arbitrary_scales(
      alpha_, 0, frames_per_sequence_,
      num_sequences_ * den_graph_.SpecialHmmState(), num_sequences_);
  CuMatrix<BaseFloat> log_inv_arbitrary_scales(
      inv_arbitrary_scales);
  log_inv_arbitrary_scales.ApplyLog();
  BaseFloat log_inv_arbitrary_scales_product =
      log_inv_arbitrary_scales.Sum();
  // penalty_correction_factor relates to how we apply 'pdf_boundary_penalty'.
  // The aim here is to penalize all pdfs in the initial and final frame that
  // are not present in the initial and final frame respectively of the
  // numerator FST, by subtracting opts_.pdf_boundary_penalty from them.  This
  // helps to simulate the effect of actually seeing context.  The way we apply
  // this is we *add* 'pdf_boundary_penalty' to the log-probs that are *in* the
  // numerator FST, and then add -2.0 * opts_.pdf_boundary_penalty to the
  // overall likelihood we return (one for the initial frame, and one for the
  // final); this is more efficient.  We also have to handle the case where
  // initial_pdf_ids_ is empty, which may be encountered in test code: in this
  // case, the user did not supply that information.
  BaseFloat penalty_correction_factor =
      (initial_pdf_ids_.empty() ? 0 : -2.0 * opts_.pdf_boundary_penalty);
  return tot_log_prob + log_inv_arbitrary_scales_product +
      penalty_correction_factor;
}



void DenominatorComputation::Backward(
    BaseFloat deriv_weight,
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  BetaLastFrame();
  for (int32 t = frames_per_sequence_ - 1; t >= 0; t--)
    BetaGeneralFrame(t);
  nnet_output_deriv->AddMat(deriv_weight, nnet_output_deriv_transposed_,
                            kTrans);
}

void DenominatorComputation::BetaLastFrame() {
  // sets up the beta on the last frame (frame == frames_per_sequence_).  Note that
  // the betas we use here contain a 1/(tot-prob) factor in order to simplify
  // the backprop.

  int32 t = frames_per_sequence_;
  BaseFloat *last_frame_beta = beta_.RowData(t % 2);

  // create a 'fake matrix' - view this row as a matrix.
  CuSubMatrix<BaseFloat> beta_mat(last_frame_beta,
                                  den_graph_.NumStates(),
                                  num_sequences_,
                                  num_sequences_);
  CuVector<BaseFloat> inv_tot_prob(tot_prob_);
  inv_tot_prob.InvertElements();
  // the beta values at the end of the file only vary with the sequence-index,
  // not with the HMM-index.  There is no notion of final-prob; the sequence
  // ends when it ends, and at that point we treat all states as having a
  // final-prob of one (which we treat as p of being final given that it just
  // ended, i.e. the probability of a sure thing, which is one).
  beta_mat.CopyRowsFromVec(inv_tot_prob);
}

void DenominatorComputation::BetaGeneralFrame(int32 t) {
  KALDI_ASSERT(t >= 0 && t < frames_per_sequence_);
  int32 num_pdfs = exp_nnet_output_transposed_.NumRows();
  const BaseFloat *this_alpha = alpha_.RowData(t),
      *next_beta = beta_.RowData((t + 1) % 2);
  BaseFloat *this_beta = beta_.RowData(t % 2);
  const Int32Pair *forward_transitions = den_graph_.ForwardTransitions();
  const DenominatorGraphTransition *transitions = den_graph_.Transitions();
  // 'probs' is the matrix of pseudo-likelihoods for frame t.
  CuSubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                               t * num_sequences_, num_sequences_),
      log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                     t * num_sequences_, num_sequences_);
  // we assume that if two matrices have been allocated with the same dimension,
  // they have the same stride.  If this is found later on to not be the case, we
  // can change the CUDA kernel.
  KALDI_ASSERT(probs.Stride() == log_prob_deriv.Stride());

  int32 num_hmm_states = den_graph_.NumStates(),
      num_sequences = num_sequences_,
      special_hmm_state = den_graph_.SpecialHmmState();

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(std::min<int32>(CU1DBLOCK, num_sequences), 1, 1);
    dim3 dimGrid(n_blocks(num_sequences, dimBlock.x), num_hmm_states, 1);
    cuda_chain_hmm_backward(dimGrid, dimBlock, forward_transitions, transitions, t,
                          num_sequences, special_hmm_state,
                          probs.Data(), probs.Stride(), this_alpha, next_beta,
                          this_beta, log_prob_deriv.Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    int32 prob_stride = probs.Stride();
    const BaseFloat *prob_data = probs.Data();
    BaseFloat *log_prob_deriv_data = log_prob_deriv.Data();
    for (int32 h = 0; h < num_hmm_states; h++) {
      for (int32 s = 0; s < num_sequences; s++) {
        BaseFloat this_alpha_prob = this_alpha[h * num_sequences + s],
            inv_arbitrary_scale =
            this_alpha[special_hmm_state * num_sequences + s];
        double tot_variable_factor = 0.0;
        BaseFloat
            occupation_factor = this_alpha_prob / inv_arbitrary_scale;
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
          log_prob_deriv_data[pdf_id * prob_stride + s] += occupation_prob;
        }
        this_beta[h * num_sequences + s] =
            tot_variable_factor / inv_arbitrary_scale;
      }
    }
  }
}


}  // namespace chain
}  // namespace kaldi

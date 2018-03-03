// chain/chain-generic-numerator.cc

// Copyright      2017   Hossein Hadian

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


#include "chain/chain-generic-numerator.h"
#include "chain/chain-kernels-ansi.h"

namespace kaldi {
namespace chain {

// GenericNumeratorComputation is responsible for the forward-backward of the
// end-to-end 'supervision' (numerator) FST. It is used in chain-training.cc
// (similar to NumeratorComputation) to compute the numerator derivatives
// for end-to-end training 'supervision's.

GenericNumeratorComputation::GenericNumeratorComputation(
    const Supervision &supervision,
    const CuMatrixBase<BaseFloat> &nnet_output):
    supervision_(supervision),
    nnet_output_deriv_transposed_(
        nnet_output.NumCols(),
        std::min<int32>(nnet_output.NumRows(),
                        static_cast<int32>(kMaxDerivTimeSteps) *
                        supervision.num_sequences)),
    tot_prob_(supervision.num_sequences, kUndefined),
    ok_(true) {
  KALDI_ASSERT(supervision.num_sequences *
               supervision.frames_per_sequence == nnet_output.NumRows() &&
               supervision.label_dim == nnet_output.NumCols());
  {
    CuMatrix<BaseFloat> exp_nnet_output_transposed_gpu(nnet_output, kTrans);
    exp_nnet_output_transposed_gpu.ApplyExp();
    exp_nnet_output_transposed_.Resize(nnet_output.NumCols(),
                                       nnet_output.NumRows(), kUndefined);
    exp_nnet_output_transposed_.CopyFromMat(exp_nnet_output_transposed_gpu);
  }

  using std::vector;
  int32 B = supervision_.num_sequences,
      num_frames = supervision_.frames_per_sequence;
  KALDI_ASSERT(supervision_.e2e_fsts.size() == B);

  // Find the maximum number of HMM states and then
  // initialize final probs, alpha, and beta.
  max_num_hmm_states_ = 0;
  for (int32 i = 0; i < B; i++) {
    KALDI_ASSERT(supervision_.e2e_fsts[i].Properties(fst::kIEpsilons, true)
                 == 0);
    if (supervision_.e2e_fsts[i].NumStates() > max_num_hmm_states_)
      max_num_hmm_states_ = supervision_.e2e_fsts[i].NumStates();
  }
  final_probs_.Resize(max_num_hmm_states_, B, kSetZero);
  alpha_.Resize(num_frames + 1,
                max_num_hmm_states_ * B + B,
                kSetZero);
  // The extra B is for storing alpha sums
  beta_.Resize(2, max_num_hmm_states_ * B, kSetZero);

  // Initialize incoming transitions for easy access
  in_transitions_.resize(B); // indexed by seq, state
  out_transitions_.resize(B); // indexed by seq, state
  for (int32 seq = 0; seq < B; seq++) {
    in_transitions_[seq] = vector<vector<DenominatorGraphTransition> >(
        supervision_.e2e_fsts[seq].NumStates());
    out_transitions_[seq] = vector<vector<DenominatorGraphTransition> >(
        supervision_.e2e_fsts[seq].NumStates());
  }

  offsets_.Resize(B);
  for (int32 seq = 0; seq < B; seq++) {
    for (int32 s = 0; s < supervision_.e2e_fsts[seq].NumStates(); s++) {
      final_probs_(s, seq) = exp(-supervision_.e2e_fsts[seq].Final(s).Value());
      BaseFloat offset = 0.0;
      if (s == 0) {
        for (fst::ArcIterator<fst::StdVectorFst> aiter(
                 supervision_.e2e_fsts[seq], s);
             !aiter.Done();
             aiter.Next())
          if (aiter.Value().weight.Value() > offset)
            offset = aiter.Value().weight.Value();
        offsets_(seq) = offset;
      }

      for (fst::ArcIterator<fst::StdVectorFst> aiter(
               supervision_.e2e_fsts[seq], s);
           !aiter.Done();
           aiter.Next()) {
        const fst::StdArc &arc = aiter.Value();
        DenominatorGraphTransition transition;
        transition.transition_prob = exp(-(arc.weight.Value() - offset));
        transition.pdf_id = arc.ilabel - 1;
        transition.hmm_state = s;
        in_transitions_[seq][arc.nextstate].push_back(transition);
        transition.hmm_state = arc.nextstate;
        out_transitions_[seq][s].push_back(transition);
      }
    }
  }
}


void GenericNumeratorComputation::AlphaFirstFrame() {
  const int32 num_sequences = supervision_.num_sequences,
      num_states = max_num_hmm_states_;
  // Set alpha_0(0) for all sequences to 1.0 and leave the rest to be 0.0.
  double *first_frame_alpha = alpha_.RowData(0);
  SubVector<double> alpha_hmm_state0(first_frame_alpha, num_sequences);
  alpha_hmm_state0.Set(1.0);

  // Now compute alpha-sums for t=0 which is obviously 1.0 for each sequence
  SubVector<double> alpha_sum_vec(first_frame_alpha +
                                  num_states * num_sequences,
                                  num_sequences);
  alpha_sum_vec.Set(1.0);
}


// The alpha computation for some 0 < t <= num_time_steps_.
void GenericNumeratorComputation::AlphaGeneralFrame(int32 t) {
  // Define some variables to make things nicer
  const int32
      num_sequences = supervision_.num_sequences,
      num_frames = supervision_.frames_per_sequence,
      num_pdfs = exp_nnet_output_transposed_.NumRows(),
      num_states = max_num_hmm_states_;
  KALDI_ASSERT(t > 0 && t <= num_frames);

  SubMatrix<double> this_alpha(alpha_.RowData(t), num_states,
                               num_sequences, num_sequences);
  const SubMatrix<double> prev_alpha(alpha_.RowData(t - 1), num_states + 1,
                                     num_sequences, num_sequences);
  // 'probs' is the matrix of pseudo-likelihoods for frame t - 1.
  SubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                             (t - 1) * num_sequences, num_sequences);

  for (int32 seq = 0; seq < num_sequences; seq++) {
    double inv_arbitrary_scale = prev_alpha(num_states, seq);
    for (int32 h = 0; h < supervision_.e2e_fsts[seq].NumStates(); h++) {
      for (auto tr = in_transitions_[seq][h].begin();
           tr != in_transitions_[seq][h].end(); tr++) {
        double transition_prob = tr->transition_prob;
        int32 pdf_id = tr->pdf_id, prev_hmm_state = tr->hmm_state;
        double prob = probs(pdf_id, seq);
        this_alpha(h, seq) += prev_alpha(prev_hmm_state, seq) /
            inv_arbitrary_scale * transition_prob * prob;
      }
    }
  }

  if (t == num_frames)  // last alpha
    this_alpha.MulElements(final_probs_);
  // Now compute alpha-sums for frame t:
  SubVector<double> alpha_sum_vec(alpha_.RowData(t) + num_states * num_sequences,
                                  num_sequences);
  alpha_sum_vec.AddRowSumMat(1.0, this_alpha, 0.0);
}

BaseFloat GenericNumeratorComputation::Forward() {
  AlphaFirstFrame();
  for (int32 t = 1; t <= supervision_.frames_per_sequence; t++) {
    AlphaGeneralFrame(t);
  }
  return ComputeTotLogLike();
}

BaseFloat GenericNumeratorComputation::ComputeTotLogLike() {
  const int32
      num_sequences = supervision_.num_sequences,
      num_frames = supervision_.frames_per_sequence,
      num_states = max_num_hmm_states_;

  // View the last alpha as a matrix of size num-hmm-states by num-sequences.
  SubMatrix<double> last_alpha(alpha_.RowData(num_frames), num_states,
                               num_sequences, num_sequences);
  tot_prob_.AddRowSumMat(1.0, last_alpha, 0.0);
  Vector<double> tot_log_probs(tot_prob_);
  tot_log_probs.ApplyLog();
  tot_log_probs.AddVec(-1.0, offsets_);
  double tot_log_prob = tot_log_probs.Sum();
  SubMatrix<double> inv_arbitrary_scales(alpha_, 0, num_frames,
                                         num_sequences * num_states,
                                         num_sequences);
  Matrix<double> log_inv_arbitrary_scales(inv_arbitrary_scales);
  log_inv_arbitrary_scales.ApplyLog();
  double log_inv_arbitrary_scales_product =
      log_inv_arbitrary_scales.Sum();
  return tot_log_prob + log_inv_arbitrary_scales_product;
}


bool GenericNumeratorComputation::Backward(
    CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  const int32
      num_sequences = supervision_.num_sequences,
      num_frames = supervision_.frames_per_sequence,
      num_pdfs = exp_nnet_output_transposed_.NumRows();
  BetaLastFrame();
  for (int32 t = num_frames - 1; t >= 0; t--) {
    BetaGeneralFrame(t);
    if (GetVerboseLevel() >= 1 || t == 0 || t == num_frames - 1)
      BetaGeneralFrameDebug(t);
    if (t % kMaxDerivTimeSteps == 0) {
      // Commit the derivative stored in exp_nnet_output_transposed_ by adding
      // its transpose to the appropriate sub-matrix of 'nnet_output_deriv'.
      int32 chunk_frames = std::min<int32>(static_cast<int32>(kMaxDerivTimeSteps),
                                           num_frames - t);
      SubMatrix<BaseFloat> transposed_deriv_part(
          nnet_output_deriv_transposed_,
          0, num_pdfs,
          0, chunk_frames * num_sequences);
      CuMatrix<BaseFloat> tmp(transposed_deriv_part);
      CuSubMatrix<BaseFloat> output_deriv_part(
          *nnet_output_deriv,
          t * num_sequences, chunk_frames * num_sequences,
          0, num_pdfs);
      output_deriv_part.AddMat(supervision_.weight, tmp, kTrans);
      if (t != 0)
        transposed_deriv_part.SetZero();
    }
  }
  return ok_;
}

void GenericNumeratorComputation::BetaLastFrame() {
  // Sets up the beta quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.
  int32 t = supervision_.frames_per_sequence;
  double *last_frame_beta = beta_.RowData(t % 2);

  SubMatrix<double> beta_mat(last_frame_beta,
                             max_num_hmm_states_,
                             supervision_.num_sequences,
                             supervision_.num_sequences);

  Vector<double> inv_tot_prob(tot_prob_);
  inv_tot_prob.InvertElements();

  beta_mat.CopyRowsFromVec(inv_tot_prob);
  beta_mat.MulElements(final_probs_);
}

void GenericNumeratorComputation::BetaGeneralFrame(int32 t) {
  const int32
      num_sequences = supervision_.num_sequences,
      num_frames = supervision_.frames_per_sequence,
      num_pdfs = exp_nnet_output_transposed_.NumRows(),
      num_states = max_num_hmm_states_;
  KALDI_ASSERT(t >= 0 && t < num_frames);

  // t_wrapped gives us the time-index we use when indexing
  // nnet_output_deriv_transposed_; to save memory we limit the size of the
  // matrix, storing only chunks of frames at a time, and we add it to the
  // non-transposed output whenever we finish a chunk.
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps);
  const SubMatrix<double> this_alpha(alpha_.RowData(t), num_states,
                               num_sequences, num_sequences);
  SubMatrix<double> this_beta(beta_.RowData(t % 2), num_states,
                               num_sequences, num_sequences);
  const SubMatrix<double> next_beta(beta_.RowData((t + 1) % 2), num_states,
                               num_sequences, num_sequences);

  SubMatrix<BaseFloat> probs(exp_nnet_output_transposed_, 0, num_pdfs,
                             t * num_sequences, num_sequences),
      log_prob_deriv(nnet_output_deriv_transposed_, 0, num_pdfs,
                     t_wrapped * num_sequences, num_sequences);

  for (int32 seq = 0; seq < num_sequences; seq++) {
    for (int32 h = 0; h < supervision_.e2e_fsts[seq].NumStates(); h++) {
      BaseFloat inv_arbitrary_scale = this_alpha(num_states, seq);
      double tot_variable_factor = 0.0;
      for (auto tr = out_transitions_[seq][h].begin();
           tr != out_transitions_[seq][h].end(); tr++) {
        BaseFloat transition_prob = tr->transition_prob;
        int32 pdf_id = tr->pdf_id,
            next_hmm_state = tr->hmm_state;
        double variable_factor = transition_prob *
            next_beta(next_hmm_state, seq) *
            probs(pdf_id, seq) / inv_arbitrary_scale;
        tot_variable_factor += variable_factor;
        double occupation_prob = variable_factor * this_alpha(h, seq);
        log_prob_deriv(pdf_id, seq) += occupation_prob;
      }
      this_beta(h, seq) = tot_variable_factor;
    }
  }
}

void GenericNumeratorComputation::BetaGeneralFrameDebug(int32 t) {
  int32 alpha_beta_size = max_num_hmm_states_ * supervision_.num_sequences;
  SubVector<double> this_alpha(alpha_.RowData(t), alpha_beta_size),
      this_beta(beta_.RowData(t % 2), alpha_beta_size);
  int32 t_wrapped = t % static_cast<int32>(kMaxDerivTimeSteps),
        num_pdfs = exp_nnet_output_transposed_.NumRows();
  SubMatrix<BaseFloat> this_log_prob_deriv(
      nnet_output_deriv_transposed_, 0, num_pdfs,
      t_wrapped * supervision_.num_sequences, supervision_.num_sequences);
  double alpha_beta_product = VecVec(this_alpha,
                                     this_beta),
      this_log_prob_deriv_sum = this_log_prob_deriv.Sum();
  if (!ApproxEqual(alpha_beta_product, supervision_.num_sequences)) {
    KALDI_WARN << "On time " << t << ", alpha-beta product "
               << alpha_beta_product << " != " << supervision_.num_sequences
               << " alpha-sum = " << this_alpha.Sum()
               << ", beta-sum = " << this_beta.Sum();
    if (fabs(alpha_beta_product - supervision_.num_sequences) > 2.0
        || alpha_beta_product - alpha_beta_product != 0) {
      KALDI_WARN << "Excessive error detected, will abandon this minibatch";
      ok_ = false;
    }
  }
  // Use higher tolerance, since we are using randomized pruning for the
  // log-prob derivatives.
  if (!ApproxEqual(this_log_prob_deriv_sum,
                   supervision_.num_sequences, 0.01)) {
    KALDI_WARN << "On time " << t << ", log-prob-deriv sum "
               << this_log_prob_deriv_sum << " != "
               << supervision_.num_sequences;
    if (fabs(this_log_prob_deriv_sum - supervision_.num_sequences) > 2.0 ||
        this_log_prob_deriv_sum - this_log_prob_deriv_sum != 0) {
      KALDI_WARN << "Excessive error detected, will abandon this minibatch";
      ok_ = false;
    }
  }
}

}  // namespace chain
}  // namespace kaldi

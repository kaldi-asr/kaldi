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

#include <iterator>
#include <limits>
#include <algorithm>

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
    nnet_output_(nnet_output) {
  KALDI_ASSERT(supervision.num_sequences *
               supervision.frames_per_sequence == nnet_output.NumRows() &&
               supervision.label_dim == nnet_output.NumCols());

  using std::vector;
  int num_sequences = supervision_.num_sequences;
  KALDI_ASSERT(supervision_.e2e_fsts.size() == num_sequences);

  // Find the maximum number of HMM states and then
  // initialize final probs, alpha, and beta.
  int max_num_hmm_states = 0;
  for (int i = 0; i < num_sequences; i++) {
    KALDI_ASSERT(supervision_.e2e_fsts[i].Properties(fst::kIEpsilons, true)
                 == 0);
    if (supervision_.e2e_fsts[i].NumStates() > max_num_hmm_states)
      max_num_hmm_states = supervision_.e2e_fsts[i].NumStates();
  }
  final_probs_.Resize(num_sequences, max_num_hmm_states);

  // Initialize incoming transitions for easy access
  in_transitions_.resize(num_sequences);  // indexed by seq, state
  out_transitions_.resize(num_sequences);  // indexed by seq, state
  for (int seq = 0; seq < num_sequences; seq++) {
    in_transitions_[seq] = vector<vector<DenominatorGraphTransition> >(
        supervision_.e2e_fsts[seq].NumStates());
    out_transitions_[seq] = vector<vector<DenominatorGraphTransition> >(
        supervision_.e2e_fsts[seq].NumStates());
  }

  offsets_.Resize(num_sequences);
  std::unordered_map<int32, MatrixIndexT> pdf_to_index;
  int32 pdf_stride = nnet_output_.Stride();
  int32 view_stride = nnet_output_.Stride() * num_sequences;
  pdf_to_index.reserve(view_stride);
  nnet_output_stride_ = pdf_stride;
  for (int seq = 0; seq < num_sequences; seq++) {
    for (int32 s = 0; s < supervision_.e2e_fsts[seq].NumStates(); s++) {
      final_probs_(seq, s)= -supervision_.e2e_fsts[seq].Final(s).Value();
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
        transition.transition_prob = -(arc.weight.Value() - offset);

        int32 pdf_id = arc.ilabel - 1;  // note: the FST labels were pdf-id plus one.

        // remap  to a unique index in the remapped space
        pdf_id = pdf_id + seq * pdf_stride;
        KALDI_ASSERT(pdf_id < view_stride);

        if (pdf_to_index.find(pdf_id) == pdf_to_index.end()) {
          index_to_pdf_.push_back(pdf_id);
          pdf_to_index[pdf_id] = index_to_pdf_.size() - 1;
        }

        transition.pdf_id = pdf_to_index[pdf_id];
        transition.hmm_state = s;
        in_transitions_[seq][arc.nextstate].push_back(transition);
        transition.hmm_state = arc.nextstate;
        out_transitions_[seq][s].push_back(transition);
      }
    }
  }
}


void GenericNumeratorComputation::AlphaFirstFrame(int seq,
                                                  Matrix<BaseFloat> *alpha) {
  const int32 num_frames = supervision_.frames_per_sequence,
              num_states = supervision_.e2e_fsts[seq].NumStates();
  alpha->Resize(num_frames + 1,  num_states + 1, kSetZero);
  alpha->Set(-std::numeric_limits<BaseFloat>::infinity());
  (*alpha)(0, 0) = 0.0;
  (*alpha)(0, num_states) = 0.0;
}


void GenericNumeratorComputation::CopySpecificPdfsIndirect(
                                    const CuMatrixBase<BaseFloat> &nnet_output,
                                    const std::vector<MatrixIndexT> &indices,
                                    Matrix<BaseFloat> *out) {
  KALDI_ASSERT(nnet_output_stride_ == nnet_output_.Stride());
  const int32 num_sequences = supervision_.num_sequences,
              frames_per_sequence = supervision_.frames_per_sequence;

  const BaseFloat *starting_ptr = nnet_output.RowData(0);
  const int view_stride = num_sequences * nnet_output.Stride();

  const CuSubMatrix<BaseFloat> sequence_view(starting_ptr,
                                             frames_per_sequence,
                                             view_stride,
                                             view_stride);

  CuArray<MatrixIndexT> indices_gpu(indices);
  CuMatrix<BaseFloat> required_pdfs(frames_per_sequence,
                                        indices.size());

  required_pdfs.CopyCols(sequence_view, indices_gpu);
  out->Swap(&required_pdfs);
}

// The alpha computation for some 0 < t <= num_time_steps_.
BaseFloat GenericNumeratorComputation::AlphaRemainingFrames(int seq,
                                              const Matrix<BaseFloat> &probs,
                                              Matrix<BaseFloat> *alpha) {
  // Define some variables to make things nicer
  const int32 num_sequences = supervision_.num_sequences,
              num_frames = supervision_.frames_per_sequence;

  KALDI_ASSERT(seq >= 0 && seq < num_sequences);

  // variables for log_likelihood computation
  double log_scale_product = 0,
         log_prob_product = 0;

  for (int t = 1; t <= num_frames; ++t) {
    const BaseFloat *probs_tm1 = probs.RowData(t - 1);
    BaseFloat *alpha_t = alpha->RowData(t);
    const BaseFloat *alpha_tm1 = alpha->RowData(t - 1);

    for (int32 h = 0; h < supervision_.e2e_fsts[seq].NumStates(); h++) {
      for (auto tr = in_transitions_[seq][h].begin();
          tr != in_transitions_[seq][h].end(); ++tr) {
        BaseFloat transition_prob = tr->transition_prob;
        int32 pdf_id = tr->pdf_id,
              prev_hmm_state = tr->hmm_state;
        BaseFloat prob = probs_tm1[pdf_id];
        alpha_t[h] = LogAdd(alpha_t[h],
            alpha_tm1[prev_hmm_state] + transition_prob + prob);
      }
    }
    double sum = alpha_tm1[alpha->NumCols() - 1];
    SubMatrix<BaseFloat> alpha_t_mat(*alpha, t, 1, 0,
                                      alpha->NumCols() - 1);
    alpha_t_mat.Add(-sum);
    sum = alpha_t_mat.LogSumExp();

    alpha_t[alpha->NumCols() - 1] = sum;
    log_scale_product += sum;
  }
  SubMatrix<BaseFloat> last_alpha(*alpha, alpha->NumRows() - 1, 1,
                                  0, alpha->NumCols() - 1);
  SubVector<BaseFloat> final_probs(final_probs_.RowData(seq),
                                   alpha->NumCols() - 1);

  // adjust last_alpha
  double sum = (*alpha)(alpha->NumRows() - 1, alpha->NumCols() - 1);
  log_scale_product -= sum;
  last_alpha.AddVecToRows(1.0, final_probs);
  sum = last_alpha.LogSumExp();
  (*alpha)(alpha->NumRows() - 1, alpha->NumCols() - 1) = sum;

  // second part of criterion
  log_prob_product = sum - offsets_(seq);

  return log_prob_product + log_scale_product;
}

bool GenericNumeratorComputation::ForwardBackward(
                                 BaseFloat *total_loglike,
                                 CuMatrixBase<BaseFloat> *nnet_output_deriv) {
  KALDI_ASSERT(total_loglike != NULL);
  KALDI_ASSERT(nnet_output_deriv != NULL);
  KALDI_ASSERT(nnet_output_deriv->NumCols() == nnet_output_.NumCols());
  KALDI_ASSERT(nnet_output_deriv->NumRows() == nnet_output_.NumRows());

  BaseFloat partial_loglike = 0;
  const int32 num_sequences = supervision_.num_sequences;

  bool ok = true;
  Matrix<BaseFloat> alpha;
  Matrix<BaseFloat> beta;
  Matrix<BaseFloat> probs;
  Matrix<BaseFloat> derivs;

  // We selectively copy only those pdfs we need
  CopySpecificPdfsIndirect(nnet_output_, index_to_pdf_, &probs);

  derivs.Resize(probs.NumRows(), probs.NumCols());
  derivs.Set(-std::numeric_limits<BaseFloat>::infinity());

  for (int seq = 0; seq < num_sequences; ++seq) {
    // Forward part
    AlphaFirstFrame(seq, &alpha);
    partial_loglike += AlphaRemainingFrames(seq, probs, &alpha);

    // Backward part
    BetaLastFrame(seq, alpha, &beta);
    BetaRemainingFrames(seq, probs, alpha, &beta, &derivs);
    if (GetVerboseLevel() >= 1)
      ok = ok && CheckValues(seq, probs, alpha, beta, derivs);
  }
  // Transfer and add the derivatives to the values in the matrix
  AddSpecificPdfsIndirect(&derivs, index_to_pdf_, nnet_output_deriv);
  *total_loglike = partial_loglike;
  return ok;
}

BaseFloat GenericNumeratorComputation::ComputeObjf() {
  BaseFloat partial_loglike = 0;
  const int32 num_sequences = supervision_.num_sequences;

  Matrix<BaseFloat> alpha;
  Matrix<BaseFloat> probs;

  // We selectively copy only those pdfs we need
  CopySpecificPdfsIndirect(nnet_output_, index_to_pdf_, &probs);

  for (int seq = 0; seq < num_sequences; ++seq) {
    // Forward part
    AlphaFirstFrame(seq, &alpha);
    partial_loglike += AlphaRemainingFrames(seq, probs, &alpha);
  }
  return partial_loglike;
}

BaseFloat GenericNumeratorComputation::GetTotalProb(
                                          const Matrix<BaseFloat> &alpha) {
  return alpha(alpha.NumRows() - 1, alpha.NumCols() - 1);
}

void GenericNumeratorComputation::BetaLastFrame(int seq,
                                                const Matrix<BaseFloat> &alpha,
                                                Matrix<BaseFloat> *beta) {
  // Sets up the beta quantity on the last frame (frame ==
  // frames_per_sequence_).  Note that the betas we use here contain a
  // 1/(tot-prob) factor in order to simplify the backprop.
  const int32 num_frames = supervision_.frames_per_sequence,
              num_states = supervision_.e2e_fsts[seq].NumStates();
  float tot_prob = GetTotalProb(alpha);

  beta->Resize(2, num_states);
  beta->Set(-std::numeric_limits<BaseFloat>::infinity());

  SubVector<BaseFloat> beta_mat(beta->RowData(num_frames % 2), num_states);
  SubVector<BaseFloat> final_probs(final_probs_.RowData(seq), num_states);

  BaseFloat inv_tot_prob = -tot_prob;
  beta_mat.Set(inv_tot_prob);
  beta_mat.AddVec(1.0, final_probs);
}

void GenericNumeratorComputation::BetaRemainingFrames(int seq,
                                                const Matrix<BaseFloat> &probs,
                                                const Matrix<BaseFloat> &alpha,
                                                Matrix<BaseFloat> *beta,
                                                Matrix<BaseFloat> *derivs) {
  const int32
      num_sequences = supervision_.num_sequences,
      num_frames = supervision_.frames_per_sequence,
      num_states = supervision_.e2e_fsts[seq].NumStates();
  KALDI_ASSERT(seq >= 0 && seq < num_sequences);

  for (int t = num_frames - 1; t >= 0; --t) {
    const BaseFloat *alpha_t = alpha.RowData(t),
        *beta_tp1 = beta->RowData((t + 1) % 2),
        *probs_t = probs.RowData(t);
    BaseFloat *log_prob_deriv_t = derivs->RowData(t),
        *beta_t = beta->RowData(t % 2);

    BaseFloat inv_arbitrary_scale = alpha_t[num_states];
    for (int32 h = 0; h < supervision_.e2e_fsts[seq].NumStates(); h++) {
      BaseFloat tot_variable_factor;
      tot_variable_factor = -std::numeric_limits<BaseFloat>::infinity();
      for (auto tr = out_transitions_[seq][h].begin();
               tr != out_transitions_[seq][h].end(); ++tr) {
        BaseFloat transition_prob = tr->transition_prob;
        int32 pdf_id = tr->pdf_id,
            next_hmm_state = tr->hmm_state;
        BaseFloat variable_factor = transition_prob +
            beta_tp1[next_hmm_state] +
            probs_t[pdf_id] - inv_arbitrary_scale;
        tot_variable_factor = LogAdd(tot_variable_factor,
                                     variable_factor);

        BaseFloat occupation_prob = variable_factor + alpha_t[h];
        log_prob_deriv_t[pdf_id] = LogAdd(log_prob_deriv_t[pdf_id],
                                           occupation_prob);
      }
      beta_t[h] = tot_variable_factor;
    }
  }
}


void GenericNumeratorComputation::AddSpecificPdfsIndirect(
                                 Matrix<BaseFloat> *logprobs,
                                 const std::vector<MatrixIndexT> &indices,
                                 CuMatrixBase<BaseFloat> *output) {
  const int32 num_sequences = supervision_.num_sequences,
              frames_per_sequence = supervision_.frames_per_sequence;

  const int view_stride = output->Stride() * num_sequences;

  KALDI_ASSERT(frames_per_sequence * num_sequences == output->NumRows());

  CuMatrix<BaseFloat> specific_pdfs;
  specific_pdfs.Swap(logprobs);
  specific_pdfs.ApplyExp();
  specific_pdfs.Scale(supervision_.weight);

  std::vector<MatrixIndexT> indices_expanded(view_stride, -1);
  for (int i = 0; i < indices.size(); ++i) {
    int pdf_index = indices[i];
    int sequence_local_pdf_index = pdf_index % nnet_output_stride_;
    int sequence_index = pdf_index / nnet_output_stride_;
    pdf_index = sequence_local_pdf_index
                + sequence_index * output->Stride();
    KALDI_ASSERT(pdf_index < view_stride);
    KALDI_ASSERT(i < specific_pdfs.NumCols());
    indices_expanded[pdf_index] = i;
  }

  CuArray<MatrixIndexT> cu_indices(indices_expanded);
  CuSubMatrix<BaseFloat> out(output->Data(), frames_per_sequence,
                             view_stride, view_stride);

  out.AddCols(specific_pdfs, cu_indices);
}

bool GenericNumeratorComputation::CheckValues(int seq,
                                            const Matrix<BaseFloat> &probs,
                                            const Matrix<BaseFloat> &alpha,
                                            const Matrix<BaseFloat> &beta,
                                            const Matrix<BaseFloat> &derivs) const {
  const int32 num_frames = supervision_.frames_per_sequence;
  // only check the derivs for the first and last frames
  const std::vector<int32> times = {0, num_frames - 1};
  for (const int32 t: times) {
    BaseFloat deriv_sum = 0.0;
    for (int32 n = 0; n < probs.NumCols(); n++) {
      int32 pdf_stride = nnet_output_.Stride();
      int32 pdf2seq = index_to_pdf_[n] / pdf_stride;
      if (pdf2seq != seq)  // this pdf is not in the space of this sequence
        continue;
      deriv_sum += Exp(derivs(t, n));
    }

    if (!ApproxEqual(deriv_sum, 1.0)) {
      KALDI_WARN << "On time " << t
                 << " for seq " << seq << ", deriv sum "
                 << deriv_sum << " != 1.0";
      if (fabs(deriv_sum - 1.0) > 0.05 || deriv_sum - deriv_sum != 0) {
        KALDI_WARN << "Excessive error detected, will abandon this minibatch";
        return false;
      }
    }
  }
  return true;
}

}  // namespace chain
}  // namespace kaldi

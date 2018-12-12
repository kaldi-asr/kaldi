// src/rnnlm/rnnlm-compute-state.cc

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)
//                2017  Yiming Wang
//                2017  Hainan Xu

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

#include "rnnlm/rnnlm-compute-state.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compile-looped.h"

namespace kaldi {
namespace rnnlm {

RnnlmComputeStateInfo::RnnlmComputeStateInfo(
    const RnnlmComputeStateComputationOptions &opts,
    const kaldi::nnet3::Nnet &rnnlm,
    const CuMatrix<BaseFloat> &word_embedding_mat):
    opts(opts), rnnlm(rnnlm), word_embedding_mat(word_embedding_mat) {
  KALDI_ASSERT(IsSimpleNnet(rnnlm));
  int32 left_context, right_context;
  ComputeSimpleNnetContext(rnnlm, &left_context, &right_context);
  if (0 != left_context || 0 != right_context) {
    KALDI_ERR << "Non-zero left or right context. Please check your script";
  }
  int32 frame_subsampling_factor = 1;
  int32 embedding_dim = word_embedding_mat.NumCols();
  if (embedding_dim != rnnlm.OutputDim("output")) {
    KALDI_ERR << "Embedding file and nnet have different embedding sizes. ";
  }

  if (opts.bos_index <= 0 || opts.bos_index >= word_embedding_mat.NumRows()) {
    KALDI_ERR << "--bos-symbol option isn't set correctly.";
  }

  if (opts.eos_index <= 0 || opts.eos_index >= word_embedding_mat.NumRows()) {
    KALDI_ERR << "--eos-symbol option isn't set correctly.";
  }

  nnet3::ComputationRequest request1, request2, request3;
  CreateLoopedComputationRequestSimple(rnnlm,
                                       1, // num_frames
                                       frame_subsampling_factor,
                                       1, // ivector_period = 1
                                       0, // extra_left_context_initial == 0
                                       0, // extra_right_context == 0
                                       1, // num_sequnces == 1
                                       &request1, &request2, &request3);

  CompileLooped(rnnlm, opts.optimize_config, request1, request2,
                request3, &computation);
  computation.ComputeCudaIndexes();
  if (GetVerboseLevel() >= 3) {
    KALDI_VLOG(3) << "Computation is:";
    computation.Print(std::cerr, rnnlm);
  }
}

RnnlmComputeState::RnnlmComputeState(const RnnlmComputeStateInfo &info,
                                     int32 bos_index) :
    info_(info),
    computer_(info_.opts.compute_config, info_.computation,
              info_.rnnlm, NULL),  // NULL is 'nnet_to_update'
    previous_word_(-1),
    normalization_factor_(0.0) {
  AddWord(bos_index);
}

RnnlmComputeState::RnnlmComputeState(const RnnlmComputeState &other):
  info_(other.info_), computer_(other.computer_),
  previous_word_(other.previous_word_),
  normalization_factor_(other.normalization_factor_)
{}

RnnlmComputeState* RnnlmComputeState::GetSuccessorState(int32 next_word) const {
  RnnlmComputeState *ans = new RnnlmComputeState(*this);
  ans->AddWord(next_word);
  return ans;
}

void RnnlmComputeState::AddWord(int32 word_index) {
  KALDI_ASSERT(word_index > 0 && word_index < info_.word_embedding_mat.NumRows());
  previous_word_ = word_index;
  AdvanceChunk();

  const CuMatrix<BaseFloat> &word_embedding_mat = info_.word_embedding_mat;
  if (info_.opts.normalize_probs) {
    CuVector<BaseFloat> log_probs(info_.word_embedding_mat.NumRows());

    log_probs.AddMatVec(1.0, word_embedding_mat, kNoTrans,
                        predicted_word_embedding_->Row(0), 0.0);
    log_probs.ApplyExp();

    // We excluding the <eps> symbol which is always 0.
    normalization_factor_ = log(log_probs.Range(1, log_probs.Dim() - 1).Sum());
  }
}

BaseFloat RnnlmComputeState::LogProbOfWord(int32 word_index) const {
  const CuMatrix<BaseFloat> &word_embedding_mat = info_.word_embedding_mat;

  BaseFloat log_prob = VecVec(predicted_word_embedding_->Row(0),
                              word_embedding_mat.Row(word_index));

  // Even without explicit normalization, the log-probs will be close to
  // correctly normalized due to the way the model was trained.
  if (info_.opts.normalize_probs) {
    log_prob -= normalization_factor_;
  }
  return log_prob;
}

void RnnlmComputeState::GetLogProbOfWords(CuMatrixBase<BaseFloat> *output) const {
  const CuMatrix<BaseFloat> &word_embedding_mat = info_.word_embedding_mat;

  KALDI_ASSERT(output->NumRows() == 1
                && output->NumCols() == word_embedding_mat.NumCols());
  output->Row(0).AddMatVec(1.0, word_embedding_mat, kNoTrans,
                   predicted_word_embedding_->Row(0), 0.0);

  // Even without explicit normalization, the log-probs will be close to
  // correctly normalized due to the way the model was trained.
  if (info_.opts.normalize_probs) {
    output->Add(normalization_factor_);
  }

  // making sure <eps> has almost 0 prob
  output->ColRange(0, 1).Set(-99.0);
}

void RnnlmComputeState::AdvanceChunk() {
  CuMatrix<BaseFloat> input_embeddings(1, info_.word_embedding_mat.NumCols());
  input_embeddings.Row(0).AddVec(1.0,
                                 info_.word_embedding_mat.Row(previous_word_));
  computer_.AcceptInput("input", &input_embeddings);
  computer_.Run();
  {
    // Note: here GetOutput() is used instead of GetOutputDestructive(), since
    // here we have recurrence that goes directly from the output, and the call
    // to GetOutputDestructive() would cause a crash on the next chunk.
    const CuMatrixBase<BaseFloat> &output(computer_.GetOutput("output"));
    predicted_word_embedding_ = &output;
  }
}

RnnlmComputeStateInfoAdapt::RnnlmComputeStateInfoAdapt(
    const RnnlmComputeStateComputationOptions &opts,
    const kaldi::nnet3::Nnet &rnnlm,
    const CuMatrix<BaseFloat> &word_embedding_mat_large,
    const CuMatrix<BaseFloat> &word_embedding_mat_med,
    const CuMatrix<BaseFloat> &word_embedding_mat_small,
    const int32 cutofflarge,
    const int32 cutoffmed):
    opts(opts), rnnlm(rnnlm), word_embedding_mat_large(word_embedding_mat_large), word_embedding_mat_med(word_embedding_mat_med),
    word_embedding_mat_small(word_embedding_mat_small), cutoff_large(cutofflarge), cutoff_med(cutoffmed) {

  KALDI_ASSERT(IsSimpleNnet(rnnlm));
  int32 left_context, right_context;
  ComputeSimpleNnetContext(rnnlm, &left_context, &right_context);
  if (0 != left_context || 0 != right_context) {
    KALDI_ERR << "Non-zero left or right context. Please check your script";
  }
  int32 frame_subsampling_factor = 1;
  int32 embedding_dim = word_embedding_mat_large.NumCols();
  if (embedding_dim != rnnlm.OutputDim("output")) {
    KALDI_ERR << "Embedding file and nnet have different embedding sizes. ";
  }
  embedding_dim = word_embedding_mat_med.NumCols();
  if (embedding_dim != rnnlm.OutputDim("outputmed")) {
    KALDI_ERR << "Embedding file and nnet have different embedding sizes. ";
  }
  embedding_dim = word_embedding_mat_small.NumCols();
  if (embedding_dim != rnnlm.OutputDim("outputsmall")) {
    KALDI_ERR << "Embedding file and nnet have different embedding sizes. ";
  }

  total_emb_size_ = word_embedding_mat_large.NumRows() + word_embedding_mat_med.NumRows() + word_embedding_mat_small.NumRows();

  if (opts.bos_index <= 0 || opts.bos_index >= total_emb_size_) {
    KALDI_ERR << "--bos-symbol option isn't set correctly.";
  }

  if (opts.eos_index <= 0 || opts.eos_index >= total_emb_size_) {
    KALDI_ERR << "--eos-symbol option isn't set correctly.";
  }

  nnet3::ComputationRequest request1, request2, request3;
  CreateLoopedComputationRequestSimple(rnnlm,
                                       1, // num_frames
                                       frame_subsampling_factor,
                                       1, // ivector_period = 1
                                       0, // extra_left_context_initial == 0
                                       0, // extra_right_context == 0
                                       1, // num_sequnces == 1
                                       &request1, &request2, &request3);

  CompileLooped(rnnlm, opts.optimize_config, request1, request2,
                request3, &computation);
  computation.ComputeCudaIndexes();
  if (GetVerboseLevel() >= 3) {
    KALDI_VLOG(3) << "Computation is:";
    computation.Print(std::cerr, rnnlm);
  }
}

RnnlmComputeStateAdapt::RnnlmComputeStateAdapt(const RnnlmComputeStateInfoAdapt &info,
                                     int32 bos_index):
    info_(info),
    computer_(info_.opts.compute_config, info_.computation,
              info_.rnnlm, NULL),  // NULL is 'nnet_to_update'
    previous_word_(-1),
    normalization_factor_(0.0) {
  AddWord(bos_index);
}

RnnlmComputeStateAdapt::RnnlmComputeStateAdapt(const RnnlmComputeStateAdapt &other):
  info_(other.info_), computer_(other.computer_),
  previous_word_(other.previous_word_),
  normalization_factor_(other.normalization_factor_)
{}

RnnlmComputeStateAdapt* RnnlmComputeStateAdapt::GetSuccessorState(int32 next_word) const {
  RnnlmComputeStateAdapt *ans = new RnnlmComputeStateAdapt(*this);
  ans->AddWord(next_word);
  return ans;
}

void RnnlmComputeStateAdapt::AddWord(int32 word_index) {
  KALDI_ASSERT(word_index > 0 && word_index < total_emb_size_);
  previous_word_ = word_index;
  AdvanceChunk();

  if (info_.opts.normalize_probs) {
    const CuMatrix<BaseFloat> &word_embedding_mat_large = info_.word_embedding_mat_large;
    const CuMatrix<BaseFloat> &word_embedding_mat_med = info_.word_embedding_mat_med;
    const CuMatrix<BaseFloat> &word_embedding_mat_small = info_.word_embedding_mat_small;
    CuVector<BaseFloat> log_probs(total_emb_size_, kUndefined);
    int32 large_size = word_embedding_mat_large.NumRows();
    int32 med_size = word_embedding_mat_med.NumRows();
    int32 small_size = word_embedding_mat_small.NumRows();
    CuVector<BaseFloat> log_probs_large(large_size, kUndefined);
    CuVector<BaseFloat> log_probs_med(med_size, kUndefined);
    CuVector<BaseFloat> log_probs_small(small_size, kUndefined);

    log_probs_large.AddMatVec(1.0, word_embedding_mat_large, kNoTrans,
                        predicted_word_embedding_large_->Row(0), 0.0);
    log_probs_med.AddMatVec(1.0, word_embedding_mat_med, kNoTrans,
                        predicted_word_embedding_med_->Row(0), 0.0);
    log_probs_small.AddMatVec(1.0, word_embedding_mat_small, kNoTrans,
                        predicted_word_embedding_small_->Row(0), 0.0);

    log_probs.Range(0, large_size).CopyFromVec(log_probs_large);
    log_probs.Range(large_size, med_size).CopyFromVec(log_probs_med);
    log_probs.Range(large_size + med_size, small_size).CopyFromVec(log_probs_small);

    log_probs.ApplyExp();

    // We excluding the <eps> symbol which is always 0.
    normalization_factor_ = log(log_probs.Range(1, log_probs.Dim() - 1).Sum());
  }
}

BaseFloat RnnlmComputeStateAdapt::LogProbOfWord(int32 word_index) const {
  const CuMatrix<BaseFloat> &word_embedding_mat_large = info_.word_embedding_mat_large;
  const CuMatrix<BaseFloat> &word_embedding_mat_med = info_.word_embedding_mat_med;
  const CuMatrix<BaseFloat> &word_embedding_mat_small = info_.word_embedding_mat_small;
  int32 cutoff_large = info_.cutoff_large;
  int32 cutoff_med = info_.cutoff_med;
  BaseFloat log_prob;
  if (word_index < cutoff_large) {
    log_prob = VecVec(predicted_word_embedding_large_->Row(0),
                                word_embedding_mat_large.Row(word_index));
  } else if (word_index < (cutoff_large + cutoff_med)) {
    log_prob = VecVec(predicted_word_embedding_med_->Row(0),
                                word_embedding_mat_med.Row(word_index - cutoff_large));
  } else {
    log_prob = VecVec(predicted_word_embedding_small_->Row(0),
                                word_embedding_mat_small.Row(word_index - cutoff_large - cutoff_med));
  }

  // Even without explicit normalization, the log-probs will be close to
  // correctly normalized due to the way the model was trained.
  if (info_.opts.normalize_probs) {
    log_prob -= normalization_factor_;
  }
  return log_prob;
}


void RnnlmComputeStateAdapt::AdvanceChunk() {
  CuMatrix<BaseFloat> input_embeddings_large(1, info_.word_embedding_mat_large.NumCols());
  CuMatrix<BaseFloat> input_embeddings_med(1, info_.word_embedding_mat_med.NumCols());
  CuMatrix<BaseFloat> input_embeddings_small(1, info_.word_embedding_mat_small.NumCols());
  int32 cutoff_large = info_.cutoff_large;
  int32 cutoff_med = info_.cutoff_med;

  if (previous_word_ < cutoff_large) {
    input_embeddings_large.Row(0).AddVec(1.0,
                                 info_.word_embedding_mat_large.Row(previous_word_));
  } else if (previous_word_ < (cutoff_large + cutoff_med)) {
    input_embeddings_med.Row(0).AddVec(1.0,
                                 info_.word_embedding_mat_med.Row(previous_word_ - cutoff_large));
  } else {
    input_embeddings_small.Row(0).AddVec(1.0,
                                 info_.word_embedding_mat_small.Row(previous_word_ - cutoff_large - cutoff_med));
  }

  computer_.AcceptInput("input", &input_embeddings_large);
  computer_.AcceptInput("inputmed", &input_embeddings_med);
  computer_.AcceptInput("inputsmall", &input_embeddings_small);
  computer_.Run();
  {
    // Note: here GetOutput() is used instead of GetOutputDestructive(), since
    // here we have recurrence that goes directly from the output, and the call
    // to GetOutputDestructive() would cause a crash on the next chunk.
    const CuMatrixBase<BaseFloat> &output(computer_.GetOutput("output"));
    predicted_word_embedding_large_ = &output;
    const CuMatrixBase<BaseFloat> &outputmed(computer_.GetOutput("outputmed"));
    predicted_word_embedding_med_ = &outputmed;
    const CuMatrixBase<BaseFloat> &outputsmall(computer_.GetOutput("outputsmall"));
    predicted_word_embedding_small_ = &outputsmall;

  }
}

} // namespace rnnlm
} // namespace kaldi

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

} // namespace rnnlm
} // namespace kaldi

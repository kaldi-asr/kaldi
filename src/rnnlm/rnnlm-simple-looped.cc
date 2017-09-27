// rnnlm/kaldi-rnnlm-simple-looped.cc

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

#include "rnnlm/rnnlm-simple-looped.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compile-looped.h"

namespace kaldi {
namespace nnet3 {


RnnlmComputeStateInfo::RnnlmComputeStateInfo(
    const RnnlmComputeStateComputationOptions &opts,
    const kaldi::nnet3::Nnet &rnnlm,
    const CuMatrix<BaseFloat> &word_embedding_mat):
    opts(opts), rnnlm(rnnlm), word_embedding_mat(word_embedding_mat) {
  Init(opts, rnnlm, word_embedding_mat);
}

void RnnlmComputeStateInfo::Init(
    const RnnlmComputeStateComputationOptions &opts,
    const kaldi::nnet3::Nnet &rnnlm,
    const CuMatrix<BaseFloat> &word_embedding_mat) {
  opts.Check();
  KALDI_ASSERT(IsSimpleNnet(rnnlm));
  int32 left_context, right_context;
  ComputeSimpleNnetContext(rnnlm, &left_context, &right_context);
  KALDI_ASSERT(0 == left_context);
  KALDI_ASSERT(0 == right_context);
  int32 frame_subsampling_factor = 1;
  nnet_output_dim = rnnlm.OutputDim("output");
  KALDI_ASSERT(nnet_output_dim > 0);

  ComputationRequest request1, request2, request3;
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

RnnlmComputeState::RnnlmComputeState(
    const RnnlmComputeStateInfo &info) :
    info_(info),
    computer_(info_.opts.compute_config, info_.computation,
              info_.rnnlm, NULL),  // NULL is 'nnet_to_update'
    feats_(-1),
    current_log_post_offset_(-1)
{}

RnnlmComputeState::RnnlmComputeState(const RnnlmComputeState &other):
  info_(other.info_), computer_(other.computer_), feats_(other.feats_),
  current_nnet_output_(other.current_nnet_output_),
  current_log_post_offset_(other.current_log_post_offset_)
{}

void RnnlmComputeState::TakeFeatures(int32 word_index) {
  feats_ = word_index;
  current_log_post_offset_ = -1;
}

BaseFloat RnnlmComputeState::LogProbOfWord(int32 word_index,
                               const CuVectorBase<BaseFloat> &hidden) const {
  const CuMatrix<BaseFloat> &word_embedding_mat = info_.word_embedding_mat;
  BaseFloat log_prob;

  if (info_.opts.force_normalize) {
    CuVector<BaseFloat> log_probs(word_embedding_mat.NumRows());

    log_probs.AddMatVec(1.0, word_embedding_mat, kTrans, hidden, 0.0);
    log_probs.ApplySoftMax();
    log_probs.ApplyLog();
    log_prob = log_probs(word_index);
  } else {
    log_prob = VecVec(hidden, word_embedding_mat.Row(word_index));
  }
  return log_prob;
}

CuVector<BaseFloat>* RnnlmComputeState::GetOutput() {
  AdvanceChunk();
  CuMatrix<BaseFloat> current_nnet_output_gpu;
  current_nnet_output_gpu.Swap(&current_nnet_output_);
  const CuSubVector<BaseFloat> hidden(current_nnet_output_gpu,
                                      -current_log_post_offset_);
  return new CuVector<BaseFloat>(hidden);
}

void RnnlmComputeState::AdvanceChunk() {
  CuMatrix<BaseFloat> input_embeddings(1, info_.word_embedding_mat.NumCols());
  int32 word_index = feats_;
  input_embeddings.RowRange(0, 1).AddMat(1.0, info_.word_embedding_mat.RowRange(word_index, 1), kNoTrans);
  computer_.AcceptInput("input", &input_embeddings);

  computer_.Run();

  {
    // Note: here GetOutput() is used instead of GetOutputDestructive(), since
    // here we have recurrence that goes directly from the output, and the call
    // to GetOutputDestructive() would cause a crash on the next chunk.
    CuMatrix<BaseFloat> output(computer_.GetOutput("output"));

    current_nnet_output_.Resize(0, 0);
    current_nnet_output_.Swap(&output);
  }
  KALDI_ASSERT(current_nnet_output_.NumRows() == 1 &&
               current_nnet_output_.NumCols() == info_.nnet_output_dim);

  current_log_post_offset_ = 0;
}

} // namespace nnet3
} // namespace kaldi

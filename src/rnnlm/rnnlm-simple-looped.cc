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


RnnlmSimpleLoopedInfo::RnnlmSimpleLoopedInfo(
    const RnnlmSimpleLoopedComputationOptions &opts,
    const kaldi::nnet3::Nnet &rnnlm,
    const CuMatrix<BaseFloat> &word_embedding_mat):
    opts(opts), rnnlm(rnnlm), word_embedding_mat(word_embedding_mat) {
  Init(opts, rnnlm, word_embedding_mat);
}

void RnnlmSimpleLoopedInfo::Init(
    const RnnlmSimpleLoopedComputationOptions &opts,
    const kaldi::nnet3::Nnet &rnnlm,
    const CuMatrix<BaseFloat> &word_embedding_mat) {
  opts.Check();
  KALDI_ASSERT(IsSimpleNnet(rnnlm));
  int32 left_context, right_context;
  ComputeSimpleNnetContext(rnnlm, &left_context, &right_context);
  frames_left_context = left_context;
  frames_right_context = right_context;
  int32 frame_subsampling_factor = 1;
  frames_per_chunk = GetChunkSize(rnnlm, frame_subsampling_factor,
                                  opts.frames_per_chunk);
  KALDI_ASSERT(frames_per_chunk == opts.frames_per_chunk);
  nnet_output_dim = rnnlm.OutputDim("output");
  KALDI_ASSERT(nnet_output_dim > 0);

  int32 ivector_period = frames_per_chunk;
  int32 extra_right_context = 0;
  int32 num_sequences = 1;  // we're processing one word sequence at a time.
  CreateLoopedComputationRequestSimple(rnnlm, frames_per_chunk,
                                       frame_subsampling_factor,
                                       ivector_period,
                                       0, // extra_left_context_initial == 0
                                       extra_right_context,
                                       num_sequences,
                                       &request1, &request2, &request3);

  CompileLooped(rnnlm, opts.optimize_config, request1, request2,
                request3, &computation);
  computation.ComputeCudaIndexes();
  if (GetVerboseLevel() >= 3) {
    KALDI_VLOG(3) << "Computation is:";
    computation.Print(std::cerr, rnnlm);
  }
}

RnnlmSimpleLooped::RnnlmSimpleLooped(
    const RnnlmSimpleLoopedInfo &info) :
    info_(info),
    computer_(info_.opts.compute_config, info_.computation,
              info_.rnnlm, NULL),  // NULL is 'nnet_to_update'
    // since everytime we provide one chunk to the object, the size of
    // feats_ == frames_per_chunk
    feats_(info_.frames_per_chunk,
           info_.word_embedding_mat.NumRows()),
    current_log_post_offset_(-1)
{}

RnnlmSimpleLooped::RnnlmSimpleLooped(const RnnlmSimpleLooped &other):
  info_(other.info_), computer_(other.computer_), feats_(other.feats_),
  current_nnet_output_(other.current_nnet_output_),
  current_log_post_offset_(other.current_log_post_offset_)
{}

void RnnlmSimpleLooped::TakeFeatures(
    const std::vector<int32> &word_indexes) {
  KALDI_ASSERT(word_indexes.size() == feats_.NumRows());
  std::vector<std::vector<std::pair<MatrixIndexT, BaseFloat> > >
      pairs(word_indexes.size());
  for (int32 i = 0; i < word_indexes.size(); i++) {
    std::pair<MatrixIndexT, BaseFloat> one_hot_index(word_indexes[i], 1.0);
    std::vector<std::pair<MatrixIndexT, BaseFloat> > row(1, one_hot_index);
    pairs[i] = row;
  }
  SparseMatrix<BaseFloat> feats_temp(feats_.NumCols(), pairs);
  feats_.Swap(&feats_temp);
  // resets offset so that AdvanceChunk() would be called in GetOutput() and
  // GetNnetOutputForFrame() after taking new features
  current_log_post_offset_ = -1;
}

//void RnnlmSimpleLooped::GetNnetOutputForFrame(
//    int32 frame, VectorBase<BaseFloat> *output) {
//  KALDI_ASSERT(frame >= 0 && frame < feats_.NumRows());
//  if (frame >= current_log_post_offset_ + current_nnet_output_.NumRows())
//    AdvanceChunk();
//  output->CopyFromVec(current_nnet_output_.Row(frame -
//                                               current_log_post_offset_));
//}

BaseFloat RnnlmSimpleLooped::LogProbOfWord(int32 word_index,
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

CuVector<BaseFloat>* RnnlmSimpleLooped::GetOutput(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame < feats_.NumRows());
  if (frame >= current_log_post_offset_ + current_nnet_output_.NumRows())
    AdvanceChunk();

//  int32 embedding_dim = info_.word_embedding_mat.NumCols();
//  int32 num_words = info_.word_embedding_mat.NumRows();


  CuMatrix<BaseFloat> current_nnet_output_gpu;
  current_nnet_output_gpu.Swap(&current_nnet_output_);
  const CuSubVector<BaseFloat> hidden(current_nnet_output_gpu,
                                      frame - current_log_post_offset_);
  return new CuVector<BaseFloat>(hidden);
//
//  // swap the pointer back so that this function can be called multiple times
//  // with the same returned value before taking next new feats
//  current_nnet_output_.Swap(&current_nnet_output_gpu);
//  return log_prob;
}

void RnnlmSimpleLooped::AdvanceChunk() {
  int32 begin_input_frame, end_input_frame;
  begin_input_frame = -info_.frames_left_context;
  // note: end is last plus one.
  end_input_frame = info_.frames_per_chunk + info_.frames_right_context;
  // currently there is no left/right context and frames_per_chunk == 1
  KALDI_ASSERT(begin_input_frame == 0 && end_input_frame == 1);

  SparseMatrix<BaseFloat> feats_chunk(end_input_frame - begin_input_frame,
                                      feats_.NumCols());
  int32 num_features = feats_.NumRows();
  for (int32 r = begin_input_frame; r < end_input_frame; r++) {
    int32 input_frame = r;
    if (input_frame < 0) input_frame = 0;
    if (input_frame >= num_features) input_frame = num_features - 1;
    feats_chunk.SetRow(r - begin_input_frame, feats_.Row(input_frame));
  }

  CuMatrix<BaseFloat> input_embeddings(1, info_.word_embedding_mat.NumCols());
  int32 word_index = feats_chunk.Row(0).GetElement(0).first;
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
  KALDI_ASSERT(current_nnet_output_.NumRows() == info_.frames_per_chunk &&
               current_nnet_output_.NumCols() == info_.nnet_output_dim);

  current_log_post_offset_ = 0;
}


} // namespace nnet3
} // namespace kaldi

// nnet2/online-nnet2-decodable.cc

// Copyright  2014  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/online-nnet2-decodable.h"

namespace kaldi {
namespace nnet2 {

DecodableNnet2Online::DecodableNnet2Online(
    const AmNnet &nnet,
    const TransitionModel &trans_model,
    const DecodableNnet2OnlineOptions &opts,
    OnlineFeatureInterface *input_feats):
    features_(input_feats),
    nnet_(nnet),
    trans_model_(trans_model),
    opts_(opts),
    feat_dim_(input_feats->Dim()),
    left_context_(nnet.GetNnet().LeftContext()),
    right_context_(nnet.GetNnet().RightContext()),
    num_pdfs_(nnet.GetNnet().OutputDim()),
    begin_frame_(-1) {
  KALDI_ASSERT(opts_.max_nnet_batch_size > 0);
  log_priors_ = nnet_.Priors();
  KALDI_ASSERT(log_priors_.Dim() == trans_model_.NumPdfs() &&
               "Priors in neural network not set up (or mismatch "
               "with transition model).");
  log_priors_.ApplyLog();
}



BaseFloat DecodableNnet2Online::LogLikelihood(int32 frame, int32 index) {
  ComputeForFrame(frame);
  int32 pdf_id = trans_model_.TransitionIdToPdf(index);
  KALDI_ASSERT(frame >= begin_frame_ &&
               frame < begin_frame_ + scaled_loglikes_.NumRows());
  return scaled_loglikes_(frame - begin_frame_, pdf_id);
}


bool DecodableNnet2Online::IsLastFrame(int32 frame) const {
  if (opts_.pad_input) { // normal case
    return features_->IsLastFrame(frame);
  } else {
    return features_->IsLastFrame(frame + left_context_ + right_context_);
  }
}

int32 DecodableNnet2Online::NumFramesReady() const {
  int32 features_ready = features_->NumFramesReady();
  if (features_ready == 0)
    return 0;
  bool input_finished = features_->IsLastFrame(features_ready - 1);
  if (opts_.pad_input) {
    // normal case... we'll pad with duplicates of first + last frame to get the
    // required left and right context.
    if (input_finished) return features_ready;
    else return std::max<int32>(0, features_ready - right_context_);
  } else {
    return std::max<int32>(0, features_ready - right_context_ - left_context_);
  }
}

void DecodableNnet2Online::ComputeForFrame(int32 frame) {
  int32 features_ready = features_->NumFramesReady();
  bool input_finished = features_->IsLastFrame(features_ready - 1);
  KALDI_ASSERT(frame >= 0);
  if (frame >= begin_frame_ &&
      frame < begin_frame_ + scaled_loglikes_.NumRows())
    return;
  KALDI_ASSERT(frame < NumFramesReady());

  int32 input_frame_begin;
  if (opts_.pad_input)
    input_frame_begin = frame - left_context_;
  else
    input_frame_begin = frame;
  int32 max_possible_input_frame_end = features_ready;
  if (input_finished && opts_.pad_input)
    max_possible_input_frame_end += right_context_;
  int32 input_frame_end = std::min<int32>(max_possible_input_frame_end,
                                          input_frame_begin +
                                          left_context_ + right_context_ +
                                          opts_.max_nnet_batch_size);
  KALDI_ASSERT(input_frame_end > input_frame_begin);
  Matrix<BaseFloat> features(input_frame_end - input_frame_begin,
                             feat_dim_);
  for (int32 t = input_frame_begin; t < input_frame_end; t++) {
    SubVector<BaseFloat> row(features, t - input_frame_begin);
    int32 t_modified = t;
    // The next two if-statements take care of "pad_input"
    if (t_modified < 0)
      t_modified = 0;
    if (t_modified >= features_ready)
      t_modified = features_ready - 1;
    features_->GetFrame(t_modified, &row);
  }
  CuMatrix<BaseFloat> cu_features;
  cu_features.Swap(&features);  // Copy to GPU, if we're using one.


  int32 num_frames_out = input_frame_end - input_frame_begin -
      left_context_ - right_context_;

  CuMatrix<BaseFloat> cu_posteriors(num_frames_out, num_pdfs_);

  // The "false" below tells it not to pad the input: we've already done
  // any padding that we needed to do.
  NnetComputation(nnet_.GetNnet(), cu_features,
                  false, &cu_posteriors);

  cu_posteriors.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
  cu_posteriors.ApplyLog();
  // subtract log-prior (divide by prior)
  cu_posteriors.AddVecToRows(-1.0, log_priors_);
  // apply probability scale.
  cu_posteriors.Scale(opts_.acoustic_scale);

  // Transfer the scores the CPU for faster access by the
  // decoding process.
  scaled_loglikes_.Resize(0, 0);
  cu_posteriors.Swap(&scaled_loglikes_);

  begin_frame_ = frame;
}

} // namespace nnet2
} // namespace kaldi

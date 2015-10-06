// nnet3/nnet-am-decodable-simple.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Vimal Manohar

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

#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-simple-computer.h"

namespace kaldi {
namespace nnet3 {


DecodableAmNnetSimple::DecodableAmNnetSimple(
    const DecodableAmNnetSimpleOptions &opts,
    const TransitionModel &trans_model,
    const AmNnetSimple &am_nnet,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    NnetSimpleComputer(opts.simple_computer_opts, am_nnet.GetNnet(), feats, 
        am_nnet.LeftContext(), am_nnet.RightContext(), 
        ivector, online_ivectors, online_ivector_period),
    opts_(opts),
    trans_model_(trans_model),
    am_nnet_(am_nnet),
    priors_(am_nnet.Priors()) {
  priors_.ApplyLog();
}

DecodableAmNnetSimple::DecodableAmNnetSimple(
    const DecodableAmNnetSimpleOptions &opts,
    const TransitionModel &trans_model,
    const AmNnetSimple &am_nnet,
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> &ivectors,
    int32 online_ivector_period):
    NnetSimpleComputer(opts.simple_computer_opts, am_nnet.GetNnet(), feats, 
        am_nnet.LeftContext(), am_nnet.RightContext(),
        NULL, &ivectors, online_ivector_period),
    opts_(opts),
    trans_model_(trans_model),
    am_nnet_(am_nnet),
    priors_(am_nnet.Priors()) {
  priors_.ApplyLog();
}

DecodableAmNnetSimple::DecodableAmNnetSimple(
    const DecodableAmNnetSimpleOptions &opts,
    const TransitionModel &trans_model,
    const AmNnetSimple &am_nnet,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> &ivector):
    NnetSimpleComputer(opts.simple_computer_opts, am_nnet.GetNnet(), feats, 
        am_nnet.LeftContext(), am_nnet.RightContext(), 
        &ivector, NULL, 0), 
    opts_(opts),
    trans_model_(trans_model),
    am_nnet_(am_nnet),
    priors_(am_nnet.Priors()) {
  priors_.ApplyLog();
}

BaseFloat DecodableAmNnetSimple::LogLikelihood(int32 frame,
                                               int32 transition_id) {
  if (frame < current_log_post_offset_ ||
      frame >= current_log_post_offset_ + current_log_post_.NumRows())
    EnsureFrameIsComputed(frame);
  int32 pdf_id = trans_model_.TransitionIdToPdf(transition_id);
  return current_log_post_(frame - current_log_post_offset_,
                           pdf_id);
}

void DecodableAmNnetSimple::DoNnetComputation(
    int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    const VectorBase<BaseFloat> &ivector,
    int32 output_t_start,
    int32 num_output_frames) {
  CuMatrix<BaseFloat> cu_output;
  DoNnetComputationInternal(input_t_start, input_feats, ivector, 
                            output_t_start, num_output_frames, &cu_output);
  // subtract log-prior (divide by prior)
  cu_output.AddVecToRows(-1.0, priors_);
  // apply the acoustic scale
  cu_output.Scale(opts_.acoustic_scale);
  current_log_post_.Resize(0, 0);
  // the following statement just swaps the pointers if we're not using a GPU.
  cu_output.Swap(&current_log_post_);
  current_log_post_offset_ = output_t_start;
}

} // namespace nnet3
} // namespace kaldi

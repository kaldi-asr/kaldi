// nnet3/online-nnet3-decodable.cc

// Copyright  2014  Johns Hopkins University (author: Daniel Povey)
//            2016  Api.ai (Author: Ilya Platonov)

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

#include "nnet3/online-nnet3-decodable.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

DecodableNnet3Online::DecodableNnet3Online(
    const AmNnetSimple &nnet,
    const TransitionModel &trans_model,
    const DecodableNnet3OnlineOptions &opts,
    OnlineFeatureInterface *input_feats):
    compiler_(nnet.GetNnet(), opts_.optimize_config),
    features_(input_feats),
    nnet_(nnet),
    trans_model_(trans_model),
    opts_(opts),
    feat_dim_(input_feats->Dim()),
    num_pdfs_(nnet.GetNnet().OutputDim("output")),
    begin_frame_(-1) {
  KALDI_ASSERT(opts_.max_nnet_batch_size > 0);
  log_priors_ = nnet_.Priors();
  KALDI_ASSERT((log_priors_.Dim() == 0 || log_priors_.Dim() == trans_model_.NumPdfs()) &&
               "Priors in neural network must match with transition model (if exist).");

  ComputeSimpleNnetContext(nnet_.GetNnet(), &left_context_, &right_context_);
  log_priors_.ApplyLog();
}



BaseFloat DecodableNnet3Online::LogLikelihood(int32 frame, int32 index) {
  ComputeForFrame(frame);
  int32 pdf_id = trans_model_.TransitionIdToPdf(index);
  KALDI_ASSERT(frame >= begin_frame_ &&
               frame < begin_frame_ + scaled_loglikes_.NumRows());
  return scaled_loglikes_(frame - begin_frame_, pdf_id);
}


bool DecodableNnet3Online::IsLastFrame(int32 frame) const {
  KALDI_ASSERT(false && "Method is not imlemented");
  return false;
}

int32 DecodableNnet3Online::NumFramesReady() const {
  int32 features_ready = features_->NumFramesReady();
  if (features_ready == 0)
    return 0;
  bool input_finished = features_->IsLastFrame(features_ready - 1);
  if (opts_.pad_input) {
    // normal case... we'll pad with duplicates of first + last frame to get the
    // required left and right context.
    if (input_finished) return subsampling(features_ready);
    else return std::max<int32>(0, subsampling(features_ready - right_context_));
  } else {
    return std::max<int32>(0, subsampling(features_ready - right_context_ - left_context_));
  }
}

int32 DecodableNnet3Online::subsampling(int32 num_frames) const {
  return (num_frames) / opts_.frame_subsampling_factor;
}

void DecodableNnet3Online::ComputeForFrame(int32 subsampled_frame) {
  int32 features_ready = features_->NumFramesReady();
  bool input_finished = features_->IsLastFrame(features_ready - 1);  
  KALDI_ASSERT(subsampled_frame >= 0);
  if (subsampled_frame >= begin_frame_ &&
      subsampled_frame < begin_frame_ + scaled_loglikes_.NumRows())
    return;
  KALDI_ASSERT(subsampled_frame < NumFramesReady());

  int32 subsample = opts_.frame_subsampling_factor;

  int32 input_frame_begin;
  if (opts_.pad_input)
    input_frame_begin = subsampled_frame * subsample  - left_context_;
  else
    input_frame_begin = subsampled_frame * subsample;
  int32 max_possible_input_frame_end = features_ready /* - ( features_ready - right_context_) % subsample */;
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

  int32 num_subsampled_frames = subsampling(input_frame_end - input_frame_begin -
          left_context_ - right_context_);
  DoNnetComputation(input_frame_begin,
      features, subsampled_frame * subsample, num_subsampled_frames);

  begin_frame_ = subsampled_frame;
}

void DecodableNnet3Online::DoNnetComputation(
    int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    int32 output_t_start,
    int32 num_subsampled_frames) {
  ComputationRequest request;
  request.need_model_derivative = false;
  request.store_component_stats = false;

  bool shift_time = true; // shift the 'input' and 'output' to a consistent
                          // time, to take advantage of caching in the compiler.
                          // An optimization.
  int32 time_offset = (shift_time ? -output_t_start : 0);

  // First add the regular features-- named "input".
  request.inputs.reserve(2);
  request.inputs.push_back(
      IoSpecification("input", time_offset + input_t_start,
                      time_offset + input_t_start + input_feats.NumRows()));
  IoSpecification output_spec;
  output_spec.name = "output";
  output_spec.has_deriv = false;
  int32 subsample = opts_.frame_subsampling_factor;
  output_spec.indexes.resize(num_subsampled_frames);
  // leave n and x values at 0 (the constructor sets these).
  for (int32 i = 0; i < num_subsampled_frames; i++)
    output_spec.indexes[i].t = time_offset + output_t_start + i * subsample;
  request.outputs.resize(1);
  request.outputs[0].Swap(&output_spec);

  const NnetComputation *computation = compiler_.Compile(request);
  Nnet *nnet_to_update = NULL;  // we're not doing any update.
  NnetComputer computer(opts_.compute_config, *computation,
                        nnet_.GetNnet(), nnet_to_update);

  CuMatrix<BaseFloat> input_feats_cu(input_feats);
  computer.AcceptInput("input", &input_feats_cu);
  CuMatrix<BaseFloat> ivector_feats_cu;
  computer.Forward();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  // subtract log-prior (divide by prior)
  if (log_priors_.Dim() != 0)
    cu_output.AddVecToRows(-1.0, log_priors_);
  // apply the acoustic scale
  cu_output.Scale(opts_.acoustic_scale);
  scaled_loglikes_.Resize(0, 0);
  // the following statement just swaps the pointers if we're not using a GPU.
  cu_output.Swap(&scaled_loglikes_);
//  current_log_post_subsampled_offset_ = output_t_start / subsample;
}

} // namespace nnet3
} // namespace kaldi

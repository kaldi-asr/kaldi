// nnet3/nnet-am-decodable-simple.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {


DecodableNnetSimple::DecodableNnetSimple(
    const NnetSimpleComputationOptions &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors,
    const MatrixBase<BaseFloat> &feats,
    CachingOptimizingCompiler *compiler,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    opts_(opts),
    nnet_(nnet),
    output_dim_(nnet_.OutputDim("output")),
    log_priors_(priors),
    feats_(feats),
    ivector_(ivector), online_ivector_feats_(online_ivectors),
    online_ivector_period_(online_ivector_period),
    compiler_(*compiler),
    current_log_post_subsampled_offset_(0) {
  num_subsampled_frames_ =
      (feats_.NumRows() + opts_.frame_subsampling_factor - 1) /
      opts_.frame_subsampling_factor;
  KALDI_ASSERT(IsSimpleNnet(nnet));
  ComputeSimpleNnetContext(nnet, &nnet_left_context_, &nnet_right_context_);
  KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
  KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period <= 0 &&
                 "You need to set the --online-ivector-period option!"));
  log_priors_.ApplyLog();
  CheckAndFixConfigs();
}


DecodableAmNnetSimple::DecodableAmNnetSimple(
    const NnetSimpleComputationOptions &opts,
    const TransitionModel &trans_model,
    const AmNnetSimple &am_nnet,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period,
    CachingOptimizingCompiler *compiler):
    compiler_(am_nnet.GetNnet(), opts.optimize_config),
    decodable_nnet_(opts, am_nnet.GetNnet(), am_nnet.Priors(),
                    feats, compiler != NULL ? compiler : &compiler_,
                    ivector, online_ivectors,
                    online_ivector_period),
    trans_model_(trans_model) {
  // note: we only use compiler_ if the passed-in 'compiler' is NULL.
}



BaseFloat DecodableAmNnetSimple::LogLikelihood(int32 frame,
                                               int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdf(transition_id);
  return decodable_nnet_.GetOutput(frame, pdf_id);
}

int32 DecodableNnetSimple::GetIvectorDim() const {
  if (ivector_ != NULL)
    return ivector_->Dim();
  else if (online_ivector_feats_ != NULL)
    return online_ivector_feats_->NumCols();
  else
    return 0;
}

void DecodableNnetSimple::EnsureFrameIsComputed(int32 subsampled_frame) {
  KALDI_ASSERT(subsampled_frame >= 0 &&
               subsampled_frame < num_subsampled_frames_);
  int32 feature_dim = feats_.NumCols(),
      ivector_dim = GetIvectorDim(),
      nnet_input_dim = nnet_.InputDim("input"),
      nnet_ivector_dim = std::max<int32>(0, nnet_.InputDim("ivector"));
  if (feature_dim != nnet_input_dim)
    KALDI_ERR << "Neural net expects 'input' features with dimension "
              << nnet_input_dim << " but you provided "
              << feature_dim;
  if (ivector_dim != std::max<int32>(0, nnet_.InputDim("ivector")))
    KALDI_ERR << "Neural net expects 'ivector' features with dimension "
              << nnet_ivector_dim << " but you provided " << ivector_dim;

  int32 current_subsampled_frames_computed = current_log_post_.NumRows(),
      current_subsampled_offset = current_log_post_subsampled_offset_;
  KALDI_ASSERT(subsampled_frame < current_subsampled_offset ||
               subsampled_frame >= current_subsampled_offset +
               current_subsampled_frames_computed);

  // all subsampled frames pertain to the output of the network,
  // they are output frames divided by opts_.frame_subsampling_factor.
  int32 subsampling_factor = opts_.frame_subsampling_factor,
      subsampled_frames_per_chunk = opts_.frames_per_chunk / subsampling_factor,
      start_subsampled_frame = subsampled_frame,
      num_subsampled_frames = std::min<int32>(num_subsampled_frames_ -
                                              start_subsampled_frame,
                                              subsampled_frames_per_chunk),
      last_subsampled_frame = start_subsampled_frame + num_subsampled_frames - 1;
  KALDI_ASSERT(num_subsampled_frames > 0);
  // the output-frame numbers are the subsampled-frame numbers
  int32 first_output_frame = start_subsampled_frame * subsampling_factor,
      last_output_frame = last_subsampled_frame * subsampling_factor;

  KALDI_ASSERT(opts_.extra_left_context >= 0 && opts_.extra_right_context >= 0);
  int32 extra_left_context = opts_.extra_left_context,
      extra_right_context = opts_.extra_right_context;
  if (first_output_frame == 0 && opts_.extra_left_context_initial >= 0)
    extra_left_context = opts_.extra_left_context_initial;
  if (last_subsampled_frame == num_subsampled_frames_ - 1 &&
      opts_.extra_right_context_final >= 0)
    extra_right_context = opts_.extra_right_context_final;
  int32 left_context = nnet_left_context_ + extra_left_context,
      right_context = nnet_right_context_ + extra_right_context;
  int32 first_input_frame = first_output_frame - left_context,
      last_input_frame = last_output_frame + right_context,
      num_input_frames = last_input_frame + 1 - first_input_frame;
  Vector<BaseFloat> ivector;
  GetCurrentIvector(first_output_frame,
                    last_output_frame - first_output_frame,
                    &ivector);

  Matrix<BaseFloat> input_feats;
  if (first_input_frame >= 0 &&
      last_input_frame < feats_.NumRows()) {
    SubMatrix<BaseFloat> input_feats(feats_.RowRange(first_input_frame,
                                                     num_input_frames));
    DoNnetComputation(first_input_frame, input_feats, ivector,
                      first_output_frame, num_subsampled_frames);
  } else {
    Matrix<BaseFloat> feats_block(num_input_frames, feats_.NumCols());
    int32 tot_input_feats = feats_.NumRows();
    for (int32 i = 0; i < num_input_frames; i++) {
      SubVector<BaseFloat> dest(feats_block, i);
      int32 t = i + first_input_frame;
      if (t < 0) t = 0;
      if (t >= tot_input_feats) t = tot_input_feats - 1;
      const SubVector<BaseFloat> src(feats_, t);
      dest.CopyFromVec(src);
    }
    DoNnetComputation(first_input_frame, feats_block, ivector,
                      first_output_frame, num_subsampled_frames);
  }
}

// note: in the normal case (with no frame subsampling) you can ignore the
// 'subsampled_' in the variable name.
void DecodableNnetSimple::GetOutputForFrame(int32 subsampled_frame,
                                            VectorBase<BaseFloat> *output) {
  if (subsampled_frame < current_log_post_subsampled_offset_ ||
      subsampled_frame >= current_log_post_subsampled_offset_ +
      current_log_post_.NumRows())
    EnsureFrameIsComputed(subsampled_frame);
  output->CopyFromVec(current_log_post_.Row(
      subsampled_frame - current_log_post_subsampled_offset_));
}

void DecodableNnetSimple::GetCurrentIvector(int32 output_t_start,
                                            int32 num_output_frames,
                                            Vector<BaseFloat> *ivector) {
  if (ivector_ != NULL) {
    *ivector = *ivector_;
    return;
  } else if (online_ivector_feats_ == NULL) {
    return;
  }
  KALDI_ASSERT(online_ivector_period_ > 0);
  // frame_to_search is the frame that we want to get the most recent iVector
  // for.  We choose a point near the middle of the current window, the concept
  // being that this is the fairest comparison to nnet2.   Obviously we could do
  // better by always taking the last frame's iVector, but decoding with
  // 'online' ivectors is only really a mechanism to simulate online operation.
  int32 frame_to_search = output_t_start + num_output_frames / 2;
  int32 ivector_frame = frame_to_search / online_ivector_period_;
  KALDI_ASSERT(ivector_frame >= 0);
  if (ivector_frame >= online_ivector_feats_->NumRows()) {
    int32 margin = ivector_frame - (online_ivector_feats_->NumRows() - 1);
    if (margin * online_ivector_period_ > 50) {
      // Half a second seems like too long to be explainable as edge effects.
      KALDI_ERR << "Could not get iVector for frame " << frame_to_search
                << ", only available till frame "
                << online_ivector_feats_->NumRows()
                << " * ivector-period=" << online_ivector_period_
                << " (mismatched --ivector-period?)";
    }
    ivector_frame = online_ivector_feats_->NumRows() - 1;
  }
  *ivector = online_ivector_feats_->Row(ivector_frame);
}


void DecodableNnetSimple::DoNnetComputation(
    int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    const VectorBase<BaseFloat> &ivector,
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
  if (ivector.Dim() != 0) {
    std::vector<Index> indexes;
    indexes.push_back(Index(0, 0, 0));
    request.inputs.push_back(IoSpecification("ivector", indexes));
  }
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
                        nnet_, nnet_to_update);

  CuMatrix<BaseFloat> input_feats_cu(input_feats);
  computer.AcceptInput("input", &input_feats_cu);
  CuMatrix<BaseFloat> ivector_feats_cu;
  if (ivector.Dim() > 0) {
    ivector_feats_cu.Resize(1, ivector.Dim());
    ivector_feats_cu.Row(0).CopyFromVec(ivector);
    computer.AcceptInput("ivector", &ivector_feats_cu);
  }
  computer.Forward();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  // subtract log-prior (divide by prior)
  if (log_priors_.Dim() != 0)
    cu_output.AddVecToRows(-1.0, log_priors_);
  // apply the acoustic scale
  cu_output.Scale(opts_.acoustic_scale);
  current_log_post_.Resize(0, 0);
  // the following statement just swaps the pointers if we're not using a GPU.
  cu_output.Swap(&current_log_post_);
  current_log_post_subsampled_offset_ = output_t_start / subsample;
}

void DecodableNnetSimple::CheckAndFixConfigs() {
  static bool warned_modulus = false,
      warned_subsampling = false;
  int32 nnet_modulus = nnet_.Modulus();
  if (opts_.frame_subsampling_factor < 1 ||
      opts_.frames_per_chunk < 1)
    KALDI_ERR << "--frame-subsampling-factor and --frames-per-chunk must be > 0";
  if (opts_.frames_per_chunk % opts_.frame_subsampling_factor != 0) {
    int32 f = opts_.frame_subsampling_factor,
        frames_per_chunk = f * ((opts_.frames_per_chunk + f - 1) / f);
    if (!warned_subsampling) {
      warned_subsampling = true;
      KALDI_LOG << "Increasing --frames-per-chunk from "
                << opts_.frames_per_chunk << " to "
                << frames_per_chunk << " to make it a multiple of "
                << "--frame-subsampling-factor="
                << opts_.frame_subsampling_factor;
    }
    opts_.frames_per_chunk = frames_per_chunk;
  }
  if (opts_.frames_per_chunk % nnet_modulus != 0 && !warned_modulus) {
    warned_modulus = true;
    KALDI_WARN << "It may be more efficient to set the --frames-per-chunk "
               << "(currently " << opts_.frames_per_chunk << " to a "
               << "multiple of the network's shift-invariance modulus "
               << nnet_modulus;
  }
}


DecodableAmNnetSimpleParallel::DecodableAmNnetSimpleParallel(
    const NnetSimpleComputationOptions &opts,
    const TransitionModel &trans_model,
    const AmNnetSimple &am_nnet,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    compiler_(am_nnet.GetNnet(), opts.optimize_config),
    trans_model_(trans_model),
    feats_copy_(NULL),
    ivector_copy_(NULL),
    online_ivectors_copy_(NULL),
    decodable_nnet_(NULL) {
  try {
    feats_copy_ = new Matrix<BaseFloat>(feats);
    if (ivector != NULL)
      ivector_copy_ = new Vector<BaseFloat>(*ivector);
    if (online_ivectors != NULL)
      online_ivectors_copy_ = new Matrix<BaseFloat>(*online_ivectors);
    decodable_nnet_ = new DecodableNnetSimple(opts, am_nnet.GetNnet(),
                                              am_nnet.Priors(), *feats_copy_,
                                              &compiler_, ivector_copy_,
                                              online_ivectors_copy_,
                                              online_ivector_period);

  } catch (...) {
    DeletePointers();
    KALDI_ERR << "Error occurred in constructor (see above)";
  }
}

void DecodableAmNnetSimpleParallel::DeletePointers() {
  // delete[] does nothing for null pointers, so we have no checks.
  delete decodable_nnet_;
  decodable_nnet_ = NULL;
  delete feats_copy_;
  feats_copy_ = NULL;
  delete ivector_copy_;
  ivector_copy_ = NULL;
  delete online_ivectors_copy_;
  online_ivectors_copy_ = NULL;
}


BaseFloat DecodableAmNnetSimpleParallel::LogLikelihood(int32 frame,
                                                       int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdf(transition_id);
  return decodable_nnet_->GetOutput(frame, pdf_id);
}


} // namespace nnet3
} // namespace kaldi

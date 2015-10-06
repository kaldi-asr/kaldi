// nnet3/nnet-simple-computer.cc

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

#include "nnet3/nnet-simple-computer.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

 NnetSimpleComputer::NnetSimpleComputer(
  const NnetSimpleComputerOptions &opts,
  const Nnet &nnet,
  const MatrixBase<BaseFloat> &feats,
  int32 left_context, int32 right_context,
  const VectorBase<BaseFloat> *ivector,
  const MatrixBase<BaseFloat> *online_ivectors,
  int32 online_ivector_period): 
 opts_(opts),
 nnet_(nnet),
 feats_(feats),
 ivector_(ivector), online_ivector_feats_(online_ivectors),
 online_ivector_period_(online_ivector_period),
 compiler_(nnet_, opts_.optimize_config),
 current_log_post_offset_(0), 
 left_context_(left_context), right_context_(right_context) {
 KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
 KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period <= 0 &&
    "You need to set the --online-ivector-period option!"));
 PossiblyWarnForFramesPerChunk();
}

NnetSimpleComputer::NnetSimpleComputer(
  const NnetSimpleComputerOptions &opts,
  const Nnet &nnet,
  const MatrixBase<BaseFloat> &feats,
  const VectorBase<BaseFloat> *ivector,
  const MatrixBase<BaseFloat> *online_ivectors,
  int32 online_ivector_period):
 opts_(opts),
 nnet_(nnet),
 feats_(feats),
 ivector_(ivector), online_ivector_feats_(online_ivectors),
 online_ivector_period_(online_ivector_period),
 compiler_(nnet_, opts_.optimize_config),
 current_log_post_offset_(0) {
 KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
 KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period <= 0 &&
    "You need to set the --online-ivector-period option!"));
 PossiblyWarnForFramesPerChunk();
 ComputeSimpleNnetContext(nnet_, &left_context_, &right_context_);
}

NnetSimpleComputer::NnetSimpleComputer(
  const NnetSimpleComputerOptions &opts,
  const Nnet &nnet,
  const MatrixBase<BaseFloat> &feats,
  const MatrixBase<BaseFloat> &ivectors,
  int32 online_ivector_period):
 opts_(opts),
 nnet_(nnet),
 feats_(feats),
 ivector_(NULL),
 online_ivector_feats_(&ivectors),
 online_ivector_period_(online_ivector_period),
 compiler_(nnet, opts_.optimize_config),
 current_log_post_offset_(0) {
 PossiblyWarnForFramesPerChunk();
 ComputeSimpleNnetContext(nnet_, &left_context_, &right_context_);
}  

NnetSimpleComputer::NnetSimpleComputer(
  const NnetSimpleComputerOptions &opts,
  const Nnet &nnet,
  const MatrixBase<BaseFloat> &feats,
  const VectorBase<BaseFloat> &ivector):
 opts_(opts),
 nnet_(nnet),
 feats_(feats),
 ivector_(&ivector),
 online_ivector_feats_(NULL),
 online_ivector_period_(0),
 compiler_(nnet, opts_.optimize_config),
 current_log_post_offset_(0) {
 PossiblyWarnForFramesPerChunk();
 ComputeSimpleNnetContext(nnet_, &left_context_, &right_context_);
}      

int32 NnetSimpleComputer::GetIvectorDim() const {
  if (ivector_ != NULL)
    return ivector_->Dim();
  else if (online_ivector_feats_ != NULL)
    return online_ivector_feats_->NumCols();
  else
    return 0;
}

void NnetSimpleComputer::EnsureFrameIsComputed(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame  < feats_.NumRows());

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

  int32 current_frames_computed = current_log_post_.NumRows(),
      current_offset = current_log_post_offset_;
  KALDI_ASSERT(frame < current_offset ||
               frame >= current_offset + current_frames_computed);
  // allow the output to be computed for frame 0 ... num_input_frames - 1.
  int32 start_output_frame = frame,
      num_output_frames = std::min<int32>(feats_.NumRows() - start_output_frame,
                                          opts_.frames_per_chunk);
  KALDI_ASSERT(num_output_frames > 0);
  KALDI_ASSERT(opts_.extra_left_context >= 0);
  int32 left_context = LeftContext() + opts_.extra_left_context;
  int32 first_input_frame = start_output_frame - left_context,
      num_input_frames = left_context + num_output_frames +
                         RightContext();
  Vector<BaseFloat> ivector;
  GetCurrentIvector(start_output_frame, num_output_frames, &ivector);
  
  Matrix<BaseFloat> input_feats;
  if (first_input_frame >= 0 &&
      first_input_frame + num_input_frames <= feats_.NumRows()) {
    SubMatrix<BaseFloat> input_feats(feats_.RowRange(first_input_frame,
                                                     num_input_frames));
    DoNnetComputation(first_input_frame, input_feats, ivector,
                      start_output_frame, num_output_frames);
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
                      start_output_frame, num_output_frames);
  }  
}

void NnetSimpleComputer::GetCurrentIvector(int32 output_t_start,
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
  

void NnetSimpleComputer::DoNnetComputation(
    int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    const VectorBase<BaseFloat> &ivector,
    int32 output_t_start,
    int32 num_output_frames) {
  CuMatrix<BaseFloat> cu_output;
  DoNnetComputationInternal(input_t_start, input_feats, ivector, 
                            output_t_start, num_output_frames, &cu_output);
  current_log_post_.Resize(0, 0);
  // the following statement just swaps the pointers if we're not using a GPU.
  cu_output.Swap(&current_log_post_);
  current_log_post_offset_ = output_t_start;
}

void NnetSimpleComputer::DoNnetComputationInternal(
    int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    const VectorBase<BaseFloat> &ivector,
    int32 output_t_start,
    int32 num_output_frames,
    CuMatrix<BaseFloat> *cu_output) {
  KALDI_ASSERT(cu_output != NULL);

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
  request.outputs.push_back(
      IoSpecification("output", time_offset + output_t_start,
                      time_offset + output_t_start + num_output_frames));
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
  computer.GetOutputDestructive("output", cu_output);
}

void NnetSimpleComputer::PossiblyWarnForFramesPerChunk() const {
  static bool warned = false;
  int32 nnet_modulus = nnet_.Modulus();  
  if (opts_.frames_per_chunk % nnet_modulus != 0 && !warned) {
    warned = true;
    KALDI_WARN << "It may be more efficient to set the --frames-per-chunk "
               << "(currently " << opts_.frames_per_chunk << " to a "
               << "multiple of the network's shift-invariance modulus "
               << nnet_modulus;
  }
}

void NnetSimpleComputer::GetOutput(Matrix<BaseFloat> *output) {
  for (size_t frame = 0; frame < feats_.NumRows(); 
       frame += opts_.frames_per_chunk) {
   EnsureFrameIsComputed(frame);
   if (frame == 0)
    output->Resize(feats_.NumRows(), current_log_post_.NumCols());
   SubMatrix<BaseFloat> this_output(*output, current_log_post_offset_, 
                                    current_log_post_.NumRows(), 
                                    0, current_log_post_.NumCols());
   this_output.CopyFromMat(current_log_post_);
  }
}

}
}

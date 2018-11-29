// nnet3/decodable-simple-looped.cc

// Copyright      2016  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/decodable-simple-looped.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compile-looped.h"

namespace kaldi {
namespace nnet3 {


DecodableNnetSimpleLoopedInfo::DecodableNnetSimpleLoopedInfo(
    const NnetSimpleLoopedComputationOptions &opts,
    Nnet *nnet):
    opts(opts), nnet(*nnet) {
  Init(opts, nnet);
}

DecodableNnetSimpleLoopedInfo::DecodableNnetSimpleLoopedInfo(
    const NnetSimpleLoopedComputationOptions &opts,
    const Vector<BaseFloat> &priors,
    Nnet *nnet):
    opts(opts), nnet(*nnet), log_priors(priors) {
  if (log_priors.Dim() != 0)
    log_priors.ApplyLog();
  Init(opts, nnet);
}


DecodableNnetSimpleLoopedInfo::DecodableNnetSimpleLoopedInfo(
    const NnetSimpleLoopedComputationOptions &opts,
    AmNnetSimple *am_nnet):
    opts(opts), nnet(am_nnet->GetNnet()), log_priors(am_nnet->Priors()) {
  if (log_priors.Dim() != 0)
    log_priors.ApplyLog();
  Init(opts, &(am_nnet->GetNnet()));
}


void DecodableNnetSimpleLoopedInfo::Init(
    const NnetSimpleLoopedComputationOptions &opts,
    Nnet *nnet) {
  opts.Check();
  KALDI_ASSERT(IsSimpleNnet(*nnet));
  has_ivectors = (nnet->InputDim("ivector") > 0);
  int32 left_context, right_context;
  int32 extra_right_context = 0;
  ComputeSimpleNnetContext(*nnet, &left_context, &right_context);
  frames_left_context = left_context + opts.extra_left_context_initial;
  frames_right_context = right_context + extra_right_context;
  frames_per_chunk = GetChunkSize(*nnet, opts.frame_subsampling_factor,
                                  opts.frames_per_chunk);
  output_dim = nnet->OutputDim("output");
  KALDI_ASSERT(output_dim > 0);
  // note, ivector_period is hardcoded to the same as frames_per_chunk_.
  int32 ivector_period = frames_per_chunk;
  if (has_ivectors)
    ModifyNnetIvectorPeriod(ivector_period, nnet);

  int32 num_sequences = 1;  // we're processing one utterance at a time.

  CreateLoopedComputationRequest(*nnet, frames_per_chunk,
                                 opts.frame_subsampling_factor,
                                 ivector_period,
                                 frames_left_context,
                                 frames_right_context,
                                 num_sequences,
                                 &request1, &request2, &request3);

  CompileLooped(*nnet, opts.optimize_config, request1, request2, request3,
                &computation);
  computation.ComputeCudaIndexes();
  if (GetVerboseLevel() >= 3) {
    KALDI_VLOG(3) << "Computation is:";
    computation.Print(std::cerr, *nnet);
  }
}


DecodableNnetSimpleLooped::DecodableNnetSimpleLooped(
    const DecodableNnetSimpleLoopedInfo &info,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    info_(info),
    computer_(info_.opts.compute_config, info_.computation,
              info_.nnet, NULL),  // NULL is 'nnet_to_update'
    feats_(feats),
    ivector_(ivector), online_ivector_feats_(online_ivectors),
    online_ivector_period_(online_ivector_period),
    num_chunks_computed_(0),
    current_log_post_subsampled_offset_(-1) {
  num_subsampled_frames_ =
      (feats_.NumRows() + info_.opts.frame_subsampling_factor - 1) /
      info_.opts.frame_subsampling_factor;
  KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
  KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period <= 0 &&
                 "You need to set the --online-ivector-period option!"));
}


void DecodableNnetSimpleLooped::GetOutputForFrame(
    int32 subsampled_frame, VectorBase<BaseFloat> *output) {
    KALDI_ASSERT(subsampled_frame >= current_log_post_subsampled_offset_ &&
                 "Frames must be accessed in order.");
    while (subsampled_frame >= current_log_post_subsampled_offset_ +
                            current_log_post_.NumRows())
      AdvanceChunk();
    output->CopyFromVec(current_log_post_.Row(
        subsampled_frame - current_log_post_subsampled_offset_));
}

int32 DecodableNnetSimpleLooped::GetIvectorDim() const {
  if (ivector_ != NULL)
    return ivector_->Dim();
  else if (online_ivector_feats_ != NULL)
    return online_ivector_feats_->NumCols();
  else
    return 0;
}


void DecodableNnetSimpleLooped::AdvanceChunk() {
  int32 begin_input_frame, end_input_frame;
  if (num_chunks_computed_ == 0) {
    begin_input_frame = -info_.frames_left_context;
    // note: end is last plus one.
    end_input_frame = info_.frames_per_chunk + info_.frames_right_context;
  } else {
    begin_input_frame = num_chunks_computed_ * info_.frames_per_chunk +
        info_.frames_right_context;
    end_input_frame = begin_input_frame + info_.frames_per_chunk;
  }
  CuMatrix<BaseFloat> feats_chunk(end_input_frame - begin_input_frame,
                                  feats_.NumCols(), kUndefined);

  int32 num_features = feats_.NumRows();
  if (begin_input_frame >= 0 && end_input_frame <= num_features) {
    SubMatrix<BaseFloat> this_feats(feats_,
                                    begin_input_frame,
                                    end_input_frame - begin_input_frame,
                                    0, feats_.NumCols());
    feats_chunk.CopyFromMat(this_feats);
  } else {
    Matrix<BaseFloat> this_feats(end_input_frame - begin_input_frame,
                                 feats_.NumCols());
    for (int32 r = begin_input_frame; r < end_input_frame; r++) {
      int32 input_frame = r;
      if (input_frame < 0) input_frame = 0;
      if (input_frame >= num_features) input_frame = num_features - 1;
      this_feats.Row(r - begin_input_frame).CopyFromVec(
          feats_.Row(input_frame));
    }
    feats_chunk.CopyFromMat(this_feats);
  }
  computer_.AcceptInput("input", &feats_chunk);

  if (info_.has_ivectors) {
    KALDI_ASSERT(info_.request1.inputs.size() == 2);
    // all but the 1st chunk should have 1 iVector, but no need
    // to assume this.
    int32 num_ivectors = (num_chunks_computed_ == 0 ?
			  info_.request1.inputs[1].indexes.size() :
			  info_.request2.inputs[1].indexes.size());
    KALDI_ASSERT(num_ivectors > 0);

    Vector<BaseFloat> ivector;
    // we just get the iVector from the last input frame we needed...
    // we don't bother trying to be 'accurate' in getting the iVectors
    // for their 'correct' frames, because in general using the
    // iVector from as large 't' as possible will be better.
    GetCurrentIvector(end_input_frame, &ivector);
    Matrix<BaseFloat> ivectors(num_ivectors,
			       ivector.Dim());
    ivectors.CopyRowsFromVec(ivector);
    CuMatrix<BaseFloat> cu_ivectors(ivectors);
    computer_.AcceptInput("ivector", &cu_ivectors);
  }
  computer_.Run();

  {
    // Note: it's possible in theory that if you had weird recurrence that went
    // directly from the output, the call to GetOutputDestructive() would cause
    // a crash on the next chunk.  If that happens, GetOutput() should be used
    // instead of GetOutputDestructive().  But we don't anticipate this will
    // happen in practice.
    CuMatrix<BaseFloat> output;
    computer_.GetOutputDestructive("output", &output);

    if (info_.log_priors.Dim() != 0) {
      // subtract log-prior (divide by prior)
      output.AddVecToRows(-1.0, info_.log_priors);
    }
    // apply the acoustic scale
    output.Scale(info_.opts.acoustic_scale);
    current_log_post_.Resize(0, 0);
    current_log_post_.Swap(&output);
  }
  KALDI_ASSERT(current_log_post_.NumRows() == info_.frames_per_chunk /
               info_.opts.frame_subsampling_factor &&
               current_log_post_.NumCols() == info_.output_dim);

  num_chunks_computed_++;

  current_log_post_subsampled_offset_ =
      (num_chunks_computed_ - 1) *
      (info_.frames_per_chunk / info_.opts.frame_subsampling_factor);
}


void DecodableNnetSimpleLooped::GetCurrentIvector(int32 input_frame,
                                                  Vector<BaseFloat> *ivector) {
  if (!info_.has_ivectors)
    return;
  if (ivector_ != NULL) {
    *ivector = *ivector_;
    return;
  } else if (online_ivector_feats_ == NULL) {
    KALDI_ERR << "Neural net expects iVectors but none provided.";
  }
  KALDI_ASSERT(online_ivector_period_ > 0);
  int32 ivector_frame = input_frame / online_ivector_period_;
  KALDI_ASSERT(ivector_frame >= 0);
  if (ivector_frame >= online_ivector_feats_->NumRows())
    ivector_frame = online_ivector_feats_->NumRows() - 1;
  KALDI_ASSERT(ivector_frame >= 0 && "ivector matrix cannot be empty.");
  *ivector = online_ivector_feats_->Row(ivector_frame);
}


DecodableAmNnetSimpleLooped::DecodableAmNnetSimpleLooped(
    const DecodableNnetSimpleLoopedInfo &info,
    const TransitionModel &trans_model,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    decodable_nnet_(info, feats, ivector, online_ivectors, online_ivector_period),
    trans_model_(trans_model) { }

BaseFloat DecodableAmNnetSimpleLooped::LogLikelihood(int32 frame,
                                                     int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdfFast(transition_id);
  return decodable_nnet_.GetOutput(frame, pdf_id);
}



} // namespace nnet3
} // namespace kaldi

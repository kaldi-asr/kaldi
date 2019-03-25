// nnet3/decodable-online-looped.cc

// Copyright  2017  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/decodable-online-looped.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

DecodableNnetLoopedOnlineBase::DecodableNnetLoopedOnlineBase(
    const DecodableNnetSimpleLoopedInfo &info,
    OnlineFeatureInterface *input_features,
    OnlineFeatureInterface *ivector_features):
    num_chunks_computed_(0),
    current_log_post_subsampled_offset_(-1),
    info_(info),
    frame_offset_(0),
    input_features_(input_features),
    ivector_features_(ivector_features),
    computer_(info_.opts.compute_config, info_.computation,
              info_.nnet, NULL) {   // NULL is 'nnet_to_update'
  // Check that feature dimensions match.
  KALDI_ASSERT(input_features_ != NULL);
  int32 nnet_input_dim = info_.nnet.InputDim("input"),
      nnet_ivector_dim = info_.nnet.InputDim("ivector"),
        feat_input_dim = input_features_->Dim(),
      feat_ivector_dim = (ivector_features_ != NULL ?
                          ivector_features_->Dim() : -1);
  if (nnet_input_dim != feat_input_dim) {
    KALDI_ERR << "Input feature dimension mismatch: got " << feat_input_dim
              << " but network expects " << nnet_input_dim;
  }
  if (nnet_ivector_dim != feat_ivector_dim) {
    KALDI_ERR << "Ivector feature dimension mismatch: got " << feat_ivector_dim
              << " but network expects " << nnet_ivector_dim;
  }
}


int32 DecodableNnetLoopedOnlineBase::NumFramesReady() const {
  // note: the ivector_features_ may have 2 or 3 fewer frames ready than
  // input_features_, but we don't wait for them; we just use the most recent
  // iVector we can.
  int32 features_ready = input_features_->NumFramesReady();
  if (features_ready == 0)
    return 0;
  bool input_finished = input_features_->IsLastFrame(features_ready - 1);

  int32 sf = info_.opts.frame_subsampling_factor;

  if (input_finished) {
    // if the input has finished,... we'll pad with duplicates of the last frame
    // as needed to get the required right context.
    return (features_ready + sf - 1) / sf - frame_offset_;
  } else {
    // note: info_.right_context_ includes both the model context and any
    // extra_right_context_ (but this
    int32 non_subsampled_output_frames_ready =
        std::max<int32>(0, features_ready - info_.frames_right_context);
    int32 num_chunks_ready = non_subsampled_output_frames_ready /
                             info_.frames_per_chunk;
    // note: the division by the frame subsampling factor 'sf' below
    // doesn't need any attention to rounding because info_.frames_per_chunk
    // is always a multiple of 'sf' (see 'frames_per_chunk = GetChunksize..."
    // in decodable-simple-looped.cc).
    return num_chunks_ready * info_.frames_per_chunk / sf - frame_offset_;
  }
}


// note: the frame-index argument is on the output of the network, i.e. after any
// subsampling, so we call it 'subsampled_frame'.
bool DecodableNnetLoopedOnlineBase::IsLastFrame(
    int32 subsampled_frame) const {
  // To understand this code, compare it with the code of NumFramesReady(),
  // it follows the same structure.
  int32 features_ready = input_features_->NumFramesReady();
  if (features_ready == 0) {
    if (subsampled_frame == -1 && input_features_->IsLastFrame(-1)) {
      // the attempt to handle this rather pathological case (input finished
      // but no frames ready) is a little quixotic as we have not properly
      // tested this and other parts of the code may die.
      return true;
    } else {
      return false;
    }
  }
  bool input_finished = input_features_->IsLastFrame(features_ready - 1);
  if (!input_finished)
    return false;
  int32 sf = info_.opts.frame_subsampling_factor,
     num_subsampled_frames_ready = (features_ready + sf - 1) / sf;
  return (subsampled_frame + frame_offset_ == num_subsampled_frames_ready - 1);
}

void DecodableNnetLoopedOnlineBase::SetFrameOffset(int32 frame_offset) {
  KALDI_ASSERT(0 <= frame_offset &&
               frame_offset <= frame_offset_ + NumFramesReady());
  frame_offset_ = frame_offset;
}

void DecodableNnetLoopedOnlineBase::AdvanceChunk() {
  // Prepare the input data for the next chunk of features.
  // note: 'end' means one past the last.
  int32 begin_input_frame, end_input_frame;
  if (num_chunks_computed_ == 0) {
    begin_input_frame = -info_.frames_left_context;
    // note: end is last plus one.
    end_input_frame = info_.frames_per_chunk + info_.frames_right_context;
  } else {
    // note: begin_input_frame will be the same as the previous end_input_frame.
    // you can verify this directly if num_chunks_computed_ == 0, and then by
    // induction.
    begin_input_frame = num_chunks_computed_ * info_.frames_per_chunk +
        info_.frames_right_context;
    end_input_frame = begin_input_frame + info_.frames_per_chunk;
  }

  int32 num_feature_frames_ready = input_features_->NumFramesReady();
  bool is_finished = input_features_->IsLastFrame(num_feature_frames_ready - 1);

  if (end_input_frame > num_feature_frames_ready && !is_finished) {
    // we shouldn't be attempting to read past the end of the available features
    // until we have reached the end of the input (i.e. the end-user called
    // InputFinished(), announcing that there is no more waveform; at this point
    // we pad as needed with copies of the last frame, to flush out the last of
    // the output.
    // If the following error happens, it likely indicates a bug in this
    // decodable code somewhere (although it could possibly indicate the
    // user asking for a frame that was not ready, which would be a misuse
    // of this class.. it can be figured out from gdb as in either case it
    // would be a bug in the code.
    KALDI_ERR << "Attempt to access frame past the end of the available input";
  }


  CuMatrix<BaseFloat> feats_chunk;
  { // this block sets 'feats_chunk'.
    Matrix<BaseFloat> this_feats(end_input_frame - begin_input_frame,
                                 input_features_->Dim());
    for (int32 i = begin_input_frame; i < end_input_frame; i++) {
      SubVector<BaseFloat> this_row(this_feats, i - begin_input_frame);
      int32 input_frame = i;
      if (input_frame < 0) input_frame = 0;
      if (input_frame >= num_feature_frames_ready)
        input_frame = num_feature_frames_ready - 1;
      input_features_->GetFrame(input_frame, &this_row);
    }
    feats_chunk.Swap(&this_feats);
  }
  computer_.AcceptInput("input", &feats_chunk);

  if (info_.has_ivectors) {
    KALDI_ASSERT(ivector_features_ != NULL);
    KALDI_ASSERT(info_.request1.inputs.size() == 2);
    // all but the 1st chunk should have 1 iVector, but there is no need to
    // assume this.
    int32 num_ivectors = (num_chunks_computed_ == 0 ?
			  info_.request1.inputs[1].indexes.size() :
			  info_.request2.inputs[1].indexes.size());
    KALDI_ASSERT(num_ivectors > 0);

    Vector<BaseFloat> ivector(ivector_features_->Dim());
    // we just get the iVector from the last input frame we needed,
    // reduced as necessary
    // we don't bother trying to be 'accurate' in getting the iVectors
    // for their 'correct' frames, because in general using the
    // iVector from as large 't' as possible will be better.

    int32 most_recent_input_frame = num_feature_frames_ready - 1,
      num_ivector_frames_ready = ivector_features_->NumFramesReady();

    if (num_ivector_frames_ready > 0) {
      int32 ivector_frame_to_use = std::min<int32>(
          most_recent_input_frame, num_ivector_frames_ready - 1);
      ivector_features_->GetFrame(ivector_frame_to_use,
                                  &ivector);
    }
    // else just leave the iVector zero (would only happen with very small
    // chunk-size, like a chunk size of 2 which would be very inefficient; and
    // only at file begin.

    // note: we expect num_ivectors to be 1 in practice.
    Matrix<BaseFloat> ivectors(num_ivectors,
			       ivector.Dim());
    ivectors.CopyRowsFromVec(ivector);
    CuMatrix<BaseFloat> cu_ivectors;
    cu_ivectors.Swap(&ivectors);
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

BaseFloat DecodableNnetLoopedOnline::LogLikelihood(int32 subsampled_frame,
                                                    int32 index) {
  subsampled_frame += frame_offset_;
  EnsureFrameIsComputed(subsampled_frame);
  // note: we index by 'inde
  return current_log_post_(
      subsampled_frame - current_log_post_subsampled_offset_,
      index - 1);
}


BaseFloat DecodableAmNnetLoopedOnline::LogLikelihood(int32 subsampled_frame,
                                                    int32 index) {
  subsampled_frame += frame_offset_;
  EnsureFrameIsComputed(subsampled_frame);
  return current_log_post_(
      subsampled_frame - current_log_post_subsampled_offset_,
      trans_model_.TransitionIdToPdfFast(index));
}


} // namespace nnet3
} // namespace kaldi

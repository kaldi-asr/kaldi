// feat/online-feature.cc

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)

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

#include "feat/online-feature.h"

namespace kaldi {


template<class C>
void OnlineMfccOrPlp<C>::GetFeature(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(frame >= 0 && frame < num_frames_);
  KALDI_ASSERT(feat->Dim() == Dim());
  feat->CopyFromVec(features_.Row(frame));
};

template<class C>
bool OnlineMfccOrPlp<C>::IsLastFrame(int32 frame) const {
  return (frame == num_frames_ - 1 && input_finished_);
}

template<class C>
OnlineMfccOrPlp<C>::OnlineMfccOrPlp(const typename C::Options &opts):
    mfcc_or_plp_(opts), input_finished_(false), num_frames_(0),
    sampling_frequency_(opts.frame_opts.samp_freq) { }

template<class C>
void OnlineMfccOrPlp<C>::AcceptWaveform(BaseFloat sampling_rate,
                                        const VectorBase<BaseFloat> &waveform) {
  if (waveform.Dim() == 0) {
    return; // Nothing to do.
  }
  if (input_finished_) {
    KALDI_ERR << "AcceptWaveform called after InputFinished() was called.";
  }
  if (sampling_rate != sampling_frequency_) {
    KALDI_ERR << "Sampling frequency mismatch, expected "
              << sampling_frequency_ << ", got " << sampling_rate;
  }

  Vector<BaseFloat> appended_wave;

  const VectorBase<BaseFloat> &wave_to_use = (waveform_remainder_.Dim() != 0 ?
                                              appended_wave : waveform);
  if (waveform_remainder_.Dim() != 0) {
    appended_wave.Resize(waveform_remainder_.Dim() +
                         waveform.Dim());
    appended_wave.Range(0, waveform_remainder_.Dim()).CopyFromVec(
        waveform_remainder_);
    appended_wave.Range(waveform_remainder_.Dim(),
                        waveform.Dim()).CopyFromVec(waveform);
  }
  waveform_remainder_.Resize(0);
  
  Matrix<BaseFloat> feats;
  BaseFloat vtln_warp = 1.0; // We don't support VTLN warping in this wrapper.
  mfcc_or_plp_.Compute(wave_to_use, vtln_warp, &feats, &waveform_remainder_);

  if (feats.NumRows() == 0) {
    // Presumably we got a very small waveform and could output no whole
    // features.  The waveform will have been appended to waveform_remainder_.
    return;
  }
  int32 new_num_frames = num_frames_ + feats.NumRows();
  BaseFloat increase_ratio = 1.5;  // This is a tradeoff between memory and
                                   // compute; it's the factor by which we
                                   // increase the memory used each time.
  if (new_num_frames > features_.NumRows()) {
    int32 new_num_rows = std::max<int32>(new_num_frames,
                                         features_.NumRows() * increase_ratio);
    // Increase the size of the features_ matrix and copy over any existing
    // data.
    features_.Resize(new_num_rows, Dim(), kCopyData);
  }
  features_.Range(num_frames_, feats.NumRows(), 0, Dim()).CopyFromMat(feats);
  num_frames_ = new_num_frames;
}

// instantiate the templates defined here for MFCC and PLP classes.
template class OnlineMfccOrPlp<Mfcc>;
template class OnlineMfccOrPlp<Plp>;

int32 OnlineSpliceFrames::NumFramesReady() const {
  int32 num_frames = src_->NumFramesReady();
  if (num_frames > 0 && src_->IsLastFrame(num_frames-1))
    return num_frames;
  else
    return std::max<int32>(0, num_frames - right_context_);
}

void OnlineSpliceFrames::GetFeature(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(left_context_ >= 0 && right_context_ >= 0);
  KALDI_ASSERT(frame > 0 && frame < NumFramesReady());
  int32 dim_in = src_->Dim();
  KALDI_ASSERT(feat->Dim() == dim_in * (1 + left_context_ + right_context_));
  int32 T = src_->NumFramesReady();
  for (int32 t2 = frame - left_context_; t2 <= frame + right_context_; t2++) {
    int32 t2_limited = t2;
    if (t2_limited < 0) t2_limited = 0;
    if (t2_limited >= T) t2_limited = T - 1;
    int32 n = t2 - (frame - left_context_); // 0 for left-most frame, increases to
                                            // the right.
    SubVector<BaseFloat> part(*feat, n * dim_in, dim_in);
    src_->GetFeature(t2_limited, &part);
  }  
}

OnlineLda::OnlineLda(const Matrix<BaseFloat> &transform,
                     OnlineFeatureInterface *src): src_(src) {
  int32 src_dim = src_->Dim();
  if (transform.NumCols() == src_dim) { // Linear transform
    linear_term_ = transform;
    offset_.Resize(transform.NumRows()); // Resize() will zero it.
  } else if (transform.NumCols() == src_dim + 1) { // Affine transform
    linear_term_.CopyFromMat(transform.Range(0, transform.NumRows(),
                                             0, src_dim));
    offset_.Resize(transform.NumRows());
    offset_.CopyColFromMat(transform, src_dim);
  } else {
    KALDI_ERR << "Dimension mismatch: source features have dimension "
              << src_dim << " and LDA #cols is " << transform.NumCols();
  }
}

void OnlineLda::GetFeature(int32 frame, VectorBase<BaseFloat> *feat) {
  Vector<BaseFloat> input_feat(linear_term_.NumCols());
  src_->GetFeature(frame, &input_feat);
  feat->CopyFromVec(offset_);
  feat->AddMatVec(1.0, linear_term_, kNoTrans, input_feat, 1.0);
}


int32 OnlineDeltaFeatures::Dim() const {
  int32 src_dim = src_->Dim();
  return src_dim * (1 + opts_.order);
}

int32 OnlineDeltaFeatures::NumFramesReady() const {
  int32 num_frames = src_->NumFramesReady(),
      context = opts_.order * opts_.window;
  // "context" is the number of frames on the left or (more relevant
  // here) right which we need in order to produce the output.
  if (num_frames > 0 && src_->IsLastFrame(num_frames-1))
    return num_frames;
  else
    return std::max<int32>(0, num_frames - context);
}

void OnlineDeltaFeatures::GetFeature(int32 frame,
                                     VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(frame >= 0 && frame < NumFramesReady());
  KALDI_ASSERT(feat->Dim() == Dim());
  // We'll produce a temporary matrix containing the features we want to
  // compute deltas on, but truncated to the necessary context.
  int32 context = opts_.order * opts_.window;
  int32 left_frame = frame - context,
      right_frame = frame + context,
      src_frames_ready = src_->NumFramesReady();
  if (left_frame < 0) left_frame = 0;
  if (right_frame >= src_frames_ready)
    right_frame = src_frames_ready - 1;
  KALDI_ASSERT(right_frame >= left_frame);
  int32 temp_num_frames = right_frame + 1 - left_frame,
      src_dim = src_->Dim();
  Matrix<BaseFloat> temp_src(temp_num_frames, src_dim);
  for (int32 t = left_frame; t <= right_frame; t++) {
    SubVector<BaseFloat> temp_row(temp_src, t - left_frame);
    src_->GetFeature(t, &temp_row);
  }
  int32 temp_t = frame - left_frame; // temp_t is the offset of frame "frame"
                                     // within temp_src
  delta_features_.Process(temp_src, temp_t, feat);
}


OnlineDeltaFeatures::OnlineDeltaFeatures(const DeltaFeaturesOptions &opts,
                                         OnlineFeatureInterface *src):
    src_(src), opts_(opts), delta_features_(opts) { }


}  // namespace kaldi

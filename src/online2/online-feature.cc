// online2/online-feature.cc

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

#include "online2/online-feature.h"
#include "transform/cmvn.h"

namespace kaldi {


template<class C>
void OnlineMfccOrPlp<C>::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
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


void OnlineCmvn::ComputeStatsForFrame(int32 frame,
                                      MatrixBase<double> *stats_out) {
  KALDI_ASSERT(frame >= 0 && frame < src_->NumFramesReady());
  int32 modulus = opts_.modulus;

  int32 index = frame / modulus; // rounds down, so this is the index into
                                 // raw_stats_ of the most recent cached
                                 // set of stats.
  if (index >= raw_stats_.size())
    index = static_cast<int32>(raw_stats_.size()) - 1;
  // most_recent_frame is most recent cached frame, or -1.
  int32 most_recent_frame = (index >= 0 ? index * modulus : -1); 

  int32 dim = this->Dim();
  Matrix<double> stats(2, dim + 1);
  
  if (index >= 0)
    stats.CopyFromMat(raw_stats_[index]);

  Vector<BaseFloat> feats(dim);
  Vector<double> feats_dbl(dim);

  KALDI_ASSERT(most_recent_frame >= -1 && most_recent_frame <= frame);
  
  for (int32 f = most_recent_frame + 1; f <= frame; f++) {
    src_->GetFrame(f, &feats);
    feats_dbl.CopyFromVec(feats);
    stats.Row(0).Range(0, dim).AddVec(1.0, feats_dbl);
    stats.Row(1).Range(0, dim).AddVec2(1.0, feats_dbl);
    stats(0, dim) += 1.0;
    
    int32 prev_f = f - opts_.cmn_window; 
    if (prev_f >= 0) {
      // we need to subtract frame prev_f from the stats.
      src_->GetFrame(prev_f, &feats);
      feats_dbl.CopyFromVec(feats);
      stats.Row(0).Range(0, dim).AddVec(-1.0, feats_dbl);
      stats.Row(1).Range(0, dim).AddVec2(-1.0, feats_dbl);
      stats(0, dim) -= 1.0;
    }
    if (f % modulus == 0) {
      int32 this_index = f / modulus;
      KALDI_ASSERT(this_index == raw_stats_.size());
      raw_stats_.resize(this_index + 1);
      raw_stats_[this_index] = stats;
    }
  }
  stats_out->CopyFromMat(stats);
}


// static
void OnlineCmvn::SmoothOnlineCmvnStats(const MatrixBase<double> &speaker_stats,
                                       const MatrixBase<double> &global_stats,
                                       const OnlineCmvnOptions &opts,
                                       MatrixBase<double> *stats) {
  int32 dim = stats->NumCols() - 1;
  double cur_count = (*stats)(0, dim);
  // If count exceeded cmn_window it would be an error in how "window_stats"
  // was accumulated.
  KALDI_ASSERT(cur_count <= 1.001 * opts.cmn_window);
  if (cur_count >= opts.cmn_window) return;
  if (speaker_stats.NumRows() != 0) { // if we have speaker stats..
    double count_from_speaker = opts.cmn_window - cur_count,
        speaker_count = speaker_stats(0, dim);
    if (count_from_speaker > opts.speaker_frames)
      count_from_speaker = opts.speaker_frames;
    if (count_from_speaker > speaker_count)
      count_from_speaker = speaker_count;
    if (count_from_speaker > 0.0)
      stats->AddMat(count_from_speaker / speaker_count,
                             speaker_stats);
    cur_count = (*stats)(0, dim);
  }
  if (cur_count >= opts.cmn_window) return;  
  if (global_stats.NumRows() != 0) {
    double count_from_global = opts.cmn_window - cur_count,
        global_count = global_stats(0, dim);
    KALDI_ASSERT(global_count > 0.0);
    if (count_from_global > opts.global_frames)
      count_from_global = opts.global_frames;
    if (count_from_global > 0.0)
      stats->AddMat(count_from_global / global_count,
                             global_stats);
  } else {
    KALDI_ERR << "Global CMN stats are required";
  }
}

void OnlineCmvn::GetFrame(int32 frame,
                          VectorBase<BaseFloat> *feat) {
  src_->GetFrame(frame, feat);
  KALDI_ASSERT(feat->Dim() == this->Dim());
  int32 dim = feat->Dim();
  Matrix<double> stats(2, dim + 1);
  if (frozen_state_.NumRows() != 0) { // the CMVN state has been frozen.
    stats.CopyFromMat(frozen_state_);
  } else {
    // first get the raw CMVN stats (this involves caching..)
    this->ComputeStatsForFrame(frame, &stats);
    // now smooth them.
    SmoothOnlineCmvnStats(orig_state_.speaker_cmvn_stats,
                          orig_state_.global_cmvn_stats,
                          opts_,
                          &stats);
  }

  // call the function ApplyCmvn declared in ../transform/cmvn.h, which
  // requires a matrix.
  Matrix<BaseFloat> feat_mat(1, dim);
  feat_mat.Row(0).CopyFromVec(*feat);
  // the function ApplyCmvn takes a matrix, so form a one-row matrix to give it.
  if (opts_.normalize_mean)
    ApplyCmvn(stats, opts_.normalize_variance, &feat_mat);
  else {
    KALDI_ASSERT(!opts_.normalize_variance);
  }
  feat->CopyFromVec(feat_mat.Row(0));
}

void OnlineCmvn::Freeze(int32 cur_frame) {
  int32 dim = this->Dim();
  Matrix<double> stats(2, dim + 1);
  // get the raw CMVN stats
  this->ComputeStatsForFrame(cur_frame, &stats);
  // now smooth them.
  SmoothOnlineCmvnStats(orig_state_.speaker_cmvn_stats,
                        orig_state_.global_cmvn_stats,
                        opts_,
                        &stats);
  this->frozen_state_ = stats;
}

void OnlineCmvn::GetState(int32 cur_frame,
                          OnlineCmvnState *state_out) {
  *state_out = this->orig_state_;
  { // This block updates state_out->speaker_cmvn_stats
    int32 dim = this->Dim();
    if (state_out->speaker_cmvn_stats.NumRows() == 0)
      state_out->speaker_cmvn_stats.Resize(2, dim + 1);
    Vector<BaseFloat> feat(dim);
    Vector<double> feat_dbl(dim);
    for (int32 t = 0; t <= cur_frame; t++) {
      src_->GetFrame(t, &feat);
      feat_dbl.CopyFromVec(feat);
      state_out->speaker_cmvn_stats(0, dim) += 1.0;
      state_out->speaker_cmvn_stats.Row(0).Range(0, dim).AddVec(1.0, feat_dbl);
      state_out->speaker_cmvn_stats.Row(1).Range(0, dim).AddVec2(1.0, feat_dbl);
    }
  }
  // Store any frozen state (the effect of the user possibly
  // having called Freeze().
  state_out->frozen_state = frozen_state_;
}

void OnlineCmvn::SetState(const OnlineCmvnState &cmvn_state) {
  KALDI_ASSERT(raw_stats_.empty() &&
               "You cannot call SetState() after processing data.");
  orig_state_ = cmvn_state;
  frozen_state_ = cmvn_state.frozen_state;
}

int32 OnlineSpliceFrames::NumFramesReady() const {
  int32 num_frames = src_->NumFramesReady();
  if (num_frames > 0 && src_->IsLastFrame(num_frames-1))
    return num_frames;
  else
    return std::max<int32>(0, num_frames - right_context_);
}

void OnlineSpliceFrames::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(left_context_ >= 0 && right_context_ >= 0);
  KALDI_ASSERT(frame >= 0 && frame < NumFramesReady());
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
    src_->GetFrame(t2_limited, &part);
  }  
}

OnlineTransform::OnlineTransform(const MatrixBase<BaseFloat> &transform,
                                 OnlineFeatureInterface *src):
    src_(src) {
  int32 src_dim = src_->Dim();
  if (transform.NumCols() == src_dim) { // Linear transform
    linear_term_ = transform;
    offset_.Resize(transform.NumRows()); // Resize() will zero it.
  } else if (transform.NumCols() == src_dim + 1) { // Affine transform
    linear_term_ = transform.Range(0, transform.NumRows(), 0, src_dim);
    offset_.Resize(transform.NumRows());
    offset_.CopyColFromMat(transform, src_dim);
  } else {
    KALDI_ERR << "Dimension mismatch: source features have dimension "
              << src_dim << " and LDA #cols is " << transform.NumCols();
  }
}

void OnlineTransform::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  Vector<BaseFloat> input_feat(linear_term_.NumCols());
  src_->GetFrame(frame, &input_feat);
  feat->CopyFromVec(offset_);
  feat->AddMatVec(1.0, linear_term_, kNoTrans, input_feat, 1.0);
}


int32 OnlineDeltaFeature::Dim() const {
  int32 src_dim = src_->Dim();
  return src_dim * (1 + opts_.order);
}

int32 OnlineDeltaFeature::NumFramesReady() const {
  int32 num_frames = src_->NumFramesReady(),
      context = opts_.order * opts_.window;
  // "context" is the number of frames on the left or (more relevant
  // here) right which we need in order to produce the output.
  if (num_frames > 0 && src_->IsLastFrame(num_frames-1))
    return num_frames;
  else
    return std::max<int32>(0, num_frames - context);
}

void OnlineDeltaFeature::GetFrame(int32 frame,
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
    src_->GetFrame(t, &temp_row);
  }
  int32 temp_t = frame - left_frame; // temp_t is the offset of frame "frame"
                                     // within temp_src
  delta_features_.Process(temp_src, temp_t, feat);
}


OnlineDeltaFeature::OnlineDeltaFeature(const DeltaFeaturesOptions &opts,
                                       OnlineFeatureInterface *src):
    src_(src), opts_(opts), delta_features_(opts) { }


void OnlineCacheFeature::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(frame >= 0);
  if (static_cast<size_t>(frame) < cache_.size() && cache_[frame] != NULL) {
    feat->CopyFromVec(*(cache_[frame]));
  } else {
    if (static_cast<size_t>(frame) < cache_.size())
      cache_.resize(frame + 1, NULL);
    int32 dim = this->Dim();
    cache_[frame] = new Vector<BaseFloat>(dim);
    // The following call will crash if frame "frame" is not ready.
    src_->GetFrame(frame, cache_[frame]);
    feat->CopyFromVec(*(cache_[frame]));
  }
}

void OnlineCacheFeature::ClearCache() {
  for (size_t i = 0; i < cache_.size(); i++)
    if (cache_[i] != NULL)
      delete cache_[i];
  cache_.resize(0);
}


}  // namespace kaldi

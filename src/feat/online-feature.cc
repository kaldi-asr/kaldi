// feat/online-feature.cc

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)
//              2014  Yanqing Sun, Junjie Wang,
//                    Daniel Povey, Korbinian Riedhammer

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
#include "transform/cmvn.h"

namespace kaldi {

RecyclingVector::RecyclingVector(int items_to_hold):
  items_to_hold_(items_to_hold == 0 ? -1 : items_to_hold),
  first_available_index_(0) {
}

RecyclingVector::~RecyclingVector() {
  for (auto *item : items_) {
    delete item;
  }
}

Vector<BaseFloat> *RecyclingVector::At(int index) const {
  if (index < first_available_index_) {
    KALDI_ERR << "Attempted to retrieve feature vector that was "
                 "already removed by the RecyclingVector (index = "
              << index << "; "
              << "first_available_index = " << first_available_index_ << "; "
              << "size = " << Size() << ")";
  }
  // 'at' does size checking.
  return items_.at(index - first_available_index_);
}

void RecyclingVector::PushBack(Vector<BaseFloat> *item) {
  if (items_.size() == items_to_hold_) {
    delete items_.front();
    items_.pop_front();
    ++first_available_index_;
  }
  items_.push_back(item);
}

int RecyclingVector::Size() const {
  return first_available_index_ + items_.size();
}

template <class C>
void OnlineGenericBaseFeature<C>::GetFrame(int32 frame,
                                           VectorBase<BaseFloat> *feat) {
  feat->CopyFromVec(*(features_.At(frame)));
};

template <class C>
OnlineGenericBaseFeature<C>::OnlineGenericBaseFeature(
    const typename C::Options &opts):
    computer_(opts), window_function_(computer_.GetFrameOptions()),
    features_(opts.frame_opts.max_feature_vectors),
    input_finished_(false), waveform_offset_(0) {
  // RE the following assert: search for ONLINE_IVECTOR_LIMIT in
  // online-ivector-feature.cc.
  // Casting to uint32, an unsigned type, means that -1 would be treated
  // as `very large`.
  KALDI_ASSERT(static_cast<uint32>(opts.frame_opts.max_feature_vectors) > 200);
}


template <class C>
void OnlineGenericBaseFeature<C>::MaybeCreateResampler(
    BaseFloat sampling_rate) {
  BaseFloat expected_sampling_rate = computer_.GetFrameOptions().samp_freq;

  if (resampler_ != nullptr) {
    KALDI_ASSERT(resampler_->GetInputSamplingRate() == sampling_rate);
    KALDI_ASSERT(resampler_->GetOutputSamplingRate() == expected_sampling_rate);
  } else if (((sampling_rate > expected_sampling_rate) &&
              computer_.GetFrameOptions().allow_downsample) ||
             ((sampling_rate < expected_sampling_rate) &&
              computer_.GetFrameOptions().allow_upsample)) {
    resampler_.reset(new LinearResample(
        sampling_rate, expected_sampling_rate,
        std::min(sampling_rate / 2, expected_sampling_rate / 2), 6));
  } else if (sampling_rate != expected_sampling_rate) {
    KALDI_ERR << "Sampling frequency mismatch, expected "
              << expected_sampling_rate << ", got " << sampling_rate
              << "\nPerhaps you want to use the options "
                 "--allow_{upsample,downsample}";
  }
}

template <class C>
void OnlineGenericBaseFeature<C>::InputFinished() {
  if (resampler_ != nullptr) {
    // There may be a few samples left once we flush the resampler_ object, telling it
    // that the file has finished.  This should rarely make any difference.
    Vector<BaseFloat> appended_wave;
    Vector<BaseFloat> resampled_wave;
    resampler_->Resample(appended_wave, true, &resampled_wave);

    if (resampled_wave.Dim() != 0) {
      appended_wave.Resize(waveform_remainder_.Dim() +
                           resampled_wave.Dim());
      if (waveform_remainder_.Dim() != 0)
        appended_wave.Range(0, waveform_remainder_.Dim())
            .CopyFromVec(waveform_remainder_);
      appended_wave.Range(waveform_remainder_.Dim(), resampled_wave.Dim())
          .CopyFromVec(resampled_wave);
      waveform_remainder_.Swap(&appended_wave);
    }
  }
  input_finished_ = true;
  ComputeFeatures();
}

template <class C>
void OnlineGenericBaseFeature<C>::AcceptWaveform(
    BaseFloat sampling_rate, const VectorBase<BaseFloat> &original_waveform) {
  if (original_waveform.Dim() == 0)
    return;  // Nothing to do.
  if (input_finished_)
    KALDI_ERR << "AcceptWaveform called after InputFinished() was called.";

  Vector<BaseFloat> appended_wave;
  Vector<BaseFloat> resampled_wave;

  const VectorBase<BaseFloat> *waveform;

  MaybeCreateResampler(sampling_rate);
  if (resampler_ == nullptr) {
    waveform = &original_waveform;
  } else {
    resampler_->Resample(original_waveform, false, &resampled_wave);
    waveform = &resampled_wave;
  }

  appended_wave.Resize(waveform_remainder_.Dim() + waveform->Dim());
  if (waveform_remainder_.Dim() != 0)
    appended_wave.Range(0, waveform_remainder_.Dim())
        .CopyFromVec(waveform_remainder_);
  appended_wave.Range(waveform_remainder_.Dim(), waveform->Dim())
      .CopyFromVec(*waveform);
  waveform_remainder_.Swap(&appended_wave);
  ComputeFeatures();
}

template <class C>
void OnlineGenericBaseFeature<C>::ComputeFeatures() {
  const FrameExtractionOptions &frame_opts = computer_.GetFrameOptions();
  int64 num_samples_total = waveform_offset_ + waveform_remainder_.Dim();
  int32 num_frames_old = features_.Size(),
      num_frames_new = NumFrames(num_samples_total, frame_opts,
                                 input_finished_);
  KALDI_ASSERT(num_frames_new >= num_frames_old);

  Vector<BaseFloat> window;
  bool need_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 frame = num_frames_old; frame < num_frames_new; frame++) {
    BaseFloat raw_log_energy = 0.0;
    ExtractWindow(waveform_offset_, waveform_remainder_, frame,
                  frame_opts, window_function_, &window,
                  need_raw_log_energy ? &raw_log_energy : NULL);
    Vector<BaseFloat> *this_feature = new Vector<BaseFloat>(computer_.Dim(),
                                                            kUndefined);
    // note: this online feature-extraction code does not support VTLN.
    BaseFloat vtln_warp = 1.0;
    computer_.Compute(raw_log_energy, vtln_warp, &window, this_feature);
    features_.PushBack(this_feature);
  }
  // OK, we will now discard any portion of the signal that will not be
  // necessary to compute frames in the future.
  int64 first_sample_of_next_frame = FirstSampleOfFrame(num_frames_new,
                                                        frame_opts);
  int32 samples_to_discard = first_sample_of_next_frame - waveform_offset_;
  if (samples_to_discard > 0) {
    // discard the leftmost part of the waveform that we no longer need.
    int32 new_num_samples = waveform_remainder_.Dim() - samples_to_discard;
    if (new_num_samples <= 0) {
      // odd, but we'll try to handle it.
      waveform_offset_ += waveform_remainder_.Dim();
      waveform_remainder_.Resize(0);
    } else {
      Vector<BaseFloat> new_remainder(new_num_samples);
      new_remainder.CopyFromVec(waveform_remainder_.Range(samples_to_discard,
                                                          new_num_samples));
      waveform_offset_ += samples_to_discard;
      waveform_remainder_.Swap(&new_remainder);
    }
  }
}

// instantiate the templates defined here for MFCC, PLP and filterbank classes.
template class OnlineGenericBaseFeature<MfccComputer>;
template class OnlineGenericBaseFeature<PlpComputer>;
template class OnlineGenericBaseFeature<FbankComputer>;

OnlineCmvnState::OnlineCmvnState(const OnlineCmvnState &other):
    speaker_cmvn_stats(other.speaker_cmvn_stats),
    global_cmvn_stats(other.global_cmvn_stats),
    frozen_state(other.frozen_state) { }

void OnlineCmvnState::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<OnlineCmvnState>");  // magic string.
  WriteToken(os, binary, "<SpeakerCmvnStats>");
  speaker_cmvn_stats.Write(os, binary);
  WriteToken(os, binary, "<GlobalCmvnStats>");
  global_cmvn_stats.Write(os, binary);
  WriteToken(os, binary, "<FrozenState>");
  frozen_state.Write(os, binary);
  WriteToken(os, binary, "</OnlineCmvnState>");
}

void OnlineCmvnState::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<OnlineCmvnState>");  // magic string.
  ExpectToken(is, binary, "<SpeakerCmvnStats>");
  speaker_cmvn_stats.Read(is, binary);
  ExpectToken(is, binary, "<GlobalCmvnStats>");
  global_cmvn_stats.Read(is, binary);
  ExpectToken(is, binary, "<FrozenState>");
  frozen_state.Read(is, binary);
  ExpectToken(is, binary, "</OnlineCmvnState>");
}

OnlineCmvn::OnlineCmvn(const OnlineCmvnOptions &opts,
                       const OnlineCmvnState &cmvn_state,
                       OnlineFeatureInterface *src):
    opts_(opts), temp_stats_(2, src->Dim() + 1),
    temp_feats_(src->Dim()), temp_feats_dbl_(src->Dim()),
    src_(src) {
  SetState(cmvn_state);
  if (!SplitStringToIntegers(opts.skip_dims, ":", false, &skip_dims_))
    KALDI_ERR << "Bad --skip-dims option (should be colon-separated list of "
              <<  "integers)";
}

OnlineCmvn::OnlineCmvn(const OnlineCmvnOptions &opts,
                       OnlineFeatureInterface *src):
    opts_(opts), temp_stats_(2, src->Dim() + 1),
    temp_feats_(src->Dim()), temp_feats_dbl_(src->Dim()),
    src_(src) {
  if (!SplitStringToIntegers(opts.skip_dims, ":", false, &skip_dims_))
    KALDI_ERR << "Bad --skip-dims option (should be colon-separated list of "
              <<  "integers)";
}


void OnlineCmvn::GetMostRecentCachedFrame(int32 frame,
                                          int32 *cached_frame,
                                          MatrixBase<double> *stats) {
  KALDI_ASSERT(frame >= 0);
  InitRingBufferIfNeeded();
  // look for a cached frame on a previous frame as close as possible in time
  // to "frame".  Return if we get one.
  for (int32 t = frame; t >= 0 && t >= frame - opts_.ring_buffer_size; t--) {
    if (t % opts_.modulus == 0) {
      // if this frame should be cached in cached_stats_modulo_, then
      // we'll look there, and we won't go back any further in time.
      break;
    }
    int32 index = t % opts_.ring_buffer_size;
    if (cached_stats_ring_[index].first == t) {
      *cached_frame = t;
      stats->CopyFromMat(cached_stats_ring_[index].second);
      return;
    }
  }
  int32 n = frame / opts_.modulus;
  if (n >= cached_stats_modulo_.size()) {
    if (cached_stats_modulo_.size() == 0) {
      *cached_frame = -1;
      stats->SetZero();
      return;
    } else {
      n = static_cast<int32>(cached_stats_modulo_.size() - 1);
    }
  }
  *cached_frame = n * opts_.modulus;
  KALDI_ASSERT(cached_stats_modulo_[n] != NULL);
  stats->CopyFromMat(*(cached_stats_modulo_[n]));
}

// Initialize ring buffer for caching stats.
void OnlineCmvn::InitRingBufferIfNeeded() {
  if (cached_stats_ring_.empty() && opts_.ring_buffer_size > 0) {
    Matrix<double> temp(2, this->Dim() + 1);
    cached_stats_ring_.resize(opts_.ring_buffer_size,
                              std::pair<int32, Matrix<double> >(-1, temp));
  }
}

void OnlineCmvn::CacheFrame(int32 frame, const MatrixBase<double> &stats) {
  KALDI_ASSERT(frame >= 0);
  if (frame % opts_.modulus == 0) {  // store in cached_stats_modulo_.
    int32 n = frame / opts_.modulus;
    if (n >= cached_stats_modulo_.size()) {
      // The following assert is a limitation on in what order you can call
      // CacheFrame.  Fortunately the calling code always calls it in sequence,
      // which it has to because you need a previous frame to compute the
      // current one.
      KALDI_ASSERT(n == cached_stats_modulo_.size());
      cached_stats_modulo_.push_back(new Matrix<double>(stats));
    } else {
      KALDI_WARN << "Did not expect to reach this part of code.";
      // do what seems right, but we shouldn't get here.
      cached_stats_modulo_[n]->CopyFromMat(stats);
    }
  } else {  // store in the ring buffer.
    InitRingBufferIfNeeded();
    if (!cached_stats_ring_.empty()) {
      int32 index = frame % cached_stats_ring_.size();
      cached_stats_ring_[index].first = frame;
      cached_stats_ring_[index].second.CopyFromMat(stats);
    }
  }
}

OnlineCmvn::~OnlineCmvn() {
  for (size_t i = 0; i < cached_stats_modulo_.size(); i++)
    delete cached_stats_modulo_[i];
  cached_stats_modulo_.clear();
}

void OnlineCmvn::ComputeStatsForFrame(int32 frame,
                                      MatrixBase<double> *stats_out) {
  KALDI_ASSERT(frame >= 0 && frame < src_->NumFramesReady());

  int32 dim = this->Dim(), cur_frame;
  GetMostRecentCachedFrame(frame, &cur_frame, stats_out);

  Vector<BaseFloat> &feats(temp_feats_);
  Vector<double> &feats_dbl(temp_feats_dbl_);
  while (cur_frame < frame) {
    cur_frame++;
    src_->GetFrame(cur_frame, &feats);
    feats_dbl.CopyFromVec(feats);
    stats_out->Row(0).Range(0, dim).AddVec(1.0, feats_dbl);
    if (opts_.normalize_variance)
      stats_out->Row(1).Range(0, dim).AddVec2(1.0, feats_dbl);
    (*stats_out)(0, dim) += 1.0;
    // it's a sliding buffer; a frame at the back may be
    // leaving the buffer so we have to subtract that.
    int32 prev_frame = cur_frame - opts_.cmn_window;
    if (prev_frame >= 0) {
      // we need to subtract frame prev_f from the stats.
      src_->GetFrame(prev_frame, &feats);
      feats_dbl.CopyFromVec(feats);
      stats_out->Row(0).Range(0, dim).AddVec(-1.0, feats_dbl);
      if (opts_.normalize_variance)
        stats_out->Row(1).Range(0, dim).AddVec2(-1.0, feats_dbl);
      (*stats_out)(0, dim) -= 1.0;
    }
    CacheFrame(cur_frame, (*stats_out));
  }
}


// static
void OnlineCmvn::SmoothOnlineCmvnStats(const MatrixBase<double> &speaker_stats,
                                       const MatrixBase<double> &global_stats,
                                       const OnlineCmvnOptions &opts,
                                       MatrixBase<double> *stats) {
  if (speaker_stats.NumRows() == 2 && !opts.normalize_variance) {
    // this is just for efficiency: don't operate on the variance if it's not
    // needed.
    int32 cols = speaker_stats.NumCols();  // dim + 1
    SubMatrix<double> stats_temp(*stats, 0, 1, 0, cols);
    SmoothOnlineCmvnStats(speaker_stats.RowRange(0, 1),
                          global_stats.RowRange(0, 1),
                          opts, &stats_temp);
    return;
  }
  int32 dim = stats->NumCols() - 1;
  double cur_count = (*stats)(0, dim);
  // If count exceeded cmn_window it would be an error in how "window_stats"
  // was accumulated.
  KALDI_ASSERT(cur_count <= 1.001 * opts.cmn_window);
  if (cur_count >= opts.cmn_window)
    return;
  if (speaker_stats.NumRows() != 0) {  // if we have speaker stats..
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
  if (cur_count >= opts.cmn_window)
    return;
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
  Matrix<double> &stats(temp_stats_);
  stats.Resize(2, dim + 1, kUndefined);  // Will do nothing if size was correct.
  if (frozen_state_.NumRows() != 0) {  // the CMVN state has been frozen.
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

  if (!skip_dims_.empty())
    FakeStatsForSomeDims(skip_dims_, &stats);

  // call the function ApplyCmvn declared in ../transform/cmvn.h, which
  // requires a matrix.
  // 1 row; num-cols == dim; stride  == dim.
  SubMatrix<BaseFloat> feat_mat(feat->Data(), 1, dim, dim);
  // the function ApplyCmvn takes a matrix, so form a one-row matrix to give it.
  if (opts_.normalize_mean)
    ApplyCmvn(stats, opts_.normalize_variance, &feat_mat);
  else
    KALDI_ASSERT(!opts_.normalize_variance);
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
  KALDI_ASSERT(cached_stats_modulo_.empty() &&
               "You cannot call SetState() after processing data.");
  orig_state_ = cmvn_state;
  frozen_state_ = cmvn_state.frozen_state;
}

int32 OnlineSpliceFrames::NumFramesReady() const {
  int32 num_frames = src_->NumFramesReady();
  if (num_frames > 0 && src_->IsLastFrame(num_frames - 1))
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
    int32 n = t2 - (frame - left_context_);  // 0 for left-most frame,
                                             // increases to the right.
    SubVector<BaseFloat> part(*feat, n * dim_in, dim_in);
    src_->GetFrame(t2_limited, &part);
  }
}

OnlineTransform::OnlineTransform(const MatrixBase<BaseFloat> &transform,
                                 OnlineFeatureInterface *src):
    src_(src) {
  int32 src_dim = src_->Dim();
  if (transform.NumCols() == src_dim) {  // Linear transform
    linear_term_ = transform;
    offset_.Resize(transform.NumRows());  // Resize() will zero it.
  } else if (transform.NumCols() == src_dim + 1) {  // Affine transform
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

void OnlineTransform::GetFrames(
    const std::vector<int32> &frames, MatrixBase<BaseFloat> *feats) {
  KALDI_ASSERT(static_cast<int32>(frames.size()) == feats->NumRows());
  int32 num_frames = feats->NumRows(),
      input_dim = linear_term_.NumCols();
  Matrix<BaseFloat> input_feats(num_frames, input_dim, kUndefined);
  src_->GetFrames(frames, &input_feats);
  feats->CopyRowsFromVec(offset_);
  feats->AddMatMat(1.0, input_feats, kNoTrans, linear_term_, kTrans, 1.0);
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
  int32 temp_t = frame - left_frame;  // temp_t is the offset of frame "frame"
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
    if (static_cast<size_t>(frame) >= cache_.size())
      cache_.resize(frame + 1, NULL);
    int32 dim = this->Dim();
    cache_[frame] = new Vector<BaseFloat>(dim);
    // The following call will crash if frame "frame" is not ready.
    src_->GetFrame(frame, cache_[frame]);
    feat->CopyFromVec(*(cache_[frame]));
  }
}

void OnlineCacheFeature::GetFrames(
    const std::vector<int32> &frames, MatrixBase<BaseFloat> *feats) {
  int32 num_frames = frames.size();
  // non_cached_frames will be the subset of 't' values in 'frames' which were
  // not previously cached, which we therefore need to get from src_.
  std::vector<int32> non_cached_frames;
  // 'non_cached_indexes' stores the indexes 'i' into 'frames' corresponding to
  // the corresponding frames in 'non_cached_frames'.
  std::vector<int32> non_cached_indexes;
  non_cached_frames.reserve(frames.size());
  non_cached_indexes.reserve(frames.size());
  for (int32 i = 0; i < num_frames; i++) {
    int32 t = frames[i];
    if (static_cast<size_t>(t) < cache_.size() && cache_[t] != NULL) {
      feats->Row(i).CopyFromVec(*(cache_[t]));
    } else {
      non_cached_frames.push_back(t);
      non_cached_indexes.push_back(i);
    }
  }
  if (non_cached_frames.empty())
    return;
  int32 num_non_cached_frames = non_cached_frames.size(),
      dim = this->Dim();
  Matrix<BaseFloat> non_cached_feats(num_non_cached_frames, dim,
                                     kUndefined);
  src_->GetFrames(non_cached_frames, &non_cached_feats);
  for (int32 i = 0; i < num_non_cached_frames; i++) {
    int32 t = non_cached_frames[i];
    if (static_cast<size_t>(t) < cache_.size() && cache_[t] != NULL) {
      // We can reach this point due to repeat indexes in 'non_cached_frames'.
      feats->Row(non_cached_indexes[i]).CopyFromVec(*(cache_[t]));
    } else {
      SubVector<BaseFloat> this_feat(non_cached_feats, i);
      feats->Row(non_cached_indexes[i]).CopyFromVec(this_feat);
      if (static_cast<size_t>(t) >= cache_.size())
        cache_.resize(t + 1, NULL);
      cache_[t] = new Vector<BaseFloat>(this_feat);
    }
  }
}


void OnlineCacheFeature::ClearCache() {
  for (size_t i = 0; i < cache_.size(); i++)
    delete cache_[i];
  cache_.resize(0);
}


void OnlineAppendFeature::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(feat->Dim() == Dim());

  SubVector<BaseFloat> feat1(*feat, 0, src1_->Dim());
  SubVector<BaseFloat> feat2(*feat, src1_->Dim(), src2_->Dim());
  src1_->GetFrame(frame, &feat1);
  src2_->GetFrame(frame, &feat2);
};


}  // namespace kaldi

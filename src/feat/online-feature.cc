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
#include "transform/cmvn.h"

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


// TODO rename and move to OnlineCmvn class? 
/// The statistics were collected also using the vector feat.
/// The function UndoStats recomputes the stats,
/// so they corresponds to statistics computed without feat vector.
void UndoStats(const VectorBase<BaseFloat> & feat, 
                Matrix<BaseFloat> *stats);

// TODO heavily based on src/transform/cmvn.h:ApplyCmvn.
void ApplyCmvn(const MatrixBase<double> &stats, 
                bool var_norm,
                VectorBase<BaseFloat> *feat) ;

void OnlineCmvn::GetFeature(int32 frame, VectorBase<BaseFloat> *feat) {
  src_->GetFeature(frame, feat);
  if (frame < stats_.size()) {
    // Apply the normalization from cached statistics.
    ApplyCmvn(stats_[frame], norm_var_, feat);
  } else {
    VectorBase<BaseFloat> fresh;
    fresh.CopyFromVec(*feat);
    window_.push(fresh);  // window_ is FIFO
    if(WindowSize() > cmvn_window_) {
      // Before removing the frame update the sliding_stat_
      UndoStats(window_.first(), &sliding_stat_);
      window_.pop();  // Removes the old frame
    } else if (WindowSize() < min_window_) {
      KALDI_ERR << "Supply more frames for input!" << std::endl
                << "Can not compute CM[V]N on " << WindowSize() << 
                "frames with minimal window size " << min_window_ << 
                "!" << std::endl;
    }
    // Add the statistics for the new frame
    // Storing count, sum and sum_square is the best way how to compute
    // on the fly variance since 
    // Var(a[n+1]) = Sum(a[0:n+1]^2) - Mean(a[0:n+1])^2 =
    //             = Sum(a[0:n]^2) + (a[n]^2) - Mean(a[0:n+1])^2
    AccCmvnStats(fresh, 1.0, &sliding_stat_);
    // Store them to the cache
    stats_.push_back(sliding_stat_);
    KALDI_ASSERT(stats_.size() == frame);
    // Apply the freshly updated cmvn stats
    ApplyCmvn(sliding_stat_, norm_var_, feat);
  }
}

void OnlineCmvn::ApplyStats(const Matrix<BaseFloat> &stats) {
  KALDI_ASSERT(stats.NumCols() == sliding_stat_.NumCols());
  KALDI_ASSERT(stats.NumRows() == sliding_stat_.NumRows());
  sliding_stat_.CopyFromMat(stats);
   
  // Create artificial frames which corresponds to stored statistics
  window_.clear();
  int32 win_size = WindowSize();
  VectorBase<BaseFloat > artificial1;
  VectorBase<BaseFloat> artificial2;
  // If variance>0 then we need atleast 2 different vectors
  // TODO fill it with non-zero variance vector
  // TODO scale to fit the variance. 
  // TODO shift it in other direction than mean
  // for(int32 i = 0; i < win_size; ++i) {
  //   window_.push(artificial);
  // }
}

void UndoStats(const VectorBase<BaseFloat> &feat, Matrix<BaseFloat> *stats) {
  // TODO 
}

void ApplyCmvn(const MatrixBase<double> &stats,
               bool var_norm,
               VectorBase<BaseFloat> *feats) {
  KALDI_ASSERT(feats != NULL);
  int32 dim = stats.NumCols() - 1;
  if (stats.NumRows() > 2 || stats.NumRows() < 1 || feats->Dim() != dim) {
    KALDI_ERR << "Dim mismatch in ApplyCmvn: cmvn "
              << stats.NumRows() << 'x' << stats.NumCols()
              << ", feats " << "1x" << feats->Dim();
  }
  if (stats.NumRows() == 1 && var_norm)
    KALDI_ERR << "You requested variance normalization but no variance stats "
              << "are supplied.";

  double count = stats(0, dim);
  // Do not change the threshold of 1.0 here: in the balanced-cmvn code, when
  // computing an offset and representing it as stats, we use a count of one.
  if (count < 1.0)
    KALDI_ERR << "Insufficient stats for cepstral mean and variance normalization: "
              << "count = " << count;

  Matrix<BaseFloat> norm(2, dim);  // norm(0, d) = mean offset
  // norm(1, d) = scale, e.g. x(d) <-- x(d)*norm(1, d) + norm(0, d).
  for (int32 d = 0; d < dim; d++) {
    double mean, offset, scale;
    mean = stats(0, d)/count;
    if (!var_norm) {
      scale = 1.0;
      offset = -mean;
    } else {
      double var = (stats(1, d)/count) - mean*mean,
          floor = 1.0e-20;
      if (var < floor) {
        KALDI_WARN << "Flooring cepstral variance from " << var << " to "
                   << floor;
        var = floor;
      }
      scale = 1.0 / sqrt(var);
      if (scale != scale || 1/scale == 0.0)
        KALDI_ERR << "NaN or infinity in cepstral mean/variance computation\n";
      offset = -(mean*scale);
    }
    norm(0, d) = offset;
    norm(1, d) = scale;
  }

  // Apply the normalization.
  for (int32 d = 0; d < dim; d++) {
    BaseFloat &f = (*feats)(d);
    f = norm(0, d) + f*norm(1, d);
  }
}

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
<<<<<<< HEAD
                     OnlineFeatureInterface *src):
    src_(src) {
=======
                     OnlineFeatureInterface *src): 
    src_(src), is_online_(true) {
>>>>>>> sb-online: Delete is_online from constructor functions
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
<<<<<<< HEAD
                                         OnlineFeatureInterface *src):
    src_(src), opts_(opts), delta_features_(opts) { }
=======
                                OnlineFeatureInterface *src):
    src_(src), opts_(opts), delta_features_(opts), is_online_(true) { }
>>>>>>> sb-online: Delete is_online from constructor functions


}  // namespace kaldi

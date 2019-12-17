// feat/online-feature.h

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)
//           2014   Yanqing Sun, Junjie Wang,
//                  Daniel Povey, Korbinian Riedhammer

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


#ifndef KALDI_FEAT_ONLINE_FEATURE_H_
#define KALDI_FEAT_ONLINE_FEATURE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/feature-functions.h"
#include "feat/feature-mfcc.h"
#include "feat/feature-plp.h"
#include "feat/feature-fbank.h"
#include "itf/online-feature-itf.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{


/// This class serves as a storage for feature vectors with an option to limit
/// the memory usage by removing old elements. The deleted frames indices are
/// "remembered" so that regardless of the MAX_ITEMS setting, the user always
/// provides the indices as if no deletion was being performed.
/// This is useful when processing very long recordings which would otherwise
/// cause the memory to eventually blow up when the features are not being removed.
class RecyclingVector {
public:
  /// By default it does not remove any elements.
  RecyclingVector(int items_to_hold = -1);

  /// The ownership is being retained by this collection - do not delete the item.
  Vector<BaseFloat> *At(int index) const;

  /// The ownership of the item is passed to this collection - do not delete the item.
  void PushBack(Vector<BaseFloat> *item);

  /// This method returns the size as if no "recycling" had happened,
  /// i.e. equivalent to the number of times the PushBack method has been called.
  int Size() const;

  ~RecyclingVector();

private:
  std::deque<Vector<BaseFloat>*> items_;
  int items_to_hold_;
  int first_available_index_;
};


/// This is a templated class for online feature extraction;
/// it's templated on a class like MfccComputer or PlpComputer
/// that does the basic feature extraction.
template<class C>
class OnlineGenericBaseFeature: public OnlineBaseFeature {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return computer_.Dim(); }

  // Note: IsLastFrame() will only ever return true if you have called
  // InputFinished() (and this frame is the last frame).
  virtual bool IsLastFrame(int32 frame) const {
    return input_finished_ && frame == NumFramesReady() - 1;
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return computer_.GetFrameOptions().frame_shift_ms / 1000.0f;
  }

  virtual int32 NumFramesReady() const { return features_.Size(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  // Next, functions that are not in the interface.


  // Constructor from options class
  explicit OnlineGenericBaseFeature(const typename C::Options &opts);

  // This would be called from the application, when you get
  // more wave data.  Note: the sampling_rate is only provided so
  // the code can assert that it matches the sampling rate
  // expected in the options.
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);


  // InputFinished() tells the class you won't be providing any
  // more waveform.  This will help flush out the last frame or two
  // of features, in the case where snip-edges == false; it also
  // affects the return value of IsLastFrame().
  virtual void InputFinished();

 private:
  // This function computes any additional feature frames that it is possible to
  // compute from 'waveform_remainder_', which at this point may contain more
  // than just a remainder-sized quantity (because AcceptWaveform() appends to
  // waveform_remainder_ before calling this function).  It adds these feature
  // frames to features_, and shifts off any now-unneeded samples of input from
  // waveform_remainder_ while incrementing waveform_offset_ by the same amount.
  void ComputeFeatures();

  void MaybeCreateResampler(BaseFloat sampling_rate);

  C computer_;  // class that does the MFCC or PLP or filterbank computation

  // resampler in cases when the input sampling frequency is not equal to
  // the expected sampling rate
  std::unique_ptr<LinearResample> resampler_;

  FeatureWindowFunction window_function_;

  // features_ is the Mfcc or Plp or Fbank features that we have already computed.

  RecyclingVector features_;

  // True if the user has called "InputFinished()"
  bool input_finished_;

  // The sampling frequency, extracted from the config.  Should
  // be identical to the waveform supplied.
  BaseFloat sampling_frequency_;

  // waveform_offset_ is the number of samples of waveform that we have
  // already discarded, i.e. that were prior to 'waveform_remainder_'.
  int64 waveform_offset_;

  // waveform_remainder_ is a short piece of waveform that we may need to keep
  // after extracting all the whole frames we can (whatever length of feature
  // will be required for the next phase of computation).
  Vector<BaseFloat> waveform_remainder_;
};

typedef OnlineGenericBaseFeature<MfccComputer> OnlineMfcc;
typedef OnlineGenericBaseFeature<PlpComputer> OnlinePlp;
typedef OnlineGenericBaseFeature<FbankComputer> OnlineFbank;


/// This class takes a Matrix<BaseFloat> and wraps it as an
/// OnlineFeatureInterface: this can be useful where some earlier stage of
/// feature processing has been done offline but you want to use part of the
/// online pipeline.
class OnlineMatrixFeature: public OnlineFeatureInterface {
 public:
  /// Caution: this class maintains the const reference from the constructor, so
  /// don't let it go out of scope while this object exists.
  explicit OnlineMatrixFeature(const MatrixBase<BaseFloat> &mat): mat_(mat) { }

  virtual int32 Dim() const { return mat_.NumCols(); }

  virtual BaseFloat FrameShiftInSeconds() const {
    return 0.01f;
  }

  virtual int32 NumFramesReady() const { return mat_.NumRows(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
    feat->CopyFromVec(mat_.Row(frame));
  }

  virtual bool IsLastFrame(int32 frame) const {
    return (frame + 1 == mat_.NumRows());
  }


 private:
  const MatrixBase<BaseFloat> &mat_;
};


// Note the similarity with SlidingWindowCmnOptions, but there
// are also differences.  One which doesn't appear in the config
// itself, because it's a difference between the setups, is that
// in OnlineCmn, we carry over data from the previous utterance,
// or, if no previous utterance is available, from global stats,
// or, if previous utterances are available but the total amount
// of data is less than prev_frames, we pad with up to "global_frames"
// frames from the global stats.
struct OnlineCmvnOptions {
  int32 cmn_window;
  int32 speaker_frames;  // must be <= cmn_window
  int32 global_frames;  // must be <= speaker_frames.
  bool normalize_mean;  // Must be true if normalize_variance==true.
  bool normalize_variance;

  int32 modulus;  // not configurable from command line, relates to how the
                  // class computes the cmvn internally.  smaller->more
                  // time-efficient but less memory-efficient.  Must be >= 1.
  int32 ring_buffer_size;  // not configurable from command line; size of ring
                           // buffer used for caching CMVN stats.  Must be >=
                           // modulus.
  std::string skip_dims; // Colon-separated list of dimensions to skip normalization
                         // of, e.g. 13:14:15.

  OnlineCmvnOptions():
      cmn_window(600),
      speaker_frames(600),
      global_frames(200),
      normalize_mean(true),
      normalize_variance(false),
      modulus(20),
      ring_buffer_size(20),
      skip_dims("") { }

  void Check() const {
    KALDI_ASSERT(speaker_frames <= cmn_window && global_frames <= speaker_frames
                 && modulus > 0);
  }

  void Register(ParseOptions *po) {
    po->Register("cmn-window", &cmn_window, "Number of frames of sliding "
                 "context for cepstral mean normalization.");
    po->Register("global-frames", &global_frames, "Number of frames of "
                 "global-average cepstral mean normalization stats to use for "
                 "first utterance of a speaker");
    po->Register("speaker-frames", &speaker_frames, "Number of frames of "
                 "previous utterance(s) from this speaker to use in cepstral "
                 "mean normalization");
    // we name the config string "norm-vars" for compatibility with
    // ../featbin/apply-cmvn.cc
    po->Register("norm-vars", &normalize_variance, "If true, do "
                 "cepstral variance normalization in addition to cepstral mean "
                 "normalization ");
    po->Register("norm-means", &normalize_mean, "If true, do mean normalization "
                 "(note: you cannot normalize the variance but not the mean)");
    po->Register("skip-dims", &skip_dims, "Dimensions to skip normalization of "
                 "(colon-separated list of integers)");}
};



/** Struct OnlineCmvnState stores the state of CMVN adaptation between
    utterances (but not the state of the computation within an utterance).  It
    stores the global CMVN stats and the stats of the current speaker (if we
    have seen previous utterances for this speaker), and possibly will have a
    member "frozen_state": if the user has called the function Freeze() of class
    OnlineCmvn, to fix the CMVN so we can estimate fMLLR on top of the fixed
    value of cmvn.  If nonempty, "frozen_state" will reflect how we were
    normalizing the mean and (if applicable) variance at the time when that
    function was called.
*/
struct OnlineCmvnState {
  // The following is the total CMVN stats for this speaker (up till now), in
  // the same format.
  Matrix<double> speaker_cmvn_stats;

  // The following is the global CMVN stats, in the usual
  // format, of dimension 2 x (dim+1), as [  sum-stats          count
  //                                       sum-squared-stats   0    ]
  Matrix<double> global_cmvn_stats;

  // If nonempty, contains CMVN stats representing the "frozen" state
  // of CMVN that reflects how we were normalizing the data when the
  // user called the Freeze() function in class OnlineCmvn.
  Matrix<double> frozen_state;

  OnlineCmvnState() { }

  explicit OnlineCmvnState(const Matrix<double> &global_stats):
      global_cmvn_stats(global_stats) { }

  // Copy constructor
  OnlineCmvnState(const OnlineCmvnState &other);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Use the default assignment operator.
};

/**
   This class does an online version of the cepstral mean and [optionally]
   variance, but note that this is not equivalent to the offline version.  This
   is necessarily so, as the offline computation involves looking into the
   future.  If you plan to use features normalized with this type of CMVN then
   you need to train in a `matched' way, i.e. with the same type of features.
   We normally only do so in the "online" GMM-based decoding, e.g.  in
   online2bin/online2-wav-gmm-latgen-faster.cc; see also the script
   steps/online/prepare_online_decoding.sh and steps/online/decode.sh.

   In the steady state (in the middle of a long utterance), this class
   accumulates CMVN statistics from the previous "cmn_window" frames (default 600
   frames, or 6 seconds), and uses these to normalize the mean and possibly
   variance of the current frame.

   The config variables "speaker_frames" and "global_frames" relate to what
   happens at the beginning of the utterance when we have seen fewer than
   "cmn_window" frames of context, and so might not have very good stats to
   normalize with.  Basically, we first augment any existing stats with up
   to "speaker_frames" frames of stats from previous utterances of the current
   speaker, and if this doesn't take us up to the required "cmn_window" frame
   count, we further augment with up to "global_frames" frames of global
   stats.  The global stats are CMVN stats accumulated from training or testing
   data, that give us a reasonable source of mean and variance for "typical"
   data.
 */
class OnlineCmvn: public OnlineFeatureInterface {
 public:

  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return src_->Dim(); }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  // The online cmvn does not introduce any additional latency.
  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //

  /// Initializer that sets the cmvn state.  If you don't have previous
  /// utterances from the same speaker you are supposed to initialize the CMVN
  /// state from some global CMVN stats, which you can get from summing all cmvn
  /// stats you have in your training data using "sum-matrix".  This just gives
  /// it a reasonable starting point at the start of the file.
  /// If you do have previous utterances from the same speaker or at least a
  /// similar environment, you are supposed to initialize it by calling GetState
  /// from the previous utterance
  OnlineCmvn(const OnlineCmvnOptions &opts,
             const OnlineCmvnState &cmvn_state,
             OnlineFeatureInterface *src);

  /// Initializer that does not set the cmvn state:
  /// after calling this, you should call SetState().
  OnlineCmvn(const OnlineCmvnOptions &opts,
             OnlineFeatureInterface *src);

  // Outputs any state information from this utterance to "cmvn_state".
  // The value of "cmvn_state" before the call does not matter: the output
  // depends on the value of OnlineCmvnState the class was initialized
  // with, the input feature values up to cur_frame, and the effects
  // of the user possibly having called Freeze().
  // If cur_frame is -1, it will just output the unmodified original
  // state that was supplied to this object.
  void GetState(int32 cur_frame,
                OnlineCmvnState *cmvn_state);

  // This function can be used to modify the state of the CMVN computation
  // from outside, but must only be called before you have processed any data
  // (otherwise it will crash).  This "state" is really just the information
  // that is propagated between utterances, not the state of the computation
  // inside an utterance.
  void SetState(const OnlineCmvnState &cmvn_state);

  // From this point it will freeze the CMN to what it would have been if
  // measured at frame "cur_frame", and it will stop it from changing
  // further. This also applies retroactively for this utterance, so if you
  // call GetFrame() on previous frames, it will use the CMVN stats
  // from cur_frame; and it applies in the future too if you then
  // call OutputState() and use this state to initialize the next
  // utterance's CMVN object.
  void Freeze(int32 cur_frame);

  virtual ~OnlineCmvn();
 private:

  /// Smooth the CMVN stats "stats" (which are stored in the normal format as a
  /// 2 x (dim+1) matrix), by possibly adding some stats from "global_stats"
  /// and/or "speaker_stats", controlled by the config.  The best way to
  /// understand the smoothing rule we use is just to look at the code.
  static void SmoothOnlineCmvnStats(const MatrixBase<double> &speaker_stats,
                                    const MatrixBase<double> &global_stats,
                                    const OnlineCmvnOptions &opts,
                                    MatrixBase<double> *stats);

  /// Get the most recent cached frame of CMVN stats.  [If no frames
  /// were cached, sets up empty stats for frame zero and returns that].
  void GetMostRecentCachedFrame(int32 frame,
                                int32 *cached_frame,
                                MatrixBase<double> *stats);

  /// Cache this frame of stats.
  void CacheFrame(int32 frame, const MatrixBase<double> &stats);

  /// Initialize ring buffer for caching stats.
  inline void InitRingBufferIfNeeded();

  /// Computes the raw CMVN stats for this frame, making use of (and updating if
  /// necessary) the cached statistics in raw_stats_.  This means the (x,
  /// x^2, count) stats for the last up to opts_.cmn_window frames.
  void ComputeStatsForFrame(int32 frame,
                            MatrixBase<double> *stats);


  OnlineCmvnOptions opts_;
  std::vector<int32> skip_dims_; // Skip CMVN for these dimensions.  Derived from opts_.
  OnlineCmvnState orig_state_;   // reflects the state before we saw this
                                 // utterance.
  Matrix<double> frozen_state_;  // If the user called Freeze(), this variable
                                 // will reflect the CMVN state that we froze
                                 // at.

  // The variable below reflects the raw (count, x, x^2) statistics of the
  // input, computed every opts_.modulus frames.  raw_stats_[n / opts_.modulus]
  // contains the (count, x, x^2) statistics for the frames from
  // std::max(0, n - opts_.cmn_window) through n.
  std::vector<Matrix<double>*> cached_stats_modulo_;
  // the variable below is a ring-buffer of cached stats.  the int32 is the
  // frame index.
  std::vector<std::pair<int32, Matrix<double> > > cached_stats_ring_;

  // Some temporary variables used inside functions of this class, which
  // put here to avoid reallocation.
  Matrix<double> temp_stats_;
  Vector<BaseFloat> temp_feats_;
  Vector<double> temp_feats_dbl_;

  OnlineFeatureInterface *src_;  // Not owned here
};


struct OnlineSpliceOptions {
  int32 left_context;
  int32 right_context;
  OnlineSpliceOptions(): left_context(4), right_context(4) { }
  void Register(ParseOptions *po) {
    po->Register("left-context", &left_context, "Left-context for frame "
                 "splicing prior to LDA");
    po->Register("right-context", &right_context, "Right-context for frame "
                 "splicing prior to LDA");
  }
};

class OnlineSpliceFrames: public OnlineFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const {
    return src_->Dim() * (1 + left_context_ + right_context_);
  }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const;

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //
  OnlineSpliceFrames(const OnlineSpliceOptions &opts,
                     OnlineFeatureInterface *src):
      left_context_(opts.left_context), right_context_(opts.right_context),
      src_(src) { }

 private:
  int32 left_context_;
  int32 right_context_;
  OnlineFeatureInterface *src_;  // Not owned here
};

/// This online-feature class implements any affine or linear transform.
class OnlineTransform: public OnlineFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return offset_.Dim(); }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void GetFrames(const std::vector<int32> &frames,
                         MatrixBase<BaseFloat> *feats);

  //
  // Next, functions that are not in the interface.
  //

  /// The transform can be a linear transform, or an affine transform
  /// where the last column is the offset.
  OnlineTransform(const MatrixBase<BaseFloat> &transform,
                  OnlineFeatureInterface *src);


 private:
  OnlineFeatureInterface *src_;  // Not owned here
  Matrix<BaseFloat> linear_term_;
  Vector<BaseFloat> offset_;
};

class OnlineDeltaFeature: public OnlineFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const;

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const;

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //
  OnlineDeltaFeature(const DeltaFeaturesOptions &opts,
                     OnlineFeatureInterface *src);

 private:
  OnlineFeatureInterface *src_;  // Not owned here
  DeltaFeaturesOptions opts_;
  DeltaFeatures delta_features_;  // This class contains just a few
                                  // coefficients.
};


/// This feature type can be used to cache its input, to avoid
/// repetition of computation in a multi-pass decoding context.
class OnlineCacheFeature: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return src_->Dim(); }

  virtual bool IsLastFrame(int32 frame) const {
    return src_->IsLastFrame(frame);
  }
  virtual BaseFloat FrameShiftInSeconds() const {
    return src_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual void GetFrames(const std::vector<int32> &frames,
                         MatrixBase<BaseFloat> *feats);

  virtual ~OnlineCacheFeature() { ClearCache(); }

  // Things that are not in the shared interface:

  void ClearCache();  // this should be called if you change the underlying
                      // features in some way.

  explicit OnlineCacheFeature(OnlineFeatureInterface *src): src_(src) { }
 private:

  OnlineFeatureInterface *src_;  // Not owned here
  std::vector<Vector<BaseFloat>* > cache_;
};




/// This online-feature class implements combination of two feature
/// streams (such as pitch, plp) into one stream.
class OnlineAppendFeature: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return src1_->Dim() + src2_->Dim(); }

  virtual bool IsLastFrame(int32 frame) const {
    return (src1_->IsLastFrame(frame) || src2_->IsLastFrame(frame));
  }
  // Hopefully sources have the same rate
  virtual BaseFloat FrameShiftInSeconds() const {
    return src1_->FrameShiftInSeconds();
  }

  virtual int32 NumFramesReady() const {
    return std::min(src1_->NumFramesReady(), src2_->NumFramesReady());
  }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual ~OnlineAppendFeature() {  }

  OnlineAppendFeature(OnlineFeatureInterface *src1,
      OnlineFeatureInterface *src2): src1_(src1), src2_(src2) { }
 private:

  OnlineFeatureInterface *src1_;
  OnlineFeatureInterface *src2_;
};

/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi

#endif  // KALDI_FEAT_ONLINE_FEATURE_H_

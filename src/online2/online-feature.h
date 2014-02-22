// online2/online-feature.h

// Copyright 2013   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE2_ONLINE_FEATURE_H_
#define KALDI_ONLINE2_ONLINE_FEATURE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/feature-functions.h"
#include "feat/feature-mfcc.h"
#include "feat/feature-plp.h"
#include "itf/online-feature-itf.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{


/// Add a virtual class for "source" features such as MFCC or PLP.
class OnlineBaseFeature: public OnlineFeatureInterface {
 public:
  // This would be called from the application, when you get
  // more wave data.  Note: the sampling_rate is only provided so
  // the code can assert that it matches the sampling rate
  // expected in the options.   
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform) = 0;

};


template<class C>
class OnlineMfccOrPlp: public OnlineBaseFeature {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return mfcc_or_plp_.Dim(); }

  // Note: this will only ever return true if you call InputFinished(), which
  // isn't really necessary to do unless you want to make sure to flush out the
  // last few frames of delta or LDA features to exactly match a non-online
  // decode of some data.
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const { return num_frames_; }
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //
  explicit OnlineMfccOrPlp(const typename C::Options &opts);

  // This would be called from the application, when you get
  // more wave data.  Note: the sampling_rate is only provided so
  // the code can assert that it matches the sampling rate
  // expected in the options.
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);


  // InputFinished() tells the class you won't be providing any
  // more waveform.  This will help flush out the last few frames
  // of delta or LDA features.
  void InputFinished() { input_finished_= true; }


 private:
  C mfcc_or_plp_; // class that does the MFCC or PLP computation

  // features_ is the MFCC or PLP features that we have already computed.
  Matrix<BaseFloat> features_;

  // True if the user has called "InputFinished()"
  bool input_finished_;

  // num_frames_ is the number of frames of MFCC features we have
  // already computed.  It may be less than the size of features_,
  // because when we resize that matrix we leave some extra room,
  // so that we don't spend too much time resizing.
  int32 num_frames_;

  // The sampling frequency, extracted from the config.  Should
  // be identical to the waveform supplied.
  BaseFloat sampling_frequency_;

  // waveform_remainder_ is a short piece of waveform that we may need to keep
  // after extracting all the whole frames we can (whatever length of feature
  // will be required for the next phase of computation).
  Vector<BaseFloat> waveform_remainder_;
};

typedef OnlineMfccOrPlp<Mfcc> OnlineMfcc;
typedef OnlineMfccOrPlp<Plp> OnlinePlp;


/// This class takes a Matrix<BaseFloat> and wraps it as an
/// OnlineFeatureInterface: this can be useful where some earlier stage of
/// feature processing has been done offline but you want to use part of the
/// online pipeline.
class OnlineMatrixFeature: public OnlineFeatureInterface {
 public:
  /// Caution: this class maintains the const reference from the constructor, so
  /// don't let it go out of scope while this object exists.
  OnlineMatrixFeature(const Matrix<BaseFloat> &mat): mat_(mat) { }

  virtual int32 Dim() const { return mat_.NumCols(); }

  virtual int32 NumFramesReady() const { return mat_.NumRows(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
    feat->CopyFromVec(mat_.Row(frame));
  }

  virtual bool IsLastFrame(int32 frame) const {
    return (frame + 1 == mat_.NumCols());
  }

 private:
  const Matrix<BaseFloat> &mat_;
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
  int32 speaker_frames; // must be <= cmn_window
  int32 global_frames; // must be <= speaker_frames.
  bool normalize_mean; // Must be true if normalize_variance==true.
  bool normalize_variance;

  int32 modulus; // not configurable from command line, relates to how the class
                 // computes the cmvn internally.
  
  OnlineCmvnOptions():
      cmn_window(600),
      speaker_frames(600),
      global_frames(200),
      normalize_mean(true),
      normalize_variance(false),
      modulus(10) { }
  
  void Check() {
    KALDI_ASSERT(speaker_frames <= cmn_window && global_frames <= speaker_frames
                 && modulus > 0);
  }

  void Register(ParseOptions *po) {
    po->Register("cmn-window", &cmn_window, "Number of frames of sliding context "
                 "for cepstral mean normalization.");
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
    po->Register("norm-mean", &normalize_mean, "If true, do mean normalization "
                 "(note: you cannot normalize the variance but not the mean)");
  }
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
  //                                       sum-sqared-stats   0    ]
  Matrix<double> global_cmvn_stats;

  // If nonempty, contains CMVN stats representing the "frozen" state
  // of CMVN that reflects how we were normalizing the data when the
  // user called the Freeze() function in class OnlineCmvn. 
  Matrix<double> frozen_state;
  
  OnlineCmvnState() { }

  OnlineCmvnState(const Matrix<double> &global_stats):
      global_cmvn_stats(global_stats) { }
};


class OnlineCmvn : public OnlineFeatureInterface {
 public:

  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return src_->Dim(); }

  virtual bool IsLastFrame(int32 frame) const { return src_->IsLastFrame(frame); }

  // The online cmvn does not introduce any additional latency.
  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }
    
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  
  //
  // Next, functions that are not in the interface.
  //

  /// Initializer that sets the cmvn state.
  OnlineCmvn(const OnlineCmvnOptions &opts,
             const OnlineCmvnState &cmvn_state,
             OnlineFeatureInterface *src):
      opts_(opts), src_(src) { SetState(cmvn_state); }

  /// Initializer that does not set the cmvn state:
  /// after calling this, you should call SetState().
  OnlineCmvn(const OnlineCmvnOptions &opts,
             OnlineFeatureInterface *src): opts_(opts), src_(src) { }
  
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
  // (otherwise it will crash).  This "state" is really just the information that
  // is propagated between utterances, not the state of the computation inside
  // an utterance.
  void SetState(const OnlineCmvnState &cmvn_state);

  // From this point it will freeze the CMN to what it would have
  // been if measured at frame "cur_frame", and it will stop it
  // from changing further. This also applies retroactively for this utterance, so if you
  // call GetFrame() on previous frames, it will use the CMVN stats
  // from cur_frame; and it applies in the future too if you then
  // call OutputState() and use this state to initialize the next
  // utterance's CMVN object.
  void Freeze(int32 cur_frame); 
  
 private:

  /// Smooth the CMVN stats "stats" (which are stored in the normal format as a
  /// 2 x (dim+1) matrix), by possibly adding some stats from "global_stats"
  /// and/or "speaker_stats", controlled by the config.  The best way to
  /// understand the smoothing rule we use is just to look at the code.
  static void SmoothOnlineCmvnStats(const MatrixBase<double> &speaker_stats,
                                    const MatrixBase<double> &global_stats,
                                    const OnlineCmvnOptions &opts,
                                    MatrixBase<double> *stats);
  
  /// Computes the raw CMVN stats for this frame, making use of (and updating if
  /// necessary) the cached statistics in raw_stats_.  This means the (x,
  /// x^2, count) stats for the last up to opts_.cmn_window frames.
  void ComputeStatsForFrame(int32 frame,
                            MatrixBase<double> *stats);
  
  
  OnlineCmvnOptions opts_;
  OnlineCmvnState orig_state_; // reflects the state before we saw this utterance.
  Matrix<double> frozen_state_; // If the user called Freeze(), this variable
                                // will reflect the CMVN state that we froze at.
  
  // The variable below reflects the raw (count, x, x^2) statistics of the input, computed 
  // every opts_.modulus frames.  raw_stats_[n / opts_.modulus] contains
  // the (count, x, x^2) statistics for the frames from std::max(0, n - opts_.cmn_window)
  // through n.
  std::vector<Matrix<double> > raw_stats_;

  OnlineFeatureInterface *src_; // Not owned here
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
    return src_->Dim() * (1 + left_context_ * right_context_);
  }

  virtual bool IsLastFrame(int32 frame) const { return src_->IsLastFrame(frame); }
  
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
  OnlineFeatureInterface *src_; // Not owned here
};

/// This online-feature class implements any affine or linear transform.
class OnlineTransform: public OnlineFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return offset_.Dim(); }

  virtual bool IsLastFrame(int32 frame) const { return src_->IsLastFrame(frame); }

  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //

  /// The transform can be a linear transform, or an affine transform
  /// where the last column is the offset.
  OnlineTransform(const MatrixBase<BaseFloat> &transform,
                  OnlineFeatureInterface *src);


 private:
  OnlineFeatureInterface *src_; // Not owned here
  Matrix<BaseFloat> linear_term_;
  Vector<BaseFloat> offset_;
};

class OnlineDeltaFeature: public OnlineFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const;

  virtual bool IsLastFrame(int32 frame) const { return src_->IsLastFrame(frame); }

  virtual int32 NumFramesReady() const;

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //
  OnlineDeltaFeature(const DeltaFeaturesOptions &opts,
                     OnlineFeatureInterface *src);

 private:
  OnlineFeatureInterface *src_; // Not owned here
  DeltaFeaturesOptions opts_;
  DeltaFeatures delta_features_; // This class contains just a few coefficients.
};


/// This feature type can be used to cache its input, to avoid
/// repetition of computation in a multi-pass decoding context.
class OnlineCacheFeature: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const { return src_->Dim(); }

  virtual bool IsLastFrame(int32 frame) const { return src_->IsLastFrame(frame); }

  virtual int32 NumFramesReady() const { return src_->NumFramesReady(); }
  
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  virtual ~OnlineCacheFeature() { ClearCache(); }

  // Things that are not in the shared interface:

  void ClearCache(); // this should be called if you change the underlying
                     // features in some way.

  OnlineCacheFeature(OnlineFeatureInterface *src): src_(src) { }
 private:
  
  OnlineFeatureInterface *src_; // Not owned here
  std::vector<Vector<BaseFloat>* > cache_;
};


/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_FEATURE_

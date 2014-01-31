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

template<class C>
class OnlineMfccOrPlp: public OnlineFeatureInterface {
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
  void AcceptWaveform(BaseFloat sampling_rate,
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


/**
   This struct stores stats fromthe OnlineCmvn feature-processing
   object.  It is useful in transferring stats between utterances.
   It also has a bool "frozen" value that is true if someone has
   called Freeze() on the OnlineCmvn object that was used to create
   these stats.  In that case the stats will represent the stats
   up to the point where we froze the CMVN.  
*/
struct OnlineCmvnStats {
  OnlineCmvnStats(): frozen(false) { }
  Matrix<BaseFloat> stats;
  bool frozen; /// if true, we're not further updating the CMVN
               /// but will leave it at its current value.
};

class OnlineCmvn : public OnlineFeatureInterface {
 public:
  // Initializer that includes prior stats (e.g. global stats,
  // or stats from the same utterance).  Just provide empty matrices
  // if you don't have stats available for either category
  OnlineCmvn(SlidingWindowCmnOptions &opts,
             const Matrix<BaseFloat> &global_stats,
             const OnlineCmvnStats &prev_utt_stats);
  
  
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return src_->Dim(); }

  void Freeze(); // From this point it will freeze the CMN and will stop it
                 // from changing.  Note: it won't freeze it to its current
                 // value from online sliding-window adaptation, it will freeze it
                 // to the whole-file value.  You can use GetCurrentStats()

  // Gets the current
  void GetCurrentState(
      Matrix<BaseFloat> *cmvn_stats);

  virtual bool IsLastFrame(int32 frame) const { return src_->IsLastFrame(frame); }

  // FIXME now CMVN "eats" min-window frames
  virtual int32 NumFramesReady() {
    return std::max(src_->NumFramesReady() - min_window_, 0); 
  }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //
  OnlineCmvn(int32 cmvn_window, OnlineFeatureInterface *src):
    norm_var_(true), cmvn_window_(cmvn_window), src_(src), is_online_(true) 
    { sliding_stat_.Resize(2, src->Dim() + 1, kSetZero); }

  /// Should be called after calling GetFrame on the last available frame
  void GetStats(Matrix<BaseFloat> *stats) const {
    stats->CopyFromMat(sliding_stat_);
  }

  /// Start using immediately the statistics for normalisation.
  void ApplyStats(const Matrix<BaseFloat> &stats);

  int32 WindowSize() { 
    int32 last = sliding_stat_(0, sliding_stat_.NumCols() - 1);
    int32 r = sliding_stat_(0, last); 
    KALDI_ASSERT(r == window_.size());
    return r;
  }

 private:
  // TODO move norm_var_, min_window_, cmn_window_, is_online to OnlineCmvnOpts struct
  bool norm_var_;  // FIXME does not store 2. row if norm_var_==false
  int32 min_window_;
  int32 cmvn_window_;
  Matrix<double> sliding_stat_;  // first row of Matrix is cumulated sum,
                              // second row is the cumulated squared sum 
                              // Matrix size is (2, Dim() + 1)
                              // At Matrix(0, dim) is stored used window size
                              // At Matrix(1, dim) is always stored 0 -> dummy
  std::deque<Matrix<double> > window_;
  std::vector<Matrix<double> > stats_; 
  OnlineFeatureInterface *src_;
  bool is_online_;
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
  OnlineSpliceFrames(int32 left_context, int32 right_context,
                     OnlineFeatureInterface *src):
      left_context_(left_context), right_context_(right_context),
      src_(src), is_online_(true) { }

 private:
  int32 left_context_;
  int32 right_context_;
  OnlineFeatureInterface *src_;
  bool is_online_;
};

/// This online-feature class implements LDA, or more generally any linear or
/// affine transform.  It doesn't do frame splicing: for that, use
/// class OnlineSpliceFrames.
class OnlineLda: public OnlineFeatureInterface {
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const { return offset_.Dim(); }

  virtual bool IsLastFrame(int32 frame) { return src_->IsLastFrame(frame); }

  virtual bool NumFramesReady() { return src_->NumFramesReady(); }

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  //
  // Next, functions that are not in the interface.
  //

  /// The transform can be a linear transform, or an affine transform
  /// where the last column is the offset.
  OnlineLda(const Matrix<BaseFloat> &transform,
            OnlineFeatureInterface *src);

 private:
  OnlineFeatureInterface *src_;
  Matrix<BaseFloat> linear_term_;
  Vector<BaseFloat> offset_;
  bool is_online_;
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
  OnlineFeatureInterface *src_;
  DeltaFeaturesOptions opts_;
  DeltaFeatures delta_features_; // This class contains just a few coefficients.
  bool is_online_;
};

class OnlineCacheFeature: public OnlineFeatureInterface {
 public:
  virtual int32 Dim() const;

  void EmptyCache(); // this should be called if you change the underlying
                     // features in some way.
 private:
  // ...
};


/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_FEATURE_H_

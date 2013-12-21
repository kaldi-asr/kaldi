// feat/online-feature-functions.h

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


#ifndef KALDI_FEAT_FEATURE_FUNCTIONS_H_
#define KALDI_FEAT_FEATURE_FUNCTIONS_H_

#include <string>
#include <vector>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/feature-functions.h"
#include "feat/feature-mfcc.h"
#include "feat/feature-plp.h"
#include "itf/online-feature-input.h"

namespace kaldi {
/// @addtogroup  onilinefeat OnlineFeatureExtraction
/// @{



class OnlineMfcc: public OnlineFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const;
  virtual void Recompute() { } // This does nothing since MFCC computation is
                               // not affected by future context.

  // Note: this will only ever return true if you call InputFinished(), which
  // isn't really necessary to do unless you want to make sure to flush out the
  // last few frames of delta or LDA features to exactly match a non-online
  // decode of some data.  
  virtual bool IsLastFrame(int32 frame);
  virtual int32 NumFramesReady();
  virtual void GetFeature(int32 frame, VectorBase<BaseFloat> *feat);  

  //
  // Next, functions that are not in the interface.
  //
  explicit OnlineMfcc(const MfccOptions &opts);

  // This would be called from the application, when you get
  // more wave data.  Note: the sampling_rate is only provided so
  // the code can assert that it matches the sampling rate
  // expected in the options.
  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);


  // InputFinished() tells the class you won't be providing any
  // more waveform.  This will help flush out the last few frames
  // of delta or LDA features.
  void InputFinished();
  
 private:
  // TODO: implement this.
};

class OnlineSpliceFrames: public OnlineFeatureInterface {
 public:
  //
  // First, functions that are present in the interface:
  //
  virtual int32 Dim() const {
    return src_->Dim() * (1 + left_context_ * right_context_);
  }

  virtual void Recompute() { src_->Recompute(); }

  virtual bool IsLastFrame(int32 frame) { return src_->IsLastFrame(frame); }

  virtual int32 NumFramesReady() { // TODO: move to .cc
    int32 nf = src_->NumFramesReady();
    if (nf > 0 && src_->IsLastFrame(nf-1)) return nf;
    else return std::max<int32>(0, nf - right_context_);
  }

  virtual void GetFeature(int32 frame, VectorBase<BaseFloat> *feat);  
  
  //
  // Next, functions that are not in the interface.
  //
  OnlineSpliceFrames(int32 left_context, int32 right_context,
                     OnlineFeatureInterface *src):
      left_context_(left_context), right_context_(right_context), src_(src) { }
  
 private:
  int32 left_context_;
  int32 right_context_;
  OnlineFeatureInterface *src_;
};

// This implements LDA, or more generally any linear or affine transform.
class OnlineLda: public OnlineFeatureInterface {
  //
  // First, functions that are present in the interface:
  //
  // TODO.

  //
  // Next, functions that are not in the interface.
  //
  OnlineLda(const Matrix<BaseFloat> &transform,
            OnlineFeatureInterface *src);
  
 private:
  Matrix<BaseFloat> linear_part_;
  Vector<BaseFloat> offset_;  
};




/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi



#endif  // KALDI_FEAT_ONLINE_FEATURE_FUNCTIONS_H_

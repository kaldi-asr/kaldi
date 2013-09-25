// tied/decodable-am-tied-full-gmm.h

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#ifndef KALDI_TIED_DECODABLE_AM_TIED_FULL_GMM_H_
#define KALDI_TIED_DECODABLE_AM_TIED_FULL_GMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "tied/tied-gmm.h"
#include "tied/am-tied-full-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"

namespace kaldi {

class DecodableAmTiedFullGmm : public DecodableInterface {
 public:
  DecodableAmTiedFullGmm(const AmTiedFullGmm &am,
                         const TransitionModel &tm,
                         const Matrix<BaseFloat> &feats)
      : acoustic_model_(am), feature_matrix_(feats), trans_model_(tm),
        previous_frame_(-1) {
    ResetLogLikeCache();
    acoustic_model_.SetupPerFrameVars(&per_frame_vars_);
  }

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid));
  }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumFrames() { return feature_matrix_.NumRows(); }
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }
  
  virtual bool ComparePdfId(int32 tid1, int32 tid2) {
    int32 index1 = trans_model_.TransitionIdToPdf(tid1);
    int32 index2 = trans_model_.TransitionIdToPdf(tid2);
    return index1 == index2;
  }

  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

  void ResetLogLikeCache();

 protected:
  virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 pdf_index);
 
  const AmTiedFullGmm &acoustic_model_;
  const Matrix<BaseFloat> &feature_matrix_;
  const TransitionModel &trans_model_;
  
  /// we can save some breath by remembering the last frame
  int32 previous_frame_;
  TiedGmmPerFrameVars per_frame_vars_;

  /// Defines a cache record for a pdf
  struct LikelihoodCacheRecord {
    BaseFloat log_like;  ///< Cache value
    int32 hit_time;      ///< Frame for which this value is relevant
  };
  
  std::vector<LikelihoodCacheRecord> log_like_cache_;

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmTiedFullGmm);
};

class DecodableAmTiedFullGmmScaled : public DecodableAmTiedFullGmm {
 public:
  DecodableAmTiedFullGmmScaled(const AmTiedFullGmm &am,
                               const TransitionModel &tm,
                               const Matrix<BaseFloat> &feats,
                               BaseFloat scale)
      : DecodableAmTiedFullGmm(am, tm, feats), scale_(scale) { }

  // Note, frames are numbered from zero but transition-ids from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_ * LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid));
  }

 private:
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmTiedFullGmmScaled);
};


}  // namespace kaldi

#endif  // KALDI_TIED_DECODABLE_AM_TIED_FULL_GMM_H_

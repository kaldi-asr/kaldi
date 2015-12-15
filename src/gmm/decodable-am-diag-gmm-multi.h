// gmm/decodable-am-diag-gmm.h

// Copyright 2015 Hainan Xu

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

#ifndef KALDI_GMM_DECODABLE_AM_DIAG_GMM_H_MULTI_
#define KALDI_GMM_DECODABLE_AM_DIAG_GMM_H_MULTI_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "transform/regression-tree.h"
#include "transform/regtree-fmllr-diag-gmm.h"
#include "transform/regtree-mllr-diag-gmm.h"

#include <vector>
using std::vector;

namespace kaldi {

// for multi-tree decoding
class DecodableAmDiagGmmUnmappedMulti : public DecodableInterface {
 public:
  /// If you set log_sum_exp_prune to a value greater than 0 it will prune
  /// in the LogSumExp operation (larger = more exact); I suggest 5.
  /// This is advisable if it's spending a long time doing exp 
  /// operations. 
  DecodableAmDiagGmmUnmappedMulti(const vector<AmDiagGmm> &ams,
                           const unordered_map<int32, vector<int32> >& mapping,
                           const Matrix<BaseFloat> &feats,
                           BaseFloat exp_weight = 0.1,
                           BaseFloat log_sum_exp_prune = -1.0):
    acoustic_models_(ams), feature_matrix_(feats),
    previous_frame_(-1), log_sum_exp_prune_(log_sum_exp_prune),
    mapping_(mapping), 
    exp_weight_(exp_weight),
    data_squared_(feats.NumCols()) {
    ResetLogLikeCache();
  }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 state_index) {
    return LogLikelihoodZeroBased(frame, state_index - 1);
  }

  int32 NumFrames() const { return feature_matrix_.NumRows(); }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return mapping_.size(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

 protected:
  void ResetLogLikeCache();
  virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 state_index);

  const vector<AmDiagGmm> &acoustic_models_;
  const Matrix<BaseFloat> &feature_matrix_;
  int32 previous_frame_;
  BaseFloat log_sum_exp_prune_;
  unordered_map<int32, vector<int32> > mapping_;

  /// Defines a cache record for a state
  struct LikelihoodCacheRecord {
    BaseFloat log_like;  ///< Cache value
    int32 hit_time;     ///< Frame for which this value is relevant
  };
  std::vector<LikelihoodCacheRecord> log_like_cache_;
  BaseFloat exp_weight_;
 private:
  Vector<BaseFloat> data_squared_;  ///< Cache for fast likelihood calculation

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmUnmappedMulti);
};


class DecodableAmDiagGmmScaledMulti: public DecodableAmDiagGmmUnmappedMulti {
 public:
  DecodableAmDiagGmmScaledMulti(const vector<AmDiagGmm> &ams,
        const unordered_map<int32, vector<int32> > &mapping,
                           TransitionModel &tm,
                           const Matrix<BaseFloat> &feats,
                           BaseFloat scale,
                           BaseFloat exp_weight = 0.1,
                           BaseFloat log_sum_exp_prune = -1.0):
      DecodableAmDiagGmmUnmappedMulti(ams, mapping, feats,
                                      exp_weight, log_sum_exp_prune),
      trans_model_(tm),
      scale_(scale), delete_feats_(NULL) {
  }

  // This version of the initializer takes ownership of the pointer
  // "feats" and will delete it when this class is destroyed.
  DecodableAmDiagGmmScaledMulti(const vector<AmDiagGmm> &ams,
        const unordered_map<int32, vector<int32> > &mapping,
                           const TransitionModel &tm,
                           BaseFloat scale,
                           BaseFloat log_sum_exp_prune,
                           Matrix<BaseFloat> *feats):
      DecodableAmDiagGmmUnmappedMulti(ams, mapping, *feats, log_sum_exp_prune),
      trans_model_(tm),  scale_(scale), delete_feats_(feats) {
  }

  
  // Note, frames are numbered from zero but transition-ids from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_*LogLikelihoodZeroBased(frame,
                                         trans_model_.TransitionIdToPdf(tid));
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  // virtual int32 NumIndices() { return mapping_.size(); }
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  const TransitionModel *TransModel() { return &trans_model_; }

  virtual ~DecodableAmDiagGmmScaledMulti() {
    if (delete_feats_) delete delete_feats_;
  }
  
 private: // want to access it public to have pdf id information
  const TransitionModel& trans_model_;  // for transition-id to pdf mapping
  BaseFloat scale_;
  Matrix<BaseFloat> *delete_feats_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmScaledMulti);
};

} // namespace kaldi

#endif

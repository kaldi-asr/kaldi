// gmm/decodable-am-diag-gmm.h

// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Lukas Burget
//                2013  Johns Hopkins Universith (author: Daniel Povey)

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

#ifndef KALDI_GMM_DECODABLE_AM_DIAG_GMM_H_
#define KALDI_GMM_DECODABLE_AM_DIAG_GMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "transform/regression-tree.h"
#include "transform/regtree-fmllr-diag-gmm.h"
#include "transform/regtree-mllr-diag-gmm.h"

namespace kaldi {

/// DecodableAmDiagGmmUnmapped is a decodable object that
/// takes indices that correspond to pdf-id's plus one.
/// This may be used in future in a decoder that doesn't need
/// to output alignments, if we create FSTs that have the pdf-ids
/// plus one as the input labels (we couldn't use the pdf-ids
/// themselves because they start from zero, and the graph might
/// have epsilon transitions).

class DecodableAmDiagGmmUnmapped : public DecodableInterface {
 public:
  /// If you set log_sum_exp_prune to a value greater than 0 it will prune
  /// in the LogSumExp operation (larger = more exact); I suggest 5.
  /// This is advisable if it's spending a long time doing exp 
  /// operations. 
  DecodableAmDiagGmmUnmapped(const AmDiagGmm &am,
                             const Matrix<BaseFloat> &feats,
                             BaseFloat log_sum_exp_prune = -1.0):
    acoustic_model_(am), feature_matrix_(feats),
    previous_frame_(-1), log_sum_exp_prune_(log_sum_exp_prune), 
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
  virtual int32 NumIndices() const { return acoustic_model_.NumPdfs(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

 protected:
  void ResetLogLikeCache();
  virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 state_index);

  const AmDiagGmm &acoustic_model_;
  const Matrix<BaseFloat> &feature_matrix_;
  int32 previous_frame_;
  BaseFloat log_sum_exp_prune_;

  /// Defines a cache record for a state
  struct LikelihoodCacheRecord {
    BaseFloat log_like;  ///< Cache value
    int32 hit_time;     ///< Frame for which this value is relevant
  };
  std::vector<LikelihoodCacheRecord> log_like_cache_;
 private:
  Vector<BaseFloat> data_squared_;  ///< Cache for fast likelihood calculation


  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmUnmapped);
};


class DecodableAmDiagGmm: public DecodableAmDiagGmmUnmapped {
 public:
  DecodableAmDiagGmm(const AmDiagGmm &am,
                     const TransitionModel &tm,
                     const Matrix<BaseFloat> &feats,
                     BaseFloat log_sum_exp_prune = -1.0)
    : DecodableAmDiagGmmUnmapped(am, feats, log_sum_exp_prune),
      trans_model_(tm) {}

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return LogLikelihoodZeroBased(frame,
                                  trans_model_.TransitionIdToPdf(tid));
  }
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  const TransitionModel *TransModel() { return &trans_model_; }
 private: // want to access public to have pdf id information
  const TransitionModel &trans_model_;  // for tid to pdf mapping
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmm);
};

class DecodableAmDiagGmmScaled: public DecodableAmDiagGmmUnmapped {
 public:
  DecodableAmDiagGmmScaled(const AmDiagGmm &am,
                           const TransitionModel &tm,
                           const Matrix<BaseFloat> &feats,
                           BaseFloat scale,
                           BaseFloat log_sum_exp_prune = -1.0):
      DecodableAmDiagGmmUnmapped(am, feats, log_sum_exp_prune), trans_model_(tm),
      scale_(scale), delete_feats_(NULL) {}

  // This version of the initializer takes ownership of the pointer
  // "feats" and will delete it when this class is destroyed.
  DecodableAmDiagGmmScaled(const AmDiagGmm &am,
                           const TransitionModel &tm,
                           BaseFloat scale,
                           BaseFloat log_sum_exp_prune,
                           Matrix<BaseFloat> *feats):
      DecodableAmDiagGmmUnmapped(am, *feats, log_sum_exp_prune),
      trans_model_(tm),  scale_(scale), delete_feats_(feats) {}

  
  // Note, frames are numbered from zero but transition-ids from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_*LogLikelihoodZeroBased(frame,
                                         trans_model_.TransitionIdToPdf(tid));
  }
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

  const TransitionModel *TransModel() { return &trans_model_; }

  virtual ~DecodableAmDiagGmmScaled() {
    if (delete_feats_) delete delete_feats_;
  }
  
 private: // want to access it public to have pdf id information
  const TransitionModel &trans_model_;  // for transition-id to pdf mapping
  BaseFloat scale_;
  Matrix<BaseFloat> *delete_feats_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmScaled);
};

}  // namespace kaldi

#endif  // KALDI_GMM_DECODABLE_AM_DIAG_GMM_H_

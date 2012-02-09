// decoder/decodable-am-diag-gmm.h

// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Lukas Burget

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

#ifndef KALDI_DECODER_DECODABLE_AM_DIAG_GMM_H_
#define KALDI_DECODER_DECODABLE_AM_DIAG_GMM_H_

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
  DecodableAmDiagGmmUnmapped(const AmDiagGmm &am,
                             const Matrix<BaseFloat> &feats)
      : acoustic_model_(am), feature_matrix_(feats),
        previous_frame_(-1), data_squared_(feats.NumCols()) {
    ResetLogLikeCache();
  }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 state_index) {
    return LogLikelihoodZeroBased(frame, state_index - 1);
  }
  int32 NumFrames() { return feature_matrix_.NumRows(); }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return acoustic_model_.NumPdfs(); }

  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

  void ResetLogLikeCache();
 protected:
  virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 state_index);

  const AmDiagGmm &acoustic_model_;
  const Matrix<BaseFloat> &feature_matrix_;
  int32 previous_frame_;

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
                     const Matrix<BaseFloat> &feats)
      : DecodableAmDiagGmmUnmapped(am, feats), trans_model_(tm) {}

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return LogLikelihoodZeroBased(frame,
                                  trans_model_.TransitionIdToPdf(tid));
  }
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

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
                           BaseFloat scale)
      : DecodableAmDiagGmmUnmapped(am, feats), trans_model_(tm),
        scale_(scale) {}

  // Note, frames are numbered from zero but transition-ids from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_*LogLikelihoodZeroBased(frame,
                                         trans_model_.TransitionIdToPdf(tid));
  }
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

  const TransitionModel *TransModel() { return &trans_model_; }
 private: // want to access it public to have pdf id information
  const TransitionModel &trans_model_;  // for transition-id to pdf mapping
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmScaled);
};

class DecodableAmDiagGmmRegtreeFmllr: public DecodableAmDiagGmmUnmapped {
 public:
  DecodableAmDiagGmmRegtreeFmllr(const AmDiagGmm &am,
                                 const TransitionModel &tm,
                                 const Matrix<BaseFloat> &feats,
                                 const RegtreeFmllrDiagGmm &fmllr_xform,
                                 const RegressionTree &regtree,
                                 BaseFloat scale)
      : DecodableAmDiagGmmUnmapped(am, feats), trans_model_(tm), scale_(scale),
        fmllr_xform_(fmllr_xform), regtree_(regtree), valid_logdets_(false) {}

  // Note, frames are numbered from zero but transition-ids (tid) from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_*LogLikelihoodZeroBased(frame,
                                         trans_model_.TransitionIdToPdf(tid));
  }

  virtual int32 NumFrames() { return feature_matrix_.NumRows(); }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

 protected:
  virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 state_index);

  const TransitionModel *TransModel() { return &trans_model_; }

 private:
  const TransitionModel &trans_model_;  // for transition-id to pdf mapping
  BaseFloat scale_;
  const RegtreeFmllrDiagGmm &fmllr_xform_;
  const RegressionTree &regtree_;
  std::vector< Vector<BaseFloat> > xformed_data_;
  std::vector< Vector<BaseFloat> > xformed_data_squared_;
  Vector<BaseFloat> logdets_;
  bool valid_logdets_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmRegtreeFmllr);
};

class DecodableAmDiagGmmRegtreeMllr: public DecodableAmDiagGmmUnmapped {
 public:
  DecodableAmDiagGmmRegtreeMllr(const AmDiagGmm &am,
                                const TransitionModel &tm,
                                const Matrix<BaseFloat> &feats,
                                const RegtreeMllrDiagGmm &mllr_xform,
                                const RegressionTree &regtree,
                                BaseFloat scale)
      : DecodableAmDiagGmmUnmapped(am, feats), trans_model_(tm), scale_(scale),
        mllr_xform_(mllr_xform), regtree_(regtree),
        data_squared_(feats.NumCols()) { InitCache(); }
  ~DecodableAmDiagGmmRegtreeMllr();

  // Note, frames are numbered from zero but transition-ids (tid) from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_*LogLikelihoodZeroBased(frame,
                                         trans_model_.TransitionIdToPdf(tid));
  }

  virtual int32 NumFrames() { return feature_matrix_.NumRows(); }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

  const TransitionModel *TransModel() { return &trans_model_; }
  
 protected:
  virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 state_index);

 private:
  /// Initializes the mean & gconst caches
  void InitCache();
  /// Get the transformed means times inverse variances for a given pdf, and
  /// cache them. The 'state_index' is 0-based.
  const Matrix<BaseFloat>& GetXformedMeanInvVars(int32 state_index);
  /// Get the cached (while computing transformed means) gconsts for
  /// likelihood calculation. The 'state_index' is 0-based.
  const Vector<BaseFloat>& GetXformedGconsts(int32 state_index);

  const TransitionModel &trans_model_;  // for transition-id to pdf mapping
  BaseFloat scale_;
  const RegtreeMllrDiagGmm &mllr_xform_;
  const RegressionTree &regtree_;
  // we want it public to have access to the pdf ids

  /// Cache of transformed means time inverse variances for each state.
  std::vector< Matrix<BaseFloat>* > xformed_mean_invvars_;
  /// Cache of transformed gconsts for each state.
  std::vector< Vector<BaseFloat>* > xformed_gconsts_;
  /// Boolean variable per state to indicate whether the transformed means for
  /// that state are cahced.
  std::vector<bool> is_cached_;

  Vector<BaseFloat> data_squared_;  ///< Cached for fast likelihood calculation

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmRegtreeMllr);
};

}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_AM_DIAG_GMM_H_

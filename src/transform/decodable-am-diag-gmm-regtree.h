// transform/decodable-am-diag-gmm-regtree.h

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

#ifndef KALDI_TRANSFORM_DECODABLE_AM_DIAG_GMM_REGTREE_H_
#define KALDI_TRANSFORM_DECODABLE_AM_DIAG_GMM_REGTREE_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "transform/regression-tree.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "transform/regtree-fmllr-diag-gmm.h"
#include "transform/regtree-mllr-diag-gmm.h"

namespace kaldi {

class DecodableAmDiagGmmRegtreeFmllr: public DecodableAmDiagGmmUnmapped {
 public:
  DecodableAmDiagGmmRegtreeFmllr(const AmDiagGmm &am,
                                 const TransitionModel &tm,
                                 const Matrix<BaseFloat> &feats,
                                 const RegtreeFmllrDiagGmm &fmllr_xform,
                                 const RegressionTree &regtree,
                                 BaseFloat scale,
                                 BaseFloat log_sum_exp_prune = -1.0)
    : DecodableAmDiagGmmUnmapped(am, feats, log_sum_exp_prune), trans_model_(tm),
      scale_(scale), fmllr_xform_(fmllr_xform), regtree_(regtree),
      valid_logdets_(false) {}

  // Note, frames are numbered from zero but transition-ids (tid) from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_*LogLikelihoodZeroBased(frame,
                                         trans_model_.TransitionIdToPdf(tid));
  }

  virtual int32 NumFramesReady() const { return feature_matrix_.NumRows(); }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

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
                                BaseFloat scale,
                                BaseFloat log_sum_exp_prune = -1.0):
      DecodableAmDiagGmmUnmapped(am, feats, log_sum_exp_prune),
      trans_model_(tm), scale_(scale), mllr_xform_(mllr_xform),
      regtree_(regtree), data_squared_(feats.NumCols()) { InitCache(); }
  ~DecodableAmDiagGmmRegtreeMllr();

  // Note, frames are numbered from zero but transition-ids (tid) from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_*LogLikelihoodZeroBased(frame,
                                         trans_model_.TransitionIdToPdf(tid));
  }

  virtual int32 NumFramesReady() const { return feature_matrix_.NumRows(); }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

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
  /// that state are cached.
  std::vector<bool> is_cached_;

  Vector<BaseFloat> data_squared_;  ///< Cached for fast likelihood calculation

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmDiagGmmRegtreeMllr);
};

}  // namespace kaldi

#endif  // KALDI_TRANSFORM_DECODABLE_AM_DIAG_GMM_REGTREE_H_

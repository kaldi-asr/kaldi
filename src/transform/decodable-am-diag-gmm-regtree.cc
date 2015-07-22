// transform/decodable-am-diag-gmm-regtree.cc

// Copyright 2009-2011  Saarland University;  Lukas Burget
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

#include <vector>
using std::vector;

#include "transform/decodable-am-diag-gmm-regtree.h"

namespace kaldi {


BaseFloat DecodableAmDiagGmmRegtreeFmllr::LogLikelihoodZeroBased(int32 frame,
                                                          int32 state) {
  KALDI_ASSERT(frame < NumFramesReady() && frame >= 0);
  KALDI_ASSERT(state < NumIndices() && state >= 0);

  if (!valid_logdets_) {
    logdets_.Resize(fmllr_xform_.NumRegClasses());
    fmllr_xform_.GetLogDets(&logdets_);
    valid_logdets_ = true;
  }

  if (log_like_cache_[state].hit_time == frame) {
    return log_like_cache_[state].log_like;  // return cached value, if found
  }

  const DiagGmm &pdf = acoustic_model_.GetPdf(state);
  const VectorBase<BaseFloat> &data = feature_matrix_.Row(frame);

  // check if everything is in order
  if (pdf.Dim() != data.Dim()) {
    KALDI_ERR << "Dim mismatch: data dim = "  << data.Dim()
        << " vs. model dim = " << pdf.Dim();
  }
  if (!pdf.valid_gconsts()) {
    KALDI_ERR << "State "  << (state)  << ": Must call ComputeGconsts() "
        "before computing likelihood.";
  }

  if (frame != previous_frame_) {  // cache the transformed & squared stats.
    fmllr_xform_.TransformFeature(data, &xformed_data_);
    xformed_data_squared_ = xformed_data_;
    vector< Vector <BaseFloat> >::iterator it = xformed_data_squared_.begin(),
        end = xformed_data_squared_.end();
    for (; it != end; ++it) { it->ApplyPow(2.0); }
    previous_frame_ = frame;
  }

  Vector<BaseFloat> loglikes(pdf.gconsts());  // need to recreate for each pdf
  int32 baseclass, regclass;
  for (int32 comp_id = 0, num_comp = pdf.NumGauss(); comp_id < num_comp;
      ++comp_id) {
    baseclass = regtree_.Gauss2BaseclassId(state, comp_id);
    regclass = fmllr_xform_.Base2RegClass(baseclass);
    // loglikes +=  means * inv(vars) * data.
    loglikes(comp_id) += VecVec(pdf.means_invvars().Row(comp_id),
                                xformed_data_[regclass]);
    // loglikes += -0.5 * inv(vars) * data_sq.
    loglikes(comp_id) -= 0.5 * VecVec(pdf.inv_vars().Row(comp_id),
                                      xformed_data_squared_[regclass]);
    loglikes(comp_id) += logdets_(regclass);
  }

  BaseFloat log_sum = loglikes.LogSumExp(log_sum_exp_prune_);
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

  log_like_cache_[state].log_like = log_sum;
  log_like_cache_[state].hit_time = frame;

  return log_sum;
}

DecodableAmDiagGmmRegtreeMllr::~DecodableAmDiagGmmRegtreeMllr() {
  DeletePointers(&xformed_mean_invvars_);
  DeletePointers(&xformed_gconsts_);
}


void DecodableAmDiagGmmRegtreeMllr::InitCache() {
  if (xformed_mean_invvars_.size() != 0)
    DeletePointers(&xformed_mean_invvars_);
  if (xformed_gconsts_.size() != 0)
    DeletePointers(&xformed_gconsts_);
  int32 num_pdfs = acoustic_model_.NumPdfs();
  xformed_mean_invvars_.resize(num_pdfs);
  xformed_gconsts_.resize(num_pdfs);
  is_cached_.resize(num_pdfs, false);
  ResetLogLikeCache();
}


// This is almost the same code as DiagGmm::ComputeGconsts, except that
// means are used instead of means * inv(vars). This saves some computation.
static void ComputeGconsts(const VectorBase<BaseFloat> &weights,
                           const MatrixBase<BaseFloat> &means,
                           const MatrixBase<BaseFloat> &inv_vars,
                           VectorBase<BaseFloat> *gconsts_out) {
  int32 num_gauss = weights.Dim();
  int32 dim = means.NumCols();
  KALDI_ASSERT(means.NumRows() == num_gauss
      && inv_vars.NumRows() == num_gauss && inv_vars.NumCols() == dim);
  KALDI_ASSERT(gconsts_out->Dim() == num_gauss);

  BaseFloat offset = -0.5 * M_LOG_2PI * dim;  // constant term in gconst.
  int32 num_bad = 0;

  for (int32 gauss = 0; gauss < num_gauss; gauss++) {
    KALDI_ASSERT(weights(gauss) >= 0);  // Cannot have negative weights.
    BaseFloat gc = Log(weights(gauss)) + offset;  // May be -inf if weights == 0
    for (int32 d = 0; d < dim; d++) {
      gc += 0.5 * Log(inv_vars(gauss, d)) - 0.5 * means(gauss, d)
        * means(gauss, d) * inv_vars(gauss, d);  // diff from DiagGmm version.
    }

    if (KALDI_ISNAN(gc)) {  // negative infinity is OK but NaN is not acceptable
      KALDI_ERR << "At component "  << gauss
                << ", not a number in gconst computation";
    }
    if (KALDI_ISINF(gc)) {
      num_bad++;
      // If positive infinity, make it negative infinity.
      // Want to make sure the answer becomes -inf in the end, not NaN.
      if (gc > 0) gc = -gc;
    }
    (*gconsts_out)(gauss) = gc;
  }
  if (num_bad > 0)
    KALDI_WARN << num_bad << " unusable components found while computing "
               << "gconsts.";
}


const Matrix<BaseFloat>& DecodableAmDiagGmmRegtreeMllr::GetXformedMeanInvVars(
    int32 state) {
  if (is_cached_[state]) {  // found in cache
    KALDI_ASSERT(xformed_mean_invvars_[state] != NULL);
    KALDI_VLOG(3) << "For PDF index " << state << ": transformed means "
                  << "found in cache.";
    return *xformed_mean_invvars_[state];
  } else {  // transform the means and cache them
    KALDI_ASSERT(xformed_mean_invvars_[state] == NULL);
    KALDI_VLOG(3) << "For PDF index " << state << ": transforming means.";
    int32 num_gauss = acoustic_model_.GetPdf(state).NumGauss(),
        dim = acoustic_model_.Dim();
    const Vector<BaseFloat> &weights = acoustic_model_.GetPdf(state).weights();
    const Matrix<BaseFloat> &invvars = acoustic_model_.GetPdf(state).inv_vars();
    xformed_mean_invvars_[state] = new Matrix<BaseFloat>(num_gauss, dim);
    mllr_xform_.GetTransformedMeans(regtree_, acoustic_model_, state,
                                    xformed_mean_invvars_[state]);
    xformed_gconsts_[state] = new Vector<BaseFloat>(num_gauss);
    // At this point, the transformed means haven't been multiplied with
    // the inv vars, and they are used to compute gconsts first.
    ComputeGconsts(weights, *xformed_mean_invvars_[state], invvars,
                   xformed_gconsts_[state]);
    // Finally, multiply the transformed means with the inv vars.
    xformed_mean_invvars_[state]->MulElements(invvars);
    is_cached_[state] = true;
    return *xformed_mean_invvars_[state];
  }
}

const Vector<BaseFloat>& DecodableAmDiagGmmRegtreeMllr::GetXformedGconsts(
    int32 state) {
  if (!is_cached_[state]) {
    KALDI_ERR << "GConsts not cached for state: " << state << ". Must call "
              << "GetXformedMeanInvVars() first.";
  }
  KALDI_ASSERT(xformed_gconsts_[state] != NULL);
  return *xformed_gconsts_[state];
}

BaseFloat DecodableAmDiagGmmRegtreeMllr::LogLikelihoodZeroBased(int32 frame,
                                                                int32 state) {
//  KALDI_ERR << "Function not completely implemented yet.";
  KALDI_ASSERT(frame < NumFramesReady() && frame >= 0);
  KALDI_ASSERT(state < NumIndices() && state >= 0);

  if (log_like_cache_[state].hit_time == frame) {
    return log_like_cache_[state].log_like;  // return cached value, if found
  }

  const DiagGmm &pdf = acoustic_model_.GetPdf(state);
  const VectorBase<BaseFloat> &data = feature_matrix_.Row(frame);

  // check if everything is in order
  if (pdf.Dim() != data.Dim()) {
    KALDI_ERR << "Dim mismatch: data dim = "  << data.Dim()
        << " vs. model dim = " << pdf.Dim();
  }

  if (frame != previous_frame_) {  // cache the squared stats.
    data_squared_.CopyFromVec(feature_matrix_.Row(frame));
    data_squared_.ApplyPow(2.0);
    previous_frame_ = frame;
  }

  const Matrix<BaseFloat> &means_invvars = GetXformedMeanInvVars(state);
  const Vector<BaseFloat> &gconsts = GetXformedGconsts(state);

  Vector<BaseFloat> loglikes(gconsts);  // need to recreate for each pdf
  // loglikes +=  means * inv(vars) * data.
  loglikes.AddMatVec(1.0, means_invvars, kNoTrans, data, 1.0);
  // loglikes += -0.5 * inv(vars) * data_sq.
  loglikes.AddMatVec(-0.5, pdf.inv_vars(), kNoTrans, data_squared_, 1.0);

  BaseFloat log_sum = loglikes.LogSumExp(log_sum_exp_prune_);
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

  log_like_cache_[state].log_like = log_sum;
  log_like_cache_[state].hit_time = frame;

  return log_sum;
}

}  // namespace kaldi

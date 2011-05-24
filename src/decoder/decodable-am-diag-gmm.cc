// decoder/decodable-am-diag-gmm.cc

// Copyright 2009-2011  Arnab Ghoshal (Saarland University)  Lukas Burget

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

#include "decoder/decodable-am-diag-gmm.h"

namespace kaldi {

BaseFloat DecodableAmDiagGmmUnmapped::LogLikelihoodZeroBased(
    int32 frame, int32 state) {
  assert(static_cast<size_t>(frame) < static_cast<size_t>(NumFrames()));
  assert(static_cast<size_t>(state) < static_cast<size_t>(NumIndices()));

  if (log_like_cache_[state].hit_time == frame) {
    return log_like_cache_[state].log_like;  // return cached value, if found
  }

  if (frame != previous_frame_) {  // cache the squared stats.
    data_squared_.CopyFromVec(feature_matrix_.Row(frame));
    data_squared_.ApplyPow(2.0);
    previous_frame_ = frame;
  }

  const DiagGmm &pdf = acoustic_model_.GetPdf(state);
  const VectorBase<BaseFloat> &data = feature_matrix_.Row(frame);

  // check if everything is in order
  if (pdf.Dim() != data.Dim()) {
    KALDI_ERR << "Dim mismatch: data dim = "  << data.Dim()
        << "vs. model dim = " << pdf.Dim();
  }
  if (!pdf.valid_gconsts()) {
    KALDI_ERR << "State "  << (state)  << ": Must call ComputeGconsts() "
        "before computing likelihood.";
  }

  Vector<BaseFloat> loglikes(pdf.gconsts());  // need to recreate for each pdf
  // loglikes +=  means * inv(vars) * data.
  loglikes.AddMatVec(1.0, pdf.means_invvars(), kNoTrans, data, 1.0);
  // loglikes += -0.5 * inv(vars) * data_sq.
  loglikes.AddMatVec(-0.5, pdf.inv_vars(), kNoTrans, data_squared_, 1.0);

  BaseFloat log_sum = loglikes.LogSumExp();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

  log_like_cache_[state].log_like = log_sum;
  log_like_cache_[state].hit_time = frame;

  return log_sum;
}

void DecodableAmDiagGmmUnmapped::ResetLogLikeCache() {
  if (static_cast<int32>(log_like_cache_.size()) != acoustic_model_.NumPdfs()) {
    log_like_cache_.resize(acoustic_model_.NumPdfs());
  }
  vector<LikelihoodCacheRecord>::iterator it = log_like_cache_.begin(),
      end = log_like_cache_.end();
  for (; it != end; ++it) { it->hit_time = -1; }
}

BaseFloat DecodableAmDiagGmmRegtreeFmllr::LogLikelihoodZeroBased(int32 frame,
                                                          int32 state) {
  KALDI_ASSERT(frame < NumFrames() && frame >= 0);
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
        << "vs. model dim = " << pdf.Dim();
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

  BaseFloat log_sum = loglikes.LogSumExp();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

  log_like_cache_[state].log_like = log_sum;
  log_like_cache_[state].hit_time = frame;

  return log_sum;
}

}  // namespace kaldi

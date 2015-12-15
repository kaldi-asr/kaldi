// gmm/decodable-am-diag-gmm.cc

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


#include "gmm/decodable-am-diag-gmm-multi.h"

namespace kaldi {

BaseFloat DecodableAmDiagGmmUnmappedMulti::LogLikelihoodZeroBased(
    int32 frame, int32 state) {
  KALDI_ASSERT(static_cast<size_t>(frame) < static_cast<size_t>(NumFrames()));
  KALDI_ASSERT(static_cast<size_t>(state) < static_cast<size_t>(NumIndices()));

  if (log_like_cache_[state].hit_time == frame) {
    return log_like_cache_[state].log_like;  // return cached value, if found
  }

  if (frame != previous_frame_) {  // cache the squared stats.
    data_squared_.CopyFromVec(feature_matrix_.Row(frame));
    data_squared_.ApplyPow(2.0);
    previous_frame_ = frame;
  }

  vector<int32> states = mapping_[state];  // get the states of different trees

  vector<BaseFloat> log_sums;
  BaseFloat log_sum = 0.0;

  for (size_t i = 0; i < acoustic_models_.size(); i++) {
    const DiagGmm &pdf = acoustic_models_[i].GetPdf(states[i]);
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

    Vector<BaseFloat> loglikes(pdf.gconsts());  // need to recreate for each pdf
    // loglikes +=  means * inv(vars) * data.
    loglikes.AddMatVec(1.0, pdf.means_invvars(), kNoTrans, data, 1.0);
    // loglikes += -0.5 * inv(vars) * data_sq.
    loglikes.AddMatVec(-0.5, pdf.inv_vars(), kNoTrans, data_squared_, 1.0);

    log_sum = loglikes.LogSumExp(log_sum_exp_prune_);
    log_sums.push_back(log_sum);

    if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
      KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  }

  // seperate this loop out s.t.
  // we could change to other means/operations
  double weighted_sum = 0;
  double weight = 0;

  // change to a weighted sum, hope it works
  for (size_t i = 0; i < acoustic_models_.size(); i++) {
    weighted_sum += log_sums[i] * exp(log_sums[i] * exp_weight_);
    weight += exp(log_sums[i] * exp_weight_);
  }
  weighted_sum /= weight;

  log_like_cache_[state].log_like = weighted_sum;

  log_like_cache_[state].hit_time = frame;
  return weighted_sum;
}

void DecodableAmDiagGmmUnmappedMulti::ResetLogLikeCache() {
  if (static_cast<int32>(log_like_cache_.size()) != mapping_.size()) {
    log_like_cache_.resize(mapping_.size());
  }
  vector<LikelihoodCacheRecord>::iterator it = log_like_cache_.begin(),
      end = log_like_cache_.end();
  for (; it != end; ++it) { it->hit_time = -1; }
}

} // namespace kaldi

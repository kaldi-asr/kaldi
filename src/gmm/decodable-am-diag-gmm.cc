// gmm/decodable-am-diag-gmm.cc

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

#include "gmm/decodable-am-diag-gmm.h"

namespace kaldi {

BaseFloat DecodableAmDiagGmmUnmapped::LogLikelihoodZeroBased(
    int32 frame, int32 state) {
  KALDI_ASSERT(static_cast<size_t>(frame) <
               static_cast<size_t>(NumFramesReady()));
  KALDI_ASSERT(static_cast<size_t>(state) < static_cast<size_t>(NumIndices()) &&
               "Likely graph/model mismatch, e.g. using wrong HCLG.fst");

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

  BaseFloat log_sum = loglikes.LogSumExp(log_sum_exp_prune_);
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


}  // namespace kaldi

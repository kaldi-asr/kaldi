// sgmm/decodable-am-sgmm.cc

// Copyright 2009-2011  Saarland University;  Lukas Burget

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

#include "sgmm/decodable-am-sgmm.h"

namespace kaldi {

BaseFloat DecodableAmSgmm::LogLikelihoodZeroBased(int32 frame, int32 pdf_id) {
  KALDI_ASSERT(frame >= 0 && frame < NumFrames());
  KALDI_ASSERT(pdf_id >= 0 && pdf_id < NumIndices());

  if (log_like_cache_[pdf_id].hit_time == frame) {
    return log_like_cache_[pdf_id].log_like;  // return cached value, if found
  }

  const VectorBase<BaseFloat> &data = feature_matrix_.Row(frame);
  // check if everything is in order
  if (acoustic_model_.FeatureDim() != data.Dim()) {
    KALDI_ERR << "Dim mismatch: data dim = "  << data.Dim()
        << "vs. model dim = " << acoustic_model_.FeatureDim();
  }

  if (frame != previous_frame_) {  // Per-frame precomputation for SGMM.
    if (gselect_all_.empty())
      acoustic_model_.GaussianSelection(sgmm_config_, data, &gselect_);
    else {
      KALDI_ASSERT(frame < gselect_all_.size());
      gselect_ = gselect_all_[frame];
    }
    acoustic_model_.ComputePerFrameVars(data, gselect_, spk_,
                                        0.0 /*FMLLR logdet*/, &per_frame_vars_);
    previous_frame_ = frame;
  }

  BaseFloat loglike = acoustic_model_.LogLikelihood(per_frame_vars_, pdf_id,
                                                    log_prune_);
  if (KALDI_ISNAN(loglike) || KALDI_ISINF(loglike))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  log_like_cache_[pdf_id].log_like = loglike;
  log_like_cache_[pdf_id].hit_time = frame;
  return loglike;
}

void DecodableAmSgmm::ResetLogLikeCache() {
  if (log_like_cache_.size() != acoustic_model_.NumPdfs()) {
    log_like_cache_.resize(acoustic_model_.NumPdfs());
  }
  vector<LikelihoodCacheRecord>::iterator it = log_like_cache_.begin(),
      end = log_like_cache_.end();
  for (; it != end; ++it) { it->hit_time = -1; }
}

}  // namespace kaldi

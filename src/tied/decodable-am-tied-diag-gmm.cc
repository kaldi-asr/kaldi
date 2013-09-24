// tied/decodable-am-tied-diag-gmm.cc

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

#include <vector>
using std::vector;

#include "tied/decodable-am-tied-diag-gmm.h"

namespace kaldi {

BaseFloat DecodableAmTiedDiagGmm::LogLikelihoodZeroBased(int32 frame, 
                                                         int32 pdf_index) {
  KALDI_ASSERT(frame >= 0 && frame < NumFrames());
  KALDI_ASSERT(pdf_index >= 0 && pdf_index < NumIndices());

  if (log_like_cache_[pdf_index].hit_time == frame) {
    return log_like_cache_[pdf_index].log_like;
  }

  const VectorBase<BaseFloat> &data = feature_matrix_.Row(frame);
  // check if everything is in order
  if (acoustic_model_.Dim() != data.Dim()) {
    KALDI_ERR << "Dim mismatch: data dim = "  << data.Dim()
        << "vs. model dim = " << acoustic_model_.Dim();
  }

  if (frame != previous_frame_) {
    // different frame, prepare the per-frame-vars
    acoustic_model_.ComputePerFrameVars(data, &per_frame_vars_);
    previous_frame_ = frame;
  }

  BaseFloat loglike = acoustic_model_.LogLikelihood(pdf_index, 
                                                    &per_frame_vars_);
  
  if (KALDI_ISNAN(loglike) || KALDI_ISINF(loglike))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
    
  log_like_cache_[pdf_index].log_like = loglike;
  log_like_cache_[pdf_index].hit_time = frame;
  
  return loglike;
}

void DecodableAmTiedDiagGmm::ResetLogLikeCache() {
  if (log_like_cache_.size() != acoustic_model_.NumTiedPdfs()) {
    log_like_cache_.resize(acoustic_model_.NumTiedPdfs());
  }
  
  vector<LikelihoodCacheRecord>::iterator it = log_like_cache_.begin(),
      end = log_like_cache_.end();
      
  for (; it != end; ++it)
    it->hit_time = -1;
}

}  // namespace kaldi

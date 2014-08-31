// online/online-decodable.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#include "online/online-decodable.h"

namespace kaldi {

OnlineDecodableDiagGmmScaled::OnlineDecodableDiagGmmScaled(
    const AmDiagGmm &am, const TransitionModel &trans_model,
    const BaseFloat scale, OnlineFeatureMatrix *input_feats):  
      features_(input_feats), ac_model_(am),
      ac_scale_(scale), trans_model_(trans_model),
      feat_dim_(input_feats->Dim()), cur_frame_(-1) {
  if (!input_feats->IsValidFrame(0)) {
    // It's not safe to throw from a constructor, so please check
    // this condition yourself before reaching this point in the code.
    KALDI_ERR << "Attempt to initialize decodable object with empty "
              << "input: please check this before the initializer!";
  }
  int32 num_pdfs = trans_model_.NumPdfs();
  cache_.resize(num_pdfs, std::pair<int32,BaseFloat>(-1, 0.0));
}

void OnlineDecodableDiagGmmScaled::CacheFrame(int32 frame) {
  KALDI_ASSERT(frame >= 0);
  cur_feats_.Resize(feat_dim_);
  if (!features_->IsValidFrame(frame))
    KALDI_ERR << "Request for invalid frame (you need to check IsLastFrame, or, "
              << "for frame zero, check that the input is valid.";
  cur_feats_.CopyFromVec(features_->GetFrame(frame));
  cur_frame_ = frame;
}

BaseFloat OnlineDecodableDiagGmmScaled::LogLikelihood(int32 frame, int32 index) {
  if (frame != cur_frame_)
    CacheFrame(frame);
  int32 pdf_id = trans_model_.TransitionIdToPdf(index);
  if (cache_[pdf_id].first == frame)
    return cache_[pdf_id].second;
  BaseFloat ans = ac_model_.LogLikelihood(pdf_id, cur_feats_) * ac_scale_;
  cache_[pdf_id].first = frame;
  cache_[pdf_id].second = ans;
  return ans;
}


bool OnlineDecodableDiagGmmScaled::IsLastFrame(int32 frame) const {
  return !features_->IsValidFrame(frame+1);
}

} // namespace kaldi

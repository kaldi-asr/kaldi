// online2/online-gmm-decodable.cc

// Copyright 2012  Cisco Systems (author: Matthias Paulik)
//           2013  Vassil Panayotov
//           2014  Johns Hopkins University (author: Daniel Povey)

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

#include "online2/online-gmm-decodable.h"

namespace kaldi {

DecodableDiagGmmScaledOnline::DecodableDiagGmmScaledOnline(
    const AmDiagGmm &am, const TransitionModel &trans_model,
    const BaseFloat scale, OnlineFeatureInterface *input_feats):  
      features_(input_feats), ac_model_(am),
      ac_scale_(scale), trans_model_(trans_model),
      feat_dim_(input_feats->Dim()), cur_feats_(feat_dim_),
      cur_frame_(-1) {
  int32 num_pdfs = trans_model_.NumPdfs();
  cache_.resize(num_pdfs, std::pair<int32,BaseFloat>(-1, 0.0f));
}

void DecodableDiagGmmScaledOnline::CacheFrame(int32 frame) {
  // The call below will fail if "frame" is an invalid index, i.e. <0
  // or >= features_->NumFramesReady(), so there
  // is no need to check again.
  features_->GetFrame(frame, &cur_feats_);
  cur_frame_ = frame;
}

BaseFloat DecodableDiagGmmScaledOnline::LogLikelihood(int32 frame, int32 index) {
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


bool DecodableDiagGmmScaledOnline::IsLastFrame(int32 frame) const {
  return features_->IsLastFrame(frame);
}

int32 DecodableDiagGmmScaledOnline::NumFramesReady() const {
  return features_->NumFramesReady();
}

} // namespace kaldi

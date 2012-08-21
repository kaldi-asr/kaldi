// online/online-decodable.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

BaseFloat
OnlineDecodableDiagGmmScaled::LogLikelihood(int32 frame, int32 index) {
  uint32 feat_size = feat_matrix_.NumRows();
  if (frame >= feat_offset_ + feat_size) {
    KALDI_ASSERT(!finished_ && "Out-of-bounds request");

    // Init the underlying Decodable with a new batch of features
    timeout_tmp_ = timeout_;
    feat_matrix_.Resize(batch_size_, feat_dim_, kUndefined);
    finished_ = !input_->Compute(&feat_matrix_,
                                 timeout_ > 0? &timeout_tmp_: 0);
    if (feat_matrix_.NumRows() == 0)
      throw std::runtime_error("Unexpected end-of-features");
    feat_offset_ += feat_size;
    if (decodable_) delete decodable_;
    decodable_ = new DecodableAmDiagGmmScaled(ac_model_, trans_model_,
                                              feat_matrix_, ac_scale_);
  }
  int32 req_frame = frame - feat_offset_;
  return decodable_->LogLikelihood(req_frame, index);
}


bool
OnlineDecodableDiagGmmScaled::IsLastFrame(int32 frame) {
  if (!finished_)
    return false;
  else
    return (frame >= (feat_offset_ + feat_matrix_.NumRows() - 1));
}

} // namespace kaldi

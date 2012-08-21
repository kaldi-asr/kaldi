// online/online-feat-extract.h

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

#ifndef KALDI_ONLINE_DECODABLE_H_
#define KALDI_ONLINE_DECODABLE_H_

#include "online-feat-input.h"
#include "decoder/decodable-am-diag-gmm.h"

namespace kaldi {

// A decodable, taking input from an OnlineFeatureInput object on-demand
class OnlineDecodableDiagGmmScaled : public DecodableInterface {
 public:
  OnlineDecodableDiagGmmScaled(OnlineFeatInputItf *feat_input,
                               const AmDiagGmm &am,
                               const TransitionModel &trans_model,
                               const BaseFloat scale,
                               const uint32 batch_size,
                               const uint32 feat_dim,
                               const int32 timeout)
      : input_(feat_input), ac_model_(am),
        ac_scale_(scale), trans_model_(trans_model),
        decodable_(0), batch_size_(batch_size), feat_dim_(feat_dim),
        feat_offset_(0), finished_(false), timeout_(timeout) {}

  /// Returns the log likelihood, which will be negated in the decoder.
  virtual BaseFloat LogLikelihood(int32 frame, int32 index);

  virtual bool IsLastFrame(int32 frame);

  /// Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

  virtual ~OnlineDecodableDiagGmmScaled() {
    if (decodable_ != 0)
      delete decodable_;
  }

 private:
  OnlineFeatInputItf *input_;
  const AmDiagGmm &ac_model_;
  BaseFloat ac_scale_;
  const TransitionModel &trans_model_;
  DecodableAmDiagGmmScaled *decodable_;
  const uint32 batch_size_; // how many features to request/process in one go
  const uint32 feat_dim_; // dimensionality of the input features
  Matrix<BaseFloat> feat_matrix_; // the current batch of features
  uint32 feat_offset_; // the offset of the first frame in the current batch
  bool finished_; // is the input already exhausted?
  const int32 timeout_; // the value used when requesting new features
  uint32 timeout_tmp_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineDecodableDiagGmmScaled);
};

} // namespace kaldi

#endif // KALDI_ONLINE_DECODABLE_H_

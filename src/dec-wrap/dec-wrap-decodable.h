// dec-wrap/dec-wrap-decodable.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov,
//   Johns Hopkins University (author: Daniel Povey)
//   Ondrej Platek

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

#ifndef KALDI_DEC_WRAP_DEC_WRAP_DECODABLE_H_
#define KALDI_DEC_WRAP_DEC_WRAP_DECODABLE_H_

#include "dec-wrap/dec-wrap-feat-input.h"
#include "gmm/decodable-am-diag-gmm.h"

namespace kaldi {

// A decodable, taking input from an OnlFeatureInput object on-demand
class OnlDecodableDiagGmmScaled : public DecodableInterface {
 public:
  OnlDecodableDiagGmmScaled(const AmDiagGmm &am,
                               const TransitionModel &trans_model,
                               const BaseFloat scale,
                               OnlFeatureMatrix *input_feats);


  /// Returns the log likelihood, which will be negated in the decoder.
  virtual BaseFloat LogLikelihood(int32 frame, int32 index);

  virtual bool IsLastFrame(int32 frame);

  /// Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

  void Reset();

  BaseFloat GetAcousticScale() const { return ac_scale_; }

 private:
  void GetFrame(int32 frame);

  OnlFeatureMatrix *features_;
  const AmDiagGmm &ac_model_;
  BaseFloat ac_scale_;
  const TransitionModel &trans_model_;
  const int32 feat_dim_; // dimensionality of the input features
  Vector<BaseFloat> cur_feats_;
  int32 cur_frame_;
  std::vector<std::pair<int32, BaseFloat> > cache_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlDecodableDiagGmmScaled);
};

} // namespace kaldi

#endif // KALDI_DEC_WRAP_DEC_WRAP_DECODABLE_H_

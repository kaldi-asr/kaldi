// durmod/kaldi-durmod.h

// Copyright (c) 2015, Johns Hopkins University (Yenda Trmal<jtrmal@gmail.com>)
//                                               Hossein Hadian

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
#ifndef DURMOD_KALDI_DURMOD_H_
#define DURMOD_KALDI_DURMOD_H_

#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "base/kaldi-common.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-nnet.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"

namespace kaldi {
  using nnet3::Nnet;
  using nnet3::NnetExample;
  using nnet3::NnetIo;

struct PhoneDurationModelOptions {
  int left_ctx, right_ctx;

  void Register(OptionsItf *po);

  PhoneDurationModelOptions():
    left_ctx(4),
    right_ctx(2)
    { }
};

class PhoneDurationModel {
  friend class PhoneDurationEgsMaker;
 public:
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

  PhoneDurationModel() {}
  PhoneDurationModel(const PhoneDurationModelOptions &opts,
                     const std::vector<std::vector<int32> > &roots,
                     const std::vector<std::vector<int32> > &questions):
    left_context_(opts.left_ctx),
    right_context_(opts.right_ctx),
    roots_(roots),
    questions_(questions),
    max_duration_(30) {}

  void InitNnet(int input_dim, int dim1,
                int dim2, int output_dim);
  void SetMaxDuration(int32 max_duration_in_frames) {
    max_duration_ = max_duration_in_frames;
  }
  const Nnet &GetNnet() const { return nnet_; }
  Nnet &GetNnet() { return nnet_; }

 private:
  Nnet nnet_;

  // info related to generating features
  int32 left_context_, right_context_;
  std::vector<std::vector<int32> > roots_;
  std::vector<std::vector<int32> > questions_;
  int32 max_duration_;
};

class PhoneDurationEgsMaker {
 public:
  explicit PhoneDurationEgsMaker(const PhoneDurationModel &model);

  void InitFeatureMaker(const PhoneDurationModel &model);

  void AlignmentToNnetExamples(
      const std::vector<std::pair<int32, int32> > &alignment,
      std::vector<NnetExample> *egs) const;

  void MakeFeatureVector(
                    const std::vector<std::pair<int32, int32> > &phone_context,
                    int phone_index,
                    SparseVector<BaseFloat> *feat) const;

  int32 FeatureDim() { return feature_dim_; }
  int32 NumBinaryFeatures() { return num_binary_features_; }
  int32 NumPhoneIdentities() { return num_phone_identities_; }
  int32 OutputDim() { return max_duration_; }

 private:
  unordered_map<int32, std::vector<int32> > binary_feats_;
  unordered_map<int32, int32> phone_id_;
  int32 num_binary_features_;
  int32 num_phone_identities_;
  int32 left_ctx_, right_ctx_;
  int32 max_duration_;
  int32 feature_dim_;

  inline BaseFloat NormalizeDuration(int32 duration_in_frames) const {
    BaseFloat normalized_duration =
                            2.0 / (1.0 + Exp(-0.01f * duration_in_frames)) - 1;
    return normalized_duration;
  }
};

}  // namespace kaldi
#endif  // DURMOD_KALDI_DURMOD_H_


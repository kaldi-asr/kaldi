// durmod/kaldi-durmod.cc
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
#include <algorithm>
#include "durmod/kaldi-durmod.h"
namespace kaldi {
void PhoneDurationModelOptions::Register(OptionsItf *opts) {
  opts->Register("left-ctx", &left_ctx, "Number of left context frames");
  opts->Register("right-ctx", &right_ctx, "Number of right context frames");
}
PhoneDurationEgsMaker::PhoneDurationEgsMaker(const PhoneDurationModel &model) {
  InitFeatureMaker(model);
}
void PhoneDurationEgsMaker::MakeFeatureVector(
                    const std::vector<std::pair<int32, int32> > &phone_context,
                    int phone_index,
                    SparseVector<BaseFloat> *feat) const {
  std::vector<std::pair<MatrixIndexT, BaseFloat> > feat_elements;
  int feat_idx = 0;  // current index in the feat vector
  for (int i = (phone_index - left_ctx_);
           i <= (phone_index + right_ctx_); i++) {
    int phone_id;
    int phone_duration;
    std::vector<int32> binary_feats;
    if (i < 0 || i >= phone_context.size()) { // not available in the context
                                              // set all binary feat elements
                                              // and duration to zero
                                              // phone_id = 0
                                              // TODO(hhadian): all-zero elems
                                              // for phone_id?
      phone_id = 0;
      phone_duration = 0;
    } else {
      int32 phone = phone_context[i].first;
      phone_duration = phone_context[i].second;
      if (phone_id_.find(phone) == phone_id_.end()) {
        KALDI_ERR << "No phone identity found for phone "
                  << phone
                  << ". Check your roots.int";
      }
      phone_id = phone_id_.find(phone)->second;
      if (binary_feats_.find(phone) != binary_feats_.end())
        binary_feats = binary_feats_.find(phone)->second;
    }
    feat_elements.push_back(std::make_pair(feat_idx + phone_id, 1.0));
    feat_idx += num_phone_identities_;
    for (int j = 0; j < binary_feats.size(); j++) {
      feat_elements.push_back(std::make_pair(feat_idx + binary_feats[j], 1.0));
    }
    feat_idx += num_binary_features_;
    if (i < phone_index) {
      feat_elements.push_back(
                             std::make_pair(feat_idx++,
                                            NormalizeDuration(phone_duration)));
    }
  }
  SparseVector<BaseFloat> tmp(feature_dim_, feat_elements);
  feat->CopyFromSvec(tmp);
}
void PhoneDurationEgsMaker::AlignmentToNnetExamples(
                      const std::vector<std::pair<int32, int32> > &alignment,
                      std::vector<NnetExample> *egs) const {
// I commented these lines because we need edge examples too:
//  if (alignment.size() < (model.left_ctx + model.right_ctx + 1)) {
//    return;
//  }
  for (int i = 0; i < alignment.size(); i++) {
    SparseVector<BaseFloat> feat;
    MakeFeatureVector(alignment, i, &feat);
    int32 phone_duration = alignment[i].second;
    SparseMatrix<BaseFloat> feat_mat(1, feat.Dim());
    feat_mat.SetRow(0, feat);
//    Matrix<BaseFloat> out(1, 1);
//    out(0, 0) = NormalizeDuration(phone_duration);
    int32 output_dim = max_duration_;
    Posterior output_elements(1);
    int32 duration_id = (phone_duration > max_duration_) ?
                                                         (max_duration_ - 1):
                                                         (phone_duration - 1);
    output_elements[0].push_back(std::make_pair(duration_id, 1.0));
    SparseMatrix<BaseFloat> output_mat(output_dim, output_elements);
    NnetIo input, output;
    input.name = "input";
    input.features = feat_mat;
    input.indexes.resize(1);
    output.name = "output";
    output.features = output_mat;
    output.indexes.resize(1);
    NnetExample eg;
    eg.io.push_back(input);
    eg.io.push_back(output);
    egs->push_back(eg);
  }
}
void PhoneDurationEgsMaker::InitFeatureMaker(const PhoneDurationModel &model) {
  num_binary_features_ = model.questions_.size();
  num_phone_identities_ = model.roots_.size() + 1;  // id=0 is for not-available
                                                    // phones (at edges).
  left_ctx_ = model.left_context_;
  right_ctx_ = model.right_context_;
  max_duration_ = model.max_duration_;
  KALDI_LOG << "Max Duration: " << max_duration_;
  int input_dim_phones = num_phone_identities_ * (left_ctx_ + right_ctx_ + 1);
  int input_dim_durations = left_ctx_;
  int input_dim_binary = num_binary_features_ * (left_ctx_ + right_ctx_ + 1);
  feature_dim_ = input_dim_phones + input_dim_binary + input_dim_durations;
  // create the reverse map for questions/cluster membership
  for (int i = 0; i < model.questions_.size(); i++) {
    for (int j = 0; j < model.questions_[i].size(); j++) {
      int phone = model.questions_[i][j];
      binary_feats_[phone].push_back(i);
    }
  }
  // and the reverse map for phoneme tree roots
  int max_phone_idx = 0;  // TODO(hhadian): in the end, check if all phones have
                          // been categorized.
  for (int i = 0; i < model.roots_.size(); i++) {
    for (int j = 0; j < model.roots_[i].size(); j++) {
      int phone = model.roots_[i][j];
      KALDI_ASSERT(phone != 0);  // 0 has a specific meaning, it's
                                 // not allowed to be in roots.int
      KALDI_ASSERT(phone_id_[phone] == 0);  // only one root for each phone
      phone_id_[phone] = i + 1;  // phone identities start from 1
      if (max_phone_idx < phone)
        max_phone_idx = phone;
      if (binary_feats_.count(phone) <= 0) {
        KALDI_WARN << "Phone " << phone << "does not have any "
                   << "acoustic question associated.";
      }
    }
  }
}
void PhoneDurationModel::InitNnet(int input_dim, int dim1,
                                  int dim2, int output_dim) {
  std::stringstream config;

  // TODO(hhadian): to be later moved to scripts:

  KALDI_LOG << "DurModel.Nnet: in-dim: " << input_dim
            << ", dim1: " << dim1
            << ", dim2: " << dim2
            << ", out-dim: " << output_dim;
  config << "component name=affine1 type=NaturalGradientAffineComponent"
         << " learning-rate=0.003 param-stddev=0.017 bias-stddev=0"
         << " input-dim=" << input_dim << " output-dim=" << dim1 << std::endl;
  config << "component name=relu1 type=RectifiedLinearComponent"
         << " dim=" << dim1 << std::endl;
  config << "component name=norm1 type=NormalizeComponent"
         << " dim=" << dim1 << std::endl;
  config << "component name=affine2 type=NaturalGradientAffineComponent"
         << " learning-rate=0.003 param-stddev=0.017 bias-stddev=0"
         << " input-dim=" << dim1 << " output-dim=" << dim2 << std::endl;
  config << "component name=relu2 type=RectifiedLinearComponent"
         << " dim=" << dim2 << std::endl;
  config << "component name=norm2 type=NormalizeComponent"
         << " dim=" << dim2 << std::endl;
  config << "component name=affine3 type=NaturalGradientAffineComponent"
         << " learning-rate=0.003 param-stddev=0.017 bias-stddev=0"
         << " input-dim=" << dim2 << " output-dim=" << output_dim << std::endl;
  config << "component name=softmax type=LogSoftmaxComponent"
         << " dim=" << output_dim << std::endl;
  config << "input-node name=input dim=" << input_dim << std::endl;
  config << "component-node name=affine1_node component=affine1 input=input"
         << std::endl;
  config << "component-node name=relu1_node component=relu1 input=affine1_node"
         << std::endl;
  config << "component-node name=norm1_node component=norm1 input=relu1_node"
         << std::endl;
  config << "component-node name=affine2_node "
         << "component=affine2 input=norm1_node"
         << std::endl;
  config << "component-node name=relu2_node component=relu2 input=affine2_node"
         << std::endl;
  config << "component-node name=norm2_node component=norm2 input=relu2_node"
         << std::endl;
  config << "component-node name=affine3_node "
         << "component=affine3 input=norm2_node"
         << std::endl;
  config << "component-node name=softmax_node "
         << "component=softmax input=affine3_node"
         << std::endl;
  config << "output-node name=output input=softmax_node" << std::endl;
  nnet_.ReadConfig(config);
}
void PhoneDurationModel::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<PhoneDurationModel>");
  ReadBasicType(is, binary, &left_context_);
  ReadBasicType(is, binary, &right_context_);
  ReadBasicType(is, binary, &max_duration_);
  ExpectToken(is, binary, "<Roots>");
  int32 size;
  ReadBasicType(is, binary, &size);
  roots_.resize(size);
  for (int i = 0; i < roots_.size(); i++)
    ReadIntegerVector(is, binary, &(roots_[i]));
  ExpectToken(is, binary, "</Roots>");
  ExpectToken(is, binary, "<Questions>");
  ReadBasicType(is, binary, &size);
  questions_.resize(size);
  for (int i = 0; i < questions_.size(); i++)
    ReadIntegerVector(is, binary, &(questions_[i]));
  ExpectToken(is, binary, "</Questions>");
  nnet_.Read(is, binary);
  ExpectToken(is, binary, "</PhoneDurationModel>");
}
void PhoneDurationModel::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PhoneDurationModel>");
  WriteBasicType(os, binary, left_context_);
  WriteBasicType(os, binary, right_context_);
  WriteBasicType(os, binary, max_duration_);
  WriteToken(os, binary, "<Roots>");
  WriteBasicType(os, binary, roots_.size());
  for (int i = 0; i < roots_.size(); i++)
    WriteIntegerVector(os, binary, roots_[i]);
  WriteToken(os, binary, "</Roots>");
  WriteToken(os, binary, "<Questions>");
  WriteBasicType(os, binary, questions_.size());
  for (int i = 0; i < questions_.size(); i++)
    WriteIntegerVector(os, binary, questions_[i]);
  WriteToken(os, binary, "</Questions>");
  nnet_.Write(os, binary);
  WriteToken(os, binary, "</PhoneDurationModel>");
}
}  // namespace kaldi

// nnet3/nnet-component-itf.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-general-component.h"
#include "nnet3/nnet-convolutional-component.h"
#include "nnet3/nnet-attention-component.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-computation-graph.h"



// \file This file contains some more-generic component code: things in base classes.
//       See nnet-component.cc for the code of the actual Components.

namespace kaldi {
namespace nnet3 {

ComponentPrecomputedIndexes* ComponentPrecomputedIndexes::ReadNew(std::istream &is,
                                                                  bool binary) {
  std::string token;
  ReadToken(is, binary, &token); // e.g. "<DistributePrecomputedComponentIndexes>".
  token.erase(0, 1); // erase "<".
  token.erase(token.length()-1); // erase ">".
  ComponentPrecomputedIndexes *ans = NewComponentPrecomputedIndexesOfType(token);
  if (!ans)
   KALDI_ERR << "Unknown ComponentPrecomputedIndexes type " << token;
  ans->Read(is, binary);
  return ans;
}

ComponentPrecomputedIndexes* ComponentPrecomputedIndexes::NewComponentPrecomputedIndexesOfType(
                                           const std::string &cpi_type) {
  ComponentPrecomputedIndexes *ans = NULL;
  if (cpi_type == "DistributeComponentPrecomputedIndexes") {
    ans = new DistributeComponentPrecomputedIndexes();
  } else if (cpi_type == "StatisticsExtractionComponentPrecomputedIndexes") {
    ans = new StatisticsExtractionComponentPrecomputedIndexes();
  } else if (cpi_type == "StatisticsPoolingComponentPrecomputedIndexes") {
    ans = new StatisticsPoolingComponentPrecomputedIndexes();
  } else if (cpi_type == "BackpropTruncationComponentPrecomputedIndexes") {
    ans = new BackpropTruncationComponentPrecomputedIndexes();
  } else if (cpi_type == "TimeHeightConvolutionComponentPrecomputedIndexes") {
    ans = new TimeHeightConvolutionComponent::PrecomputedIndexes();
  } else if (cpi_type == "RestrictedAttentionComponentPrecomputedIndexes") {
    ans = new RestrictedAttentionComponent::PrecomputedIndexes();
  }
  if (ans != NULL) {
    KALDI_ASSERT(cpi_type == ans->Type());
  }
  return ans;
}

// static
Component* Component::ReadNew(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token); // e.g. "<SigmoidComponent>".
  token.erase(0, 1); // erase "<".
  token.erase(token.length()-1); // erase ">".
  Component *ans = NewComponentOfType(token);
  if (!ans)
    KALDI_ERR << "Unknown component type " << token;
  ans->Read(is, binary);
  return ans;
}


// static
Component* Component::NewComponentOfType(const std::string &component_type) {
  Component *ans = NULL;
  if (component_type == "SigmoidComponent") {
    ans = new SigmoidComponent();
  } else if (component_type == "TanhComponent") {
    ans = new TanhComponent();
  } else if (component_type == "SoftmaxComponent") {
    ans = new SoftmaxComponent();
  } else if (component_type == "LogSoftmaxComponent") {
    ans = new LogSoftmaxComponent();
  } else if (component_type == "RectifiedLinearComponent") {
    ans = new RectifiedLinearComponent();
  } else if (component_type == "NormalizeComponent") {
    ans = new NormalizeComponent();
  } else if (component_type == "PnormComponent") {
    ans = new PnormComponent();
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
  } else if (component_type == "LinearComponent") {
    ans = new LinearComponent();
  } else if (component_type == "NaturalGradientAffineComponent") {
    ans = new NaturalGradientAffineComponent();
  } else if (component_type == "PerElementScaleComponent") {
    ans = new PerElementScaleComponent();
  } else if (component_type == "NaturalGradientPerElementScaleComponent") {
    ans = new NaturalGradientPerElementScaleComponent();
  } else if (component_type == "PerElementOffsetComponent") {
    ans = new PerElementOffsetComponent();
  } else if (component_type == "SumGroupComponent") {
    ans = new SumGroupComponent();
  } else if (component_type == "FixedAffineComponent") {
    ans = new FixedAffineComponent();
  } else if (component_type == "FixedScaleComponent") {
    ans = new FixedScaleComponent();
  } else if (component_type == "FixedBiasComponent") {
    ans = new FixedBiasComponent();
  } else if (component_type == "NoOpComponent") {
    ans = new NoOpComponent();
  } else if (component_type == "ClipGradientComponent") {
    ans = new ClipGradientComponent();
  } else if (component_type == "ElementwiseProductComponent") {
    ans = new ElementwiseProductComponent();
  } else if (component_type == "ConvolutionComponent") {
    ans = new ConvolutionComponent();
  } else if (component_type == "MaxpoolingComponent") {
    ans = new MaxpoolingComponent();
  } else if (component_type == "PermuteComponent") {
    ans = new PermuteComponent();
  } else if (component_type == "DistributeComponent") {
    ans = new DistributeComponent();
  } else if (component_type == "CompositeComponent") {
    ans = new CompositeComponent();
  } else if (component_type == "RepeatedAffineComponent") {
    ans = new RepeatedAffineComponent();
  } else if (component_type == "BlockAffineComponent") {
    ans = new BlockAffineComponent();
  } else if (component_type == "NaturalGradientRepeatedAffineComponent") {
    ans = new NaturalGradientRepeatedAffineComponent();
  } else if (component_type == "StatisticsExtractionComponent") {
    ans = new StatisticsExtractionComponent();
  } else if (component_type == "StatisticsPoolingComponent") {
    ans = new StatisticsPoolingComponent();
  } else if (component_type == "ConstantFunctionComponent") {
    ans = new ConstantFunctionComponent();
  } else if (component_type == "ConstantComponent") {
    ans = new ConstantComponent();
  } else if (component_type == "DropoutComponent") {
    ans = new DropoutComponent();
  } else if (component_type == "DropoutMaskComponent") {
    ans = new DropoutMaskComponent();
  } else if (component_type == "BackpropTruncationComponent") {
    ans = new BackpropTruncationComponent();
  } else if (component_type == "LstmNonlinearityComponent") {
    ans = new LstmNonlinearityComponent();
  } else if (component_type == "BatchNormComponent") {
    ans = new BatchNormComponent();
  } else if (component_type == "TimeHeightConvolutionComponent") {
    ans = new TimeHeightConvolutionComponent();
  } else if (component_type == "RestrictedAttentionComponent") {
    ans = new RestrictedAttentionComponent();
  } else if (component_type == "SumBlockComponent") {
    ans = new SumBlockComponent();
  } else if (component_type == "ScaleAndOffsetComponent") {
    ans = new ScaleAndOffsetComponent();
  }
  if (ans != NULL) {
    KALDI_ASSERT(component_type == ans->Type());
  }
  return ans;
}

std::string Component::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim();
  return stream.str();
}

void Component::GetInputIndexes(const MiscComputationInfo &misc_info,
                                const Index &output_index,
                                std::vector<Index> *input_indexes) const {
  input_indexes->resize(1);
  (*input_indexes)[0] = output_index;
}

bool Component::IsComputable(const MiscComputationInfo &misc_info,
                             const Index &output_index,
                             const IndexSet &input_index_set,
                             std::vector<Index> *used_inputs) const {
  // the default Component dependency is for an output index to map directly to
  // the same input index, which is required to compute the output.
  if (!input_index_set(output_index))
    return false;
  if (used_inputs) {
    used_inputs->clear();
    used_inputs->push_back(output_index);
  }
  return true;
}


UpdatableComponent::UpdatableComponent(const UpdatableComponent &other):
    learning_rate_(other.learning_rate_),
    learning_rate_factor_(other.learning_rate_factor_),
    l2_regularize_(other.l2_regularize_),
    is_gradient_(other.is_gradient_),
    max_change_(other.max_change_) { }


void UpdatableComponent::SetUpdatableConfigs(
    const UpdatableComponent &other) {
  learning_rate_ = other.learning_rate_;
  learning_rate_factor_ = other.learning_rate_factor_;
  l2_regularize_ = other.l2_regularize_;
  is_gradient_ = other.is_gradient_;
  max_change_ = other.max_change_;
}

// If these defaults are changed, the defaults in the constructor that
// takes no arguments should be changed too.
void UpdatableComponent::InitLearningRatesFromConfig(ConfigLine *cfl) {
  learning_rate_ = 0.001;
  cfl->GetValue("learning-rate", &learning_rate_);
  learning_rate_factor_ = 1.0;
  cfl->GetValue("learning-rate-factor", &learning_rate_factor_);
  max_change_ = 0.0;
  cfl->GetValue("max-change", &max_change_);
  l2_regularize_ = 0.0;
  cfl->GetValue("l2-regularize", &l2_regularize_);
  if (learning_rate_ < 0.0 || learning_rate_factor_ < 0.0 ||
      max_change_ < 0.0 || l2_regularize_ < 0.0)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}


std::string UpdatableComponent::ReadUpdatableCommon(std::istream &is,
                                                    bool binary) {
  std::ostringstream opening_tag;
  opening_tag << '<' << this->Type() << '>';
  std::string token;
  ReadToken(is, binary, &token);
  if (token == opening_tag.str()) {
    // if the first token is the opening tag, then
    // ignore it and get the next tag.
    ReadToken(is, binary, &token);
  }
  if (token == "<LearningRateFactor>") {
    ReadBasicType(is, binary, &learning_rate_factor_);
    ReadToken(is, binary, &token);
  } else {
    learning_rate_factor_ = 1.0;
  }
  if (token == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ReadToken(is, binary, &token);
  } else {
    is_gradient_ = false;
  }
  if (token == "<MaxChange>") {
    ReadBasicType(is, binary, &max_change_);
    ReadToken(is, binary, &token);
  } else {
    max_change_ = 0.0;
  }
  if (token == "<L2Regularize>") {
    ReadBasicType(is, binary, &l2_regularize_);
    ReadToken(is, binary, &token);
  } else {
    l2_regularize_ = 0.0;
  }
  if (token == "<LearningRate>") {
    ReadBasicType(is, binary, &learning_rate_);
    return "";
  } else {
    return token;
  }
}

void UpdatableComponent::WriteUpdatableCommon(std::ostream &os,
                                              bool binary) const {
  std::ostringstream opening_tag;
  opening_tag << '<' << this->Type() << '>';
  std::string token;
  WriteToken(os, binary, opening_tag.str());
  if (learning_rate_factor_ != 1.0) {
    WriteToken(os, binary, "<LearningRateFactor>");
    WriteBasicType(os, binary, learning_rate_factor_);
  }
  if (is_gradient_) {
    WriteToken(os, binary, "<IsGradient>");
    WriteBasicType(os, binary, is_gradient_);
  }
  if (max_change_ > 0.0) {
    WriteToken(os, binary, "<MaxChange>");
    WriteBasicType(os, binary, max_change_);
  }
  if (l2_regularize_ > 0.0) {
    WriteToken(os, binary, "<L2Regularize>");
    WriteBasicType(os, binary, l2_regularize_);
  }
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
}


std::string UpdatableComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", learning-rate="
         << LearningRate();
  if (is_gradient_)
    stream << ", is-gradient=true";
  if (l2_regularize_ != 0.0)
    stream << ", l2-regularize=" << l2_regularize_;
  if (learning_rate_factor_ != 1.0)
    stream << ", learning-rate-factor=" << learning_rate_factor_;
  if (max_change_ > 0.0)
    stream << ", max-change=" << max_change_;
  return stream.str();
}

void NonlinearComponent::StoreStatsInternal(
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> *deriv) {
  KALDI_ASSERT(out_value.NumCols() == InputDim());

  // Check we have the correct dimensions.
  if (value_sum_.Dim() != InputDim() ||
      (deriv != NULL && deriv_sum_.Dim() != InputDim())) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (value_sum_.Dim() != InputDim()) {
      value_sum_.Resize(InputDim());
      count_ = 0.0;
    }
    if (deriv != NULL && deriv_sum_.Dim() != InputDim()) {
      deriv_sum_.Resize(InputDim());
      count_ = 0.0;
      value_sum_.SetZero();
    }
  }
  count_ += out_value.NumRows();
  CuVector<BaseFloat> temp(InputDim());
  temp.AddRowSumMat(1.0, out_value, 0.0);
  value_sum_.AddVec(1.0, temp);
  if (deriv != NULL) {
    temp.AddRowSumMat(1.0, *deriv, 0.0);
    deriv_sum_.AddVec(1.0, temp);
  }
}

void NonlinearComponent::ZeroStats() {
  value_sum_.SetZero();
  deriv_sum_.SetZero();
  count_ = 0.0;
  num_dims_self_repaired_ = 0.0;
  num_dims_processed_ = 0.0;
}

std::string NonlinearComponent::Info() const {
  std::stringstream stream;
  if (InputDim() == OutputDim()) {
    stream << Type() << ", dim=" << InputDim();
  } else {
    stream << Type() << ", input-dim=" << InputDim()
           << ", output-dim=" << OutputDim();
  }
  if (block_dim_ != dim_)
    stream << ", block-dim=" << block_dim_;
  if (self_repair_lower_threshold_ != BaseFloat(kUnsetThreshold))
    stream << ", self-repair-lower-threshold=" << self_repair_lower_threshold_;
  if (self_repair_upper_threshold_ != BaseFloat(kUnsetThreshold))
    stream << ", self-repair-upper-threshold=" << self_repair_upper_threshold_;
  if (self_repair_scale_ != 0.0)
    stream << ", self-repair-scale=" << self_repair_scale_;
  if (count_ > 0 && value_sum_.Dim() == dim_) {
    stream << ", count=" << std::setprecision(3) << count_
           << std::setprecision(6);
    stream << ", self-repaired-proportion="
           << (num_dims_processed_ > 0 ?
               num_dims_self_repaired_ / num_dims_processed_ : 0);
    Vector<double> value_avg_dbl(value_sum_);
    Vector<BaseFloat> value_avg(value_avg_dbl);
    value_avg.Scale(1.0 / count_);
    stream << ", value-avg=" << SummarizeVector(value_avg);
    if (deriv_sum_.Dim() == dim_) {
      Vector<double> deriv_avg_dbl(deriv_sum_);
      Vector<BaseFloat> deriv_avg(deriv_avg_dbl);
      deriv_avg.Scale(1.0 / count_);
      stream << ", deriv-avg=" << SummarizeVector(deriv_avg);
    }
  }
  return stream.str();
}

void NonlinearComponent::Scale(BaseFloat scale) {
  value_sum_.Scale(scale);
  deriv_sum_.Scale(scale);
  count_ *= scale;
  num_dims_self_repaired_ *= scale;
  num_dims_processed_ *= scale;
}

void NonlinearComponent::Add(BaseFloat alpha, const Component &other_in) {
  const NonlinearComponent *other =
      dynamic_cast<const NonlinearComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  if (value_sum_.Dim() == 0 && other->value_sum_.Dim() != 0)
    value_sum_.Resize(other->value_sum_.Dim());
  if (deriv_sum_.Dim() == 0 && other->deriv_sum_.Dim() != 0)
    deriv_sum_.Resize(other->deriv_sum_.Dim());
  if (other->value_sum_.Dim() != 0)
    value_sum_.AddVec(alpha, other->value_sum_);
  if (other->deriv_sum_.Dim() != 0)
    deriv_sum_.AddVec(alpha, other->deriv_sum_);
  count_ += alpha * other->count_;
  num_dims_self_repaired_ += alpha * other->num_dims_self_repaired_;
  num_dims_processed_ += alpha * other->num_dims_processed_;
}

void NonlinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
  if (PeekToken(is, binary) == 'B') {
    ExpectToken(is, binary, "<BlockDim>");
    ReadBasicType(is, binary, &block_dim_);
  } else {
    block_dim_ = dim_;
  }
  ExpectToken(is, binary, "<ValueAvg>");
  value_sum_.Read(is, binary);
  ExpectToken(is, binary, "<DerivAvg>");
  deriv_sum_.Read(is, binary);
  ExpectToken(is, binary, "<Count>");
  ReadBasicType(is, binary, &count_);
  value_sum_.Scale(count_);
  deriv_sum_.Scale(count_);

  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<NumDimsSelfRepaired>") {
    ReadBasicType(is, binary, &num_dims_self_repaired_);
    ReadToken(is, binary, &token);
  }
  if (token == "<NumDimsProcessed>") {
    ReadBasicType(is, binary, &num_dims_processed_);
    ReadToken(is, binary, &token);
  }
  if (token == "<SelfRepairLowerThreshold>") {
    ReadBasicType(is, binary, &self_repair_lower_threshold_);
    ReadToken(is, binary, &token);
  }
  if (token == "<SelfRepairUpperThreshold>") {
    ReadBasicType(is, binary, &self_repair_upper_threshold_);
    ReadToken(is, binary, &token);
  }
  if (token == "<SelfRepairScale>") {
    ReadBasicType(is, binary, &self_repair_scale_);
    ReadToken(is, binary, &token);
  }
  if (token != ostr_end.str()) {
    KALDI_ERR << "Expected token " << ostr_end.str()
              << ", got " << token;
  }
}

void NonlinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  if (block_dim_ != dim_) {
    WriteToken(os, binary, "<BlockDim>");
    WriteBasicType(os, binary, block_dim_);
  }
  // Write the values and derivatives in a count-normalized way, for
  // greater readability in text form.
  WriteToken(os, binary, "<ValueAvg>");
  Vector<BaseFloat> temp(value_sum_);
  if (count_ != 0.0) temp.Scale(1.0 / count_);
  temp.Write(os, binary);
  WriteToken(os, binary, "<DerivAvg>");

  temp.Resize(deriv_sum_.Dim(), kUndefined);
  temp.CopyFromVec(deriv_sum_);
  if (count_ != 0.0) temp.Scale(1.0 / count_);
  temp.Write(os, binary);
  WriteToken(os, binary, "<Count>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "<NumDimsSelfRepaired>");
  WriteBasicType(os, binary, num_dims_self_repaired_);
  WriteToken(os, binary, "<NumDimsProcessed>");
  WriteBasicType(os, binary, num_dims_processed_);
  if (self_repair_lower_threshold_ != kUnsetThreshold) {
    WriteToken(os, binary, "<SelfRepairLowerThreshold>");
    WriteBasicType(os, binary, self_repair_lower_threshold_);
  }
  if (self_repair_upper_threshold_ != kUnsetThreshold) {
    WriteToken(os, binary, "<SelfRepairUpperThreshold>");
    WriteBasicType(os, binary, self_repair_upper_threshold_);
  }
  if (self_repair_scale_ != 0.0) {
    WriteToken(os, binary, "<SelfRepairScale>");
    WriteBasicType(os, binary, self_repair_scale_);
  }
  WriteToken(os, binary, ostr_end.str());
}

NonlinearComponent::NonlinearComponent():
    dim_(-1), block_dim_(-1), count_(0.0),
    num_dims_self_repaired_(0.0), num_dims_processed_(0.0),
    self_repair_lower_threshold_(kUnsetThreshold),
    self_repair_upper_threshold_(kUnsetThreshold),
    self_repair_scale_(0.0) { }

NonlinearComponent::NonlinearComponent(const NonlinearComponent &other):
    dim_(other.dim_), block_dim_(other.block_dim_),
    value_sum_(other.value_sum_), deriv_sum_(other.deriv_sum_),
    count_(other.count_),
    num_dims_self_repaired_(other.num_dims_self_repaired_),
    num_dims_processed_(other.num_dims_processed_),
    self_repair_lower_threshold_(other.self_repair_lower_threshold_),
    self_repair_upper_threshold_(other.self_repair_upper_threshold_),
    self_repair_scale_(other.self_repair_scale_) { }

void NonlinearComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = cfl->GetValue("dim", &dim_);
  block_dim_ = dim_;
  cfl->GetValue("block-dim", &block_dim_);
  cfl->GetValue("self-repair-lower-threshold", &self_repair_lower_threshold_);
  cfl->GetValue("self-repair-upper-threshold", &self_repair_upper_threshold_);
  cfl->GetValue("self-repair-scale", &self_repair_scale_);
  if (!ok || cfl->HasUnusedValues() || dim_ <= 0 ||
      block_dim_ <= 0 || dim_ % block_dim_ != 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
}



} // namespace nnet3
} // namespace kaldi

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
#include "rnnlm/rnnlm-component-itf.h"
#include "rnnlm/rnnlm-component.h"
#include "nnet3/nnet-general-component.h"
#include "rnnlm/nnet-parse.h"
#include "nnet3/nnet-computation-graph.h"

// \file This file contains some more-generic component code: things in base classes.
//       See nnet-component.cc for the code of the actual Components.

namespace kaldi {
//using namespace nnet3;
namespace rnnlm {

// static
LmComponent* LmComponent::ReadNew(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token); // e.g. "<SigmoidComponent>".
  token.erase(0, 1); // erase "<".
  token.erase(token.length()-1); // erase ">".
  LmComponent *ans = NewComponentOfType(token);
  if (!ans)
    KALDI_ERR << "Unknown component type " << token;
  ans->Read(is, binary);
  return ans;
}


// static
LmComponent* LmComponent::NewComponentOfType(const std::string &component_type) {
  LmComponent *ans = NULL;
  if (component_type == "LmSoftmaxComponent") {
    ans = new LmSoftmaxComponent();
  } else if (component_type == "LmLogSoftmaxComponent") {
    ans = new LmLogSoftmaxComponent();
  } else if (component_type == "LmAffineComponent") {
    ans = new LmAffineComponent();
//  } else if (component_type == "NaturalGradientAffineComponent") {
//    ans = new LmNaturalGradientAffineComponent();
  } else if (component_type == "LmFixedAffineComponent") {
    ans = new LmFixedAffineComponent();
  }
  if (ans != NULL) {
    KALDI_ASSERT(component_type == ans->Type());
  }
  return ans;
}

std::string LmComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim();
  return stream.str();
}

void LmComponent::GetInputIndexes(const MiscComputationInfo &misc_info,
                                const Index &output_index,
                                std::vector<Index> *input_indexes) const {
  input_indexes->resize(1);
  (*input_indexes)[0] = output_index;
}

bool LmComponent::IsComputable(const MiscComputationInfo &misc_info,
                             const Index &output_index,
                             const nnet3::IndexSet &input_index_set,
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


void LmUpdatableComponent::InitLearningRatesFromConfig(ConfigLine *cfl) {
  cfl->GetValue("learning-rate", &learning_rate_);
  cfl->GetValue("learning-rate-factor", &learning_rate_factor_);
  if (learning_rate_ < 0.0 || learning_rate_factor_ < 0.0)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}


void LmUpdatableComponent::ReadUpdatableCommon(std::istream &is, bool binary) {
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
  if (token == "<LearningRate>") {
    ReadBasicType(is, binary, &learning_rate_);
  } else {
    KALDI_ERR << "Expected token <LearningRate>, got "
              << token;
  }
}

void LmUpdatableComponent::WriteUpdatableCommon(std::ostream &os,
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
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
}


std::string LmUpdatableComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", learning-rate="
         << LearningRate();
  if (is_gradient_)
    stream << ", is-gradient=true";
  if (learning_rate_factor_ != 1.0)
    stream << ", learning-rate-factor=" << learning_rate_factor_;
  return stream.str();
}

void LmNonlinearComponent::StoreStatsInternal(
    const MatrixBase<BaseFloat> &out_value,
    const MatrixBase<BaseFloat> *deriv) {
  KALDI_ASSERT(out_value.NumCols() == InputDim());
  // Check we have the correct dimensions.
  if (value_sum_.Dim() != InputDim() ||
      (deriv != NULL && deriv_sum_.Dim() != InputDim())) {
    mutex_.Lock();
    if (value_sum_.Dim() != InputDim()) {
      value_sum_.Resize(InputDim());
      count_ = 0.0;
    }
    if (deriv != NULL && deriv_sum_.Dim() != InputDim()) {
      deriv_sum_.Resize(InputDim());
      count_ = 0.0;
      value_sum_.SetZero();
    }
    mutex_.Unlock();
  }
  count_ += out_value.NumRows();
  Vector<BaseFloat> temp(InputDim());
  temp.AddRowSumMat(1.0, out_value, 0.0);
  value_sum_.AddVec(1.0, temp);
  if (deriv != NULL) {
    temp.AddRowSumMat(1.0, *deriv, 0.0);
    deriv_sum_.AddVec(1.0, temp);
  }
}

void LmNonlinearComponent::ZeroStats() {
  value_sum_.SetZero();
  deriv_sum_.SetZero();
  count_ = 0.0;
}

std::string LmNonlinearComponent::Info() const {
  std::stringstream stream;
  if (InputDim() == OutputDim())
    stream << Type() << ", dim=" << InputDim();
  else
    stream << Type() << ", input-dim=" << InputDim()
           << ", output-dim=" << OutputDim()
           << ", add-log-stddev=true";

  if (self_repair_lower_threshold_ != BaseFloat(kUnsetThreshold))
    stream << ", self-repair-lower-threshold=" << self_repair_lower_threshold_;
  if (self_repair_upper_threshold_ != BaseFloat(kUnsetThreshold))
    stream << ", self-repair-upper-threshold=" << self_repair_upper_threshold_;
  if (self_repair_scale_ != 0.0)
    stream << ", self-repair-scale=" << self_repair_scale_;
  if (count_ > 0 && value_sum_.Dim() == dim_ &&  deriv_sum_.Dim() == dim_) {
    stream << ", count=" << std::setprecision(3) << count_
           << std::setprecision(6);
    Vector<double> value_avg_dbl(value_sum_);
    Vector<BaseFloat> value_avg(value_avg_dbl);
    value_avg.Scale(1.0 / count_);
    stream << ", value-avg=" << SummarizeVector(value_avg);
    Vector<double> deriv_avg_dbl(deriv_sum_);
    Vector<BaseFloat> deriv_avg(deriv_avg_dbl);
    deriv_avg.Scale(1.0 / count_);
    stream << ", deriv-avg=" << SummarizeVector(deriv_avg);
  }
  return stream.str();
}

void LmNonlinearComponent::Scale(BaseFloat scale) {
  value_sum_.Scale(scale);
  deriv_sum_.Scale(scale);
  count_ *= scale;
}

void LmNonlinearComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LmNonlinearComponent *other =
      dynamic_cast<const LmNonlinearComponent*>(&other_in);
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
}

void LmNonlinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
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

void LmNonlinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
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

LmNonlinearComponent::LmNonlinearComponent():
    dim_(-1), count_(0.0),
    self_repair_lower_threshold_(kUnsetThreshold),
    self_repair_upper_threshold_(kUnsetThreshold),
    self_repair_scale_(0.0) { }

LmNonlinearComponent::LmNonlinearComponent(const LmNonlinearComponent &other):
    dim_(other.dim_), value_sum_(other.value_sum_), deriv_sum_(other.deriv_sum_),
    count_(other.count_),
    self_repair_lower_threshold_(other.self_repair_lower_threshold_),
    self_repair_upper_threshold_(other.self_repair_upper_threshold_),
    self_repair_scale_(other.self_repair_scale_) { }

void LmNonlinearComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = cfl->GetValue("dim", &dim_);
  cfl->GetValue("self-repair-lower-threshold", &self_repair_lower_threshold_);
  cfl->GetValue("self-repair-upper-threshold", &self_repair_upper_threshold_);
  cfl->GetValue("self-repair-scale", &self_repair_scale_);
  if (!ok || cfl->HasUnusedValues() || dim_ <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
}



} // namespace nnet3
} // namespace kaldi

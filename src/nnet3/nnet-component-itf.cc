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
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-computation-graph.h"

// \file This file contains some more-generic component code: things in base classes.
//       See nnet-component.cc for the code of the actual Components.

namespace kaldi {
namespace nnet3 {

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
  } else if (component_type == "SumReduceComponent") {
    ans = new SumReduceComponent();
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
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


void UpdatableComponent::InitLearningRatesFromConfig(ConfigLine *cfl) {
  cfl->GetValue("learning-rate", &learning_rate_);
  cfl->GetValue("learning-rate-factor", &learning_rate_factor_);
  if (learning_rate_ < 0.0 || learning_rate_factor_ < 0.0)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}


void UpdatableComponent::ReadUpdatableCommon(std::istream &is, bool binary) {
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
  if (learning_rate_factor_ != 1.0)
    stream << ", learning-rate-factor=" << learning_rate_factor_;
  return stream.str();
}

void NonlinearComponent::StoreStatsInternal(
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> *deriv) {
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
}

std::string NonlinearComponent::Info() const {
  std::stringstream stream;
  if (InputDim() == OutputDim())
    stream << Type() << ", dim=" << InputDim();
  else
    stream << Type() << ", input-dim=" << InputDim()
           << ", output-dim=" << OutputDim()
           << ", add-log-stddev=true";
  
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

void NonlinearComponent::Scale(BaseFloat scale) {
  value_sum_.Scale(scale);
  deriv_sum_.Scale(scale);
  count_ *= scale;
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
}

void NonlinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
  std::string tok; // TODO: remove back-compatibility code.
  ReadToken(is, binary, &tok);
  if (tok == "<ValueSum>") {
    // this branch is for back compatibility.  TODO: delete it
    // after Dec 2015.
    value_sum_.Read(is, binary);
    ExpectToken(is, binary, "<DerivSum>");
    deriv_sum_.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    ReadBasicType(is, binary, &count_);
    ExpectToken(is, binary, ostr_end.str());
  } else {
    // The new format is more readable as we write values that are normalized by
    // the count.
    KALDI_ASSERT(tok == "<ValueAvg>");
    value_sum_.Read(is, binary);
    ExpectToken(is, binary, "<DerivAvg>");
    deriv_sum_.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    ReadBasicType(is, binary, &count_);
    value_sum_.Scale(count_);
    deriv_sum_.Scale(count_);
    ExpectToken(is, binary, ostr_end.str());
  }
}

void NonlinearComponent::Write(std::ostream &os, bool binary) const {
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
  WriteToken(os, binary, ostr_end.str());
}

NonlinearComponent::NonlinearComponent(const NonlinearComponent &other):
    dim_(other.dim_), value_sum_(other.value_sum_), deriv_sum_(other.deriv_sum_),
    count_(other.count_) { }

void NonlinearComponent::InitFromConfig(ConfigLine *cfl) {
  int32 dim;
  bool ok = cfl->GetValue("dim", &dim);
  if (!ok || cfl->HasUnusedValues() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim);
}

} // namespace nnet3
} // namespace kaldi

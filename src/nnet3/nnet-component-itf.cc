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
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
  } else if (component_type == "NaturalGradientAffineComponent") {
    ans = new NaturalGradientAffineComponent();
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
  } else if (component_type == "ElementwiseProductComponent") {
    ans = new ElementwiseProductComponent();
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



void UpdatableComponent::Init(BaseFloat lr, bool is_gradient) {
  learning_rate_ = lr;
  is_gradient_ = is_gradient;
}

std::string UpdatableComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", learning-rate="
         << LearningRate();
  if (is_gradient_)
    stream << ", is-gradient=true";
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

std::string NonlinearComponent::Info() const {
  std::stringstream stream;
  KALDI_ASSERT(InputDim() == OutputDim());  // always the case
  stream << Type() << ", dim=" << InputDim();
  return stream.str();
}

void NonlinearComponent::Scale(BaseFloat scale) {
  value_sum_.Scale(scale);
  deriv_sum_.Scale(scale);
  count_ *= scale;
}

void NonlinearComponent::Add(BaseFloat alpha, const NonlinearComponent &other) {
  if (value_sum_.Dim() == 0 && other.value_sum_.Dim() != 0)
    value_sum_.Resize(other.value_sum_.Dim());
  if (deriv_sum_.Dim() == 0 && other.deriv_sum_.Dim() != 0)
    deriv_sum_.Resize(other.deriv_sum_.Dim());
  if (other.value_sum_.Dim() != 0)
    value_sum_.AddVec(alpha, other.value_sum_);
  if (other.deriv_sum_.Dim() != 0)
    deriv_sum_.AddVec(alpha, other.deriv_sum_);
  count_ += alpha * other.count_;
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
    value_sum_.Read(is, binary);
    ExpectToken(is, binary, "<DerivSum>");
    deriv_sum_.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    ReadBasicType(is, binary, &count_);
    ExpectToken(is, binary, ostr_end.str());
  } else if (tok == "<Counts>") { // Back-compat code for SoftmaxComponent.
    value_sum_.Read(is, binary); // Set both value_sum_ and deriv_sum_ to the same value,
    // and count_ to its sum.
    count_ = value_sum_.Sum();
    ExpectToken(is, binary, ostr_end.str());
  } else {
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void NonlinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<ValueSum>");
  value_sum_.Write(os, binary);
  WriteToken(os, binary, "<DerivSum>");
  deriv_sum_.Write(os, binary);
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

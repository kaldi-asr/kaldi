// nnet2/nnet-component.cc

// Copyright 2011-2012  Karel Vesely
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
//	              2013  Xiaohui Zhang	

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

#include <sstream>
#include "nnet2/nnet-component.h"
#include "nnet2/nnet-precondition.h"
#include "nnet2/nnet-precondition-online.h"
#include "util/text-utils.h"
#include "util/kaldi-io.h"

namespace kaldi {
namespace nnet2 {

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
  } else if (component_type == "PowerComponent") {
    ans = new PowerComponent();
  } else if (component_type == "SoftmaxComponent") {
    ans = new SoftmaxComponent();
  } else if (component_type == "RectifiedLinearComponent") {
    ans = new RectifiedLinearComponent();
  } else if (component_type == "NormalizeComponent") {
    ans = new NormalizeComponent();
  } else if (component_type == "SoftHingeComponent") {
    ans = new SoftHingeComponent();
  } else if (component_type == "PnormComponent") {
    ans = new PnormComponent();
  } else if (component_type == "MaxoutComponent") {
    ans = new MaxoutComponent();
  } else if (component_type == "ScaleComponent") {
    ans = new ScaleComponent();
  } else if (component_type == "PowerExpandComponent") {
    ans = new PowerExpandComponent();
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
  } else if (component_type == "PiecewiseLinearComponent") {
    ans = new PiecewiseLinearComponent();
  } else if (component_type == "AffineComponentA") {
    ans = new AffineComponentA();
  } else if (component_type == "AffineComponentPreconditioned") {
    ans = new AffineComponentPreconditioned();
  } else if (component_type == "AffineComponentPreconditionedOnline") {
    ans = new AffineComponentPreconditionedOnline();
  } else if (component_type == "AffineComponentModified") {
    ans = new AffineComponentModified();
  } else if (component_type == "AffinePreconInputComponent") {
    ans = new AffinePreconInputComponent();
  } else if (component_type == "MixtureProbComponent") {
    ans = new MixtureProbComponent();
  } else if (component_type == "SumGroupComponent") {
    ans = new SumGroupComponent();
  } else if (component_type == "BlockAffineComponent") {
    ans = new BlockAffineComponent();
  } else if (component_type == "BlockAffineComponentPreconditioned") {
    ans = new BlockAffineComponentPreconditioned();
  } else if (component_type == "PermuteComponent") {
    ans = new PermuteComponent();
  } else if (component_type == "DctComponent") {
    ans = new DctComponent();
  } else if (component_type == "FixedLinearComponent") {
    ans = new FixedLinearComponent();
  } else if (component_type == "FixedAffineComponent") {
    ans = new FixedAffineComponent();
  } else if (component_type == "FixedScaleComponent") {
    ans = new FixedScaleComponent();
  } else if (component_type == "FixedBiasComponent") {
    ans = new FixedBiasComponent();
  } else if (component_type == "SpliceComponent") {
    ans = new SpliceComponent();
  } else if (component_type == "SpliceMaxComponent") {
    ans = new SpliceMaxComponent();
  } else if (component_type == "DropoutComponent") {
    ans = new DropoutComponent();
  } else if (component_type == "AdditiveNoiseComponent") {
    ans = new AdditiveNoiseComponent();
  }
  return ans;
}

// static
Component* Component::NewFromString(const std::string &initializer_line) {
  std::istringstream istr(initializer_line);
  std::string component_type; // e.g. "SigmoidComponent".
  istr >> component_type >> std::ws; 
  std::string rest_of_line;
  getline(istr, rest_of_line);
  Component *ans = NewComponentOfType(component_type);
  if (ans == NULL)
    KALDI_ERR << "Bad initializer line (no such type of Component): "
              << initializer_line;
  ans->InitFromString(rest_of_line);
  return ans;
}


// This is like ExpectToken but for two tokens, and it
// will either accept token1 and then token2, or just token2.
// This is useful in Read functions where the first token
// may already have been consumed.
static void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                                 const std::string &token1,
                                 const std::string &token2) {
  KALDI_ASSERT(token1 != token2);
  std::string temp;
  ReadToken(is, binary, &temp);
  if (temp == token1) {
    ExpectToken(is, binary, token2);
  } else {
    if (temp != token2) {
      KALDI_ERR << "Expecting token " << token1 << " or " << token2
                << " but got " << temp;
    }
  }
}


// static
bool ParseFromString(const std::string &name, std::string *string,
                     int32 *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!ConvertStringToInteger(split_string[i].substr(len), param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     bool *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      std::string b = split_string[i].substr(len);
      if (b.empty())
        KALDI_ERR << "Bad option " << split_string[i];
      if (b[0] == 'f' || b[0] == 'F') *param = false;
      else if (b[0] == 't' || b[0] == 'T') *param = true;
      else
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     BaseFloat *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!ConvertStringToReal(split_string[i].substr(len), param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;      
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     std::string *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      *param = split_string[i].substr(len);

      // Set "string" to all the pieces but the one we used.
      *string = "";
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;      
    }
  }
  return false;
}

bool ParseFromString(const std::string &name, std::string *string,
                     std::vector<int32> *param) {
  std::vector<std::string> split_string;
  SplitStringToVector(*string, " \t", true,
                      &split_string);
  std::string name_equals = name + "="; // the name and then the equals sign.
  size_t len = name_equals.length();
  
  for (size_t i = 0; i < split_string.size(); i++) {
    if (split_string[i].compare(0, len, name_equals) == 0) {
      if (!SplitStringToIntegers(split_string[i].substr(len), ":",
                                 false, param))
        KALDI_ERR << "Bad option " << split_string[i];
      *string = "";
      // Set "string" to all the pieces but the one we used.
      for (size_t j = 0; j < split_string.size(); j++) {
        if (j != i) {
          if (!string->empty()) *string += " ";
          *string += split_string[j];
        }
      }
      return true;
    }
  }
  return false;
}


Component *PermuteComponent::Copy() const {
  PermuteComponent *ans = new PermuteComponent();
  ans->reorder_ = reorder_;
  return ans;
}
void PermuteComponent::Init(const std::vector<int32> &reorder) {
  reorder_ = reorder;
  KALDI_ASSERT(!reorder.empty());
  std::vector<int32> indexes(reorder);
  std::sort(indexes.begin(), indexes.end());
  for (int32 i = 0; i < static_cast<int32>(indexes.size()); i++)
    KALDI_ASSERT(i == indexes[i] && "Not a permutation");
}


std::string Component::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim();
  return stream.str();
}

std::string UpdatableComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", learning-rate="
         << LearningRate();
  return stream.str();
}


void PowerExpandComponent::Init(int32 dim,
                                int32 max_power,
                                BaseFloat higher_power_scale) {
  input_dim_ = dim;
  max_power_ = max_power;
  higher_power_scale_ = higher_power_scale;
  KALDI_ASSERT(input_dim_ > 0 && max_power >= 1 && higher_power_scale > 0.0);
}

void PowerExpandComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim, max_power = 2;
  BaseFloat higher_power_scale = 1.0;
  ParseFromString("max-power", &args, &max_power); // Optional.
  ParseFromString("higher-power-scale", &args, &higher_power_scale); // Optional.
  // Accept either "dim" or "input-dim" to specify the input dim.
  // "input-dim" is the canonical one; "dim" simplifies the testing code.
  bool ok = (ParseFromString("dim", &args, &dim) ||
             ParseFromString("input-dim", &args, &dim));
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, max_power, higher_power_scale);
}


void PowerExpandComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols() * max_power_, kUndefined);
  for (int32 p = 1; p <= max_power_; p++) {
    CuSubMatrix<BaseFloat> out_part(*out, 0, in.NumRows(),
                                  in.NumCols() * (p - 1), in.NumCols());
    out_part.CopyFromMat(in);
    if (p != 1) {
      out_part.ApplyPow(p);
      if (higher_power_scale_ != 1.0)
        out_part.Scale(higher_power_scale_);
    }
  }
}

void PowerExpandComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                    const CuMatrixBase<BaseFloat> &,// out_value,
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kUndefined);
  CuMatrix<BaseFloat> temp(in_value.NumRows(), in_value.NumCols(), kUndefined);
  for (int32 p = 1; p <= max_power_; p++) {
    const CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, in_value.NumRows(),
                                              in_value.NumCols() * (p - 1),
                                              in_value.NumCols());
    if (p == 1) {
      in_deriv->CopyFromMat(out_deriv_part);
    } else {
      // in scalar terms: in_deriv += p * in_value^(p-1) * [out_deriv w.r.t. this power]
      temp.CopyFromMat(in_value);
      if (p > 2) temp.ApplyPow(p - 1);
      temp.MulElements(out_deriv_part);
      in_deriv->AddMat(p * higher_power_scale_, temp);
    }
  }
}

void PowerExpandComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PowerExpandComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<MaxPower>");
  ReadBasicType(is, binary, &max_power_);
  ExpectToken(is, binary, "<HigherPowerScale>");
  ReadBasicType(is, binary, &higher_power_scale_);
  ExpectToken(is, binary, "</PowerExpandComponent>");
}

void PowerExpandComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PowerExpandComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<MaxPower>");
  WriteBasicType(os, binary, max_power_);
  WriteToken(os, binary, "<HigherPowerScale>");
  WriteBasicType(os, binary, higher_power_scale_);
  WriteToken(os, binary, "</PowerExpandComponent>");
}

std::string PowerExpandComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", max-power=" << max_power_;
  return stream.str();
}

void NonlinearComponent::SetDim(int32 dim) {
  KALDI_ASSERT(dim>0);
  dim_ = dim;
  value_sum_.Resize(dim);
  deriv_sum_.Resize(dim);
  count_ = 0.0;
}

void NonlinearComponent::UpdateStats(const CuMatrixBase<BaseFloat> &out_value,
                                     const CuMatrixBase<BaseFloat> *deriv) {
  KALDI_ASSERT(out_value.NumCols() == InputDim());
  if (value_sum_.Dim() != InputDim()) {
    value_sum_.Resize(InputDim());
    if (deriv != NULL) deriv_sum_.Resize(InputDim());
    count_ = 0.0;
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

void NonlinearComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  bool ok = ParseFromString("dim", &args, &dim);
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim);
}

void MaxoutComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  if (input_dim_ == 0)
    input_dim_ = 10 * output_dim_; // default group size : 10
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0);
  KALDI_ASSERT(input_dim_ % output_dim_ == 0) 
}

void MaxoutComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim = 0;
  int32 output_dim = 0;
  bool ok = ParseFromString("output-dim", &args, &output_dim) &&
      ParseFromString("input-dim", &args, &input_dim);
  KALDI_LOG << output_dim << " " << input_dim << " " << ok;
  if (!ok || !args.empty() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(input_dim, output_dim);
}


void MaxoutComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                int32 num_chunks,
                                CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), output_dim_, kUndefined);
  int32 group_size = input_dim_ / output_dim_;
  for (MatrixIndexT j = 0; j < output_dim_; j++) {
    CuSubMatrix<BaseFloat> pool(out->ColRange(j, 1));
    pool.Set(-1e20);
    for (MatrixIndexT i = 0; i < group_size; i++)
      pool.Max(in.ColRange(j * group_size + i, 1));
  }
}

void MaxoutComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               int32, // num_chunks
                               Component *to_update, // to_update
                               CuMatrix<BaseFloat> *in_deriv) const {
  int32 group_size = input_dim_ / output_dim_;
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);
  for (MatrixIndexT j = 0; j < output_dim_; j++) {
    CuSubMatrix<BaseFloat> out_j(out_value.ColRange(j, 1));
    for (MatrixIndexT i = 0; i < group_size; i++) {
        CuSubMatrix<BaseFloat> in_i(in_value.ColRange(j * group_size + i, 1));
        CuSubMatrix<BaseFloat> in_deriv_i(in_deriv->ColRange(j * group_size + i, 1));
        CuMatrix<BaseFloat> out_deriv_j(out_deriv.ColRange(j, 1));

        // Only the pool-inputs with 'max-values' are used to back-propagate into,
        // the rest of derivatives is zeroed-out by a mask.
        CuMatrix<BaseFloat> mask;
        in_i.EqualElementMask(out_j, &mask);
        out_deriv_j.MulElements(mask);
        in_deriv_i.AddMat(1.0, out_deriv_j); 
    }
  }
}

void MaxoutComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MaxoutComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</MaxoutComponent>");
}

void MaxoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MaxoutComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</MaxoutComponent>");
}

std::string MaxoutComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim = " << input_dim_
         << ", output-dim = " << output_dim_;
  return stream.str();
}

void PnormComponent::Init(int32 input_dim, int32 output_dim, BaseFloat p)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  if (input_dim_ == 0)
    input_dim_ = 10 * output_dim_; // default group size : 10
  p_ = p;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0 && p_ >= 0);
  KALDI_ASSERT(input_dim_ % output_dim_ == 0) 
}

void PnormComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim = 0;
  int32 output_dim = 0;
  BaseFloat p = 2;
  bool ok = ParseFromString("output-dim", &args, &output_dim) &&
      ParseFromString("input-dim", &args, &input_dim);
  ParseFromString("p", &args, &p);
  if (!ok || !args.empty() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(input_dim, output_dim, p);
}


void PnormComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                               int32 num_chunks,
                               CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), output_dim_, kUndefined);
  out->GroupPnorm(in, p_);
}

void PnormComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              int32, // num_chunks
                              Component *to_update, // to_update
                              CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);
  in_deriv->GroupPnormDeriv(in_value, out_value, p_);
  in_deriv->MulRowsGroupMat(out_deriv); 
}

void PnormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PnormComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<P>");
  ReadBasicType(is, binary, &p_);
  ExpectToken(is, binary, "</PnormComponent>");
}

void PnormComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PnormComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<P>");
  WriteBasicType(os, binary, p_);
  WriteToken(os, binary, "</PnormComponent>");
}

std::string PnormComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim = " << input_dim_
         << ", output-dim = " << output_dim_
	 << ", p = " << p_;
  return stream.str();
}


const BaseFloat NormalizeComponent::kNormFloor = pow(2.0, -66);
// This component modifies the vector of activations by scaling it so that the
// root-mean-square equals 1.0.

void NormalizeComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  *out = in;
  CuVector<BaseFloat> in_norm(in.NumRows());
  in_norm.AddDiagMat2(1.0 / in.NumCols(),
                      in, kNoTrans, 0.0);
  in_norm.ApplyFloor(kNormFloor);
  in_norm.ApplyPow(-0.5);
  out->MulRowsVec(in_norm);
}

/*
  A note on the derivative of NormalizeComponent...
  let both row_in and row_out be vectors of dimension D.
  Let p = row_in^T row_in / D, and let
      f = 1 / sqrt(max(kNormFloor, p)), and we compute row_out as:
row_out = f row_in.
  Suppose we have a quantity deriv_out which is the derivative
  of the objective function w.r.t. row_out.  We want to compute
  deriv_in which is the derivative of the objective function w.r.t.
  row_in.  Let the objective function be F.  One term is obvious: we have
     deriv_in = f deriv_out + ....
  next we have to take into account the derivative that gets back-propagated
  through f.  Obviously, dF/df = deriv_out^T row_in.
  And df/dp = (p <= kNormFloor ? 0.0 : -0.5 p^{-1.5}) = (f == 1 / sqrt(kNormFloor) ? 0.0 : -0.5 f^3),
  and dp/d(row_in) = 2/D row_in. [it's vector_valued].
  So this term in dF/d(row_in) equals:
    dF/df df/dp dp/d(row_in)   =    2/D (f == 1 / sqrt(kNormFloor)  ? 0.0 : -0.5 f^3) (deriv_out^T row_in) row_in
  So
     deriv_in = f deriv_out + (f == 1.0 ? 0.0 : -f^3 / D) (deriv_out^T row_in) row_in

*/

void NormalizeComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  int32, // num_chunks
                                  Component *to_update,
                                  CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  
  CuVector<BaseFloat> in_norm(in_value.NumRows());
  in_norm.AddDiagMat2(1.0 / in_value.NumCols(),
                      in_value, kNoTrans, 0.0);
  in_norm.ApplyFloor(kNormFloor);
  in_norm.ApplyPow(-0.5);
  in_deriv->AddDiagVecMat(1.0, in_norm, out_deriv, kNoTrans, 0.0);
  in_norm.ReplaceValue(1.0 / sqrt(kNormFloor), 0.0);
  in_norm.ApplyPow(3.0);
  CuVector<BaseFloat> dot_products(in_deriv->NumRows());
  dot_products.AddDiagMatMat(1.0, out_deriv, kNoTrans, in_value, kTrans, 0.0);
  dot_products.MulElements(in_norm);
  
  in_deriv->AddDiagVecMat(-1.0 / in_value.NumCols(), dot_products, in_value, kNoTrans, 1.0);
}

void SigmoidComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  out->Sigmoid(in);
}

void SigmoidComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *to_update,
                                CuMatrix<BaseFloat> *in_deriv) const {
  // we ignore in_value and to_update.

  // The element by element equation would be:
  // in_deriv = out_deriv * out_value * (1.0 - out_value);
  // We can accomplish this via calls to the matrix library.

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->Set(1.0);
  in_deriv->AddMat(-1.0, out_value);
  // now in_deriv = 1.0 - out_value [element by element]
  in_deriv->MulElements(out_value);
  // now in_deriv = out_value * (1.0 - out_value) [element by element], i.e.
  // it contains the element-by-element derivative of the nonlinearity.
  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
  // now in_deriv = out_deriv * out_value * (1.0 - out_value) [element by element]
}


void TanhComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  // Apply tanh function to each element of the output...
  // the tanh function may be written as -1 + ( 2 / (1 + e^{-2 x})),
  // which is a scaled and shifted sigmoid.
  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->Tanh(in);
}

void TanhComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             int32, // num_chunks
                             Component *to_update,
                             CuMatrix<BaseFloat> *in_deriv) const {
  /*
    Note on the derivative of the tanh function:
    tanh'(x) = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)

    The element by element equation of what we're doing would be:
    in_deriv = out_deriv * (1.0 - out_value^2).
    We can accomplish this via calls to the matrix library. */

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->CopyFromMat(out_value);
  in_deriv->ApplyPow(2.0);
  in_deriv->Scale(-1.0);
  in_deriv->Add(1.0);
  // now in_deriv = (1.0 - out_value^2), the element-by-element derivative of
  // the nonlinearity.
  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
}  

void PowerComponent::Init(int32 dim, BaseFloat power) {
  dim_ = dim;
  power_ = power;
  KALDI_ASSERT(dim > 0 && power >= 0);
}

void PowerComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat power = 2.0;
  ParseFromString("power", &args, &power); // Optional.
  // Accept either "dim" or "input-dim" to specify the input dim.
  // "input-dim" is the canonical one; "dim" simplifies the testing code.
  bool ok = (ParseFromString("dim", &args, &dim) ||
             ParseFromString("input-dim", &args, &dim));
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, power);
}

void PowerComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  // Apply power operation to each element of the input...
  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->CopyFromMat(in);
  out->ApplyPowAbs(power_);
}

void PowerComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             int32, // num_chunks
                             Component *to_update,
                             CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols());
  // in scalar terms: in_deriv += p * in_value^(p-1) * out_deriv
  in_deriv->CopyFromMat(in_value); 
  in_deriv->ApplyPowAbs(power_ - 1.0, true);
  in_deriv->Scale(power_);
  in_deriv->MulElements(out_deriv);
}

void PowerComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PowerComponent>", "<InputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<Power>");
  ReadBasicType(is, binary, &power_);
  ExpectToken(is, binary, "</PowerComponent>");
}

void PowerComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PowerComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Power>");
  WriteBasicType(os, binary, power_);
  WriteToken(os, binary, "</PowerComponent>");
}

std::string PowerComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", dim = " << dim_
	 << ", power = " << power_;
  return stream.str();
}

void RectifiedLinearComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              CuMatrix<BaseFloat> *out) const {
  // Apply rectified linear function (x >= 0 ? 1.0 : 0.0) 
  *out = in;
  out->ApplyFloor(0.0);
}

void RectifiedLinearComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                        const CuMatrixBase<BaseFloat> &out_value,
                                        const CuMatrixBase<BaseFloat> &out_deriv,
                                        int32, // num_chunks
                                        Component *to_update,
                                        CuMatrix<BaseFloat> *in_deriv) const {

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols(),
                   kUndefined);
  in_deriv->CopyFromMat(out_value);
  in_deriv->ApplyHeaviside();
  // Now in_deriv(i, j) equals (out_value(i, j) > 0.0 ? 1.0 : 0.0),
  // which is the derivative of the nonlinearity (well, except at zero
  // where it's undefined).
  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
}  

void SoftHingeComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                   int32, // num_chunks
                                   CuMatrix<BaseFloat> *out) const {
  // Apply function x = log(1 + exp(x))
  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->SoftHinge(in);
}

void SoftHingeComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  int32, // num_chunks
                                  Component *to_update,
                                  CuMatrix<BaseFloat> *in_deriv) const {

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols(),
                   kUndefined);
  // note: d/dx: log(1 + exp(x)) = (exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x)),
  // which is the sigmoid function.
  
  // if the output is y, then dy/dx =  (exp(x) / (1 + exp(x)),
  // and using y = log(1 + exp(x)) -> exp(x) = exp(y) - 1, we have
  // dy/dx = (exp(y) - 1) / exp(y)
  

  in_deriv->Sigmoid(in_value);

  if (to_update != NULL)
    dynamic_cast<NonlinearComponent*>(to_update)->UpdateStats(out_value,
                                                              in_deriv);
  in_deriv->MulElements(out_deriv);
}  


void ScaleComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                   int32, // num_chunks
                                   CuMatrix<BaseFloat> *out) const {
  *out = in;
  out->Scale(scale_);
}

void ScaleComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                              const CuMatrixBase<BaseFloat> &, // out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              int32, // num_chunks
                              Component *, // to_update
                              CuMatrix<BaseFloat> *in_deriv) const {

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols(),
                   kUndefined);
  in_deriv->CopyFromMat(out_deriv);
  in_deriv->Scale(scale_);
}  

void ScaleComponent::Init(int32 dim, BaseFloat scale) {
  dim_ = dim;
  scale_ = scale;
  KALDI_ASSERT(dim_ > 0);
  KALDI_ASSERT(scale_ != 0.0);
}

void ScaleComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat scale;
  if (!ParseFromString("dim", &args, &dim))
    KALDI_ERR << "Dimension not specified for ScaleComponent in config file";
  if (!ParseFromString("scale", &args, &scale))
    KALDI_ERR << "Scale not specified for ScaleComponent in config file";
  Init(dim, scale);
}

void ScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ScaleComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Scale>");
  WriteBasicType(os, binary, scale_);
  WriteToken(os, binary, "</ScaleComponent>");
}

void ScaleComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<ScaleComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<Scale>");
  ReadBasicType(is, binary, &scale_);
  ExpectToken(is, binary, "</ScaleComponent>");
}

std::string ScaleComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", dim=" << dim_ << ", scale=" << scale_;
  return stream.str();
}

void SoftmaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 CuMatrix<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).

  out->Resize(in.NumRows(), in.NumCols(), kUndefined);
  out->ApplySoftMaxPerRow(in);
  
  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);
}

void SoftmaxComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32 num_chunks,
                                Component *to_update, // only thing updated is counts_.
                                CuMatrix<BaseFloat> *in_deriv) const {
  /*
    Note on the derivative of the softmax function: let it be
    p_i = exp(x_i) / sum_i exp_i
    The [matrix-valued] Jacobian of this function is
    diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
    d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).    
  */
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());  
  KALDI_ASSERT(SameDim(out_value, out_deriv) && SameDim(out_value, *in_deriv));
  const CuMatrixBase<BaseFloat> &P(out_value), &E(out_deriv);
  CuMatrixBase<BaseFloat> &D (*in_deriv);


#if 1
  D.CopyFromMat(P);
  D.MulElements(E);
  // At this point, D = P .* E (in matlab notation)
  CuVector<BaseFloat> pe_vec(D.NumRows()); // For each row i, the dot product (p_t . e_t).
  pe_vec.AddDiagMatMat(1.0, P, kNoTrans, E, kTrans, 0.0);

  D.AddDiagVecMat(-1.0, pe_vec, P, kNoTrans, 1.0); // does D -= diag(pe_vec) * P.
#else  
  // The old code, where we did stuff row-by-row, is as follows;
  //   we had to rework it to use whole-matrix operations in order
  //   to use CUDA more effectively. 
  for (int32 r = 0; r < P.NumRows(); r++) {
    CuSubVector<BaseFloat> p(P, r), e(E, r), d(D, r);
    d.AddVecVec(1.0, p, e, 0.0); // d_i = p_i e_i.
    BaseFloat pT_e = VecVec(p, e); // p^T e.
    d.AddVec(-pT_e, p); // d_i -= (p^T e) p_i
  }
#endif
  
  // The SoftmaxComponent does not have any real trainable parameters, but
  // during the backprop we store some statistics on the average counts;
  // these may be used in mixing-up.
  if (to_update != NULL) {
    NonlinearComponent *to_update_nonlinear =
        dynamic_cast<NonlinearComponent*>(to_update);
    to_update_nonlinear->UpdateStats(out_value);
  }
}

void AffineComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void AffineComponent::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

AffineComponent::AffineComponent(const AffineComponent &component):
    UpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_),
    is_gradient_(component.is_gradient_) { }

AffineComponent::AffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 const CuVectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    UpdatableComponent(learning_rate),
    linear_params_(linear_params),
    bias_params_(bias_params) {
  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
  is_gradient_ = false;
}



void AffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
  if (treat_as_gradient)
    is_gradient_ = true;
}

void AffineComponent::SetParams(const VectorBase<BaseFloat> &bias,
                                const MatrixBase<BaseFloat> &linear) {
  bias_params_ = bias;
  linear_params_ = linear;
  KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
}

void AffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string AffineComponent::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_stddev =
      std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
                linear_params_size),
      bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                              bias_params_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", linear-params-stddev=" << linear_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate();
  return stream.str();
}

Component* AffineComponent::Copy() const {
  AffineComponent *ans = new AffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

BaseFloat AffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void AffineComponent::Init(BaseFloat learning_rate, 
                           int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

void AffineComponent::Init(BaseFloat learning_rate,
                           std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);  
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

void AffineComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  if (ParseFromString("matrix", &args, &matrix_filename)) {    
    Init(learning_rate, matrix_filename);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    Init(learning_rate, input_dim, output_dim,
         param_stddev, bias_stddev);    
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
}


void AffineComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                int32, // num_chunks
                                CuMatrix<BaseFloat> *out) const {
  // No need for asserts as they'll happen within the matrix operations.
  out->Resize(in.NumRows(), linear_params_.NumRows());
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

void AffineComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void AffineComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &,  // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               int32, //  num_chunks
                               Component *to_update_in,
                               CuMatrix<BaseFloat> *in_deriv) const {
  AffineComponent *to_update = dynamic_cast<AffineComponent*>(to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  // Propagate the derivative back to the input.
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                      0.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(in_value, out_deriv);  // by child classes.
  }
}

void AffineComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  // might not see the "<AffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  std::string tok;
  // back-compatibility code.  TODO: re-do this later.
  ReadToken(is, binary, &tok);
  if (tok == "<AvgInput>") { // discard the following.
    CuVector<BaseFloat> avg_input;
    avg_input.Read(is, binary);
    BaseFloat avg_input_count;
    ExpectToken(is, binary, "<AvgInputCount>");
    ReadBasicType(is, binary, &avg_input_count);
    ReadToken(is, binary, &tok);
  }
  if (tok == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ExpectToken(is, binary, ostr_end.str());
  } else {
    is_gradient_ = false;
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void AffineComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, ostr_end.str());
}

int32 AffineComponent::GetParameterDim() const {
  return (InputDim() + 1) * OutputDim();
}
void AffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
  params->Range(InputDim() * OutputDim(),
                OutputDim()).CopyFromVec(bias_params_);
}
void AffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  bias_params_.CopyFromVec(params.Range(InputDim() * OutputDim(),
                                        OutputDim()));
}

void AffineComponent::LimitRank(int32 d,
                                AffineComponent **a, AffineComponent **b) const {
  KALDI_ASSERT(d <= InputDim());

  // We'll limit the rank of just the linear part, keeping the bias vector full.
  Matrix<BaseFloat> M (linear_params_);
  int32 rows = M.NumRows(), cols = M.NumCols(), rc_min = std::min(rows, cols);
  Vector<BaseFloat> s(rc_min);
  Matrix<BaseFloat> U(rows, rc_min), Vt(rc_min, cols);
  // Do the destructive svd M = U diag(s) V^T.  It actually outputs the transpose of V.
  M.DestructiveSvd(&s, &U, &Vt);
  SortSvd(&s, &U, &Vt); // Sort the singular values from largest to smallest.
  BaseFloat old_svd_sum = s.Sum();
  U.Resize(rows, d, kCopyData);
  s.Resize(d, kCopyData);
  Vt.Resize(d, cols, kCopyData);
  BaseFloat new_svd_sum = s.Sum();
  KALDI_LOG << "Reduced rank from "
            << rc_min <<  " to " << d << ", SVD sum reduced from "
            << old_svd_sum << " to " << new_svd_sum;

  // U.MulColsVec(s); // U <-- U diag(s)
  Vt.MulRowsVec(s); // Vt <-- diag(s) Vt.

  *a = dynamic_cast<AffineComponent*>(this->Copy());
  *b = dynamic_cast<AffineComponent*>(this->Copy());
  
  (*a)->bias_params_.Resize(d, kSetZero);
  (*a)->linear_params_ = Vt;
  
  (*b)->bias_params_ = this->bias_params_;
  (*b)->linear_params_ = U;
}

Component *AffineComponent::CollapseWithNext(
    const AffineComponent &next_component) const {
  AffineComponent *ans = dynamic_cast<AffineComponent*>(this->Copy());
  KALDI_ASSERT(ans != NULL);
  // Note: it's possible that "ans" is really of a derived type such
  // as AffineComponentPreconditioned, but this will still work.
  // the "copy" call will copy things like learning rates, "alpha" value
  // for preconditioned component, etc.
  ans->linear_params_.Resize(next_component.OutputDim(), InputDim());
  ans->bias_params_ = next_component.bias_params_;

  ans->linear_params_.AddMatMat(1.0, next_component.linear_params_, kNoTrans,
                                this->linear_params_, kNoTrans, 0.0);
  ans->bias_params_.AddMatVec(1.0, next_component.linear_params_, kNoTrans,
                              this->bias_params_, 1.0);
  return ans;
}

Component *AffineComponent::CollapseWithNext(
    const FixedAffineComponent &next_component) const {
  // If at least one was non-updatable, make the whole non-updatable.
  FixedAffineComponent *ans =
      dynamic_cast<FixedAffineComponent*>(next_component.Copy());
  KALDI_ASSERT(ans != NULL);
  ans->linear_params_.Resize(next_component.OutputDim(), InputDim());
  ans->bias_params_ = next_component.bias_params_;

  ans->linear_params_.AddMatMat(1.0, next_component.linear_params_, kNoTrans,
                                this->linear_params_, kNoTrans, 0.0);
  ans->bias_params_.AddMatVec(1.0, next_component.linear_params_, kNoTrans,
                              this->bias_params_, 1.0);
  return ans;
}

Component *AffineComponent::CollapseWithPrevious(
    const FixedAffineComponent &prev_component) const {
  // If at least one was non-updatable, make the whole non-updatable.
  FixedAffineComponent *ans =
      dynamic_cast<FixedAffineComponent*>(prev_component.Copy());
  KALDI_ASSERT(ans != NULL);

  ans->linear_params_.Resize(this->OutputDim(), prev_component.InputDim());
  ans->bias_params_ = this->bias_params_;

  ans->linear_params_.AddMatMat(1.0, this->linear_params_, kNoTrans,
                                prev_component.linear_params_, kNoTrans, 0.0);
  ans->bias_params_.AddMatVec(1.0, this->linear_params_, kNoTrans,
                              prev_component.bias_params_, 1.0);
  return ans;
}


void PiecewiseLinearComponent::Scale(BaseFloat scale) {
  params_.Scale(scale);
}

void PiecewiseLinearComponent::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const PiecewiseLinearComponent *other =
      dynamic_cast<const PiecewiseLinearComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
}

PiecewiseLinearComponent::PiecewiseLinearComponent(const PiecewiseLinearComponent &component):
    UpdatableComponent(component),
    params_(component.params_),
    is_gradient_(component.is_gradient_),
    max_change_(component.max_change_) { }

void PiecewiseLinearComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    is_gradient_ = true;    
  }
  params_.SetZero();
}

void PiecewiseLinearComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_);
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}

std::string PiecewiseLinearComponent::Info() const {
  std::stringstream stream;
  BaseFloat params_size = static_cast<BaseFloat>(params_.NumRows())
      * static_cast<BaseFloat>(params_.NumCols());
  BaseFloat stddev =
      std::sqrt(TraceMatMat(params_, params_, kTrans) / params_size);
  CuVector<BaseFloat> per_dim_mean(params_.NumCols());
  CuVector<BaseFloat> per_dim_stddev(params_.NumCols());
  for (int32 dim = 0; dim < params_.NumCols(); dim++) {
    CuVector<BaseFloat> temp(params_.NumRows());
    temp.CopyColFromMat(params_, dim);
    BaseFloat mean = temp.Sum() / temp.Dim(),
        scatter = VecVec(temp, temp) / temp.Dim(),
        var = scatter - mean * mean,
        stddev = std::sqrt(var);
    per_dim_mean(dim) = mean;
    per_dim_stddev(dim) = stddev;
  }
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", N=" << (params_.NumCols() - 2)
         << ", global-params-stddev=" << stddev
         << ", params-mean=" << per_dim_mean
         << ", params-stddev=" << per_dim_stddev
         << ", learning-rate=" << LearningRate()
         << ", max-change=" << max_change_;
  return stream.str();
}

Component* PiecewiseLinearComponent::Copy() const {
  PiecewiseLinearComponent *ans = new PiecewiseLinearComponent();
  ans->learning_rate_ = learning_rate_;
  ans->params_ = params_;
  ans->is_gradient_ = is_gradient_;
  ans->max_change_ = max_change_;
  return ans;
}

BaseFloat PiecewiseLinearComponent::DotProduct(const UpdatableComponent &other_in) const {
  const PiecewiseLinearComponent *other =
      dynamic_cast<const PiecewiseLinearComponent*>(&other_in);
  return TraceMatMat(params_, other->params_, kTrans);
}

void PiecewiseLinearComponent::Init(int32 dim, int32 N,
                                    BaseFloat learning_rate,
                                    BaseFloat max_change) {
  UpdatableComponent::Init(learning_rate);
  params_.Resize(dim, N + 2); // will set them to zero.
  KALDI_ASSERT(N >= 3 && N % 2 == 1 &&
               "PiecewiseLinearComponent: must have N >= 3 and odd.");
  for (int32 i = 0; i < dim; i++) {
    // The "middle" gamma index has c_i = 0.  If we
    // initialize with all parameters zero except beta = 0.5 and the middle gamma
    // = 0.5, then we have f(x) = 0.5 x + 0.5 |x| = max(x, 0), which is the
    // same as the ReLU function.
    BaseFloat beta = 0.5, middle_gamma = 0.5;
    int32 middle_index = (N - 1) / 2;
    params_(i, 1) = beta;
    params_(i, middle_index + 2) = middle_gamma;
  }
  max_change_ = max_change;
}

void PiecewiseLinearComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_, max_change = 0.0;
  int32 dim = -1, N = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("max-change", &args, &max_change); // optional.
  ok = ok && ParseFromString("dim", &args, &dim);
  ok = ok && ParseFromString("N", &args, &N);

  Init(dim, N, learning_rate, max_change);
  
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
}


void PiecewiseLinearComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                         int32, // num_chunks
                                         CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), OutputDim());

  KALDI_ASSERT(in.NumCols() == InputDim());
  

  CuVector<BaseFloat> temp(InputDim());
  int32 dim = OutputDim(), num_frames = in.NumRows(), N = params_.NumCols() - 2;
  BaseFloat tick = 2.0 / (N - 1);
  // "tick" is the distance between the c_i.
  
  for (int32 t = 0; t < num_frames; t++) {
    for (int32 d = 0; d < dim; d++) {
      BaseFloat x = in(t, d);
      BaseFloat alpha = params_(d, 0), beta = params_(d, 1);
      BaseFloat y = alpha + x * beta;
      for (int32 n = 0; n < N; n++) {
        BaseFloat c_n = -1.0 + tick * n, gamma_n = params_(d, n + 2);
        y += gamma_n * std::abs(x - c_n);
      }
      (*out)(t, d) = y;
    }
  }
}


void PiecewiseLinearComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                        const CuMatrixBase<BaseFloat> &,  // out_value
                                        const CuMatrixBase<BaseFloat> &out_deriv,
                                        int32, //  num_chunks
                                        Component *to_update_in,
                                        CuMatrix<BaseFloat> *in_deriv) const {

  PiecewiseLinearComponent *to_update =
      dynamic_cast<PiecewiseLinearComponent*>(to_update_in);

  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == InputDim());
  
  in_deriv->Resize(in_value.NumRows(), InputDim());
  
  int32 dim = OutputDim(), num_frames = in_value.NumRows(),
      N = params_.NumCols() - 2;
  BaseFloat tick = 2.0 / (N - 1);
  // "tick" is the distance between the c_i.
  
  CuMatrix<BaseFloat> param_deriv(params_.NumRows(), params_.NumCols());
  
  for (int32 t = 0; t < num_frames; t++) {
    for (int32 d = 0; d < dim; d++) {
      BaseFloat x = in_value(t, d), oderiv = out_deriv(t, d), ideriv = 0.0;
      BaseFloat beta = params_(d, 1);
      // in forward: y = alpha + x * beta.
      ideriv += beta * oderiv; 
      param_deriv(d, 0) += 1.0 * oderiv;
      param_deriv(d, 1) += x * oderiv;
      
      for (int32 n = 0; n < N; n++) {
        BaseFloat c_n = -1.0 + tick * n, gamma_n = params_(d, n + 2);
        // in forward: y += gamma_n * std::abs(x - c_n);
        ideriv += oderiv * gamma_n * (x >= c_n ? 1.0 : -1.0);
        param_deriv(d, n + 2) += oderiv * std::abs(x - c_n);
      }
      (*in_deriv)(t, d) = ideriv;
    }
  }
  if (to_update != NULL) {
    if (to_update->is_gradient_ || to_update->max_change_ == 0.0) {
      to_update->params_.AddMat(to_update->learning_rate_, param_deriv);
    } else {
      param_deriv.Scale(to_update->learning_rate_);
      param_deriv.ApplyCeiling(to_update->max_change_);
      param_deriv.ApplyFloor(-to_update->max_change_);
      to_update->params_.AddMat(1.0, param_deriv);
    }
  }
}

void PiecewiseLinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<PiecewiseLinearComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</PiecewiseLinearComponent>"
  // might not see the "<PiecewiseLinearComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "<MaxChange>");
  ReadBasicType(is, binary, &max_change_);
  ExpectToken(is, binary, ostr_end.str());
}


void PiecewiseLinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<PiecewiseLinearComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</PiecewiseLinearComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<MaxChange>");
  WriteBasicType(os, binary, max_change_);
  WriteToken(os, binary, ostr_end.str());
}

int32 PiecewiseLinearComponent::GetParameterDim() const {
  return params_.NumRows() * params_.NumCols();
}

void PiecewiseLinearComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyRowsFromMat(params_);
}
void PiecewiseLinearComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  params_.CopyRowsFromVec(params);
}


void AffineComponentPreconditioned::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponentPreconditioned>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponentPreconditioned>"
  // might not see the "<AffineComponentPreconditioned>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  // todo: remove back-compat code.  Will just be:
  // ExpectToken(is, binary, "<MaxChange>");
  // ReadBasicType(is, binary, &max_change_);
  // ExpectToken(is, binary, ostr_end);
  // [end of function]
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<MaxChange>") {
    ReadBasicType(is, binary, &max_change_);
    ExpectToken(is, binary, ostr_end.str());
  } else {
    max_change_ = 0.0;
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void AffineComponentPreconditioned::InitFromString(std::string args) {
  std::string orig_args(args);
  std::string matrix_filename;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat alpha = 0.1, max_change = 0.0;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("alpha", &args, &alpha);
  ParseFromString("max-change", &args, &max_change);

  if (ParseFromString("matrix", &args, &matrix_filename)) {
    Init(learning_rate, alpha, max_change, matrix_filename);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    bool ok = true;
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    if (!ok)
      KALDI_ERR << "Bad initializer " << orig_args;
    Init(learning_rate, input_dim, output_dim, param_stddev,
         bias_stddev, alpha, max_change);
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
}

void AffineComponentPreconditioned::Init(BaseFloat learning_rate,
                                         BaseFloat alpha, BaseFloat max_change,
                                         std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  alpha_ = alpha;
  max_change_ = max_change;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

void AffineComponentPreconditioned::Init(
    BaseFloat learning_rate, 
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev,
    BaseFloat alpha, BaseFloat max_change) {
  UpdatableComponent::Init(learning_rate);
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  alpha_ = alpha;
  KALDI_ASSERT(alpha_ > 0.0);
  max_change_ = max_change; // Note: any value of max_change_is valid, but
  // only values > 0.0 will actually activate the code.
}


void AffineComponentPreconditioned::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<MaxChange>");
  WriteBasicType(os, binary, max_change_);
  WriteToken(os, binary, ostr_end.str());
}

std::string AffineComponentPreconditioned::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_stddev =
      std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
                linear_params_size),
      bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                              bias_params_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", linear-params-stddev=" << linear_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate()
         << ", alpha=" << alpha_
         << ", max-change=" << max_change_;
  return stream.str();
}

Component* AffineComponentPreconditioned::Copy() const {
  AffineComponentPreconditioned *ans = new AffineComponentPreconditioned();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->alpha_ = alpha_;
  ans->max_change_ = max_change_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}


BaseFloat AffineComponentPreconditioned::GetScalingFactor(
    const CuMatrix<BaseFloat> &in_value_precon,
    const CuMatrix<BaseFloat> &out_deriv_precon) {
  static int scaling_factor_printed = 0;

  KALDI_ASSERT(in_value_precon.NumRows() == out_deriv_precon.NumRows());
  CuVector<BaseFloat> in_norm(in_value_precon.NumRows()),
      out_deriv_norm(in_value_precon.NumRows());
  in_norm.AddDiagMat2(1.0, in_value_precon, kNoTrans, 0.0);
  out_deriv_norm.AddDiagMat2(1.0, out_deriv_precon, kNoTrans, 0.0);
  // Get the actual l2 norms, not the squared l2 norm.
  in_norm.ApplyPow(0.5);
  out_deriv_norm.ApplyPow(0.5);
  BaseFloat sum = learning_rate_ * VecVec(in_norm, out_deriv_norm);
  // sum is the product of norms that we are trying to limit
  // to max_value_.
  KALDI_ASSERT(sum == sum && sum - sum == 0.0 &&
               "NaN in backprop");
  KALDI_ASSERT(sum >= 0.0);
  if (sum <= max_change_) return 1.0;
  else {
    BaseFloat ans = max_change_ / sum;
    if (scaling_factor_printed < 10) {
      KALDI_LOG << "Limiting step size to " << max_change_
                << " using scaling factor " << ans << ", for component index "
                << Index();
      scaling_factor_printed++;
    }
    return ans;
  }
}

void AffineComponentPreconditioned::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  CuMatrix<BaseFloat> in_value_temp;
  
  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"  
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);
  
  CuMatrix<BaseFloat> in_value_precon(in_value_temp.NumRows(),
                                      in_value_temp.NumCols(), kUndefined),
      out_deriv_precon(out_deriv.NumRows(),
                       out_deriv.NumCols(), kUndefined);
  // each row of in_value_precon will be that same row of
  // in_value, but multiplied by the inverse of a Fisher
  // matrix that has been estimated from all the other rows,
  // smoothed by some appropriate amount times the identity
  // matrix (this amount is proportional to \alpha).
  PreconditionDirectionsAlphaRescaled(in_value_temp, alpha_, &in_value_precon);
  PreconditionDirectionsAlphaRescaled(out_deriv, alpha_, &out_deriv_precon);

  BaseFloat minibatch_scale = 1.0;

  if (max_change_ > 0.0)
    minibatch_scale = GetScalingFactor(in_value_precon, out_deriv_precon);
  
  
  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_precon,
                                            0, in_value_precon.NumRows(),
                                            0, in_value_precon.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_precon.NumRows());
  
  precon_ones.CopyColFromMat(in_value_precon, in_value_precon.NumCols() - 1);

  BaseFloat local_lrate = minibatch_scale * learning_rate_;
  bias_params_.AddMatVec(local_lrate, out_deriv_precon, kTrans,
                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_precon, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void AffineComponentPreconditionedOnline::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">";
  ostr_end << "</" << Type() << ">";
  // might not see the "<AffineComponentPreconditionedOnline>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<Rank>") {  // back-compatibility (temporary)
    ReadBasicType(is, binary, &rank_in_);
    rank_out_ = rank_in_;
  } else {
    KALDI_ASSERT(tok == "<RankIn>");
    ReadBasicType(is, binary, &rank_in_);
    ExpectToken(is, binary, "<RankOut>");
    ReadBasicType(is, binary, &rank_out_);    
  }
  ReadToken(is, binary, &tok);
  if (tok == "<UpdatePeriod>") {
    ReadBasicType(is, binary, &update_period_);
    ExpectToken(is, binary, "<NumSamplesHistory>");
  } else {
    update_period_ = 1;
    KALDI_ASSERT(tok == "<NumSamplesHistory>");
  }
  ReadBasicType(is, binary, &num_samples_history_);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<MaxChangePerSample>");
  ReadBasicType(is, binary, &max_change_per_sample_);
  ExpectToken(is, binary, ostr_end.str());
  SetPreconditionerConfigs();
}

void AffineComponentPreconditionedOnline::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  std::string matrix_filename;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_sample = 0.1;
  int32 input_dim = -1, output_dim = -1, rank_in = 30, rank_out = 80,
      update_period = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("num-samples-history", &args, &num_samples_history);
  ParseFromString("alpha", &args, &alpha);
  ParseFromString("max-change-per-sample", &args, &max_change_per_sample);
  ParseFromString("rank-in", &args, &rank_in);
  ParseFromString("rank-out", &args, &rank_out);
  ParseFromString("update-period", &args, &update_period);

  if (ParseFromString("matrix", &args, &matrix_filename)) {
    Init(learning_rate, rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample,
         matrix_filename);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    Init(learning_rate, input_dim, output_dim, param_stddev,
         bias_stddev, rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample);
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
}

void AffineComponentPreconditionedOnline::SetPreconditionerConfigs() {
  preconditioner_in_.SetRank(rank_in_);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_in_.SetAlpha(alpha_);
  preconditioner_in_.SetUpdatePeriod(update_period_);
  preconditioner_out_.SetRank(rank_out_);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_out_.SetAlpha(alpha_);
  preconditioner_out_.SetUpdatePeriod(update_period_);
}

void AffineComponentPreconditionedOnline::Init(
    BaseFloat learning_rate, int32 rank_in, int32 rank_out,
    int32 update_period, BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample,
    std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetPreconditionerConfigs();
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

AffineComponentPreconditionedOnline::AffineComponentPreconditionedOnline(
    const AffineComponent &orig,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha):
    max_change_per_sample_(0.1) {
  this->linear_params_ = orig.linear_params_;
  this->bias_params_ = orig.bias_params_;
  this->learning_rate_ = orig.learning_rate_;
  this->is_gradient_ = orig.is_gradient_;
  this->rank_in_ = rank_in;
  this->rank_out_ = rank_out;
  this->update_period_ = update_period;
  this->num_samples_history_ = num_samples_history;
  this->alpha_ = alpha;
  SetPreconditionerConfigs();
}

void AffineComponentPreconditionedOnline::Init(
    BaseFloat learning_rate, 
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample) {
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0 &&
               bias_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetPreconditionerConfigs();
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
}


void AffineComponentPreconditionedOnline::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, rank_in_);
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, rank_out_);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period_);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history_);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<MaxChangePerSample>");
  WriteBasicType(os, binary, max_change_per_sample_);
  WriteToken(os, binary, ostr_end.str());
}

std::string AffineComponentPreconditionedOnline::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_stddev =
      std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
                linear_params_size),
      bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                              bias_params_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", linear-params-stddev=" << linear_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate()
         << ", rank-in=" << rank_in_
         << ", rank-out=" << rank_out_
         << ", num_samples_history=" << num_samples_history_
         << ", update_period=" << update_period_
         << ", alpha=" << alpha_
         << ", max-change-per-sample=" << max_change_per_sample_;
  return stream.str();
}

Component* AffineComponentPreconditionedOnline::Copy() const {
  AffineComponentPreconditionedOnline *ans = new AffineComponentPreconditionedOnline();
  ans->learning_rate_ = learning_rate_;
  ans->rank_in_ = rank_in_;
  ans->rank_out_ = rank_out_;
  ans->update_period_ = update_period_;
  ans->num_samples_history_ = num_samples_history_;
  ans->alpha_ = alpha_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->preconditioner_in_ = preconditioner_in_;
  ans->preconditioner_out_ = preconditioner_out_;
  ans->max_change_per_sample_ = max_change_per_sample_;
  ans->is_gradient_ = is_gradient_;
  ans->SetPreconditionerConfigs();
  return ans;
}



BaseFloat AffineComponentPreconditionedOnline::GetScalingFactor(
    const CuVectorBase<BaseFloat> &in_products,
    BaseFloat learning_rate_scale,
    CuVectorBase<BaseFloat> *out_products) {
  static int scaling_factor_printed = 0;
  int32 minibatch_size = in_products.Dim();

  out_products->MulElements(in_products);
  out_products->ApplyPow(0.5);
  BaseFloat prod_sum = out_products->Sum();
  BaseFloat tot_objf_change = learning_rate_scale * learning_rate_ * prod_sum,
      max_objf_change = max_change_per_sample_ * minibatch_size;
  // tot_objf_change is the product of norms that we are trying to limit
  // to max_value_.
  KALDI_ASSERT(tot_objf_change - tot_objf_change == 0.0 && "NaN in backprop");
  KALDI_ASSERT(tot_objf_change >= 0.0);
  if (tot_objf_change <= max_objf_change) return 1.0;
  else {
    BaseFloat factor = max_objf_change / tot_objf_change;
    if (scaling_factor_printed < 10) {
      KALDI_LOG << "Limiting step size using scaling factor "
                << factor << ", for component index " << Index();
      scaling_factor_printed++;
    }
    return factor;
  }
}

void AffineComponentPreconditionedOnline::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  CuMatrix<BaseFloat> in_value_temp;
  
  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);
  
  CuMatrix<BaseFloat> out_deriv_temp(out_deriv);

  CuMatrix<BaseFloat> row_products(2,
                                   in_value.NumRows());
  CuSubVector<BaseFloat> in_row_products(row_products, 0),
      out_row_products(row_products, 1);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale, out_scale;
  
  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
                                            &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
                                             &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale;
  BaseFloat minibatch_scale = 1.0;
  
  if (max_change_per_sample_ > 0.0)
    minibatch_scale = GetScalingFactor(in_row_products, scale,
                                       &out_row_products);
  
  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, in_value_temp.NumRows(),
                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());
  
  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);
  
  BaseFloat local_lrate = scale * minibatch_scale * learning_rate_;
  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}


void AffineComponentModified::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  // might not see the "<AffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<CutoffLength>");
  ReadBasicType(is, binary, &cutoff_length_);
  ExpectToken(is, binary, "<MaxChange>");
  ReadBasicType(is, binary, &max_change_);
  ExpectToken(is, binary, ostr_end.str());
}


void AffineComponentModified::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  std::string matrix_filename;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat cutoff_length = 0.25, max_change = 0.1;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("cutoff-length", &args, &cutoff_length);
  ParseFromString("max-change", &args, &max_change);

  if (ParseFromString("matrix", &args, &matrix_filename)) {
    Init(learning_rate, cutoff_length, max_change, matrix_filename);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 0.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    Init(learning_rate, input_dim, output_dim, param_stddev,
         bias_stddev, cutoff_length, max_change);
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
}

void AffineComponentModified::Init(BaseFloat learning_rate, BaseFloat length_cutoff,
                                   BaseFloat max_change, std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  cutoff_length_ = cutoff_length_;
  max_change_ = max_change;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
}

void AffineComponentModified::Init(
    BaseFloat learning_rate, 
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev,
    BaseFloat cutoff_length, BaseFloat max_change) {
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  cutoff_length_ = cutoff_length;
  KALDI_ASSERT(max_change_ > 0.0);
  max_change_ = max_change; // Note: any value of max_change_is valid, but
  // only values > 0.0 will actually activate the code.
}


void AffineComponentModified::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<AffineComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</AffineComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<CutoffLength>");
  WriteBasicType(os, binary, cutoff_length_);
  WriteToken(os, binary, "<MaxChange>");
  WriteBasicType(os, binary, max_change_);
  WriteToken(os, binary, ostr_end.str());
}

std::string AffineComponentModified::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_stddev =
      std::sqrt(TraceMatMat(linear_params_, linear_params_, kTrans) /
                linear_params_size),
      bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                              bias_params_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", linear-params-stddev=" << linear_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate()
         << ", cutoff_length=" << cutoff_length_
         << ", max-change=" << max_change_;
  return stream.str();
}

Component* AffineComponentModified::Copy() const {
  AffineComponentModified *ans = new AffineComponentModified();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->cutoff_length_ = cutoff_length_;
  ans->max_change_ = max_change_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

void AffineComponentModified::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  int32 output_dim = OutputDim(), input_dim = InputDim();

  CuMatrix<BaseFloat> delta_params(output_dim, input_dim + 1);
  
  { // set delta_params to the change in parameters under a
    // straightforward gradient descent.
    CuSubMatrix<BaseFloat> linear_delta_params(delta_params,
                                             0, output_dim,
                                             0, input_dim);
    linear_delta_params.AddMatMat(learning_rate_, out_deriv, kTrans,
                                  in_value, kNoTrans, 0.0);
    
    CuVector<BaseFloat> bias_delta_params(output_dim);
    bias_delta_params.AddRowSumMat(learning_rate_, out_deriv);

    delta_params.CopyColFromVec(bias_delta_params, input_dim);
  }

  // diagnostics:
  int32 num_below_cutoff = 0, num_below_cutoff_limited = 0,
      num_limited = 0;
  
  CuVector<BaseFloat> param_row(input_dim + 1);
  for (int32 d = 0; d < output_dim; d++) {
    CuSubVector<BaseFloat> delta_param_row(delta_params, d);
    // Get the corresponding row of current parameters:
    param_row.Range(0, input_dim).CopyFromVec(linear_params_.Row(d));
    param_row(input_dim) = bias_params_(d);
 
    BaseFloat length = sqrt(VecVec(param_row, param_row)),
        delta_length = sqrt(VecVec(delta_param_row, delta_param_row)),
        dot_product = VecVec(param_row, delta_param_row);
    if (length < cutoff_length_) {
      // length is below cutoff -> do normal gradient descent, except to prevent
      // very large changes, limit delta to cutoff_length_ times max_change_.
      num_below_cutoff++;
      if (delta_length > cutoff_length_ * max_change_) {
        delta_param_row.Scale((cutoff_length_ * max_change_) / delta_length);
        num_below_cutoff_limited++;
      }
    } else {
      BaseFloat scale = 1.0; // We'll later scale delta_param_row by this much.
      // First enforce that the length of delta_param_row cannot exceed the current
      // length of the row times max_change_.
      if (delta_length > length * max_change_) {
        scale = (length * max_change_) / delta_length;
        delta_length *= scale;
        dot_product *= scale;
        num_limited++;
      }
      // OK, now rescale the (param_row + delta_param_row)
      // such that its length equals the original length of param_row plus
      // the component of delta_param_row in the direction of param_row.
      BaseFloat delta_length_in_direction = dot_product / length,
          delta_length_perpendicular_sq =
          delta_length * delta_length -
          delta_length_in_direction * delta_length_in_direction;
      KALDI_ASSERT(delta_length_perpendicular_sq >= 0.0);
      // delta_length_in_direction equals the (signed) length of the component
      // of delta_param_row in the same direction as "param_row",
      // delta_length_perpendicular_sq is the squared length of the component
      // perpendicular to that.
      BaseFloat new_length = length + delta_length_in_direction;
      // "new_length" is the length that we want the sum (param_row +
      // delta_param_row) to be, but we will need to rescale to ensure this.
      BaseFloat actual_length = sqrt(new_length * new_length +
                                     delta_length_perpendicular_sq);
      BaseFloat scaling_factor = new_length / actual_length;
      // We want to scale (param_row + delta_param_row) by "scaling_factor",
      // and express the result as an offset from param_row so we can add to it:
      // we want scaling_factor * (param_row + delta_param_row) - param_row
      // which equals param_row * (scaling_factor-1) + scaling_factor * delta_param_row.

      delta_param_row.Scale(scale * scaling_factor); // The "scale" comes from a previous
      // length-limiting operation, we delayed its application until now for efficiency.
      delta_param_row.AddVec(scaling_factor - 1.0, param_row);
    }
    // Now apply the change.
    linear_params_.Row(d).AddVec(1.0, delta_param_row.Range(0, input_dim));
    bias_params_(d) += delta_param_row(input_dim);
  }
  static int32 num_messages_printed = 0;
  if (num_messages_printed < 100) {
    KALDI_LOG << "Processed " << output_dim << " parameter rows, of which "
              << num_below_cutoff << " were below length cutoff (of which "
              << num_below_cutoff_limited << " were limited); of the rest, "
              << num_limited << " had their length limited.";
    num_messages_printed++;
  }
}



void AffinePreconInputComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void AffinePreconInputComponent::Backprop(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    int32, //  num_chunks
    Component *to_update_in,
    CuMatrix<BaseFloat> *in_deriv) const {
  AffinePreconInputComponent *to_update =
      dynamic_cast<AffinePreconInputComponent*>(to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  // Propagate the derivative back to the input.
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                      0.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    // add the sum of the rows of out_deriv, to the bias_params_.
    to_update->bias_params_.AddRowSumMat(to_update->learning_rate_, out_deriv,
                                         1.0);
    if (to_update->is_gradient_) { // simple update, getting gradient.
      to_update->linear_params_.AddMatMat(to_update->learning_rate_,
                                          out_deriv, kTrans,
                                          in_value, kNoTrans,
                                          1.0);
    } else {
      // more complex update, correcting for variance of input features.  Note:
      // most likely to_update == this, but we don't insist on this.
      CuMatrix<BaseFloat> in_value_tmp(in_value);
      in_value_tmp.MulColsVec(input_precision_); // Scale each column of in_value_tmp
      // (i.e. each dimension of the input features) by the corresponding element of
      // input_precision_.
    
      to_update->linear_params_.AddMatMat(to_update->learning_rate_,
                                          out_deriv, kTrans, in_value_tmp,
                                          kNoTrans, 1.0);
      // Next update input_precision_.  Note: we don't use any scaling on the
      // samples at this point.  This really won't matter in practice, it's just
      // for preconditioning.  Note: avg_samples_ is not very precisely a number
      // of samples to average over, just in an approximate dimensional sense; the
      // inverse of this is the constant in the exponential averaging.  Note: the
      // least we can actually average over is one minibatch; this is where the
      // std::max comes in below.
      int32 num_frames = in_value_tmp.NumRows();
      BaseFloat avg_samples_scaled =
          std::max(1.0, static_cast<double>(avg_samples_ / num_frames));
      BaseFloat cur_scale = 1.0 / avg_samples_scaled,
               prev_scale = 1.0 - cur_scale;
      CuVector<BaseFloat> &input_precision = to_update->input_precision_;
      input_precision.InvertElements();
      input_precision.AddDiagMat2(cur_scale, in_value, kTrans, prev_scale);
      if (input_precision.ApplyFloor(1.0e-10) > 0)
        KALDI_WARN << "Flooring elements of input feature variance.";
      input_precision.InvertElements();
    }
  }
}

void AffinePreconInputComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<AffinePreconInputComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<AvgSamples>");
  ReadBasicType(is, binary, &avg_samples_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<InputPrecision>");
  input_precision_.Read(is, binary);
  ExpectToken(is, binary, "</AffinePreconInputComponent>");  
}

void AffinePreconInputComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AffinePreconInputComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<AvgSamples>");
  WriteBasicType(os, binary, avg_samples_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<InputPrecision>");
  input_precision_.Write(os, binary);
  WriteToken(os, binary, "</AffinePreconInputComponent>");  
}

void AffinePreconInputComponent::Init(
    BaseFloat learning_rate,
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev,
    BaseFloat bias_stddev,
    BaseFloat avg_samples) {
  is_gradient_ = false;
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  avg_samples_ = avg_samples;
  KALDI_ASSERT(avg_samples_ > 1.0);
  input_precision_.Resize(input_dim);
  input_precision_.Set(1.0); // Set to all ones, as initially we
  // have no idea what the parameter variance is.
}

void AffinePreconInputComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_,
             avg_samples = 2000.0;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("avg-samples", &args, &avg_samples); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
      bias_stddev = 1.0;
  ParseFromString("param-stddev", &args, &param_stddev);
  ParseFromString("bias-stddev", &args, &bias_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, input_dim, output_dim,
       param_stddev, bias_stddev, avg_samples);
}

Component* AffinePreconInputComponent::Copy() const {
  AffinePreconInputComponent *ans = new AffinePreconInputComponent();
  ans->learning_rate_ = learning_rate_;
  ans->avg_samples_ = avg_samples_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->input_precision_ = input_precision_;
  return ans;
}

void BlockAffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void BlockAffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

BaseFloat BlockAffineComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const BlockAffineComponent *other =
      dynamic_cast<const BlockAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

Component* BlockAffineComponent::Copy() const {
  BlockAffineComponent *ans = new BlockAffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->num_blocks_ = num_blocks_;
  return ans;
}

void BlockAffineComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void BlockAffineComponent::Add(BaseFloat alpha,
                               const UpdatableComponent &other_in) {
  const BlockAffineComponent *other =
      dynamic_cast<const BlockAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void BlockAffineComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32, // num_chunks
                                     CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), bias_params_.Dim());

  // The matrix has a block structure where each matrix has input dim
  // (#rows) equal to input_block_dim.  The blocks are stored in linear_params_
  // as [ M
  //      N
  //      O ] but we actually treat it as:
  // [ M 0 0
  //   0 N 0
  //   0 0 O ]
  int32 input_block_dim = linear_params_.NumCols(),
       output_block_dim = linear_params_.NumRows() / num_blocks_,
             num_frames = in.NumRows();
  KALDI_ASSERT(in.NumCols() == input_block_dim * num_blocks_);
  KALDI_ASSERT(out->NumCols() == output_block_dim * num_blocks_);
  KALDI_ASSERT(in.NumRows() == out->NumRows());

  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  
  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_block(in, 0, num_frames,
                                  b * input_block_dim, input_block_dim),
        out_block(*out, 0, num_frames,
                  b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    out_block.AddMatMat(1.0, in_block, kNoTrans, param_block, kTrans, 1.0);
  }
}

void BlockAffineComponent::UpdateSimple(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  int32 input_block_dim = linear_params_.NumCols(),
      output_block_dim = linear_params_.NumRows() / num_blocks_,
      num_frames = in_value.NumRows();

  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        b * input_block_dim,
                                        input_block_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    // Update the parameters.
    param_block.AddMatMat(learning_rate_, out_deriv_block, kTrans,
                          in_value_block, kNoTrans, 1.0);
  }
}

void BlockAffineComponent::Backprop(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    int32, // num_chunks
    Component *to_update_in,
    CuMatrix<BaseFloat> *in_deriv) const {
  // This code mirrors the code in Propagate().
  int32 num_frames = in_value.NumRows();
  BlockAffineComponent *to_update = dynamic_cast<BlockAffineComponent*>(
      to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  int32 input_block_dim = linear_params_.NumCols(),
       output_block_dim = linear_params_.NumRows() / num_blocks_;
  KALDI_ASSERT(in_value.NumCols() == input_block_dim * num_blocks_);
  KALDI_ASSERT(out_deriv.NumCols() == output_block_dim * num_blocks_);

  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        b * input_block_dim,
                                        input_block_dim),
        in_deriv_block(*in_deriv, 0, num_frames,
                       b * input_block_dim, input_block_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);

    // Propagate the derivative back to the input.
    in_deriv_block.AddMatMat(1.0, out_deriv_block, kNoTrans,
                             param_block, kNoTrans, 0.0);
  }
  if (to_update != NULL)
    to_update->Update(in_value, out_deriv);
}


void BlockAffineComponent::Init(BaseFloat learning_rate,
                                int32 input_dim, int32 output_dim,
                                BaseFloat param_stddev,
                                BaseFloat bias_stddev,
                                int32 num_blocks) {
  UpdatableComponent::Init(learning_rate);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  KALDI_ASSERT(input_dim % num_blocks == 0 && output_dim % num_blocks == 0);

  linear_params_.Resize(output_dim, input_dim / num_blocks);
  bias_params_.Resize(output_dim);

  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  num_blocks_ = num_blocks;
}

void BlockAffineComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  int32 input_dim = -1, output_dim = -1, num_blocks = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  ok = ok && ParseFromString("num-blocks", &args, &num_blocks);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
      bias_stddev = 1.0;
  ParseFromString("param-stddev", &args, &param_stddev);
  ParseFromString("bias-stddev", &args, &bias_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, input_dim, output_dim,
       param_stddev, bias_stddev, num_blocks);
}
  

void BlockAffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BlockAffineComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<NumBlocks>");
  ReadBasicType(is, binary, &num_blocks_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</BlockAffineComponent>");  
}

void BlockAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BlockAffineComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<NumBlocks>");
  WriteBasicType(os, binary, num_blocks_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</BlockAffineComponent>");  
}


int32 BlockAffineComponent::GetParameterDim() const {
  // Note: num_blocks_ should divide both InputDim() and OutputDim().
  return InputDim() * OutputDim() / num_blocks_;
}

void BlockAffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  int32 l = linear_params_.NumRows() * linear_params_.NumCols(),
      b = bias_params_.Dim();
  params->Range(0, l).CopyRowsFromMat(linear_params_);
  params->Range(l, b).CopyFromVec(bias_params_);
}
void BlockAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  int32 l = linear_params_.NumRows() * linear_params_.NumCols(),
      b = bias_params_.Dim();
  linear_params_.CopyRowsFromVec(params.Range(0, l));
  bias_params_.CopyFromVec(params.Range(l, b));
}


void BlockAffineComponentPreconditioned::Init(BaseFloat learning_rate,
                                              int32 input_dim, int32 output_dim,
                                              BaseFloat param_stddev,
                                              BaseFloat bias_stddev,
                                              int32 num_blocks,
                                              BaseFloat alpha) {
  BlockAffineComponent::Init(learning_rate, input_dim, output_dim,
                             param_stddev, bias_stddev, num_blocks);
  is_gradient_ = false;
  KALDI_ASSERT(alpha > 0.0);
  alpha_ = alpha;
}

void BlockAffineComponentPreconditioned::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat alpha = 4.0;
  int32 input_dim = -1, output_dim = -1, num_blocks = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("alpha", &args, &alpha);
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  ok = ok && ParseFromString("num-blocks", &args, &num_blocks);
  
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
      bias_stddev = 1.0;
  ParseFromString("param-stddev", &args, &param_stddev);
  ParseFromString("bias-stddev", &args, &bias_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, input_dim, output_dim,
       param_stddev, bias_stddev, num_blocks,
       alpha);
}

void BlockAffineComponentPreconditioned::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient)
    is_gradient_ = true;
  BlockAffineComponent::SetZero(treat_as_gradient);
}  

void BlockAffineComponentPreconditioned::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BlockAffineComponentPreconditioned>",
                       "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<NumBlocks>");
  ReadBasicType(is, binary, &num_blocks_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</BlockAffineComponentPreconditioned>");  
}

void BlockAffineComponentPreconditioned::Write(std::ostream &os,
                                               bool binary) const {
  WriteToken(os, binary, "<BlockAffineComponentPreconditioned>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<NumBlocks>");
  WriteBasicType(os, binary, num_blocks_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</BlockAffineComponentPreconditioned>");  
}

Component* BlockAffineComponentPreconditioned::Copy() const {
  BlockAffineComponentPreconditioned *ans = new
      BlockAffineComponentPreconditioned();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->num_blocks_ = num_blocks_;
  ans->alpha_ = alpha_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

void BlockAffineComponentPreconditioned::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  if (is_gradient_) {
    UpdateSimple(in_value, out_deriv);
    // does the baseline update with no preconditioning.
    return;
  }
  int32 input_block_dim = linear_params_.NumCols(),
      output_block_dim = linear_params_.NumRows() / num_blocks_,
      num_frames = in_value.NumRows();

  CuMatrix<BaseFloat> in_value_temp(num_frames, input_block_dim + 1, kUndefined),
      in_value_precon(num_frames, input_block_dim + 1, kUndefined);
  in_value_temp.Set(1.0); // so last row will have value 1.0.
  CuSubMatrix<BaseFloat> in_value_temp_part(in_value_temp, 0, num_frames,
                                            0, input_block_dim); // all but last 1.0
  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_precon, 0, num_frames,
                                            0, input_block_dim);
  CuVector<BaseFloat> precon_ones(num_frames);
  CuMatrix<BaseFloat> out_deriv_precon(num_frames, output_block_dim, kUndefined);
  
  for (int32 b = 0; b < num_blocks_; b++) {
    CuSubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        b * input_block_dim,
                                        input_block_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    in_value_temp_part.CopyFromMat(in_value_block);

    PreconditionDirectionsAlphaRescaled(in_value_temp, alpha_,
                                        &in_value_precon);
    PreconditionDirectionsAlphaRescaled(out_deriv_block, alpha_,
                                        &out_deriv_precon);
    
    
    // Update the parameters.
    param_block.AddMatMat(learning_rate_, out_deriv_precon, kTrans,
                          in_value_precon_part, kNoTrans, 1.0);
    precon_ones.CopyColFromMat(in_value_precon, input_block_dim);
    bias_params_.Range(b * output_block_dim, output_block_dim).
        AddMatVec(learning_rate_, out_deriv_precon, kTrans,
                  precon_ones, 1.0);
  }
}


void PermuteComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PermuteComponent>", "<Reorder>");
  ReadIntegerVector(is, binary, &reorder_);
  ExpectToken(is, binary, "</PermuteComponent>");
}

void PermuteComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PermuteComponent>");
  WriteToken(os, binary, "<Reorder>");
  WriteIntegerVector(os, binary, reorder_);
  WriteToken(os, binary, "</PermuteComponent>");
}

void PermuteComponent::Init(int32 dim) {
  KALDI_ASSERT(dim > 0);
  reorder_.resize(dim);
  for (int32 i = 0; i < dim; i++) reorder_[i] = i;
  std::random_shuffle(reorder_.begin(), reorder_.end());
}

void PermuteComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  bool ok = ParseFromString("dim", &args, &dim);
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim);
}

void PermuteComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  std::vector<int32> reverse_reorder(reorder_.size());
  for (size_t i = 0; i < reorder_.size(); i++)
    reverse_reorder[reorder_[i]] = i;
  out->CopyCols(in, reverse_reorder);
}

void PermuteComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *to_update,
                                CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  KALDI_ASSERT(out_deriv.NumCols() == OutputDim());
  in_deriv->CopyCols(out_deriv, reorder_);
}

void MixtureProbComponent::Refresh() {
  KALDI_ASSERT(params_.size() == log_params_.size());
  for (size_t i = 0; i < params_.size(); i++) {
    // Make it so each column of params_ sums to one
    CuVector<BaseFloat> col(params_[i].NumRows());
    for (int32 c = 0; c < params_[i].NumCols(); c++) {
      col.CopyColFromMat(log_params_[i], c);
      col.ApplyExp();
      KALDI_ASSERT(col.Sum() > 0.0);
      col.Scale(1.0 / col.Sum()); // make it sum to one.
      params_[i].CopyColFromVec(col, c);
    }
  }
}

void MixtureProbComponent::PerturbParams(BaseFloat stddev) {
  for (size_t i = 0; i < log_params_.size(); i++) {
    CuMatrix<BaseFloat> &log_params(log_params_[i]);
    CuMatrix<BaseFloat> rand(log_params.NumRows(), log_params.NumCols());
    rand.SetRandn();
    log_params.AddMat(stddev, rand);
  }
  Refresh();
}


Component* MixtureProbComponent::Copy() const {
  MixtureProbComponent *ans = new MixtureProbComponent();
  ans->learning_rate_ = learning_rate_;
  ans->log_params_ = log_params_;
  ans->params_ = params_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  return ans;
}

BaseFloat MixtureProbComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const MixtureProbComponent *other =
      dynamic_cast<const MixtureProbComponent*>(&other_in);
  BaseFloat ans = 0.0;
  KALDI_ASSERT(log_params_.size() == other->log_params_.size());

  for (size_t i = 0; i < params_.size(); i++) {
    const CuMatrix<BaseFloat> &log_params(log_params_[i]),
        &other_log_params(other->log_params_[i]);
    ans += TraceMatMat(log_params, other_log_params, kTrans);
  }
  return ans;
}

void MixtureProbComponent::Scale(BaseFloat scale) {
  for (size_t i = 0; i < params_.size(); i++) {
    CuMatrix<BaseFloat> &log_params(log_params_[i]);
    log_params.Scale(scale);
  }
  Refresh();
}

void MixtureProbComponent::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const MixtureProbComponent *other =
      dynamic_cast<const MixtureProbComponent*>(&other_in);
  KALDI_ASSERT(other != NULL && other->params_.size() == params_.size());
  
  for (size_t i = 0; i < params_.size(); i++) {
    CuMatrix<BaseFloat> log_params(log_params_[i]),
        other_log_params(other->log_params_[i]);
    log_params.AddMat(alpha, other_log_params); // <- This is the key line.
  }
  Refresh();
}


void MixtureProbComponent::Init(BaseFloat learning_rate,
                                BaseFloat diag_element,
                                const std::vector<int32> &sizes) {
  UpdatableComponent::Init(learning_rate);
  input_dim_ = 0;
  output_dim_ = 0;
  params_.resize(sizes.size());
  log_params_.resize(sizes.size());
  KALDI_ASSERT(diag_element > 0.0 && diag_element < 1.0);
  // Initialize to a block-diagonal matrix consisting of a series of square
  // blocks, with sizes specified in "sizes".  Note: each block will typically
  // correspond to a number of clustered states, so this whole thing implements
  // an idea similar to the "state clustered tied mixture" system.
  for (size_t i = 0; i < sizes.size(); i++) {
    KALDI_ASSERT(sizes[i] > 0);
    int32 size = sizes[i];
    params_[i].Resize(size, size);
    input_dim_ += size;
    output_dim_ += size;
    if (size == 1) {
      params_[i](0,0) = 1.0;
    } else {
      BaseFloat off_diag_element = (1.0 - diag_element) / (size - 0.999999);
      params_[i].Set(off_diag_element);
      for (int32 j = 0; j < size; j++)
        params_[i](j, j) = diag_element;
    }
    log_params_[i] = params_[i];
    log_params_[i].ApplyLog(); // From now, log_params_ will be the
    // "primary" parameters, with params_ treated as derived quantities.
  }
}  

// e.g. args="learning-rate=0.01 diag-element=0.9 dims=3:4:5"
void MixtureProbComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_,
             diag_element = 0.9;
  std::vector<int32> dims;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("diag-element", &args, &diag_element); // optional.
  ok = ok && ParseFromString("dims", &args, &dims); // dims is colon-separated list.
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, diag_element, dims);
}

// For back-compatibility, we read and write the "params".
void MixtureProbComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MixtureProbComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<Params>");
  int32 size;
  ReadBasicType(is, binary, &size);
  input_dim_ = 0;
  output_dim_ = 0;
  KALDI_ASSERT(size >= 0);
  params_.resize(size);
  log_params_.resize(size);
  for (int32 i = 0; i < size; i++) {
    params_[i].Read(is, binary);
    input_dim_ += params_[i].NumCols();
    output_dim_ += params_[i].NumRows();
    log_params_[i] = params_[i];
    log_params_[i].ApplyLog();
  }

#if 0 // this is back-compatibility code, now disabled.  Will remove eventually.
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<IsGradient>") { // Back-compatibility code,
    // remove this later.
    bool tmp;
    ReadBasicType(is, binary, &tmp);
    ExpectToken(is, binary, "</MixtureProbComponent>");  
  } else {
    KALDI_ASSERT(token == "</MixtureProbComponent>");
  }
#else
  ExpectToken(is, binary, "</MixtureProbComponent>");
#endif
  
}

void MixtureProbComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MixtureProbComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<Params>");
  int32 size = params_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    params_[i].Write(os, binary);
  WriteToken(os, binary, "</MixtureProbComponent>");  
}

void MixtureProbComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
  }
  for (size_t i = 0; i < params_.size(); i++)
    log_params_[i].SetZero();
  Refresh();
}

void MixtureProbComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32, // num_chunks
                                     CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == InputDim());
  out->Resize(in.NumRows(), OutputDim());
  
  int32 num_frames = in.NumRows(),
      input_offset = 0,
      output_offset = 0;

  for (size_t i = 0; i < params_.size(); i++) {
    int32 this_input_dim = params_[i].NumCols(), // input dim of this block.
         this_output_dim = params_[i].NumRows();
    KALDI_ASSERT(this_input_dim > 0 && this_output_dim > 0);
    CuSubMatrix<BaseFloat> in_block(in, 0, num_frames,
                                  input_offset, this_input_dim),
        out_block(*out, 0, num_frames, output_offset, this_output_dim);
    const CuMatrix<BaseFloat> &param_block(params_[i]);
    out_block.AddMatMat(1.0, in_block, kNoTrans, param_block, kTrans, 0.0);
    input_offset += this_input_dim;
    output_offset += this_output_dim;   
  }
  KALDI_ASSERT(input_offset == InputDim() && output_offset == OutputDim());
}

void MixtureProbComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                    const CuMatrixBase<BaseFloat> &,// out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *to_update_in,
                                    CuMatrix<BaseFloat> *in_deriv) const {
  MixtureProbComponent *to_update = dynamic_cast<MixtureProbComponent*>(
      to_update_in);

  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == InputDim() && out_deriv.NumCols() == OutputDim());
  int32 num_frames = in_value.NumRows(),
      input_offset = 0,
     output_offset = 0;
  
  for (size_t i = 0; i < params_.size(); i++) {
    int32 this_input_dim = params_[i].NumCols(), // input dim of this block.
        this_output_dim = params_[i].NumRows();   
    KALDI_ASSERT(this_input_dim > 0 && this_output_dim > 0);
    CuSubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        input_offset, this_input_dim),
        in_deriv_block(*in_deriv, 0, num_frames,
                       input_offset, this_input_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        output_offset, this_output_dim);
    const CuMatrix<BaseFloat> &param_block(params_[i]);
    
    // Propagate gradient back to in_deriv.
    in_deriv_block.AddMatMat(1.0, out_deriv_block, kNoTrans, param_block,
                             kNoTrans, 0.0);
    
    if (to_update != NULL) {
      CuMatrix<BaseFloat> &log_param_block_to_update(to_update->log_params_[i]);
      const CuMatrix<BaseFloat> &param_block(this->params_[i]);

      int32 num_rows = this_output_dim, num_cols = this_input_dim;

      CuMatrix<BaseFloat> gradient(num_rows, num_cols); // gradient 
      // in space of derived params "params_".
      gradient.AddMatMat(1.0, out_deriv_block,
                         kTrans, in_value_block, kNoTrans,
                         0.0);
      
      CuVector<BaseFloat> param_col(num_rows),
          gradient_col(num_rows),
          log_gradient_col(num_rows),
          log_param_col(num_rows);
      for (int32 col = 0; col < num_cols; col++) {
        param_col.CopyColFromMat(param_block, col);
        gradient_col.CopyColFromMat(gradient, col);
        BaseFloat cT_g = VecVec(param_col, gradient_col);

        log_gradient_col.AddVecVec(1.0, param_col, gradient_col, 0.0); // h <-- diag(c) g.
        log_gradient_col.AddVec(-cT_g, param_col); // h -= (c^T g) c .  This is the
        // effect on the derivative of the sum-to-one constraint.
        log_param_col.CopyColFromMat(log_param_block_to_update, col);
        log_param_col.AddVec(to_update->learning_rate_,
                             log_gradient_col);
        // Gradient step in unnormalized log-prob space.
        log_param_block_to_update.CopyColFromVec(log_param_col, col); // Write back.
      }
    }
    input_offset += this_input_dim;
    output_offset += this_output_dim;   
  }
  if (to_update != NULL)
    to_update->Refresh();
  KALDI_ASSERT(input_offset == InputDim() && output_offset == OutputDim());
}

int32 MixtureProbComponent::GetParameterDim() const {
  int32 ans = 0;
  for (size_t i = 0; i < params_.size(); i++)
    ans += params_[i].NumRows() * params_[i].NumCols();
  return ans;
}

void MixtureProbComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  int32 offset = 0;
  for (size_t i = 0; i < params_.size(); i++) {
    int32 size = params_[i].NumRows() * params_[i].NumCols();
    params->Range(offset, size).CopyRowsFromMat(params_[i]);
    offset += size;
  }
  KALDI_ASSERT(offset == params->Dim());
}

void MixtureProbComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  int32 offset = 0;
  for (size_t i = 0; i < params_.size(); i++) {
    int32 size = params_[i].NumRows() * params_[i].NumCols();
    params_[i].CopyRowsFromVec(params.Range(offset, size));
    offset += size;
  }
  KALDI_ASSERT(offset == params.Dim());
}

void SumGroupComponent::Init(const std::vector<int32> &sizes) {
  KALDI_ASSERT(!sizes.empty());
  std::vector<Int32Pair> cpu_vec(sizes.size());
  std::vector<int32> reverse_cpu_vec;
  int32 cur_index = 0;
  for (size_t i = 0; i < sizes.size(); i++) {
    KALDI_ASSERT(sizes[i] > 0);
    cpu_vec[i].first = cur_index;
    cpu_vec[i].second = cur_index + sizes[i];
    cur_index += sizes[i];
    for (int32 j = cpu_vec[i].first; j < cpu_vec[i].second; j++)
      reverse_cpu_vec.push_back(i);
  }
  this->indexes_ = cpu_vec;
  this->reverse_indexes_ = reverse_cpu_vec;
  this->input_dim_ = cur_index;
  this->output_dim_ = sizes.size();
}

void SumGroupComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  std::vector<int32> sizes;
  bool ok = ParseFromString("sizes", &args, &sizes);

  if (!ok || !args.empty() || sizes.empty())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  this->Init(sizes);
}
  
Component* SumGroupComponent::Copy() const {
  SumGroupComponent *ans = new SumGroupComponent();
  ans->indexes_ = indexes_;
  ans->reverse_indexes_ = reverse_indexes_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  return ans;
}

void SumGroupComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SumGroupComponent>", "<Sizes>");
  std::vector<int32> sizes;
  ReadIntegerVector(is, binary, &sizes);
  ExpectToken(is, binary, "<SumGroupComponent>");
  this->Init(sizes);
}

void SumGroupComponent::GetSizes(std::vector<int32> *sizes) const {
  std::vector<Int32Pair> indexes;
  indexes_.CopyToVec(&indexes);
  sizes->resize(indexes.size());
  for (size_t i = 0; i < indexes.size(); i++) {
    (*sizes)[i] = indexes[i].second - indexes[i].first;
    if (i == 0) { KALDI_ASSERT(indexes[i].first == 0); }
    else { KALDI_ASSERT(indexes[i].first == indexes[i-1].second); }
    KALDI_ASSERT(indexes[i].second > indexes[i].first);
    (*sizes)[i] = indexes[i].second - indexes[i].first;
  }
}

void SumGroupComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SumGroupComponent>");
  WriteToken(os, binary, "<Sizes>");
  std::vector<int32> sizes;
  this->GetSizes(&sizes);
  WriteIntegerVector(os, binary, sizes);
  WriteToken(os, binary, "<SumGroupComponent>");
}

void SumGroupComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                  int32 num_chunks,
                                  CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), this->OutputDim(), kUndefined);
  out->SumColumnRanges(in, indexes_);
}

void SumGroupComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value,
                                 const CuMatrixBase<BaseFloat> &, // out_value,
                                 const CuMatrixBase<BaseFloat> &out_deriv,
                                 int32 num_chunks,
                                 Component *to_update,
                                 CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  in_deriv->CopyCols(out_deriv, reverse_indexes_);
}


std::string SpliceComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", context=" << left_context_ << "/" << right_context_;
  if (const_component_dim_ != 0)
    stream << ", const_component_dim=" << const_component_dim_;

  return stream.str();
}

void SpliceComponent::Init(int32 input_dim, int32 left_context,
                           int32 right_context, int32 const_component_dim) {
  input_dim_ = input_dim;
  const_component_dim_ = const_component_dim;
  left_context_ = left_context;
  right_context_ = right_context;
  KALDI_ASSERT(input_dim_ > 0 && left_context >= 0 && right_context >= 0);
  KALDI_ASSERT(const_component_dim_ >= 0 && const_component_dim_ < input_dim_);
}


// e.g. args == "input-dim=10 left-context=2 right-context=2
void SpliceComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim, left_context, right_context;
  bool ok = ParseFromString("input-dim", &args, &input_dim) &&
            ParseFromString("left-context", &args, &left_context) &&
            ParseFromString("right-context", &args, &right_context);

  int32 const_component_dim = 0;
  ParseFromString("const-component-dim", &args, &const_component_dim);
  
  if (!ok || !args.empty() || input_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(input_dim, left_context, right_context, const_component_dim);
}

int32 SpliceComponent::OutputDim() const {
  return (input_dim_  - const_component_dim_)
      * (1 + left_context_ + right_context_)
      + const_component_dim_;
}

void SpliceComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                int32 num_chunks,
                                CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() > 0 && num_chunks > 0);
  if (in.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << in.NumRows();
  int32 input_chunk_size = in.NumRows() / num_chunks,
       output_chunk_size = input_chunk_size - left_context_ - right_context_,
               input_dim = in.NumCols(),
              output_dim = OutputDim();
  if (output_chunk_size <= 0)
    KALDI_ERR << "Splicing features: output will have zero dimension. "
              << "Probably a code error.";
  out->Resize(num_chunks * output_chunk_size, output_dim);

  // 'indexes' is, for each index from 0 to (left_context_+right_context_+1)-1,
  // then for each row of "out", the corresponding row of "in" that we copy from.
  int32 num_splice = left_context_ + right_context_ + 1,
      const_dim = const_component_dim_;
  std::vector<std::vector<int32> > indexes(num_splice);
  // const_component_dim_ != 0, "const_indexes" will be used to determine which
  // row of "in" we copy the last part of each row of "out" from (this part is
  // not subject to splicing, it's assumed constant for each frame of "input".
  std::vector<int32> const_indexes(const_dim == 0 ? 0 : out->NumRows());

  for (int32 c = 0; c < num_splice; c++) 
    indexes[c].resize(out->NumRows());

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    for (int32 c = 0; c < num_splice; c++) {
      for (int32 offset = 0; offset < output_chunk_size; offset++) {
        indexes[c][chunk * output_chunk_size + offset] =
            chunk * input_chunk_size + c + offset;
      }
    }
    if (const_dim != 0) {
      for (int32 offset = 0; offset < output_chunk_size; offset++)
        const_indexes[chunk * output_chunk_size + offset] =
            chunk * input_chunk_size + offset; // there is
      // an arbitrariness here; since we assume the const_component
      // is constant within a chunk, it doesn't matter from where we copy.
    }
  }
  for (int32 c = 0; c < num_splice; c++) {
    int32 dim = input_dim - const_dim; // dimension we
    // are splicing
    CuSubMatrix<BaseFloat> in_part(in, 0, in.NumRows(),
                                   0, dim),
        out_part(*out, 0, out->NumRows(),
                 c * dim, dim);
    out_part.CopyRows(in_part, indexes[c]);
  }
  if (const_dim != 0) {
    CuSubMatrix<BaseFloat> in_part(in, 0, in.NumRows(),
                                   in.NumCols() - const_dim, const_dim),
        out_part(*out, 0, out->NumRows(),
                 out->NumCols() - const_dim, const_dim);
    out_part.CopyRows(in_part, const_indexes);
  }
}

void SpliceComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                               const CuMatrixBase<BaseFloat> &, // out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               int32 num_chunks,
                               Component *to_update, // may == "this".
                               CuMatrix<BaseFloat> *in_deriv) const {
 
  KALDI_ASSERT(out_deriv.NumRows() > 0 && num_chunks > 0);

  if (out_deriv.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << out_deriv.NumRows();
  
  int32 output_chunk_size = out_deriv.NumRows() / num_chunks,
      input_chunk_size = output_chunk_size + left_context_ + right_context_,
      output_dim = out_deriv.NumCols(),
      input_dim = InputDim();
 
  KALDI_ASSERT( OutputDim() == output_dim );

  in_deriv->Resize(num_chunks * input_chunk_size, input_dim, kUndefined);

  int32 num_splice = left_context_ + right_context_ + 1,
      const_dim = const_component_dim_;
  // 'indexes' is, for each index from 0 to num_splice - 1,
  // then for each row of "in_deriv", the corresponding row of "out_deriv" that
  // we add, or -1 if.
    
  std::vector<std::vector<int32> > indexes(num_splice);
  // const_dim != 0, "const_indexes" will be used to determine which
  // row of "in" we copy the last part of each row of "out" from (this part is
  // not subject to splicing, it's assumed constant for each frame of "input".
  std::vector<int32> const_indexes(const_dim == 0 ? 0 : in_deriv->NumRows(),
                                   -1);

  for (int32 c = 0; c < indexes.size(); c++) 
    indexes[c].resize(in_deriv->NumRows(), -1); // set to -1 by default,
  // this gets interpreted by the CopyRows() code as a signal to zero the output...

  int32 dim = input_dim - const_dim; // dimension we are splicing

  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    for (int32 c = 0; c < num_splice; c++)
      for (int32 offset = 0; offset < output_chunk_size; offset++)
        indexes[c][chunk * input_chunk_size + c + offset] =
            chunk * output_chunk_size + offset;

    // Note: when changing over to the CUDA code, we also changed
    // how the derivatives are propagated through the splicing layer
    // for the const-component-dim.  The code was never being used,
    // so it doesn't matter.  The way we now do it probably makes more
    // sense (to get the derivative, you'd have to sum over time, not
    // pick an arbitrary time)
    if (const_dim != 0)
      for (int32 offset = 0; offset < output_chunk_size; offset++)
        const_indexes[chunk * input_chunk_size + offset] =
            chunk * output_chunk_size + offset;
  }
    
  CuMatrix<BaseFloat> temp_mat(in_deriv->NumRows(), dim, kUndefined);
    
  for (int32 c = 0; c < num_splice; c++) {
    int32 dim = input_dim - const_dim; // dimension we
    // are splicing
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_deriv.NumRows(),
                                          c * dim, dim),
        in_deriv_part(*in_deriv, 0, in_deriv->NumRows(),
                      0, dim);
    if (c == 0)
      in_deriv_part.CopyRows(out_deriv_part, indexes[c]);
    else {
      temp_mat.CopyRows(out_deriv_part, indexes[c]);
      in_deriv_part.AddMat(1.0, temp_mat);
    }
  }
  if (const_dim != 0) {
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_deriv.NumRows(),
                                          out_deriv.NumCols() - const_dim,
                                          const_dim),
        in_deriv_part(*in_deriv, 0, in_deriv->NumRows(),
                      in_deriv->NumCols() - const_dim, const_dim);
    in_deriv_part.CopyRows(out_deriv_part, const_indexes);
  }
}

Component *SpliceComponent::Copy() const {
  SpliceComponent *ans = new SpliceComponent();
  ans->input_dim_ = input_dim_;
  ans->left_context_ = left_context_;
  ans->right_context_ = right_context_;
  ans->const_component_dim_ = const_component_dim_;
  return ans;
}

void SpliceComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SpliceComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<ConstComponentDim>");
  ReadBasicType(is, binary, &const_component_dim_);
  ExpectToken(is, binary, "</SpliceComponent>");
}

void SpliceComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SpliceComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<ConstComponentDim>");
  WriteBasicType(os, binary, const_component_dim_);
  WriteToken(os, binary, "</SpliceComponent>");  
}


std::string SpliceMaxComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", context=" << left_context_
         << "/" << right_context_;
  return stream.str();
}

void SpliceMaxComponent::Init(int32 dim, int32 left_context,
                              int32 right_context) {
  dim_ = dim;
  left_context_ = left_context;
  right_context_ = right_context;
  KALDI_ASSERT(dim_ > 0 && left_context >= 0 && right_context >= 0);
}


// e.g. args == "dim=10 left-context=2 right-context=2
void SpliceMaxComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim, left_context, right_context;
  bool ok = ParseFromString("dim", &args, &dim) &&
            ParseFromString("left-context", &args, &left_context) &&
            ParseFromString("right-context", &args, &right_context);
  
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, left_context, right_context);
}

void SpliceMaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                   int32 num_chunks,
                                   CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() > 0 && in.NumCols() == InputDim());
  if (in.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << in.NumRows();
  int32 input_chunk_size = in.NumRows() / num_chunks,
       output_chunk_size = input_chunk_size - left_context_ - right_context_,
                     dim = in.NumCols();
  if (output_chunk_size <= 0)
    KALDI_ERR << "Splicing features: output will have zero dimension. "
              << "Probably a code error.";
  out->Resize(num_chunks * output_chunk_size, dim);
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> input_chunk(in,
                                     chunk * input_chunk_size, input_chunk_size,
                                     0, dim),
                        output_chunk(*out,
                                     chunk * output_chunk_size, output_chunk_size,
                                     0, dim);
    for (int32 offset = 0;
         offset < 1 + left_context_ + right_context_;
         offset++) {
      CuSubMatrix<BaseFloat> input_chunk_part(input_chunk,
                                            offset, output_chunk_size, 0, dim);
      if (offset == 0) output_chunk.CopyFromMat(input_chunk_part);
      else {
        output_chunk.Max(input_chunk_part);
      }
    }
  }  
}

void SpliceMaxComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &, // out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  int32 num_chunks,
                                  Component *to_update, // may == "this".
                                  CuMatrix<BaseFloat> *in_deriv) const {
 KALDI_ASSERT(out_deriv.NumRows() > 0 && num_chunks > 0);

  if (out_deriv.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << out_deriv.NumRows();
  
  int32 output_chunk_size = out_deriv.NumRows() / num_chunks,
         input_chunk_size = output_chunk_size + left_context_ + right_context_,
                      dim = out_deriv.NumCols();

  KALDI_ASSERT(dim == InputDim());

  in_deriv->Resize(num_chunks * input_chunk_size, dim); // Will zero it.
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> in_deriv_chunk(*in_deriv, 
                                        chunk * input_chunk_size,
                                        input_chunk_size, 
                                        0, dim),
                         in_value_chunk(in_value,
                                        chunk * input_chunk_size,
                                        input_chunk_size, 
                                        0, dim),
                        out_deriv_chunk(out_deriv,
                                        chunk * output_chunk_size,
                                        output_chunk_size,
                                        0, dim);
    for (int32 r = 0; r < out_deriv_chunk.NumRows(); r++) {
      for (int32 c = 0; c < dim; c++) {
        int32 in_r_begin = r, in_r_end = r + left_context_ + right_context_ + 1;
        int32 in_r_max = -1;
        BaseFloat max_input = -std::numeric_limits<BaseFloat>::infinity();
        for (int32 in_r = in_r_begin; in_r < in_r_end; in_r++) {
          BaseFloat input = in_value_chunk(in_r, c);
          if (input > max_input) {
            max_input = input;
            in_r_max = in_r;
          }
        }
        KALDI_ASSERT(in_r_max != -1);
        (*in_deriv)(in_r_max, c) += out_deriv_chunk(r, c);
      }
    }
  }
}

Component *SpliceMaxComponent::Copy() const {
  SpliceMaxComponent *ans = new SpliceMaxComponent();
  ans->Init(dim_, left_context_, right_context_);
  return ans;
}

void SpliceMaxComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SpliceMaxComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "</SpliceMaxComponent>");
}

void SpliceMaxComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SpliceMaxComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "</SpliceMaxComponent>");  
}

std::string DctComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", dct_dim=" << dct_mat_.NumCols();
  if (dct_mat_.NumCols() != dct_mat_.NumRows())
    stream << ", dct_keep_dim=" << dct_mat_.NumRows();

  return stream.str();
}

void DctComponent::Init(int32 dim, int32 dct_dim, bool reorder, int32 dct_keep_dim) {
  int dct_keep_dim_ = (dct_keep_dim > 0) ? dct_keep_dim : dct_dim;

  KALDI_ASSERT(dim > 0 && dct_dim > 0);
  KALDI_ASSERT(dim % dct_dim == 0); // dct_dim must divide dim.
  KALDI_ASSERT(dct_dim >= dct_keep_dim_)
  dim_ = dim;
  dct_mat_.Resize(dct_keep_dim_, dct_dim);
  reorder_ = reorder;
  Matrix<BaseFloat> dct_mat(dct_keep_dim_, dct_dim);
  ComputeDctMatrix(&dct_mat);
  dct_mat_ = dct_mat;
}



void DctComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim, dct_dim;
  bool reorder = false;
  int32 dct_keep_dim = 0;

  bool ok = ParseFromString("dim", &args, &dim) &&
            ParseFromString("dct-dim", &args, &dct_dim);
  ParseFromString("reorder", &args, &reorder);
  ParseFromString("dct-keep-dim", &args, &dct_keep_dim);

  if (!ok || !args.empty() || dim <= 0 || dct_dim <= 0 || dct_keep_dim < 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim, dct_dim, reorder, dct_keep_dim);
}

void DctComponent::Reorder(CuMatrixBase<BaseFloat> *mat, bool reverse) const {
  // reorders into contiguous blocks of dize "dct_dim_", assuming that
  // such blocks were interlaced before.  if reverse==true, does the
  // reverse.
  int32 dct_dim = dct_mat_.NumCols(),
      dct_keep_dim = dct_mat_.NumRows(),
      block_size_in = dim_ / dct_dim,
      block_size_out = dct_keep_dim;

  //This does not necesarily needs to be true anymore -- output must be reordered as well, but the dimension differs... 
  //KALDI_ASSERT(mat->NumCols() == dim_);
  if (reverse) std::swap(block_size_in, block_size_out);

  CuVector<BaseFloat> temp(mat->NumCols());
  for (int32 i = 0; i < mat->NumRows(); i++) {
    CuSubVector<BaseFloat> row(*mat, i);
    int32 num_blocks_in = block_size_out;
    for (int32 b = 0; b < num_blocks_in; b++) {
      for (int32 j = 0; j < block_size_in; j++) {
        temp(j * block_size_out + b) = row(b * block_size_in + j);
      }
    }
    row.CopyFromVec(temp);
  }
}

void DctComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                             int32, // num_chunks
                             CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == InputDim());
  
  int32 dct_dim = dct_mat_.NumCols(),
        dct_keep_dim = dct_mat_.NumRows(),
        num_chunks = dim_ / dct_dim,
        num_rows = in.NumRows();
  
  out->Resize(num_rows, num_chunks * dct_keep_dim);

  CuMatrix<BaseFloat> in_tmp;
  if (reorder_) {
    in_tmp = in;
    Reorder(&in_tmp, false);
  }
  
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> in_mat(reorder_ ? in_tmp : in,
                                0, num_rows, dct_dim * chunk, dct_dim),
                        out_mat(*out, 
                                0, num_rows, dct_keep_dim * chunk, dct_keep_dim);

    out_mat.AddMatMat(1.0, in_mat, kNoTrans, dct_mat_, kTrans, 0.0);
  }
  if (reorder_)
    Reorder(out, true);
}

void DctComponent::Backprop(const CuMatrixBase<BaseFloat>&, // in_value,
                            const CuMatrixBase<BaseFloat>&, // out_value,
                            const CuMatrixBase<BaseFloat> &out_deriv,
                            int32, // num_chunks
                            Component*,// to_update
                            CuMatrix<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(out_deriv.NumCols() == OutputDim());

  int32 dct_dim = dct_mat_.NumCols(),
        dct_keep_dim = dct_mat_.NumRows(),
        num_chunks = dim_ / dct_dim,
        num_rows = out_deriv.NumRows();

  in_deriv->Resize(num_rows, dim_);
  
  CuMatrix<BaseFloat> out_deriv_tmp;
  if (reorder_) {
    out_deriv_tmp = out_deriv;
    Reorder(&out_deriv_tmp, false);
  }
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    CuSubMatrix<BaseFloat> in_deriv_mat(*in_deriv,
                                      0, num_rows, dct_dim * chunk, dct_dim),
                        out_deriv_mat(reorder_ ? out_deriv_tmp : out_deriv,
                                      0, num_rows, dct_keep_dim * chunk, dct_keep_dim);

    // Note: in the reverse direction the DCT matrix is transposed.  This is
    // normal when computing derivatives; the necessity for the transpose is
    // obvious if you consider what happens when the input and output dims
    // differ.
    in_deriv_mat.AddMatMat(1.0, out_deriv_mat, kNoTrans,
                           dct_mat_, kNoTrans, 0.0);
  }
  if (reorder_)
    Reorder(in_deriv, true);
}

Component* DctComponent::Copy() const {
  DctComponent *ans = new DctComponent();
  ans->dct_mat_ = dct_mat_;
  ans->dim_ = dim_;
  ans->reorder_ = reorder_;
  return ans;
}

void DctComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DctComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<DctDim>");
  int32 dct_dim = dct_mat_.NumCols();
  WriteBasicType(os, binary, dct_dim);
  WriteToken(os, binary, "<Reorder>");
  WriteBasicType(os, binary, reorder_);
  WriteToken(os, binary, "<DctKeepDim>");
  int32 dct_keep_dim = dct_mat_.NumRows();
  WriteBasicType(os, binary, dct_keep_dim);
  WriteToken(os, binary, "</DctComponent>");  
}

void DctComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DctComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  
  ExpectToken(is, binary, "<DctDim>");
  int32 dct_dim; 
  ReadBasicType(is, binary, &dct_dim);
  
  ExpectToken(is, binary, "<Reorder>");
  ReadBasicType(is, binary, &reorder_);

  int32 dct_keep_dim = dct_dim;
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<DctKeepDim>") {
    ReadBasicType(is, binary, &dct_keep_dim);
    ExpectToken(is, binary, "</DctComponent>");
  } else if (token != "</DctComponent>") {
    KALDI_ERR << "Expected token \"</DctComponent>\", got instead \""
              << token << "\".";
  }

  KALDI_ASSERT(dct_dim > 0 && dim_ > 0 && dim_ % dct_dim == 0);
  Init(dim_, dct_dim, reorder_, dct_keep_dim);
  //idct_mat_.Resize(dct_keep_dim, dct_dim);
  //ComputeDctMatrix(&dct_mat_);
}

void FixedLinearComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("matrix", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  bool binary;
  Input ki(filename, &binary);
  CuMatrix<BaseFloat> mat;
  mat.Read(ki.Stream(), binary);
  KALDI_ASSERT(mat.NumRows() != 0);
  Init(mat);
}


std::string FixedLinearComponent::Info() const {
  std::stringstream stream;
  BaseFloat mat_size = static_cast<BaseFloat>(mat_.NumRows())
      * static_cast<BaseFloat>(mat_.NumCols()),
      mat_stddev = std::sqrt(TraceMatMat(mat_, mat_, kTrans) /
                         mat_size); 
  stream << Component::Info() << ", params-stddev=" << mat_stddev;
  return stream.str();
}

void FixedLinearComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), mat_.NumRows());
  out->AddMatMat(1.0, in, kNoTrans, mat_, kTrans, 0.0);
}

void FixedLinearComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), mat_.NumCols());
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, mat_, kNoTrans, 0.0);
}

Component* FixedLinearComponent::Copy() const {
  FixedLinearComponent *ans = new FixedLinearComponent();
  ans->Init(mat_);
  return ans;
}


void FixedLinearComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedLinearComponent>");
  WriteToken(os, binary, "<CuMatrix>");
  mat_.Write(os, binary);
  WriteToken(os, binary, "</FixedLinearComponent>");  
}

void FixedLinearComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedLinearComponent>", "<CuMatrix>");
  mat_.Read(is, binary);
  ExpectToken(is, binary, "</FixedLinearComponent>");
}

void FixedAffineComponent::Init(const CuMatrixBase<BaseFloat> &mat) {
  KALDI_ASSERT(mat.NumCols() > 1);
  linear_params_ = mat.Range(0, mat.NumRows(),
                             0, mat.NumCols() - 1);
  bias_params_.Resize(mat.NumRows());
  bias_params_.CopyColFromMat(mat, mat.NumCols() - 1);
}


void FixedAffineComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("matrix", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  bool binary;
  Input ki(filename, &binary);
  CuMatrix<BaseFloat> mat;
  mat.Read(ki.Stream(), binary);
  KALDI_ASSERT(mat.NumRows() != 0);
  Init(mat);
}


std::string FixedAffineComponent::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size = static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols()),
      linear_params_stddev =
      std::sqrt(TraceMatMat(linear_params_,
                            linear_params_, kTrans) /
                linear_params_size),
      bias_params_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                                     bias_params_.Dim());
      
  stream << Component::Info() << ", linear-params-stddev=" << linear_params_stddev
         << ", bias-params-stddev=" << bias_params_stddev;
  return stream.str();
}

void FixedAffineComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), linear_params_.NumRows());
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 0.0);
  out->AddVecToRows(1.0, bias_params_);
}

void FixedAffineComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), linear_params_.NumCols());
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans, 0.0);
}

Component* FixedAffineComponent::Copy() const {
  FixedAffineComponent *ans = new FixedAffineComponent();
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  return ans;
}


void FixedAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedAffineComponent>");
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</FixedAffineComponent>");  
}

void FixedAffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedAffineComponent>", "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);  
  ExpectToken(is, binary, "</FixedAffineComponent>");
}


void FixedScaleComponent::Init(const CuVectorBase<BaseFloat> &scales) {
  KALDI_ASSERT(scales.Dim() != 0);
  scales_ = scales;
}

void FixedScaleComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("scales", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  CuVector<BaseFloat> vec;
  ReadKaldiObject(filename, &vec);
  Init(vec);
}


std::string FixedScaleComponent::Info() const {
  std::stringstream stream;
  BaseFloat scales_size = static_cast<BaseFloat>(scales_.Dim()),
      scales_mean = scales_.Sum() / scales_size,
      scales_stddev = std::sqrt(VecVec(scales_, scales_) / scales_size)
       - (scales_mean * scales_mean);
  stream << Component::Info() << ", scales-mean=" << scales_mean
         << ", scales-stddev=" << scales_stddev;
  return stream.str();
}

void FixedScaleComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  *out = in;
  out->MulColsVec(scales_);
}

void FixedScaleComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  *in_deriv = out_deriv;
  in_deriv->MulColsVec(scales_);
}

Component* FixedScaleComponent::Copy() const {
  FixedScaleComponent *ans = new FixedScaleComponent();
  ans->scales_ = scales_;
  return ans;
}


void FixedScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedScaleComponent>");
  WriteToken(os, binary, "<Scales>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "</FixedScaleComponent>");  
}

void FixedScaleComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedScaleComponent>", "<Scales>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "</FixedScaleComponent>");
}

void FixedBiasComponent::Init(const CuVectorBase<BaseFloat> &bias) {
  KALDI_ASSERT(bias.Dim() != 0);
  bias_ = bias;
}

void FixedBiasComponent::InitFromString(std::string args) {
  std::string orig_args = args;
  std::string filename;
  bool ok = ParseFromString("bias", &args, &filename);

  if (!ok || !args.empty()) 
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";

  CuVector<BaseFloat> vec;
  ReadKaldiObject(filename, &vec);
  Init(vec);
}


std::string FixedBiasComponent::Info() const {
  std::stringstream stream;
  BaseFloat bias_size = static_cast<BaseFloat>(bias_.Dim()),
      bias_mean = bias_.Sum() / bias_size,
      bias_stddev = std::sqrt(VecVec(bias_, bias_) / bias_size)
       - (bias_mean * bias_mean);
  stream << Component::Info() << ", bias-mean=" << bias_mean
         << ", bias-stddev=" << bias_stddev;
  return stream.str();
}

void FixedBiasComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     CuMatrix<BaseFloat> *out) const {
  *out = in;
  out->AddVecToRows(1.0, bias_, 1.0);
}

void FixedBiasComponent::Backprop(const CuMatrixBase<BaseFloat> &, // in_value
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    CuMatrix<BaseFloat> *in_deriv) const {
  *in_deriv = out_deriv;
}

Component* FixedBiasComponent::Copy() const {
  FixedBiasComponent *ans = new FixedBiasComponent();
  ans->bias_ = bias_;
  return ans;
}


void FixedBiasComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FixedBiasComponent>");
  WriteToken(os, binary, "<Bias>");
  bias_.Write(os, binary);
  WriteToken(os, binary, "</FixedBiasComponent>");  
}

void FixedBiasComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedBiasComponent>", "<Bias>");
  bias_.Read(is, binary);
  ExpectToken(is, binary, "</FixedBiasComponent>");
}




std::string DropoutComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", dropout_proportion = "
         << dropout_proportion_ << ", dropout_scale = "
         << dropout_scale_;
  return stream.str();
}

void DropoutComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat dropout_proportion = 0.5, dropout_scale = 0.0;
  bool ok = ParseFromString("dim", &args, &dim);
  ParseFromString("dropout-proportion", &args, &dropout_proportion);
  ParseFromString("dropout-scale", &args, &dropout_scale);
  
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type DropoutComponent: \""
              << orig_args << "\"";
  Init(dim, dropout_proportion, dropout_scale);
}

void DropoutComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DropoutComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<DropoutScale>");
  ReadBasicType(is, binary, &dropout_scale_);
  ExpectToken(is, binary, "<DropoutProportion>");
  ReadBasicType(is, binary, &dropout_proportion_);
  ExpectToken(is, binary, "</DropoutComponent>");
}

void DropoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DropoutComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<DropoutScale>");
  WriteBasicType(os, binary, dropout_scale_);
  WriteToken(os, binary, "<DropoutProportion>");
  WriteBasicType(os, binary, dropout_proportion_);
  WriteToken(os, binary, "</DropoutComponent>");  
}


void DropoutComponent::Init(int32 dim,
                            BaseFloat dropout_proportion,
                            BaseFloat dropout_scale){
  dim_ = dim;
  dropout_proportion_ = dropout_proportion;
  dropout_scale_ = dropout_scale;
}
  
void DropoutComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
    				 int32 num_chunks,
    				 CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == this->InputDim());
  out->Resize(in.NumRows(), in.NumCols());

  BaseFloat dp = dropout_proportion_;
  KALDI_ASSERT(dp < 1.0 && dp >= 0.0);
  KALDI_ASSERT(dropout_scale_ <= 1.0 && dropout_scale_ >= 0.0);

  BaseFloat low_scale = dropout_scale_,
      high_scale = (1.0 - (dp * low_scale)) / (1.0 - dp),
      average = (low_scale * dp) +
                (high_scale * (1.0 - dp));
  KALDI_ASSERT(fabs(average - 1.0) < 0.01);

  out->Resize(in.NumRows(), in.NumCols(), kUndefined);

  // This const_cast is only safe assuming you don't attempt
  // to use multi-threaded code with the GPU.
  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);

  
  out->Add(-dp); // now, a proportion "dp" will be <0.0
  out->ApplyHeaviside(); // apply the function (x>0?1:0).  Now, a proportion "dp" will
                         // be zero and (1-dp) will be 1.0.
  if ((high_scale - low_scale) != 1.0)
    out->Scale(high_scale - low_scale); // now, "dp" are 0 and (1-dp) are "high_scale-low_scale".
  if (low_scale != 0.0)
    out->Add(low_scale); // now "dp" equal "low_scale" and (1.0-dp) equal "high_scale".

  out->MulElements(in);
}

void DropoutComponent::Backprop(const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *, // to_update
                                CuMatrix<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(SameDim(in_value, out_value) && SameDim(in_value, out_deriv));
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->AddMatMatDivMat(out_deriv, out_value, in_value);
}

Component* DropoutComponent::Copy() const {
  return new DropoutComponent(dim_,
                              dropout_proportion_,
                              dropout_scale_);
}

void AdditiveNoiseComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat stddev = 1.0;
  bool ok = ParseFromString("dim", &args, &dim);
  ParseFromString("stddev", &args, &stddev);  
  
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type AdditiveNoiseComponent: \""
              << orig_args << "\"";
  Init(dim, stddev);
}

void AdditiveNoiseComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<AdditiveNoiseComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<Stddev>");
  ReadBasicType(is, binary, &stddev_);
  ExpectToken(is, binary, "</AdditiveNoiseComponent>");
}

void AdditiveNoiseComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AdditiveNoiseComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Stddev>");
  WriteBasicType(os, binary, stddev_);
  WriteToken(os, binary, "</AdditiveNoiseComponent>");  
}

void AdditiveNoiseComponent::Init(int32 dim, BaseFloat stddev) {
  dim_ = dim;
  stddev_ = stddev;
}
  
void AdditiveNoiseComponent::Propagate(
    const CuMatrixBase<BaseFloat> &in,
    int32 num_chunks,
    CuMatrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == this->InputDim());
  *out = in;
  CuMatrix<BaseFloat> rand(in.NumRows(), in.NumCols());
  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand);
  out->AddMat(stddev_, rand);
}

void AffineComponentA::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<AffineComponentA>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "<InputScatter>");
  input_scatter_.Read(is, binary);
  ExpectToken(is, binary, "<OutputScatter>");
  output_scatter_.Read(is, binary);  
  ExpectToken(is, binary, "<InC>");
  in_C_.Read(is, binary);
  ExpectToken(is, binary, "<InCInv>");
  in_C_inv_.Read(is, binary);
  ExpectToken(is, binary, "<OutC>");
  out_C_.Read(is, binary);
  ExpectToken(is, binary, "<OutCInv>");
  out_C_inv_.Read(is, binary);
  ExpectToken(is, binary, "<InvFisherIn>");
  inv_fisher_in_.Read(is, binary);
  ExpectToken(is, binary, "<InvFisherOut>");
  inv_fisher_out_.Read(is, binary);
  ExpectToken(is, binary, "</AffineComponentA>");
}

void AffineComponentA::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AffineComponentA>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<InputScatter>");
  input_scatter_.Write(os, binary);
  WriteToken(os, binary, "<OutputScatter>");
  output_scatter_.Write(os, binary);
  WriteToken(os, binary, "<InC>");
  in_C_.Write(os, binary);
  WriteToken(os, binary, "<InCInv>");
  in_C_inv_.Write(os, binary);
  WriteToken(os, binary, "<OutC>");
  out_C_.Write(os, binary);
  WriteToken(os, binary, "<OutCInv>");
  out_C_inv_.Write(os, binary);
  WriteToken(os, binary, "<InvFisherIn>");
  inv_fisher_in_.Write(os, binary);
  WriteToken(os, binary, "<InvFisherOut>");
  inv_fisher_out_.Write(os, binary);
  WriteToken(os, binary, "</AffineComponentA>");
}

AffineComponentA::AffineComponentA(const AffineComponent &component):
    AffineComponent(component) { }


void AffineComponentA::InitializeScatter() {
  KALDI_ASSERT(is_gradient_ &&
               "InitializeScatter should only be called on gradients.");
  KALDI_ASSERT(input_scatter_.NumRows() == 0 &&
               output_scatter_.NumRows() == 0 &&
               "InitializeScatter called when already initialized.");
  input_scatter_.Resize(InputDim() + 1); // + 1 because of the bias; we include
  // that in the input dimension.
  output_scatter_.Resize(OutputDim());
}

void AffineComponentA::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
  input_scatter_.Scale(scale);
  output_scatter_.Scale(scale);
  // Remove all precomputed quantities, they'll be invalid.
  ClearPrecomputedQuantities();
}

void AffineComponentA::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const AffineComponentA *other =
      dynamic_cast<const AffineComponentA*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
  input_scatter_.AddSp(alpha, other->input_scatter_);
  output_scatter_.AddSp(alpha, other->output_scatter_);  
  // Remove all precomputed quantities, they'll be invalid.
  ClearPrecomputedQuantities();
}

Component* AffineComponentA::Copy() const {
  // The initializer below will be the one that takes AffineComponent,
  // so we need to take care of the remaining parameters.
  AffineComponentA *ans = new AffineComponentA(*this);
  ans->input_scatter_ = input_scatter_;
  ans->output_scatter_ = output_scatter_;
  return ans;
}

void AffineComponentA::ClearPrecomputedQuantities() {
  in_C_.Resize(0);
  in_C_inv_.Resize(0);
  out_C_.Resize(0);
  out_C_inv_.Resize(0);
  inv_fisher_in_.Resize(0);
  inv_fisher_out_.Resize(0);
}

void AffineComponentA::UpdateSimple(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  KALDI_ASSERT(this->is_gradient_);
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);

  // The rest of this function is about updating the scatters.
  if (input_scatter_.NumRows() != 0) { // scatter is to be accumulated..
    CuMatrix<double> in_value_dbl(in_value.NumCols() + 1,
                                in_value.NumRows());
    in_value_dbl.Range(0, in_value.NumCols(),
                       0, in_value.NumRows()).CopyFromMat(in_value, kTrans);
    in_value_dbl.Row(in_value.NumCols()).Set(1.0);
    input_scatter_.AddMat2(1.0, in_value_dbl, kNoTrans, 1.0);
  }
  if (output_scatter_.NumRows() != 0) {
    CuMatrix<double> out_deriv_dbl(out_deriv, kTrans);
    output_scatter_.AddMat2(1.0, out_deriv_dbl, kNoTrans, 1.0);
  }
}

// static
void AffineComponentA::ComputeTransforms(const CuSpMatrix<double> &scatter_in,
                                         const PreconditionConfig &config,
                                         double tot_count,
                                         CuTpMatrix<double> *C,
                                         CuTpMatrix<double> *C_inv) {
  CuSpMatrix<double> scatter(scatter_in);
  KALDI_ASSERT(scatter.Trace() > 0);

  scatter.Scale(1.0 / tot_count);
  // Smooth using "alpha"-- smoothing with the unit matrix.
  
  double d = config.alpha * scatter.Trace() / scatter.NumRows();
  for (int32 i = 0; i < scatter.NumRows(); i++)
    scatter(i, i) += d;
  
  C->Resize(scatter.NumRows());
  C->Cholesky(scatter);
  *C_inv = *C;
  C_inv->Invert();
}
/*
  // "transform" is now the cholesky factor C.

  // Now, the scatter may be viewed as a scatter of gradients (not parameters),
  // so call it S = \sum g g^T.  [we omit the index on g.]  We now have S = C
  // C^T.  the transformed g would be g' = C^{-1} g.  [this would make S' unit.]
  // If renormalize == true, we want to ensure that trace(C^{-1} S C^{-T}) equals
  // trace(S).  This is a way of ensuring that the magnitude of the gradients is
  // about the same after transformation.  Now, trace(C^{-1} S C^{-T}) is trace(I) =
  // dim(S).  So to renormalize to make it equal to trace(S), we'd have to scale
  // by trace(S)/dim(S), which is equivalent to scaling C itself by
  // [trace(S)/dim(S)]^{-0.5}.  Note: this assumes that alpha is small.
  // We may have to revisit this later

  if (config.renormalize)
    transform->Scale(pow(scatter.Trace() / scatter.NumRows(), -0.5));
  
  // Now take care of whether it should be inverted or not, and
  // transposed or not.
  if (is_gradient) {
    if (forward) transform->Invert(); // g' <-- C^{-1} g
    // else: g <-- C g'
    *trans = kNoTrans;  
  } else {
    if (!forward) transform->Invert(); // p <-- C^{-T} p'
    // else: p' <-- C^T p
    *trans = kTrans;
  }
}
*/

// static
void AffineComponentA::ComputePreconditioner(const CuSpMatrix<double> &scatter_in,
                                             const PreconditionConfig &config,
                                             double tot_count,
                                             CuSpMatrix<double> *inv_fisher) {
  CuSpMatrix<double> scatter(scatter_in);
  KALDI_ASSERT(scatter.Trace() > 0);

  scatter.Scale(1.0 / tot_count);
  // Smooth using "alpha"-- smoothing with the unit matrix.
  
  double d = config.alpha * scatter.Trace() / scatter.NumRows();
  for (int32 i = 0; i < scatter.NumRows(); i++)
    scatter(i, i) += d;
  
  inv_fisher->Resize(scatter.NumRows());
  inv_fisher->CopyFromSp(scatter);
  inv_fisher->Invert();

  if (config.renormalize) {
    // renormalize so trace(inv_fisher . scatter) equals
    // trace(scatter . unit-matrix).
    inv_fisher->Scale(scatter.Trace() / TraceSpSp(*inv_fisher, scatter));
  }
}


void AffineComponentA::Transform(
    const PreconditionConfig &config,
    bool forward,
    AffineComponent *component) {
  if (!config.do_precondition) return; // There is nothing to do in this case.
  // (this option will probably only be used for testing.)
  
  KALDI_ASSERT(component != NULL);

  if (in_C_.NumRows() == 0) { // Need to pre-compute some things.
    double tot_count = input_scatter_(InputDim(), InputDim());
    // This equals the total count, because for each frame the last
    // element of the extended input vector is 1.
    ComputeTransforms(input_scatter_, config, tot_count, &in_C_, &in_C_inv_);
    ComputeTransforms(output_scatter_, config, tot_count, &out_C_, &out_C_inv_);
  }

  // "invert" is true if these two bools have the same value.
  bool is_gradient = component->is_gradient_,
      invert = (is_gradient == forward);
  
  // "params" are the parameters of "component" that we'll be changing.
  // Get them as a single matrix.
  CuMatrix<double> params(OutputDim(), InputDim() + 1);
  params.Range(0, OutputDim(), 0, InputDim()).CopyFromMat(
      component->linear_params_);
  params.CopyColFromVec(CuVector<double>(component->bias_params_),
                        InputDim());
  

  MatrixTransposeType transpose_in = (is_gradient ? kTrans : kNoTrans);
  
  CuMatrix<double> params_temp(OutputDim(), InputDim() + 1);
  params_temp.AddMatTp(1.0, params, kNoTrans,
                       invert ? in_C_inv_ : in_C_,
                       transpose_in, 0.0);
  
  MatrixTransposeType transpose_out = (is_gradient ? kNoTrans : kTrans);
  params.AddTpMat(1.0, invert ? out_C_inv_ : out_C_, transpose_out,
                  params_temp, kNoTrans, 0.0);
  
  // OK, we've done transforming the parameters or gradients.
  
  // Copy the "params" back to "component".
  component->linear_params_.CopyFromMat(
      params.Range(0, OutputDim(), 0, InputDim()));
  component->bias_params_.CopyColFromMat(params,
                                         InputDim());  
}


void AffineComponentA::Precondition(
    const PreconditionConfig &config,
    AffineComponent *component) {
  
  if (!config.do_precondition) return; // There is nothing to do in this case.
  // (this option will probably only be used for testing.)
  
  KALDI_ASSERT(component != NULL);

  if (inv_fisher_in_.NumRows() == 0) { // Need to pre-compute some things.
    double tot_count = input_scatter_(InputDim(), InputDim());
    // This equals the total count, because for each frame the last
    // element of the extended input vector is 1.
    ComputePreconditioner(input_scatter_, config, tot_count, &inv_fisher_in_);
    ComputePreconditioner(output_scatter_, config, tot_count, &inv_fisher_out_);
  }

  // "params" are the parameters of "component" that we'll be changing.
  // Get them as a single matrix.
  CuMatrix<double> params(OutputDim(), InputDim() + 1);
  params.Range(0, OutputDim(), 0, InputDim()).CopyFromMat(
      component->linear_params_);
  params.CopyColFromVec(CuVector<double>(component->bias_params_),
                        InputDim());
  
  CuMatrix<double> params_temp(OutputDim(), InputDim() + 1);
  params_temp.AddMatSp(1.0, params, kNoTrans, inv_fisher_in_, 0.0);
  
  params.AddSpMat(1.0, inv_fisher_out_, params_temp, kNoTrans, 0.0);
  
  // OK, we've done transforming the parameters or gradients.
  // Copy the "params" back to "component".
  component->linear_params_.CopyFromMat(
      params.Range(0, OutputDim(), 0, InputDim()));
  component->bias_params_.CopyColFromMat(params,
                                         InputDim());  
}

} // namespace nnet2
} // namespace kaldi


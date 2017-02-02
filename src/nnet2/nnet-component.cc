// nnet2/nnet-component.cc

// Copyright 2011-2012  Karel Vesely
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//                2014  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen

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
#include "nnet2/nnet-component.h"
#include "nnet2/nnet-precondition.h"
#include "nnet2/nnet-precondition-online.h"
#include "util/stl-utils.h"
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
  } else if (component_type == "LogSoftmaxComponent") {
    ans = new LogSoftmaxComponent();
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
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
  } else if (component_type == "AffineComponentPreconditioned") {
    ans = new AffineComponentPreconditioned();
  } else if (component_type == "AffineComponentPreconditionedOnline") {
    ans = new AffineComponentPreconditionedOnline();
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
  } else if (component_type == "Convolutional1dComponent") {
    ans = new Convolutional1dComponent();
  } else if (component_type == "MaxpoolingComponent") {
    ans = new MaxpoolingComponent();
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


void NonlinearComponent::SetDim(int32 dim) {
  KALDI_ASSERT(dim > 0);
  dim_ = dim;
  value_sum_.Resize(dim);
  deriv_sum_.Resize(dim);
  count_ = 0.0;
}

void NonlinearComponent::UpdateStats(const CuMatrixBase<BaseFloat> &out_value,
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
  ExpectToken(is, binary, "<ValueSum>");
  value_sum_.Read(is, binary);
  ExpectToken(is, binary, "<DerivSum>");
  deriv_sum_.Read(is, binary);
  ExpectToken(is, binary, "<Count>");
  ReadBasicType(is, binary, &count_);
  ExpectToken(is, binary, ostr_end.str());
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
  KALDI_ASSERT(input_dim_ % output_dim_ == 0);
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


void MaxoutComponent::Propagate(const ChunkInfo &in_info,
                                const ChunkInfo &out_info,
                                const CuMatrixBase<BaseFloat> &in,
                                CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
  out->GroupMax(in);
}

void MaxoutComponent::Backprop(const ChunkInfo &, // in_info,
                               const ChunkInfo &, // out_info,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               Component *to_update,
                               CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);
  in_deriv->GroupMaxDeriv(in_value, out_value);
  in_deriv->MulRowsGroupMat(out_deriv);
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
  KALDI_ASSERT(input_dim_ % output_dim_ == 0);
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


void PnormComponent::Propagate(const ChunkInfo &in_info,
                               const ChunkInfo &out_info,
                               const CuMatrixBase<BaseFloat> &in,
                               CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  out->GroupPnorm(in, p_);
}

void PnormComponent::Backprop(const ChunkInfo &,  // in_info,
                              const ChunkInfo &,  // out_info,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              Component *to_update,
                                // may be identical to "this".
                              CuMatrix<BaseFloat> *in_deriv) const  {
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);
  in_deriv->DiffGroupPnorm(in_value, out_value, out_deriv, p_);
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

void NormalizeComponent::Propagate(const ChunkInfo &in_info,
                                   const ChunkInfo &out_info,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const  {
  cu::NormalizePerRow(in, BaseFloat(1), false, out);
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

void NormalizeComponent::Backprop(const ChunkInfo &,  // in_info,
                                  const ChunkInfo &,  // out_info,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *to_update,
                                    // may be identical to "this".
                                  CuMatrix<BaseFloat> *in_deriv) const  {
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

void SigmoidComponent::Propagate(const ChunkInfo &in_info,
                                 const ChunkInfo &out_info,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  out->Sigmoid(in);
}

void SigmoidComponent::Backprop(const ChunkInfo &,  //in_info,
                                const ChunkInfo &,  //out_info,
                                const CuMatrixBase<BaseFloat> &,  //in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *to_update, // may be identical to "this".
                                CuMatrix<BaseFloat> *in_deriv) const  {
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


void TanhComponent::Propagate(const ChunkInfo &in_info,
                              const ChunkInfo &out_info,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const  {
  // Apply tanh function to each element of the output...
  // the tanh function may be written as -1 + ( 2 / (1 + e^{-2 x})),
  // which is a scaled and shifted sigmoid.

  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
  out->Tanh(in);
}

void TanhComponent::Backprop(const ChunkInfo &, //in_info,
                             const ChunkInfo &, //out_info,
                             const CuMatrixBase<BaseFloat> &, //in_value,
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             Component *to_update, // may be identical to "this".
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

void PowerComponent::Propagate(const ChunkInfo &in_info,
                               const ChunkInfo &out_info,
                               const CuMatrixBase<BaseFloat> &in,
                               CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  // Apply power operation to each element of the input...
  out->CopyFromMat(in);
  out->ApplyPowAbs(power_);
}

void PowerComponent::Backprop(const ChunkInfo &,  //in_info,
                              const ChunkInfo &,  //out_info,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              Component *to_update, // may be identical to "this".
                              CuMatrix<BaseFloat> *in_deriv) const  {
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

void RectifiedLinearComponent::Propagate(const ChunkInfo &in_info,
                                         const ChunkInfo &out_info,
                                         const CuMatrixBase<BaseFloat> &in,
                                         CuMatrixBase<BaseFloat> *out) const  {
  // Apply rectified linear function (x >= 0 ? 1.0 : 0.0)
  out->CopyFromMat(in);
  out->ApplyFloor(0.0);
}

void RectifiedLinearComponent::Backprop(const ChunkInfo &,  //in_info,
                                        const ChunkInfo &,  //out_info,
                                        const CuMatrixBase<BaseFloat> &,  //in_value,
                                        const CuMatrixBase<BaseFloat> &out_value,
                                        const CuMatrixBase<BaseFloat> &out_deriv,
                                        Component *to_update, // may be identical to "this".
                                        CuMatrix<BaseFloat> *in_deriv) const  {

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

void SoftHingeComponent::Propagate(const ChunkInfo &in_info,
                                   const ChunkInfo &out_info,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
  // Apply function x = log(1 + exp(x))
  out->SoftHinge(in);
}

void SoftHingeComponent::Backprop(const ChunkInfo &,  //in_info,
                                  const ChunkInfo &,  //out_info,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *to_update, // may be identical to "this".
                                  CuMatrix<BaseFloat> *in_deriv) const  {

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


void ScaleComponent::Propagate(const ChunkInfo &in_info,
                               const ChunkInfo &out_info,
                               const CuMatrixBase<BaseFloat> &in,
                               CuMatrixBase<BaseFloat> *out) const  {
  out->CopyFromMat(in);
  out->Scale(scale_);
}

void ScaleComponent::Backprop(const ChunkInfo &,  //in_info,
                              const ChunkInfo &,  //out_info,
                              const CuMatrixBase<BaseFloat> &,  //in_value,
                              const CuMatrixBase<BaseFloat> &,  //out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              Component *, //to_update, // may be identical to "this".
                              CuMatrix<BaseFloat> *in_deriv) const  {

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

void SoftmaxComponent::Propagate(const ChunkInfo &in_info,
                                 const ChunkInfo &out_info,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).

  out->ApplySoftMaxPerRow(in);

  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);
}

void SoftmaxComponent::Backprop(const ChunkInfo &in_info,
                                const ChunkInfo &out_info,
                                const CuMatrixBase<BaseFloat> &,  //in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *to_update, // only thing updated is counts_.
                                CuMatrix<BaseFloat> *in_deriv) const  {
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
  in_deriv->DiffSoftmaxPerRow(out_value, out_deriv);

  // The SoftmaxComponent does not have any real trainable parameters, but
  // during the backprop we store some statistics on the average counts;
  // these may be used in mixing-up.
  if (to_update != NULL) {
    NonlinearComponent *to_update_nonlinear =
        dynamic_cast<NonlinearComponent*>(to_update);
    to_update_nonlinear->UpdateStats(out_value);
  }
}

void LogSoftmaxComponent::Propagate(const ChunkInfo &in_info,
                                    const ChunkInfo &out_info,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  // Applies log softmax function to each row of the output. For each row, we do
  // x_i = x_i - log(sum_j exp(x_j))
  out->ApplyLogSoftMaxPerRow(in);

  // Just to be consistent with SoftmaxComponent::Propagate()
  out->ApplyFloor(Log(1.0e-20));
}

void LogSoftmaxComponent::Backprop(const ChunkInfo &in_info,
                                   const ChunkInfo &out_info,
                                   const CuMatrixBase<BaseFloat> &,  //in_value,
                                   const CuMatrixBase<BaseFloat> &out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *to_update,
                                   CuMatrix<BaseFloat> *in_deriv) const  {
  /*
    Let the output be y, then
      y_i = x_i - log(sum_i exp(x_i))
    where x_i is the input to the component. The Jacobian matrix of this
    function is
      J = I - 1 exp(y^T)
    where 1 is a vector of ones. Let the derivative vector at the output be e,
    and at the input be d, then we have
      d = e - exp(y) Sum(e)
      d_i = e_i - exp(y_i) Sum(e)
  */
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  KALDI_ASSERT(SameDim(out_value, out_deriv) && SameDim(out_value, *in_deriv));

  in_deriv->DiffLogSoftmaxPerRow(out_value, out_deriv);

  // Updates stats.
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

// virtual
void AffineComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
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


void AffineComponent::Propagate(const ChunkInfo &in_info,
                                const ChunkInfo &out_info,
                                const CuMatrixBase<BaseFloat> &in,
                                CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  // No need for asserts as they'll happen within the matrix operations.
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

void AffineComponent::Backprop(const ChunkInfo &, //in_info,
                               const ChunkInfo &, //out_info,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, //out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               Component *to_update_in, // may be identical to "this".
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

Component *AffineComponent::CollapseWithNext(
    const FixedScaleComponent &next_component) const {
  KALDI_ASSERT(this->OutputDim() == next_component.InputDim());
  AffineComponent *ans =
      dynamic_cast<AffineComponent*>(this->Copy());
  KALDI_ASSERT(ans != NULL);
  ans->linear_params_.MulRowsVec(next_component.scales_);
  ans->bias_params_.MulElements(next_component.scales_);

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


// virtual
void AffineComponentPreconditionedOnline::Resize(
    int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 1 && output_dim > 1);
  if (rank_in_ >= input_dim) rank_in_ = input_dim - 1;
  if (rank_out_ >= output_dim) rank_out_ = output_dim - 1;
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
  OnlinePreconditioner temp;
  preconditioner_in_ = temp;
  preconditioner_out_ = temp;
  SetPreconditionerConfigs();
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
  BaseFloat tot_change_norm = learning_rate_scale * learning_rate_ * prod_sum,
      max_change_norm = max_change_per_sample_ * minibatch_size;
  // tot_change_norm is the product of norms that we are trying to limit
  // to max_value_.
  KALDI_ASSERT(tot_change_norm - tot_change_norm == 0.0 && "NaN in backprop");
  KALDI_ASSERT(tot_change_norm >= 0.0);
  if (tot_change_norm <= max_change_norm) return 1.0;
  else {
    BaseFloat factor = max_change_norm / tot_change_norm;
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

void BlockAffineComponent::Propagate(const ChunkInfo &in_info,
                                     const ChunkInfo &out_info,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

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

void BlockAffineComponent::Backprop(const ChunkInfo &,  //in_info,
                                    const ChunkInfo &,  //out_info,
                                    const CuMatrixBase<BaseFloat> &in_value,
                                    const CuMatrixBase<BaseFloat> &,  //out_value,
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    Component *to_update_in,
                                    CuMatrix<BaseFloat> *in_deriv) const  {

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

void PermuteComponent::Propagate(const ChunkInfo &in_info,
                                 const ChunkInfo &out_info,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  std::vector<int32> reverse_reorder(reorder_.size());
  for (size_t i = 0; i < reorder_.size(); i++)
    reverse_reorder[reorder_[i]] = i;
  // Note: if we were actually using this component type we could make the
  // CuArray a member variable for efficiency.
  CuArray<int32> cu_reverse_reorder(reverse_reorder);
  out->CopyCols(in, cu_reverse_reorder);
}

void PermuteComponent::Backprop(const ChunkInfo &,  //in_info,
                                const ChunkInfo &,  //out_info,
                                const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *to_update,
                                CuMatrix<BaseFloat> *in_deriv) const  {
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  KALDI_ASSERT(out_deriv.NumCols() == OutputDim());
  // Note: if we were actually using this component type we could make the
  // CuArray a member variable for efficiency.
  CuArray<int32> cu_reorder(reorder_);
  in_deriv->CopyCols(out_deriv, cu_reorder);
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

  std::string token;
  ReadToken(is, binary, &token);
  if (!(token == "<SumGroupComponent>" ||
        token == "</SumGroupComponent>")) {
    KALDI_ERR << "Expected </SumGroupComponent>, got " << token;
  }
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
  WriteToken(os, binary, "</SumGroupComponent>");
}

void SumGroupComponent::Propagate(const ChunkInfo &in_info,
                                  const ChunkInfo &out_info,
                                  const CuMatrixBase<BaseFloat> &in,
                                  CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  out->SumColumnRanges(in, indexes_);
}

void SumGroupComponent::Backprop(const ChunkInfo &in_info,
                                 const ChunkInfo &out_info,
                                 const CuMatrixBase<BaseFloat> &, //in_value,
                                 const CuMatrixBase<BaseFloat> &, //out_value,
                                 const CuMatrixBase<BaseFloat> &out_deriv,
                                 Component *to_update, // may be identical to "this".
                                 CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  in_deriv->CopyCols(out_deriv, reverse_indexes_);
}


std::string SpliceComponent::Info() const {
  std::stringstream stream;
  std::ostringstream os;
  std::copy(context_.begin(), context_.end(),
            std::ostream_iterator<int32>(os, " "));
  stream << Component::Info() << ", context=" << os.str();
  if (const_component_dim_ != 0)
    stream << ", const_component_dim=" << const_component_dim_;

  return stream.str();
}

void SpliceComponent::Init(int32 input_dim, std::vector<int32> context,
                           int32 const_component_dim) {
  input_dim_ = input_dim;
  const_component_dim_ = const_component_dim;
  context_ = context;
  KALDI_ASSERT(context_.size() > 0);
  KALDI_ASSERT(input_dim_ > 0 && context_.front() <= 0 && context_.back() >= 0);
  KALDI_ASSERT(IsSortedAndUniq(context));
  KALDI_ASSERT(const_component_dim_ >= 0 && const_component_dim_ < input_dim_);
}


// e.g. args == "input-dim=10 left-context=2 right-context=2
void SpliceComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim, left_context, right_context;
  std::vector <int32> context;
  bool in_dim_ok = ParseFromString("input-dim", &args, &input_dim);
  bool context_ok = ParseFromString("context", &args, &context);
  bool left_right_context_ok = ParseFromString("left-context", &args,
                                               &left_context) &&
                               ParseFromString("right-context", &args,
                                               &right_context);
  int32 const_component_dim = 0;
  ParseFromString("const-component-dim", &args, &const_component_dim);

  if (!(in_dim_ok && (context_ok || left_right_context_ok)) ||
      !args.empty() || input_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  if (left_right_context_ok)  {
    KALDI_ASSERT(context.size() == 0);
    for (int32 i = -left_context; i <= right_context; i++)
      context.push_back(i);
  }
  Init(input_dim, context, const_component_dim);
}

int32 SpliceComponent::OutputDim() const {
  return (input_dim_  - const_component_dim_)
      * (context_.size())
      + const_component_dim_;
}

int32 ChunkInfo::GetIndex(int32 offset) const  {
  if (offsets_.empty()) {  // if data is contiguous
    KALDI_ASSERT((offset <= last_offset_) && (offset >= first_offset_));
    return offset - first_offset_;
  } else  {
    std::vector<int32>::const_iterator iter =
        std::lower_bound(offsets_.begin(), offsets_.end(), offset);
    // make sure offset is present in the vector
    KALDI_ASSERT(iter != offsets_.end() && *iter == offset);
    return static_cast<int32>(iter - offsets_.begin());
  }
}

int32 ChunkInfo::GetOffset(int32 index) const {
  if (offsets_.empty()) { // if data is contiguous
    int32 offset = index + first_offset_;  // just offset by the first_offset_
    KALDI_ASSERT((offset <= last_offset_) && (offset >= first_offset_));
    return offset;
  } else  {
    KALDI_ASSERT((index >= 0) && (index < offsets_.size()));
    return offsets_[index];
  }
}

void ChunkInfo::Check() const {
  // Checking sanity of the ChunkInfo object
  KALDI_ASSERT((feat_dim_ > 0) && (num_chunks_ > 0));

  if (! offsets_.empty()) {
    KALDI_ASSERT((first_offset_ == offsets_.front()) &&
                 (last_offset_ == offsets_.back()));
  } else  {
    KALDI_ASSERT((first_offset_ >= 0) && (last_offset_ >= first_offset_));
    // asserting the chunk is not contiguous, as offsets is not empty
    KALDI_ASSERT ( last_offset_ - first_offset_ + 1 > offsets_.size() );
  }
  KALDI_ASSERT(NumRows() % num_chunks_ == 0);

}

void ChunkInfo::CheckSize(const CuMatrixBase<BaseFloat> &mat) const {
  KALDI_ASSERT((mat.NumRows()  ==  NumRows()) && (mat.NumCols() == NumCols()));
}

/*
 * This method was used for debugging, make changes in nnet-component.h to
 * expose it
void ChunkInfo::ToString() const  {
    KALDI_LOG << "feat_dim  " << feat_dim_;
    KALDI_LOG << "num_chunks  " << num_chunks_;
    KALDI_LOG << "first_index  " << first_offset_;
    KALDI_LOG << "last_index  " << last_offset_;
    for (size_t i = 0; i < offsets_.size(); i++)
      KALDI_LOG << offsets_[i];
}
*/


void SpliceComponent::Propagate(const ChunkInfo &in_info,
                                const ChunkInfo &out_info,
                                const CuMatrixBase<BaseFloat> &in,
                                CuMatrixBase<BaseFloat> *out) const  {

  // Check the inputs are correct and resize output
  in_info.Check();
  out_info.Check();
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  int32 in_chunk_size  = in_info.ChunkSize(),
        out_chunk_size = out_info.ChunkSize(),
        input_dim = in_info.NumCols();

  if (out_chunk_size <= 0)
    KALDI_ERR << "Splicing features: output will have zero dimension. "
              << "Probably a code error.";

  // 'indexes' is, for each index from 0 to context_.size() - 1,
  // then for each row of "out", the corresponding row of "in" that we copy from
  int32 num_splice = context_.size();
  std::vector<std::vector<int32> > indexes(num_splice);
  for (int32 c = 0; c < num_splice; c++)
    indexes[c].resize(out->NumRows());
  // const_component_dim_ != 0, "const_indexes" will be used to determine which
  // row of "in" we copy the last part of each row of "out" from (this part is
  // not subject to splicing, it's assumed constant for each frame of "input".
  int32 const_dim = const_component_dim_;
  std::vector<int32> const_indexes(const_dim == 0 ? 0 : out->NumRows());

  for (int32 chunk = 0; chunk < in_info.NumChunks(); chunk++) {
    if (chunk == 0) {
      // this branch could be used for all chunks in the matrix,
      // but is restricted to chunk 0 for efficiency reasons
      for (int32 c = 0; c < num_splice; c++) {
        for (int32 out_index = 0; out_index < out_chunk_size; out_index++) {
          int32 out_offset = out_info.GetOffset(out_index);
          int32 in_index = in_info.GetIndex(out_offset + context_[c]);
          indexes[c][chunk * out_chunk_size + out_index] =
              chunk * in_chunk_size + in_index;
        }
      }
    } else {  // just copy the indices from the previous chunk
              // and offset these by input chunk size
     for (int32 c = 0; c < num_splice; c++) {
       for (int32 out_index = 0; out_index < out_chunk_size; out_index++) {
         int32 last_value = indexes[c][(chunk-1) * out_chunk_size + out_index];
         indexes[c][chunk * out_chunk_size + out_index] =
             (last_value == -1 ? -1 : last_value + in_chunk_size);
       }
     }
   }
    if (const_dim != 0) {
      for (int32 out_index = 0; out_index < out_chunk_size; out_index++)
        const_indexes[chunk * out_chunk_size + out_index] =
            chunk * in_chunk_size + out_index;  // there is
      // an arbitrariness here; since we assume the const_component
      // is constant within a chunk, it doesn't matter from where we copy.
    }
  }


  for (int32 c = 0; c < num_splice; c++) {
    int32 dim = input_dim - const_dim;  // dimension we
    // are splicing
    CuSubMatrix<BaseFloat> in_part(in, 0, in.NumRows(),
                                   0, dim),
        out_part(*out, 0, out->NumRows(),
                 c * dim, dim);
    CuArray<int32> cu_indexes(indexes[c]);
    out_part.CopyRows(in_part, cu_indexes);
  }
  if (const_dim != 0) {
    CuSubMatrix<BaseFloat> in_part(in, 0, in.NumRows(),
                                   in.NumCols() - const_dim, const_dim),
        out_part(*out, 0, out->NumRows(),
                 out->NumCols() - const_dim, const_dim);

    CuArray<int32> cu_const_indexes(const_indexes);
    out_part.CopyRows(in_part, cu_const_indexes);
  }
}

void SpliceComponent::Backprop(const ChunkInfo &in_info,
                               const ChunkInfo &out_info,
                               const CuMatrixBase<BaseFloat> &,  // in_value,
                               const CuMatrixBase<BaseFloat> &,  // out_value,
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               Component *to_update,
                               CuMatrix<BaseFloat> *in_deriv) const {
  in_info.Check();
  out_info.Check();
  out_info.CheckSize(out_deriv);
  in_deriv->Resize(in_info.NumRows(), in_info.NumCols(), kUndefined);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
  int32 num_chunks = in_info.NumChunks();
  // rewrite backpropagate

  int32 out_chunk_size = out_info.ChunkSize(),
         in_chunk_size = in_info.ChunkSize(),
            output_dim = out_deriv.NumCols(),
             input_dim = InputDim();

  KALDI_ASSERT(OutputDim() == output_dim);

  int32 num_splice = context_.size(),
      const_dim = const_component_dim_;
  // 'indexes' is, for each index from 0 to num_splice - 1,
  // then for each row of "in_deriv", the corresponding row of "out_deriv" that
  // we add, or -1 if.

  std::vector<std::vector<int32> > indexes(num_splice);
  // const_dim != 0, "const_indexes" will be used to determine which
  // row of "in" we copy the last part of each row of "out" from (this part is
  // not subject to splicing, it's assumed constant for each frame of "input".
  std::vector<int32> const_indexes(const_dim == 0 ? 0 : in_deriv->NumRows(), -1);

  for (int32 c = 0; c < indexes.size(); c++)
    indexes[c].resize(in_deriv->NumRows(), -1);  // set to -1 by default,
  // this gets interpreted by the CopyRows() code
  // as a signal to zero the output...

  int32 dim = input_dim - const_dim;  // dimension we are splicing
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    if (chunk == 0) { // this branch can be taken for all chunks, but is not
                      // taken for efficiency reasons
      for (int32 c = 0; c < num_splice; c++)  {
        for (int32 out_index = 0; out_index < out_chunk_size; out_index++) {
          int32 out_offset = out_info.GetOffset(out_index);
          int32 in_index = in_info.GetIndex(out_offset + context_[c]);
          indexes[c][chunk * in_chunk_size + in_index] =
              chunk * out_chunk_size + out_index;
        }
      }
    } else {  // just copy the indexes from the previous chunk
      for (int32 c = 0; c < num_splice; c++)  {
        for (int32 in_index = 0; in_index < in_chunk_size; in_index++) {
          int32 last_value = indexes[c][(chunk-1) * in_chunk_size + in_index];
          indexes[c][chunk * in_chunk_size + in_index] =
              (last_value == -1 ? -1 : last_value + out_chunk_size);
        }
      }
    }
    // this code corresponds to the way the forward propagation works; see
    // comments there.
    if (const_dim != 0) {
      for (int32 out_index = 0; out_index < out_chunk_size; out_index++)  {
        const_indexes[chunk * in_chunk_size + out_index] =
            chunk * out_chunk_size + out_index;
      }
    }
  }

  CuMatrix<BaseFloat> temp_mat(in_deriv->NumRows(), dim, kUndefined);

  for (int32 c = 0; c < num_splice; c++) {
    CuArray<int32> cu_indexes(indexes[c]);
    int32 dim = input_dim - const_dim;  // dimension we
    // are splicing
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_deriv.NumRows(),
                                          c * dim, dim),
        in_deriv_part(*in_deriv, 0, in_deriv->NumRows(),
                      0, dim);
    if (c == 0) {
      in_deriv_part.CopyRows(out_deriv_part, cu_indexes);
    } else {
      temp_mat.CopyRows(out_deriv_part, cu_indexes);
      in_deriv_part.AddMat(1.0, temp_mat);
    }
  }
  if (const_dim != 0) {
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_deriv.NumRows(),
                                          out_deriv.NumCols() - const_dim,
                                          const_dim),
        in_deriv_part(*in_deriv, 0, in_deriv->NumRows(),
                      in_deriv->NumCols() - const_dim, const_dim);
    CuArray<int32> cu_const_indexes(const_indexes);
    in_deriv_part.CopyRows(out_deriv_part, cu_const_indexes);
  }
}

Component *SpliceComponent::Copy() const {
  SpliceComponent *ans = new SpliceComponent();
  ans->input_dim_ = input_dim_;
  ans->context_ = context_;
  ans->const_component_dim_ = const_component_dim_;
  return ans;
}

void SpliceComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SpliceComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  std::string token;
  ReadToken(is, false, &token);
  if (token == "<LeftContext>") {
    int32 left_context=0, right_context=0;
    std::vector<int32> context;
    ReadBasicType(is, binary, &left_context);
    ExpectToken(is, binary, "<RightContext>");
    ReadBasicType(is, binary, &right_context);
    for (int32 i = -1 * left_context; i <= right_context; i++)
      context.push_back(i);
    context_ = context;
  } else  if (token == "<Context>") {
    ReadIntegerVector(is, binary, &context_);
  } else  {
    KALDI_ERR << "Unknown token" << token
              << ", the model might be corrupted";
  }
  ExpectToken(is, binary, "<ConstComponentDim>");
  ReadBasicType(is, binary, &const_component_dim_);
  ExpectToken(is, binary, "</SpliceComponent>");
}

void SpliceComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SpliceComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<Context>");
  WriteIntegerVector(os, binary, context_);
  WriteToken(os, binary, "<ConstComponentDim>");
  WriteBasicType(os, binary, const_component_dim_);
  WriteToken(os, binary, "</SpliceComponent>");
}


std::string SpliceMaxComponent::Info() const {
  std::stringstream stream;
  std::ostringstream os;
  std::copy(context_.begin(), context_.end(),
            std::ostream_iterator<int32>(os, " "));
  stream << Component::Info() << ", context=" << os.str();
  return stream.str();
}

void SpliceMaxComponent::Init(int32 dim,
                              std::vector<int32> context)  {
  dim_ = dim;
  context_ = context;
  KALDI_ASSERT(dim_ > 0 && context_.front() <= 0 && context_.back() >= 0);
}


// e.g. args == "dim=10 left-context=2 right-context=2
void SpliceMaxComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim, left_context, right_context;
  std::vector <int32> context;
  bool dim_ok = ParseFromString("dim", &args, &dim);
  bool context_ok = ParseFromString("context", &args, &context);
  bool left_right_context_ok = ParseFromString("left-context",
                                               &args, &left_context) &&
                               ParseFromString("right-context", &args,
                                               &right_context);

  if (!(dim_ok && (context_ok || left_right_context_ok)) ||
      !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  if (left_right_context_ok)  {
    KALDI_ASSERT(context.size() == 0);
    for (int32 i = -1 * left_context; i <= right_context; i++)
      context.push_back(i);
  }
  Init(dim, context);
}


void SpliceMaxComponent::Propagate(const ChunkInfo &in_info,
                                   const ChunkInfo &out_info,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const  {
  in_info.Check();
  out_info.Check();
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
  int32 in_chunk_size  = in_info.ChunkSize(),
        out_chunk_size = out_info.ChunkSize(),
        dim = in_info.NumCols();

  CuMatrix<BaseFloat> input_chunk_part(out_chunk_size, dim);
  for (int32 chunk = 0; chunk < in_info.NumChunks(); chunk++) {
    CuSubMatrix<BaseFloat> input_chunk(in,
                                     chunk * in_chunk_size, in_chunk_size,
                                     0, dim),
                        output_chunk(*out,
                                     chunk * out_chunk_size,
                                     out_chunk_size, 0, dim);
    for (int32 offset = 0; offset < context_.size(); offset++) {
      // computing the indices to copy into input_chunk_part from input_chunk
      // copy the rows of the input matrix which correspond to the current
      // context index
      std::vector<int32> input_chunk_inds(out_chunk_size);
      for (int32 i = 0; i < out_chunk_size; i++) {
        int32 out_chunk_ind  = i;
        int32 out_chunk_offset =
            out_info.GetOffset(out_chunk_ind);
        input_chunk_inds[i] =
            in_info.GetIndex(out_chunk_offset + context_[offset]);
      }
      CuArray<int32> cu_chunk_inds(input_chunk_inds);
      input_chunk_part.CopyRows(input_chunk, cu_chunk_inds);
      if (offset == 0)  {
        output_chunk.CopyFromMat(input_chunk_part);
      } else {
        output_chunk.Max(input_chunk_part);
      }
    }
  }
}

void SpliceMaxComponent::Backprop(const ChunkInfo &in_info,
                                  const ChunkInfo &out_info,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &,  // out_value
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *to_update,
                                  CuMatrix<BaseFloat> *in_deriv) const  {
  in_info.Check();
  out_info.Check();
  in_info.CheckSize(in_value);
  out_info.CheckSize(out_deriv);
  in_deriv->Resize(in_info.NumRows(), in_info.NumCols());
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  int32 out_chunk_size = out_info.ChunkSize(),
         in_chunk_size = in_info.ChunkSize(),
                      dim = out_deriv.NumCols();

  KALDI_ASSERT(dim == InputDim());

  for (int32 chunk = 0; chunk < in_info.NumChunks(); chunk++) {
    CuSubMatrix<BaseFloat> in_deriv_chunk(*in_deriv,
                                        chunk * in_chunk_size,
                                        in_chunk_size,
                                        0, dim),
                         in_value_chunk(in_value,
                                        chunk * in_chunk_size,
                                        in_chunk_size,
                                        0, dim),
                        out_deriv_chunk(out_deriv,
                                        chunk * out_chunk_size,
                                        out_chunk_size,
                                        0, dim);
    for (int32 r = 0; r < out_deriv_chunk.NumRows(); r++) {
      int32 out_chunk_ind = r;
      int32 out_chunk_offset =
          out_info.GetOffset(out_chunk_ind);

      for (int32 c = 0; c < dim; c++) {
        int32 in_r_max = -1;
        BaseFloat max_input = -std::numeric_limits<BaseFloat>::infinity();
        for (int32 context_ind = 0;
             context_ind < context_.size(); context_ind++) {
          int32 in_r =
              in_info.GetIndex(out_chunk_offset + context_[context_ind]);
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
  ans->Init(dim_, context_);
  return ans;
}

void SpliceMaxComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SpliceMaxComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  std::string token;
  ReadToken(is, false, &token);
  if (token == "<LeftContext>") {
    int32 left_context = 0, right_context = 0;
    std::vector<int32> context;
    ReadBasicType(is, binary, &left_context);
    ExpectToken(is, binary, "<RightContext>");
    ReadBasicType(is, binary, &right_context);
    for (int32 i = -1 * left_context; i <= right_context; i++)
      context.push_back(i);
    context_ = context;
  } else  if (token == "<Context>") {
    ReadIntegerVector(is, binary, &context_);
  } else  {
    KALDI_ERR << "Unknown token" << token << ", the model might be corrupted";
  }
  ExpectToken(is, binary, "</SpliceMaxComponent>");
}

void SpliceMaxComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SpliceMaxComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Context>");
  WriteIntegerVector(os, binary, context_);
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
  KALDI_ASSERT(dct_dim >= dct_keep_dim_);
  dim_ = dim;
  dct_mat_.Resize(dct_keep_dim_, dct_dim);
  reorder_ = reorder;
  Matrix<BaseFloat> dct_mat(dct_keep_dim_, dct_dim);
  ComputeDctMatrix(&dct_mat);
  dct_mat_ = dct_mat;
}



void DctComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim, dct_dim, dct_keep_dim = 0;
  bool reorder = false;

  bool ok = ParseFromString("dim", &args, &dim);
  ok = ParseFromString("dct-dim", &args, &dct_dim) && ok;
  ok = ParseFromString("reorder", &args, &reorder) && ok;
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

void DctComponent::Propagate(const ChunkInfo &in_info,
                             const ChunkInfo &out_info,
                             const CuMatrixBase<BaseFloat> &in,
                             CuMatrixBase<BaseFloat> *out) const  {
  KALDI_ASSERT(in.NumCols() == InputDim());
  int32 dct_dim = dct_mat_.NumCols(),
        dct_keep_dim = dct_mat_.NumRows(),
        num_rows = in.NumRows(),
        num_chunks = dim_ / dct_dim;

  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(num_rows == out_info.NumRows());
  KALDI_ASSERT(num_chunks * dct_keep_dim == out_info.NumCols());

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

void DctComponent::Backprop(const ChunkInfo &,  //in_info,
                            const ChunkInfo &,  //out_info,
                            const CuMatrixBase<BaseFloat> &,  //in_value,
                            const CuMatrixBase<BaseFloat> &,  //out_value,
                            const CuMatrixBase<BaseFloat> &out_deriv,
                            Component *,  //to_update,
                            CuMatrix<BaseFloat> *in_deriv) const  {
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

void FixedLinearComponent::Propagate(const ChunkInfo &in_info,
                                     const ChunkInfo &out_info,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  out->AddMatMat(1.0, in, kNoTrans, mat_, kTrans, 0.0);
}

void FixedLinearComponent::Backprop(const ChunkInfo &,  //in_info,
                                    const ChunkInfo &,  //out_info,
                                    const CuMatrixBase<BaseFloat> &,  //in_value,
                                    const CuMatrixBase<BaseFloat> &,  //out_value,
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    Component *,  //to_update, // may be identical to "this".
                                    CuMatrix<BaseFloat> *in_deriv) const  {
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

void FixedAffineComponent::Propagate(const ChunkInfo &in_info,
                                     const ChunkInfo &out_info,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 0.0);
  out->AddVecToRows(1.0, bias_params_);
}

void FixedAffineComponent::Backprop(const ChunkInfo &,  //in_info,
                                    const ChunkInfo &,  //out_info,
                                    const CuMatrixBase<BaseFloat> &,  //in_value,
                                    const CuMatrixBase<BaseFloat> &,  //out_value,
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    Component *,  //to_update, // may be identical to "this".
                                    CuMatrix<BaseFloat> *in_deriv) const  {
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
      scales_stddev = std::sqrt(VecVec(scales_, scales_) / scales_size
       - (scales_mean * scales_mean));
  stream << Component::Info() << ", scales-mean=" << scales_mean
         << ", scales-stddev=" << scales_stddev;
  return stream.str();
}

void FixedScaleComponent::Propagate(const ChunkInfo &in_info,
                                    const ChunkInfo &out_info,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const  {
  out->CopyFromMat(in);
  out->MulColsVec(scales_);
}

void FixedScaleComponent::Backprop(const ChunkInfo &, //in_info,
                                   const ChunkInfo &, //out_info,
                                   const CuMatrixBase<BaseFloat> &, //in_value,
                                   const CuMatrixBase<BaseFloat> &, //out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *, //to_update, // may be identical to "this".
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

void FixedBiasComponent::Propagate(const ChunkInfo &in_info,
                                   const ChunkInfo &out_info,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const  {
  out->CopyFromMat(in);
  out->AddVecToRows(1.0, bias_, 1.0);
}

void FixedBiasComponent::Backprop(const ChunkInfo &, //in_info,
                                  const ChunkInfo &, //out_info,
                                  const CuMatrixBase<BaseFloat> &,  //in_value,
                                  const CuMatrixBase<BaseFloat> &,  //out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *,  //to_update,
                                  CuMatrix<BaseFloat> *in_deriv) const  {
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

void DropoutComponent::Propagate(const ChunkInfo &in_info,
                                 const ChunkInfo &out_info,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
  KALDI_ASSERT(in.NumCols() == this->InputDim());

  BaseFloat dp = dropout_proportion_;
  KALDI_ASSERT(dp < 1.0 && dp >= 0.0);
  KALDI_ASSERT(dropout_scale_ <= 1.0 && dropout_scale_ >= 0.0);

  BaseFloat low_scale = dropout_scale_,
      high_scale = (1.0 - (dp * low_scale)) / (1.0 - dp),
      average = (low_scale * dp) +
                (high_scale * (1.0 - dp));
  KALDI_ASSERT(fabs(average - 1.0) < 0.01);

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

void DropoutComponent::Backprop(const ChunkInfo &,  //in_info,
                                const ChunkInfo &,  //out_info,
                                const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *,  //to_update
                                CuMatrix<BaseFloat> *in_deriv) const  {
  KALDI_ASSERT(SameDim(in_value, out_value) && SameDim(in_value, out_deriv));
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->SetMatMatDivMat(out_deriv, out_value, in_value);
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

void AdditiveNoiseComponent::Propagate(const ChunkInfo &in_info,
                                       const ChunkInfo &out_info,
                                       const CuMatrixBase<BaseFloat> &in,
                                       CuMatrixBase<BaseFloat> *out) const  {
  KALDI_ASSERT(in.NumCols() == this->InputDim());
  out->CopyFromMat(in);
  CuMatrix<BaseFloat> rand(in.NumRows(), in.NumCols());
  const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&rand);
  out->AddMat(stddev_, rand);
}

Convolutional1dComponent::Convolutional1dComponent():
    UpdatableComponent(),
    patch_dim_(0), patch_step_(0), patch_stride_(0),
    appended_conv_(false), is_gradient_(false) {}

Convolutional1dComponent::Convolutional1dComponent(const Convolutional1dComponent &component):
    UpdatableComponent(component),
    filter_params_(component.filter_params_),
    bias_params_(component.bias_params_),
    appended_conv_(component.appended_conv_),
    is_gradient_(component.is_gradient_) {}

Convolutional1dComponent::Convolutional1dComponent(const CuMatrixBase<BaseFloat> &filter_params,
                                                   const CuVectorBase<BaseFloat> &bias_params,
                                                   BaseFloat learning_rate):
    UpdatableComponent(learning_rate),
    filter_params_(filter_params),
    bias_params_(bias_params) {
  KALDI_ASSERT(filter_params.NumRows() == bias_params.Dim() &&
               bias_params.Dim() != 0);
  appended_conv_ = false;
  is_gradient_ = false;
}

// aquire input dim
int32 Convolutional1dComponent::InputDim() const {
  int32 filter_dim = filter_params_.NumCols();
  int32 num_splice = filter_dim / patch_dim_;
  return patch_stride_ * num_splice;
}

// aquire output dim
int32 Convolutional1dComponent::OutputDim() const {
  int32 num_filters = filter_params_.NumRows();
  int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
  return num_patches * num_filters;
}

// initialize the component using hyperparameters
void Convolutional1dComponent::Init(BaseFloat learning_rate,
                                    int32 input_dim, int32 output_dim,
                                    int32 patch_dim, int32 patch_step,
                                    int32 patch_stride, BaseFloat param_stddev,
                                    BaseFloat bias_stddev, bool appended_conv) {
  UpdatableComponent::Init(learning_rate);
  patch_dim_ = patch_dim;
  patch_step_ = patch_step;
  patch_stride_ = patch_stride;
  appended_conv_ = appended_conv;
  int32 num_splice = input_dim / patch_stride;
  int32 filter_dim = num_splice * patch_dim;
  int32 num_patches = 1 + (patch_stride - patch_dim) / patch_step;
  int32 num_filters = output_dim / num_patches;
  KALDI_ASSERT(input_dim % patch_stride == 0);
  KALDI_ASSERT((patch_stride - patch_dim) % patch_step == 0);
  KALDI_ASSERT(output_dim % num_patches == 0);

  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
  KALDI_ASSERT(param_stddev >= 0.0 && bias_stddev >= 0.0);
  filter_params_.SetRandn();
  filter_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

// initialize the component using predefined matrix file
void Convolutional1dComponent::Init(BaseFloat learning_rate, int32 patch_dim,
                                    int32 patch_step, int32 patch_stride,
                                    std::string matrix_filename,
                                    bool appended_conv) {
  UpdatableComponent::Init(learning_rate);
  patch_dim_ = patch_dim;
  patch_step_ = patch_step;
  patch_stride_ = patch_stride;
  appended_conv_ = appended_conv;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat);
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 filter_dim = mat.NumCols() - 1, num_filters = mat.NumRows();
  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
  filter_params_.CopyFromMat(mat.Range(0, num_filters, 0, filter_dim));
  bias_params_.CopyColFromMat(mat, filter_dim);
}

// resize the component, setting the parameters to zero, while
// leaving any other configuration values the same
void Convolutional1dComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  int32 num_splice = input_dim / patch_stride_;
  int32 filter_dim = num_splice * patch_dim_;
  int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
  int32 num_filters = output_dim / num_patches;
  KALDI_ASSERT(input_dim % patch_stride_ == 0);
  KALDI_ASSERT((patch_stride_ - patch_dim_) % patch_step_ == 0);
  KALDI_ASSERT(output_dim % num_patches == 0);
  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
}

// display information about component
std::string Convolutional1dComponent::Info() const {
  std::stringstream stream;
  BaseFloat filter_params_size = static_cast<BaseFloat>(filter_params_.NumRows())
                                 * static_cast<BaseFloat>(filter_params_.NumCols());
  BaseFloat filter_stddev =
            std::sqrt(TraceMatMat(filter_params_, filter_params_, kTrans) /
                      filter_params_size),
            bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                                    bias_params_.Dim());

  int32 num_splice = InputDim() / patch_stride_;
  int32 filter_dim = num_splice * patch_dim_;
  int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
  int32 num_filters = OutputDim() / num_patches;

  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", num-splice=" << num_splice
         << ", num-patches=" << num_patches
         << ", num-filters=" << num_filters
         << ", filter-dim=" << filter_dim
         << ", filter-params-stddev=" << filter_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", appended-conv=" << appended_conv_
         << ", learning-rate=" << LearningRate();
  return stream.str();
}

// initialize the component using configuration file
void Convolutional1dComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true, appended_conv = false;
  BaseFloat learning_rate = learning_rate_;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  int32 patch_dim = -1, patch_step = -1, patch_stride = -1;
  ParseFromString("learning-rate", &args, &learning_rate);
  ParseFromString("appended-conv", &args, &appended_conv);
  ok = ok && ParseFromString("patch-dim", &args, &patch_dim);
  ok = ok && ParseFromString("patch-step", &args, &patch_step);
  ok = ok && ParseFromString("patch-stride", &args, &patch_stride);
  if (ParseFromString("matrix", &args, &matrix_filename)) {
    // initialize from prefined parameter matrix
    Init(learning_rate, patch_dim, patch_step, patch_stride,
         matrix_filename, appended_conv);
    if (ParseFromString("input-dim", &args, &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
               "input-dim mismatch vs. matrix.");
    if (ParseFromString("output-dim", &args, &output_dim))
            KALDI_ASSERT(output_dim == OutputDim() &&
                     "output-dim mismatch vs. matrix.");
  } else {
    // initialize from configuration
    ok = ok && ParseFromString("input-dim", &args, &input_dim);
    ok = ok && ParseFromString("output-dim", &args, &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim), bias_stddev = 1.0;
    ParseFromString("param-stddev", &args, &param_stddev);
    ParseFromString("bias-stddev", &args, &bias_stddev);
    Init(learning_rate, input_dim, output_dim, patch_dim,
         patch_step, patch_stride, param_stddev, bias_stddev, appended_conv);
  }
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: " << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
}

// propagation function

/*
   In Convolution1dComponent, filter is defined $num-filters x $filter-dim,
   and bias vector B is defined by length $num-filters. The propatation is
   Y = X o A' + B
   where "o" is executing matrix-matrix convolution, which consists of a group
   of vector-matrix convolutions.
   For instance, the convolution of X(t) and the i-th filter A(i) is
   Y(t,i) = X(t) o A'(i) + B(i)
   The convolution used here is valid convolution. Meaning that the
   output of M o N is of dim |M| - |N| + 1, assuming M is not shorter then N.

   By default, input is arranged by
   x (time), y (channel), z(frequency)
   and output is arranged by
   x (time), y (frequency), z(channel).
   When appending convolutional1dcomponent, appended_conv_ should be
   set ture for the appended convolutional1dcomponent.
*/
void Convolutional1dComponent::Propagate(const ChunkInfo &in_info,
                                         const ChunkInfo &out_info,
                                         const CuMatrixBase<BaseFloat> &in,
                                         CuMatrixBase<BaseFloat> *out) const {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

  // dims
  int32 num_splice = InputDim() / patch_stride_;
  int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
  int32 num_filters = filter_params_.NumRows();
  int32 num_frames = in.NumRows();
  int32 filter_dim = filter_params_.NumCols();

  /** Buffer of reshaped inputs:
   *  1row = vectorized rectangular feature patches
   *  1col = dim over speech frames,
   */
  CuMatrix<BaseFloat> patches(num_frames, filter_dim * num_patches, kUndefined);
  // column_map is indexed by the column-index of "patches",
  // and the value is the corresponding column-index of "in".
  std::vector<int32> column_map(filter_dim * num_patches);

  // build-up a column selection map
  for (int32 patch = 0, index = 0; patch < num_patches; patch++) {
    int32 fstride = patch * patch_step_;
    for (int32 splice = 0; splice < num_splice; splice++) {
      int32 cstride = splice * patch_stride_;
      for (int32 d = 0; d < patch_dim_; d++, index++) {
        if (appended_conv_)
          column_map[index] = (fstride + d) * num_splice + splice;
        else
          column_map[index] = fstride + cstride + d;
      }
    }
  }
  CuArray<int32> cu_cols(column_map);
  patches.CopyCols(in, cu_cols);

  //
  // compute filter activations
  //

  std::vector<CuSubMatrix<BaseFloat>* > tgt_batch, patch_batch, filter_params_batch;

  CuSubMatrix<BaseFloat>* filter_params_elem = new CuSubMatrix<BaseFloat>(
      filter_params_, 0, filter_params_.NumRows(), 0, filter_params_.NumCols());

  // form batch in vector container
  for (int32 p = 0; p < num_patches; p++) {
    // form batch in vector container. for filter_params_batch, all elements
    // point to the same copy filter_params_elem
    tgt_batch.push_back(new CuSubMatrix<BaseFloat>(out->ColRange(p * num_filters,
                                                                 num_filters)));
    patch_batch.push_back(new CuSubMatrix<BaseFloat>(
        patches.ColRange(p * filter_dim, filter_dim)));
    filter_params_batch.push_back(filter_params_elem);

    tgt_batch[p]->AddVecToRows(1.0, bias_params_, 0.0); // add bias
  }

  // apply all filters
  AddMatMatBatched<BaseFloat>(1.0, tgt_batch, patch_batch, kNoTrans,
                              filter_params_batch, kTrans, 1.0);

  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < num_patches; p++) {
    delete tgt_batch[p];
    delete patch_batch[p];
  }
}

// scale the parameters
void Convolutional1dComponent::Scale(BaseFloat scale) {
  filter_params_.Scale(scale);
  bias_params_.Scale(scale);
}

// add another convolution component
void Convolutional1dComponent::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const Convolutional1dComponent *other =
      dynamic_cast<const Convolutional1dComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  filter_params_.AddMat(alpha, other->filter_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

/*
 This function does an operation similar to reversing a map,
 except it handles maps that are not one-to-one by outputting
 the reversed map as a vector of lists.
 @param[in] forward_indexes is a vector of int32, each of whose
            elements is between 0 and input_dim - 1.
 @param[in] input_dim. See definitions of forward_indexes and
            backward_indexes.
 @param[out] backward_indexes is a vector of dimension input_dim
            of lists, The list at (backward_indexes[i]) is a list
            of all indexes j such that forward_indexes[j] = i.
*/
void Convolutional1dComponent::ReverseIndexes(const std::vector<int32> &forward_indexes,
                                              int32 input_dim,
                                              std::vector<std::vector<int32> > *backward_indexes) {
  int32 i, size = forward_indexes.size();
  int32 reserve_size = 2 + size / input_dim;
  backward_indexes->resize(input_dim);
  std::vector<std::vector<int32> >::iterator iter = backward_indexes->begin(),
    end = backward_indexes->end();
  for (; iter != end; ++iter)
    iter->reserve(reserve_size);
  for (int32 j = 0; j < forward_indexes.size(); j++) {
    i = forward_indexes[j];
    KALDI_ASSERT(i < input_dim);
    (*backward_indexes)[i].push_back(j);
  }
}

/*
 This function transforms a vector of lists into a list of vectors,
 padded with -1.
 @param[in] The input vector of lists. Let in.size() be D, and let
            the longest list length (i.e. the max of in[i].size()) be L.
 @param[out] The output list of vectors. The length of the list will
            be L, each vector-dimension will be D (i.e. out[i].size() == D),
            and if in[i] == j, then for some k we will have that
            out[k][j] = i. The output vectors are padded with -1
            where necessary if not all the input lists have the same side.
*/
void Convolutional1dComponent::RearrangeIndexes(const std::vector<std::vector<int32> > &in,
                                                std::vector<std::vector<int32> > *out) {
  int32 D = in.size();
  int32 L = 0;
  for (int32 i = 0; i < D; i++)
    if (in[i].size() > L)
      L = in[i].size();
  out->resize(L);
  for (int32 i = 0; i < L; i++)
    (*out)[i].resize(D, -1);
  for (int32 i = 0; i < D; i++) {
    for (int32 j = 0; j < in[i].size(); j++) {
      (*out)[j][i] = in[i][j];
    }
  }
}

// back propagation function
void Convolutional1dComponent::Backprop(const ChunkInfo &in_info,
                                        const ChunkInfo &out_info,
                                        const CuMatrixBase<BaseFloat> &in_value,
                                        const CuMatrixBase<BaseFloat> &out_value,
                                        const CuMatrixBase<BaseFloat> &out_deriv,
                                        Component *to_update_in,
                                        CuMatrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  Convolutional1dComponent *to_update = dynamic_cast<Convolutional1dComponent*>(to_update_in);
  int32 num_splice = InputDim() / patch_stride_;
  int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
  int32 num_filters = filter_params_.NumRows();
  int32 num_frames = out_deriv.NumRows();
  int32 filter_dim = filter_params_.NumCols();

  /** Buffer for backpropagation:
   *  derivatives in the domain of 'patches_',
   *  1row = vectorized rectangular feature patches,
   *  1col = dim over speech frames,
   */
  CuMatrix<BaseFloat> patches_deriv(num_frames, filter_dim * num_patches, kSetZero);

  //
  // backpropagate to vector of matrices
  // (corresponding to position of a filter)
  //
  std::vector<CuSubMatrix<BaseFloat>* > patch_deriv_batch, out_deriv_batch,
      filter_params_batch;

  CuSubMatrix<BaseFloat>* filter_params_elem = new CuSubMatrix<BaseFloat>(
      filter_params_, 0, filter_params_.NumRows(), 0, filter_params_.NumCols());

  // form batch in vector container
  for (int32 p = 0; p < num_patches; p++) {
    // form batch in vector container. for filter_params_batch, all elements
    // point to the same copy filter_params_elem
    patch_deriv_batch.push_back(new CuSubMatrix<BaseFloat>(patches_deriv.ColRange(
        p * filter_dim, filter_dim)));
    out_deriv_batch.push_back(new CuSubMatrix<BaseFloat>(out_deriv.ColRange(
        p * num_filters, num_filters)));
    filter_params_batch.push_back(filter_params_elem);
  }
  AddMatMatBatched<BaseFloat>(1.0, patch_deriv_batch, out_deriv_batch, kNoTrans,
                              filter_params_batch, kNoTrans, 0.0);

  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < num_patches; p++) {
    delete patch_deriv_batch[p];
    delete out_deriv_batch[p];
  }

  // sum the derivatives into in_deriv
  std::vector<int32> column_map(filter_dim * num_patches);
  for (int32 patch = 0, index = 0; patch < num_patches; patch++) {
    int32 fstride = patch * patch_step_;
    for (int32 splice = 0; splice < num_splice; splice++) {
      int32 cstride = splice * patch_stride_;
      for (int32 d = 0; d < patch_dim_; d++, index++) {
        if (appended_conv_)
          column_map[index] = (fstride + d) * num_splice + splice;
        else
          column_map[index] = fstride + cstride + d;
      }
    }
  }
  std::vector<std::vector<int32> > reversed_column_map;
  ReverseIndexes(column_map, InputDim(), &reversed_column_map);
  std::vector<std::vector<int32> > rearranged_column_map;
  RearrangeIndexes(reversed_column_map, &rearranged_column_map);
  for (int32 p = 0; p < rearranged_column_map.size(); p++) {
    CuArray<int32> cu_cols(rearranged_column_map[p]);
    in_deriv->AddCols(patches_deriv, cu_cols);
  }

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    to_update->Update(in_value, out_deriv);
  }
}

void Convolutional1dComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
  }
  filter_params_.SetZero();
  bias_params_.SetZero();
  if (treat_as_gradient) {
    is_gradient_ = true;
  }
}

void Convolutional1dComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<Convolutional1dComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</Convolutional1dComponent>"
  // might not see the "<Convolutional1dComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<PatchDim>");
  ReadBasicType(is, binary, &patch_dim_);
  ExpectToken(is, binary, "<PatchStep>");
  ReadBasicType(is, binary, &patch_step_);
  ExpectToken(is, binary, "<PatchStride>");
  ReadBasicType(is, binary, &patch_stride_);
  // back-compatibility
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<AppendedConv>") {
    ReadBasicType(is, binary, &appended_conv_);
    ExpectToken(is, binary, "<FilterParams>");
  } else {
    appended_conv_ = false;
    KALDI_ASSERT(tok == "<FilterParams>");
  }
  filter_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ReadToken(is, binary, &tok);
  if (tok == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ExpectToken(is, binary, ostr_end.str());
  } else {
    is_gradient_ = false;
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void Convolutional1dComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<Convolutional1dComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</Convolutional1dComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<PatchDim>");
  WriteBasicType(os, binary, patch_dim_);
  WriteToken(os, binary, "<PatchStep>");
  WriteBasicType(os, binary, patch_step_);
  WriteToken(os, binary, "<PatchStride>");
  WriteBasicType(os, binary, patch_stride_);
  WriteToken(os, binary, "<AppendedConv>");
  WriteBasicType(os, binary, appended_conv_);
  WriteToken(os, binary, "<FilterParams>");
  filter_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, ostr_end.str());
}

BaseFloat Convolutional1dComponent::DotProduct(const UpdatableComponent &other_in) const {
  const Convolutional1dComponent *other =
      dynamic_cast<const Convolutional1dComponent*>(&other_in);
  return TraceMatMat(filter_params_, other->filter_params_, kTrans)
         + VecVec(bias_params_, other->bias_params_);
}

Component* Convolutional1dComponent::Copy() const {
  Convolutional1dComponent *ans = new Convolutional1dComponent();
  ans->learning_rate_ = learning_rate_;
  ans->patch_dim_ = patch_dim_;
  ans->patch_step_ = patch_step_;
  ans->patch_stride_ = patch_stride_;
  ans->filter_params_ = filter_params_;
  ans->bias_params_ = bias_params_;
  ans->appended_conv_ = appended_conv_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

void Convolutional1dComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_filter_params(filter_params_);
  temp_filter_params.SetRandn();
  filter_params_.AddMat(stddev, temp_filter_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

void Convolutional1dComponent::SetParams(const VectorBase<BaseFloat> &bias,
                                         const MatrixBase<BaseFloat> &filter) {
  bias_params_ = bias;
  filter_params_ = filter;
  KALDI_ASSERT(bias_params_.Dim() == filter_params_.NumRows());
}

int32 Convolutional1dComponent::GetParameterDim() const {
  return (filter_params_.NumCols() + 1) * filter_params_.NumRows();
}

// update parameters
void Convolutional1dComponent::Update(const CuMatrixBase<BaseFloat> &in_value,
                                      const CuMatrixBase<BaseFloat> &out_deriv) {
  // useful dims
  int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
  int32 num_filters = filter_params_.NumRows();
  int32 filter_dim = filter_params_.NumCols();
  int32 num_frames = in_value.NumRows();
  int32 num_splice = InputDim() / patch_stride_;
  CuMatrix<BaseFloat> filters_grad;
  CuVector<BaseFloat> bias_grad;

  /** Buffer of reshaped inputs:
   *  1row = vectorized rectangular feature patches
   *  1col = dim over speech frames,
   */
  CuMatrix<BaseFloat> patches(num_frames, filter_dim * num_patches, kUndefined);
  std::vector<int32> column_map(filter_dim * num_patches);
  for (int32 patch = 0, index = 0; patch < num_patches; patch++) {
    int32 fstride = patch * patch_step_;
    for (int32 splice = 0; splice < num_splice; splice++) {
      int32 cstride = splice * patch_stride_;
      for (int32 d = 0; d < patch_dim_; d++, index++) {
        if (appended_conv_)
          column_map[index] = (fstride + d) * num_splice + splice;
        else
          column_map[index] = fstride + cstride + d;
      }
    }
  }
  CuArray<int32> cu_cols(column_map);
  patches.CopyCols(in_value, cu_cols);

  //
  // calculate the gradient
  //
  filters_grad.Resize(num_filters, filter_dim, kSetZero); // reset
  bias_grad.Resize(num_filters, kSetZero); // reset

  //
  // use all the patches
  //

  // create a single large matrix holding the smaller matrices
  // from the vector container filters_grad_batch along the rows
  CuMatrix<BaseFloat> filters_grad_blocks_batch(
      num_patches * filters_grad.NumRows(), filters_grad.NumCols());

  std::vector<CuSubMatrix<BaseFloat>* > filters_grad_batch, diff_patch_batch,
      patch_batch;
  for (int32 p = 0; p < num_patches; p++) {
    // form batch in vector container
    filters_grad_batch.push_back(new CuSubMatrix<BaseFloat>(
        filters_grad_blocks_batch.RowRange(
            p * filters_grad.NumRows(),
            filters_grad.NumRows())));
    diff_patch_batch.push_back(new CuSubMatrix<BaseFloat>(out_deriv.ColRange(
        p * num_filters, num_filters)));
    patch_batch.push_back(new CuSubMatrix<BaseFloat>(patches.ColRange(
        p * filter_dim, filter_dim)));
  }

  AddMatMatBatched<BaseFloat>(1.0, filters_grad_batch, diff_patch_batch,
                              kTrans, patch_batch, kNoTrans, 1.0);

  // add the row blocks together to filters_grad
  filters_grad.AddMatBlocks(1.0, filters_grad_blocks_batch);

  // create a matrix holding the col blocks sum of out_deriv
  CuMatrix<BaseFloat> out_deriv_col_blocks_sum(out_deriv.NumRows(), num_filters);

  // add the col blocks together to out_deriv_col_blocks_sum
  out_deriv_col_blocks_sum.AddMatBlocks(1.0, out_deriv);

  bias_grad.AddRowSumMat(1.0, out_deriv_col_blocks_sum, 1.0);

  // release memory
  for (int32 p = 0; p < num_patches; p++) {
    delete filters_grad_batch[p];
    delete diff_patch_batch[p];
    delete patch_batch[p];
  }

  //
  // update
  //
  filter_params_.AddMat(learning_rate_, filters_grad);
  bias_params_.AddVec(learning_rate_, bias_grad);
}

void MaxpoolingComponent::Init(int32 input_dim, int32 output_dim,
                               int32 pool_size, int32 pool_stride)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  pool_size_ = pool_size;
  pool_stride_ = pool_stride;

  // sanity check
  // number of patches
  KALDI_ASSERT(input_dim_ % pool_stride_ == 0);
  int32 num_patches = input_dim_ / pool_stride_;
  // number of pools
  KALDI_ASSERT(num_patches % pool_size_ == 0);
  int32 num_pools = num_patches / pool_size_;
  // check output dim
  KALDI_ASSERT(output_dim_ == num_pools * pool_stride_);
}

void MaxpoolingComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 input_dim = 0;
  int32 output_dim = 0;
  int32 pool_size = -1, pool_stride = -1;
  bool ok = true;

  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  ok = ok && ParseFromString("pool-size", &args, &pool_size);
  ok = ok && ParseFromString("pool-stride", &args, &pool_stride);

  KALDI_LOG << output_dim << " " << input_dim << " " << ok;
  KALDI_LOG << "Pool: " << pool_size << " "
            << pool_stride << " " << ok;
  if (!ok || !args.empty() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(input_dim, output_dim, pool_size, pool_stride);
}

/*
   Input and output of maxpooling component is arranged as
   x (time), y (frequency), z (channel)
   for efficient pooling.
 */
void MaxpoolingComponent::Propagate(const ChunkInfo &in_info,
                                    const ChunkInfo &out_info,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const  {
  in_info.CheckSize(in);
  out_info.CheckSize(*out);
  KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());
  int32 num_patches = input_dim_ / pool_stride_;
  int32 num_pools = num_patches / pool_size_;

  // do the max-pooling
  for (int32 q = 0; q < num_pools; q++) {
    // get output buffer of the pool
    CuSubMatrix<BaseFloat> pool(out->ColRange(q * pool_stride_, pool_stride_));
    pool.Set(-1e20); // reset a large negative value
    for (int32 r = 0; r < pool_size_; r++) {
      // col-by-col block comparison pool
      int32 p = r + q * pool_size_;
      pool.Max(in.ColRange(p * pool_stride_, pool_stride_));
    }
  }
}

void MaxpoolingComponent::Backprop(const ChunkInfo &, // in_info,
                                   const ChunkInfo &, // out_info,
                                   const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *to_update,
                                   CuMatrix<BaseFloat> *in_deriv) const {
  int32 num_patches = input_dim_ / pool_stride_;
  int32 num_pools = num_patches / pool_size_;
  std::vector<int32> patch_summands(num_patches, 0);
  in_deriv->Resize(in_value.NumRows(), in_value.NumCols(), kSetZero);

  for(int32 q = 0; q < num_pools; q++) {
    for(int32 r = 0; r < pool_size_; r++) {
      int32 p = r + q * pool_size_;
      CuSubMatrix<BaseFloat> in_p(in_value.ColRange(p * pool_stride_, pool_stride_));
      CuSubMatrix<BaseFloat> out_q(out_value.ColRange(q * pool_stride_, pool_stride_));
      CuSubMatrix<BaseFloat> tgt(in_deriv->ColRange(p * pool_stride_, pool_stride_));
      CuMatrix<BaseFloat> src(out_deriv.ColRange(q * pool_stride_, pool_stride_));
      // zero-out mask
      CuMatrix<BaseFloat> mask;
      in_p.EqualElementMask(out_q, &mask);
      src.MulElements(mask);
      tgt.AddMat(1.0, src);
      // summed deriv info
      patch_summands[p] += 1;
    }
  }

  // scale in_deriv of overlaped pools
  for(int32 p = 0; p < num_patches; p++) {
    CuSubMatrix<BaseFloat> tgt(in_deriv->ColRange(p * pool_stride_, pool_stride_));
    KALDI_ASSERT(patch_summands[p] > 0);
    tgt.Scale(1.0 / patch_summands[p]);
  }
}

void MaxpoolingComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MaxpoolingComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "<PoolSize>");
  ReadBasicType(is, binary, &pool_size_);
  ExpectToken(is, binary, "<PoolStride>");
  ReadBasicType(is, binary, &pool_stride_);
  ExpectToken(is, binary, "</MaxpoolingComponent>");
}

void MaxpoolingComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MaxpoolingComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "<PoolSize>");
  WriteBasicType(os, binary, pool_size_);
  WriteToken(os, binary, "<PoolStride>");
  WriteBasicType(os, binary, pool_stride_);
  WriteToken(os, binary, "</MaxpoolingComponent>");
}

std::string MaxpoolingComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim = " << input_dim_
         << ", output-dim = " << output_dim_
         << ", pool-size = " << pool_size_
         << ", pool-stride = " << pool_stride_;
  return stream.str();
}

} // namespace nnet2
} // namespace kaldi

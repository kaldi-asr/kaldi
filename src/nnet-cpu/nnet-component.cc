// nnet/nnet-component.cc

// Copyright 2011-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "nnet-cpu/nnet-component.h"
#include "nnet-cpu/nnet-precondition.h"
#include "util/text-utils.h"
#include "util/kaldi-io.h"

namespace kaldi {

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
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
  } else if (component_type == "AffineComponentA") {
    ans = new AffineComponentA();
  } else if (component_type == "AffineComponentNobias") {
    ans = new AffineComponentNobias();
  } else if (component_type == "AffineComponentPreconditioned") {
    ans = new AffineComponentPreconditioned();
  } else if (component_type == "AffinePreconInputComponent") {
    ans = new AffinePreconInputComponent();
  } else if (component_type == "MixtureProbComponent") {
    ans = new MixtureProbComponent();
  } else if (component_type == "BlockAffineComponent") {
    ans = new BlockAffineComponent();
  } else if (component_type == "PermuteComponent") {
    ans = new PermuteComponent();
  } else if (component_type == "DctComponent") {
    ans = new DctComponent();
  } else if (component_type == "FixedLinearComponent") {
    ans = new FixedLinearComponent();
  } else if (component_type == "SpliceComponent") {
    ans = new SpliceComponent();
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

std::string Component::Info() const {
  std::stringstream stream;
  stream << Type() << " component, inputDim=" << InputDim()
         << ", outputDim=" << OutputDim();
  return stream.str();
}

void NonlinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
  ExpectToken(is, binary, ostr_end.str());  
}

void NonlinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, ostr_end.str());  
}

void NonlinearComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  bool ok = ParseFromString("dim", &args, &dim);
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << orig_args << "\"";
  Init(dim); // calls a virtual function that will generally
  // just set dim, but does more for SoftmaxComponent
  // (sets up counts).
}

void SigmoidComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 Matrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  int32 num_rows = in.NumRows(), num_cols = in.NumCols();
  for(int32 r = 0; r < num_rows; r++) {
    const BaseFloat *in_data = in.RowData(r),
        *in_data_end = in_data + num_cols;
    BaseFloat *out_data = out->RowData(r);
    for (; in_data != in_data_end; ++in_data, ++out_data) {
      if (*in_data > 0.0) {
        *out_data = 1.0 / (1.0 + exp(- *in_data));
      } else { // avoid exponentiating positive number; instead,
        // use 1/(1+exp(-x)) = exp(x) / (exp(x)+1)
        BaseFloat f = exp(*in_data);
        *out_data = f / (f + 1.0);
      }
    }
  }
}

void SigmoidComponent::Backprop(const MatrixBase<BaseFloat> &, // in_value
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *, // to_update
                                Matrix<BaseFloat> *in_deriv) const {
  // we ignore in_value and to_update.

  // The element by element equation would be:
  // in_deriv = out_deriv * out_value * (1.0 - out_value);
  // We can accomplish this via calls to the matrix library.

  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  in_deriv->Set(1.0);
  in_deriv->AddMat(-1.0, out_value);
  // now in_deriv = 1.0 - out_value [element by element]
  in_deriv->MulElements(out_value);
  // now in_deriv = out_value * (1.0 - out_value) [element by element]
  in_deriv->MulElements(out_deriv);
  // now in_deriv = out_deriv * out_value * (1.0 - out_value) [element by element]
}


void TanhComponent::Propagate(const MatrixBase<BaseFloat> &in,
                              int32, // num_chunks
                              Matrix<BaseFloat> *out) const {
  // Apply tanh function to each element of the output...
  // the tanh function may be written as -1 + ( 2 / (1 + e^{-2 x})),
  // which is a scaled and shifted sigmoid.
  out->Resize(in.NumRows(), in.NumCols());
  int32 num_rows = in.NumRows(), num_cols = in.NumCols();
  for(int32 r = 0; r < num_rows; r++) {
    const BaseFloat *in_data = in.RowData(r),
        *in_data_end = in_data + num_cols;
    BaseFloat *out_data = out->RowData(r);
    for (; in_data != in_data_end; ++in_data, ++out_data) {
      if (*in_data > 0.0) {
        *out_data = -1.0 + 2.0 / (1.0 + exp(-2.0 * *in_data));
      } else {
        *out_data = 1.0 - 2.0 / (1.0 + exp(2.0 * *in_data));
      }
    }
  }
}

void TanhComponent::Backprop(const MatrixBase<BaseFloat> &, // in_value
                             const MatrixBase<BaseFloat> &out_value,
                             const MatrixBase<BaseFloat> &out_deriv,
                             int32, // num_chunks
                             Component *, // to_update
                             Matrix<BaseFloat> *in_deriv) const {
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
  in_deriv->Add(1.0); // now in_deriv = (1.0 - out_value^2).
  in_deriv->MulElements(out_deriv);
}  

void SoftmaxComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 Matrix<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).
  *out = in; // Resizes also.
  int32 num_rows = out->NumRows();
  for(int32 r = 0; r < num_rows; r++) {
    SubVector<BaseFloat> row(*out, r);
    row.ApplySoftMax();
  }
}

void SoftmaxComponent::Backprop(const MatrixBase<BaseFloat> &, // in_value
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                int32 num_chunks,
                                Component *to_update, // only thing updated is counts_.
                                Matrix<BaseFloat> *in_deriv) const {
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
  const MatrixBase<BaseFloat> &P(out_value), &E(out_deriv);
  MatrixBase<BaseFloat> &D (*in_deriv);
  
  for (int32 r = 0; r < P.NumRows(); r++) {
    SubVector<BaseFloat> p(P, r), e(E, r), d(D, r);
    d.AddVecVec(1.0, p, e, 0.0); // d_i = p_i e_i.
    BaseFloat pT_e = VecVec(p, e); // p^T e.
    d.AddVec(-pT_e, p); // d_i -= (p^T e) p_i
  }

  // The SoftmaxComponent does not have any real trainable parameters, but
  // during the backprop we store some statistics on the average counts;
  // these may be used in mixing-up.
  if (to_update != NULL) {
    SoftmaxComponent *to_update_softmax =
        dynamic_cast<SoftmaxComponent*>(to_update);
    // The next loop updates the counts_ variable, which is the soft-count of
    // each output dimension.
    int32 chunk_size = out_value.NumRows() / num_chunks;
    KALDI_ASSERT(num_chunks > 0 && chunk_size * num_chunks == out_value.NumRows());
    to_update_softmax->counts_.AddRowSumMat(1.0, out_value, 1.0);
  }
}

void SoftmaxComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SoftmaxComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
  ExpectToken(is, binary, "<Counts>");
  counts_.Read(is, binary);
  ExpectToken(is, binary, "</SoftmaxComponent>");
}

void SoftmaxComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SoftmaxComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<Counts>");
  counts_.Write(os, binary);
  WriteToken(os, binary, "</SoftmaxComponent>");
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
    avg_input_(component.avg_input_),
    avg_input_count_(component.avg_input_count_),
    is_gradient_(component.is_gradient_) { }

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
  if (linear.NumRows() != linear_params_.NumRows()) {
    avg_input_.Resize(linear.NumRows()); // zeroes it.
    avg_input_count_ = 0.0;
  }
  bias_params_ = bias;
  linear_params_ = linear;
  KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
}

void AffineComponent::ZeroOccupancy() {
  avg_input_count_ = 0.0;
  avg_input_.Set(0.0);
}

void AffineComponent::PerturbParams(BaseFloat stddev) {
  Matrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  Vector<BaseFloat> temp_bias_params(bias_params_);
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
  stream << Type() << " component, inputDim=" << InputDim()
         << ", outputDim=" << OutputDim()
         << ", linear-params stddev = " << linear_stddev
         << ", bias-params stddev = " << bias_stddev
         << ", learning rate = " << LearningRate();
  return stream.str();
}

Component* AffineComponent::Copy() const {
  AffineComponent *ans = new AffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->avg_input_ = avg_input_;
  ans->avg_input_count_ = avg_input_count_;
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
                           BaseFloat param_stddev, BaseFloat bias_stddev,
                           bool precondition) {
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  avg_input_.Resize(input_dim);
  avg_input_count_ = 0.0;
}

void AffineComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  bool precondition = false;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
      bias_stddev = 1.0;
  ParseFromString("param-stddev", &args, &param_stddev);
  ParseFromString("bias-stddev", &args, &bias_stddev);
  ParseFromString("precondition", &args, &precondition);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, input_dim, output_dim,
       param_stddev, bias_stddev, precondition);
}


void AffineComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                int32, // num_chunks
                                Matrix<BaseFloat> *out) const {
  // No need for asserts as they'll happen within the matrix operations.
  out->Resize(in.NumRows(), linear_params_.NumRows());
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

// This is an improved, preconditioned update with mean removal
// for the features.
void AffineComponentNobias::Update(
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &out_deriv) {
  
  // The idea with the linear_params is: take an input dimension
  // m and and output dimension n.  Let mu be the mean of this
  // input feature, and f be the feature value.  Let's train as
  // if (f - mu) were the feature.  So in this case the gradient
  // descent rule would be not learning_rate . in_value . output_deriv,
  // but learning_rate . (in_value - mu) . output_deriv.
  // And we'll represent in this model in the normal way, so we
  // need to update the bias parameter to take into account this change
  // in value of our (f - mu) feature.  So the bias term would change as:
  // bias_n += -mu * (learning_rate . (in_value - mu) . output_deriv).
  
  
  Vector<BaseFloat> out_deriv_sum(OutputDim());
  out_deriv_sum.AddRowSumMat(1.0, out_deriv, 0.0);

  Vector<BaseFloat> avg_input(InputDim());
  avg_input.AddRowSumMat(1.0 / in_value.NumRows(), in_value, 0.0);
  BaseFloat avg_input_count = 1.0;
  
  // The following term is already there in the basic update.
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
  // The next term is the new term for updating the linear params,
  // corresponding to (learning_rate . -mu . output_deriv.).  Here,
  // mu is avg_input_(i) / avg_input_count_.
  linear_params_.AddVecVec(-learning_rate_/avg_input_count,
                           out_deriv_sum, avg_input);

  // The next term is the "normal" term for updating the bias parameters.
  bias_params_.AddVec(learning_rate_, out_deriv_sum);   
  
  // Next we handle the expression (from above)
  // bias_weight += -mu * (learning_rate . (in_value - mu) . output_deriv).
  // Here, we do this for each frame, and for each frame it happens
  // with "bias_weight" and "output" deriv both indexed by the same value
  // (the output dim)-- and the parts involving mu and in_value are summed
  // over the input dimensions.
  Vector<BaseFloat> sum(in_value.NumRows());  
  sum.Set(VecVec(avg_input, avg_input) /
          (avg_input_count * avg_input_count));
  // note: below, "sum" in sum(frame) just refers to the variable named "sum".
  // now sum(frame) = \sum_i (-mu(i) *  -mu(i)),
  // where \mu_i is avg_input_(i)/avg_input_count_.
  // the next line does: sum(frame) -= \sum_i mu(i) * input(i), giving
  // sum(frame) = - \sum_i mu(i) * (in_value(i) - mu(i)).
  sum.AddMatVec(-1.0 / avg_input_count,
                in_value, kNoTrans, avg_input, 1.0);
  // The next line updates the bias params with this "correction term":
  // for a scalar, it's doing:
  // bias_weight += -mu * (learning_rate . (in_value - mu) . output_deriv).
  bias_params_.AddMatVec(learning_rate_, out_deriv, kTrans,
                         sum, 1.0);
  if (bias_params_.Max() > 100.0 || bias_params_.Min() < -100.0) {
    KALDI_ERR << "Bias params getting too large.";
  }
}

void AffineComponent::UpdateSimple(const MatrixBase<BaseFloat> &in_value,
                                   const MatrixBase<BaseFloat> &out_deriv) {
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void AffineComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                               const MatrixBase<BaseFloat> &,  // out_value
                               const MatrixBase<BaseFloat> &out_deriv,
                               int32, //  num_chunks
                               Component *to_update_in,
                               Matrix<BaseFloat> *in_deriv) const {
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
    
    if (!to_update->is_gradient_) {
      // Update the stats on the average input. 
      // This is only for diagnostics.  We ignore the chunk weights.
      if (to_update->avg_input_.Dim() != to_update->InputDim())
        to_update->avg_input_.Resize(to_update->InputDim());
      to_update->avg_input_.AddRowSumMat(1.0, in_value, 1.0);
      to_update->avg_input_count_ += in_value.NumRows();

      BaseFloat scale_per_minibatch = 0.95; // Should be configurable, but for now
      // this will do.  Average over ~20 minibatches.
      to_update->avg_input_.Scale(scale_per_minibatch);
      to_update->avg_input_count_ *= scale_per_minibatch;
    }
  }
}

Component* AffineComponentNobias::Copy() const {
  AffineComponentNobias *ans = new AffineComponentNobias();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->avg_input_ = avg_input_;
  ans->avg_input_count_ = avg_input_count_;
  return ans;
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
  ExpectToken(is, binary, "<AvgInput>");
  avg_input_.Read(is, binary);
  ExpectToken(is, binary, "<AvgInputCount>");
  ReadBasicType(is, binary, &avg_input_count_);
  std::string tok;
  // back-compatibility code.  TODO: re-do this later.
  ReadToken(is, binary, &tok);
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
  WriteToken(os, binary, "<AvgInput>");
  avg_input_.Write(os, binary);
  WriteToken(os, binary, "<AvgInputCount>");
  WriteBasicType(os, binary, avg_input_count_);
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

void AffineComponentPreconditioned::Read(std::istream &is, bool binary) {
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
  ExpectToken(is, binary, "<AvgInput>");
  avg_input_.Read(is, binary);
  ExpectToken(is, binary, "<AvgInputCount>");
  ReadBasicType(is, binary, &avg_input_count_);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, ostr_end.str());
}

void AffineComponentPreconditioned::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  bool precondition = false;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
             bias_stddev = 1.0, alpha = 0.1;
  ParseFromString("param-stddev", &args, &param_stddev);
  ParseFromString("bias-stddev", &args, &bias_stddev);
  ParseFromString("precondition", &args, &precondition);
  ParseFromString("alpha", &args, &alpha);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, input_dim, output_dim,
       param_stddev, bias_stddev, precondition, alpha);
}

void AffineComponentPreconditioned::Init(
    BaseFloat learning_rate, 
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev,
    bool precondition, BaseFloat alpha) {
  UpdatableComponent::Init(learning_rate);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  avg_input_.Resize(input_dim);
  avg_input_count_ = 0.0;
  alpha_ = alpha;
  KALDI_ASSERT(alpha > 0.0);
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
  WriteToken(os, binary, "<AvgInput>");
  avg_input_.Write(os, binary);
  WriteToken(os, binary, "<AvgInputCount>");
  WriteBasicType(os, binary, avg_input_count_);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
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
  stream << Type() << " component, inputDim=" << InputDim()
         << ", outputDim=" << OutputDim()
         << ", linear-params stddev = " << linear_stddev
         << ", bias-params stddev = " << bias_stddev
         << ", learning rate = " << LearningRate()
         << ", alpha = " << alpha_;
  return stream.str();
}

Component* AffineComponentPreconditioned::Copy() const {
  AffineComponentPreconditioned *ans = new AffineComponentPreconditioned();
  ans->learning_rate_ = learning_rate_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->avg_input_ = avg_input_;
  ans->avg_input_count_ = avg_input_count_;
  ans->alpha_ = alpha_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

void AffineComponentPreconditioned::Update(
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &out_deriv) {
  Matrix<BaseFloat> in_value_precon(in_value.NumRows(),
                                    in_value.NumCols()),
      out_deriv_precon(out_deriv.NumRows(),
                       out_deriv.NumCols());
  // each row of in_value_precon will be that same row of
  // in_value, but multiplied by the inverse of a Fisher
  // matrix that has been estimated from all the other rows,
  // smoothed by some appropriate amount times the identity
  // matrix (this amount is proportional to \alpha).
  PreconditionDirectionsAlphaRescaled(in_value, alpha_, &in_value_precon);
  PreconditionDirectionsAlphaRescaled(out_deriv, alpha_, &out_deriv_precon);
  
  bias_params_.AddRowSumMat(learning_rate_, out_deriv_precon, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv_precon, kTrans,
                           in_value_precon, kNoTrans, 1.0);
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
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &, // out_value
    const MatrixBase<BaseFloat> &out_deriv,
    int32, //  num_chunks
    Component *to_update_in,
    Matrix<BaseFloat> *in_deriv) const {
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
      Matrix<BaseFloat> in_value_tmp(in_value);
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
      Vector<BaseFloat> &input_precision = to_update->input_precision_;
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
  Matrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  Vector<BaseFloat> temp_bias_params(bias_params_);
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

void BlockAffineComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                     int32, // num_chunks
                                     Matrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), bias_params_.Dim());
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.

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
    SubMatrix<BaseFloat> in_block(in, 0, num_frames,
                                  b * input_block_dim, input_block_dim),
        out_block(*out, 0, num_frames,
                  b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    out_block.AddMatMat(1.0, in_block, kNoTrans, param_block, kTrans, 1.0);
  }
}

void BlockAffineComponent::Backprop(
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &, // out_value
    const MatrixBase<BaseFloat> &out_deriv,
    int32, // num_chunks
    Component *to_update_in,
    Matrix<BaseFloat> *in_deriv) const {
  // This code mirrors the code in Propagate().
  int32 num_frames = in_value.NumRows();
  BlockAffineComponent *to_update = dynamic_cast<BlockAffineComponent*>(
      to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  int32 input_block_dim = linear_params_.NumCols(),
       output_block_dim = linear_params_.NumRows() / num_blocks_;
  KALDI_ASSERT(in_value.NumCols() == input_block_dim * num_blocks_);
  KALDI_ASSERT(out_deriv.NumCols() == output_block_dim * num_blocks_);

  // add the sum of the rows of out_deriv, to the bias_params_.
  if (to_update != NULL)
    to_update->bias_params_.AddRowSumMat(to_update->learning_rate_, out_deriv,
                                         1.0);
  
  for (int32 b = 0; b < num_blocks_; b++) {
    SubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
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
    
    if (to_update != NULL) {
      SubMatrix<BaseFloat> param_block_to_update(
          to_update->linear_params_,
          b * output_block_dim, output_block_dim,
          0, input_block_dim);
      // Update the parameters.
      param_block_to_update.AddMatMat(
          to_update->learning_rate_,
          out_deriv_block, kTrans, in_value_block, kNoTrans, 1.0);
    }
  }  
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

void PermuteComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                 int32, // num_chunks
                                 Matrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), in.NumCols());
  
  int32 num_rows = in.NumRows(), num_cols = in.NumCols();
  for (int32 r = 0; r < num_rows; r++) {
    const BaseFloat *in_data = in.RowData(r);
    BaseFloat *out_data = out->RowData(r);
    for (int32 c = 0; c < num_cols; c++)
      out_data[reorder_[c]] = in_data[c];
  }
}

void PermuteComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *to_update,
                                Matrix<BaseFloat> *in_deriv) const {
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  KALDI_ASSERT(out_deriv.NumCols() == OutputDim());
  
  int32 num_rows = in_deriv->NumRows(), num_cols = in_deriv->NumCols();
  for (int32 r = 0; r < num_rows; r++) {
    const BaseFloat *out_deriv_data = out_deriv.RowData(r);
    BaseFloat *in_deriv_data = in_deriv->RowData(r);
    for (int32 c = 0; c < num_cols; c++)
      in_deriv_data[c] = out_deriv_data[reorder_[c]];
  }
}


void MixtureProbComponent::PerturbParams(BaseFloat stddev) {
  // We need to preserve the sum-to-one constraint when
  // perturbing these parameters.
  for (size_t i = 0; i < params_.size(); i++) {
    Matrix<BaseFloat> params(params_[i]);
    params.ApplyFloor(1.0e-20);
    params.ApplyLog();
    Vector<BaseFloat> col(params.NumRows()),
        rand(params.NumRows());
    for (int32 j = 0; j < params.NumCols(); j++) {
      col.CopyColFromMat(params, j);
      rand.SetRandn();
      col.AddVec(stddev, rand);  // *Perturb the parameters.*
      col.ApplyExp(); // convert back to non-log form.
      KALDI_ASSERT(col.Sum() > 0.0);
      col.Scale(1.0 / col.Sum()); // make it sum to one.
      params.CopyColFromVec(col, j);
    }      
    params_[i].CopyFromMat(params);
  }
}


Component* MixtureProbComponent::Copy() const {
  MixtureProbComponent *ans = new MixtureProbComponent();
  ans->learning_rate_ = learning_rate_;
  ans->params_ = params_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

/// dot product is only possible between parameters and gradients.
BaseFloat MixtureProbComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const MixtureProbComponent *other =
      dynamic_cast<const MixtureProbComponent*>(&other_in);
  BaseFloat ans = 0.0;
  KALDI_ASSERT(params_.size() == other->params_.size());
  if (this->is_gradient_) {
    KALDI_ASSERT(!other->is_gradient_);
    return other_in.DotProduct(*this);
  }
  KALDI_ASSERT(other->is_gradient_ && !this->is_gradient_);
  for (size_t i = 0; i < params_.size(); i++) {
    Matrix<BaseFloat> log_params(params_[i]);
    log_params.ApplyFloor(1.0e-20);
    log_params.ApplyLog();
    ans += TraceMatMat(log_params, other->params_[i], kTrans);
  }
  return ans;
}

void MixtureProbComponent::Scale(BaseFloat scale) {
  for (size_t i = 0; i < params_.size(); i++) {
    if (this->is_gradient_) { // just scale.
      params_[i].Scale(scale);
    } else {
      // scale in log-space.  From its external interface, this class acts like
      // its parameters are stored in log space, although they are not.
      Matrix<BaseFloat> &params(params_[i]);
      params.ApplyFloor(1.0e-20);
      params.ApplyLog();
      params.Scale(scale);  // **scale in log-space.**
      params.ApplyExp();
      // Now re-normalize each column to sum to one.
      Vector<BaseFloat> col(params.NumRows());
      for (int32 c = 0; c < params.NumCols(); c++) {
        col.CopyColFromMat(params, c);
        KALDI_ASSERT(col.Sum() > 0.0);
        col.Scale(1.0 / col.Sum()); // make it sum to one.
        params.CopyColFromVec(col, c);
      }
    }
  }
}

void MixtureProbComponent::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const MixtureProbComponent *other =
      dynamic_cast<const MixtureProbComponent*>(&other_in);
  KALDI_ASSERT(other != NULL && other->is_gradient_ == is_gradient_
               && other->params_.size() == params_.size());

  for (size_t i = 0; i < params_.size(); i++) {
    if (this->is_gradient_) { // just add in the normal way.
      params_[i].AddMat(alpha, other->params_[i]);
    } else {
      KALDI_ASSERT(!other->is_gradient_); // if we need this to work when
      // "other" is a gradient, we'd do it slightly differently; don't support
      // this for now.
      
      // Do the addition in log-space.  From its external interface, this class
      // acts like its parameters are stored in log space, although they are
      // not.
      Matrix<BaseFloat> params(params_[i]), other_params(other->params_[i]);
      params.ApplyFloor(1.0e-20);
      params.ApplyLog();
      other_params.ApplyFloor(1.0e-20);
      other_params.ApplyLog();
      params.AddMat(alpha, other_params);  // **add in log-space.**
      params.ApplyExp();
      // Now re-normalize each column to sum to one.
      Vector<BaseFloat> col(params.NumRows());
      for (int32 c = 0; c < params.NumCols(); c++) {
        col.CopyColFromMat(params, c);
        KALDI_ASSERT(col.Sum() > 0.0);
        col.Scale(1.0 / col.Sum()); // make it sum to one.
        params.CopyColFromVec(col, c);
      }
      params_[i].CopyFromMat(params);
    }
  }
}


void MixtureProbComponent::Init(BaseFloat learning_rate,
                                BaseFloat diag_element,
                                const std::vector<int32> &sizes) {
  UpdatableComponent::Init(learning_rate);
  is_gradient_ = false;
  input_dim_ = 0;
  output_dim_ = 0;
  params_.resize(sizes.size());
  KALDI_ASSERT(diag_element > 0.0 && diag_element <= 1.0);
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
  for (int32 i = 0; i < size; i++) {
    params_[i].Read(is, binary);
    input_dim_ += params_[i].NumCols();
    output_dim_ += params_[i].NumRows();
  }        
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</MixtureProbComponent>");  
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
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</MixtureProbComponent>");  
}

void MixtureProbComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    is_gradient_ = true;
  }
  for (size_t i = 0; i < params_.size(); i++)
    params_[i].SetZero();
}

void MixtureProbComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                     int32, // num_chunks
                                     Matrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == InputDim());
  out->Resize(in.NumRows(), OutputDim());
  
  int32 num_frames = in.NumRows(),
      input_offset = 0,
     output_offset = 0;

  for (size_t i = 0; i < params_.size(); i++) {
    int32 this_input_dim = params_[i].NumCols(), // input dim of this block.
         this_output_dim = params_[i].NumRows();
    KALDI_ASSERT(this_input_dim > 0 && this_output_dim > 0);
    SubMatrix<BaseFloat> in_block(in, 0, num_frames,
                                  input_offset, this_input_dim),
        out_block(*out, 0, num_frames, output_offset, this_output_dim);
    const Matrix<BaseFloat> &param_block(params_[i]);
    out_block.AddMatMat(1.0, in_block, kNoTrans, param_block, kTrans, 0.0);
    input_offset += this_input_dim;
    output_offset += this_output_dim;   
  }
  KALDI_ASSERT(input_offset == InputDim() && output_offset == OutputDim());
}

void MixtureProbComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                    const MatrixBase<BaseFloat> &,// out_value
                                    const MatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *to_update_in,
                                    Matrix<BaseFloat> *in_deriv) const {
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
    SubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        input_offset, this_input_dim),
        in_deriv_block(*in_deriv, 0, num_frames,
                       input_offset, this_input_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        output_offset, this_output_dim);
    const Matrix<BaseFloat> &param_block(params_[i]);
    
    // Propagate gradient back to in_deriv.
    in_deriv_block.AddMatMat(1.0, out_deriv_block, kNoTrans, param_block,
                             kNoTrans, 0.0);
    
    if (to_update != NULL) {
      Matrix<BaseFloat> &param_block_to_update(to_update->params_[i]);
      if (to_update->is_gradient_) { // We're just storing
        // the gradient there-- w.r.t. the unnormalized log params.
        KALDI_ASSERT(to_update->learning_rate_ == 1.0);
        // tmp_mat will be d/d(probs).  We want to store d/d(unnormalized
        // log-probs).
        Matrix<BaseFloat> tmp_mat(param_block_to_update.NumRows(),
                                  param_block_to_update.NumCols());
        tmp_mat.AddMatMat(1.0, out_deriv_block, kTrans, in_value_block,
                          kNoTrans, 0.0);
        // we want param_block_to_update to be d/d(unnormalized log-probs).
        // First multiply each element by the corresponding parameter
        // (which is a probability).  This comes from differentiating the exp.
        tmp_mat.MulElements(this->params_[i]);
        // Now, the sum-to-one constraint is per column of the params.
        // We normalize from the unnormalized log-probs, e.g.
        // p_i = exp(l_i) / \sum_j exp(l_j)
        // and this translates into invariance w.r.t. a constant aded
        // factor on the log-probs, which means there should be no
        // gradient w.r.t. adding to a given column of the matrix,
        // or the sum of the gradients over the column should be zero.
        // So we subtract the average from each column.
        Vector<BaseFloat> tmp_col(param_block_to_update.NumRows()),
            param_col(param_block_to_update.NumRows());
        for (int32 j = 0; j < param_block_to_update.NumCols(); j++) {
          tmp_col.CopyColFromMat(tmp_mat, j);
          param_col.CopyColFromMat(this->params_[i], j);
          // The next line relates to the sum-to-one constraint.
          tmp_col.AddVec(-1.0 * tmp_col.Sum(), param_col);
          tmp_mat.CopyColFromVec(tmp_col, j);
        }
        param_block_to_update.AddMat(1.0, tmp_mat);
      } else {
        /*
          We do gradient descent in the space of log probabilities.  We enforce the
          sum-to-one constraint; this affects the gradient (I think you can derive
          this using lagrangian multipliers).
          
          For a column c of the matrix, we have a gradient g.
          Let l be the vector of unnormalized log-probs of that row; it has an arbitrary
          offset, but we just choose the point where it coincides with correctly normalized
          log-probs, so for each element:
          l_i = log(c_i).
          The functional relationship between l_i and c_i is:
          c_i = exp(l_i) / sum_j exp(l_j) . [softmax function from l to c.]
          Let h_i be the gradient w.r.t l_i.  We can compute this as follows.  The softmax function
          has a Jacobian equal to diag(c) - c c^T.  We have:
          h = (diag(c) - c c^T)  g
          We do the gradient-descent step on h, and when we convert back to c, we renormalize.
          [note: the renormalization would not even be necessary if the step size were infinitesimal;
          it's only needed due to second-order effects which slightly unnormalize each column.]
        */        
        int32 num_rows = this_output_dim, num_cols = this_input_dim;
        Matrix<BaseFloat> gradient(num_rows, num_cols);
        gradient.AddMatMat(1.0, out_deriv_block, kTrans, in_value_block, kNoTrans,
                           0.0);
        for (int32 col = 0; col < num_cols; col++) {
          Vector<BaseFloat> param_col(num_rows);
          param_col.CopyColFromMat(param_block_to_update, col);
          Vector<BaseFloat> log_param_col(param_col);
          log_param_col.ApplyLog(); // note: works even for zero, but may have -inf
          log_param_col.Scale(1.0); // relates to l2 regularization-- applied at log
          // parameter level.
          for (int32 i = 0; i < num_rows; i++)
            if (log_param_col(i) < -1.0e+20)
              log_param_col(i) = -1.0e+20; // get rid of -inf's,as
          // as we're not sure exactly how BLAS will deal with them.
          Vector<BaseFloat> gradient_col(num_rows);
          gradient_col.CopyColFromMat(gradient, col);
          Vector<BaseFloat> log_gradient(num_rows);
          log_gradient.AddVecVec(1.0, param_col, gradient_col, 0.0); // h <-- diag(c) g.
          BaseFloat cT_g = VecVec(param_col, gradient_col);
          log_gradient.AddVec(-cT_g, param_col); // h -= (c^T g) c .
          log_param_col.AddVec(learning_rate_, log_gradient); // Gradient step,
          // in unnormalized log-prob space.      
          log_param_col.ApplySoftMax(); // Go back to probabilities, renormalizing.
          param_block_to_update.CopyColFromVec(log_param_col, col); // Write back.
        }
      }
    }
    input_offset += this_input_dim;
    output_offset += this_output_dim;   
  }
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

void SpliceComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                int32 num_chunks,
                                Matrix<BaseFloat> *out) const {
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
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    SubMatrix<BaseFloat> input_chunk(in,
                                     chunk * input_chunk_size, input_chunk_size,
                                     0, input_dim),
                        output_chunk(*out,
                                     chunk * output_chunk_size, output_chunk_size,
                                     0, output_dim);

    for (int32 c = 0; c < left_context_ + right_context_ + 1; c++) {
      SubMatrix<BaseFloat> input_part(input_chunk, 
                                      c, output_chunk_size,
                                      0, input_dim - const_component_dim_),
                           output_part(output_chunk, 
                                       0, output_chunk_size,
                                       (input_dim - const_component_dim_) * c,
                                       input_dim - const_component_dim_);
      output_part.CopyFromMat(input_part);
    }
    //Append the constant component at the end of the output vector
    if (const_component_dim_ != 0) {
      SubMatrix<BaseFloat> input_part(input_chunk, 
                                      0, output_chunk_size,
                                      InputDim() - const_component_dim_,
                                      const_component_dim_),
                           output_part(output_chunk, 
                                       0, output_chunk_size,
                                       OutputDim() - const_component_dim_,
                                       const_component_dim_);
      output_part.CopyFromMat(input_part);
    }
  }  
}

void SpliceComponent::Backprop(const MatrixBase<BaseFloat> &, // in_value
                               const MatrixBase<BaseFloat> &, // out_value,
                               const MatrixBase<BaseFloat> &out_deriv,
                               int32 num_chunks,
                               Component *to_update, // may == "this".
                               Matrix<BaseFloat> *in_deriv) const {
 
 KALDI_ASSERT(out_deriv.NumRows() > 0 && num_chunks > 0);

  if (out_deriv.NumRows() % num_chunks != 0)
    KALDI_ERR << "Number of chunks " << num_chunks << "does not divide "
              << "number of frames " << out_deriv.NumRows();
  
  int32 output_chunk_size = out_deriv.NumRows() / num_chunks,
         input_chunk_size = output_chunk_size + left_context_ + right_context_,
               output_dim = out_deriv.NumCols(),
                input_dim = InputDim();
 
 KALDI_ASSERT( OutputDim() == output_dim );

  in_deriv->Resize(num_chunks * input_chunk_size, input_dim); // Will zero it.
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    SubMatrix<BaseFloat> in_deriv_chunk(*in_deriv, 
                            chunk * input_chunk_size, input_chunk_size, 
                            0, input_dim),
                        out_deriv_chunk(out_deriv,
                            chunk * output_chunk_size, output_chunk_size,
                            0, output_dim);


    for (int32 c = 0; c < left_context_ + right_context_ + 1; c++) {
      SubMatrix<BaseFloat> in_deriv_part(in_deriv_chunk, 
                                         c, output_chunk_size,
                                         0, input_dim - const_component_dim_),
                          out_deriv_part(out_deriv_chunk, 
                                        0, output_chunk_size,
                                        c * (input_dim - const_component_dim_),
                                         input_dim - const_component_dim_);
      in_deriv_part.AddMat(1.0, out_deriv_part);
    }
    
    if (const_component_dim_ > 0) {
      SubMatrix<BaseFloat> out_deriv_const_part(out_deriv_chunk, 
                              chunk * output_chunk_size, 1,
                              output_dim - const_component_dim_,
                                                const_component_dim_);
                      
      for (int32 c = 0; c < in_deriv_chunk.NumRows(); c++) {
        SubMatrix<BaseFloat> in_deriv_part(in_deriv_chunk, c, 1,
                                           input_dim - const_component_dim_,
                                           const_component_dim_);
        in_deriv_part.CopyFromMat(out_deriv_const_part);
      } 
    }
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
  ComputeDctMatrix(&dct_mat_);
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

void DctComponent::Reorder(MatrixBase<BaseFloat> *mat, bool reverse) const {
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

  Vector<BaseFloat> temp(mat->NumCols());
  for (int32 i = 0; i < mat->NumRows(); i++) {
    SubVector<BaseFloat> row(*mat, i);
    int32 num_blocks_in = block_size_out;
    for (int32 b = 0; b < num_blocks_in; b++) {
      for (int32 j = 0; j < block_size_in; j++) {
        temp(j * block_size_out + b) = row(b * block_size_in + j);
      }
    }
    row.CopyFromVec(temp);
  }
}

void DctComponent::Propagate(const MatrixBase<BaseFloat> &in,
                             int32, // num_chunks
                             Matrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == InputDim());
  
  int32 dct_dim = dct_mat_.NumCols(),
        dct_keep_dim = dct_mat_.NumRows(),
        num_chunks = dim_ / dct_dim,
        num_rows = in.NumRows();
  
  out->Resize(num_rows, num_chunks * dct_keep_dim);

  Matrix<BaseFloat> in_tmp;
  if (reorder_) {
    in_tmp = in;
    Reorder(&in_tmp, false);
  }
  
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    SubMatrix<BaseFloat> in_mat(reorder_ ? in_tmp : in,
                                0, num_rows, dct_dim * chunk, dct_dim),
                        out_mat(*out, 
                                0, num_rows, dct_keep_dim * chunk, dct_keep_dim);

    out_mat.AddMatMat(1.0, in_mat, kNoTrans, dct_mat_, kTrans, 0.0);
  }
  if (reorder_)
    Reorder(out, true);
}

void DctComponent::Backprop(const MatrixBase<BaseFloat>&, // in_value,
                            const MatrixBase<BaseFloat>&, // out_value,
                            const MatrixBase<BaseFloat> &out_deriv,
                            int32, // num_chunks
                            Component*,// to_update
                            Matrix<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(out_deriv.NumCols() == OutputDim());

  int32 dct_dim = dct_mat_.NumCols(),
        dct_keep_dim = dct_mat_.NumRows(),
        num_chunks = dim_ / dct_dim,
        num_rows = out_deriv.NumRows();

  in_deriv->Resize(num_rows, dim_);
  
  Matrix<BaseFloat> out_deriv_tmp;
  if (reorder_) {
    out_deriv_tmp = out_deriv;
    Reorder(&out_deriv_tmp, false);
  }
  for (int32 chunk = 0; chunk < num_chunks; chunk++) {
    SubMatrix<BaseFloat> in_deriv_mat(*in_deriv,
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

std::string FixedLinearComponent::Info() const {
  std::stringstream stream;
  stream << Component::Info() << ", input_dim=" << mat_.NumCols()
         << ", output_dim = " << mat_.NumRows();
  return stream.str();
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
  Matrix<BaseFloat> mat;
  mat.Read(ki.Stream(), binary);
  KALDI_ASSERT(mat.NumRows() != 0);
  Init(mat);
}


void FixedLinearComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                     int32 num_chunks,
                                     Matrix<BaseFloat> *out) const {
  out->Resize(in.NumRows(), mat_.NumRows());
  out->AddMatMat(1.0, in, kNoTrans, mat_, kTrans, 0.0);
}

void FixedLinearComponent::Backprop(const MatrixBase<BaseFloat> &, // in_value
                                    const MatrixBase<BaseFloat> &, // out_value
                                    const MatrixBase<BaseFloat> &out_deriv,
                                    int32, // num_chunks
                                    Component *, // to_update
                                    Matrix<BaseFloat> *in_deriv) const {
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
  WriteToken(os, binary, "<Matrix>");
  mat_.Write(os, binary);
  WriteToken(os, binary, "</FixedLinearComponent>");  
}

void FixedLinearComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<FixedLinearComponent>", "<Matrix>");
  mat_.Read(is, binary);
  ExpectToken(is, binary, "</FixedLinearComponent>");
}

void DropoutComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  int32 dim;
  BaseFloat dropout_proportion = 0.5;
  bool ok = ParseFromString("dim", &args, &dim);
  ParseFromString("dropout-proportion", &args, &dropout_proportion);  
  
  if (!ok || !args.empty() || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type DropoutComponent: \""
              << orig_args << "\"";
  Init(dim, dropout_proportion);
}

void DropoutComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<DropoutComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<DropoutProportion>");
  ReadBasicType(is, binary, &dropout_proportion_);
  ExpectToken(is, binary, "</DropoutComponent>");
}

void DropoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DropoutComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<DropoutProportion>");
  WriteBasicType(os, binary, dropout_proportion_);
  WriteToken(os, binary, "</DropoutComponent>");  
}


void DropoutComponent::Init(int32 dim, BaseFloat dropout_proportion){
  dim_ = dim;
  dropout_proportion_ = dropout_proportion;
}
  
void DropoutComponent::Propagate(
    const MatrixBase<BaseFloat> &in,
    int32 num_chunks,
    Matrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == this->InputDim());
  out->Resize(in.NumRows(), in.NumCols());

  KALDI_ASSERT(dropout_proportion_ < 1.0 && dropout_proportion_ >= 0.0);
  int32 dim = InputDim(), num_keep = static_cast<int32>(dim * dropout_proportion_);
  KALDI_ASSERT(num_keep > 0);
  BaseFloat scale = dim / static_cast<BaseFloat>(num_keep); // scale on the
  // dimensions that we keep.
  Vector<BaseFloat> scales(dim);
  BaseFloat *begin = scales.Data(), *end = begin + dim;
  std::fill(begin, begin + num_keep, scale);
  std::fill(begin + num_keep, end, 0.0);

  out->CopyFromMat(in);
  for (int32 r = 0; r < out->NumRows(); r++) {
    SubVector<BaseFloat> out_row(*out, r);
    std::random_shuffle(begin, end); // get new random ordering of kept components.
    out_row.MulElements(scales);
  }
}

void DropoutComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                int32, // num_chunks
                                Component *, // to_update
                                Matrix<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(SameDim(in_value, out_value) && SameDim(in_value, out_deriv));
  in_deriv->Resize(out_deriv.NumRows(), out_deriv.NumCols());
  for (int32 r = 0; r < in_value.NumRows(); r++) { // each frame...
    for (int32 c = 0; c < in_value.NumCols(); c++) {
      BaseFloat i = in_value(r, c), o = out_value(r, c), od = out_deriv(r, c),
          id;
      if (i != 0.0) {
        id = od * (o / i); /// o / i is either zero or "scale".
      } else {
        id = od; /// Just imagine the scale was 1.0.  This is somehow true in
        /// expectation; anyway, this case should basically never happen so it doesn't
        /// really matter.
      }
      (*in_deriv)(r, c) = id;
    }
  }
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
    const MatrixBase<BaseFloat> &in,
    int32 num_chunks,
    Matrix<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == this->InputDim());
  *out = in;
  Matrix<BaseFloat> rand(in.NumRows(), in.NumCols());
  out->AddMat(stddev_, rand);
}

void AffineComponentA::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<AffineComponentA>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<AvgInput>");
  avg_input_.Read(is, binary);
  ExpectToken(is, binary, "<AvgInputCount>");
  ReadBasicType(is, binary, &avg_input_count_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "<InputScatter>");
  input_scatter_.Read(is, binary);
  ExpectToken(is, binary, "<OutputScatter>");
  output_scatter_.Read(is, binary);  
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
  WriteToken(os, binary, "<AvgInput>");
  avg_input_.Write(os, binary);
  WriteToken(os, binary, "<AvgInputCount>");
  WriteBasicType(os, binary, avg_input_count_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<InputScatter>");
  input_scatter_.Write(os, binary);
  WriteToken(os, binary, "<OutputScatter>");
  output_scatter_.Write(os, binary);  
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
}

void AffineComponentA::Add(BaseFloat alpha, const UpdatableComponent &other_in) {
  const AffineComponentA *other =
      dynamic_cast<const AffineComponentA*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
  input_scatter_.AddSp(alpha, other->input_scatter_);
  output_scatter_.AddSp(alpha, other->output_scatter_);  
}

Component* AffineComponentA::Copy() const {
  // The initializer below will be the one that takes AffineComponent,
  // so we need to take care of the remaining parameters.
  AffineComponentA *ans = new AffineComponentA(*this);
  ans->input_scatter_ = input_scatter_;
  ans->output_scatter_ = output_scatter_;
  return ans;
}

void AffineComponentA::UpdateSimple(
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &out_deriv) {
  KALDI_ASSERT(this->is_gradient_);
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);

  // The rest of this function is about updating the scatters.
  if (input_scatter_.NumRows() != 0) { // scatter is to be accumulated..
    Matrix<double> in_value_dbl(in_value.NumCols() + 1,
                                in_value.NumRows());
    in_value_dbl.Range(0, in_value.NumCols(),
                       0, in_value.NumRows()).CopyFromMat(in_value, kTrans);
    in_value_dbl.Row(in_value.NumCols()).Set(1.0);
    input_scatter_.AddMat2(1.0, in_value_dbl, kNoTrans, 1.0);
  }
  if (output_scatter_.NumRows() != 0) {
    Matrix<double> out_deriv_dbl(out_deriv, kTrans);
    output_scatter_.AddMat2(1.0, out_deriv_dbl, kNoTrans, 1.0);
  }
}

// static
void AffineComponentA::ComputeTransforms(const SpMatrix<double> &scatter_in,
                                         const PreconditionConfig &config,
                                         double tot_count,
                                         TpMatrix<double> *C,
                                         TpMatrix<double> *C_inv) {
  SpMatrix<double> scatter(scatter_in);
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
  Matrix<double> params(OutputDim(), InputDim() + 1);
  params.Range(0, OutputDim(), 0, InputDim()).CopyFromMat(
      component->linear_params_);
  params.CopyColFromVec(Vector<double>(component->bias_params_),
                        InputDim());
  

  MatrixTransposeType transpose_in = (is_gradient ? kTrans : kNoTrans);
  
  Matrix<double> params_temp(OutputDim(), InputDim() + 1);
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


} // namespace kaldi



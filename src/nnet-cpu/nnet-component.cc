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

#include "nnet-cpu/nnet-component.h"
#include "util/text-utils.h"

#include <sstream>

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
  } else if (component_type == "SpliceComponent") {
    ans = new SpliceComponent();
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
  stream << Type() << " component, inputDim=" << InputDim() <<", outputDim=" << OutputDim();
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
                                const VectorBase<BaseFloat> &, //chunk_weights
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
                             const VectorBase<BaseFloat> &, // chunk_weights
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
                                const VectorBase<BaseFloat> &chunk_weights, // chunk_weights
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
  // during the backprop we
  if (to_update) {
    SoftmaxComponent *to_update_softmax =
        dynamic_cast<SoftmaxComponent*>(to_update);
    // The next loop updates the counts_ variable, which is the soft-count of
    // each output dimension.
    int32 num_chunks = chunk_weights.Dim(),
          chunk_size = out_value.NumRows() / num_chunks;
    KALDI_ASSERT(num_chunks > 0 && chunk_size * num_chunks == out_value.NumRows());
    for (int32 chunk = 0; chunk < num_chunks; chunk++) {
      BaseFloat chunk_weight = chunk_weights(chunk) / chunk_size; // the "chunk_weights"
      // variable stores the weighting factor times the number of labeled frames
      // in the chunk, which happens to be convenient when implementing l2
      // regularization with SGD.  Here we want the actual weighting factor on
      // the chunk; typically these weights will be close to one.  Note: this
      // code assumes that this softmax layer doesn't undergo any frame splicing
      // before the output, so the chunk_size equals the number of labels at the
      // output.  This is a safe assumption, as we anticipate we won't be
      // needing this count information in cases where such splicing might be
      // applied.
      SubMatrix<BaseFloat> output_chunk(out_value, chunk * chunk_size, chunk_size,
                                        0, out_value.NumCols());
      to_update_softmax->counts_.AddRowSumMat(chunk_weight, output_chunk, 1.0);
      // Add the sum of the frames in the chunk to the counts, weighted by the
      // chunk weight.
    }
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

void AffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    SetL2Penalty(0.0);
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void AffineComponent::PerturbParams(BaseFloat stddev) {
  Matrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  Vector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

Component* AffineComponent::Copy() const {
  AffineComponent *ans = new AffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->l2_penalty_ = l2_penalty_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  return ans;
}

BaseFloat AffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void AffineComponent::Init(BaseFloat learning_rate, BaseFloat l2_penalty,
                           int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {
  UpdatableComponent::Init(learning_rate, l2_penalty);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(param_stddev);
}

void AffineComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_,
               l2_penalty = l2_penalty_;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("l2-penalty", &args, &l2_penalty); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
  ParseFromString("param-stddev", &args, &param_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, l2_penalty, input_dim, output_dim, param_stddev);
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

void AffineComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                               const MatrixBase<BaseFloat> &,  // out_value
                               const MatrixBase<BaseFloat> &out_deriv,
                               const VectorBase<BaseFloat> &chunk_weights,
                               Component *to_update_in,
                               Matrix<BaseFloat> *in_deriv) const {
  AffineComponent *to_update = dynamic_cast<AffineComponent*>(to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  // Propagate the derivative back to the input.
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                      0.0);

  if (to_update) {
    BaseFloat old_weight = to_update->OldWeight(chunk_weights.Sum());
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    // add the sum of the rows of out_deriv, to the bias_params_.
    to_update->bias_params_.AddRowSumMat(to_update->learning_rate_, out_deriv,
                                         old_weight);
    to_update->linear_params_.AddMatMat(to_update->learning_rate_,
                                        out_deriv, kTrans, in_value, kNoTrans,
                                        old_weight);
  }
}


void AffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<AffineComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<L2Penalty>");
  ReadBasicType(is, binary, &l2_penalty_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</AffineComponent>");  
}

void AffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AffineComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<L2Penalty>");
  WriteBasicType(os, binary, l2_penalty_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</AffineComponent>");  
}


void AffinePreconInputComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    SetL2Penalty(0.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void AffinePreconInputComponent::Backprop(
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &, // out_value
    const MatrixBase<BaseFloat> &out_deriv,
    const VectorBase<BaseFloat> &chunk_weights,
    Component *to_update_in,
    Matrix<BaseFloat> *in_deriv) const {
  AffinePreconInputComponent *to_update =
      dynamic_cast<AffinePreconInputComponent*>(to_update_in);
  in_deriv->Resize(out_deriv.NumRows(), InputDim());
  // Propagate the derivative back to the input.
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                      0.0);

  if (to_update) {
    BaseFloat old_weight = to_update->OldWeight(chunk_weights.Sum());
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    // add the sum of the rows of out_deriv, to the bias_params_.
    to_update->bias_params_.AddRowSumMat(to_update->learning_rate_, out_deriv,
                                         old_weight);
    if (to_update->is_gradient_) { // simple update, getting gradient.
      to_update->linear_params_.AddMatMat(to_update->learning_rate_,
                                          out_deriv, kTrans,
                                          in_value, kNoTrans,
                                          old_weight);
    } else {
      // more complex update, correcting for variance of input features.  Note:
      // most likely to_update == this, but we don't insist on this.
      Matrix<BaseFloat> in_value_tmp(in_value);
      in_value_tmp.MulColsVec(input_precision_); // Scale each column of in_value_tmp
      // (i.e. each dimension of the input features) by the corresponding element of
      // input_precision_.
    
      to_update->linear_params_.AddMatMat(to_update->learning_rate_,
                                          out_deriv, kTrans, in_value_tmp,
                                          kNoTrans, old_weight);
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
  ExpectToken(is, binary, "<L2Penalty>");
  ReadBasicType(is, binary, &l2_penalty_);
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
  WriteToken(os, binary, "<L2Penalty>");
  WriteBasicType(os, binary, l2_penalty_);
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
    BaseFloat learning_rate, BaseFloat l2_penalty,
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev,
    BaseFloat avg_samples) {
  is_gradient_ = false;
  UpdatableComponent::Init(learning_rate, l2_penalty);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(param_stddev);
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
               l2_penalty = l2_penalty_,
             avg_samples = 2000.0;
  int32 input_dim = -1, output_dim = -1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("l2-penalty", &args, &l2_penalty); // optional.
  ParseFromString("avg-samples", &args, &avg_samples); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
  ParseFromString("param-stddev", &args, &param_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, l2_penalty, input_dim, output_dim,
       param_stddev, avg_samples);
}

Component* AffinePreconInputComponent::Copy() const {
  AffinePreconInputComponent *ans = new AffinePreconInputComponent();
  ans->learning_rate_ = learning_rate_;
  ans->l2_penalty_ = l2_penalty_;
  ans->avg_samples_ = avg_samples_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->input_precision_ = input_precision_;
  return ans;
}

void BlockAffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    SetL2Penalty(0.0);
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
  ans->l2_penalty_ = l2_penalty_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->num_blocks_ = num_blocks_;
  return ans;
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
    const VectorBase<BaseFloat> &chunk_weights,
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
  if (to_update)
    to_update->bias_params_.AddRowSumMat(to_update->learning_rate_, out_deriv,
                                         to_update->OldWeight(chunk_weights.Sum()));
  
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
    
    if (to_update) {
      SubMatrix<BaseFloat> param_block_to_update(
          to_update->linear_params_,
          b * output_block_dim, output_block_dim,
          0, input_block_dim);
      // Update the parameters.
      param_block_to_update.AddMatMat(
          to_update->learning_rate_,
          out_deriv_block, kTrans, in_value_block, kNoTrans,
          to_update->OldWeight(chunk_weights.Sum()));
    }
  }  
}

BaseFloat UpdatableComponent::OldWeight(BaseFloat tot_weight) const {
  // tot_weight would equal #frames if we did not have frame weighting.
  return std::pow(static_cast<BaseFloat>(1.0 - 2.0 * learning_rate_ * l2_penalty_),
                  static_cast<BaseFloat>(tot_weight));
}

void BlockAffineComponent::Init(BaseFloat learning_rate, BaseFloat l2_penalty,
                                int32 input_dim, int32 output_dim,
                                BaseFloat param_stddev, int32 num_blocks) {
  UpdatableComponent::Init(learning_rate, l2_penalty);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  KALDI_ASSERT(input_dim % num_blocks == 0 && output_dim % num_blocks == 0);

  linear_params_.Resize(output_dim, input_dim / num_blocks);
  bias_params_.Resize(output_dim);

  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(param_stddev);
  num_blocks_ = num_blocks;
}

void BlockAffineComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_,
               l2_penalty = l2_penalty_;
  int32 input_dim = -1, output_dim = -1, num_blocks = 1;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("l2-penalty", &args, &l2_penalty); // optional.
  ok = ok && ParseFromString("input-dim", &args, &input_dim);
  ok = ok && ParseFromString("output-dim", &args, &output_dim);
  ok = ok && ParseFromString("num-blocks", &args, &num_blocks);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
  ParseFromString("param-stddev", &args, &param_stddev);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, l2_penalty, input_dim, output_dim, param_stddev,
       num_blocks);
}
  

void BlockAffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BlockAffineComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<L2Penalty>");
  ReadBasicType(is, binary, &l2_penalty_);
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
  WriteToken(os, binary, "<L2Penalty>");
  WriteBasicType(os, binary, l2_penalty_);
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
                                const VectorBase<BaseFloat> &, // chunk_weights
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
  for (size_t i = 0; i < params_.size(); i++) {
    Matrix<BaseFloat> temp_params(params_[i]);
    temp_params.SetRandn();
    params_[i].AddMat(stddev, temp_params);
  }
}


Component* MixtureProbComponent::Copy() const {
  MixtureProbComponent *ans = new MixtureProbComponent();
  ans->learning_rate_ = learning_rate_;
  ans->l2_penalty_ = l2_penalty_;
  ans->params_ = params_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}


BaseFloat MixtureProbComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const MixtureProbComponent *other =
      dynamic_cast<const MixtureProbComponent*>(&other_in);
  BaseFloat ans = 0.0;
  KALDI_ASSERT(params_.size() == other->params_.size());
  for (size_t i = 0; i < params_.size(); i++)
    ans += TraceMatMat(params_[i], other->params_[i], kTrans);
  return ans;
}

void MixtureProbComponent::Init(BaseFloat learning_rate, BaseFloat l2_penalty,
                                BaseFloat diag_element,
                                const std::vector<int32> &sizes) {
  UpdatableComponent::Init(learning_rate, l2_penalty);
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

void MixtureProbComponent::InitFromString(std::string args) {
  std::string orig_args(args);
  bool ok = true;
  BaseFloat learning_rate = learning_rate_,
               l2_penalty = l2_penalty_,
             diag_element = 0.9;
  std::vector<int32> dims;
  ParseFromString("learning-rate", &args, &learning_rate); // optional.
  ParseFromString("l2-penalty", &args, &l2_penalty); // optional.
  ParseFromString("diag-element", &args, &diag_element); // optional.
  ok = ok && ParseFromString("dims", &args, &dims);
  if (!args.empty())
    KALDI_ERR << "Could not process these elements in initializer: "
              << args;
  if (!ok)
    KALDI_ERR << "Bad initializer " << orig_args;
  Init(learning_rate, l2_penalty, diag_element, dims);
}


void MixtureProbComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MixtureProbComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<L2Penalty>");
  ReadBasicType(is, binary, &l2_penalty_);
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
  WriteToken(os, binary, "<L2Penalty>");
  WriteBasicType(os, binary, l2_penalty_);
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
    SetL2Penalty(0.0);
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
    output_offset += this_input_dim;   
  }
}

void MixtureProbComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                    const MatrixBase<BaseFloat> &,// out_value
                                    const MatrixBase<BaseFloat> &out_deriv,
                                    const VectorBase<BaseFloat> &chunk_weights,
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

    if (to_update) {
      Matrix<BaseFloat> &param_block_to_update(to_update->params_[i]);
      if (to_update->is_gradient_) { // We're just storing
        // the gradient there, so it's a linear update rule as for any other layer.
        // Note: most likely the learning_rate_ will be 1.0 and OldWeight() will
        // be 1.0 because of zero l2_penalty_.
        KALDI_ASSERT(to_update->OldWeight(chunk_weights.Sum()) == 1.0 &&
                     to_update->learning_rate_ == 1.0);
        param_block_to_update.AddMatMat(1.0, out_deriv_block, kTrans, in_value_block,
                                        kNoTrans, 1.0);
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
        BaseFloat old_weight = to_update->OldWeight(chunk_weights.Sum());
        for (int32 col = 0; col < num_cols; col++) {
          Vector<BaseFloat> param_col(num_rows);
          param_col.CopyColFromMat(param_block_to_update, col);
          Vector<BaseFloat> log_param_col(param_col);
          log_param_col.ApplyLog(); // note: works even for zero, but may have -inf
          log_param_col.Scale(old_weight); // relates to l2 regularization-- applied at log
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
    output_offset += this_input_dim;   
  }
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
                                      (input_dim - const_component_dim_) * c, input_dim - const_component_dim_);
      output_part.CopyFromMat(input_part);
    }
    //Append the constant component at the end of the output vector
    if (const_component_dim_ != 0) {
      SubMatrix<BaseFloat> input_part(input_chunk, 
                                    0, output_chunk_size,
                                    InputDim() - const_component_dim_, const_component_dim_),
                           output_part(output_chunk, 
                                    0, output_chunk_size,
                                    OutputDim() - const_component_dim_, const_component_dim_);
      output_part.CopyFromMat(input_part);
    }

  }  
}

void SpliceComponent::Backprop(const MatrixBase<BaseFloat> &, // in_value
                               const MatrixBase<BaseFloat> &, // out_value,
                               const MatrixBase<BaseFloat> &out_deriv,
                               const VectorBase<BaseFloat> &chunk_weights,
                               Component *to_update, // may == "this".
                               Matrix<BaseFloat> *in_deriv) const {
  int32 num_chunks = chunk_weights.Dim();
 
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
                            const VectorBase<BaseFloat> &chunk_weights,
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

} // namespace kaldi

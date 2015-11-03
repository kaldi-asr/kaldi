// nnet3/nnet-simple-component.cc

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
#include <algorithm>
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {

void PnormComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  if (input_dim_ == 0)
    input_dim_ = 10 * output_dim_; // default group size : 10
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0);
  KALDI_ASSERT(input_dim_ % output_dim_ == 0);
}

void PnormComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || cfl->HasUnusedValues() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim);
}


void PnormComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in,
                               CuMatrixBase<BaseFloat> *out) const {
  BaseFloat p = 2.0;
  out->GroupPnorm(in, p);  // TODO: when done, replace with Group2Norm function.
}

void PnormComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  BaseFloat p = 2.0;
  // TODO: use Group2NormDeriv when done.
  in_deriv->GroupPnormDeriv(in_value, out_value, p);
  in_deriv->MulRowsGroupMat(out_deriv);
}

void PnormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PnormComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</PnormComponent>");
}

void PnormComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PnormComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</PnormComponent>");
}

std::string PnormComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_;
  return stream.str();
}


void ElementwiseProductComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ >= 0);
  KALDI_ASSERT(input_dim_ > output_dim_);
  KALDI_ASSERT(input_dim_ % output_dim_ == 0);
}

void ElementwiseProductComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || cfl->HasUnusedValues() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim);
}


void ElementwiseProductComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == input_dim_);
  int32 num_inputs = input_dim_ / output_dim_;
  for (int32 i = 0; i < num_inputs; i++)  {
    CuSubMatrix<BaseFloat> current_in(in, 0, in.NumRows(),
                                      i * output_dim_, output_dim_);
    if (i == 0) {
      out->CopyFromMat(current_in);
    } else  {
      out->MulElements(current_in);
    }
  }
}

void ElementwiseProductComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  int32 num_inputs = input_dim_ / output_dim_;
  for (int32 i = 0; i < num_inputs; i++)  {
    CuSubMatrix<BaseFloat> current_in_deriv(*in_deriv, 0, in_deriv->NumRows(),
                                            i * output_dim_,
                                            output_dim_);
    current_in_deriv.CopyFromMat(out_deriv);
    for (int32 j = 0; j < num_inputs; j++)  {
      if (i == j)
        continue;
      CuSubMatrix<BaseFloat> in_value_partition(in_value, 0,
                                                in_value.NumRows(),
                                                j * output_dim_,
                                                output_dim_);
      current_in_deriv.MulElements(in_value_partition);
    }
  }
}

void ElementwiseProductComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<ElementwiseProductComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</ElementwiseProductComponent>");
}

void ElementwiseProductComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ElementwiseProductComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</ElementwiseProductComponent>");
}

std::string ElementwiseProductComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_;
  return stream.str();
}

const BaseFloat NormalizeComponent::kNormFloor = pow(2.0, -66);
// This component modifies the vector of activations by scaling it so that the
// root-mean-square equals 1.0.  It's important that its square root
// be exactly representable in float.

void NormalizeComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  CuVector<BaseFloat> in_norm(in.NumRows());
  in_norm.AddDiagMat2(1.0 / in.NumCols(),
                      in, kNoTrans, 0.0);
  in_norm.ApplyFloor(kNormFloor);
  in_norm.ApplyPow(-0.5);
  out->CopyFromMat(in);
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
void NormalizeComponent::Backprop(const std::string &debug_info,
                                  const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &, // out_value
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *to_update,
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  CuVector<BaseFloat> dot_products(out_deriv.NumRows());
  dot_products.AddDiagMatMat(1.0, out_deriv, kNoTrans, in_value, kTrans, 0.0);
  CuVector<BaseFloat> in_norm(in_value.NumRows());
  in_norm.AddDiagMat2(1.0 / in_value.NumCols(),
                      in_value, kNoTrans, 0.0);
  in_norm.ApplyFloor(kNormFloor);
  in_norm.ApplyPow(-0.5);

  if (in_deriv) {
    if (in_deriv->Data() != out_deriv.Data())
      in_deriv->AddDiagVecMat(1.0, in_norm, out_deriv, kNoTrans, 0.0);
    else
      in_deriv->MulRowsVec(in_norm);
  }
  in_norm.ReplaceValue(1.0 / sqrt(kNormFloor), 0.0);
  in_norm.ApplyPow(3.0);
  dot_products.MulElements(in_norm);
  in_deriv->AddDiagVecMat(-1.0 / in_value.NumCols(),
                          dot_products, in_value,
                          kNoTrans, 1.0);
}

void SigmoidComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->Sigmoid(in);
}

void SigmoidComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL)
    in_deriv->DiffSigmoid(out_value, out_deriv);
}

void SigmoidComponent::StoreStats(const CuMatrixBase<BaseFloat> &out_value) {
  // derivative of the nonlinearity is out_value * (1.0 - out_value);
  CuMatrix<BaseFloat> temp_deriv(out_value.NumRows(), out_value.NumCols(),
                                 kUndefined);
  temp_deriv.Set(1.0);
  temp_deriv.AddMat(-1.0, out_value);
  temp_deriv.MulElements(out_value);
  StoreStatsInternal(out_value, &temp_deriv);
}



void NoOpComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
}

void NoOpComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  in_deriv->CopyFromMat(out_deriv);
}

void ClipGradientComponent::Read(std::istream &is, bool binary) {
  // might not see the "<NaturalGradientAffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, "<ClipGradientComponent>",
                       "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<ClippingThreshold>");
  ReadBasicType(is, binary, &clipping_threshold_);
  ExpectToken(is, binary, "<NormBasedClipping>");
  ReadBasicType(is, binary, &norm_based_clipping_);
  ExpectToken(is, binary, "<NumElementsClipped>");
  ReadBasicType(is, binary, &num_clipped_);
  ExpectToken(is, binary, "<NumElementsProcessed>");
  ReadBasicType(is, binary, &count_);
  ExpectToken(is, binary, "</ClipGradientComponent>");
}

void ClipGradientComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ClipGradientComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<ClippingThreshold>");
  WriteBasicType(os, binary, clipping_threshold_);
  WriteToken(os, binary, "<NormBasedClipping>");
  WriteBasicType(os, binary, norm_based_clipping_);
  WriteToken(os, binary, "<NumElementsClipped>");
  WriteBasicType(os, binary, num_clipped_);
  WriteToken(os, binary, "<NumElementsProcessed>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "</ClipGradientComponent>");
}

std::string ClipGradientComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", norm-based-clipping="
         << (norm_based_clipping_ ? "true" : "false")
         << ", clipping-threshold=" << clipping_threshold_
         << ", clipped-proportion="
         << (count_ > 0 ? static_cast<BaseFloat>(num_clipped_)/count_ : 0);
  return stream.str();
}

void ClipGradientComponent::Init(int32 dim,
                                 BaseFloat clipping_threshold,
                                 bool norm_based_clipping,
                                 int32 num_clipped,
                                 int32 count)  {
  KALDI_ASSERT(clipping_threshold >= 0 && dim > 0);
  dim_ = dim;
  norm_based_clipping_ = norm_based_clipping;
  clipping_threshold_ = clipping_threshold;
  num_clipped_ = num_clipped;
  count_ = count;
}

void ClipGradientComponent::InitFromConfig(ConfigLine *cfl) {
  int32 dim = 0;
  bool ok = cfl->GetValue("dim", &dim);
  bool norm_based_clipping = false;
  BaseFloat clipping_threshold = 15.0;
  cfl->GetValue("clipping-threshold", &clipping_threshold);
  cfl->GetValue("norm-based-clipping", &norm_based_clipping);
  if (!ok || cfl->HasUnusedValues() ||
      clipping_threshold < 0 || dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim, clipping_threshold, norm_based_clipping, 0, 0);
}

void ClipGradientComponent::Propagate(
                                 const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
}


void ClipGradientComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             Component *to_update_in, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  // the following statement will do nothing if in_deriv and out_deriv have same
  // memory.
  in_deriv->CopyFromMat(out_deriv);

  ClipGradientComponent *to_update =
      dynamic_cast<ClipGradientComponent*>(to_update_in);
  KALDI_ASSERT(to_update != NULL);

  if (clipping_threshold_ > 0) {
    if (norm_based_clipping_) {
      // each row in the derivative matrix, which corresponds to one sample in
      // the mini-batch, is scaled to have a max-norm of clipping_threshold_
      CuVector<BaseFloat> clipping_scales(in_deriv->NumRows());
      clipping_scales.AddDiagMat2(pow(clipping_threshold_, -2), *in_deriv,
                                  kNoTrans, 0.0);
     // now clipping_scales contains the squared (norm of each row divided by
     //  clipping_threshold)
      int32 num_not_scaled = clipping_scales.ApplyFloor(1.0);
     // now clipping_scales contains min(1,
     //    squared-(norm/clipping_threshold))
      if (num_not_scaled != clipping_scales.Dim()) {
        clipping_scales.ApplyPow(-0.5);
        // now clipping_scales contains max(1,
        //       clipping_threshold/vector_norm)
        in_deriv->MulRowsVec(clipping_scales);
        to_update->num_clipped_ += (clipping_scales.Dim() - num_not_scaled);
       }
      to_update->count_ += clipping_scales.Dim();
    } else {
      // each element of the derivative matrix, is clipped to be below the
      // clipping_threshold_
      in_deriv->ApplyCeiling(clipping_threshold_);
      in_deriv->ApplyFloor(-1 * clipping_threshold_);
    }
  }
}

void ClipGradientComponent::ZeroStats()  {
  count_ = 0.0;
  num_clipped_ = 0.0;
}

void ClipGradientComponent::Scale(BaseFloat scale) {
  count_ *= scale;
  num_clipped_ *= scale;
}

void ClipGradientComponent::Add(BaseFloat alpha, const Component &other_in) {
  const ClipGradientComponent *other =
      dynamic_cast<const ClipGradientComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  count_ += alpha * other->count_;
  num_clipped_ += alpha * other->num_clipped_;
}

void TanhComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in,
                              CuMatrixBase<BaseFloat> *out) const {
  // Apply tanh function to each element of the output...
  // the tanh function may be written as -1 + ( 2 / (1 + e^{-2 x})),
  // which is a scaled and shifted sigmoid.
  out->Tanh(in);
}

void TanhComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             Component *to_update, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL)
    in_deriv->DiffTanh(out_value, out_deriv);
}

/*
  Note on the derivative of the tanh function:
  tanh'(x) = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)

  The element by element equation of what we're doing would be:
  in_deriv = out_deriv * (1.0 - out_value^2).
  We can accomplish this via calls to the matrix library. */
void TanhComponent::StoreStats(const CuMatrixBase<BaseFloat> &out_value) {
  // derivative of the onlinearity is out_value * (1.0 - out_value);
  CuMatrix<BaseFloat> temp_deriv(out_value);
  temp_deriv.ApplyPow(2.0);
  temp_deriv.Scale(-1.0);
  temp_deriv.Add(1.0);
  StoreStatsInternal(out_value, &temp_deriv);
}


void RectifiedLinearComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  // Apply rectified linear function (x >= 0 ? 1.0 : 0.0)
  out->CopyFromMat(in);
  out->ApplyFloor(0.0);
}

void RectifiedLinearComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    in_deriv->CopyFromMat(out_value);
    in_deriv->ApplyHeaviside();
    in_deriv->MulElements(out_deriv);
  }
}

void RectifiedLinearComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &out_value) {
  CuMatrix<BaseFloat> temp_deriv(out_value);
  temp_deriv.ApplyHeaviside();
  StoreStatsInternal(out_value, &temp_deriv);
}

void AffineComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void AffineComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
}

void AffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

AffineComponent::AffineComponent(const AffineComponent &component):
    UpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_) { }

AffineComponent::AffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 const CuVectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    UpdatableComponent(learning_rate),
    linear_params_(linear_params),
    bias_params_(bias_params) {
  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
}



void AffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
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
         << ", learning-rate=" << LearningRate()
         << ", is-gradient=" << (is_gradient_ ? "true" : "false");
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

void AffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  cfl->GetValue("learning-rate", &learning_rate); // optional.
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(learning_rate, matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(learning_rate, input_dim, output_dim,
         param_stddev, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}




void AffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {

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

void AffineComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               Component *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  AffineComponent *to_update = dynamic_cast<AffineComponent*>(to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                        1.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
  }
}

void AffineComponent::Read(std::istream &is, bool binary) {
  // might not see the "<AffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, "</AffineComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</AffineComponent>");
}

void AffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AffineComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</AffineComponent>");
}

int32 AffineComponent::NumParameters() const {
  return (InputDim() + 1) * OutputDim();
}
void AffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
  params->Range(InputDim() * OutputDim(),
                OutputDim()).CopyFromVec(bias_params_);
}
void AffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  bias_params_.CopyFromVec(params.Range(InputDim() * OutputDim(),
                                        OutputDim()));
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

Component *AffineComponent::CollapseWithNext(
    const PerElementScaleComponent &next_component) const {
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


void PerElementScaleComponent::Scale(BaseFloat scale) {
  scales_.Scale(scale);
}

void PerElementScaleComponent::Resize(int32 dim) {
  KALDI_ASSERT(dim > 0);
  scales_.Resize(dim);
}

void PerElementScaleComponent::Add(BaseFloat alpha,
                                   const Component &other_in) {
  const PerElementScaleComponent *other =
      dynamic_cast<const PerElementScaleComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  scales_.AddVec(alpha, other->scales_);
}

PerElementScaleComponent::PerElementScaleComponent(
    const PerElementScaleComponent &component):
    UpdatableComponent(component),
    scales_(component.scales_) { }

void PerElementScaleComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    is_gradient_ = true;
  }
  scales_.SetZero();
}

void PerElementScaleComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_scales(scales_);
  temp_scales.SetRandn();
  scales_.AddVec(stddev, temp_scales);
}

std::string PerElementScaleComponent::Info() const {
  std::stringstream stream;
  BaseFloat scales_stddev = std::sqrt(VecVec(scales_, scales_) /
                              scales_.Dim());
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", scales-stddev=" << scales_stddev
         << ", learning-rate=" << LearningRate();
  return stream.str();
}

Component* PerElementScaleComponent::Copy() const {
  PerElementScaleComponent *ans = new PerElementScaleComponent();
  ans->learning_rate_ = learning_rate_;
  ans->scales_ = scales_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}

BaseFloat PerElementScaleComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const PerElementScaleComponent *other =
      dynamic_cast<const PerElementScaleComponent*>(&other_in);
  return VecVec(scales_, other->scales_);
}

void PerElementScaleComponent::Init(
    BaseFloat learning_rate, int32 dim,
    BaseFloat param_mean, BaseFloat param_stddev) {
  UpdatableComponent::Init(learning_rate);
  scales_.Resize(dim);
  KALDI_ASSERT(dim > 0 && param_stddev >= 0.0);
  scales_.SetRandn();
  scales_.Scale(param_stddev);
  scales_.Add(param_mean);
}

void PerElementScaleComponent::Init(BaseFloat learning_rate,
                                    std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() == 1);
  int32 dim = mat.NumRows();
  scales_.Resize(dim);
  scales_.CopyColFromMat(mat, 0);
}

void PerElementScaleComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  std::string matrix_filename;
  int32 dim = -1;
  cfl->GetValue("learning-rate", &learning_rate); // optional.
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(learning_rate, matrix_filename);
    if (cfl->GetValue("dim", &dim))
      KALDI_ASSERT(dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("dim", &dim);
    BaseFloat param_mean = 1.0, param_stddev = 0.0;
    cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);
    Init(learning_rate, dim, param_mean, param_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void PerElementScaleComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  out->MulColsVec(scales_);
}

void PerElementScaleComponent::UpdateSimple(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  scales_.AddDiagMatMat(learning_rate_, out_deriv, kTrans,
                        in_value, kNoTrans, 1.0);
}

void PerElementScaleComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  PerElementScaleComponent *to_update =
      dynamic_cast<PerElementScaleComponent*>(to_update_in);

  if (in_deriv) {
    // Propagate the derivative back to the input.
    in_deriv->CopyFromMat(out_deriv);
    in_deriv->MulColsVec(scales_);
  }

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
  }
}

void PerElementScaleComponent::Read(std::istream &is, bool binary) {
  // might not see the begin marker part because of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, "<PerElementScaleComponent>", "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<Params>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</PerElementScaleComponent>");
}

void PerElementScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PerElementScaleComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<Params>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</PerElementScaleComponent>");
}

int32 PerElementScaleComponent::NumParameters() const {
  return InputDim();
}

void PerElementScaleComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyFromVec(scales_);
}

void PerElementScaleComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  scales_.CopyFromVec(params);
}

NaturalGradientAffineComponent::NaturalGradientAffineComponent(): max_change_per_sample_(0.0),
  update_count_(0.0), active_scaling_count_(0.0), max_change_scale_stats_(0.0) { }

// virtual
void NaturalGradientAffineComponent::Resize(
    int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 1 && output_dim > 1);
  if (rank_in_ >= input_dim) rank_in_ = input_dim - 1;
  if (rank_out_ >= output_dim) rank_out_ = output_dim - 1;
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
  OnlineNaturalGradient temp;
  preconditioner_in_ = temp;
  preconditioner_out_ = temp;
  SetNaturalGradientConfigs();
}


void NaturalGradientAffineComponent::Read(std::istream &is, bool binary) {
  // might not see the "<NaturalGradientAffineComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, "<NaturalGradientAffineComponent>",
                       "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<RankIn>");
  ReadBasicType(is, binary, &rank_in_);
  ExpectToken(is, binary, "<RankOut>");
  ReadBasicType(is, binary, &rank_out_);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period_);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history_);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<MaxChangePerSample>");
  ReadBasicType(is, binary, &max_change_per_sample_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<UpdateCount>") {
    ReadBasicType(is, binary, &update_count_);
    ExpectToken(is, binary, "<ActiveScalingCount>");
    ReadBasicType(is, binary, &active_scaling_count_);
    ExpectToken(is, binary, "<MaxChangeScaleStats>");
    ReadBasicType(is, binary, &max_change_scale_stats_);
    ExpectToken(is, binary, "<NaturalGradientAffineComponent>");
  } else {
    if (token != "<NaturalGradientAffineComponent>")
      KALDI_ERR << "Expected <NaturalGradientAffineComponent>, got " << token;
  }
  SetNaturalGradientConfigs();
}

void NaturalGradientAffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  BaseFloat learning_rate = learning_rate_;
  BaseFloat num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_sample = 0.0;
  int32 input_dim = -1, output_dim = -1, rank_in = 20, rank_out = 80,
      update_period = 4;
  cfl->GetValue("learning-rate", &learning_rate); // optional.
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("max-change-per-sample", &max_change_per_sample);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);

  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(learning_rate, rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample,
         matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0, bias_mean = 0.0, bias_init = 0.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    cfl->GetValue("bias-mean", &bias_mean);
    cfl->GetValue("bias-init", &bias_init);
    Init(learning_rate, input_dim, output_dim, param_stddev,
         bias_init, bias_mean, bias_stddev, rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void NaturalGradientAffineComponent::SetNaturalGradientConfigs() {
  preconditioner_in_.SetRank(rank_in_);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_in_.SetAlpha(alpha_);
  preconditioner_in_.SetUpdatePeriod(update_period_);
  preconditioner_out_.SetRank(rank_out_);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_out_.SetAlpha(alpha_);
  preconditioner_out_.SetUpdatePeriod(update_period_);
}

void NaturalGradientAffineComponent::Init(
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
  SetNaturalGradientConfigs();
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
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void NaturalGradientAffineComponent::Init(
    BaseFloat learning_rate,
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_init,
    BaseFloat bias_mean, BaseFloat bias_stddev,
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
  if (bias_init != 0.0) {
    bias_params_.Set(bias_init);
  } else {
    bias_params_.SetRandn();
    bias_params_.Add(bias_mean);
    bias_params_.Scale(bias_stddev);
  }
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  if (max_change_per_sample > 0.0)
    KALDI_WARN << "You are setting a positive max_change_per_sample for "
               << "NaturalGradientAffineComponent. But the per-component "
               << "gradient clipping mechansim has been removed. Instead it's currently "
               << "done at the whole model level.";
  max_change_per_sample_ = max_change_per_sample;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void NaturalGradientAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NaturalGradientAffineComponent>");
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
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<UpdateCount>");
  WriteBasicType(os, binary, update_count_);
  WriteToken(os, binary, "<ActiveScalingCount>");
  WriteBasicType(os, binary, active_scaling_count_);
  WriteToken(os, binary, "<MaxChangeScaleStats>");
  WriteBasicType(os, binary, max_change_scale_stats_);
  WriteToken(os, binary, "<NaturalGradientAffineComponent>");
}

std::string NaturalGradientAffineComponent::Info() const {
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
  if (update_count_ > 0.0) {
    stream << ", avg-scaling-factor=" << max_change_scale_stats_ / update_count_
           << ", active-scaling-portion="
           << active_scaling_count_ / update_count_;
  }
  return stream.str();
}

Component* NaturalGradientAffineComponent::Copy() const {
  NaturalGradientAffineComponent *ans = new NaturalGradientAffineComponent();
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
  ans->update_count_ = update_count_;
  ans->active_scaling_count_ = active_scaling_count_;
  ans->max_change_scale_stats_ = max_change_scale_stats_;
  ans->SetNaturalGradientConfigs();
  return ans;
}

void NaturalGradientAffineComponent::Update(
    const std::string &debug_info,
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

  CuSubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, in_value_temp.NumRows(),
                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  CuVector<BaseFloat> precon_ones(in_value_temp.NumRows());

  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

  BaseFloat local_lrate = scale * learning_rate_;
  update_count_ += 1.0;
  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void NaturalGradientAffineComponent::ZeroStats()  {
  update_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
  active_scaling_count_ = 0.0;
}

void NaturalGradientAffineComponent::Scale(BaseFloat scale) {
  update_count_ *= scale;
  max_change_scale_stats_ *= scale;
  active_scaling_count_ *= scale;
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void NaturalGradientAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const NaturalGradientAffineComponent *other =
      dynamic_cast<const NaturalGradientAffineComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  update_count_ += alpha * other->update_count_;
  max_change_scale_stats_ += alpha * other->max_change_scale_stats_;
  active_scaling_count_ += alpha * other->active_scaling_count_;
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

std::string FixedAffineComponent::Info() const {
  std::stringstream stream;
  BaseFloat linear_params_size =
      static_cast<BaseFloat>(linear_params_.NumRows())
      * static_cast<BaseFloat>(linear_params_.NumCols());
  BaseFloat linear_params_stddev =
      std::sqrt(TraceMatMat(linear_params_,
                            linear_params_, kTrans) / linear_params_size);
  BaseFloat bias_params_stddev =
      std::sqrt(VecVec(bias_params_, bias_params_) / bias_params_.Dim());

  stream << Component::Info() << ", linear-params-stddev="
      << linear_params_stddev << ", bias-params-stddev=" << bias_params_stddev;
  return stream.str();
}

void FixedAffineComponent::Init(const CuMatrixBase<BaseFloat> &mat) {
  KALDI_ASSERT(mat.NumCols() > 1);
  linear_params_ = mat.Range(0, mat.NumRows(), 0, mat.NumCols() - 1);
  bias_params_.Resize(mat.NumRows());
  bias_params_.CopyColFromMat(mat, mat.NumCols() - 1);
}

void FixedAffineComponent::InitFromConfig(ConfigLine *cfl) {
  std::string filename;
  // Two forms allowed: "matrix=<rxfilename>", or "input-dim=x output-dim=y"
  // (for testing purposes only).
  if (cfl->GetValue("matrix", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";

    bool binary;
    Input ki(filename, &binary);
    CuMatrix<BaseFloat> mat;
    mat.Read(ki.Stream(), binary);
    KALDI_ASSERT(mat.NumRows() != 0);
    Init(mat);
  } else {
    int32 input_dim, output_dim;
    if (!cfl->GetValue("input-dim", &input_dim) ||
        !cfl->GetValue("output-dim", &output_dim) || cfl->HasUnusedValues()) {
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    }
    CuMatrix<BaseFloat> mat(output_dim, input_dim + 1);
    mat.SetRandn();
    Init(mat);
  }
}


void FixedAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const  {
  out->CopyRowsFromVec(bias_params_); // Adds the bias term first.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

void FixedAffineComponent::Backprop(const std::string &debug_info,
                                    const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &, //in_value
                                    const CuMatrixBase<BaseFloat> &, //out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    Component *, //to_update
                                    CuMatrixBase<BaseFloat> *in_deriv) const {
  // kBackpropAdds is true. It's the user's responsibility to zero out
  // <in_deriv> if they need it to be so.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans,
                        linear_params_, kNoTrans, 1.0);
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

void SumGroupComponent::InitFromConfig(ConfigLine *cfl) {
  std::vector<int32> sizes;
  bool ok = cfl->GetValue("sizes", &sizes);

  if (!ok || cfl->HasUnusedValues() || sizes.empty())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
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

void SumGroupComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &in,
                                  CuMatrixBase<BaseFloat> *out) const {
  out->SumColumnRanges(in, indexes_);
}

void SumGroupComponent::Backprop(const std::string &debug_info,
                                 const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &, // in_value,
                                 const CuMatrixBase<BaseFloat> &, // out_value
                                 const CuMatrixBase<BaseFloat> &out_deriv,
                                 Component *to_update_in,
                                 CuMatrixBase<BaseFloat> *in_deriv) const {
  in_deriv->CopyCols(out_deriv, reverse_indexes_);
}

void SoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).
  out->ApplySoftMaxPerRow(in);

  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);
}

void SoftmaxComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &, // in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv == NULL)
    return;
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
  const CuMatrixBase<BaseFloat> &P(out_value), &E(out_deriv);
  CuMatrixBase<BaseFloat> &D (*in_deriv);

  D.CopyFromMat(P);
  D.MulElements(E);
  // At this point, D = P .* E (in matlab notation)
  CuVector<BaseFloat> pe_vec(D.NumRows()); // For each row i, the dot product (p_t . e_t).
  pe_vec.AddDiagMatMat(1.0, P, kNoTrans, E, kTrans, 0.0);

  D.AddDiagVecMat(-1.0, pe_vec, P, kNoTrans, 1.0); // does D -= diag(pe_vec) * P.
}

void SoftmaxComponent::StoreStats(const CuMatrixBase<BaseFloat> &out_value) {
  // We don't store derivative stats for this component type, just activation
  // stats.
  StoreStatsInternal(out_value, NULL);
}


void LogSoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  // Applies log softmax function to each row of the output. For each row, we do
  // x_i = x_i - log(sum_j exp(x_j))
  out->ApplyLogSoftMaxPerRow(in);
}

void LogSoftmaxComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &, // in_value
                                   const CuMatrixBase<BaseFloat> &out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *, // to_update
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv == NULL)
    return;

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
  const CuMatrixBase<BaseFloat> &Y(out_value), &E(out_deriv);
  CuMatrixBase<BaseFloat> &D (*in_deriv);

  D.CopyFromMat(Y);
  D.ApplyExp();                           // exp(y)
  CuVector<BaseFloat> E_sum(D.NumRows()); // Initializes to zero
  E_sum.AddColSumMat(1.0, E);             // Sum(e)
  D.MulRowsVec(E_sum);                    // exp(y) Sum(e)
  D.Scale(-1.0);                          // - exp(y) Sum(e)
  D.AddMat(1.0, E, kNoTrans);             // e - exp(y_i) Sum(e)
}


void FixedScaleComponent::Init(const CuVectorBase<BaseFloat> &scales) {
  KALDI_ASSERT(scales.Dim() != 0);
  scales_ = scales;
}


void FixedScaleComponent::InitFromConfig(ConfigLine *cfl) {
  std::string filename;
  // Accepts "scales" config (for filename) or "dim" -> random init, for testing.
  if (cfl->GetValue("scales", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    CuVector<BaseFloat> vec;
    ReadKaldiObject(filename, &vec);
    Init(vec);
  } else {
    int32 dim;
    if (!cfl->GetValue("dim", &dim) || cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    KALDI_ASSERT(dim > 0);
    CuVector<BaseFloat> vec(dim);
    vec.SetRandn();
    Init(vec);
  }
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

void FixedScaleComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);  // does nothing if same matrix.
  out->MulColsVec(scales_);
}

void FixedScaleComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &, // in_value
                                   const CuMatrixBase<BaseFloat> &, // out_value
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *, // to_update
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  in_deriv->CopyFromMat(out_deriv);  // does nothing if same memory.
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

void FixedBiasComponent::InitFromConfig(ConfigLine *cfl) {
  std::string filename;
  // Accepts "bias" config (for filename) or "dim" -> random init, for testing.
  if (cfl->GetValue("bias", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    CuVector<BaseFloat> vec;
    ReadKaldiObject(filename, &vec);
    Init(vec);
  } else {
    int32 dim;
    if (!cfl->GetValue("dim", &dim) || cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    KALDI_ASSERT(dim > 0);
    CuVector<BaseFloat> vec(dim);
    vec.SetRandn();
    Init(vec);
  }
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

void FixedBiasComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);  // will do nothing if in and out have same memory.
  out->AddVecToRows(1.0, bias_, 1.0);
}

void FixedBiasComponent::Backprop(const std::string &debug_info,
                                  const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &, // in_value
                                  const CuMatrixBase<BaseFloat> &, // out_value
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *, // to_update
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  // the following statement will do nothing if in_deriv and out_deriv have same
  // memory.
  in_deriv->CopyFromMat(out_deriv);
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


void NaturalGradientPerElementScaleComponent::Read(
    std::istream &is, bool binary) {
  // might not see the begin marker part because of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, "<NaturalGradientPerElementScaleComponent>",
                       "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<Params>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  int32 rank, update_period;
  ExpectToken(is, binary, "<Rank>");
  ReadBasicType(is, binary, &rank);
  preconditioner_.SetRank(rank);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);
  preconditioner_.SetUpdatePeriod(update_period);
  BaseFloat num_samples_history, alpha;
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  preconditioner_.SetNumSamplesHistory(num_samples_history);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  preconditioner_.SetAlpha(alpha);
  ExpectToken(is, binary, "<MaxChangePerMinibatch>");
  ReadBasicType(is, binary, &max_change_per_minibatch_);
  ExpectToken(is, binary, "</NaturalGradientPerElementScaleComponent>");
}

void NaturalGradientPerElementScaleComponent::Write(std::ostream &os,
                                                    bool binary) const {
  WriteToken(os, binary, "<NaturalGradientPerElementScaleComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<Params>");
  scales_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<Rank>");
  WriteBasicType(os, binary, preconditioner_.GetRank());
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, preconditioner_.GetUpdatePeriod());
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, preconditioner_.GetNumSamplesHistory());
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, preconditioner_.GetAlpha());
  WriteToken(os, binary, "<MaxChangePerMinibatch>");
  WriteBasicType(os, binary, max_change_per_minibatch_);
  WriteToken(os, binary, "</NaturalGradientPerElementScaleComponent>");
}

std::string NaturalGradientPerElementScaleComponent::Info() const {
  std::stringstream stream;
  stream << PerElementScaleComponent::Info()
         << ", rank=" << preconditioner_.GetRank()
         << ", update-period=" << preconditioner_.GetUpdatePeriod()
         << ", num-samples-history=" << preconditioner_.GetNumSamplesHistory()
         << ", alpha=" << preconditioner_.GetAlpha()
         << ", max-change-per-minibatch=" << max_change_per_minibatch_;
  return stream.str();
}

void NaturalGradientPerElementScaleComponent::InitFromConfig(ConfigLine *cfl) {
  // First set various configuration values that have defaults.
  int32 rank = 8,  // Use a small rank because in this case the amount of memory
                   // for the preconditioner actually exceeds the memory for the
                   // parameters (by "rank").
      update_period = 10;
  // the max_change_per_minibatch is the maximum amount of parameter-change, in 2-norm,
  // that we allow per minibatch; if change is greater than that, we scale down
  // the parameter-change.  It has the same purpose as the max-change-per-sample in
  // the NaturalGradientAffineComponent.
  BaseFloat num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_minibatch = 0.0,
      learning_rate = learning_rate_;  // default to value from constructor.
  cfl->GetValue("rank", &rank);
  cfl->GetValue("update-period", &update_period);
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("max-change-per-minibatch", &max_change_per_minibatch);
  cfl->GetValue("learning-rate", &learning_rate);

  std::string filename;
  // Accepts "scales" config (for filename) or "dim" -> random init, for testing.
  if (cfl->GetValue("scales", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    Init(learning_rate, filename, rank, update_period, num_samples_history,
         alpha, max_change_per_minibatch);
  } else {
    BaseFloat param_mean = 1.0, param_stddev = 0.0;
    cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);

    int32 dim;
    if (!cfl->GetValue("dim", &dim) || cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    KALDI_ASSERT(dim > 0);

    Init(learning_rate, dim, param_mean, param_stddev, rank, update_period,
         num_samples_history, alpha, max_change_per_minibatch);
  }
}

void NaturalGradientPerElementScaleComponent::Init(
    BaseFloat learning_rate, int32 dim, BaseFloat param_mean,
    BaseFloat param_stddev, int32 rank, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_minibatch) {
  PerElementScaleComponent::Init(learning_rate, dim, param_mean,
                                 param_stddev);
  preconditioner_.SetRank(rank);
  preconditioner_.SetUpdatePeriod(update_period);
  preconditioner_.SetNumSamplesHistory(num_samples_history);
  preconditioner_.SetAlpha(alpha);
  max_change_per_minibatch_ = max_change_per_minibatch;
  if (max_change_per_minibatch > 0.0)
    KALDI_WARN << "You are setting a positive max_change_per_minibatch for "
               << "NaturalGradientPerElementScaleComponent. But the per-component "
               << "gradient clipping mechansim has been removed. Instead it's currently "
               << "done at the whole model level.";
}

void NaturalGradientPerElementScaleComponent::Init(
    BaseFloat learning_rate, std::string vector_filename,
    int32 rank, int32 update_period, BaseFloat num_samples_history,
    BaseFloat alpha, BaseFloat max_change_per_minibatch) {
  PerElementScaleComponent::Init(learning_rate, vector_filename);
  preconditioner_.SetRank(rank);
  preconditioner_.SetUpdatePeriod(update_period);
  preconditioner_.SetNumSamplesHistory(num_samples_history);
  preconditioner_.SetAlpha(alpha);
  max_change_per_minibatch_ = max_change_per_minibatch;
}


NaturalGradientPerElementScaleComponent::NaturalGradientPerElementScaleComponent(
    const NaturalGradientPerElementScaleComponent &other):
    PerElementScaleComponent(other),
    max_change_per_minibatch_(other.max_change_per_minibatch_),
    preconditioner_(other.preconditioner_) { }




Component* NaturalGradientPerElementScaleComponent::Copy() const {
  return new NaturalGradientPerElementScaleComponent(*this);
}

void NaturalGradientPerElementScaleComponent::Update(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  CuMatrix<BaseFloat> derivs_per_frame(in_value);
  derivs_per_frame.MulElements(out_deriv);
  // the non-natural-gradient update would just do
  // scales_.AddRowSumMat(learning_rate_, derivs_per_frame).

  BaseFloat scale;
  preconditioner_.PreconditionDirections(&derivs_per_frame, NULL, &scale);

  CuVector<BaseFloat> delta_scales(scales_.Dim());
  delta_scales.AddRowSumMat(scale * learning_rate_, derivs_per_frame);
  scales_.AddVec(1.0, delta_scales);
}

// Constructors for the convolution component
ConvolutionComponent::ConvolutionComponent():
    UpdatableComponent(),
    input_x_dim_(0), input_y_dim_(0), input_z_dim_(0),
    filt_x_dim_(0), filt_y_dim_(0),
    filt_x_step_(0), filt_y_step_(0),
    input_vectorization_(kZyx),
    is_gradient_(false) {}

ConvolutionComponent::ConvolutionComponent(
    const ConvolutionComponent &component):
    UpdatableComponent(component),
    input_x_dim_(component.input_x_dim_),
    input_y_dim_(component.input_y_dim_),
    input_z_dim_(component.input_z_dim_),
    filt_x_dim_(component.filt_x_dim_),
    filt_y_dim_(component.filt_y_dim_),
    filt_x_step_(component.filt_x_step_),
    filt_y_step_(component.filt_y_step_),
    input_vectorization_(component.input_vectorization_),
    filter_params_(component.filter_params_),
    bias_params_(component.bias_params_),
    is_gradient_(component.is_gradient_) {}

ConvolutionComponent::ConvolutionComponent(
    const CuMatrixBase<BaseFloat> &filter_params,
    const CuVectorBase<BaseFloat> &bias_params,
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step,
    TensorVectorizationType input_vectorization,
    BaseFloat learning_rate):
    UpdatableComponent(learning_rate),
    input_x_dim_(input_x_dim),
    input_y_dim_(input_y_dim),
    input_z_dim_(input_z_dim),
    filt_x_dim_(filt_x_dim),
    filt_y_dim_(filt_y_dim),
    filt_x_step_(filt_x_step),
    filt_y_step_(filt_y_step),
    input_vectorization_(input_vectorization),
    filter_params_(filter_params),
    bias_params_(bias_params){
  KALDI_ASSERT(filter_params.NumRows() == bias_params.Dim() &&
               bias_params.Dim() != 0);
  KALDI_ASSERT(filter_params.NumCols() == filt_x_dim * filt_y_dim * input_z_dim);
  is_gradient_ = false;
}

// aquire input dim
int32 ConvolutionComponent::InputDim() const {
  return input_x_dim_ * input_y_dim_ * input_z_dim_;
}

// aquire output dim
int32 ConvolutionComponent::OutputDim() const {
  int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_);
  int32 num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_);
  int32 num_filters = filter_params_.NumRows();
  return num_x_steps * num_y_steps * num_filters;
}

// initialize the component using hyperparameters
void ConvolutionComponent::Init(
    BaseFloat learning_rate,
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step, int32 num_filters,
    TensorVectorizationType input_vectorization,
    BaseFloat param_stddev, BaseFloat bias_stddev) {
  UpdatableComponent::Init(learning_rate);
  input_x_dim_ = input_x_dim;
  input_y_dim_ = input_y_dim;
  input_z_dim_ = input_z_dim;
  filt_x_dim_ = filt_x_dim;
  filt_y_dim_ = filt_y_dim;
  filt_x_step_ = filt_x_step;
  filt_y_step_ = filt_y_step;
  input_vectorization_ = input_vectorization;
  KALDI_ASSERT((input_x_dim_ - filt_x_dim_) % filt_x_step_ == 0);
  KALDI_ASSERT((input_y_dim_ - filt_y_dim_) % filt_y_step_ == 0);
  int32 filter_dim = filt_x_dim_ * filt_y_dim_ * input_z_dim_;
  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
  KALDI_ASSERT(param_stddev >= 0.0 && bias_stddev >= 0.0);
  filter_params_.SetRandn();
  filter_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

// initialize the component using predefined matrix file
void ConvolutionComponent::Init(
    BaseFloat learning_rate,
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step,
    TensorVectorizationType input_vectorization,
    std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  input_x_dim_ = input_x_dim;
  input_y_dim_ = input_y_dim;
  input_z_dim_ = input_z_dim;
  filt_x_dim_ = filt_x_dim;
  filt_y_dim_ = filt_y_dim;
  filt_x_step_ = filt_x_step;
  filt_y_step_ = filt_y_step;
  input_vectorization_ = input_vectorization;
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat);
  int32 filter_dim = (filt_x_dim_ * filt_y_dim_ * input_z_dim_);
  int32 num_filters = mat.NumRows();
  KALDI_ASSERT(mat.NumCols() == (filter_dim + 1));
  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
  filter_params_.CopyFromMat(mat.Range(0, num_filters, 0, filter_dim));
  bias_params_.CopyColFromMat(mat, filter_dim);
}

// display information about component
std::string ConvolutionComponent::Info() const {
  std::stringstream stream;
  BaseFloat filter_params_size =
      static_cast<BaseFloat>(filter_params_.NumRows())
      * static_cast<BaseFloat>(filter_params_.NumCols());
  BaseFloat filter_stddev =
            std::sqrt(TraceMatMat(filter_params_, filter_params_, kTrans) /
                      filter_params_size),
            bias_stddev = std::sqrt(VecVec(bias_params_, bias_params_) /
                                    bias_params_.Dim());

  stream << Type() << ", input-x-dim=" << input_x_dim_
         << ", input-y-dim=" << input_y_dim_
         << ", input-z-dim=" << input_z_dim_
         << ", filt-x-dim=" << filt_x_dim_
         << ", filt-y-dim=" << filt_y_dim_
         << ", filt-x-step=" << filt_x_step_
         << ", filt-y-step=" << filt_y_step_
         << ", input-vectorization=" << input_vectorization_
         << ", num-filters=" << filter_params_.NumRows()
         << ", filter-params-stddev=" << filter_stddev
         << ", bias-params-stddev=" << bias_stddev
         << ", learning-rate=" << LearningRate();
  return stream.str();
}

// initialize the component using configuration file
void ConvolutionComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  std::string matrix_filename;
  int32 input_x_dim = -1, input_y_dim = -1, input_z_dim = -1,
        filt_x_dim = -1, filt_y_dim = -1,
        filt_x_step = -1, filt_y_step = -1,
        num_filters = -1;
  std::string input_vectorization_order = "zyx";
  cfl->GetValue("learning-rate", &learning_rate); //optional
  ok = ok && cfl->GetValue("input-x-dim", &input_x_dim);
  ok = ok && cfl->GetValue("input-y-dim", &input_y_dim);
  ok = ok && cfl->GetValue("input-z-dim", &input_z_dim);
  ok = ok && cfl->GetValue("filt-x-dim", &filt_x_dim);
  ok = ok && cfl->GetValue("filt-y-dim", &filt_y_dim);
  ok = ok && cfl->GetValue("filt-x-step", &filt_x_step);
  ok = ok && cfl->GetValue("filt-y-step", &filt_y_step);

  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
  // optional argument
  TensorVectorizationType input_vectorization;
  cfl->GetValue("input-vectorization-order", &input_vectorization_order);
  if (input_vectorization_order.compare("zyx") == 0) {
    input_vectorization = kZyx;
  } else if (input_vectorization_order.compare("yzx") == 0) {
    input_vectorization = kYzx;
  } else {
    KALDI_ERR << "Unknown or unsupported input vectorization order "
              << input_vectorization_order
              << " accepted candidates are 'yzx' and 'zyx'";
  }

  if (cfl->GetValue("matrix", &matrix_filename)) {
    // initialize from prefined parameter matrix
    Init(learning_rate, input_x_dim, input_y_dim, input_z_dim,
         filt_x_dim, filt_y_dim,
         filt_x_step, filt_y_step,
         input_vectorization,
         matrix_filename);
  } else {
    ok = ok && cfl->GetValue("num-filters", &num_filters);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    // initialize from configuration
    int32 filter_input_dim = filt_x_dim * filt_y_dim * input_z_dim;
    BaseFloat param_stddev = 1.0 / std::sqrt(filter_input_dim), bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(learning_rate, input_x_dim, input_y_dim, input_z_dim,
         filt_x_dim, filt_y_dim, filt_x_step, filt_y_step, num_filters,
         input_vectorization, param_stddev, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
	      << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

// Inline methods to convert from tensor index i.e., (x,y,z) index
// to index in yzx or zyx vectorized tensors
inline int32 YzxVectorIndex(int32 x, int32 y, int32 z,
                            int32 input_x_dim,
                            int32 input_y_dim,
                            int32 input_z_dim) {
  KALDI_PARANOID_ASSERT(x < input_x_dim && y < input_y_dim && z < input_z_dim);
  return (input_y_dim * input_z_dim) * x + (input_y_dim) * z + y;
}

inline int32 ZyxVectorIndex(int32 x, int32 y, int32 z,
                            int32 input_x_dim,
                            int32 input_y_dim,
                            int32 input_z_dim) {
  KALDI_PARANOID_ASSERT(x < input_x_dim && y < input_y_dim && z < input_z_dim);
  return (input_y_dim * input_z_dim) * x + (input_z_dim) * y + z;
}

// Method to convert from a matrix representing a minibatch of vectorized
// 3D tensors to patches for convolution, each patch corresponds to
// one dot product in the convolution
void ConvolutionComponent::InputToInputPatches(
    const CuMatrixBase<BaseFloat>& in,
    CuMatrix<BaseFloat> *patches) const{
  int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_);
  int32 num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_);
  const int32 filt_x_step = filt_x_step_,
              filt_y_step = filt_y_step_,
              filt_x_dim = filt_x_dim_,
              filt_y_dim = filt_y_dim_,
              input_x_dim = input_x_dim_,
              input_y_dim = input_y_dim_,
              input_z_dim = input_z_dim_,
              filter_dim = filter_params_.NumCols();

  std::vector<int32> column_map(patches->NumCols());
  int32 column_map_size = column_map.size();
  for (int32 x_step = 0; x_step < num_x_steps; x_step++) {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      int32 patch_start_index = patch_number * filter_dim;
      for (int32 x = 0, index = patch_start_index; x < filt_x_dim; x++)  {
        for (int32 y = 0; y < filt_y_dim; y++)  {
          for (int32 z = 0; z < input_z_dim; z++, index++)  {
            KALDI_ASSERT(index < column_map_size);
            if (input_vectorization_ == kZyx)  {
              column_map[index] = ZyxVectorIndex(x_step * filt_x_step + x,
                                                 y_step * filt_y_step + y, z,
                                                 input_x_dim, input_y_dim,
                                                 input_z_dim);
            } else if (input_vectorization_ == kYzx)  {
              column_map[index] = YzxVectorIndex(x_step * filt_x_step + x,
                                                  y_step * filt_y_step + y, z,
                                                  input_x_dim, input_y_dim,
                                                  input_z_dim);
            }
          }
        }
      }
    }
  }
  CuArray<int32> cu_cols(column_map);
  patches->CopyCols(in, cu_cols);
}


// propagation function
// see function declaration in nnet-simple-component.h for details
void ConvolutionComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                         const CuMatrixBase<BaseFloat> &in,
                                         CuMatrixBase<BaseFloat> *out) const {
  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              num_filters = filter_params_.NumRows(),
              num_frames = in.NumRows(),
              filter_dim = filter_params_.NumCols();
  KALDI_ASSERT((*out).NumRows() == num_frames &&
               (*out).NumCols() == (num_filters * num_x_steps * num_y_steps));

  CuMatrix<BaseFloat> patches(num_frames,
                              num_x_steps * num_y_steps * filter_dim,
                              kUndefined);
  InputToInputPatches(in, &patches);
  CuSubMatrix<BaseFloat>* filter_params_elem = new CuSubMatrix<BaseFloat>(
		  filter_params_, 0, filter_params_.NumRows(), 0,
		  filter_params_.NumCols());
  std::vector<CuSubMatrix<BaseFloat>* > tgt_batch, patch_batch,
      filter_params_batch;

  for (int32 x_step = 0; x_step < num_x_steps; x_step++)  {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      tgt_batch.push_back(new CuSubMatrix<BaseFloat>(
              out->ColRange(patch_number * num_filters, num_filters)));
      patch_batch.push_back(new CuSubMatrix<BaseFloat>(
              patches.ColRange(patch_number * filter_dim, filter_dim)));
      filter_params_batch.push_back(filter_params_elem);
      tgt_batch[patch_number]->AddVecToRows(1.0, bias_params_, 1.0); // add bias
    }
  }
  // apply all filters
  AddMatMatBatched(1.0f, tgt_batch, patch_batch, kNoTrans, filter_params_batch,
		  kTrans, 1.0f);
  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < tgt_batch.size(); p++) {
    delete tgt_batch[p];
    delete patch_batch[p];
  }
}

// scale the parameters
void ConvolutionComponent::Scale(BaseFloat scale) {
  filter_params_.Scale(scale);
  bias_params_.Scale(scale);
}

// add another convolution component
void ConvolutionComponent::Add(BaseFloat alpha, const Component &other_in) {
  const ConvolutionComponent *other =
      dynamic_cast<const ConvolutionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  filter_params_.AddMat(alpha, other->filter_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
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
void RearrangeIndexes(const std::vector<std::vector<int32> > &in,
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

// Method to compute the input derivative matrix from the input derivatives
// for patches, where each patch corresponds to one dot product
// in the convolution
void ConvolutionComponent::InderivPatchesToInderiv(
    const CuMatrix<BaseFloat>& in_deriv_patches,
    CuMatrixBase<BaseFloat> *in_deriv) const {

  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              filt_x_step = filt_x_step_,
              filt_y_step = filt_y_step_,
              filt_x_dim = filt_x_dim_,
              filt_y_dim = filt_y_dim_,
              input_x_dim = input_x_dim_,
              input_y_dim = input_y_dim_,
              input_z_dim = input_z_dim_,
              filter_dim = filter_params_.NumCols();

  // Compute the reverse column_map from the matrix with input
  // derivative patches to input derivative matrix
  std::vector<std::vector<int32> > reverse_column_map(in_deriv->NumCols());
  int32 rev_col_map_size = reverse_column_map.size();
  for (int32 x_step = 0; x_step < num_x_steps; x_step++) {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      int32 patch_start_index = patch_number * filter_dim;
      for (int32 x = 0, index = patch_start_index; x < filt_x_dim; x++)  {
        for (int32 y = 0; y < filt_y_dim; y++)  {
          for (int32 z = 0; z < input_z_dim; z++, index++)  {
            int32 vector_index;
            if (input_vectorization_ == kZyx)  {
              vector_index = ZyxVectorIndex(x_step * filt_x_step + x,
                                            y_step * filt_y_step + y, z,
                                            input_x_dim, input_y_dim,
                                            input_z_dim);
            } else if (input_vectorization_ == kYzx)  {
              vector_index = YzxVectorIndex(x_step * filt_x_step + x,
                                            y_step * filt_y_step + y, z,
                                            input_x_dim, input_y_dim,
                                            input_z_dim);
            }
            KALDI_ASSERT(vector_index < rev_col_map_size);
            reverse_column_map[vector_index].push_back(index);
          }
        }
      }
    }
  }
  std::vector<std::vector<int32> > rearranged_column_map;
  RearrangeIndexes(reverse_column_map, &rearranged_column_map);
  for (int32 p = 0; p < rearranged_column_map.size(); p++) {
    CuArray<int32> cu_cols(rearranged_column_map[p]);
    in_deriv->AddCols(in_deriv_patches, cu_cols);
  }
}

// back propagation function
// see function declaration in nnet-simple-component.h for details
void ConvolutionComponent::Backprop(const std::string &debug_info,
                                    const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in_value,
                                    const CuMatrixBase<BaseFloat> &, // out_value,
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    Component *to_update_in,
                                    CuMatrixBase<BaseFloat> *in_deriv) const {
  ConvolutionComponent *to_update =
      dynamic_cast<ConvolutionComponent*>(to_update_in);
  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              num_filters = filter_params_.NumRows(),
              num_frames = out_deriv.NumRows(),
              filter_dim = filter_params_.NumCols();

  KALDI_ASSERT(out_deriv.NumRows() == num_frames &&
               out_deriv.NumCols() ==
               (num_filters * num_x_steps * num_y_steps));

  // Compute inderiv patches
  CuMatrix<BaseFloat> in_deriv_patches(num_frames,
                                       num_x_steps * num_y_steps * filter_dim,
                                       kSetZero);

  std::vector<CuSubMatrix<BaseFloat>* > patch_deriv_batch, out_deriv_batch,
	  filter_params_batch;
  CuSubMatrix<BaseFloat>* filter_params_elem = new CuSubMatrix<BaseFloat>(
		  filter_params_, 0, filter_params_.NumRows(), 0,
		  filter_params_.NumCols());

  for (int32 x_step = 0; x_step < num_x_steps; x_step++)  {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;

      patch_deriv_batch.push_back(new CuSubMatrix<BaseFloat>(
              in_deriv_patches.ColRange(
              patch_number * filter_dim, filter_dim)));
      out_deriv_batch.push_back(new CuSubMatrix<BaseFloat>(out_deriv.ColRange(
              patch_number * num_filters, num_filters)));
      filter_params_batch.push_back(filter_params_elem);
    }
  }
  AddMatMatBatched(1.0f, patch_deriv_batch, out_deriv_batch, kNoTrans,
		  filter_params_batch, kNoTrans, 0.0f);


  if (in_deriv) {
    // combine the derivatives from the individual input deriv patches
    // to compute input deriv matrix
    InderivPatchesToInderiv(in_deriv_patches, in_deriv);
  }

  if (to_update != NULL)  {
    to_update->Update(debug_info, in_value, out_deriv, out_deriv_batch);
  }

  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < patch_deriv_batch.size(); p++) {
    delete patch_deriv_batch[p];
    delete out_deriv_batch[p];
  }
}


// update parameters
// see function declaration in nnet-simple-component.h for details
void ConvolutionComponent::Update(const std::string &debug_info,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  const std::vector<CuSubMatrix<BaseFloat> *>& out_deriv_batch) {
  // useful dims
  const int32 num_x_steps = (1 + (input_x_dim_ - filt_x_dim_) / filt_x_step_),
              num_y_steps = (1 + (input_y_dim_ - filt_y_dim_) / filt_y_step_),
              num_filters = filter_params_.NumRows(),
              num_frames = out_deriv.NumRows(),
              filter_dim = filter_params_.NumCols();
  KALDI_ASSERT(out_deriv.NumRows() == num_frames &&
               out_deriv.NumCols() ==
               (num_filters * num_x_steps * num_y_steps));


  CuMatrix<BaseFloat> filters_grad;
  CuVector<BaseFloat> bias_grad;

  CuMatrix<BaseFloat> input_patches(num_frames,
                                    filter_dim * num_x_steps * num_y_steps,
                                    kUndefined);
  InputToInputPatches(in_value, &input_patches);

  filters_grad.Resize(num_filters, filter_dim, kSetZero); // reset
  bias_grad.Resize(num_filters, kSetZero); // reset

  // create a single large matrix holding the smaller matrices
  // from the vector container filters_grad_batch along the rows
  CuMatrix<BaseFloat> filters_grad_blocks_batch(
      num_x_steps * num_y_steps * filters_grad.NumRows(),
      filters_grad.NumCols());

  std::vector<CuSubMatrix<BaseFloat>* > filters_grad_batch, input_patch_batch;

  for (int32 x_step = 0; x_step < num_x_steps; x_step++)  {
    for (int32 y_step = 0; y_step < num_y_steps; y_step++)  {
      int32 patch_number = x_step * num_y_steps + y_step;
      filters_grad_batch.push_back(new CuSubMatrix<BaseFloat>(
              filters_grad_blocks_batch.RowRange(
				      patch_number * filters_grad.NumRows(),
				    filters_grad.NumRows())));

      input_patch_batch.push_back(new CuSubMatrix<BaseFloat>(
              input_patches.ColRange(patch_number * filter_dim, filter_dim)));
    }
  }

  AddMatMatBatched(1.0f, filters_grad_batch, out_deriv_batch, kTrans,
                   input_patch_batch, kNoTrans, 1.0f);

  // add the row blocks together to filters_grad
  filters_grad.AddMatBlocks(1.0, filters_grad_blocks_batch);

  // create a matrix holding the col blocks sum of out_deriv
  CuMatrix<BaseFloat> out_deriv_col_blocks_sum(out_deriv.NumRows(),
                                               num_filters);

  // add the col blocks together to out_deriv_col_blocks_sum
  out_deriv_col_blocks_sum.AddMatBlocks(1.0, out_deriv);

  bias_grad.AddRowSumMat(1.0, out_deriv_col_blocks_sum, 1.0);

  // release memory
  for (int32 p = 0; p < input_patch_batch.size(); p++) {
    delete filters_grad_batch[p];
    delete input_patch_batch[p];
  }

  //
  // update
  //
  filter_params_.AddMat(learning_rate_, filters_grad);
  bias_params_.AddVec(learning_rate_, bias_grad);
}

void ConvolutionComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
  }
  filter_params_.SetZero();
  bias_params_.SetZero();
  if (treat_as_gradient) {
    is_gradient_ = true;
  }
}

void ConvolutionComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<ConvolutionComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</ConvolutionComponent>"
  // might not see the "<ConvolutionComponent>" part because
  // of how ReadNew() works.
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<InputXDim>");
  ReadBasicType(is, binary, &input_x_dim_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<InputYDim>");
  ReadBasicType(is, binary, &input_y_dim_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<InputZDim>");
  ReadBasicType(is, binary, &input_z_dim_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<FiltXDim>");
  ReadBasicType(is, binary, &filt_x_dim_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<FiltYDim>");
  ReadBasicType(is, binary, &filt_y_dim_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<FiltXStep>");
  ReadBasicType(is, binary, &filt_x_step_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<FiltYStep>");
  ReadBasicType(is, binary, &filt_y_step_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<InputVectorization>");
  int32 input_vectorization;
  ReadBasicType(is, binary, &input_vectorization);
  input_vectorization_ = static_cast<TensorVectorizationType>(input_vectorization);
  ExpectToken(is, binary, "<FilterParams>");
  filter_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<IsGradient>") {
    ReadBasicType(is, binary, &is_gradient_);
    ExpectToken(is, binary, ostr_end.str());
  } else {
    is_gradient_ = false;
    KALDI_ASSERT(tok == ostr_end.str());
  }
}

void ConvolutionComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<Convolutional1dComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</Convolutional1dComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<InputXDim>");
  WriteBasicType(os, binary, input_x_dim_);
  WriteToken(os, binary, "<InputYDim>");
  WriteBasicType(os, binary, input_y_dim_);
  WriteToken(os, binary, "<InputZDim>");
  WriteBasicType(os, binary, input_z_dim_);
  WriteToken(os, binary, "<FiltXDim>");
  WriteBasicType(os, binary, filt_x_dim_);
  WriteToken(os, binary, "<FiltYDim>");
  WriteBasicType(os, binary, filt_y_dim_);
  WriteToken(os, binary, "<FiltXStep>");
  WriteBasicType(os, binary, filt_x_step_);
  WriteToken(os, binary, "<FiltYStep>");
  WriteBasicType(os, binary, filt_y_step_);
  WriteToken(os, binary, "<InputVectorization>");
  WriteBasicType(os, binary, static_cast<int32>(input_vectorization_));
  WriteToken(os, binary, "<FilterParams>");
  filter_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, ostr_end.str());
}

BaseFloat ConvolutionComponent::DotProduct(const UpdatableComponent &other_in) const {
  const ConvolutionComponent *other =
      dynamic_cast<const ConvolutionComponent*>(&other_in);
  return TraceMatMat(filter_params_, other->filter_params_, kTrans)
         + VecVec(bias_params_, other->bias_params_);
}

Component* ConvolutionComponent::Copy() const {
  ConvolutionComponent *ans = new ConvolutionComponent(*this);
  return ans;
}

void ConvolutionComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_filter_params(filter_params_);
  temp_filter_params.SetRandn();
  filter_params_.AddMat(stddev, temp_filter_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

void ConvolutionComponent::SetParams(const VectorBase<BaseFloat> &bias,
                                     const MatrixBase<BaseFloat> &filter) {
  bias_params_ = bias;
  filter_params_ = filter;
  KALDI_ASSERT(bias_params_.Dim() == filter_params_.NumRows());
}

int32 ConvolutionComponent::NumParameters() const {
  return (filter_params_.NumCols() + 1) * filter_params_.NumRows();
}

Convolutional1dComponent::Convolutional1dComponent():
    UpdatableComponent(),
    patch_dim_(0), patch_step_(0), patch_stride_(0), is_gradient_(false) {}

Convolutional1dComponent::Convolutional1dComponent(const Convolutional1dComponent &component):
    UpdatableComponent(component),
    filter_params_(component.filter_params_),
    bias_params_(component.bias_params_),
    is_gradient_(component.is_gradient_) {}

Convolutional1dComponent::Convolutional1dComponent(const CuMatrixBase<BaseFloat> &filter_params,
                                                   const CuVectorBase<BaseFloat> &bias_params,
                                                   BaseFloat learning_rate):
    UpdatableComponent(learning_rate),
    filter_params_(filter_params),
    bias_params_(bias_params) {
  KALDI_ASSERT(filter_params.NumRows() == bias_params.Dim() &&
               bias_params.Dim() != 0);
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
                                    int32 patch_dim, int32 patch_step, int32 patch_stride,
                                    BaseFloat param_stddev, BaseFloat bias_stddev) {
  UpdatableComponent::Init(learning_rate);
  patch_dim_ = patch_dim;
  patch_step_ = patch_step;
  patch_stride_ = patch_stride;
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
void Convolutional1dComponent::Init(BaseFloat learning_rate,
                                    int32 patch_dim, int32 patch_step, int32 patch_stride,
                                    std::string matrix_filename) {
  UpdatableComponent::Init(learning_rate);
  patch_dim_ = patch_dim;
  patch_step_ = patch_step;
  patch_stride_ = patch_stride;
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
         << ", learning-rate=" << LearningRate();
  return stream.str();
}

// initialize the component using configuration file
void Convolutional1dComponent::InitFromConfig(ConfigLine *cfl) {
  KALDI_WARN << "Convolutional1dComponent has been deprecated."
             << " Please use ConvolutionComponent.";
  bool ok = true;
  BaseFloat learning_rate = learning_rate_;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  int32 patch_dim = -1, patch_step = -1, patch_stride = -1;
  cfl->GetValue("learning-rate", &learning_rate); //optional
  ok = ok && cfl->GetValue("patch-dim", &patch_dim);
  ok = ok && cfl->GetValue("patch-step", &patch_step);
  ok = ok && cfl->GetValue("patch-stride", &patch_stride);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    // initialize from prefined parameter matrix
    Init(learning_rate, patch_dim, patch_step, patch_stride, matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
               "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
               "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    // initialize from configuration
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim), bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(learning_rate, input_dim, output_dim,
         patch_dim, patch_step, patch_stride, param_stddev, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
	      << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

// propagation function

/* Convolutional propagation is explained:
 - Recall the AffineComponent, input X is defined #frames x $input-dim,
   linear matrix A is defined $output-dim x $input-dim, and bias
   vector B is defined by length $output-dim. The propagation is
   Y = X * A' + B                                     (1)
   where "*" is row-by-row processing of X, executing vector-matrix
   multiplication
   Y(t) = X(t) * A' + B                               (2)
   which converts each row of input of dim $input-dim to a row of output of
   dim $output-dim by A' (' defines transpose).
 - In Convolution1dComponent, A is redefined $num-filters x $filter-dim,
   and bias vector B is redefined by length $num-filters. The propatation is
   Y = X o A' + B                                     (3)
   where "o" is also row-by-row processing of X, but executing vector-matrix
   convolution, which consists of a group of vector-vector convolutions.
   For instance, the convolution of X(t) and the i-th filter A(i) is
   Y(t,i) = X(t) o A'(i) + B(i)                       (4)
   The convolution used here is valid convolution. Meaning that the
   output of M o N is of dim |M| - |N| + 1, assuming M is not shorter then N.

   Note that in all the equations, B is extended to proper dimensions
   for legal addition.
*/
void Convolutional1dComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                         const CuMatrixBase<BaseFloat> &in,
                                         CuMatrixBase<BaseFloat> *out) const {
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
  for (int32 p = 0, index = 0; p < num_patches; p++) {
    for (int32 s = 0; s < num_splice; s++) {
        for (int32 d = 0; d < patch_dim_; d++, index++) {
        column_map[index] = p * patch_step_ + s * patch_stride_ + d;
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
		  filter_params_, 0, filter_params_.NumRows(), 0,
		  filter_params_.NumCols());

  // form batch in vector container
  for (int32 p = 0; p < num_patches; p++) {
    // form batch in vector container. for filter_params_batch, all elements
    // point to the same copy filter_params_elem
    tgt_batch.push_back(new CuSubMatrix<BaseFloat>(out->ColRange(p * num_filters,
				    num_filters)));
    patch_batch.push_back(new CuSubMatrix<BaseFloat>(patches.ColRange(p * filter_dim,
				    filter_dim)));
    filter_params_batch.push_back(filter_params_elem);

    tgt_batch[p]->AddVecToRows(1.0, bias_params_, 1.0); // add bias
  }

  // apply all filters
  AddMatMatBatched(1.0f, tgt_batch, patch_batch, kNoTrans, filter_params_batch,
		  kTrans, 1.0f);

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
void Convolutional1dComponent::Add(BaseFloat alpha, const Component &other_in) {
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
void Convolutional1dComponent::Backprop(const std::string &debug_info,
                                        const ComponentPrecomputedIndexes *indexes,
                                        const CuMatrixBase<BaseFloat> &in_value,
                                        const CuMatrixBase<BaseFloat> &, // out_value,
                                        const CuMatrixBase<BaseFloat> &out_deriv,
                                        Component *to_update_in,
                                        CuMatrixBase<BaseFloat> *in_deriv) const {
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
		  filter_params_, 0, filter_params_.NumRows(), 0,
		  filter_params_.NumCols());

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
  AddMatMatBatched(1.0f, patch_deriv_batch, out_deriv_batch, kNoTrans,
		  filter_params_batch, kNoTrans, 0.0f);

  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < num_patches; p++) {
    delete patch_deriv_batch[p];
    delete out_deriv_batch[p];
  }

  // sum the derivatives into in_deriv
  std::vector<int32> column_map(filter_dim * num_patches);
  for (int32 p = 0, index = 0; p < num_patches; p++) {
    for (int32 s = 0; s < num_splice; s++) {
      for (int32 d = 0; d < patch_dim_; d++, index++) {
        column_map[index] = p * patch_step_ + s * patch_stride_ + d;
      }
    }
  }

  if (in_deriv) {
    std::vector<std::vector<int32> > reversed_column_map;
    ReverseIndexes(column_map, InputDim(), &reversed_column_map);
    std::vector<std::vector<int32> > rearranged_column_map;
    RearrangeIndexes(reversed_column_map, &rearranged_column_map);
    for (int32 p = 0; p < rearranged_column_map.size(); p++) {
      CuArray<int32> cu_cols(rearranged_column_map[p]);
      in_deriv->AddCols(patches_deriv, cu_cols);
    }
  }

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    to_update->Update(debug_info, in_value, out_deriv);
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
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<PatchDim>");
  ReadBasicType(is, binary, &patch_dim_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<PatchStep>");
  ReadBasicType(is, binary, &patch_step_);
  ExpectOneOrTwoTokens(is, binary, ostr_beg.str(), "<PatchStride>");
  ReadBasicType(is, binary, &patch_stride_);
  ExpectToken(is, binary, "<FilterParams>");
  filter_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  std::string tok;
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

int32 Convolutional1dComponent::NumParameters() const {
  return (filter_params_.NumCols() + 1) * filter_params_.NumRows();
}

// update parameters
void Convolutional1dComponent::Update(const std::string &debug_info,
		                      const CuMatrixBase<BaseFloat> &in_value,
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
  for (int32 p = 0, index = 0; p < num_patches; p++) {
    for (int32 s = 0; s < num_splice; s++) {
      for (int32 d = 0; d < patch_dim_; d++, index++) {
        column_map[index] = p * patch_step_ + s * patch_stride_ + d;
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

  AddMatMatBatched(1.0f, filters_grad_batch, diff_patch_batch, kTrans, patch_batch,
		  kNoTrans, 1.0f);

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

void MaxpoolingComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  int32 pool_size = -1, pool_stride = -1;
  bool ok = true;

  ok = ok && cfl->GetValue("input-dim", &input_dim);
  ok = ok && cfl->GetValue("output-dim", &output_dim);
  ok = ok && cfl->GetValue("pool-size", &pool_size);
  ok = ok && cfl->GetValue("pool-stride", &pool_stride);

  KALDI_LOG << output_dim << " " << input_dim << " " << ok;
  KALDI_LOG << "Pool: " << pool_size << " "
            << pool_stride << " " << ok;
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
	      << cfl->UnusedValues();
  if (!ok || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim, pool_size, pool_stride);
}

void MaxpoolingComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
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

void MaxpoolingComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *, // to_update,
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  int32 num_patches = input_dim_ / pool_stride_;
  int32 num_pools = num_patches / pool_size_;
  std::vector<int32> patch_summands(num_patches, 0);

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

void PermuteComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const  {
  out->CopyCols(in, column_map_);
}
void PermuteComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &, //in_value
                                const CuMatrixBase<BaseFloat> &, // out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *to_update,
                                CuMatrixBase<BaseFloat> *in_deriv) const  {
  // computing the reverse column_map
  std::vector<int32> reverse_column_map(column_map_.Dim()),
                     column_map(column_map_.Dim());
  column_map_.CopyToVec(&column_map);
  int32 column_map_size = column_map.size();
  for (int32 i = 0; i < column_map_size; i++)  {
    reverse_column_map[column_map[i]] = i;
  }
  CuArray<int32> cu_reverse_column_map(reverse_column_map);
  in_deriv->CopyCols(out_deriv, cu_reverse_column_map);
}

void PermuteComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string new_column_order;
  ok = ok && cfl->GetValue("new-column-order", &new_column_order);
  std::vector<int32> column_map;
  SplitStringToIntegers(new_column_order, ",", true, &column_map);
  CuArray<int32> cu_column_map(column_map);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
	      << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(cu_column_map);
}

void PermuteComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PermuteComponent>", "<ColumnMap>");
  CuVector<BaseFloat> cu_column_map;
  cu_column_map.Read(is, binary);
  std::vector<int32> column_map(cu_column_map.Dim());
  for (int32 i = 0; i < cu_column_map.Dim(); i++)
    column_map[i] = static_cast<int32>(cu_column_map(i));
  column_map_.CopyFromVec(column_map);
  ExpectToken(is, binary, "</PermuteComponent>");
}

void PermuteComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PermuteComponent>");
  WriteToken(os, binary, "<ColumnMap>");
  std::ostringstream buffer;
  std::vector<int32> column_map(column_map_.Dim());
  column_map_.CopyToVec(&column_map);
  CuVector<BaseFloat> cu_column_map(column_map.size());
  for (int32 i = 0; i < column_map.size() -1; i++)
    cu_column_map(i) = static_cast<BaseFloat>(column_map[i]);
  cu_column_map.Write(os, binary);
  WriteToken(os, binary, "</PermuteComponent>");
}

std::string PermuteComponent::Info() const {
  std::stringstream stream;
  stream << Type() << ", dim=" << column_map_.Dim();
  stream << " , new-column-order=";
  std::vector<int32> column_map(column_map_.Dim());
  column_map_.CopyToVec(&column_map);
  CuVector<BaseFloat> cu_column_map(column_map.size());
  for (int32 i = 0; i < column_map.size() -1; i++)
    cu_column_map(i) = static_cast<BaseFloat>(column_map[i]);
  cu_column_map.Write(stream, false);

  return stream.str();
}


} // namespace nnet3
} // namespace kaldi

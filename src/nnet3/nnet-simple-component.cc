// nnet3/nnet-simple-component.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen
//                2015  Daniel Galvez

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
#include <iomanip>
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

void PnormComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0 &&
               input_dim_ % output_dim_ == 0);
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
  out->GroupPnorm(in, p);
}

void PnormComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)
    return;
  BaseFloat p = 2.0;
  in_deriv->DiffGroupPnorm(in_value, out_value, out_deriv, p);
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


void DropoutComponent::Init(int32 dim, BaseFloat dropout_proportion,
                            bool dropout_per_frame) {
  dropout_proportion_ = dropout_proportion;
  dropout_per_frame_ = dropout_per_frame;
  dim_ = dim;
}

void DropoutComponent::InitFromConfig(ConfigLine *cfl) {
  int32 dim = 0;
  BaseFloat dropout_proportion = 0.0;
  bool dropout_per_frame = false;
  bool ok = cfl->GetValue("dim", &dim) &&
    cfl->GetValue("dropout-proportion", &dropout_proportion);
  cfl->GetValue("dropout-per-frame", &dropout_per_frame);
    // for this stage, dropout is hard coded in
    // normal mode if not declared in config
  if (!ok || cfl->HasUnusedValues() || dim <= 0 ||
      dropout_proportion < 0.0 || dropout_proportion > 1.0)
       KALDI_ERR << "Invalid initializer for layer of type "
                 << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim, dropout_proportion, dropout_per_frame);
}

std::string DropoutComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", dropout-proportion=" << dropout_proportion_
         << ", dropout-per-frame=" << (dropout_per_frame_ ? "true" : "false");
  return stream.str();
}

void DropoutComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() && out->NumCols() == in.NumCols()
               && in.NumCols() == dim_);

  BaseFloat dropout = dropout_proportion_;
  KALDI_ASSERT(dropout >= 0.0 && dropout <= 1.0);
  if (!dropout_per_frame_) {
    // This const_cast is only safe assuming you don't attempt
    // to use multi-threaded code with the GPU.
    const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(out);

    out->Add(-dropout);  // now, a proportion "dropout" will be <0.0
    // apply the function (x>0?1:0).  Now, a proportion
    // "dropout" will be zero and (1 - dropout) will be 1.0.
    out->ApplyHeaviside();

    out->MulElements(in);
  } else {
    // randomize the dropout matrix by row,
    // i.e. [[1,1,1,1],[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]]
    CuMatrix<BaseFloat> tmp(1, out->NumRows(), kUndefined);
    // This const_cast is only safe assuming you don't attempt
    // to use multi-threaded code with the GPU.
    const_cast<CuRand<BaseFloat>&>(random_generator_).RandUniform(&tmp);
    tmp.Add(-dropout);
    tmp.ApplyHeaviside();
    out->CopyColsFromVec(tmp.Row(0));
    out->MulElements(in);
  }
}


void DropoutComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in_value,
                                const CuMatrixBase<BaseFloat> &out_value,
                                const CuMatrixBase<BaseFloat> &out_deriv,
                                Component *to_update,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(in_value.NumRows() == out_value.NumRows() &&
               in_value.NumCols() == out_value.NumCols());

  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == out_deriv.NumCols());
  in_deriv->SetMatMatDivMat(out_deriv, out_value, in_value);
}



void DropoutComponent::Read(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<DropoutComponent>") {
    ReadToken(is, binary, &token);
  }
  KALDI_ASSERT(token == "<Dim>");
  ReadBasicType(is, binary, &dim_);  // read dimension.
  ReadToken(is, binary, &token);
  KALDI_ASSERT(token == "<DropoutProportion>");
  ReadBasicType(is, binary, &dropout_proportion_);  // read dropout rate
  ReadToken(is, binary, &token);
  if (token == "<DropoutPerFrame>") {
    ReadBasicType(is, binary, &dropout_per_frame_);  // read dropout mode
    ReadToken(is, binary, &token);
    KALDI_ASSERT(token == "</DropoutComponent>");
  } else {
    dropout_per_frame_ = false;
    KALDI_ASSERT(token == "</DropoutComponent>");
  }
}

void DropoutComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<DropoutComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<DropoutProportion>");
  WriteBasicType(os, binary, dropout_proportion_);
  WriteToken(os, binary, "<DropoutPerFrame>");
  WriteBasicType(os, binary, dropout_per_frame_);
  WriteToken(os, binary, "</DropoutComponent>");
}

void SumReduceComponent::Init(int32 input_dim, int32 output_dim)  {
  input_dim_ = input_dim;
  output_dim_ = output_dim;
  KALDI_ASSERT(input_dim_ > 0 && output_dim_ > 0 &&
               input_dim_ % output_dim_ == 0);
}

void SumReduceComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  int32 output_dim = 0;
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim);
  if (!ok || cfl->HasUnusedValues() || output_dim <= 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, output_dim);
}


void SumReduceComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows() && in.NumCols() == input_dim_
               && out->NumCols() == output_dim_);
  int32 num_blocks = input_dim_ / output_dim_;
  for (int32 i = 0; i < num_blocks; i++) {
    CuSubMatrix<BaseFloat> in_block(in, 0, in.NumRows(),
                                    i * output_dim_, output_dim_);
    if (i == 0)
      out->CopyFromMat(in_block);
    else
      out->AddMat(1.0, in_block);
  }
}

void SumReduceComponent::Backprop(const std::string &debug_info,
                                  const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &, // in_value
                                  const CuMatrixBase<BaseFloat> &, // out_value
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *, // to_update
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  KALDI_ASSERT(out_deriv.NumRows() == in_deriv->NumRows() &&
               in_deriv->NumCols() == input_dim_ &&
               out_deriv.NumCols() == output_dim_);
  int32 num_blocks = input_dim_ / output_dim_;
  for (int32 i = 0; i < num_blocks; i++) {
    CuSubMatrix<BaseFloat> in_deriv_block(*in_deriv, 0, in_deriv->NumRows(),
                                          i * output_dim_, output_dim_);
    in_deriv_block.CopyFromMat(out_deriv);
  }
}

void SumReduceComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<SumReduceComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</SumReduceComponent>");
}

void SumReduceComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<SumReduceComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</SumReduceComponent>");
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

const BaseFloat NormalizeComponent::kSquaredNormFloor =
    pow(2.0, NormalizeComponent::kExpSquaredNormFloor);

// This component modifies the vector of activations by scaling it
// so that the root-mean-square equals 1.0.  It's important that its
// square root be exactly representable in float.
void NormalizeComponent::Init(int32 input_dim, BaseFloat target_rms,
                              bool add_log_stddev) {
  KALDI_ASSERT(input_dim > 0);
  KALDI_ASSERT(target_rms > 0);
  input_dim_ = input_dim;
  target_rms_ = target_rms;
  add_log_stddev_ = add_log_stddev;
}

NormalizeComponent::NormalizeComponent(const NormalizeComponent &other):
    input_dim_(other.input_dim_), target_rms_(other.target_rms_),
    add_log_stddev_(other.add_log_stddev_) { }

void NormalizeComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = 0;
  bool add_log_stddev = false;
  BaseFloat target_rms = 1.0;
  bool ok = cfl->GetValue("dim", &input_dim) ||
      cfl->GetValue("input-dim", &input_dim);
  cfl->GetValue("target-rms", &target_rms);
  cfl->GetValue("add-log-stddev", &add_log_stddev);
  if (!ok || cfl->HasUnusedValues() || input_dim <= 0 || target_rms <= 0.0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(input_dim, target_rms, add_log_stddev);
}

void NormalizeComponent::Read(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<NormalizeComponent>") {
    ReadToken(is, binary, &token);
  }
  KALDI_ASSERT(token == "<Dim>" || token == "<InputDim>");
  ReadBasicType(is, binary, &input_dim_); // Read dimension.
  ReadToken(is, binary, &token);
  // read target_rms_ if it is available.
  if (token == "<TargetRms>") {
    ReadBasicType(is, binary, &target_rms_);
    ReadToken(is, binary, &token);
  }
  //  Read add_log_stddev_ token, if it is available.
  if (token == "<AddLogStddev>") {
    ReadBasicType(is, binary, &add_log_stddev_);
    ReadToken(is, binary, &token);
  }
  if (token == "<ValueAvg>") {
    // back-compatibility code.
    CuVector<double> temp;
    temp.Read(is, binary);
    ExpectToken(is, binary, "<DerivAvg>");
    temp.Read(is, binary);
    ExpectToken(is, binary, "<Count>");
    double count;
    ReadBasicType(is, binary, &count);
    ReadToken(is, binary, &token);
  }
  KALDI_ASSERT(token == "</NormalizeComponent>");
}

void NormalizeComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NormalizeComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<TargetRms>");
  WriteBasicType(os, binary, target_rms_);
  WriteToken(os, binary, "<AddLogStddev>");
  WriteBasicType(os, binary, add_log_stddev_);
  WriteToken(os, binary, "</NormalizeComponent>");
}

std::string NormalizeComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim() << ", target-rms=" << target_rms_
         << ", add-log-stddev=" << std::boolalpha << add_log_stddev_;
  return stream.str();
}

// The output y_i = scale * x_i,
// and we want to RMS value of the y_i to equal target_rms,
// so y^t y = D * target_rms^2 (if y is one row of the input).
// we need to have scale = 1.0 / sqrt(x^t x / (D * target_rms^2)).
// there is also flooring involved, to avoid division-by-zero
// problems.  It's important for the backprop, that the floor's
// square root is exactly representable as float.
// If add_log_stddev_ is true, log(max(epsi, sqrt(x^t x / D)))
// is an extra dimension of the output.
void NormalizeComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumCols() == in.NumCols() + (add_log_stddev_ ? 1 : 0));
  cu::NormalizePerRow(in, target_rms_, add_log_stddev_, out);
}

/*
  A note on the derivative of NormalizeComponent...
  let both row_in and row_out be vectors of dimension D.
  Let p = row_in^T row_in / (D * target_rms^2), and let
  f = 1.0 / sqrt(max(kSquaredNormFloor, p)), and we compute row_out as:
  row_out = f row_in.
  Suppose we have a quantity deriv_out which is the derivative
  of the objective function w.r.t. row_out.  We want to compute
  deriv_in which is the derivative of the objective function w.r.t.
  row_in.  Let the objective function be F.  One term is obvious: we have
  deriv_in = f deriv_out + ....
  next we have to take into account the derivative that gets back-propagated
  through f.  Obviously, dF/df = deriv_out^T row_in.
  And df/dp = (p <= kSquaredNormFloor ? 0.0 : -0.5 p^{-1.5}) = (f == 1.0 / sqrt(kSquaredNormFloor) ? 0.0 : -0.5 f^3),
  and dp/d(row_in) = 2/(D * target_rms^2) row_in. [it's vector_valued].
  So this term in dF/d(row_in) equals:
  dF/df df/dp dp/d(row_in)   =    2/(D * target_rms^2) (f == 1.0 / sqrt(kSquaredNormFloor)  ? 0.0 : -0.5 f^3) (deriv_out^T row_in) row_in
  So
  deriv_in = f deriv_out + (f == 1.0 ? 0.0 : -f^3  / (D * target_rms^2) ) (deriv_out^T row_in) row_in

  if add_log_stddev_ true, the deriv_in has another term as
  dF/dx_i = dF/df . df/dx_i => df/dx_i = x_i/(x^T x)
*/
void NormalizeComponent::Backprop(const std::string &debug_info,
                                  const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &, // out_value
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *to_update,
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)  return;
  const CuSubMatrix<BaseFloat> out_deriv_no_log(out_deriv,
                                                0, out_deriv.NumRows(),
                                                0, input_dim_);
  CuVector<BaseFloat> dot_products(out_deriv.NumRows());
  dot_products.AddDiagMatMat(1.0, out_deriv_no_log, kNoTrans,
                             in_value, kTrans, 0.0);
  CuVector<BaseFloat> in_norm(in_value.NumRows());
  BaseFloat d_scaled = (in_value.NumCols() * target_rms_ * target_rms_);
  in_norm.AddDiagMat2(1.0, in_value, kNoTrans, 0.0);

  if (add_log_stddev_) {
    CuVector<BaseFloat> log_stddev_deriv(in_norm), // log_stddev deriv as dF/dy .* (x^T x)^-1
        out_deriv_for_stddev(out_deriv.NumRows(), kUndefined);
    // f = log(sqrt(max(epsi, x^T x / D)))
    // df/dx = epsi^2 * D < x^T x ? (1/(x^T x)) * x  : 0.
    // we don't compute this exactly below for the case wehn x^2 x is very
    // small, but we do make sure that the deriv isn't infinity when the input
    // is zero.
    log_stddev_deriv.ApplyFloor(input_dim_ * kSquaredNormFloor);
    log_stddev_deriv.ApplyPow(-1.0);
    out_deriv_for_stddev.CopyColFromMat(out_deriv, (out_deriv.NumCols() - 1));
    log_stddev_deriv.MulElements(out_deriv_for_stddev);
    if (in_deriv)
      in_deriv->AddDiagVecMat(1.0, log_stddev_deriv, in_value, kNoTrans, 1.0);
  }
  in_norm.Scale(1.0 / d_scaled);
  in_norm.ApplyFloor(kSquaredNormFloor);
  in_norm.ApplyPow(-0.5);
  if (in_deriv) {
    if (in_deriv->Data() != out_deriv_no_log.Data())
      in_deriv->AddDiagVecMat(1.0, in_norm, out_deriv_no_log, kNoTrans, 1.0);
    else
      in_deriv->MulRowsVec(in_norm);
    in_norm.ReplaceValue(1.0 / sqrt(kSquaredNormFloor), 0.0);
    in_norm.ApplyPow(3.0);
    dot_products.MulElements(in_norm);

    in_deriv->AddDiagVecMat(-1.0 / d_scaled,
                            dot_products, in_value,
                            kNoTrans, 1.0);
  }
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
                                Component *to_update_in,
                                CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    in_deriv->DiffSigmoid(out_value, out_deriv);
    SigmoidComponent *to_update = dynamic_cast<SigmoidComponent*>(to_update_in);
    if (to_update != NULL)
      RepairGradients(out_value, in_deriv, to_update);
  }
}

void SigmoidComponent::RepairGradients(
    const CuMatrixBase<BaseFloat> &out_value,
    CuMatrixBase<BaseFloat> *in_deriv,
    SigmoidComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  // maximum possible derivative of SigmoidComponent is 0.25.
  // the default lower-threshold on the derivative, below which we
  // add a term to the derivative to encourage the inputs to the sigmoid
  // to be closer to zero, is 0.05, which means the derivative is on average
  // 5 times smaller than its maximum possible value.
  BaseFloat default_lower_threshold = 0.05;

  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;

  to_update->num_dims_processed_ += dim_;

  if (self_repair_scale_ == 0.0 || count_ == 0.0 || deriv_sum_.Dim() != dim_ ||
      RandUniform() > repair_probability)
    return;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold; // -1000.0
  BaseFloat lower_threshold = (self_repair_lower_threshold_ == unset ?
                               default_lower_threshold :
                               self_repair_lower_threshold_) *
      count_;
  if (self_repair_upper_threshold_ != unset) {
    KALDI_ERR << "Do not set the self-repair-upper-threshold for sigmoid "
              << "components, it does nothing.";
  }

  // thresholds_vec is actually a 1-row matrix.  (the ApplyHeaviside
  // function isn't defined for vectors).
  CuMatrix<BaseFloat> thresholds(1, dim_);
  CuSubVector<BaseFloat> thresholds_vec(thresholds, 0);
  thresholds_vec.AddVec(-1.0, deriv_sum_);
  thresholds_vec.Add(lower_threshold);
  thresholds.ApplyHeaviside();
  to_update->num_dims_self_repaired_ += thresholds_vec.Sum();

  // At this point, 'thresholds_vec' contains a 1 for each dimension of
  // the output that is 'problematic', i.e. for which the avg-deriv
  // is less than the self-repair lower threshold, and a 0 for
  // each dimension that is not problematic.

  // what we want to do is to add
  // -self_repair_scale_ / repair_probability times (2 * output-valiue - 1.0)
  // to the input derivative for each problematic dimension.

  // Here, 2 * output - 1.0 is a version of the sigmoid that goes from -1.0 to
  // 1.0, like a tanh.  the negative sign is so that for inputs <0, we push them
  // up towards 0, and for inputs >0, we push them down towards 0.
  // Our use of this sigmoid-type function here is just a convenience since
  // we have it available.  We could use just about any function that is positive
  // for inputs < 0 and negative for inputs > 0.

  // We can rearrange the above as: for only the problematic columns,
  //   input-deriv -= 2 * self-repair-scale / repair-probabilty * output
  //   input-deriv +=  self-repair-scale / repair-probabilty
  // which we can write as:
  //   input-deriv -= 2 * self-repair-scale / repair-probabilty * output * thresholds-vec
  //   input-deriv +=  self-repair-scale / repair-probabilty * thresholds-vec

  in_deriv->AddMatDiagVec(-2.0 * self_repair_scale_ / repair_probability,
                          out_value, kNoTrans, thresholds_vec);
  in_deriv->AddVecToRows(self_repair_scale_ / repair_probability,
                         thresholds_vec);
}



void SigmoidComponent::StoreStats(const CuMatrixBase<BaseFloat> &out_value) {
  // only store stats about every other minibatch.
  if (RandInt(0, 1) == 0)
    return;
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
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<SelfRepairClippedProportionThreshold>") {
    ReadBasicType(is, binary, &self_repair_clipped_proportion_threshold_);
    ExpectToken(is, binary, "<SelfRepairTarget>");
    ReadBasicType(is, binary, &self_repair_target_);
    ExpectToken(is, binary, "<SelfRepairScale>");
    ReadBasicType(is, binary, &self_repair_scale_);
    ExpectToken(is, binary, "<NumElementsClipped>");
  } else {
    self_repair_clipped_proportion_threshold_ = 1.0;
    self_repair_target_ = 0.0;
    self_repair_scale_ = 0.0;
    KALDI_ASSERT(token == "<NumElementsClipped>");
  }
  ReadBasicType(is, binary, &num_clipped_);
  ExpectToken(is, binary, "<NumElementsProcessed>");
  ReadBasicType(is, binary, &count_);
  ReadToken(is, binary, &token);
  if (token == "<NumSelfRepaired>") {
    ReadBasicType(is, binary, &num_self_repaired_);
    ExpectToken(is, binary, "<NumBackpropped>");
    ReadBasicType(is, binary, &num_backpropped_);
    ExpectToken(is, binary, "</ClipGradientComponent>");
  } else {
    num_self_repaired_ = 0;
    num_backpropped_ = 0;
    KALDI_ASSERT(token == "</ClipGradientComponent>");
  }
}

void ClipGradientComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ClipGradientComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<ClippingThreshold>");
  WriteBasicType(os, binary, clipping_threshold_);
  WriteToken(os, binary, "<NormBasedClipping>");
  WriteBasicType(os, binary, norm_based_clipping_);
  WriteToken(os, binary, "<SelfRepairClippedProportionThreshold>");
  WriteBasicType(os, binary, self_repair_clipped_proportion_threshold_);
  WriteToken(os, binary, "<SelfRepairTarget>");
  WriteBasicType(os, binary, self_repair_target_);
  WriteToken(os, binary, "<SelfRepairScale>");
  WriteBasicType(os, binary, self_repair_scale_);
  WriteToken(os, binary, "<NumElementsClipped>");
  WriteBasicType(os, binary, num_clipped_);
  WriteToken(os, binary, "<NumElementsProcessed>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "<NumSelfRepaired>");
  WriteBasicType(os, binary, num_self_repaired_);
  WriteToken(os, binary, "<NumBackpropped>");
  WriteBasicType(os, binary, num_backpropped_);
  WriteToken(os, binary, "</ClipGradientComponent>");
}

std::string ClipGradientComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", norm-based-clipping="
         << (norm_based_clipping_ ? "true" : "false")
         << ", clipping-threshold=" << clipping_threshold_
         << ", clipped-proportion="
         << (count_ > 0 ? static_cast<BaseFloat>(num_clipped_)/count_ : 0);
  if (self_repair_scale_ != 0.0)
    stream << ", self-repair-clipped-proportion-threshold="
           << self_repair_clipped_proportion_threshold_
           << ", self-repair-target=" << self_repair_target_
           << ", self-repair-scale=" << self_repair_scale_;
  return stream.str();
}

void ClipGradientComponent::Init(int32 dim,
                                 BaseFloat clipping_threshold,
                                 bool norm_based_clipping,
                                 BaseFloat self_repair_clipped_proportion_threshold,
                                 BaseFloat self_repair_target,
                                 BaseFloat self_repair_scale,
                                 int32 num_clipped,
                                 int32 count,
                                 int32 num_self_repaired,
                                 int32 num_backpropped)  {
  KALDI_ASSERT(clipping_threshold >= 0 && dim > 0 &&
      self_repair_clipped_proportion_threshold >= 0.0 &&
      self_repair_target >= 0.0 && self_repair_scale >= 0.0);
  dim_ = dim;
  norm_based_clipping_ = norm_based_clipping;
  clipping_threshold_ = clipping_threshold;
  self_repair_clipped_proportion_threshold_ =
      self_repair_clipped_proportion_threshold;
  self_repair_target_ = self_repair_target;
  self_repair_scale_ = self_repair_scale;
  num_clipped_ = num_clipped;
  count_ = count;
  num_self_repaired_ = num_self_repaired;
  num_backpropped_ = num_backpropped;
}

void ClipGradientComponent::InitFromConfig(ConfigLine *cfl) {
  int32 dim = 0;
  bool ok = cfl->GetValue("dim", &dim);
  bool norm_based_clipping = false;
  BaseFloat clipping_threshold = 15.0;
  BaseFloat self_repair_clipped_proportion_threshold = 0.01;
  BaseFloat self_repair_target = 0.0;
  BaseFloat self_repair_scale = 1.0;
  cfl->GetValue("clipping-threshold", &clipping_threshold);
  cfl->GetValue("norm-based-clipping", &norm_based_clipping);
  cfl->GetValue("self-repair-clipped-proportion-threshold",
                &self_repair_clipped_proportion_threshold);
  cfl->GetValue("self-repair-target",
                &self_repair_target);
  cfl->GetValue("self-repair-scale", &self_repair_scale);
  if (!ok || cfl->HasUnusedValues() ||
      clipping_threshold < 0 || dim <= 0 ||
      self_repair_clipped_proportion_threshold < 0.0 ||
      self_repair_target < 0.0 || self_repair_scale < 0.0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(dim, clipping_threshold, norm_based_clipping,
       self_repair_clipped_proportion_threshold,
       self_repair_target,
       self_repair_scale, 0, 0, 0, 0);
}

void ClipGradientComponent::Propagate(
                                 const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
}


void ClipGradientComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &in_value,
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
        if (to_update != NULL)
          to_update->num_clipped_ += (clipping_scales.Dim() - num_not_scaled);
       }
      if (to_update != NULL)
        to_update->count_ += clipping_scales.Dim();
    } else {
      // each element of the derivative matrix, is clipped to be below the
      // clipping_threshold_
      in_deriv->ApplyCeiling(clipping_threshold_);
      in_deriv->ApplyFloor(-1 * clipping_threshold_);
    }

    if (to_update != NULL) {
      to_update->num_backpropped_ += 1;
      RepairGradients(debug_info, in_value, in_deriv, to_update);
    }
  }
}

// This function will add a self-repair term to in-deriv, attempting to shrink
// the maginitude of the input towards self_repair_target_.
// This term is proportional to [-(input vector - self_repair_target_)].
// The avarage magnitude of this term is equal to
// [self_repair_scale_ * clipped_proportion * average norm of input derivative].
// We use norm of input derivative when computing the magnitude so that it is
// comparable to the magnitude of input derivative, especially when the gradient
// explosion is actually happening.
void ClipGradientComponent::RepairGradients(
    const std::string &debug_info,
    const CuMatrixBase<BaseFloat> &in_value,
    CuMatrixBase<BaseFloat> *in_deriv, ClipGradientComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);

  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;
  if (self_repair_clipped_proportion_threshold_ >= 1.0 ||
      self_repair_scale_ == 0.0 || count_ == 0 ||
      RandUniform() > repair_probability)
    return;

  KALDI_ASSERT(self_repair_target_ >= 0.0 && self_repair_scale_ > 0.0);

  BaseFloat clipped_proportion =
    (count_ > 0 ? static_cast<BaseFloat>(num_clipped_) / count_ : 0);
  // in-deriv would be modified only when clipped_proportion exceeds the
  // threshold
  if (clipped_proportion <= self_repair_clipped_proportion_threshold_)
    return;

  to_update->num_self_repaired_ += 1;
  if (to_update->debug_info_ == "") // get the component-node name
    to_update->debug_info_ = debug_info;
  if (to_update->num_self_repaired_ == 1)
    KALDI_LOG << "ClipGradientComponent(node_name=" << debug_info
              << ")'s self-repair was activated as the first time at the "
              << to_update->num_backpropped_
              << "-th call of Backprop() in this training job.";

  // sign_mat = sign(in_value), i.e.,
  // An element in sign_mat is 1 if its corresponding element in in_value > 0,
  // or -1 otherwise
  CuMatrix<BaseFloat> sign_mat(in_value);
  sign_mat.ApplyHeaviside();
  sign_mat.Scale(2.0);
  sign_mat.Add(-1.0);

  // repair_mat =
  // floor(abs(in_value) - self_repair_target_, 0) .* sign(in_value)
  CuMatrix<BaseFloat> repair_mat(in_value);
  repair_mat.ApplyPowAbs(1.0);
  repair_mat.Add(-self_repair_target_);
  repair_mat.ApplyFloor(0.0);
  repair_mat.MulElements(sign_mat);

  // magnitude =
  // self_repair_scale_ * clipped_proportion * average norm of in-deriv
  CuVector<BaseFloat> in_deriv_norm_vec(in_deriv->NumRows());
  in_deriv_norm_vec.AddDiagMat2(1.0, *in_deriv, kNoTrans, 0.0);
  in_deriv_norm_vec.ApplyPow(0.5);
  double in_deriv_norm_sum = in_deriv_norm_vec.Sum();
  BaseFloat magnitude = self_repair_scale_ * clipped_proportion *
                        (in_deriv_norm_sum / in_deriv_norm_vec.Dim());

  CuVector<BaseFloat> repair_mat_norm_vec(repair_mat.NumRows());
  repair_mat_norm_vec.AddDiagMat2(1.0, repair_mat, kNoTrans, 0.0);
  repair_mat_norm_vec.ApplyPow(0.5);
  double repair_mat_norm_sum = repair_mat_norm_vec.Sum();
  double scale = 0.0;
  if (repair_mat_norm_sum != 0.0)
    scale = magnitude / (repair_mat_norm_sum / repair_mat_norm_vec.Dim());
  // repair_mat is scaled so that on average the rows have the norm
  // (magnitude / repair_probability). This will give higher magnitude of
  // self-repair to input vectors that have larger absolute value, which tend to
  // be those that are diverging.
  in_deriv->AddMat(-scale / repair_probability, repair_mat);
  CuVector<BaseFloat> in_deriv_repaired_norm_vec(in_deriv->NumRows());
  in_deriv_repaired_norm_vec.AddDiagMat2(1.0, *in_deriv, kNoTrans, 0.0);
  in_deriv_repaired_norm_vec.ApplyPow(0.5);
  // scale in_deriv to have the same norm as that before adding the self-repair
  // term, in order to avoid increase of the norm caused by self-repair,
  // which may incur more clip of gradient and thus more self-repair
  double in_deriv_repaired_norm_sum = in_deriv_repaired_norm_vec.Sum();
  if (in_deriv_repaired_norm_sum != 0.0)
    in_deriv->Scale(in_deriv_norm_sum / in_deriv_repaired_norm_sum);
}

void ClipGradientComponent::ZeroStats()  {
  count_ = 0.0;
  num_clipped_ = 0.0;
  num_self_repaired_ = 0;
  num_backpropped_ = 0;
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


void TanhComponent::RepairGradients(
    const CuMatrixBase<BaseFloat> &out_value,
    CuMatrixBase<BaseFloat> *in_deriv,
    TanhComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  // maximum possible derivative of SigmoidComponent is 1.0
  // the default lower-threshold on the derivative, below which we
  // add a term to the derivative to encourage the inputs to the sigmoid
  // to be closer to zero, is 0.2, which means the derivative is on average
  // 5 times smaller than its maximum possible value.
  BaseFloat default_lower_threshold = 0.2;

  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;

  to_update->num_dims_processed_ += dim_;

  if (self_repair_scale_ == 0.0 || count_ == 0.0 || deriv_sum_.Dim() != dim_ ||
      RandUniform() > repair_probability)
    return;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold; // -1000.0
  BaseFloat lower_threshold = (self_repair_lower_threshold_ == unset ?
                               default_lower_threshold :
                               self_repair_lower_threshold_) *
      count_;
  if (self_repair_upper_threshold_ != unset) {
    KALDI_ERR << "Do not set the self-repair-upper-threshold for sigmoid "
              << "components, it does nothing.";
  }

  // thresholds_vec is actually a 1-row matrix.  (the ApplyHeaviside
  // function isn't defined for vectors).
  CuMatrix<BaseFloat> thresholds(1, dim_);
  CuSubVector<BaseFloat> thresholds_vec(thresholds, 0);
  thresholds_vec.AddVec(-1.0, deriv_sum_);
  thresholds_vec.Add(lower_threshold);
  thresholds.ApplyHeaviside();
  to_update->num_dims_self_repaired_ += thresholds_vec.Sum();

  // At this point, 'thresholds_vec' contains a 1 for each dimension of
  // the output that is 'problematic', i.e. for which the avg-deriv
  // is less than the self-repair lower threshold, and a 0 for
  // each dimension that is not problematic.

  // what we want to do is to add -self_repair_scale_ / repair_probability times
  // output-valiue) to the input derivative for each problematic dimension.
  // note that for the tanh, the output-value goes from -1.0 when the input is
  // -inf to +1.0 when the input is +inf.  The negative sign is so that for
  // inputs <0, we push them up towards 0, and for inputs >0, we push them down
  // towards 0.  Our use of the tanh here is just a convenience since we have it
  // available.  We could use just about any function that is positive for
  // inputs < 0 and negative for inputs > 0.

  // We can rearrange the above as: for only the problematic columns,
  //   input-deriv -= self-repair-scale / repair-probabilty * output
  // which we can write as:
  //   input-deriv -=  self-repair-scale / repair-probabilty * output * thresholds-vec

  in_deriv->AddMatDiagVec(-self_repair_scale_ / repair_probability,
                          out_value, kNoTrans, thresholds_vec);
}

void TanhComponent::Backprop(const std::string &debug_info,
                             const ComponentPrecomputedIndexes *indexes,
                             const CuMatrixBase<BaseFloat> &,
                             const CuMatrixBase<BaseFloat> &out_value,
                             const CuMatrixBase<BaseFloat> &out_deriv,
                             Component *to_update_in, // may be NULL; may be identical
                             // to "this" or different.
                             CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    in_deriv->DiffTanh(out_value, out_deriv);
    TanhComponent *to_update = dynamic_cast<TanhComponent*>(to_update_in);
    if (to_update != NULL)
      RepairGradients(out_value, in_deriv, to_update);
  }
}

/*
  Note on the derivative of the tanh function:
  tanh'(x) = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)

  The element by element equation of what we're doing would be:
  in_deriv = out_deriv * (1.0 - out_value^2).
  We can accomplish this via calls to the matrix library. */
void TanhComponent::StoreStats(const CuMatrixBase<BaseFloat> &out_value) {
  // only store stats about every other minibatch.
  if (RandInt(0, 1) == 0)
    return;
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
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    in_deriv->Heaviside(out_value);
    in_deriv->MulElements(out_deriv);
    RectifiedLinearComponent *to_update =
        dynamic_cast<RectifiedLinearComponent*>(to_update_in);
    if (to_update != NULL)
      RepairGradients(in_deriv, to_update);
  }
}


void RectifiedLinearComponent::RepairGradients(
    CuMatrixBase<BaseFloat> *in_deriv,
    RectifiedLinearComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  BaseFloat default_lower_threshold = 0.05,
      default_upper_threshold = 0.95;
  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;

  to_update->num_dims_processed_ += dim_;

  if (self_repair_scale_ == 0.0 || count_ == 0.0 || deriv_sum_.Dim() != dim_ ||
      RandUniform() > repair_probability)
    return;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold; // -1000.0
  BaseFloat lower_threshold = (self_repair_lower_threshold_ == unset ?
                               default_lower_threshold :
                               self_repair_lower_threshold_) *
      count_,
      upper_threshold = (self_repair_upper_threshold_ == unset ?
                         default_upper_threshold :
                         self_repair_upper_threshold_) *
      count_;

  CuMatrix<BaseFloat> storage(2, dim_ + 2, kUndefined);
  CuSubVector<BaseFloat> thresholds_vec(storage.RowData(0) + dim_, 2);
  CuSubMatrix<BaseFloat> stats_mat(storage, 0, 2, 0, dim_);
  thresholds_vec(0) = -lower_threshold;
  thresholds_vec(1) = -upper_threshold;
  CuSubVector<BaseFloat> row0(stats_mat, 0);
  CuSubVector<BaseFloat> row1(stats_mat, 1);

  row0.CopyFromVec(deriv_sum_);
  row1.CopyFromVec(row0);
  stats_mat.AddVecToCols(1.0, thresholds_vec, 1.0);
  // now row0 equals stats - lower_threshold, and
  //     row1 equals stats - upper_threshold.
  stats_mat.ApplyHeaviside();
  // now row0 equals (stats > lower_threshold ? 1 : 0), and
  //     row1 equals (stats > upper_threshold ? 1 : 0).
  // what we want is:
  // self_repair_scale * ((stats <= lower_threshold ? 1 : 0) +
  //                         (stats > upper_threshold ? -1 : 0)).
  //
  // we can get these in stats_mat.Row(0) by computing:
  // -self_repair_scale * (stats_mat.Row(1)  + stats_mat.Row(0) - 1).
  row0.AddVec(1.0, row1, 1.0);
  row0.Add(-1.0);
  CuVector<BaseFloat> temp(row0);
  temp.ApplyPow(2.0);
  to_update->num_dims_self_repaired_ += temp.Sum();
  // [actually we need to divide by repair_probability also, to
  //  correct for the fact that we only do this on some frames.]
  row0.Scale(-self_repair_scale_ / repair_probability);
  in_deriv->AddVecToRows(1.0, row0, 1.0);
}


void RectifiedLinearComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &out_value) {
  // only store stats about every other minibatch.
  if (RandInt(0, 1) == 0)
    return;
  CuMatrix<BaseFloat> temp_deriv(out_value.NumRows(),
                                 out_value.NumCols(),
                                 kUndefined);
  temp_deriv.Heaviside(out_value);
  StoreStatsInternal(out_value, &temp_deriv);
}

void EluComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  // Apply exponential linear function (x >= 0 ? x : exp(x) - 1)
  out->CopyFromMat(in);
  CuMatrix<BaseFloat> exp_negated_relu(in);
  out->ApplyFloor(0.0);
  exp_negated_relu.ApplyCeiling(0.0);
  exp_negated_relu.ApplyExp();
  out->AddMat(1.0, exp_negated_relu, kNoTrans);
  out->Add(-1.0);
}

void EluComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv != NULL) {
    in_deriv->CopyFromMat(out_value);
    in_deriv->ApplyCeiling(0.0);
    in_deriv->Add(1.0);
    in_deriv->MulElements(out_deriv);
    EluComponent *to_update =
        dynamic_cast<EluComponent*>(to_update_in);
    if (to_update != NULL)
      RepairGradients(in_deriv, to_update);
  }
}


void EluComponent::RepairGradients(
    CuMatrixBase<BaseFloat> *in_deriv,
    EluComponent *to_update) const {
  KALDI_ASSERT(to_update != NULL);
  BaseFloat default_lower_threshold = 0.05,
      default_upper_threshold = 0.95;
  // we use this 'repair_probability' (hardcoded for now) to limit
  // this code to running on about half of the minibatches.
  BaseFloat repair_probability = 0.5;

  to_update->num_dims_processed_ += dim_;

  if (self_repair_scale_ == 0.0 || count_ == 0.0 || deriv_sum_.Dim() != dim_ ||
      RandUniform() > repair_probability)
    return;

  // check that the self-repair scale is in a reasonable range.
  KALDI_ASSERT(self_repair_scale_ > 0.0 && self_repair_scale_ < 0.1);
  BaseFloat unset = kUnsetThreshold; // -1000.0
  BaseFloat lower_threshold = (self_repair_lower_threshold_ == unset ?
                               default_lower_threshold :
                               self_repair_lower_threshold_) *
      count_,
      upper_threshold = (self_repair_upper_threshold_ == unset ?
                         default_upper_threshold :
                         self_repair_upper_threshold_) *
      count_;

  CuMatrix<BaseFloat> storage(2, dim_ + 2, kUndefined);
  CuSubVector<BaseFloat> thresholds_vec(storage.RowData(0) + dim_, 2);
  CuSubMatrix<BaseFloat> stats_mat(storage, 0, 2, 0, dim_);
  thresholds_vec(0) = -lower_threshold;
  thresholds_vec(1) = -upper_threshold;
  CuSubVector<BaseFloat> row0(stats_mat, 0);
  CuSubVector<BaseFloat> row1(stats_mat, 1);

  row0.CopyFromVec(deriv_sum_);
  row1.CopyFromVec(row0);
  stats_mat.AddVecToCols(1.0, thresholds_vec, 1.0);
  // now row0 equals stats - lower_threshold, and
  //     row1 equals stats - upper_threshold.
  stats_mat.ApplyHeaviside();
  // now row0 equals (stats > lower_threshold ? 1 : 0), and
  //     row1 equals (stats > upper_threshold ? 1 : 0).
  // what we want is:
  // self_repair_scale * ((stats <= lower_threshold ? 1 : 0) +
  //                         (stats > upper_threshold ? -1 : 0)).
  //
  // we can get these in stats_mat.Row(0) by computing:
  // -self_repair_scale * (stats_mat.Row(1)  + stats_mat.Row(0) - 1).
  row0.AddVec(1.0, row1, 1.0);
  row0.Add(-1.0);
  CuVector<BaseFloat> temp(row0);
  temp.ApplyPow(2.0);
  to_update->num_dims_self_repaired_ += temp.Sum();
  // [actually we need to divide by repair_probability also, to
  //  correct for the fact that we only do this on some frames.]
  row0.Scale(-self_repair_scale_ / repair_probability);
  in_deriv->AddVecToRows(1.0, row0, 1.0);
}


void EluComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &out_value) {
  // only store stats about every other minibatch.
  if (RandInt(0, 1) == 0)
    return;
  CuMatrix<BaseFloat> temp_deriv(out_value.NumRows(),
                                 out_value.NumCols(),
                                 kUndefined);
  temp_deriv.CopyFromMat(out_value);
  temp_deriv.ApplyCeiling(0.0);
  temp_deriv.Add(1.0);
  StoreStatsInternal(out_value, &temp_deriv);
}


void AffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    // If scale == 0.0 we call SetZero() which will get rid of NaN's and inf's.
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
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
    linear_params_(linear_params),
    bias_params_(bias_params) {
  SetUnderlyingLearningRate(learning_rate);
  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
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
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

Component* AffineComponent::Copy() const {
  AffineComponent *ans = new AffineComponent(*this);
  return ans;
}

BaseFloat AffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void AffineComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

void AffineComponent::Init(std::string matrix_filename) {
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
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
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
    Init(input_dim, output_dim,
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
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</AffineComponent>");
}

void AffineComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
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

RepeatedAffineComponent::RepeatedAffineComponent(const RepeatedAffineComponent & component) :
    UpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_),
    num_repeats_(component.num_repeats_) {}


void RepeatedAffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void RepeatedAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const RepeatedAffineComponent *other =
      dynamic_cast<const RepeatedAffineComponent *>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void RepeatedAffineComponent::PerturbParams(BaseFloat stddev){
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string RepeatedAffineComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", num-repeats=" << num_repeats_;
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

Component* RepeatedAffineComponent::Copy() const {
  RepeatedAffineComponent *ans = new RepeatedAffineComponent(*this);
  return ans;
}

BaseFloat RepeatedAffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const RepeatedAffineComponent *other =
      dynamic_cast<const RepeatedAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
                     + VecVec(bias_params_, other->bias_params_);
}

void RepeatedAffineComponent::Init(int32 input_dim, int32 output_dim, int32 num_repeats,
                                   BaseFloat param_stddev, BaseFloat bias_mean,
                                   BaseFloat bias_stddev) {
  KALDI_ASSERT(input_dim % num_repeats == 0 && output_dim % num_repeats == 0);
  linear_params_.Resize(output_dim / num_repeats, input_dim / num_repeats);
  bias_params_.Resize(output_dim / num_repeats);
  num_repeats_ = num_repeats;
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  bias_params_.Add(bias_mean);
  SetNaturalGradientConfigs();
}


void RepeatedAffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  int32 num_repeats = num_repeats_;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  ok = cfl->GetValue("num-repeats", &num_repeats) && ok;
  ok = cfl->GetValue("input-dim", &input_dim) && ok;
  ok = cfl->GetValue("output-dim", &output_dim) && ok;
  KALDI_ASSERT(input_dim % num_repeats == 0 &&
               "num-repeats must divide input-dim");
  KALDI_ASSERT(output_dim % num_repeats == 0 &&
               "num-repeats must divide output-dim");
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim / num_repeats),
      bias_mean = 0.0, bias_stddev = 0.0;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-mean", &bias_mean);
  cfl->GetValue("bias-stddev", &bias_stddev);
  Init(input_dim, output_dim,
       num_repeats, param_stddev, bias_mean, bias_stddev);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void RepeatedAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                        const CuMatrixBase<BaseFloat> &in,
                                        CuMatrixBase<BaseFloat> *out) const {
  // we gave the kInputContiguous and kOutputContiguous flags-- check that they
  // are honored.
  KALDI_ASSERT(in.NumCols() == in.Stride() &&
               out->NumCols() == out->Stride() &&
               out->NumRows() == in.NumRows());

  int32 num_repeats = num_repeats_,
      num_rows = in.NumRows(),
      block_dim_out = linear_params_.NumRows(),
      block_dim_in = linear_params_.NumCols();

  CuSubMatrix<BaseFloat> in_reshaped(in.Data(), num_rows * num_repeats,
                                     block_dim_in, block_dim_in),
      out_reshaped(out->Data(), num_rows * num_repeats,
                   block_dim_out, block_dim_out);

  out_reshaped.CopyRowsFromVec(bias_params_);

  out_reshaped.AddMatMat(1.0, in_reshaped, kNoTrans,
                         linear_params_, kTrans, 1.0);
}

void RepeatedAffineComponent::Backprop(const std::string &debug_info,
                                       const ComponentPrecomputedIndexes *indexes,
                                       const CuMatrixBase<BaseFloat> &in_value,
                                       const CuMatrixBase<BaseFloat> &, // out_value
                                       const CuMatrixBase<BaseFloat> &out_deriv,
                                       Component *to_update_in,
                                       CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(out_deriv.NumCols() == out_deriv.Stride() &&
       (in_value.NumCols() == 0 || in_value.NumCols() == in_value.Stride()) &&
               (!in_deriv || in_deriv->NumCols() == in_deriv->Stride()));

  RepeatedAffineComponent *to_update = dynamic_cast<RepeatedAffineComponent*>(
      to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv) {
    int32 num_repeats = num_repeats_,
        num_rows = out_deriv.NumRows(),
        block_dim_out = linear_params_.NumRows(),
        block_dim_in = linear_params_.NumCols();

    CuSubMatrix<BaseFloat> in_deriv_reshaped(in_deriv->Data(),
                                             num_rows * num_repeats,
                                             block_dim_in, block_dim_in),
        out_deriv_reshaped(out_deriv.Data(),
                           num_rows * num_repeats,
                           block_dim_out, block_dim_out);
    in_deriv_reshaped.AddMatMat(1.0, out_deriv_reshaped, kNoTrans,
                                linear_params_, kNoTrans, 1.0);
  }

  // Next update the model (must do this 2nd so the derivatives we propagate are
  // accurate, in case this == to_update_in.)
  if (to_update != NULL)
    to_update->Update(in_value, out_deriv);
}

void RepeatedAffineComponent::Update(const CuMatrixBase<BaseFloat> &in_value,
                                     const CuMatrixBase<BaseFloat> &out_deriv) {
  KALDI_ASSERT(out_deriv.NumCols() == out_deriv.Stride() &&
               in_value.NumCols() == in_value.Stride() &&
               in_value.NumRows() == out_deriv.NumRows());


    int32 num_repeats = num_repeats_,
        num_rows = in_value.NumRows(),
        block_dim_out = linear_params_.NumRows(),
        block_dim_in = linear_params_.NumCols();

    CuSubMatrix<BaseFloat> in_value_reshaped(in_value.Data(),
                                             num_rows * num_repeats,
                                             block_dim_in, block_dim_in),
        out_deriv_reshaped(out_deriv.Data(),
                           num_rows * num_repeats,
                           block_dim_out, block_dim_out);


  linear_params_.AddMatMat(learning_rate_, out_deriv_reshaped, kTrans,
                           in_value_reshaped, kNoTrans, 1.0);
  bias_params_.AddRowSumMat(learning_rate_,
                            out_deriv_reshaped);
}

void RepeatedAffineComponent::Read(std::istream &is, bool binary) {
  // This Read function also works for NaturalGradientRepeatedAffineComponent.
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<NumRepeats>");
  ReadBasicType(is, binary, &num_repeats_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, std::string("</") + Type() + std::string(">"));
  SetNaturalGradientConfigs();
}

void RepeatedAffineComponent::Write(std::ostream &os, bool binary) const {
  // This Write function also works for NaturalGradientRepeatedAffineComponent.
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<NumRepeats>");
  WriteBasicType(os, binary, num_repeats_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  // write closing token.
  WriteToken(os, binary, std::string("</") + Type() + std::string(">"));
}

int32 RepeatedAffineComponent::NumParameters() const {
  // Note: unlike AffineComponent, InputDim() & OutputDim() are not used here and below,
  // for they are multipled by num_repeats_.
  return linear_params_.NumCols() * linear_params_.NumRows() + bias_params_.Dim();
}

void RepeatedAffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, linear_params_.NumCols() * linear_params_.NumRows()).CopyRowsFromMat(linear_params_);
  params->Range(linear_params_.NumCols() * linear_params_.NumRows(),
                bias_params_.Dim()).CopyFromVec(bias_params_);
}

void RepeatedAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, linear_params_.NumCols() * linear_params_.NumRows()));
  bias_params_.CopyFromVec(params.Range(linear_params_.NumCols() * linear_params_.NumRows(),
                                        bias_params_.Dim()));
}

void NaturalGradientRepeatedAffineComponent::SetNaturalGradientConfigs() {
  int32 rank_in = 40;
  int32 input_dim = linear_params_.NumCols();
  if (rank_in > input_dim / 2)
    rank_in = input_dim / 2;
  if (rank_in < 1)
    rank_in = 1;
  preconditioner_in_.SetRank(rank_in);
  preconditioner_in_.SetUpdatePeriod(4);
}

NaturalGradientRepeatedAffineComponent::NaturalGradientRepeatedAffineComponent(
    const NaturalGradientRepeatedAffineComponent &other):
    RepeatedAffineComponent(other),
    preconditioner_in_(other.preconditioner_in_) { }

// virtual
Component* NaturalGradientRepeatedAffineComponent::Copy() const {
  return new NaturalGradientRepeatedAffineComponent(*this);
}

void NaturalGradientRepeatedAffineComponent::Update(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  KALDI_ASSERT(out_deriv.NumCols() == out_deriv.Stride() &&
               in_value.NumCols() == in_value.Stride() &&
               in_value.NumRows() == out_deriv.NumRows());

  int32 num_repeats = num_repeats_,
      num_rows = in_value.NumRows(),
      block_dim_out = linear_params_.NumRows(),
      block_dim_in = linear_params_.NumCols();

  CuSubMatrix<BaseFloat> in_value_reshaped(in_value.Data(),
                                           num_rows * num_repeats,
                                           block_dim_in, block_dim_in),
        out_deriv_reshaped(out_deriv.Data(),
                           num_rows * num_repeats,
                           block_dim_out, block_dim_out);

  CuVector<BaseFloat> bias_deriv(block_dim_out);
  bias_deriv.AddRowSumMat(1.0, out_deriv_reshaped);

  CuMatrix<BaseFloat> deriv(block_dim_out,
                            block_dim_in + 1);
  deriv.ColRange(0, block_dim_in).AddMatMat(
      1.0, out_deriv_reshaped, kTrans,
      in_value_reshaped, kNoTrans, 1.0);
  deriv.CopyColFromVec(bias_deriv, block_dim_in);

  BaseFloat scale = 1.0;
  if (!is_gradient_) {
    try {
      // Only apply the preconditioning/natural-gradient if we're not computing
      // the exact gradient.
      preconditioner_in_.PreconditionDirections(&deriv, NULL, &scale);
    } catch (...) {
      int32 num_bad_rows = 0;
      for (int32 i = 0; i < out_deriv.NumRows(); i++) {
        BaseFloat f = out_deriv.Row(i).Sum();
        if (!(f - f == 0)) num_bad_rows++;
      }
      KALDI_ERR << "Preonditioning failed, in_value sum is "
                << in_value.Sum() << ", out_deriv sum is " << out_deriv.Sum()
                << ", out_deriv has " << num_bad_rows << " bad rows.";
    }
  }
  linear_params_.AddMat(learning_rate_ * scale,
                        deriv.ColRange(0, block_dim_in));
  bias_deriv.CopyColFromMat(deriv, block_dim_in);
  bias_params_.AddVec(learning_rate_ * scale, bias_deriv);
}

BlockAffineComponent::BlockAffineComponent(const BlockAffineComponent &other) :
  UpdatableComponent(other),
  linear_params_(other.linear_params_),
  bias_params_(other.bias_params_),
  num_blocks_(other.num_blocks_) {}

BlockAffineComponent::BlockAffineComponent(const RepeatedAffineComponent &rac) :
  UpdatableComponent(rac),
  linear_params_(rac.num_repeats_ * rac.linear_params_.NumRows(),
                 rac.linear_params_.NumCols(), kUndefined),
  bias_params_(rac.num_repeats_ * rac.linear_params_.NumRows(), kUndefined),
  num_blocks_(rac.num_repeats_) {
  // copy rac's linear_params_ and bias_params_ to this.
  int32 num_rows_in_block = rac.linear_params_.NumRows();
  for(int32 block_counter = 0; block_counter < num_blocks_; block_counter++) {
    int32 row_offset = block_counter * num_rows_in_block;
    CuSubMatrix<BaseFloat> block = this->linear_params_.RowRange(row_offset,
                                                                 num_rows_in_block);
    block.CopyFromMat(rac.linear_params_);
    CuSubVector<BaseFloat> block_bias = this->bias_params_.Range(row_offset,
                                                                 num_rows_in_block);
    block_bias.CopyFromVec(rac.bias_params_);
  }
}

Component* BlockAffineComponent::Copy() const {
  BlockAffineComponent *ans = new BlockAffineComponent(*this);
  return ans;
}

std::string BlockAffineComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", num-blocks=" << num_blocks_;
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

void BlockAffineComponent::Init(int32 input_dim,
                                int32 output_dim, int32 num_blocks,
                                BaseFloat param_stddev, BaseFloat bias_mean,
                                BaseFloat bias_stddev) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0 && num_blocks >= 1);
  KALDI_ASSERT(output_dim % num_blocks == 0 && input_dim % num_blocks == 0);
  const int32 num_columns_per_block = input_dim / num_blocks;
  linear_params_.Resize(output_dim, num_columns_per_block);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(param_stddev >= 0.0 && bias_stddev >= 0.0);
  linear_params_.SetRandn();
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  bias_params_.Add(bias_mean);
  num_blocks_ = num_blocks;
}

void BlockAffineComponent::InitFromConfig(ConfigLine *cfl) {
  int32 input_dim = -1, output_dim = -1, num_blocks = -1;
  if(!cfl->GetValue("input-dim", &input_dim) ||
     !cfl->GetValue("output-dim", &output_dim) ||
     !cfl->GetValue("num-blocks", &num_blocks))
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  InitLearningRatesFromConfig(cfl);
  BaseFloat param_stddev = 1.0 / std::sqrt(input_dim / num_blocks),
      bias_mean = 0.0, bias_stddev = 1.0;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-stddev", &bias_stddev);
  cfl->GetValue("bias-mean", &bias_mean);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";

  Init(input_dim, output_dim, num_blocks,
       param_stddev, bias_mean, bias_stddev);
}

void BlockAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const {
  out->CopyRowsFromVec(bias_params_);
  // block_dimension is both the number of columns, and the number of rows,
  // of a block.
  int32 num_rows_in_block = linear_params_.NumRows() / num_blocks_;
  int32 num_cols_in_block = linear_params_.NumCols();
  std::vector<CuSubMatrix<BaseFloat> *> in_batch, out_batch,
    linear_params_batch;
  for(int block_counter = 0; block_counter < num_blocks_; block_counter++) {
    CuSubMatrix<BaseFloat> *in_block =
      new CuSubMatrix<BaseFloat>(in.ColRange(block_counter * num_cols_in_block,
                                   num_cols_in_block));
    in_batch.push_back(in_block);

    CuSubMatrix<BaseFloat> *out_block =
      new CuSubMatrix<BaseFloat>(out->ColRange(block_counter * num_rows_in_block,
                                    num_rows_in_block));
    out_batch.push_back(out_block);

    CuSubMatrix<BaseFloat> *linear_params_block =
      new CuSubMatrix<BaseFloat>(linear_params_.RowRange(block_counter * num_rows_in_block,
                                              num_rows_in_block));
    linear_params_batch.push_back(linear_params_block);
  }
  AddMatMatBatched<BaseFloat>(1.0, out_batch, in_batch, kNoTrans,
                              linear_params_batch, kTrans, 1.0);

  DeletePointers(&in_batch);
  DeletePointers(&out_batch);
  DeletePointers(&linear_params_batch);
}

void BlockAffineComponent::Backprop(const std::string &debug_info,
                                    const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in_value,
                                    const CuMatrixBase<BaseFloat> &, // out_value
                                    const CuMatrixBase<BaseFloat> &out_deriv,
                                    Component *to_update_in,
                                    CuMatrixBase<BaseFloat> *in_deriv) const {
  BlockAffineComponent *to_update = dynamic_cast<BlockAffineComponent*>(to_update_in);

  const int32 num_rows_in_block = linear_params_.NumRows() / num_blocks_;
  const int32 num_cols_in_block = linear_params_.NumCols();

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv) {
    std::vector<CuSubMatrix<BaseFloat> *> in_deriv_batch, out_deriv_batch, linear_params_batch;

    for(int block_counter = 0; block_counter < num_blocks_; block_counter++) {
      CuSubMatrix<BaseFloat> *in_deriv_block =
        new CuSubMatrix<BaseFloat>(in_deriv->ColRange(block_counter * num_cols_in_block,
                                                      num_cols_in_block));
      in_deriv_batch.push_back(in_deriv_block);

      CuSubMatrix<BaseFloat> *out_deriv_block =
        new CuSubMatrix<BaseFloat>(out_deriv.ColRange(block_counter * num_rows_in_block,
                                                       num_rows_in_block));
      out_deriv_batch.push_back(out_deriv_block);

      CuSubMatrix<BaseFloat> *linear_params_block =
        new CuSubMatrix<BaseFloat>(linear_params_.RowRange(block_counter * num_rows_in_block,
                                                          num_rows_in_block));
      linear_params_batch.push_back(linear_params_block);
    }

    AddMatMatBatched<BaseFloat>(1.0, in_deriv_batch, out_deriv_batch, kNoTrans,
                                linear_params_batch, kNoTrans, 1.0);

    DeletePointers(&in_deriv_batch);
    DeletePointers(&out_deriv_batch);
    DeletePointers(&linear_params_batch);
  }

  if (to_update != NULL) {

    { // linear params update

      std::vector<CuSubMatrix<BaseFloat> *> in_value_batch,
        out_deriv_batch, linear_params_batch;

      for (int block_counter = 0; block_counter < num_blocks_; block_counter++) {
        CuSubMatrix<BaseFloat> *in_value_block =
          new CuSubMatrix<BaseFloat>(in_value.ColRange(block_counter * num_cols_in_block,
                                                       num_cols_in_block));
        in_value_batch.push_back(in_value_block);

        CuSubMatrix<BaseFloat> *out_deriv_block =
          new CuSubMatrix<BaseFloat>(out_deriv.ColRange(block_counter * num_rows_in_block,
                                                        num_rows_in_block));
        out_deriv_batch.push_back(out_deriv_block);

        CuSubMatrix<BaseFloat> *linear_params_block =
          new CuSubMatrix<BaseFloat>(to_update->linear_params_.RowRange(block_counter * num_rows_in_block,
                                                                        num_rows_in_block));
        linear_params_batch.push_back(linear_params_block);
      }

      AddMatMatBatched<BaseFloat>(to_update->learning_rate_,
                                  linear_params_batch,
                                  out_deriv_batch, kTrans,
                                  in_value_batch, kNoTrans, 1.0);

      DeletePointers(&in_value_batch);
      DeletePointers(&out_deriv_batch);
      DeletePointers(&linear_params_batch);
    } // end linear params update

    { // bias update
      to_update->bias_params_.AddRowSumMat(to_update->learning_rate_,
                                           out_deriv, 1.0);
    } // end bias update
  }
}

void BlockAffineComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void BlockAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
  const BlockAffineComponent *other =
    dynamic_cast<const BlockAffineComponent *>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void BlockAffineComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);

  CuVector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

BaseFloat BlockAffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const BlockAffineComponent *other =
    dynamic_cast<const BlockAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans) +
    VecVec(bias_params_, other->bias_params_);
}

void BlockAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<NumBlocks>");
  ReadBasicType(is, binary, &num_blocks_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</BlockAffineComponent>");
}

void BlockAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<NumBlocks>");
  WriteBasicType(os, binary, num_blocks_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</BlockAffineComponent>");
}

int32 BlockAffineComponent::NumParameters() const {
  return linear_params_.NumCols() * linear_params_.NumRows() + bias_params_.Dim();
}

void BlockAffineComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  int32 num_linear_params = linear_params_.NumCols() * linear_params_.NumRows();
  int32 num_bias_params = bias_params_.Dim();
  params->Range(0, num_linear_params).CopyRowsFromMat(linear_params_);
  params->Range(num_linear_params, num_bias_params).CopyFromVec(bias_params_);
}

void BlockAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  int32 num_linear_params = linear_params_.NumCols() * linear_params_.NumRows();
  int32 num_bias_params = bias_params_.Dim();
  linear_params_.CopyRowsFromVec(params.Range(0, num_linear_params));
  bias_params_.CopyFromVec(params.Range(num_linear_params, num_bias_params));
}

void PerElementScaleComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    scales_.SetZero();
  } else {
    scales_.Scale(scale);
  }
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

void PerElementScaleComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_scales(scales_.Dim(), kUndefined);
  temp_scales.SetRandn();
  scales_.AddVec(stddev, temp_scales);
}

std::string PerElementScaleComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", scales-min=" << scales_.Min()
         << ", scales-max=" << scales_.Max();
  PrintParameterStats(stream, "scales", scales_, true);
  return stream.str();
}

Component* PerElementScaleComponent::Copy() const {
  return new PerElementScaleComponent(*this);
}

BaseFloat PerElementScaleComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const PerElementScaleComponent *other =
      dynamic_cast<const PerElementScaleComponent*>(&other_in);
  return VecVec(scales_, other->scales_);
}

void PerElementScaleComponent::Init(int32 dim,
                                    BaseFloat param_mean,
                                    BaseFloat param_stddev) {
  KALDI_ASSERT(dim > 0 && param_stddev >= 0.0);
  scales_.Resize(dim);
  scales_.SetRandn();
  scales_.Scale(param_stddev);
  scales_.Add(param_mean);
}

void PerElementScaleComponent::Init(std::string vector_filename) {
  CuVector<BaseFloat> vec;
  ReadKaldiObject(vector_filename, &vec); // will abort on failure.
  scales_.Resize(vec.Dim());
  scales_.CopyFromVec(vec);
}

void PerElementScaleComponent::InitFromConfig(ConfigLine *cfl) {
  std::string vector_filename;
  int32 dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("vector", &vector_filename)) {
    Init(vector_filename);
    if (cfl->GetValue("dim", &dim))
      KALDI_ASSERT(dim == InputDim() &&
                   "input-dim mismatch vs. vector.");
  } else {
    if(!cfl->GetValue("dim", &dim))
      KALDI_ERR << "'dim' not provided in the config line.";
    BaseFloat param_mean = 1.0, param_stddev = 0.0;
    cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);
    Init(dim, param_mean, param_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
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
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate.
  ExpectToken(is, binary, "<Params>");
  scales_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</PerElementScaleComponent>");
}

void PerElementScaleComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate.
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

void PerElementOffsetComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    offsets_.SetZero();
  } else {
    offsets_.Scale(scale);
  }
}


void PerElementOffsetComponent::Add(BaseFloat alpha,
                                   const Component &other_in) {
  const PerElementOffsetComponent *other =
      dynamic_cast<const PerElementOffsetComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  offsets_.AddVec(alpha, other->offsets_);
}

PerElementOffsetComponent::PerElementOffsetComponent(
    const PerElementOffsetComponent &component):
    UpdatableComponent(component),
    offsets_(component.offsets_) { }

void PerElementOffsetComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_offsets(offsets_.Dim(), kUndefined);
  temp_offsets.SetRandn();
  offsets_.AddVec(stddev, temp_offsets);
}

std::string PerElementOffsetComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", offsets-min=" << offsets_.Min()
         << ", offsets-max=" << offsets_.Max();
  PrintParameterStats(stream, "offsets", offsets_, true);
  return stream.str();
}

Component* PerElementOffsetComponent::Copy() const {
  return new PerElementOffsetComponent(*this);
}

BaseFloat PerElementOffsetComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const PerElementOffsetComponent *other =
      dynamic_cast<const PerElementOffsetComponent*>(&other_in);
  return VecVec(offsets_, other->offsets_);
}

void PerElementOffsetComponent::Init(int32 dim,
                                     BaseFloat param_mean,
                                     BaseFloat param_stddev) {
  KALDI_ASSERT(dim > 0 && param_stddev >= 0.0);
  offsets_.Resize(dim);
  offsets_.SetRandn();
  offsets_.Scale(param_stddev);
  offsets_.Add(param_mean);
}

void PerElementOffsetComponent::Init(std::string vector_filename) {
  CuVector<BaseFloat> vec;
  ReadKaldiObject(vector_filename, &vec); // will abort on failure.
  offsets_.Resize(vec.Dim());
  offsets_.CopyFromVec(vec);
}

void PerElementOffsetComponent::InitFromConfig(ConfigLine *cfl) {
  std::string vector_filename;
  int32 dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("vector", &vector_filename)) {
    Init(vector_filename);
    if (cfl->GetValue("dim", &dim))
      KALDI_ASSERT(dim == InputDim() &&
                   "input-dim mismatch vs. vector.");
  } else {
    if(!cfl->GetValue("dim", &dim))
      KALDI_ERR << "'dim' not provided in the config line.";
    BaseFloat param_mean = 0.0, param_stddev = 0.0;
    cfl->GetValue("param-mean", &param_mean);
    cfl->GetValue("param-stddev", &param_stddev);
    Init(dim, param_mean, param_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
}

void PerElementOffsetComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  out->AddVecToRows(1.0, offsets_);
}

void PerElementOffsetComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  PerElementOffsetComponent *to_update =
      dynamic_cast<PerElementOffsetComponent*>(to_update_in);

  if (in_deriv) {
    // Propagate the derivative back to the input.
    in_deriv->CopyFromMat(out_deriv);
  }

  if (to_update != NULL)
    to_update->offsets_.AddRowSumMat(to_update->learning_rate_, out_deriv);
}

void PerElementOffsetComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate
  ExpectToken(is, binary, "<Offsets>");
  offsets_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</PerElementOffsetComponent>");
}

void PerElementOffsetComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<Offsets>");
  offsets_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</PerElementOffsetComponent>");
}

int32 PerElementOffsetComponent::NumParameters() const {
  return InputDim();
}

void PerElementOffsetComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyFromVec(offsets_);
}

void PerElementOffsetComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  offsets_.CopyFromVec(params);
}

std::string ConstantFunctionComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", " << Type() << ", input-dim=" << InputDim()
         << ", output-dim=" << OutputDim()
         << ", is-updatable=" << std::boolalpha << is_updatable_
         << ", use-natural-gradient=" << std::boolalpha
         << use_natural_gradient_;
  PrintParameterStats(stream, "output", output_, true);
  return stream.str();
}

ConstantFunctionComponent::ConstantFunctionComponent():
    UpdatableComponent(), input_dim_(-1), is_updatable_(true),
    use_natural_gradient_(true) { }

ConstantFunctionComponent::ConstantFunctionComponent(
    const ConstantFunctionComponent &other):
    UpdatableComponent(other), input_dim_(other.input_dim_),
    output_(other.output_), is_updatable_(other.is_updatable_),
    use_natural_gradient_(other.use_natural_gradient_),
    preconditioner_(other.preconditioner_) { }

void ConstantFunctionComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->CopyRowsFromVec(output_);
}

void ConstantFunctionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  // we don't update in_deriv, since we set the flag
  // kBackpropAdds, and the output doesn't depend on the
  // input, so the input-derivative is zero.
  if (to_update_in) {
    ConstantFunctionComponent *to_update =
      dynamic_cast<ConstantFunctionComponent*>(to_update_in);
    if (to_update->is_updatable_) {
      // only do the update if the is_updatable_ flag is set.
      KALDI_ASSERT(to_update && to_update->is_updatable_);
      if (to_update->use_natural_gradient_ && !to_update->is_gradient_) {
        CuMatrix<BaseFloat> out_deriv_copy(out_deriv);
        BaseFloat scale = 1.0;
        to_update->preconditioner_.PreconditionDirections(&out_deriv_copy,
                                                          NULL, &scale);
        to_update->output_.AddRowSumMat(scale * to_update->learning_rate_,
                                        out_deriv_copy);
      } else {
        to_update->output_.AddRowSumMat(to_update->learning_rate_,
                                        out_deriv);
      }
    }
  }
}

void ConstantFunctionComponent::Read(std::istream &is, bool binary) {
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<ConstantFunctionComponent>") {
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
    ReadToken(is, binary, &token);
  } else {
    learning_rate_ = 0.001;
  }
  if (token == "<InputDim>") {
    ReadBasicType(is, binary, &input_dim_);
  } else {
    KALDI_ERR << "Expected token <InputDim>, got "
              << token;
  }
  ExpectToken(is, binary, "<Output>");
  output_.Read(is, binary);
  ExpectToken(is, binary, "<IsUpdatable>");
  ReadBasicType(is, binary, &is_updatable_);
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);
  ExpectToken(is, binary, "</ConstantFunctionComponent>");
}

void ConstantFunctionComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<Output>");
  output_.Write(os, binary);
  WriteToken(os, binary, "<IsUpdatable>");
  WriteBasicType(os, binary, is_updatable_);
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  WriteToken(os, binary, "</ConstantFunctionComponent>");
}

Component* ConstantFunctionComponent::Copy() const {
  return new ConstantFunctionComponent(*this);
}

void ConstantFunctionComponent::Scale(BaseFloat scale) {
  if (is_updatable_) {
    if (scale == 0.0) {
      output_.SetZero();
    } else {
      output_.Scale(scale);
    }
  }
}

void ConstantFunctionComponent::Add(BaseFloat alpha, const Component &other_in) {
  if (is_updatable_) {
    const ConstantFunctionComponent *other =
        dynamic_cast<const ConstantFunctionComponent*>(&other_in);
    KALDI_ASSERT(other != NULL);
    output_.AddVec(alpha, other->output_);
  }
}

void ConstantFunctionComponent::PerturbParams(BaseFloat stddev) {
  CuVector<BaseFloat> temp_output(output_.Dim(), kUndefined);
  temp_output.SetRandn();
  output_.AddVec(stddev, temp_output);
}

BaseFloat ConstantFunctionComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  KALDI_ASSERT(is_updatable_);
  const ConstantFunctionComponent *other =
      dynamic_cast<const ConstantFunctionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return VecVec(output_, other->output_);
}

void ConstantFunctionComponent::InitFromConfig(ConfigLine *cfl) {
  int32 output_dim = 0;
  InitLearningRatesFromConfig(cfl);
  bool ok = cfl->GetValue("output-dim", &output_dim) &&
      cfl->GetValue("input-dim", &input_dim_);
  cfl->GetValue("is-updatable", &is_updatable_);
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  BaseFloat output_mean = 0.0, output_stddev = 0.0;
  cfl->GetValue("output-mean", &output_mean);
  cfl->GetValue("output-stddev", &output_stddev);
  if (!ok || cfl->HasUnusedValues() || input_dim_ <= 0 ||
      output_dim <= 0) {
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
  }
  Vector<BaseFloat> output(output_dim);
  output.SetRandn();
  output.Scale(output_stddev);
  output.Add(output_mean);
  output_ = output;
}

int32 ConstantFunctionComponent::NumParameters() const {
  KALDI_ASSERT(is_updatable_);
  return output_.Dim();
}

void ConstantFunctionComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  params->CopyFromVec(output_);
}

void ConstantFunctionComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  output_.CopyFromVec(params);
}


NaturalGradientAffineComponent::NaturalGradientAffineComponent():
    max_change_per_sample_(0.0),
    update_count_(0.0), active_scaling_count_(0.0),
    max_change_scale_stats_(0.0) { }

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
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
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
    ReadToken(is, binary, &token);
  }
  if (token != "<NaturalGradientAffineComponent>" &&
      token != "</NaturalGradientAffineComponent>")
    KALDI_ERR << "Expected <NaturalGradientAffineComponent> or "
              << "</NaturalGradientAffineComponent>, got " << token;
  SetNaturalGradientConfigs();
}

void NaturalGradientAffineComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  BaseFloat num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_sample = 0.0;
  int32 input_dim = -1, output_dim = -1, rank_in = 20, rank_out = 80,
      update_period = 4;
  InitLearningRatesFromConfig(cfl);
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("max-change-per-sample", &max_change_per_sample);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);

  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(rank_in, rank_out, update_period,
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
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0, bias_mean = 0.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    cfl->GetValue("bias-mean", &bias_mean);
    Init(input_dim, output_dim, param_stddev,
         bias_stddev, bias_mean, rank_in, rank_out, update_period,
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
    int32 rank_in, int32 rank_out,
    int32 update_period, BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample,
    std::string matrix_filename) {
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
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev, BaseFloat bias_mean,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample) {
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0 &&
               bias_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  bias_params_.Add(bias_mean);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  if (max_change_per_sample > 0.0)
    KALDI_WARN << "You are setting a positive max_change_per_sample for "
               << "NaturalGradientAffineComponent. But it has been deprecated. "
               << "Please use max_change for all updatable components instead "
               << "to activate the per-component max change mechanism.";
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void NaturalGradientAffineComponent::Write(std::ostream &os,
                                           bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
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
  WriteToken(os, binary, "</NaturalGradientAffineComponent>");
}

std::string NaturalGradientAffineComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  stream << ", rank-in=" << rank_in_
         << ", rank-out=" << rank_out_
         << ", num_samples_history=" << num_samples_history_
         << ", update_period=" << update_period_
         << ", alpha=" << alpha_
         << ", max-change-per-sample=" << max_change_per_sample_;
  if (update_count_ > 0.0 && max_change_per_sample_ > 0.0) {
    stream << ", avg-scaling-factor=" << max_change_scale_stats_ / update_count_
           << ", active-scaling-portion="
           << active_scaling_count_ / update_count_;
  }
  return stream.str();
}

Component* NaturalGradientAffineComponent::Copy() const {
  return new NaturalGradientAffineComponent(*this);
}

NaturalGradientAffineComponent::NaturalGradientAffineComponent(
    const NaturalGradientAffineComponent &other):
    AffineComponent(other),
    rank_in_(other.rank_in_),
    rank_out_(other.rank_out_),
    update_period_(other.update_period_),
    num_samples_history_(other.num_samples_history_),
    alpha_(other.alpha_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_),
    max_change_per_sample_(other.max_change_per_sample_),
    update_count_(other.update_count_),
    active_scaling_count_(other.active_scaling_count_),
    max_change_scale_stats_(other.max_change_scale_stats_) {
  SetNaturalGradientConfigs();
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
  std::ostringstream stream;
  stream << Component::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
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
    int32 input_dim = -1, output_dim = -1;
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


FixedAffineComponent::FixedAffineComponent(const AffineComponent &c):
    linear_params_(c.LinearParams()),
    bias_params_(c.BiasParams()) { }

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

void SumGroupComponent::Init(int32 input_dim, int32 output_dim) {
  const int32 num_groups = output_dim;
  KALDI_ASSERT(input_dim % num_groups == 0);
  const int32 group_size = input_dim / num_groups;

  std::vector<Int32Pair> cpu_vec(num_groups);
  std::vector<int32> reverse_cpu_vec;
  int32 cur_index = 0;
  for (size_t i = 0; i < num_groups; i++) {
    cpu_vec[i].first = cur_index;
    cpu_vec[i].second = cur_index + group_size;
    cur_index += group_size;
    for (int32 j = cpu_vec[i].first; j < cpu_vec[i].second; j++)
      reverse_cpu_vec.push_back(i);
  }
  this->indexes_ = cpu_vec;
  this->reverse_indexes_ = reverse_cpu_vec;
  this->input_dim_ = input_dim;
  this->output_dim_ = num_groups;
}

void SumGroupComponent::InitFromConfig(ConfigLine *cfl) {
  std::vector<int32> sizes;
  bool has_sizes = cfl->GetValue("sizes", &sizes);
  if (has_sizes) {
    if (cfl->HasUnusedValues() || sizes.empty())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    this->Init(sizes);
  } else { // each group has the same size
    int32 input_dim = -1, output_dim = -1;
    if (!cfl->GetValue("input-dim", &input_dim) ||
        !cfl->GetValue("output-dim", &output_dim) || cfl->HasUnusedValues()) {
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    }
    Init(input_dim, output_dim);
  }
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
  in_deriv->DiffSoftmaxPerRow(out_value, out_deriv);
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
  in_deriv->DiffLogSoftmaxPerRow(out_value, out_deriv);
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
  std::ostringstream stream;
  stream << Component::Info();
  PrintParameterStats(stream, "scales", scales_, true);
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
  std::ostringstream stream;
  stream << Component::Info();
  PrintParameterStats(stream, "bias", bias_, true);
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
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
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
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
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
  std::ostringstream stream;
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
      max_change_per_minibatch = 0.0;
  cfl->GetValue("rank", &rank);
  cfl->GetValue("update-period", &update_period);
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("max-change-per-minibatch", &max_change_per_minibatch);
  InitLearningRatesFromConfig(cfl);
  std::string filename;
  // Accepts "scales" config (for filename) or "dim" -> random init, for testing.
  if (cfl->GetValue("scales", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    Init(filename, rank, update_period, num_samples_history,
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

    Init(dim, param_mean, param_stddev, rank, update_period,
         num_samples_history, alpha, max_change_per_minibatch);
  }
}

void NaturalGradientPerElementScaleComponent::Init(
    int32 dim, BaseFloat param_mean,
    BaseFloat param_stddev, int32 rank, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_minibatch) {
  PerElementScaleComponent::Init(dim, param_mean,
                                 param_stddev);
  preconditioner_.SetRank(rank);
  preconditioner_.SetUpdatePeriod(update_period);
  preconditioner_.SetNumSamplesHistory(num_samples_history);
  preconditioner_.SetAlpha(alpha);
  max_change_per_minibatch_ = max_change_per_minibatch;
  if (max_change_per_minibatch > 0.0)
    KALDI_WARN << "You are setting a positive max_change_per_minibatch for "
               << "NaturalGradientPerElementScaleComponent. But it has been deprecated. "
               << "Please use max_change for all updatable components instead "
               << "to activate the per-component max change mechanism.";
}

void NaturalGradientPerElementScaleComponent::Init(
    std::string vector_filename,
    int32 rank, int32 update_period, BaseFloat num_samples_history,
    BaseFloat alpha, BaseFloat max_change_per_minibatch) {
  PerElementScaleComponent::Init(vector_filename);
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
  SetUnderlyingLearningRate(learning_rate);
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
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step, int32 num_filters,
    TensorVectorizationType input_vectorization,
    BaseFloat param_stddev, BaseFloat bias_stddev) {
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
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step,
    TensorVectorizationType input_vectorization,
    std::string matrix_filename) {
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
  std::ostringstream stream;
  stream << UpdatableComponent::Info()
         << ", input-x-dim=" << input_x_dim_
         << ", input-y-dim=" << input_y_dim_
         << ", input-z-dim=" << input_z_dim_
         << ", filt-x-dim=" << filt_x_dim_
         << ", filt-y-dim=" << filt_y_dim_
         << ", filt-x-step=" << filt_x_step_
         << ", filt-y-step=" << filt_y_step_
         << ", input-vectorization=" << input_vectorization_
         << ", num-filters=" << filter_params_.NumRows();
  PrintParameterStats(stream, "filter-params", filter_params_);
  PrintParameterStats(stream, "bias-params", bias_params_, true);
  return stream.str();
}

// initialize the component using configuration file
void ConvolutionComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_x_dim = -1, input_y_dim = -1, input_z_dim = -1,
        filt_x_dim = -1, filt_y_dim = -1,
        filt_x_step = -1, filt_y_step = -1,
        num_filters = -1;
  std::string input_vectorization_order = "zyx";
  InitLearningRatesFromConfig(cfl);
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
    Init(input_x_dim, input_y_dim, input_z_dim,
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
    Init(input_x_dim, input_y_dim, input_z_dim,
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
      filter_params_, 0, filter_params_.NumRows(), 0, filter_params_.NumCols());
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
  AddMatMatBatched<BaseFloat>(1.0, tgt_batch, patch_batch,
                              kNoTrans, filter_params_batch,
                              kTrans, 1.0);
  // release memory
  delete filter_params_elem;
  for (int32 p = 0; p < tgt_batch.size(); p++) {
    delete tgt_batch[p];
    delete patch_batch[p];
  }
}

// scale the parameters
void ConvolutionComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    filter_params_.SetZero();
    bias_params_.SetZero();
  } else {
    filter_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
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
            } else {
              KALDI_ASSERT(input_vectorization_ == kYzx);
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
      filter_params_, 0, filter_params_.NumRows(), 0, filter_params_.NumCols());

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
  AddMatMatBatched<BaseFloat>(1.0, patch_deriv_batch,
                              out_deriv_batch, kNoTrans,
                              filter_params_batch, kNoTrans, 0.0);

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
              patch_number * filters_grad.NumRows(), filters_grad.NumRows())));

      input_patch_batch.push_back(new CuSubMatrix<BaseFloat>(
              input_patches.ColRange(patch_number * filter_dim, filter_dim)));
    }
  }

  AddMatMatBatched<BaseFloat>(1.0, filters_grad_batch, out_deriv_batch, kTrans,
                              input_patch_batch, kNoTrans, 1.0);

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

void ConvolutionComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate.
  ExpectToken(is, binary, "<InputXDim>");
  ReadBasicType(is, binary, &input_x_dim_);
  ExpectToken(is, binary, "<InputYDim>");
  ReadBasicType(is, binary, &input_y_dim_);
  ExpectToken(is, binary, "<InputZDim>");
  ReadBasicType(is, binary, &input_z_dim_);
  ExpectToken(is, binary, "<FiltXDim>");
  ReadBasicType(is, binary, &filt_x_dim_);
  ExpectToken(is, binary, "<FiltYDim>");
  ReadBasicType(is, binary, &filt_y_dim_);
  ExpectToken(is, binary, "<FiltXStep>");
  ReadBasicType(is, binary, &filt_x_step_);
  ExpectToken(is, binary, "<FiltYStep>");
  ReadBasicType(is, binary, &filt_y_step_);
  ExpectToken(is, binary, "<InputVectorization>");
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
    ExpectToken(is, binary, "</ConvolutionComponent>");
  } else {
    is_gradient_ = false;
    KALDI_ASSERT(tok == "</ConvolutionComponent>");
  }
}

void ConvolutionComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // write opening tag and learning rate.
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
  WriteToken(os, binary, "</ConvolutionComponent>");
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

void ConvolutionComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  int32 num_filter_params = filter_params_.NumCols() * filter_params_.NumRows();
  params->Range(0, num_filter_params).CopyRowsFromMat(filter_params_);
  params->Range(num_filter_params, bias_params_.Dim()).CopyFromVec(bias_params_);
}
void ConvolutionComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  int32 num_filter_params = filter_params_.NumCols() * filter_params_.NumRows();
  filter_params_.CopyRowsFromVec(params.Range(0, num_filter_params));
  bias_params_.CopyFromVec(params.Range(num_filter_params, bias_params_.Dim()));
}

// aquire input dim
int32 MaxpoolingComponent::InputDim() const {
  return input_x_dim_ * input_y_dim_ * input_z_dim_;
}

MaxpoolingComponent::MaxpoolingComponent(
    const MaxpoolingComponent &component):
    input_x_dim_(component.input_x_dim_),
    input_y_dim_(component.input_y_dim_),
    input_z_dim_(component.input_z_dim_),
    pool_x_size_(component.pool_x_size_),
    pool_y_size_(component.pool_y_size_),
    pool_z_size_(component.pool_z_size_),
    pool_x_step_(component.pool_x_step_),
    pool_y_step_(component.pool_y_step_),
    pool_z_step_(component.pool_z_step_) { }

// aquire output dim
int32 MaxpoolingComponent::OutputDim() const {
  int32 num_pools_x = 1 + (input_x_dim_ - pool_x_size_) / pool_x_step_;
  int32 num_pools_y = 1 + (input_y_dim_ - pool_y_size_) / pool_y_step_;
  int32 num_pools_z = 1 + (input_z_dim_ - pool_z_size_) / pool_z_step_;
  return num_pools_x * num_pools_y * num_pools_z;
}

// check the component parameters
void MaxpoolingComponent::Check() const {
  // sanity check of the max pooling parameters
  KALDI_ASSERT(input_x_dim_ > 0);
  KALDI_ASSERT(input_y_dim_ > 0);
  KALDI_ASSERT(input_z_dim_ > 0);
  KALDI_ASSERT(pool_x_size_ > 0);
  KALDI_ASSERT(pool_y_size_ > 0);
  KALDI_ASSERT(pool_z_size_ > 0);
  KALDI_ASSERT(pool_x_step_ > 0);
  KALDI_ASSERT(pool_y_step_ > 0);
  KALDI_ASSERT(pool_z_step_ > 0);
  KALDI_ASSERT(input_x_dim_ >= pool_x_size_);
  KALDI_ASSERT(input_y_dim_ >= pool_y_size_);
  KALDI_ASSERT(input_z_dim_ >= pool_z_size_);
  KALDI_ASSERT(pool_x_size_ >= pool_x_step_);
  KALDI_ASSERT(pool_y_size_ >= pool_y_step_);
  KALDI_ASSERT(pool_z_size_ >= pool_z_step_);
  KALDI_ASSERT((input_x_dim_ - pool_x_size_) % pool_x_step_  == 0);
  KALDI_ASSERT((input_y_dim_ - pool_y_size_) % pool_y_step_  == 0);
  KALDI_ASSERT((input_z_dim_ - pool_z_size_) % pool_z_step_  == 0);
}

// initialize the component using configuration file
void MaxpoolingComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;

  ok = ok && cfl->GetValue("input-x-dim", &input_x_dim_);
  ok = ok && cfl->GetValue("input-y-dim", &input_y_dim_);
  ok = ok && cfl->GetValue("input-z-dim", &input_z_dim_);
  ok = ok && cfl->GetValue("pool-x-size", &pool_x_size_);
  ok = ok && cfl->GetValue("pool-y-size", &pool_y_size_);
  ok = ok && cfl->GetValue("pool-z-size", &pool_z_size_);
  ok = ok && cfl->GetValue("pool-x-step", &pool_x_step_);
  ok = ok && cfl->GetValue("pool-y-step", &pool_y_step_);
  ok = ok && cfl->GetValue("pool-z-step", &pool_z_step_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();

  Check();
}

// Method to convert from a matrix representing a minibatch of vectorized
// 3D tensors to patches for 3d max pooling, each patch corresponds to
// the nodes having the same local coordinatenodes from each pool
void MaxpoolingComponent::InputToInputPatches(
    const CuMatrixBase<BaseFloat>& in,
    CuMatrix<BaseFloat> *patches) const{
  int32 num_pools_x = 1 + (input_x_dim_ - pool_x_size_) / pool_x_step_;
  int32 num_pools_y = 1 + (input_y_dim_ - pool_y_size_) / pool_y_step_;
  int32 num_pools_z = 1 + (input_z_dim_ - pool_z_size_) / pool_z_step_;

  std::vector<int32> column_map(patches->NumCols());
  int32 column_map_size = column_map.size();
  for (int32 x = 0, index =0; x < pool_x_size_; x++) {
    for (int32 y = 0; y < pool_y_size_; y++) {
      for (int32 z = 0; z < pool_z_size_; z++) {
        // given the local node coordinate, group them from each pool
        // to form a patch
        for (int32 x_pool = 0; x_pool < num_pools_x; x_pool++) {
          for (int32 y_pool = 0; y_pool < num_pools_y; y_pool++) {
            for (int32 z_pool = 0; z_pool < num_pools_z; z_pool++, index++) {
              KALDI_ASSERT(index < column_map_size);
              column_map[index] = (x_pool * pool_x_step_ + x) * input_y_dim_ * input_z_dim_ +
                                  (y_pool * pool_y_step_ + y) * input_z_dim_ +
                                  (z_pool * pool_z_step_ + z);

            }
          }
        }
      }
    }
  }
  CuArray<int32> cu_cols(column_map);
  patches->CopyCols(in, cu_cols);
}

/*
  This is the 3d max pooling propagate function.
  It is assumed that each row of the input matrix
  is a vectorized 3D-tensor of type zxy.
  Similar to the propagate function of ConvolutionComponent,
  the input matrix is first arranged into patches so that
  pools (with / without overlapping) could be
  processed in a parallelizable manner.
  The output matrix is also a vectorized 3D-tensor of type zxy.
*/

void MaxpoolingComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  int32 num_frames = in.NumRows();
  int32 num_pools = OutputDim();
  int32 pool_size = pool_x_size_ * pool_y_size_ * pool_z_size_;
  CuMatrix<BaseFloat> patches(num_frames, num_pools * pool_size, kUndefined);
  InputToInputPatches(in, &patches);

  out->Set(-1e20); // reset a large negative value
  for (int32 q = 0; q < pool_size; q++)
    out->Max(patches.ColRange(q * num_pools, num_pools));
}

// Method to compute the input derivative matrix from the input derivatives
// for patches, where each patch corresponds to
// the nodes having the same local coordinatenodes from each pool
void MaxpoolingComponent::InderivPatchesToInderiv(
    const CuMatrix<BaseFloat>& in_deriv_patches,
    CuMatrixBase<BaseFloat> *in_deriv) const {

  int32 num_pools_x = 1 + (input_x_dim_ - pool_x_size_) / pool_x_step_;
  int32 num_pools_y = 1 + (input_y_dim_ - pool_y_size_) / pool_y_step_;
  int32 num_pools_z = 1 + (input_z_dim_ - pool_z_size_) / pool_z_step_;

  std::vector<std::vector<int32> > reverse_column_map(in_deriv->NumCols());
  int32 rev_col_map_size = reverse_column_map.size();
  for (int32 x = 0, index = 0; x < pool_x_size_; x++) {
    for (int32 y = 0; y < pool_y_size_; y++) {
      for (int32 z = 0; z < pool_z_size_; z++) {

        for (int32 x_pool = 0; x_pool < num_pools_x; x_pool++) {
          for (int32 y_pool = 0; y_pool < num_pools_y; y_pool++) {
            for (int32 z_pool = 0; z_pool < num_pools_z; z_pool++, index++) {
              int32 vector_index = (x_pool * pool_x_step_ + x) * input_y_dim_ * input_z_dim_ +
                                  (y_pool * pool_y_step_ + y) * input_z_dim_ +
                                  (z_pool * pool_z_step_ + z);

              KALDI_ASSERT(vector_index < rev_col_map_size);
              reverse_column_map[vector_index].push_back(index);
            }
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

/*
  3d max pooling backpropagate function
  This function backpropagate the error from
  out_deriv to in_deriv.
  In order to select the node in each pool to
  backpropagate the error, it has to compare
  the output pool value stored in the out_value
  matrix with each of its input pool member node
  stroed in the in_value matrix.
*/
void MaxpoolingComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv,
                                   Component *, // to_update,
                                   CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)
    return;

  int32 num_frames = in_value.NumRows();
  int32 num_pools = OutputDim();
  int32 pool_size = pool_x_size_ * pool_y_size_ * pool_z_size_;
  CuMatrix<BaseFloat> patches(num_frames, num_pools * pool_size, kUndefined);
  InputToInputPatches(in_value, &patches);

  for (int32 q = 0; q < pool_size; q++) {
    // zero-out mask
    CuMatrix<BaseFloat> mask;
    out_value.EqualElementMask(patches.ColRange(q * num_pools, num_pools), &mask);
    mask.MulElements(out_deriv);
    patches.ColRange(q * num_pools, num_pools).CopyFromMat(mask);
  }

  // combine the derivatives from the individual input deriv patches
  // to compute input deriv matrix
  InderivPatchesToInderiv(patches, in_deriv);
}

void MaxpoolingComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MaxpoolingComponent>", "<InputXDim>");
  ReadBasicType(is, binary, &input_x_dim_);
  ExpectToken(is, binary, "<InputYDim>");
  ReadBasicType(is, binary, &input_y_dim_);
  ExpectToken(is, binary, "<InputZDim>");
  ReadBasicType(is, binary, &input_z_dim_);
  ExpectToken(is, binary, "<PoolXSize>");
  ReadBasicType(is, binary, &pool_x_size_);
  ExpectToken(is, binary, "<PoolYSize>");
  ReadBasicType(is, binary, &pool_y_size_);
  ExpectToken(is, binary, "<PoolZSize>");
  ReadBasicType(is, binary, &pool_z_size_);
  ExpectToken(is, binary, "<PoolXStep>");
  ReadBasicType(is, binary, &pool_x_step_);
  ExpectToken(is, binary, "<PoolYStep>");
  ReadBasicType(is, binary, &pool_y_step_);
  ExpectToken(is, binary, "<PoolZStep>");
  ReadBasicType(is, binary, &pool_z_step_);
  ExpectToken(is, binary, "</MaxpoolingComponent>");
  Check();
}

void MaxpoolingComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MaxpoolingComponent>");
  WriteToken(os, binary, "<InputXDim>");
  WriteBasicType(os, binary, input_x_dim_);
  WriteToken(os, binary, "<InputYDim>");
  WriteBasicType(os, binary, input_y_dim_);
  WriteToken(os, binary, "<InputZDim>");
  WriteBasicType(os, binary, input_z_dim_);
  WriteToken(os, binary, "<PoolXSize>");
  WriteBasicType(os, binary, pool_x_size_);
  WriteToken(os, binary, "<PoolYSize>");
  WriteBasicType(os, binary, pool_y_size_);
  WriteToken(os, binary, "<PoolZSize>");
  WriteBasicType(os, binary, pool_z_size_);
  WriteToken(os, binary, "<PoolXStep>");
  WriteBasicType(os, binary, pool_x_step_);
  WriteToken(os, binary, "<PoolYStep>");
  WriteBasicType(os, binary, pool_y_step_);
  WriteToken(os, binary, "<PoolZStep>");
  WriteBasicType(os, binary, pool_z_step_);
  WriteToken(os, binary, "</MaxpoolingComponent>");
}

// display information about component
std::string MaxpoolingComponent::Info() const {
  std::ostringstream stream;
  stream << Type()
         << ", input-x-dim=" << input_x_dim_
         << ", input-y-dim=" << input_y_dim_
         << ", input-z-dim=" << input_z_dim_
         << ", pool-x-size=" << pool_x_size_
         << ", pool-y-size=" << pool_y_size_
         << ", pool-z-size=" << pool_z_size_
         << ", pool-x-step=" << pool_x_step_
         << ", pool-y-step=" << pool_y_step_
         << ", pool-z-step=" << pool_z_step_;
  return stream.str();
}

void PermuteComponent::ComputeReverseColumnMap() {
  int32 dim = column_map_.Dim();
  KALDI_ASSERT(dim > 0);
  std::vector<int32> reverse_column_map_cpu(dim, -1),
      column_map_cpu(dim);
  column_map_.CopyToVec(&column_map_cpu);
  for (int32 i = 0; i < dim; i++) {
    int32 &dest = reverse_column_map_cpu[column_map_cpu[i]];
    if (dest != -1)
      KALDI_ERR << "Column map does not represent a permutation.";
    dest = i;
  }
  reverse_column_map_.Resize(dim);
  reverse_column_map_.CopyFromVec(reverse_column_map_cpu);
}

Component* PermuteComponent::Copy() const {
  PermuteComponent *ans = new PermuteComponent();
  ans->column_map_ = column_map_;
  ans->reverse_column_map_ = reverse_column_map_;
  return ans;
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
  in_deriv->CopyCols(out_deriv, reverse_column_map_);
}

void PermuteComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string column_map_str;
  ok = ok && cfl->GetValue("column-map", &column_map_str);
  std::vector<int32> column_map;
  if (!SplitStringToIntegers(column_map_str, ",", true, &column_map))
    KALDI_ERR << "Bad initializer in PermuteComponent: column-map="
              << column_map_str;
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(column_map);
}

void PermuteComponent::Init(const std::vector<int32> &column_map) {
  KALDI_ASSERT(column_map.size() > 0);
  column_map_.CopyFromVec(column_map);
  ComputeReverseColumnMap();
}

void PermuteComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PermuteComponent>", "<ColumnMap>");
  std::vector<int32> column_map;
  if (binary && is.peek() == 'F') {
    // back-compatibility code [temporary]
    Vector<BaseFloat> float_map;
    float_map.Read(is, binary);
    column_map.resize(float_map.Dim());
    for (int32 i = 0; i < float_map.Dim(); i++) {
      // note: casting truncates toward zero: add 0.5 to approximate rounding.
      column_map[i] = static_cast<int32>(float_map(i) + 0.5);
    }
    // the next line is a workaround for a bug in the old
    // writing code, which now causes an assert failure.  it's only
    // valid for the permutations we're currently using.  anyway all this
    // code is only temporary.
    column_map.back() = float_map.Dim() - 1;
  } else {
    ReadIntegerVector(is, binary, &column_map);
  }
  column_map_.CopyFromVec(column_map);
  ExpectToken(is, binary, "</PermuteComponent>");
  ComputeReverseColumnMap();
}

void PermuteComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PermuteComponent>");
  WriteToken(os, binary, "<ColumnMap>");
  std::ostringstream buffer;
  std::vector<int32> column_map;
  column_map_.CopyToVec(&column_map);
  WriteIntegerVector(os, binary, column_map);
  WriteToken(os, binary, "</PermuteComponent>");
}

std::string PermuteComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << column_map_.Dim();
  stream << " , column-map=[ ";
  std::vector<int32> column_map(column_map_.Dim());
  column_map_.CopyToVec(&column_map);
  int32 max_size = 5;
  for (size_t i = 0; i < column_map.size() && i < max_size; i++)
    stream << column_map[i] << ' ';
  if (static_cast<int32>(column_map.size()) > max_size)
    stream << "... ";
  stream << "]";
  return stream.str();
}


bool CompositeComponent::IsUpdatable() const {
  for (std::vector<Component*>::const_iterator iter = components_.begin(),
           end = components_.end(); iter != end; ++iter)
    if (((*iter)->Properties() & kUpdatableComponent) != 0)
      return true;
  return false;
}

// virtual
int32 CompositeComponent::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
};

// virtual
int32 CompositeComponent::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
};

// virtual
int32 CompositeComponent::Properties() const {
  KALDI_ASSERT(!components_.empty());
  int32 last_component_properties = components_.back()->Properties(),
      first_component_properties = components_.front()->Properties();
  // We always assume backprop needs the input, as this would be necessary to
  // get the activations at intermediate layers, if these were not needed in
  // backprop, there would be no reason to use a CompositeComponent.
  int32 ans = kSimpleComponent | kBackpropNeedsInput |
      (last_component_properties &
       (kPropagateAdds|kBackpropNeedsOutput|kOutputContiguous)) |
       (first_component_properties &
        (kBackpropAdds|kInputContiguous)) |
       (IsUpdatable() ? kUpdatableComponent : 0);
  // note, we don't return the kStoresStats property because that function is
  // not implemented; instead, for efficiency, we call StoreStats() on any
  // sub-components as part of the backprop phase.
  if (last_component_properties & kStoresStats)
    ans |= kBackpropNeedsOutput;
  return ans;
};


MatrixStrideType CompositeComponent::GetStrideType(int32 i) const {
  int32 num_components = components_.size();
  if ((components_[i]->Properties() & kOutputContiguous) ||
      (i + 1 < num_components &&
       (components_[i + 1]->Properties() & kInputContiguous)))
    return kStrideEqualNumCols;
  else
    return kDefaultStride;
}


// virtual
void CompositeComponent::Propagate(
    const ComponentPrecomputedIndexes *, // indexes
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == out->NumRows() && in.NumCols() == InputDim() &&
               out->NumCols() == OutputDim());
  int32 num_rows = in.NumRows(),
      num_components = components_.size();
  if (max_rows_process_ > 0 && num_rows > max_rows_process_) {
    // recurse and process smaller parts of the data, to save memory.
    for (int32 row_offset = 0; row_offset < num_rows;
         row_offset += max_rows_process_) {
      int32 this_num_rows = std::min<int32>(max_rows_process_,
                                            num_rows - row_offset);
      const CuSubMatrix<BaseFloat> in_part(in, row_offset, this_num_rows,
                                           0, in.NumCols());
      CuSubMatrix<BaseFloat> out_part(*out, row_offset, this_num_rows,
                                      0, out->NumCols());
      this->Propagate(NULL, in_part, &out_part);
    }
    return;
  }
  std::vector<CuMatrix<BaseFloat> > intermediate_outputs(num_components - 1);
  for (int32 i = 0; i < num_components; i++) {
    if (i + 1 < num_components) {
      MatrixResizeType resize_type =
          ((components_[i]->Properties() & kPropagateAdds) ?
           kSetZero : kUndefined);
      intermediate_outputs[i].Resize(num_rows, components_[i]->OutputDim(),
                                     resize_type, GetStrideType(i));
    }
    components_[i]->Propagate(NULL, (i == 0 ? in : intermediate_outputs[i-1]),
               (i + 1 == num_components ? out : &(intermediate_outputs[i])));
    if (i > 0)
      intermediate_outputs[i-1].Resize(0, 0);
  }
}


void CompositeComponent::Init(const std::vector<Component*> &components,
                              int32 max_rows_process) {
  DeletePointers(&components_);  // clean up.
  components_ = components;
  KALDI_ASSERT(!components.empty());
  max_rows_process_ = max_rows_process;

  for (size_t i = 0; i < components_.size(); i++) {
    // make sure all constituent components are simple.
    KALDI_ASSERT(components_[i]->Properties() & kSimpleComponent);
    if (i > 0) {
      // make sure all the internal dimensions match up.
      KALDI_ASSERT(components_[i]->InputDim() ==
                   components_[i-1]->OutputDim());
    }
  }
}

// virtual
void CompositeComponent::Read(std::istream &is, bool binary) {
  // Because we didn't previously write out the learning rate,
  // we need some temporary code.
  int32 max_rows_process;
  if (false) {
    ReadUpdatableCommon(is, binary);
    ExpectToken(is, binary, "<MaxRowsProcess>");
    ReadBasicType(is, binary, &max_rows_process);
  } else {  // temporary code.
    std::string token;
    ReadToken(is, binary, &token);
    if (token == "<CompositeComponent>") {
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
      ReadToken(is, binary, &token);
    }
    if (token != "<MaxRowsProcess>") {
      KALDI_ERR << "Expected token <MaxRowsProcess>, got "
                << token;
    }
    ReadBasicType(is, binary, &max_rows_process);
  }
  ExpectToken(is, binary, "<NumComponents>");
  int32 num_components;
  ReadBasicType(is, binary, &num_components); // Read dimension.
  if (num_components < 0 || num_components > 100000)
    KALDI_ERR << "Bad num-components";
  std::vector<Component*> components(num_components);
  for (int32 i = 0; i < num_components; i++)
    components[i] = ReadNew(is, binary);
  Init(components, max_rows_process);
  ExpectToken(is, binary, "</CompositeComponent>");
}

// virtual
void CompositeComponent::ZeroStats() {
  // we call ZeroStats() on all components without checking their flags; this
  // will do nothing if the component doesn't store stats.  (components like
  // ReLU and sigmoid and tanh store stats on activations).
  for (size_t i = 0; i < components_.size(); i++)
   components_[i]->ZeroStats();
}

// virtual
void CompositeComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate.
  WriteToken(os, binary, "<MaxRowsProcess>");
  WriteBasicType(os, binary, max_rows_process_);
  WriteToken(os, binary, "<NumComponents>");
  int32 num_components = components_.size();
  WriteBasicType(os, binary, num_components);
  for (int32 i = 0; i < num_components; i++)
    components_[i]->Write(os, binary);
  WriteToken(os, binary, "</CompositeComponent>");
}


// virtual
void CompositeComponent::Backprop(const std::string &debug_info,
                                  const ComponentPrecomputedIndexes *indexes,
                                  const CuMatrixBase<BaseFloat> &in_value,
                                  const CuMatrixBase<BaseFloat> &out_value,
                                  const CuMatrixBase<BaseFloat> &out_deriv,
                                  Component *to_update,
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == InputDim() &&
               out_deriv.NumCols() == OutputDim());
  int32 num_rows = in_value.NumRows(),
      num_components = components_.size();
  if (max_rows_process_ > 0 && num_rows > max_rows_process_) {
    KALDI_ASSERT(max_rows_process_ > 0);
    // recurse and process smaller parts of the data, to save memory.
    for (int32 row_offset = 0; row_offset < num_rows;
         row_offset += max_rows_process_) {
      bool have_output_value = (out_value.NumRows() != 0);
      int32 this_num_rows = std::min<int32>(max_rows_process_,
                                            num_rows - row_offset);
      // out_value_part will only be used if out_value is nonempty; otherwise we
      // make it a submatrix of 'out_deriv' to avoid errors in the constructor.
      const CuSubMatrix<BaseFloat> out_value_part(have_output_value ? out_value : out_deriv,
                                                  row_offset, this_num_rows,
                                                  0, out_deriv.NumCols());
      // in_deriv_value_part will only be used if in_deriv != NULL; otherwise we
      // make it a submatrix of 'in_value' to avoid errors in the constructor.
      CuSubMatrix<BaseFloat> in_deriv_part(in_deriv != NULL ? *in_deriv : in_value,
                                            row_offset, this_num_rows,
                                            0, in_value.NumCols());
      CuSubMatrix<BaseFloat> in_value_part(in_value, row_offset, this_num_rows,
                                           0, in_value.NumCols());
      const CuSubMatrix<BaseFloat> out_deriv_part(out_deriv,
                                                  row_offset, this_num_rows,
                                                  0, out_deriv.NumCols());
      CuMatrix<BaseFloat>  empty_mat;
      this->Backprop(debug_info, NULL, in_value_part,
                     (have_output_value ? static_cast<const CuMatrixBase<BaseFloat>&>(out_value_part) :
                      static_cast<const CuMatrixBase<BaseFloat>&>(empty_mat)),
                     out_deriv_part, to_update,
                     in_deriv != NULL ? &in_deriv_part : NULL);
    }
    return;
  }
  // For now, assume all intermediate values and derivatives need to be
  // computed.  in_value and out_deriv will always be supplied.

  // intermediate_outputs[i] contains the output of component i.
  std::vector<CuMatrix<BaseFloat> > intermediate_outputs(num_components - 1);
  // intermediate_derivs[i] contains the deriative at the output of component i.
  std::vector<CuMatrix<BaseFloat> > intermediate_derivs(num_components - 1);

  // Do the propagation again, for all but the last component in the sequence.
  // later on we can try being more careful about which ones we need to
  // propagate.
  for (int32 i = 0; i + 1 < num_components; i++) {
    // skip the last-but-one component's propagate if the last component's
    // backprop doesn't need the input and the one previous to that doesn't
    // need the output.  [lowest hanging fruit for optimization]
    if (i + 2 == num_components &&
        !(components_[i+1]->Properties() & kBackpropNeedsInput) &&
        !(components_[i]->Properties() & kBackpropNeedsOutput))
      break;
    MatrixResizeType resize_type =
        ((components_[i]->Properties() & kPropagateAdds) ?
         kSetZero : kUndefined);
    intermediate_outputs[i].Resize(num_rows, components_[i]->OutputDim(),
                                   resize_type, GetStrideType(i));
    components_[i]->Propagate(NULL,
                              (i == 0 ? in_value : intermediate_outputs[i-1]),
                              &(intermediate_outputs[i]));
  }
  for (int32 i = num_components - 1; i >= 0; i--) {
    Component *component_to_update =
        (to_update == NULL ? NULL :
         dynamic_cast<CompositeComponent*>(to_update)->components_[i]);

    if (components_[i]->Properties() & kStoresStats &&
        component_to_update != NULL)
      component_to_update->StoreStats(
          (i + 1 == num_components ? out_value : intermediate_outputs[i]));

    // skip the first component's backprop if it's not updatable and in_deriv is
    // not requested.  Again, this is the lowest-hanging fruit to optimize.
    if (i == 0 && !(components_[0]->Properties() & kUpdatableComponent) &&
        in_deriv == NULL)
      break;
    if (i > 0) {
      MatrixResizeType resize_type =
          ((components_[i]->Properties() & kBackpropAdds) ?
           kSetZero : kUndefined);
      intermediate_derivs[i-1].Resize(num_rows, components_[i]->InputDim(),
                                      resize_type, GetStrideType(i - 1));
    }
    components_[i]->Backprop(debug_info, NULL,
                             (i == 0 ? in_value : intermediate_outputs[i-1]),
                             (i + 1 == num_components ? out_value : intermediate_outputs[i]),
                             (i + 1 == num_components ? out_deriv : intermediate_derivs[i]),
                             component_to_update,
                             (i == 0 ? in_deriv : &(intermediate_derivs[i-1])));
  }
}


// virtual
std::string CompositeComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << " ";
  for (size_t i = 0; i < components_.size(); i++) {
    if (i > 0) stream << ", ";
    stream << "sub-component" << (i+1) << " = { "
           << components_[i]->Info() << " }";
  }
  return stream.str();
}

// virtual
void CompositeComponent::Scale(BaseFloat scale) {
  for (size_t i = 0; i < components_.size(); i++)
    components_[i]->Scale(scale);
}

// virtual
void CompositeComponent::Add(BaseFloat alpha, const Component &other_in) {
  const CompositeComponent *other = dynamic_cast<const CompositeComponent*>(
      &other_in);
  KALDI_ASSERT(other != NULL && other->components_.size() ==
               components_.size() && "Mismatching nnet topologies");
  for (size_t i = 0; i < components_.size(); i++)
    components_[i]->Add(alpha, *(other->components_[i]));
}

// virtual
void CompositeComponent::PerturbParams(BaseFloat stddev) {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->PerturbParams(stddev);
    }
  }
}

void CompositeComponent::SetUnderlyingLearningRate(BaseFloat lrate) {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  UpdatableComponent::SetUnderlyingLearningRate(lrate);

  // apply any learning-rate-factor that's set at this level (ill-advised, but
  // we'll do it.)
  BaseFloat effective_lrate = LearningRate();
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->SetUnderlyingLearningRate(effective_lrate);
    }
  }
}

void CompositeComponent::SetActualLearningRate(BaseFloat lrate) {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  UpdatableComponent::SetActualLearningRate(lrate);
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->SetActualLearningRate(lrate);
    }
  }
}

// virtual
void CompositeComponent::SetAsGradient() {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  UpdatableComponent::SetAsGradient();
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->SetAsGradient();
    }
  }
}

// virtual
int32 CompositeComponent::NumParameters() const {
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      ans += uc->NumParameters();
    }
  }
  return ans;
}

// virtual
void CompositeComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  int32 cur_offset = 0;
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      int32 this_size = uc->NumParameters();
      SubVector<BaseFloat> params_range(*params, cur_offset, this_size);
      uc->Vectorize(&params_range);
      cur_offset += this_size;
    }
  }
  KALDI_ASSERT(cur_offset == params->Dim());
}

// virtual
void CompositeComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  int32 cur_offset = 0;
  KALDI_ASSERT(this->IsUpdatable());  // or should not be called.
  for (size_t i = 0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      int32 this_size = uc->NumParameters();
      SubVector<BaseFloat> params_range(params, cur_offset, this_size);
      uc->UnVectorize(params_range);
      cur_offset += this_size;
    }
  }
  KALDI_ASSERT(cur_offset == params.Dim());
}

// virtual
BaseFloat CompositeComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const CompositeComponent *other = dynamic_cast<const CompositeComponent*>(
      &other_in);
  KALDI_ASSERT(other != NULL && other->components_.size() ==
               components_.size() && "Mismatching nnet topologies");
  BaseFloat ans = 0.0;
  for (size_t i = 0.0; i < components_.size(); i++) {
    if (components_[i]->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc =
          dynamic_cast<UpdatableComponent*>(components_[i]);
      const UpdatableComponent *uc_other =
          dynamic_cast<UpdatableComponent*>(other->components_[i]);
      KALDI_ASSERT(uc != NULL && uc_other != NULL);
      ans += uc->DotProduct(*uc_other);
    }
  }
  return ans;
}

// virtual
Component* CompositeComponent::Copy() const {
  std::vector<Component*> components(components_.size());
  for (size_t i = 0; i < components_.size(); i++)
    components[i] = components_[i]->Copy();
  CompositeComponent *ans = new CompositeComponent();
  ans->Init(components, max_rows_process_);
  return ans;
}


// virtual
void CompositeComponent::InitFromConfig(ConfigLine *cfl) {
  int32 max_rows_process = 4096, num_components = -1;
  cfl->GetValue("max-rows-process", &max_rows_process);
  if (!cfl->GetValue("num-components", &num_components) ||
      num_components < 1)
    KALDI_ERR << "Expected num-components to be defined in "
              << "CompositeComponent config line '" << cfl->WholeLine() << "'";
  std::vector<Component*> components;
  for (int32 i = 1; i <= num_components; i++) {
    std::ostringstream name_stream;
    name_stream << "component" << i;
    std::string component_config;
    if (!cfl->GetValue(name_stream.str(), &component_config)) {
      DeletePointers(&components);
      KALDI_ERR << "Expected '" << name_stream.str() << "' to be defined in "
                << "CompositeComponent config line '" << cfl->WholeLine() << "'";
    }
    ConfigLine nested_line;
    // note: the nested line may not contain comments.
    std::string component_type;
    Component *this_component = NULL;
    if (!nested_line.ParseLine(component_config) ||
        !nested_line.GetValue("type", &component_type) ||
        !(this_component = NewComponentOfType(component_type)) ||
        nested_line.FirstToken() != "") {
      DeletePointers(&components);
      KALDI_ERR << "Could not parse config line for '" << name_stream.str()
                << "(or undefined or bad component type [type=xxx]), in "
                << "CompositeComponent config line '" << cfl->WholeLine() << "'";
    }
    if(this_component->Type() == "CompositeComponent") {
      DeletePointers(&components);
      delete this_component;
      KALDI_ERR << "Found CompositeComponent nested within CompositeComponent."
                << "Try decreasing max-rows-process instead."
                << "Nested line: '" << nested_line.WholeLine() << "'\n"
                << "Toplevel CompositeComponent line '" << cfl->WholeLine()
                << "'";
    }
    this_component->InitFromConfig(&nested_line);
    components.push_back(this_component);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  this->Init(components, max_rows_process);
}

const Component* CompositeComponent::GetComponent(int32 i) const {
  KALDI_ASSERT(static_cast<size_t>(i) < components_.size());
  return components_[i];
}

void CompositeComponent::SetComponent(int32 i, Component *component) {
  KALDI_ASSERT(static_cast<size_t>(i) < components_.size());
  delete components_[i];
  components_[i] = component;
}


int32 LstmNonlinearityComponent::InputDim() const {
  int32 cell_dim = value_sum_.NumCols();
  return cell_dim * 5;
}

int32 LstmNonlinearityComponent::OutputDim() const {
  int32 cell_dim = value_sum_.NumCols();
  return cell_dim * 2;
}


void LstmNonlinearityComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read opening tag and learning rate.
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  ExpectToken(is, binary, "<ValueAvg>");
  value_sum_.Read(is, binary);
  ExpectToken(is, binary, "<DerivAvg>");
  deriv_sum_.Read(is, binary);
  ExpectToken(is, binary, "<SelfRepairConfig>");
  self_repair_config_.Read(is, binary);
  ExpectToken(is, binary, "<SelfRepairProb>");
  self_repair_total_.Read(is, binary);

  ExpectToken(is, binary, "<Count>");
  ReadBasicType(is, binary, &count_);

  // For the on-disk format, we normalze value_sum_, deriv_sum_ and
  // self_repair_total_ by dividing by the count, but in memory they are scaled
  // by the count.  [for self_repair_total_, the scaling factor is count_ *
  // cell_dim].
  value_sum_.Scale(count_);
  deriv_sum_.Scale(count_);
  int32 cell_dim = params_.NumCols();
  self_repair_total_.Scale(count_ * cell_dim);

  InitNaturalGradient();

  ExpectToken(is, binary, "</LstmNonlinearityComponent>");

}

void LstmNonlinearityComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Read opening tag and learning rate.

  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  WriteToken(os, binary, "<ValueAvg>");
  {
    Matrix<BaseFloat> value_avg(value_sum_);
    if (count_ != 0.0)
      value_avg.Scale(1.0 / count_);
    value_avg.Write(os, binary);
  }
  WriteToken(os, binary, "<DerivAvg>");
  {
    Matrix<BaseFloat> deriv_avg(deriv_sum_);
    if (count_ != 0.0)
      deriv_avg.Scale(1.0 / count_);
    deriv_avg.Write(os, binary);
  }
  WriteToken(os, binary, "<SelfRepairConfig>");
  self_repair_config_.Write(os, binary);
  WriteToken(os, binary, "<SelfRepairProb>");
  {
    int32 cell_dim = params_.NumCols();
    Vector<BaseFloat> self_repair_prob(self_repair_total_);
    if (count_ != 0.0)
      self_repair_prob.Scale(1.0 / (count_ * cell_dim));
    self_repair_prob.Write(os, binary);
  }
  WriteToken(os, binary, "<Count>");
  WriteBasicType(os, binary, count_);
  WriteToken(os, binary, "</LstmNonlinearityComponent>");
}



std::string LstmNonlinearityComponent::Info() const {
  std::ostringstream stream;
  int32 cell_dim = params_.NumCols();
  stream << UpdatableComponent::Info() << ", cell-dim=" << cell_dim;
  PrintParameterStats(stream, "w_ic", params_.Row(0));
  PrintParameterStats(stream, "w_fc", params_.Row(1));
  PrintParameterStats(stream, "w_oc", params_.Row(2));

  // Note: some of the following code mirrors the code in
  // UpdatableComponent::Info(), in nnet-component-itf.cc.
  if (count_ > 0) {
    stream << ", count=" << std::setprecision(3) << count_
           << std::setprecision(6);
  }
  static const char *nonlin_names[] = { "i_t_sigmoid", "f_t_sigmoid", "c_t_tanh",
                                        "o_t_sigmoid", "m_t_tanh" };
  for (int32 i = 0; i < 5; i++) {
    stream << ", " << nonlin_names[i] << "={";
    stream << " self-repair-lower-threshold=" << self_repair_config_(i)
           << ", self-repair-scale=" << self_repair_config_(i + 5);

    if (count_ != 0) {
      BaseFloat self_repaired_proportion =
          self_repair_total_(i) / (count_ * cell_dim);
      stream << ", self-repaired-proportion=" << self_repaired_proportion;
      Vector<double> value_sum(value_sum_.Row(i)),
          deriv_sum(deriv_sum_.Row(i));
      Vector<BaseFloat> value_avg(value_sum), deriv_avg(deriv_sum);
      value_avg.Scale(1.0 / count_);
      deriv_avg.Scale(1.0 / count_);
      stream << ", value-avg=" << SummarizeVector(value_avg)
             << ", deriv-avg=" << SummarizeVector(deriv_avg);
    }
    stream << " }";
  }
  return stream.str();
}


Component* LstmNonlinearityComponent::Copy() const {
  return new LstmNonlinearityComponent(*this);
}

void LstmNonlinearityComponent::ZeroStats() {
  value_sum_.SetZero();
  deriv_sum_.SetZero();
  self_repair_total_.SetZero();
  count_ = 0.0;
}

void LstmNonlinearityComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    params_.SetZero();
    value_sum_.SetZero();
    deriv_sum_.SetZero();
    self_repair_total_.SetZero();
    count_ = 0.0;
  } else {
    params_.Scale(scale);
    value_sum_.Scale(scale);
    deriv_sum_.Scale(scale);
    self_repair_total_.Scale(scale);
    count_ *= scale;
  }
}

void LstmNonlinearityComponent::Add(BaseFloat alpha,
                                    const Component &other_in) {
  const LstmNonlinearityComponent *other =
      dynamic_cast<const LstmNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  params_.AddMat(alpha, other->params_);
  value_sum_.AddMat(alpha, other->value_sum_);
  deriv_sum_.AddMat(alpha, other->deriv_sum_);
  self_repair_total_.AddVec(alpha, other->self_repair_total_);
  count_ += alpha * other->count_;
}

void LstmNonlinearityComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_params(params_.NumRows(), params_.NumCols());
  temp_params.SetRandn();
  params_.AddMat(stddev, temp_params);
}

BaseFloat LstmNonlinearityComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const LstmNonlinearityComponent *other =
      dynamic_cast<const LstmNonlinearityComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return TraceMatMat(params_, other->params_, kTrans);
}

int32 LstmNonlinearityComponent::NumParameters() const {
  return params_.NumRows() * params_.NumCols();
}

void LstmNonlinearityComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == NumParameters());
  params->CopyRowsFromMat(params_);
}


void LstmNonlinearityComponent::UnVectorize(
    const VectorBase<BaseFloat> &params)  {
  KALDI_ASSERT(params.Dim() == NumParameters());
  params_.CopyRowsFromVec(params);
}


void LstmNonlinearityComponent::Propagate(
    const ComponentPrecomputedIndexes *, // indexes
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  cu::ComputeLstmNonlinearity(in, params_, out);
}


void LstmNonlinearityComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {

  if (to_update_in == NULL) {
    cu::BackpropLstmNonlinearity(in_value, params_, out_deriv,
                                 deriv_sum_, self_repair_config_,
                                 count_, in_deriv,
                                 (CuMatrixBase<BaseFloat>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<double>*) NULL,
                                 (CuMatrixBase<BaseFloat>*) NULL);
  } else {
    LstmNonlinearityComponent *to_update =
        dynamic_cast<LstmNonlinearityComponent*>(to_update_in);
    KALDI_ASSERT(to_update != NULL);

    int32 cell_dim = params_.NumCols();
    CuMatrix<BaseFloat> params_deriv(3, cell_dim, kUndefined);
    CuMatrix<BaseFloat> self_repair_total(5, cell_dim, kUndefined);

    cu::BackpropLstmNonlinearity(in_value, params_, out_deriv,
                                 deriv_sum_, self_repair_config_,
                                 count_, in_deriv, &params_deriv,
                                 &(to_update->value_sum_),
                                 &(to_update->deriv_sum_),
                                 &self_repair_total);

    CuVector<BaseFloat> self_repair_total_sum(5);
    self_repair_total_sum.AddColSumMat(1.0, self_repair_total, 0.0);
    to_update->self_repair_total_.AddVec(1.0, self_repair_total_sum);
    to_update->count_ += static_cast<double>(in_value.NumRows());

    BaseFloat scale = 1.0;
    if (!to_update->is_gradient_) {
      to_update->preconditioner_.PreconditionDirections(
          &params_deriv, NULL, &scale);
    }
    to_update->params_.AddMat(to_update->learning_rate_ * scale,
                              params_deriv);
  }
}

LstmNonlinearityComponent::LstmNonlinearityComponent(
    const LstmNonlinearityComponent &other):
    UpdatableComponent(other),
    params_(other.params_),
    value_sum_(other.value_sum_),
    deriv_sum_(other.deriv_sum_),
    self_repair_config_(other.self_repair_config_),
    self_repair_total_(other.self_repair_total_),
    count_(other.count_),
    preconditioner_(other.preconditioner_) { }

void LstmNonlinearityComponent::Init(
    int32 cell_dim, BaseFloat param_stddev,
    BaseFloat tanh_self_repair_threshold,
    BaseFloat sigmoid_self_repair_threshold,
    BaseFloat self_repair_scale) {
  KALDI_ASSERT(cell_dim > 0 && param_stddev >= 0.0 &&
               tanh_self_repair_threshold >= 0.0 &&
               tanh_self_repair_threshold <= 1.0 &&
               sigmoid_self_repair_threshold >= 0.0 &&
               sigmoid_self_repair_threshold <= 0.25 &&
               self_repair_scale >= 0.0 && self_repair_scale <= 0.1);
  params_.Resize(3, cell_dim);
  params_.SetRandn();
  params_.Scale(param_stddev);
  value_sum_.Resize(5, cell_dim);
  deriv_sum_.Resize(5, cell_dim);
  self_repair_config_.Resize(10);
  self_repair_config_.Range(0, 5).Set(sigmoid_self_repair_threshold);
  self_repair_config_(2) = tanh_self_repair_threshold;
  self_repair_config_(4) = tanh_self_repair_threshold;
  self_repair_config_.Range(5, 5).Set(self_repair_scale);
  self_repair_total_.Resize(5);
  count_ = 0.0;
  InitNaturalGradient();

}

void LstmNonlinearityComponent::InitNaturalGradient() {
  // As regards the configuration for the natural-gradient preconditioner, we
  // don't make it configurable from the command line-- it's unlikely that any
  // differences from changing this would be substantial enough to effectively
  // tune the configuration.  Because the preconditioning code doesn't 'see' the
  // derivatives from individual frames, but only averages over the minibatch,
  // there is a fairly small amount of data available to estimate the Fisher
  // information matrix, so we set the rank, update period and
  // num-samples-history to smaller values than normal.
  preconditioner_.SetRank(20);
  preconditioner_.SetUpdatePeriod(2);
  preconditioner_.SetNumSamplesHistory(1000.0);
}


void LstmNonlinearityComponent::InitFromConfig(ConfigLine *cfl) {
  InitLearningRatesFromConfig(cfl);
  bool ok = true;
  int32 cell_dim;
  // these self-repair thresholds are the normal defaults for tanh and sigmoid
  // respectively.  If, later on, we decide that we want to support different
  // self-repair config values for the individual sigmoid and tanh
  // nonlinearities, we can modify this code then.
  BaseFloat tanh_self_repair_threshold = 0.2,
      sigmoid_self_repair_threshold = 0.05,
      self_repair_scale = 1.0e-05;
  // param_stddev is the stddev of the parameters.  it may be better to
  // use a smaller value but this was the default in the python scripts
  // for a while.
  BaseFloat param_stddev = 1.0;
  ok = ok && cfl->GetValue("cell-dim", &cell_dim);
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("tanh-self-repair-threshold",
                &tanh_self_repair_threshold);
  cfl->GetValue("sigmoid-self-repair-threshold",
                &sigmoid_self_repair_threshold);
  cfl->GetValue("self-repair-scale", &self_repair_scale);

  // We may later on want to make it possible to initialize the different
  // parameters w_ic, w_fc and w_oc with different biases.  We'll implement
  // that when and if it's needed.

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (ok) {
    Init(cell_dim, param_stddev, tanh_self_repair_threshold,
         sigmoid_self_repair_threshold, self_repair_scale);
  } else {
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  }
}



} // namespace nnet3
} // namespace kaldi

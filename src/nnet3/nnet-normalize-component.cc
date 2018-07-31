// nnet3/nnet-normalize-component.cc

// Copyright      2015-2017  Johns Hopkins University (author: Daniel Povey)
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
#include "nnet3/nnet-normalize-component.h"
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

const BaseFloat NormalizeComponent::kSquaredNormFloor =
    pow(2.0, NormalizeComponent::kExpSquaredNormFloor);

NormalizeComponent::NormalizeComponent(const NormalizeComponent &other):
    input_dim_(other.input_dim_), block_dim_(other.block_dim_),
    target_rms_(other.target_rms_),
    add_log_stddev_(other.add_log_stddev_) { }

void NormalizeComponent::InitFromConfig(ConfigLine *cfl) {
  input_dim_ = 0;
  add_log_stddev_ = false;
  target_rms_ = 1.0;
  bool ok = cfl->GetValue("dim", &input_dim_) ||
      cfl->GetValue("input-dim", &input_dim_);
  block_dim_ = input_dim_;
  cfl->GetValue("block-dim", &block_dim_);
  cfl->GetValue("target-rms", &target_rms_);
  cfl->GetValue("add-log-stddev", &add_log_stddev_);
  if (!ok || cfl->HasUnusedValues() || input_dim_ <= 0 || target_rms_ <= 0.0 ||
      block_dim_ <= 0 || input_dim_ % block_dim_ != 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
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
  if (token == "<BlockDim>") {
    ReadBasicType(is, binary, &block_dim_);
    ReadToken(is, binary, &token);
  } else {
    block_dim_ = input_dim_;
  }
  // read target_rms_ if it is available.
  if (token == "<TargetRms>") {
    ReadBasicType(is, binary, &target_rms_);
    ReadToken(is, binary, &token);
  }
  //  Read add_log_stddev_ token, if it is available.
  if (token == "<AddLogStddev>") {
    ReadBasicType(is, binary, &add_log_stddev_);
    ReadToken(is, binary, &token);
  } else {
    add_log_stddev_ = false;
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
  if (block_dim_ != input_dim_) {
    WriteToken(os, binary, "<BlockDim>");
    WriteBasicType(os, binary, block_dim_);
  }
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
  if (block_dim_ != input_dim_)
    stream << ", block-dim=" << block_dim_;
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
void* NormalizeComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumCols() == InputDim() && out->NumCols() == OutputDim() &&
               in.NumRows() == out->NumRows());
  if (block_dim_ != input_dim_) {
    int32 num_blocks = input_dim_ / block_dim_,
        new_num_rows = in.NumRows() * num_blocks,
        output_block_dim = block_dim_ + (add_log_stddev_ ? 1 : 0);
    KALDI_ASSERT(in.Stride() == in.NumCols() && out->Stride() == out->NumCols());
    CuSubMatrix<BaseFloat> in_reshaped(in.Data(), new_num_rows,
                                       block_dim_, block_dim_),
        out_reshaped(out->Data(), new_num_rows,
                     output_block_dim, output_block_dim);
    cu::NormalizePerRow(in_reshaped, target_rms_, add_log_stddev_,
                        &out_reshaped);
  } else {
    cu::NormalizePerRow(in, target_rms_, add_log_stddev_, out);
  }
  return NULL;
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
                                  void *memo,
                                  Component *to_update,
                                  CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)
    return;
  if (block_dim_ != input_dim_) {
    int32 num_blocks = input_dim_ / block_dim_,
        new_num_rows = in_value.NumRows() * num_blocks,
        output_block_dim = block_dim_ + (add_log_stddev_ ? 1 : 0);
    KALDI_ASSERT(in_value.Stride() == in_value.NumCols() &&
                 out_deriv.Stride() == out_deriv.NumCols() &&
                 in_deriv->Stride() == in_deriv->NumCols());
    CuSubMatrix<BaseFloat> in_value_reshaped(in_value.Data(), new_num_rows,
                                             block_dim_, block_dim_),
        out_deriv_reshaped(out_deriv.Data(), new_num_rows,
                           output_block_dim, output_block_dim),
        in_deriv_reshaped(in_deriv->Data(), new_num_rows,
                          block_dim_, block_dim_);
    cu::DiffNormalizePerRow(in_value_reshaped, out_deriv_reshaped, target_rms_,
                            add_log_stddev_, &in_deriv_reshaped);
  } else {
    cu::DiffNormalizePerRow(in_value, out_deriv, target_rms_, add_log_stddev_,
                            in_deriv);
  }
}

void BatchNormComponent::ComputeDerived() {
  if (!test_mode_) {
    offset_.Resize(0);
    scale_.Resize(0);
    return;
  }

  if (count_ == 0.0) {
    KALDI_WARN << "Test-mode is set but there is no data count.  "
        "Creating random counts.  This only makes sense "
        "in unit-tests (or compute_prob_*.0.log).  If you see this "
        "elsewhere, something is very wrong.";
    count_ = 1.0;
    stats_sum_.SetRandn();
    stats_sumsq_.SetRandn();
    stats_sumsq_.AddVecVec(1.0, stats_sum_, stats_sum_, 1.0);
  }

  offset_.Resize(block_dim_);
  scale_.Resize(block_dim_);
  offset_.CopyFromVec(stats_sum_);
  offset_.Scale(-1.0 / count_);
  // now offset_ is -mean.
  scale_.CopyFromVec(stats_sumsq_);
  scale_.Scale(1.0 / count_);
  scale_.AddVecVec(-1.0, offset_, offset_, 1.0);
  // now scale_ is variance.
  // Mathematically the ApplyFloor statement should be a no-op; this is in case
  // of numerical roundoff.
  scale_.ApplyFloor(0.0);
  scale_.Add(epsilon_);
  BaseFloat power = -0.5;
  scale_.ApplyPow(power);
  // now scale_ = min(variance, epsilon)^power
  // next, multiply by the target RMS (normally 1.0).
  scale_.Scale(target_rms_);
  offset_.MulElements(scale_);
  // now offset_ is -(scale*mean).
}

void BatchNormComponent::SetTestMode(bool test_mode) {
  test_mode_ = test_mode;
  ComputeDerived();
}

void BatchNormComponent::Check() const {
  KALDI_ASSERT(dim_ > 0 && block_dim_ > 0 && dim_ % block_dim_ == 0 &&
               epsilon_ > 0.0 && target_rms_ > 0.0);
}

BatchNormComponent::BatchNormComponent(const BatchNormComponent &other):
    dim_(other.dim_), block_dim_(other.block_dim_),
    epsilon_(other.epsilon_), target_rms_(other.target_rms_),
    test_mode_(other.test_mode_), count_(other.count_),
    stats_sum_(other.stats_sum_), stats_sumsq_(other.stats_sumsq_) {
  ComputeDerived();
  Check();
}


std::string BatchNormComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_ << ", block-dim=" << block_dim_
         << ", epsilon=" << epsilon_ << ", target-rms=" << target_rms_
         << ", count=" << count_
         << ", test-mode=" << (test_mode_ ? "true" : "false");
  if (count_ > 0) {
    Vector<BaseFloat> mean(stats_sum_), var(stats_sumsq_);
    mean.Scale(1.0 / count_);
    var.Scale(1.0 / count_);
    // subtract mean^2 from var.
    var.AddVecVec(-1.0, mean, mean, 1.0);
    var.ApplyFloor(0.0);
    var.ApplyPow(0.5);  // make it the stddev.
    stream << ", data-mean=" << SummarizeVector(mean)
           << ", data-stddev=" << SummarizeVector(var);
  }
  return stream.str();
}

void BatchNormComponent::InitFromConfig(ConfigLine *cfl) {
  dim_ = -1;
  block_dim_ = -1;
  epsilon_ = 1.0e-03;
  target_rms_ = 1.0;
  test_mode_ = false;
  bool ok = cfl->GetValue("dim", &dim_);
  cfl->GetValue("block-dim", &block_dim_);
  cfl->GetValue("epsilon", &epsilon_);
  cfl->GetValue("target-rms", &target_rms_);
  cfl->GetValue("test-mode", &test_mode_);
  if (!ok || dim_ <= 0) {
    KALDI_ERR << "BatchNormComponent must have 'dim' specified, and > 0";
  }
  if (block_dim_ == -1)
    block_dim_ = dim_;
  if (!(block_dim_ > 0 && dim_ % block_dim_ == 0 &&
        epsilon_ > 0 && target_rms_ > 0))
    KALDI_ERR << "Invalid configuration in BatchNormComponent.";
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  count_ = 0;
  stats_sum_.Resize(block_dim_);
  stats_sumsq_.Resize(block_dim_);
  if (test_mode_) {
    ComputeDerived();
  }
}



/*
  BATCHNORM_MATH

  This comment describes the equations involved in batch normalization, and
  derives the forward and back-propagation.

  This is all dimension-by-dimension, so we just imagine the inputs
  are scalars x(i), for i=0 .. n-1.

  FORWARD PASS:

  Let 'power' be a constant, equal to -0.5 for regular batch-norm.

  To simplify the math we (conceptually, not physically) do the normalization in
  two stages: first mean, then variance, so we have x(i) -> y(i) -> z(i).

  The name 'rscale' means 'raw scale', meaning the scale before including
  target-rms.  Later we'll define 'scale = target-rms * rscale', to make some
  of the actual computations slightly more efficient.

  Define:   mean = 1/I * sum_i x(i)
            y(i) = x(i) - mean

            var = 1/I \sum_i y(i)^2
         rscale = sqrt(var + epsilon)^power   <---- For regular batchnorm, power == -0.5.
           z(i) = target-rms * rscale * y(i)


  Most of the rest of this comment derives how to compute the derivatives.  If
  you just want the formulas, please skip to the string 'BACKWARD PASS' below.

  We'll use a notation where an apostrophe on something means (the derivative of
  the objective function w.r.t. that thing), so y'(i) is df/dy(i), and so on.
  We are given y'(i).  Propagating the derivatives backward:

    rscale' = (sum_i y(i) z'(i)) * target-rms
            = (sum_i z(i) z'(i)) / rscale

  [ note: d(rscale)/d(var) = power * (var + epsilon)^{power - 1}
                           = power * rscale^{(power-1)/power}  ]

    var' = rscale' * power * rscale^{(power-1)/power}
         = power * (\sum_i z'(i) z(i)) * rscale^{(power-1)/power - 1}
         = power * (\sum_i z'(i) z(i)) * rscale^{-1/power}

  [note: the following formula is of the form "direct term" + "indirect term"]
    y'(i) =  z'(i) * target-rms * rscale   +    2/I y(i) var'

  Now, the above is inconvenient because it contains y(i) which is an intermediate
  quantity.  We reformulate in terms of z(i), using y(i) = z(i) / (target-rms * rscale), so:

  defining
   var_deriv_mod = 2/I * var' / (target-rms * rscale)
                 = 2/I * power/target-rms * (\sum_i z'(i) z(i)) * rscale^{-(1+power)/power}
 we have:
    y'(i) =  z'(i) * target-rms * rscale   +    z(i) var_deriv_mod

 Now,
    mean' = \sum_i y'(i)
          = (target-rms * rscale * \sum_i z'(i))  +  (var_deriv_mod \sum_i z(i))
     [... and the 2nd term above is zero when summed over i, because \sum_i z(i) is zero, ...]
          = target-rms * rscale * \sum_i z(i)
 and:
    x'(i) =  z'(i) * target-rms * rscale   +    z(i) var_deriv_mod   -  1/I mean'
          =  z'(i) * target-rms * rscale   +    z(i) var_deriv_mod   -  1/I * target-rms * rscale * \sum_i z'(i)
          =  target-rms * rscale * (z'(i) - 1/I * \sum_i z'(i))  +  z(i) var_deriv_mod

    It will simplify the code if we define:

      scale = target-rms * rscale.  This way, we can write as follows:

  BACKWARD PASS (recap):

   var_deriv_mod = 2 * power * target-rms^{1/power} * (1/I \sum_i z'(i) z(i)) * scale^{-(1+power)/power}
                .. which for power = -0.5, simplifies to:
   var_deriv_mod = -1.0 / (target-rms^2) * (1/I \sum_i z'(i) z(i)) * scale

           x'(i) = scale * (z'(i) - 1/I * \sum_i z'(i))  + z(i) var_deriv_mod

  */
void* BatchNormComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const CuMatrixBase<BaseFloat> &in,
                                    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(SameDim(in, *out) &&
               (in.NumCols() == dim_ || in.NumCols() == block_dim_));
  if (in.NumCols() != block_dim_) {
    // if block_dim_ != dim_, we recurse; this helps keep the main code
    // simple.
    KALDI_ASSERT(in.Stride() == in.NumCols() && out->Stride() == out->NumCols());
    int32 ratio = dim_ / block_dim_, orig_rows = in.NumRows(),
        orig_cols = in.NumCols(), new_rows = orig_rows * ratio,
        new_cols = orig_cols / ratio;
    CuSubMatrix<BaseFloat> in_reshaped(in.Data(), new_rows, new_cols, new_cols),
        out_reshaped(out->Data(), new_rows, new_cols, new_cols);
    return Propagate(indexes, in_reshaped, &out_reshaped);
  }

  // From this point, we can assume that the num-cols of 'in' and 'out'
  // equals block_dim_.

  if (!test_mode_) {
    // search in the comment above for FORWARD PASS to see what is being
    // implemented here.
    // if this takes too much time due to multiple different CUDA calls,
    // we'll consider making a single kernel for some of it.
    Memo *memo = new Memo;
    int32 num_frames = in.NumRows(), dim = block_dim_;
    memo->num_frames = num_frames;
    memo->mean_uvar_scale.Resize(5, dim);
    CuSubVector<BaseFloat> mean(memo->mean_uvar_scale, 0),
        uvar(memo->mean_uvar_scale, 1),
        scale(memo->mean_uvar_scale, 2);
    mean.AddRowSumMat(1.0 / num_frames, in, 0.0);
    uvar.AddDiagMat2(1.0 / num_frames, in, kTrans, 0.0);
    scale.CopyFromVec(uvar);

    // by applying this scale at this point, we save a multiply later on.
    BaseFloat var_scale = 1.0 / (target_rms_ * target_rms_);
    scale.AddVecVec(-var_scale, mean, mean, var_scale);
    // at this point, 'scale' contains just the variance (times target-rms^{-2}).
    scale.ApplyFloor(0.0);
    scale.Add(var_scale * epsilon_);
    // Now 'scale' contains the variance floored to zero and then with epsilon
    // added [both times 1/target-rms^2].
    scale.ApplyPow(-0.5);
    // now 'scale' is the actual scale we'll use.

    // the next command will do no work if out == in, for in-place propagation.
    out->CopyFromMat(in);
    out->AddVecToRows(-1.0, mean, 1.0);
    out->MulColsVec(scale);
    return static_cast<void*>(memo);
  } else {
    if (offset_.Dim() != block_dim_) {
      if (count_ == 0)
        KALDI_ERR << "Test mode set in BatchNormComponent, but no stats.";
      else  // why was ComputeDerived() not called?
        KALDI_ERR << "Code error in BatchNormComponent";
    }
    out->CopyFromMat(in);
    out->MulColsVec(scale_);
    out->AddVecToRows(1.0, offset_, 1.0);
    return NULL;
  }
}

void BatchNormComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,  // unused
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo_in,
    Component *to_update,  // unused
    CuMatrixBase<BaseFloat> *in_deriv) const {

  KALDI_ASSERT(SameDim(out_value, out_deriv) &&
               SameDim(out_value, *in_deriv) &&
               (out_value.NumCols() == dim_ ||
                out_value.NumCols() == block_dim_));
  if (out_value.NumCols() != block_dim_) {
    // if block_dim_ != dim_, we recurse; this helps keep the main code
    // simple.
    KALDI_ASSERT(out_value.Stride() == out_value.NumCols() &&
                 out_deriv.Stride() == out_deriv.NumCols() &&
                 in_deriv->Stride() == in_deriv->NumCols());
    int32 ratio = dim_ / block_dim_,
        orig_rows = out_value.NumRows(),
        orig_cols = out_value.NumCols(),
        new_rows = orig_rows * ratio, new_cols = orig_cols / ratio;
    CuSubMatrix<BaseFloat> out_value_reshaped(out_value.Data(), new_rows,
                                              new_cols, new_cols),
        out_deriv_reshaped(out_deriv.Data(), new_rows, new_cols, new_cols),
        in_deriv_reshaped(in_deriv->Data(), new_rows, new_cols, new_cols);
    // we'll never use in_value, so pass it in unchanged.
    Backprop(debug_info, indexes, in_value,
             out_value_reshaped, out_deriv_reshaped,
             memo_in, to_update, &in_deriv_reshaped);
    return;
  }

  Memo *memo = static_cast<Memo*>(memo_in);

  if (!test_mode_) {
    // search above for BACKWARD PASS for a comment describing the math.
    KALDI_ASSERT(memo != NULL && "memo not passed into backprop");
    int32 num_frames = memo->num_frames;
    KALDI_ASSERT(out_value.NumRows() == num_frames);
    CuSubVector<BaseFloat>
        scale(memo->mean_uvar_scale, 2),
        var_deriv_mod(memo->mean_uvar_scale, 3),
        temp(memo->mean_uvar_scale, 4);

    // var_deriv_mod is going to contain:
    //  2 * power * target-rms^{1/power} * (1/I \sum_i z'(i) z(i)) * scale^{-(1+power)/power}
    // which for power = -0.5 simplifies to:
    // -1.0 / (target_rms * target_rms).
    // but for now we don't have the power of 'scale', we'll add that later.
    BaseFloat coeff = -1.0 / (target_rms_ * target_rms_ * num_frames);

    var_deriv_mod.AddDiagMatMat(coeff, out_value, kTrans,
                                out_deriv, kNoTrans, 0.0);
    var_deriv_mod.MulElements(scale);

    temp.AddRowSumMat(-1.0 / num_frames, out_deriv, 0.0);
    // the following statement does no work if in_deriv and out_deriv are the
    // same matrix.
    in_deriv->CopyFromMat(out_deriv);
    in_deriv->AddVecToRows(1.0, temp);
    // At this point, *in_deriv contains
    // (z'(i) - 1/I * \sum_i z'(i))
    in_deriv->MulColsVec(scale);
    // At this point, *in_deriv contains
    // scale * (z'(i) - 1/I * \sum_i z'(i))

    in_deriv->AddMatDiagVec(1.0, out_value, kNoTrans,
                            var_deriv_mod, 1.0);

    // At this point, *in_deriv contains what we described in the comment
    // starting BATCHNORM_MATH as:
    // x'(i) = scale * (z'(i) - 1/I * \sum_i z'(i))  + z(i) var_deriv_mod
  } else {
    KALDI_ASSERT(offset_.Dim() == block_dim_);
    // the next call does no work if they point to the same memory.
    in_deriv->CopyFromMat(out_deriv);
    in_deriv->MulColsVec(scale_);
  }
}

void BatchNormComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    void *memo_in) {
  // in test mode this component does not store stats, it doesn't provide the
  // kStoresStats flag.
  KALDI_ASSERT(!test_mode_);
  KALDI_ASSERT(out_value.NumCols() == dim_ || out_value.NumCols() == block_dim_);
  if (out_value.NumCols() != block_dim_) {
    // if block_dim_ != dim_, we recurse; this helps keep the main code
    // simple.
    KALDI_ASSERT(out_value.Stride() == out_value.NumCols());
    int32 ratio = dim_ / block_dim_,
        orig_rows = out_value.NumRows(),
        orig_cols = out_value.NumCols(),
        new_rows = orig_rows * ratio, new_cols = orig_cols / ratio;
    CuSubMatrix<BaseFloat> out_value_reshaped(out_value.Data(), new_rows,
                                              new_cols, new_cols);
    // we'll never use in_value, so just pass it in unchanged.
    StoreStats(in_value, out_value_reshaped, memo_in);
    return;
  }

  Memo *memo = static_cast<Memo*>(memo_in);
  KALDI_ASSERT(out_value.NumRows() == memo->num_frames);

  CuSubVector<BaseFloat> mean(memo->mean_uvar_scale, 0),
      uvar(memo->mean_uvar_scale, 1);
  KALDI_ASSERT(mean.Dim() == block_dim_ && memo->num_frames > 0);
  BaseFloat num_frames = memo->num_frames;
  if (stats_sum_.Dim() != block_dim_) {
    stats_sum_.Resize(block_dim_);
    stats_sumsq_.Resize(block_dim_);
    KALDI_ASSERT(count_ == 0);
  }
  count_ += num_frames;
  stats_sum_.AddVec(num_frames, mean, 1.0);
  stats_sumsq_.AddVec(num_frames, uvar, 1.0);
}

void BatchNormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<BatchNormComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<BlockDim>");
  ReadBasicType(is, binary, &block_dim_);
  ExpectToken(is, binary, "<Epsilon>");
  ReadBasicType(is, binary, &epsilon_);
  ExpectToken(is, binary, "<TargetRms>");
  ReadBasicType(is, binary, &target_rms_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<Count>");
  ReadBasicType(is, binary, &count_);
  ExpectToken(is, binary, "<StatsMean>");
  stats_sum_.Read(is, binary);
  ExpectToken(is, binary, "<StatsVar>");
  stats_sumsq_.Read(is, binary);
  stats_sumsq_.AddVecVec(1.0, stats_sum_, stats_sum_, 1.0);
  stats_sum_.Scale(count_);
  stats_sumsq_.Scale(count_);
  ExpectToken(is, binary, "</BatchNormComponent>");
  ComputeDerived();
  Check();
}

void BatchNormComponent::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<BatchNormComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<BlockDim>");
  WriteBasicType(os, binary, block_dim_);
  WriteToken(os, binary, "<Epsilon>");
  WriteBasicType(os, binary, epsilon_);
  WriteToken(os, binary, "<TargetRms>");
  WriteBasicType(os, binary, target_rms_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<Count>");
  WriteBasicType(os, binary,  count_);
  CuVector<BaseFloat> mean(stats_sum_), var(stats_sumsq_);
  if (count_ != 0) {
    mean.Scale(1.0 / count_);
    var.Scale(1.0 / count_);
    var.AddVecVec(-1.0, mean, mean, 1.0);
  }
  WriteToken(os, binary, "<StatsMean>");
  mean.Write(os, binary);
  WriteToken(os, binary, "<StatsVar>");
  var.Write(os, binary);
  WriteToken(os, binary, "</BatchNormComponent>");
}

void BatchNormComponent::Scale(BaseFloat scale) {
  if (scale == 0) {
    count_ = 0.0;
    stats_sum_.SetZero();
    stats_sumsq_.SetZero();
  } else {
    count_ *= scale;
    stats_sum_.Scale(scale);
    stats_sumsq_.Scale(scale);
  }
}


void BatchNormComponent::Add(BaseFloat alpha, const Component &other_in) {
  const BatchNormComponent *other =
      dynamic_cast<const BatchNormComponent*>(&other_in);
  count_ += alpha * other->count_;
  stats_sum_.AddVec(alpha, other->stats_sum_);
  stats_sumsq_.AddVec(alpha, other->stats_sumsq_);
  // this operation might change offset_ and scale_, so we recompute them
  // in this instance (but not in Scale()).
  ComputeDerived();
}

void BatchNormComponent::ZeroStats() {
  // We only zero the stats if we're not in test mode.  In test mode, this would
  // be dangerous as the stats are the source for the transform, and zeroing
  // them and then calling ComputeDerived() again would remove the transform
  // parameters (offset_ and scale_).
  if (!test_mode_) {
    count_ = 0.0;
    stats_sum_.SetZero();
    stats_sumsq_.SetZero();
  }
}


} // namespace nnet3
} // namespace kaldi

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
  scale_.ApplyPow(-0.5);
  // now scale_ = min(variance, epsilon)^{-0.5}.
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
    dim_(other.dim_), block_dim_(other.block_dim_), epsilon_(other.epsilon_),
    target_rms_(other.target_rms_), test_mode_(other.test_mode_),
    count_(other.count_), stats_sum_(other.stats_sum_),
    stats_sumsq_(other.stats_sumsq_) {
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
  BATCH_NORM_MATH

  This comment describes the equations involved in batch normalization, and
  derives the forward and back-propagation.

  This is all dimension-by-dimension, so we just imagine the inputs
  are scalars x(i), for i=0 .. n-1.

  FORWARD PASS:

  Define xsum  = sum_i x(i)
         x2sum = sum_i x(i)^2
          mean = xsum / n
           var = x2sum / n - (mean*mean)
         scale = sqrt(var + epsilon)^{-0.5}
        offset = -mean * scale

      y(i) = scale * x(i) + offset

   Most of the rest of this comment derives how to compute the derivatives.  If
   you just want the formulas, please skip to the string 'BACKWARD PASS' below.

  We'll use a notation where an apostrophe on something means (the derivative of
  the objective function w.r.t. that thing), so y'(i) is df/dy(i), and so on.
  We are given y'(i).  Propagating the derivatives backward:
     offset' = sum_i y'(i)
     scale' = (sum_i y'(i) * x(i)) - offset' * mean
       var' = scale' * -0.5 * sqrt(var + epsilon)^{-1.5}
            = -0.5 * scale' * scale^3
      mean' = -offset' * scale - 2 * mean * var'
      xsum' = mean' / n
     x2sum' = var' / n

  So the derivatives propagated back to the original data are:
     x'(i) = y'(i) * scale  +  xsum'  +  x(i) * x2sum'

  The above is quite complicated to compute, but we can use some invariances
  to work out a simpler way to compute the derivatives.

  Firstly, note that x'(i) is of the form:

   x'(i) =  y'(i) * scale + [affine function of x(i)].

   [it's a 1-d affine function, i.e. offset and scale].
 This has the same functional form as:

  x'(i) =  y'(i) * scale + [affine function of y(i)].

  since y(i) is an affine function of x(i) with nonzero scale.
  Because the output is invariant to shifts in the input, sum_i x'(i)
  will be zero.  This is sufficient to determine the bias
  term in the affine function.  [Note: the scale on y(i) doesn't
  come into it because the y(i) sum to zero].  The offset
  will just be (sum_i y'(i) * scale / n); this makes the sum of x'(i) zero.
  So let's write it as

    x'(i) =  (y'(i) - 1/n sum_i y'(i)) * scale + alpha y(i).

  and it will be convenient to define:

  x_deriv_base(i) = (y'(i) - 1/n sum_i y'(i)) * scale

  which is just y'(i) with mean subtraction, scaled according to
  the scale used in the normalization.  So write

   x'(i) = x_deriv_base(i) + alpha y(i).

 The question is, what is the scale alpha.  We don't actually need to
 do any differentiation to figure this out.  First, assume there is
 no "+ epsilon" in the variance; later we'll explain why this doesn't
 matter.  The key to working out alpha is that the output is invariant
 to scaling of the input.  Assume we scale around the input's mean,
 since that makes the math simpler.  We can express this by the
 constraint that (\sum_i x'(i) * (x(i) - avg-x)) = 0.  This is
 equivalent to the constraint that (\sum_i x'(i) y (i)) = 0, since
 y(i) is x(i) - avg-x times a nonzero scale.  We'll use this contraint
 to determine alpha, Using the above expressionfor x(i), we can write
 this constraint as:
   \sum_i ( y(i) x_deriv_base(i)  + alpha y(i) y(i)) = 0.
 Now, since we said we'd ignore the epsilon, the output has unit variance,
 so we know that \sum_i y(i) y(i) = n.
 So alpha = - \sum_i y(i) x_deriv_base(i) / n.  We can actually re-imagine
 the epsilon term (or variance-flooring) as having been implemented by
 adding a couple extra rows to the matrix with suitable values, and zero
 output-deriv for those rows.  If you think about it carefully you'll see that
 the formula above is valid even if there is an extra term
 in the variance.  Anyway the correctness of the derivative will get tested
 throughly by the component unit-tests.

 So to recap, here is the backprop.

 BACKWARD PASS:

  We are given y'(i), scale, and y(i).

  We compute:
    x_deriv_base(i) = (y'(i) - 1/n sum_i y'(i)) * scale
              alpha = - \sum_i y(i) x_deriv_base(i) / n
              x'(i) = x_deriv_base(i) + alpha y(i)
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
    memo->mean_uvar_scale.Resize(4, dim);
    CuSubVector<BaseFloat> mean(memo->mean_uvar_scale, 0),
        uvar(memo->mean_uvar_scale, 1),
        scale(memo->mean_uvar_scale, 2);
    mean.AddRowSumMat(1.0 / num_frames, in, 0.0);
    uvar.AddDiagMat2(1.0 / num_frames, in, kTrans, 0.0);
    scale.CopyFromVec(uvar);
    // by applying this scale at this point, we save a multiply later on.
    BaseFloat var_scale = 1.0 / (target_rms_ * target_rms_);
    scale.AddVecVec(-var_scale, mean, mean, var_scale);
    // at this point, 'scale' contains just the variance [divided by target-rms^2].
    scale.ApplyFloor(0.0);
    scale.Add(var_scale * epsilon_);
    // Now 'scale' contains the variance floored to zero and then with epsilon
    // added [both divided by target-rms^2].
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
    CuSubVector<BaseFloat> temp(memo->mean_uvar_scale, 3),
        scale(memo->mean_uvar_scale, 2);
    temp.AddRowSumMat(-1.0 / num_frames, out_deriv, 0.0);
    // the following does no work if in_deriv and out_deriv are the same matrix.
    in_deriv->CopyFromMat(out_deriv);
    in_deriv->AddVecToRows(1.0, temp);
    in_deriv->MulColsVec(scale);
    // at this point, 'in_deriv' contains:
    // x_deriv_base(i) = (y'(i) - 1/n sum_i y'(i)) * scale
    temp.AddDiagMatMat(-1.0 / (num_frames * target_rms_ * target_rms_),
                       out_value, kTrans, *in_deriv, kNoTrans, 0.0);
    // now, 'temp' contains the quantity which we described
    // in the math as:
    // alpha = - \sum_i y(i) x_deriv_base(i) / n.
    // The factor 1 / (target_rms_ * target_rms_) comes from following
    // this additional scaling factor through the math.  In the comment I said
    // "we know that \sum_i y(i) y(i) = n".  Taking target-rms into account
    // this becomes "we know that \sum_i y(i) y(i) = n * target-rms^2".
    in_deriv->AddMatDiagVec(1.0, out_value, kNoTrans, temp, 1.0);
    // At this point, in_deriv contains  x'(i) = x_deriv_base(i) + alpha y(i).
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




/**
   MEMORY_NORM_MATH

   This comment describes the equations involved in 'memory-norm'.
   memory-norm is like batch normalization, except instead of computing
   everything on the current minibatch, we deal with decaying averages
   over time, interpreted as expectations.  We'll firm up the math later.
   The idea is to obtain a form of batch-norm that is compatible with
   use in recurrent neural nets.

   Everything is dimension by dimension here, so let's imagine the input and
   output are one-dimensional.  Any index 'i' is going to be like a frame index
   or an index referring to a sample.  We'll be writing down some expectations,
   and we're rather cavalier with notation; these basically mean
   exponentially-decaying weighted averages over time.

   The input will be x(i), and the output y(i).

   Each frame will have a weight, w(i) >= 0.  (these will be part of the
   decaying averages)...

   Let's define
      count = \sum_i w(i)
      sum =  \sum_i w(i) x(i)
      sumsq =  \sum_i w(i) x(i)^2

   We can compute:
      mean = sum / count
      var = epsilon + (sumsq / count) - (mean * mean)
      scale = target_rms * var^{-0.5}

      y(i) = (x(i) - mean) * scale.

   We are given the derivatives of the objective function w.r.t. the
   outputs; we'll write these as y'(i) [CAUTION: this is nonstandard
   notation.  An apostrophe on something means the derivative of the
   objective function w.r.t. that thing].

   Over this data, with these weights, we can compute the derivative
   of the objective w.r.t. the mean and the scale:

       mean' = -scale * \sum_i w(i) y'(i)
      scale' = \sum_i w(i) y'(i) (x(i) - mean)
             = 1/scale \sum_i w(i) y'(i) y(i)
        var' = -0.5 target_rms var^{-1.5} scale'
             = -0.5 target_rms var^{-1.5} (1/scale) \sum_i w(i) y'(i) y(i)
                 .. and using 1/scale = var^{0.5}/target_rms,
             = -0.5 var^{-1} \sum_i w(i) y'(i) y(i)                      (*)


   It will be convenient to write down 'per-frame' versions of all of these
   quantities, which are divided by the total count:
        mean_norm' = mean' / count
        scale_norm' = scale' / count
        var_norm' = var' / count
   (we keep the apostrophe on these quantities as it clarifies that they
   are derivatives of the objective function w.r.t something).

    Now, 'var' can be written as:
        var = epsilon + (1/count) \sum_i w(i) (x(i) - mean)^2
    and the following formula is more convenient to propagate the derivative
    back to an x(i).
        Note: the following has 3 terms, which we can think of as
     "direct term" (given fixed mean and scale),
     "term via mean" (term that comes via derivative of the mean)
     "term via scale" (term that comes via derivative of the scale)


        x'(i) = y'(i)*scale + mean_norm' + 2 var_norm' (x(i) - mean)
              = y'(i)*scale + mean_norm' + 2 var_norm' y(i) / scale
               ... and substituting in the equation (*) above for var', using var_norm' = var'/scale,
               and rearranging slightly:
              = y'(i)*scale + mean_norm' - y(i) * var^{-1}/scale * 1/count * \sum_i w(i) y'(i) y(i)
              .. and using scale=target-rms * var^{-0.5}, so var^{-1}/scale = var^{-0.5}/target-rms = scale/target-rms^2:
              = y'(i)*scale + mean_norm' - y(i) * scale/(count*target-rms^2) * \sum_i w(i) y'(i) y(i)
            .. and considering that the factor of 'scale' appears (directly or indirectly) in all 3
              of the terms in the above expression, we can reorganize this as:
              = scale * (y'(i) - 1/count*\sum_i w(i)*y(i) - 1/(count*target-rms^2) * \sum_i w(i) y'(i) y(i))
*/


void MemoryNormComponent::SetTestMode(bool test_mode) {
  if (test_mode && stats_count_ <= 0) {
    KALDI_WARN << "Refusing to set test-mode in MemoryNormComponent since no "
        "stats are present.";
    return;
  }
  test_mode_ = test_mode;
}

void MemoryNormComponent::Check() const {
  KALDI_ASSERT(dim_ > 0 && block_dim_ > 0 && dim_ % block_dim_ == 0 &&
               epsilon_ > 0.0 && target_rms_ > 0.0 &&
               stats_count_ >= 0.0 && backward_count_ >= 0.0);

}

MemoryNormComponent::MemoryNormComponent(const MemoryNormComponent &other):
    dim_(other.dim_), block_dim_(other.block_dim_), epsilon_(other.epsilon_),
    target_rms_(other.target_rms_),
    include_indirect_derivative_(other.include_indirect_derivative_),
    test_mode_(other.test_mode_),
    stats_count_(other.stats_count_), backward_count_(other.backward_count_),
    data_(other.data_) {
  Check();
}


std::string MemoryNormComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_ << ", block-dim=" << block_dim_
         << ", epsilon=" << epsilon_ << ", target-rms=" << target_rms_
         << ", include-indirect-derivative="
         << (include_indirect_derivative_ ? "true" : "false")
         << ", stats-count=" << stats_count_ << ", backward-count="
         << backward_count_
         << ", test-mode=" << (test_mode_ ? "true" : "false");
  if (stats_count_ > 0.0) {
    CuSubVector<BaseFloat> x_mean(data_, 0),
        y_deriv(data_, 2), y_deriv_y(data_, 3),
        scale(data_, 4);
    if (stats_count_ > 0.0)
      stream << ", x-mean=" << SummarizeVector(x_mean)
             << ", scale=" << SummarizeVector(scale);
    if (backward_count_ > 0.0)
      stream << ", y-deriv=" << SummarizeVector(y_deriv)
             << ", y-deriv-y=" << SummarizeVector(y_deriv_y);
  }
  return stream.str();
}

void MemoryNormComponent::InitFromConfig(ConfigLine *cfl) {
  dim_ = -1;
  block_dim_ = -1;
  epsilon_ = 1.0e-03;
  target_rms_ = 1.0;
  include_indirect_derivative_ = true;
  test_mode_ = false;

  bool ok = cfl->GetValue("dim", &dim_);
  cfl->GetValue("block-dim", &block_dim_);
  cfl->GetValue("epsilon", &epsilon_);
  cfl->GetValue("target-rms", &target_rms_);
  cfl->GetValue("include-indirect-derivative", &include_indirect_derivative_);
  cfl->GetValue("test-mode", &test_mode_);
  if (!ok || dim_ <= 0) {
    KALDI_ERR << "MemoryNormComponent must have 'dim' specified, and > 0";
  }
  if (block_dim_ == -1)
    block_dim_ = dim_;
  if (!(block_dim_ > 0 && dim_ % block_dim_ == 0 &&
        epsilon_ > 0 && target_rms_ > 0))
    KALDI_ERR << "Invalid configuration in MemoryNormComponent.";
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  stats_count_ = 0.0;
  backward_count_ = 0.0;
  data_.Resize(5, block_dim_);
}



void* MemoryNormComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
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

  if (out->Data() != in.Data())
    out->CopyFromMat(in);

  if (test_mode_ && stats_count_ <= 0.0)
    KALDI_ERR << "Test mode set but no stats available.";

  // From this point, we can assume that the num-cols of 'in' and 'out'
  // equals block_dim_.
  Memo *memo = NULL;
  if (!test_mode_) {
    memo = GetMemo(in);
  }

  if (test_mode_) {
    CuSubVector<BaseFloat> x_mean(data_, 0), scale(data_, 4);
    out->AddVecToRows(-1.0, x_mean);
    out->MulColsVec(scale);
  } else {
    CuSubVector<BaseFloat> x_mean(memo->data, 0),
        scale(memo->data, 4);
    out->AddVecToRows(-1.0, x_mean);
    out->MulColsVec(scale);
  }
  return memo;
}


MemoryNormComponent::Memo* MemoryNormComponent::GetMemo(
    const CuMatrixBase<BaseFloat> &in) const {
  KALDI_ASSERT(in.NumCols() == block_dim_ && !test_mode_ &&
               stats_count_ >= 0.0);
  Memo *memo = new Memo;
  BaseFloat old_stats_count = stats_count_,
      num_frames = in.NumRows(),
      new_stats_count = num_frames + old_stats_count,
      old_weight = old_stats_count / new_stats_count;

  // The information in 'memo' will be copied to *this when
  // StoreStats() is caled (we can't update it in the Propagate()
  // function for 'const' reasons).
  memo->stats_count = new_stats_count;
  memo->backward_count = backward_count_;
  memo->data = data_;

  CuSubVector<BaseFloat> x_mean(memo->data, 0),
      x_uvar(memo->data, 1), scale(memo->data, 4);
  // Each row of 'in' gets a weight of 1.0 / new_stats_count in the stats.
  x_mean.AddRowSumMat(1.0 / new_stats_count, in, old_weight);
  x_uvar.AddDiagMat2(1.0 / new_stats_count, in, kTrans, old_weight);

  scale.CopyFromVec(x_uvar);
  // we save a CUDA operation by applying the scale 'target_rms_scale' before doing
  // ApplyPow(-0.5), and this requires taking it to the power -2.
  BaseFloat target_rms_scale = 1.0 / (target_rms_ * target_rms_);
  scale.AddVecVec(-target_rms_scale, x_mean, x_mean, target_rms_scale);
  // at this point, 'scale' is the variance.
  scale.ApplyFloor(0.0);
  scale.Add(epsilon_ * target_rms_scale);
  scale.ApplyPow(-0.5);
  // OK, now 'scale' is the scale.
  return memo;
}

void MemoryNormComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &in_value,  // unused.
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo_in,
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {

  KALDI_ASSERT(SameDim(out_deriv, *in_deriv) &&
               (out_deriv.NumCols() == dim_ ||
                out_deriv.NumCols() == block_dim_));
  if (out_deriv.NumCols() != block_dim_) {
    // if block_dim_ != dim_, we recurse; this helps keep the main code
    // simple.
    KALDI_ASSERT(out_deriv.Stride() == out_deriv.NumCols() &&
                 in_deriv->Stride() == in_deriv->NumCols());
    if (out_value.NumRows() != 0) {
      KALDI_ASSERT(out_value.Stride() == out_value.NumCols());
    }
    int32 ratio = dim_ / block_dim_,
        orig_rows = out_value.NumRows(),
        orig_cols = out_value.NumCols(),
        new_rows = orig_rows * ratio, new_cols = orig_cols / ratio;
    CuSubMatrix<BaseFloat>
        out_deriv_reshaped(out_deriv.Data(), new_rows, new_cols, new_cols),
        in_deriv_reshaped(in_deriv->Data(), new_rows, new_cols, new_cols);

    // we'll never use in_value, so pass it in unchanged.
    if (out_value.NumRows() != 0) {
      CuSubMatrix<BaseFloat> out_value_reshaped(out_value.Data(), new_rows,
                                                new_cols, new_cols);
      Backprop(debug_info, indexes, in_value,
               out_value_reshaped, out_deriv_reshaped,
               memo_in, to_update_in, &in_deriv_reshaped);
    } else {
      Backprop(debug_info, indexes, in_value,
               out_value, out_deriv_reshaped,
               memo_in, to_update_in, &in_deriv_reshaped);
    }
    return;
  }

  // assume in_deriv is non-NULL, because a non-updatable Component will not
  // have the backprop called if the in_deriv is non-NULL.

  if (test_mode_) {
    // In test mode we treat it as a fixed scale and offset.
    KALDI_ASSERT(memo_in == NULL && stats_count_ != 0.0);
    // the following is a no-op if in_deriv and out_deriv are the same matrix.
    in_deriv->CopyFromMat(out_deriv);
    CuSubVector<BaseFloat> scale(data_, 4);
    in_deriv->MulColsVec(scale);
    return;
  }

  // OK, we're not in test mode.
  // Before computing 'in_deriv', we may need to store some stats.
  if (include_indirect_derivative_ && to_update_in != NULL) {
    // Store some stats which are necessary to compute the 'indirect derivative'
    // term (this is analogous to the part of the derivative in regular backprop
    // that comes from the objf derivative w.r.t. the mean and variance stats).
    //
    // Note: instead of simply adding to the stats 'y_deriv' and 'y_deriv_y',
    // the following equations do a kind of weighted combination, because
    // these stats are stored normalized by the total count (backward_count_).
    MemoryNormComponent *to_update =
        dynamic_cast<MemoryNormComponent*>(to_update_in);
    BaseFloat backward_count = to_update->backward_count_,
        num_frames = in_deriv->NumRows(),
        new_backward_count = backward_count + num_frames,
        old_weight = backward_count / new_backward_count;
    CuSubVector<BaseFloat> y_deriv(to_update->data_, 2),
        y_deriv_y(to_update->data_, 3);
    // The factor 1.0 / new_backward_count that appears below can be perhaps more
    // clearly written as follows: first define
    //       new_weight = num_frames / new_backward_count
    // and then write new_weight / num_frames, which simplifies to
    // 1.0 / new_backward_count.  The factor of 1.0 / num_frames is necessary to
    // convert from data sums to a per-frame average.
    y_deriv.AddRowSumMat(1.0 / new_backward_count, out_deriv, old_weight);
    y_deriv_y.AddDiagMatMat(1.0 / new_backward_count, out_deriv, kTrans,
                            out_value, kNoTrans, old_weight);
    to_update->backward_count_ = new_backward_count;
    // We don't bother calling to_update->ComputeDerived()-- although it would
    // be harmless-- because in the current situations where this code is
    // reached, to_update will be the delta_nnet_, and the derived parameter
    // 'scale') of delta_nnet_ aren't used.

    // to_update->ComputeDerived();
  }

  // the following does no work if in_deriv and out_deriv are the same matrix.
  in_deriv->CopyFromMat(out_deriv);

  Memo *memo = static_cast<Memo*>(memo_in);
  if (memo->backward_count != 0.0) {
    CuSubVector<BaseFloat> y_deriv(memo->data, 2),
        y_deriv_y(memo->data, 3);
    in_deriv->AddVecToRows(-1.0, y_deriv);
    in_deriv->AddMatDiagVec(-1.0 / (target_rms_ * target_rms_),
                            out_value, kNoTrans, y_deriv_y);
  }
  CuSubVector<BaseFloat> scale(memo->data, 4);
  in_deriv->MulColsVec(scale);

}


void MemoryNormComponent::ComputeDerived() {
  KALDI_ASSERT(stats_count_ >= 0.0 && data_.NumRows() == 5);
  if (stats_count_ == 0.0) {
    // zero 'scale'.
    data_.Row(4).SetZero();
    return;
  }
  CuSubVector<BaseFloat>  x_mean(data_, 0), x_uvar(data_, 1),
       scale(data_, 4);
  scale.CopyFromVec(x_uvar);
  // we save a CUDA operation by applying the scale 'target_rms_scale' before doing
  // ApplyPow(-0.5), and this requires taking it to the power -2.
  BaseFloat target_rms_scale = 1.0 / (target_rms_ * target_rms_);
  scale.AddVecVec(-target_rms_scale, x_mean, x_mean, target_rms_scale);
  // at this point, 'scale' is the variance (divided by target_rms^2).
  scale.ApplyFloor(0.0);
  scale.Add(epsilon_ * target_rms_scale);
  scale.ApplyPow(-0.5);
}

void MemoryNormComponent::StoreStats(
    const CuMatrixBase<BaseFloat> &, // in_value
    const CuMatrixBase<BaseFloat> &, // out_value
    void *memo_in) {
  // in test mode this component does not store stats; it doesn't provide the
  // kStoresStats flag so this function won't be called.
  KALDI_ASSERT(!test_mode_ && memo_in != NULL && stats_count_ >= 0.0);

  // We don't actually need 'in_value' and 'out_value', as the
  // required statistics are already stored in 'memo_in'.
  Memo *memo = static_cast<Memo*>(memo_in);

  // check that the memo's stats count is more than our stats_count_,
  // which it should be because the memo should have added extra stats,
  // and StoreStats() should be called directly after the Propagate()
  // function.
  // This could possibly fail with memo_in->stats_count == stats_count_
  // due to roundoff, if you trained with batchnorm-stats-scale set at 1,
  // but that would be a poor choice of parameters anyway as
  // roundoff would be a big problem.
  KALDI_ASSERT(memo->stats_count > stats_count_);

  stats_count_ = memo->stats_count;
  // Copying the entire data matrix should be safe because
  // StoreStats() is always called directly after the corresponding
  // Propagate(), and on the same object; and there should be
  // no possibility that other things in this->data changed in
  // the interim.
  data_.CopyFromMat(memo->data);
}

void MemoryNormComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<MemoryNormComponent>", "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "<BlockDim>");
  ReadBasicType(is, binary, &block_dim_);
  ExpectToken(is, binary, "<Epsilon>");
  ReadBasicType(is, binary, &epsilon_);
  ExpectToken(is, binary, "<TargetRms>");
  ReadBasicType(is, binary, &target_rms_);
  ExpectToken(is, binary, "<IncludeIndirectDerivative>");
  ReadBasicType(is, binary, &include_indirect_derivative_);
  ExpectToken(is, binary, "<TestMode>");
  ReadBasicType(is, binary, &test_mode_);
  ExpectToken(is, binary, "<StatsCount>");
  ReadBasicType(is, binary, &stats_count_);
  ExpectToken(is, binary, "<BackwardCount>");
  ReadBasicType(is, binary, &backward_count_);
  ExpectToken(is, binary, "<Data>");
  data_.Read(is, binary);
  ExpectToken(is, binary, "</MemoryNormComponent>");
  Check();
}

void MemoryNormComponent::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<MemoryNormComponent>");
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "<BlockDim>");
  WriteBasicType(os, binary, block_dim_);
  WriteToken(os, binary, "<Epsilon>");
  WriteBasicType(os, binary, epsilon_);
  WriteToken(os, binary, "<TargetRms>");
  WriteBasicType(os, binary, target_rms_);
  WriteToken(os, binary, "<IncludeIndirectDerivative>");
  WriteBasicType(os, binary, include_indirect_derivative_);
  WriteToken(os, binary, "<TestMode>");
  WriteBasicType(os, binary, test_mode_);
  WriteToken(os, binary, "<StatsCount>");
  WriteBasicType(os, binary,  stats_count_);
  WriteToken(os, binary, "<BackwardCount>");
  WriteBasicType(os, binary,  backward_count_);
  WriteToken(os, binary, "<Data>");
  data_.Write(os, binary);
  WriteToken(os, binary, "</MemoryNormComponent>");
}

void MemoryNormComponent::Scale(BaseFloat scale) {
  if (scale <= 0) {
    if (scale < 0.0)
      KALDI_WARN << "Setting stats to zero in MemoryNormComponent: requested scale = "
                 << scale;
    // If scale is negative we zero the stats.  This may not always be the right
    // thing to do, so we warn.
    data_.SetZero();
    stats_count_ = 0.0;
    backward_count_ = 0.0;
  } else {
    stats_count_ *= scale;
    backward_count_ *= scale;
    // 'data_' doesnt need to be changed, as all the quantities it contains are
    // normalized by the count.
  }
}


void MemoryNormComponent::Add(BaseFloat alpha, const Component &other_in) {
  const MemoryNormComponent *other =
      dynamic_cast<const MemoryNormComponent*>(&other_in);

  static bool warned = false;
  if (alpha < 0.0) {
    if (!warned) {
      warned = true;
      KALDI_WARN << "Adding MemoryNormComponent with negative scale: will do nothing "
                 << "(will not warn again).";
    }
    return;
  }

  BaseFloat
      new_stats_count = stats_count_ + alpha * other->stats_count_,
      new_backward_count = backward_count_ + alpha * other->backward_count_;

  if (new_stats_count > 0.0) {
    // This block sets rows 0 and 1 of data_, which we call 'x_mean' and
    // 'x_uvar, to the appropriate weighted combination of 'this' and 'other'.
    BaseFloat this_scale = stats_count_ / new_stats_count,
        other_scale = alpha * other->stats_count_ / new_stats_count;
    data_.RowRange(0, 2).Scale(this_scale);
    data_.RowRange(0, 2).AddMat(other_scale, other->data_.RowRange(0, 2));
  }
  if (new_backward_count > 0.0) {
    // This block sets rows 2 and 3 of data_, which we call 'y_deriv' and
    // 'y_deriv_y', to the appropriate weighted combination of 'this' and
    // 'other'.
    BaseFloat this_scale = backward_count_ / new_backward_count,
        other_scale = alpha * other->backward_count_ / new_backward_count;
    data_.RowRange(2, 2).Scale(this_scale);
    data_.RowRange(2, 2).AddMat(other_scale, other->data_.RowRange(2, 2));
  }
  stats_count_ = new_stats_count;
  backward_count_ = new_backward_count;
  ComputeDerived();
}

void MemoryNormComponent::ZeroStats() {
  // We only zero the stats if we're not in test mode.  In test mode, this would
  // be dangerous as the stats aren't really considered to be stats, they become
  // a fixed part of the model.
  if (!test_mode_) {
    stats_count_ = 0.0;
    backward_count_ = 0.0;
    data_.SetZero();
  }
}




} // namespace nnet3
} // namespace kaldi

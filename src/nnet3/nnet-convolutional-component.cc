// nnet3/nnet-convolutional-component.cc

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet3/nnet-convolutional-component.h"
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {


TimeHeightConvolutionComponent::TimeHeightConvolutionComponent():
    use_natural_gradient_(true),
    num_minibatches_history_(4.0) { }

TimeHeightConvolutionComponent::TimeHeightConvolutionComponent(
    const TimeHeightConvolutionComponent &other):
    UpdatableComponent(other),  // initialize base-class
    model_(other.model_),
    all_time_offsets_(other.all_time_offsets_),
    time_offset_required_(other.time_offset_required_),
    linear_params_(other.linear_params_),
    bias_params_(other.bias_params_),
    max_memory_mb_(other.max_memory_mb_),
    use_natural_gradient_(other.use_natural_gradient_),
    num_minibatches_history_(other.num_minibatches_history_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_) {
  Check();
}


void TimeHeightConvolutionComponent::Check() const {
  model_.Check();
  KALDI_ASSERT(bias_params_.Dim() == model_.num_filters_out &&
               linear_params_.NumRows() == model_.ParamRows() &&
               linear_params_.NumCols() == model_.ParamCols());
}

int32 TimeHeightConvolutionComponent::InputDim() const {
  return model_.InputDim();
}

int32 TimeHeightConvolutionComponent::OutputDim() const {
  return model_.OutputDim();
}

std::string TimeHeightConvolutionComponent::Info() const {
  std::ostringstream stream;
  // The output of model_.Info() has been designed to be suitable
  // as a component-level info string, it has
  // {num-filters,height}-{in-out}, offsets=[...], required-time-offsets=[...],
  // {input,output}-dim.
  stream << UpdatableComponent::Info() << ", " << model_.Info();
  PrintParameterStats(stream, "filter-params", linear_params_);
  PrintParameterStats(stream, "bias-params", bias_params_, true);
  stream << ", num-params=" << NumParameters()
         << ", max-memory-mb=" << max_memory_mb_
         << ", use-natural-gradient=" << use_natural_gradient_;
  if (use_natural_gradient_) {
    stream << ", num-minibatches-history=" << num_minibatches_history_
           << ", rank-in=" << preconditioner_in_.GetRank()
           << ", rank-out=" << preconditioner_out_.GetRank()
           << ", alpha-in=" << preconditioner_in_.GetAlpha()
           << ", alpha-out=" << preconditioner_in_.GetAlpha();
  }
  return stream.str();
}


void TimeHeightConvolutionComponent::InitUnit() {
  if (model_.num_filters_in != model_.num_filters_out) {
    KALDI_ERR << "You cannot specify init-unit if the num-filters-in "
              << "and num-filters-out differ.";
  }
  size_t i;
  int32 zero_offset = 0;
  for (i = 0; i < model_.offsets.size(); i++) {
    if (model_.offsets[i].time_offset == 0 &&
        model_.offsets[i].height_offset == 0) {
      zero_offset = i;
      break;
    }
  }
  if (i == model_.offsets.size())  // did not break.
    KALDI_ERR << "You cannot specify init-unit if the model does "
              << "not have the offset (0, 0).";

  CuSubMatrix<BaseFloat> zero_offset_block(
      linear_params_, 0, linear_params_.NumRows(),
      zero_offset * model_.num_filters_in, model_.num_filters_in);

  KALDI_ASSERT(zero_offset_block.NumRows() == zero_offset_block.NumCols());
  zero_offset_block.AddToDiag(1.0);  // set this block to the unit matrix.
}

void TimeHeightConvolutionComponent::InitFromConfig(ConfigLine *cfl) {
  // 1. Config values inherited from UpdatableComponent.
  InitLearningRatesFromConfig(cfl);

  // 2. convolution-related config values.
  model_.height_subsample_out = 1;  // default.
  max_memory_mb_ = 200.0;
  std::string height_offsets, time_offsets, required_time_offsets = "undef";

  bool ok = cfl->GetValue("num-filters-in", &model_.num_filters_in) &&
      cfl->GetValue("num-filters-out", &model_.num_filters_out) &&
      cfl->GetValue("height-in", &model_.height_in) &&
      cfl->GetValue("height-out", &model_.height_out) &&
      cfl->GetValue("height-offsets", &height_offsets) &&
      cfl->GetValue("time-offsets", &time_offsets);
  if (!ok) {
    KALDI_ERR << "Bad initializer: expected all the values "
        "num-filters-in, num-filters-out, height-in, height-out, "
        "height-offsets, time-offsets to be defined: "
              << cfl->WholeLine();
  }
  // some optional structural configs.
  cfl->GetValue("required-time-offsets", &required_time_offsets);
  cfl->GetValue("height-subsample-out", &model_.height_subsample_out);
  cfl->GetValue("max-memory-mb", &max_memory_mb_);
  KALDI_ASSERT(max_memory_mb_ > 0.0);
  {  // This block attempts to parse height_offsets, time_offsets
     // and required_time_offsets.
    std::vector<int32> height_offsets_vec,
        time_offsets_vec, required_time_offsets_vec;
    if (!SplitStringToIntegers(height_offsets, ",", false,
                               &height_offsets_vec) ||
        !SplitStringToIntegers(time_offsets, ",", false,
                               &time_offsets_vec)) {
      KALDI_ERR << "Formatting problem in time-offsets or height-offsets: "
                << cfl->WholeLine();
    }
    if (height_offsets_vec.empty() || !IsSortedAndUniq(height_offsets_vec) ||
        time_offsets_vec.empty() || !IsSortedAndUniq(time_offsets_vec)) {
      KALDI_ERR << "Options time-offsets and height-offsets must be nonempty, "
          "sorted and unique.";
    }
    if (required_time_offsets == "undef") {
      required_time_offsets_vec = time_offsets_vec;
    } else {
      if (!SplitStringToIntegers(required_time_offsets, ",", false,
                                 &required_time_offsets_vec) ||
          required_time_offsets_vec.empty() ||
          !IsSortedAndUniq(required_time_offsets_vec)) {
      KALDI_ERR << "Formatting problem in required-time-offsets: "
                << cfl->WholeLine();
      }
    }
    model_.offsets.clear();
    for (size_t i = 0; i < time_offsets_vec.size(); i++) {
      for (size_t j = 0; j < height_offsets_vec.size(); j++) {
        time_height_convolution::ConvolutionModel::Offset offset;
        offset.time_offset = time_offsets_vec[i];
        offset.height_offset = height_offsets_vec[j];
        model_.offsets.push_back(offset);
      }
    }
    model_.required_time_offsets.clear();
    model_.required_time_offsets.insert(
        required_time_offsets_vec.begin(),
        required_time_offsets_vec.end());
  }

  model_.ComputeDerived();
  if (!model_.Check(false, true)) {
    KALDI_ERR << "Parameters used to initialize TimeHeightConvolutionComponent "
              << "do not make sense,  line was: " << cfl->WholeLine();
  }
  if (!model_.Check(true, true)) {
    KALDI_WARN << "There are input heights unused in "
        "TimeHeightConvolutionComponent; consider increasing output "
        "height or decreasing height of preceding layer."
               << cfl->WholeLine();
  }

  // 3. Parameter-initialization configs.
  BaseFloat param_stddev = -1, bias_stddev = 0.0;
  bool init_unit = false;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-stddev", &bias_stddev);
  cfl->GetValue("init-unit", &init_unit);
  if (param_stddev < 0.0) {
    param_stddev = 1.0 / sqrt(model_.num_filters_in *
                              model_.offsets.size());
  }
  // initialize the parameters.
  linear_params_.Resize(model_.ParamRows(), model_.ParamCols());
  if (!init_unit) {
    linear_params_.SetRandn();
    linear_params_.Scale(param_stddev);
  } else {
    InitUnit();
  }
  bias_params_.Resize(model_.num_filters_out);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);


  // 4. Natural-gradient related configs.
  use_natural_gradient_ = true;
  num_minibatches_history_ = 4.0;
  int32 rank_out = -1, rank_in = -1;
  BaseFloat alpha_out = 4.0, alpha_in = 4.0;
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("alpha-in", &alpha_in);
  cfl->GetValue("alpha-out", &alpha_out);
  cfl->GetValue("num-minibatches-history", &num_minibatches_history_);

  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);
  int32 dim_in = linear_params_.NumCols() + 1,
      dim_out = linear_params_.NumRows();
  if (rank_in < 0) {
    rank_in = std::min<int32>(80, (dim_in + 1) / 2);
    preconditioner_in_.SetRank(rank_in);
  }
  if (rank_out < 0) {
    rank_out = std::min<int32>(80, (dim_out + 1) / 2);
    preconditioner_out_.SetRank(rank_out);
  }
  // the swapping of in and out in the lines below is intentional.  the num-rows
  // of the matrix that we give to preconditioner_in_ to precondition is
  // dim-out, and the num-rows of the matrix we give to preconditioner_out_ to
  // preconditioner is dim-in.  the preconditioner objects treat these rows
  // as separate samples, e.g. separate frames, even though they actually
  // correspond to a different dimension in the parameter space.
  preconditioner_in_.SetNumSamplesHistory(dim_out * num_minibatches_history_);
  preconditioner_out_.SetNumSamplesHistory(dim_in * num_minibatches_history_);

  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);

  ComputeDerived();
}

void* TimeHeightConvolutionComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL);
  { // this block handles the bias term.
    KALDI_ASSERT(out->Stride() == out->NumCols() &&
                 out->NumCols() == model_.height_out * model_.num_filters_out);
    CuSubMatrix<BaseFloat> out_reshaped(
        out->Data(), out->NumRows() * model_.height_out,
        model_.num_filters_out, model_.num_filters_out);
    out_reshaped.CopyRowsFromVec(bias_params_);
  }
  ConvolveForward(indexes->computation, in, linear_params_, out);
  return NULL;
}

void TimeHeightConvolutionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void*, // memo
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL);

  if (in_deriv != NULL) {
    ConvolveBackwardData(indexes->computation, linear_params_,
                         out_deriv, in_deriv);
  }
  if (to_update_in != NULL) {
    TimeHeightConvolutionComponent *to_update =
        dynamic_cast<TimeHeightConvolutionComponent*>(to_update_in);
    KALDI_ASSERT(to_update != NULL);

    if (to_update->learning_rate_ == 0.0)
      return;

    if (to_update->is_gradient_ || !to_update->use_natural_gradient_)
      to_update->UpdateSimple(*indexes, in_value, out_deriv);
    else
      to_update->UpdateNaturalGradient(*indexes, in_value, out_deriv);
  }
}

void TimeHeightConvolutionComponent::UpdateSimple(
    const PrecomputedIndexes &indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  { // this block handles the bias term.
    KALDI_ASSERT(out_deriv.Stride() == out_deriv.NumCols() &&
                 out_deriv.NumCols() ==
                 model_.height_out * model_.num_filters_out);
    CuSubMatrix<BaseFloat> out_deriv_reshaped(
        out_deriv.Data(), out_deriv.NumRows() * model_.height_out,
        model_.num_filters_out, model_.num_filters_out);
    bias_params_.AddRowSumMat(learning_rate_, out_deriv_reshaped);
  }

  ConvolveBackwardParams(indexes.computation, in_value, out_deriv,
                         learning_rate_, &linear_params_);
}


void TimeHeightConvolutionComponent::UpdateNaturalGradient(
    const PrecomputedIndexes &indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  CuVector<BaseFloat> bias_temp(bias_params_.Dim());

  { // this block computes 'bias_temp', the derivative w.r.t. the bias.
    KALDI_ASSERT(out_deriv.Stride() == out_deriv.NumCols() &&
                 out_deriv.NumCols() ==
                 model_.height_out * model_.num_filters_out);
    CuSubMatrix<BaseFloat> out_deriv_reshaped(
        out_deriv.Data(), out_deriv.NumRows() * model_.height_out,
        model_.num_filters_out, model_.num_filters_out);
    bias_temp.AddRowSumMat(1.0, out_deriv_reshaped);
  }

  CuMatrix<BaseFloat> params_temp(linear_params_.NumRows(),
                                  linear_params_.NumCols() + 1);
  params_temp.CopyColFromVec(bias_temp, linear_params_.NumCols());


  CuSubMatrix<BaseFloat> linear_params_temp(
      params_temp, 0, linear_params_.NumRows(),
      0, linear_params_.NumCols());

  ConvolveBackwardParams(indexes.computation, in_value, out_deriv,
                         1.0, &linear_params_temp);

  // the precondition-directions code outputs a scalar that
  // must be multiplied by its output (this saves one
  // CUDA operation internally).
  // We don't bother applying this scale before doing the other
  // dimenson of natural gradient, because although it's not
  // invariant to scalar multiplication of the input if the
  // scalars are different across iterations, the scalars
  // will be pretty similar on different iterations
  BaseFloat scale1, scale2;
  preconditioner_in_.PreconditionDirections(&params_temp, NULL,
                                            &scale1);


  CuMatrix<BaseFloat> params_temp_transpose(params_temp, kTrans);
  preconditioner_out_.PreconditionDirections(&params_temp_transpose,
                                             NULL, &scale2);


  linear_params_.AddMat(
      learning_rate_ * scale1 * scale2,
      params_temp_transpose.RowRange(0, linear_params_.NumCols()),
      kTrans);

  bias_params_.AddVec(learning_rate_ * scale1 * scale2,
                      params_temp_transpose.Row(linear_params_.NumCols()));
}


void TimeHeightConvolutionComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
  using namespace time_height_convolution;
  ConvolutionComputationOptions opts;
  opts.max_memory_mb = max_memory_mb_;
  ConvolutionComputation computation_temp;
  std::vector<Index> input_indexes_modified,
      output_indexes_modified;
  CompileConvolutionComputation(
      model_, *input_indexes, *output_indexes, opts,
      &computation_temp, &input_indexes_modified, &output_indexes_modified);
  input_indexes->swap(input_indexes_modified);
  output_indexes->swap(output_indexes_modified);
}

void TimeHeightConvolutionComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate.
  WriteToken(os, binary, "<Model>");
  model_.Write(os, binary);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<MaxMemoryMb>");
  WriteBasicType(os, binary, max_memory_mb_);
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  WriteToken(os, binary, "<NumMinibatchesHistory>");
  WriteBasicType(os, binary, num_minibatches_history_);
  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank();
  BaseFloat alpha_in = preconditioner_in_.GetAlpha(),
      alpha_out = preconditioner_out_.GetAlpha();
  WriteToken(os, binary, "<AlphaInOut>");
  WriteBasicType(os, binary, alpha_in);
  WriteBasicType(os, binary, alpha_out);
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "</TimeHeightConvolutionComponent>");
}

void TimeHeightConvolutionComponent::Read(std::istream &is, bool binary) {
  std::string token = ReadUpdatableCommon(is, binary);
  // the next few lines are only for back compatibility.
  if (token != "") {
    KALDI_ASSERT(token == "<Model>");
  } else {
    ExpectToken(is, binary, "<Model>");
  }
  model_.Read(is, binary);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<MaxMemoryMb>");
  ReadBasicType(is, binary, &max_memory_mb_);
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);
  ExpectToken(is, binary, "<NumMinibatchesHistory>");
  ReadBasicType(is, binary, &num_minibatches_history_);
  int32 rank_in,  rank_out;
  BaseFloat alpha_in, alpha_out;
  ExpectToken(is, binary, "<AlphaInOut>");
  ReadBasicType(is, binary, &alpha_in);
  ReadBasicType(is, binary, &alpha_out);
  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  int32 dim_in = linear_params_.NumCols() + 1,
      dim_out = linear_params_.NumRows();
  // the following lines mirror similar lines in InitFromConfig().
  // the swapping of in and out is intentional; see comment in InitFromConfig(),
  // by similar lines.
  preconditioner_in_.SetNumSamplesHistory(dim_out * num_minibatches_history_);
  preconditioner_out_.SetNumSamplesHistory(dim_in * num_minibatches_history_);
  ExpectToken(is, binary, "</TimeHeightConvolutionComponent>");
  ComputeDerived();
  Check();
}

void TimeHeightConvolutionComponent::ComputeDerived() {
  all_time_offsets_.clear();
  all_time_offsets_.insert(
      all_time_offsets_.end(),
      model_.all_time_offsets.begin(),
      model_.all_time_offsets.end());
  time_offset_required_.resize(all_time_offsets_.size());
  for (size_t i = 0; i < all_time_offsets_.size(); i++) {
    time_offset_required_[i] =
        (model_.required_time_offsets.count(all_time_offsets_[i]) > 0);
  }
}

void TimeHeightConvolutionComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  size_t size = all_time_offsets_.size();
  desired_indexes->resize(size);
  for (size_t i = 0; i < size; i++) {
    (*desired_indexes)[i].n = output_index.n;
    (*desired_indexes)[i].t = output_index.t + all_time_offsets_[i];
    (*desired_indexes)[i].x = output_index.x;
  }
}


bool TimeHeightConvolutionComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  size_t size = all_time_offsets_.size();
  Index index(output_index);
  if (used_inputs != NULL) {
    used_inputs->clear();
    used_inputs->reserve(size);
    for (size_t i = 0; i < size; i++) {
      index.t = output_index.t + all_time_offsets_[i];
      if (input_index_set(index)) {
        // This input index is available.
        used_inputs->push_back(index);
      } else {
        // This input index is not available.
        if (time_offset_required_[i]) {
          // A required offset was not present -> this output index is not
          // computable.
          used_inputs->clear();
          return false;
        }
      }
    }
    // All required time-offsets of the output were computable. -> return true.
    return true;
  } else {
    for (size_t i = 0; i < size; i++) {
      if (time_offset_required_[i]) {
        index.t = output_index.t + all_time_offsets_[i];
        if (!input_index_set(index))
          return false;
      }
    }
    return true;
  }
}


ComponentPrecomputedIndexes* TimeHeightConvolutionComponent::PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const {
  using namespace time_height_convolution;
  ConvolutionComputationOptions opts;
  opts.max_memory_mb = max_memory_mb_;
  PrecomputedIndexes *ans = new PrecomputedIndexes();
  std::vector<Index> input_indexes_modified,
      output_indexes_modified;
  CompileConvolutionComputation(
      model_, input_indexes, output_indexes, opts,
      &(ans->computation), &input_indexes_modified, &output_indexes_modified);
  if (input_indexes_modified != input_indexes ||
      output_indexes_modified != output_indexes) {
    KALDI_ERR << "Problem precomputing indexes";
  }
  return ans;
}

void TimeHeightConvolutionComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void TimeHeightConvolutionComponent::Add(BaseFloat alpha,
                                         const Component &other_in) {
  const TimeHeightConvolutionComponent *other =
      dynamic_cast<const TimeHeightConvolutionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void TimeHeightConvolutionComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_mat(linear_params_.NumRows(),
                               linear_params_.NumCols(), kUndefined);
  temp_mat.SetRandn();
  linear_params_.AddMat(stddev, temp_mat);
  CuVector<BaseFloat> temp_vec(bias_params_.Dim(), kUndefined);
  temp_vec.SetRandn();
  bias_params_.AddVec(stddev, temp_vec);
}

BaseFloat TimeHeightConvolutionComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const TimeHeightConvolutionComponent *other =
      dynamic_cast<const TimeHeightConvolutionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans) +
      VecVec(bias_params_, other->bias_params_);
}

int32 TimeHeightConvolutionComponent::NumParameters() const {
  return linear_params_.NumRows() * linear_params_.NumCols() +
      bias_params_.Dim();
}

void TimeHeightConvolutionComponent::Vectorize(
    VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == NumParameters());
  int32 linear_size = linear_params_.NumRows() * linear_params_.NumCols(),
      bias_size = bias_params_.Dim();
  params->Range(0, linear_size).CopyRowsFromMat(linear_params_);
  params->Range(linear_size, bias_size).CopyFromVec(bias_params_);
}

void TimeHeightConvolutionComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == NumParameters());
  int32 linear_size = linear_params_.NumRows() * linear_params_.NumCols(),
      bias_size = bias_params_.Dim();
  linear_params_.CopyRowsFromVec(params.Range(0, linear_size));
  bias_params_.CopyFromVec(params.Range(linear_size, bias_size));
}



TimeHeightConvolutionComponent::PrecomputedIndexes*
TimeHeightConvolutionComponent::PrecomputedIndexes::Copy() const {
  return new PrecomputedIndexes(*this);
}

void TimeHeightConvolutionComponent::PrecomputedIndexes::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<TimeHeightConvolutionComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<Computation>");
  computation.Write(os, binary);
  WriteToken(os, binary, "</TimeHeightConvolutionComponentPrecomputedIndexes>");
}

void TimeHeightConvolutionComponent::PrecomputedIndexes::Read(
    std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<TimeHeightConvolutionComponentPrecomputedIndexes>",
                       "<Computation>");
  computation.Read(is, binary);
  ExpectToken(is, binary, "</TimeHeightConvolutionComponentPrecomputedIndexes>");
}




int32 TimeConvolutionComponent::InputDim() const {
  return sub_frames_per_frame_ * samples_per_sub_frame_;
}

int32 TimeConvolutionComponent::OutputDim() const {
  return sub_frames_per_frame_ * num_filters_out_;
}

std::string TimeConvolutionComponent::Info() const {
  std::ostringstream stream;
  // The output of model_.Info() has been designed to be suitable
  // as a component-level info string, it has
  // {num-filters,height}-{in-out}, offsets=[...], required-time-offsets=[...],
  // {input,output}-dim.
  stream << UpdatableComponent::Info()
         << ", sub-frames-per-frame=" << sub_frames_per_frame_
         << ", num-filters-out=" << num_filters_out_
         << ", samples-per-sub-frame=" << samples_per_sub_frame_
         << ", sub-frames-left-context=" << sub_frames_left_context_
         << ", sub-frames-right-context=" << sub_frames_right_context_
         << ", zero-pad=" << (zero_pad_ ? "true" : "false");

  PrintParameterStats(stream, "filter-params", linear_params_);
  PrintParameterStats(stream, "bias-params", bias_params_, true);
  stream << ", num-params=" << NumParameters()
         << ", use-natural-gradient=" << use_natural_gradient_;
  if (use_natural_gradient_) {
    stream << ", num-minibatches-history=" << num_minibatches_history_
           << ", rank-in=" << preconditioner_in_.GetRank()
           << ", rank-out=" << preconditioner_out_.GetRank()
           << ", alpha-in=" << preconditioner_in_.GetAlpha()
           << ", alpha-out=" << preconditioner_in_.GetAlpha();
  }
  return stream.str();
}

void TimeConvolutionComponent::InitFromConfig(ConfigLine *cfl) {
  // 1. Config values inherited from UpdatableComponent.
  InitLearningRatesFromConfig(cfl);

  int32 input_dim = -1;
  zero_pad_ = true;

  bool ok = cfl->GetValue("input-dim", &input_dim) &&
      cfl->GetValue("sub-frames-per-frame", &sub_frames_per_frame_) &&
      cfl->GetValue("num-filters-out", &num_filters_out_) &&
      cfl->GetValue("sub-frames-left-context", &sub_frames_left_context_) &&
      cfl->GetValue("sub-frames-right-context", &sub_frames_right_context_);

  if (!ok) {
    KALDI_ERR << "Bad initializer: expected all the values "
        "input-dim, sub-frames-per-frame, num-filters-out, "
        "sub-frames-left-context and sub-frames-right-context "
        "to be defined: " << cfl->WholeLine();
  }
  // some optional configs.
  cfl->GetValue("zero-pad", &zero_pad_);
  {  // This block sets some parameters of 'model_' based on
    // the user-provided input.
    if (!(sub_frames_per_frame_ > 0 && input_dim > 0 &&
          input_dim % sub_frames_per_frame_ == 0)) {
      KALDI_ERR << "Bad values sub-frames-per-frame=" << sub_frames_per_frame_
                << ", input-dim=" << input_dim
                << ": sub-frames-per-frame must divide input-dim.";
    }
    samples_per_sub_frame_ = input_dim / sub_frames_per_frame_;
    if (sub_frames_left_context_ < 0 ||
        sub_frames_right_context_ < 0) {
      KALDI_ERR << "sub-frames-left-context and sub-frames-right-context "
                << "must be non-negative: " << cfl->WholeLine();
    }
  }

  // 3. Parameter-initialization configs.
  BaseFloat param_stddev = -1, bias_stddev = 0.0;
  bool init_unit = false;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-stddev", &bias_stddev);
  cfl->GetValue("init-unit", &init_unit);
  int32 total_context = sub_frames_left_context_ + 1 + sub_frames_right_context_;
  // initialize the parameters.
  linear_params_.Resize(num_filters_out_,
                        total_context * samples_per_sub_frame_);
  if (param_stddev < 0.0) {
    param_stddev = 1.0 / sqrt(linear_params_.NumCols());
  }

  linear_params_.SetRandn();
  linear_params_.Scale(param_stddev);
  bias_params_.Resize(num_filters_out_);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);


  // 4. Natural-gradient related configs.
  use_natural_gradient_ = true;
  num_minibatches_history_ = 4.0;
  int32 rank_out = -1, rank_in = -1;
  BaseFloat alpha_out = 4.0, alpha_in = 4.0;
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("alpha-in", &alpha_in);
  cfl->GetValue("alpha-out", &alpha_out);
  cfl->GetValue("num-minibatches-history", &num_minibatches_history_);

  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);
  int32 dim_in = linear_params_.NumCols() + 1,
      dim_out = linear_params_.NumRows();
  if (rank_in < 0) {
    rank_in = std::min<int32>(80, (dim_in + 1) / 2);
    preconditioner_in_.SetRank(rank_in);
  }
  if (rank_out < 0) {
    rank_out = std::min<int32>(80, (dim_out + 1) / 2);
    preconditioner_out_.SetRank(rank_out);
  }
  // the swapping of in and out in the lines below is intentional.  the num-rows
  // of the matrix that we give to preconditioner_in_ to precondition is
  // dim-out, and the num-rows of the matrix we give to preconditioner_out_ to
  // preconditioner is dim-in.  the preconditioner objects treat these rows
  // as separate samples, e.g. separate frames, even though they actually
  // correspond to a different dimension in the parameter space.
  preconditioner_in_.SetNumSamplesHistory(dim_out * num_minibatches_history_);
  preconditioner_out_.SetNumSamplesHistory(dim_in * num_minibatches_history_);

  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);

  ComputeDerived();
}

void TimeConvolutionComponent::ComputeDerived() {
  KALDI_ASSERT(sub_frames_per_frame_ > 0);
  int32 sf = sub_frames_per_frame_;
  frames_left_context_ = (sub_frames_left_context_ + sf - 1) / sf;
  frames_right_context_ = (sub_frames_right_context_ + sf - 1) / sf;
}

void TimeConvolutionComponent::Check() const {
  KALDI_ASSERT(sub_frames_per_frame_ > 0 &&
               sub_frames_left_context_ >= 0 &&
               sub_frames_right_context_ >= 0 &&
               samples_per_sub_frame_ > 0 &&
               num_filters_out_ > 0);
  KALDI_ASSERT(linear_params_.NumRows() == num_filters_out_ &&
               bias_params_.Dim() == num_filters_out_ &&
               linear_params_.NumCols() == samples_per_sub_frame_ *
               (1 + sub_frames_left_context_ +
                sub_frames_right_context_));
  int32 sf = sub_frames_per_frame_;
  KALDI_ASSERT(frames_left_context_ == (sub_frames_left_context_ + sf - 1) / sf &&
               frames_right_context_ == (sub_frames_right_context_ + sf - 1) / sf);

}


ComponentPrecomputedIndexes* TimeConvolutionComponent::PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const {
  using namespace time_height_convolution;

  PrecomputedIndexes *ans = new PrecomputedIndexes();
  std::vector<Index> input_indexes_modified,
      output_indexes_modified;
  Check();
  GetComputationIo(input_indexes, output_indexes, &(ans->io));
  PadComputationIoSpecial(frames_left_context_,
                          frames_right_context_,
                          &(ans->io));
  CreateOperations(ans->io, &(ans->operations));
  GetIndexesForComputation(ans->io,
                           input_indexes, output_indexes,
                           &input_indexes_modified,
                           &output_indexes_modified);
  if (input_indexes_modified != input_indexes ||
      output_indexes_modified != output_indexes) {
    KALDI_ERR << "Problem precomputing indexes";
  }
  return ans;
}


void TimeConvolutionComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
  using namespace time_height_convolution;
  ConvolutionComputationIo io;
  GetComputationIo(*input_indexes, *output_indexes, &io);
  PadComputationIoSpecial(frames_left_context_,
                          frames_right_context_,
                          &io);
  std::vector<Index> input_indexes_modified,
      output_indexes_modified;
  GetIndexesForComputation(io,
                           *input_indexes, *output_indexes,
                           &input_indexes_modified,
                           &output_indexes_modified);
  input_indexes->swap(input_indexes_modified);
  output_indexes->swap(output_indexes_modified);
}


void TimeConvolutionComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  desired_indexes->clear();
  desired_indexes->reserve(frames_left_context_ + 1 + frames_right_context_);
  for (int32 offset = -frames_left_context_;
       offset <= frames_right_context_; ++offset) {
    desired_indexes->push_back(output_index);
    desired_indexes->back().t += offset;
  }
}


bool TimeConvolutionComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  KALDI_ASSERT(output_index.t != kNoTime);
  Index index = output_index;
  bool zero_pad = zero_pad_;
  int32 t = output_index.t;
  if (used_inputs) {
    used_inputs->clear();
    used_inputs->reserve(frames_left_context_ + 1 + frames_right_context_);
    for (int32 offset = -frames_left_context_;
         offset <= frames_right_context_; ++offset) {
      index.t = t + offset;
      if (input_index_set(index)) {
        used_inputs->push_back(index);
      } else {
        if (offset == 0 || !zero_pad) {
          used_inputs->clear();
          return false;
        }
      }
    }
    return true;
  } else {
    if (zero_pad) {
      return input_index_set(index);
    } else {
      for (int32 offset = -frames_left_context_;
         offset <= frames_right_context_; ++offset) {
        index.t = t + offset;
        if (!input_index_set(index))
          return false;
      }
      return true;
    }
  }
}


void TimeConvolutionComponent::Scale(BaseFloat scale) {
  if (scale == 0.0) {
    linear_params_.SetZero();
    bias_params_.SetZero();
  } else {
    linear_params_.Scale(scale);
    bias_params_.Scale(scale);
  }
}

void TimeConvolutionComponent::Add(BaseFloat alpha,
                                         const Component &other_in) {
  const TimeConvolutionComponent *other =
      dynamic_cast<const TimeConvolutionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

void TimeConvolutionComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_mat(linear_params_.NumRows(),
                               linear_params_.NumCols(), kUndefined);
  temp_mat.SetRandn();
  linear_params_.AddMat(stddev, temp_mat);
  CuVector<BaseFloat> temp_vec(bias_params_.Dim(), kUndefined);
  temp_vec.SetRandn();
  bias_params_.AddVec(stddev, temp_vec);
}

BaseFloat TimeConvolutionComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const TimeConvolutionComponent *other =
      dynamic_cast<const TimeConvolutionComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans) +
      VecVec(bias_params_, other->bias_params_);
}

int32 TimeConvolutionComponent::NumParameters() const {
  return linear_params_.NumRows() * linear_params_.NumCols() +
      bias_params_.Dim();
}

void TimeConvolutionComponent::Vectorize(
    VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == NumParameters());
  int32 linear_size = linear_params_.NumRows() * linear_params_.NumCols(),
      bias_size = bias_params_.Dim();
  params->Range(0, linear_size).CopyRowsFromMat(linear_params_);
  params->Range(linear_size, bias_size).CopyFromVec(bias_params_);
}

void TimeConvolutionComponent::UnVectorize(
    const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == NumParameters());
  int32 linear_size = linear_params_.NumRows() * linear_params_.NumCols(),
      bias_size = bias_params_.Dim();
  linear_params_.CopyRowsFromVec(params.Range(0, linear_size));
  bias_params_.CopyFromVec(params.Range(linear_size, bias_size));
}



TimeConvolutionComponent::PrecomputedIndexes*
TimeConvolutionComponent::PrecomputedIndexes::Copy() const {
  return new PrecomputedIndexes(*this);
}

void TimeConvolutionComponent::PrecomputedIndexes::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<TimeConvolutionComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<Io>");
  io.Write(os, binary);
  int32 size = operations.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    operations[i].Write(os, binary);
  WriteToken(os, binary, "</TimeConvolutionComponentPrecomputedIndexes>");
}

void TimeConvolutionComponent::PrecomputedIndexes::Read(
    std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<TimeConvolutionComponentPrecomputedIndexes>",
                       "<Io>");
  io.Read(is, binary);
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size >= 0 && size < 100000);
  operations.resize(size);
  for (int32 i = 0; i < size; i++)
    operations[i].Read(is, binary);
  ExpectToken(is, binary, "</TimeConvolutionComponentPrecomputedIndexes>");
}

void TimeConvolutionComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate.
  WriteToken(os, binary, "<SubFramesPerFrame>");
  WriteBasicType(os, binary, sub_frames_per_frame_);
  WriteToken(os, binary, "<NumFiltersOut>");
  WriteBasicType(os, binary, num_filters_out_);
  WriteToken(os, binary, "<SamplesPerSubFrame>");
  WriteBasicType(os, binary, samples_per_sub_frame_);
  WriteToken(os, binary, "<SubFramesLeftRightContext>");
  WriteBasicType(os, binary, sub_frames_left_context_);
  WriteBasicType(os, binary, sub_frames_right_context_);
  WriteToken(os, binary, "<ZeroPad>");
  WriteBasicType(os, binary, zero_pad_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);
  WriteToken(os, binary, "<NumMinibatchesHistory>");
  WriteBasicType(os, binary, num_minibatches_history_);
  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank();
  BaseFloat alpha_in = preconditioner_in_.GetAlpha(),
      alpha_out = preconditioner_out_.GetAlpha();
  WriteToken(os, binary, "<AlphaInOut>");
  WriteBasicType(os, binary, alpha_in);
  WriteBasicType(os, binary, alpha_out);
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "</TimeConvolutionComponent>");
}

void TimeConvolutionComponent::Read(std::istream &is, bool binary) {
  std::string token = ReadUpdatableCommon(is, binary);
  KALDI_ASSERT(token == "");
  ExpectToken(is, binary, "<SubFramesPerFrame>");
  ReadBasicType(is, binary, &sub_frames_per_frame_);
  ExpectToken(is, binary, "<NumFiltersOut>");
  ReadBasicType(is, binary, &num_filters_out_);
  ExpectToken(is, binary, "<SamplesPerSubFrame>");
  ReadBasicType(is, binary, &samples_per_sub_frame_);
  ExpectToken(is, binary, "<SubFramesLeftRightContext>");
  ReadBasicType(is, binary, &sub_frames_left_context_);
  ReadBasicType(is, binary, &sub_frames_right_context_);
  ExpectToken(is, binary, "<ZeroPad>");
  ReadBasicType(is, binary, &zero_pad_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);
  ExpectToken(is, binary, "<NumMinibatchesHistory>");
  ReadBasicType(is, binary, &num_minibatches_history_);
  int32 rank_in,  rank_out;
  BaseFloat alpha_in, alpha_out;
  ExpectToken(is, binary, "<AlphaInOut>");
  ReadBasicType(is, binary, &alpha_in);
  ReadBasicType(is, binary, &alpha_out);
  preconditioner_in_.SetAlpha(alpha_in);
  preconditioner_out_.SetAlpha(alpha_out);
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  int32 dim_in = linear_params_.NumCols() + 1,
      dim_out = linear_params_.NumRows();
  // the following lines mirror similar lines in InitFromConfig().
  // the swapping of in and out is intentional; see comment in InitFromConfig(),
  // by similar lines.
  preconditioner_in_.SetNumSamplesHistory(dim_out * num_minibatches_history_);
  preconditioner_out_.SetNumSamplesHistory(dim_in * num_minibatches_history_);
  ExpectToken(is, binary, "</TimeConvolutionComponent>");
  ComputeDerived();
}


/*
   This is the core of how TimeConvolutionComponent works.

   For purposes of exposition, assume we have a grid on 'n' and 't', with all
   the 'x' values zero.  The framework is a little more general than that, but
   this makes it easier to explain.

   The input of the computation has rows corresponding to (n,t), with
   the 't' values having higher stride.  It's the same situation on
   the output.  As for the columns of the input and output, they
   have sub-frames with the highest stride, then (samples, filters)
   respectively on the input and output.

   Each operation is on a row-block of the input that has the same
   num-rows as the output.  The num-cols of an operation on the input
   side is variable, it depends how many sub-frames we need from
   a particular frame-shift (could be all the sub-frames, or just
   some of them, depending if we're at an edge or not).
 */
void TimeConvolutionComponent::CreateOperations(
    const time_height_convolution::ConvolutionComputationIo &io,
    std::vector<Operation> *operations) const {
  operations->clear();
  int32 last_t_in = io.start_t_in + io.num_t_in - 1,
      last_t_out = io.start_t_out + io.num_t_out - 1;
  KALDI_ASSERT(io.t_step_in == 1 && io.t_step_out == 1 &&
               io.start_t_in == io.start_t_out - frames_left_context_ &&
               last_t_in == last_t_out + frames_right_context_);
  for (int32 sub_frame = 0; sub_frame < sub_frames_per_frame_; ++sub_frame) {
    // 'frame_shift' determines how we shift the input relative to the
    // output, e.g. if frame_shift == -1 then we're talking about what
    // the output at time t sees from the input at time t-1.
    // Note: shifting the frame by frame_shift means shifting the sub-frame by
    // frame_shift * sub_frames_per_frame_.
    for (int32 frame_shift = -frames_left_context_;
         frame_shift <= frames_right_context_;
         frame_shift++) {
      // The variable 'first_sub_frame_offset' deserves some explanation.
      // It means: if we take the first sub-frame available on this
      // frame shift, what is its offset relative to sub-frame 'sub_frame'?
      // (the offset is measured, of course, in sub-frames).
      int32 first_sub_frame_offset =
          frame_shift * sub_frames_per_frame_ - sub_frame,
          end_sub_frame_offset =
          first_sub_frame_offset + sub_frames_per_frame_;

      // these variables will represent the part of the available range of
      // offsets, that we actually use.  (the 'end' is the last one in the
      // range, plus one).
      int32 first_used_sub_frame_offset =
          std::max(first_sub_frame_offset,
                   -sub_frames_left_context_),
          end_used_sub_frame_offset =
          std::min(end_sub_frame_offset,
                   sub_frames_right_context_ + 1);

      // if the used range is empty, we generate no operation.
      if (end_used_sub_frame_offset <= first_used_sub_frame_offset)
        continue;

      Operation operation;
      operation.output_start_col = sub_frame * num_filters_out_;
      operation.output_num_cols = num_filters_out_;
      operation.input_start_row =
          (frame_shift - (-frames_left_context_)) * io.num_images;
      operation.input_start_col =
          (first_used_sub_frame_offset - first_sub_frame_offset) *
          samples_per_sub_frame_;
      operation.input_num_cols =
          (end_used_sub_frame_offset - first_used_sub_frame_offset) *
          samples_per_sub_frame_;
      operation.params_start_col =
          (first_used_sub_frame_offset - (-sub_frames_left_context_)) *
          samples_per_sub_frame_;
      operations->push_back(operation);
    }
  }
}

void TimeConvolutionComponent::Operation::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Operation>");
  WriteBasicType(os, binary, output_start_col);
  WriteBasicType(os, binary, output_num_cols);
  WriteBasicType(os, binary, input_start_row);
  WriteBasicType(os, binary, input_start_col);
  WriteBasicType(os, binary, input_num_cols);
  WriteBasicType(os, binary, params_start_col);
  WriteToken(os, binary, "</Operation>");
}

void TimeConvolutionComponent::Operation::Read(
    std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Operation>");
  ReadBasicType(is, binary, &output_start_col);
  ReadBasicType(is, binary, &output_num_cols);
  ReadBasicType(is, binary, &input_start_row);
  ReadBasicType(is, binary, &input_start_col);
  ReadBasicType(is, binary, &input_num_cols);
  ReadBasicType(is, binary, &params_start_col);
  ExpectToken(is, binary, "</Operation>");
}


void* TimeConvolutionComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL);

  const time_height_convolution::ConvolutionComputationIo &io = indexes->io;
  KALDI_ASSERT(in.NumRows() == io.num_images * io.num_t_in &&
               in.NumCols() == InputDim() &&
               out->NumRows() == io.num_images * io.num_t_out &&
               out->NumCols() == OutputDim() &&
               out->NumCols() == out->Stride());


  { // this block handles the bias term.
    CuSubMatrix<BaseFloat> out_reshaped(
        out->Data(), out->NumRows() * sub_frames_per_frame_,
        num_filters_out_, num_filters_out_);
    out_reshaped.CopyRowsFromVec(bias_params_);
  }

  const std::vector<Operation>  &operations = indexes->operations;

  for (size_t i = 0, size = operations.size(); i < size; i++) {
    const Operation &op = operations[i];
    CuSubMatrix<BaseFloat> out_part(*out, 0, out->NumRows(),
                                    op.output_start_col, op.output_num_cols);
    CuSubMatrix<BaseFloat> in_part(in,
                                   op.input_start_row, out->NumRows(),
                                   op.input_start_col, op.input_num_cols);
    CuSubMatrix<BaseFloat> params_part(linear_params_,
                                       0, linear_params_.NumRows(),
                                       op.params_start_col, op.input_num_cols);
    out_part.AddMatMat(1.0, in_part, kNoTrans, params_part, kTrans, 1.0);
  }
  return NULL;
}


void TimeConvolutionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &, // out_value
    const CuMatrixBase<BaseFloat> &out_deriv,
    void*, // memo
    Component *to_update_in,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  const PrecomputedIndexes *indexes =
      dynamic_cast<const PrecomputedIndexes*>(indexes_in);
  KALDI_ASSERT(indexes != NULL);

  const time_height_convolution::ConvolutionComputationIo &io = indexes->io;
  const std::vector<Operation>  &operations = indexes->operations;
  KALDI_ASSERT(in_value.NumRows() == io.num_images * io.num_t_in &&
               in_value.NumCols() == InputDim() &&
               out_deriv.NumRows() == io.num_images * io.num_t_out &&
               out_deriv.NumCols() == OutputDim() &&
               out_deriv.NumCols() == out_deriv.Stride());
  int32 out_num_rows = out_deriv.NumRows();
  if (in_deriv != NULL) {
    // backprop to in_deriv.
    for (size_t i = 0, size = operations.size(); i < size; i++) {
      const Operation &op = operations[i];
      CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_num_rows,
                                            op.output_start_col,
                                            op.output_num_cols);
      CuSubMatrix<BaseFloat> in_deriv_part(*in_deriv,
                                           op.input_start_row, out_num_rows,
                                           op.input_start_col,
                                           op.input_num_cols);
      CuSubMatrix<BaseFloat> params_part(linear_params_,
                                         0, linear_params_.NumRows(),
                                         op.params_start_col,
                                         op.input_num_cols);
      in_deriv_part.AddMatMat(1.0, out_deriv_part, kNoTrans,
                              params_part, kNoTrans, 1.0);
    }
  }


  if (to_update_in != NULL) {
    TimeConvolutionComponent *to_update =
        dynamic_cast<TimeConvolutionComponent*>(to_update_in);
    KALDI_ASSERT(to_update != NULL);

    if (to_update->learning_rate_ == 0.0)
      return;
    if (to_update->is_gradient_ || !to_update->use_natural_gradient_)
      to_update->UpdateSimple(*indexes, in_value, out_deriv);
    else
      to_update->UpdateNaturalGradient(*indexes, in_value, out_deriv);
  }
}


void TimeConvolutionComponent::UpdateSimple(
    const PrecomputedIndexes &indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {
  int32 out_num_rows = out_deriv.NumRows();
  { // this block handles the bias term.
    KALDI_ASSERT(out_deriv.Stride() == out_deriv.NumCols() &&
                 out_deriv.NumCols() == OutputDim());
    CuSubMatrix<BaseFloat> out_deriv_reshaped(
        out_deriv.Data(),
        out_num_rows * sub_frames_per_frame_,
        num_filters_out_, num_filters_out_);
    bias_params_.AddRowSumMat(learning_rate_, out_deriv_reshaped);
  }

  const std::vector<Operation>  &operations = indexes.operations;
  for (size_t i = 0, size = operations.size(); i < size; i++) {
    const Operation &op = operations[i];
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_num_rows,
                                          op.output_start_col,
                                          op.output_num_cols);
    CuSubMatrix<BaseFloat> in_value_part(in_value,
                                         op.input_start_row, out_num_rows,
                                         op.input_start_col, op.input_num_cols);
    CuSubMatrix<BaseFloat> params_part(linear_params_,
                                       0, linear_params_.NumRows(),
                                       op.params_start_col, op.input_num_cols);
    params_part.AddMatMat(learning_rate_, out_deriv_part, kTrans,
                          in_value_part, kNoTrans, 1.0);
  }
}


void TimeConvolutionComponent::UpdateNaturalGradient(
    const PrecomputedIndexes &indexes,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_deriv) {

  int32 out_num_rows = out_deriv.NumRows();
  CuVector<BaseFloat> bias_temp(bias_params_.Dim());

  { // this block computes 'bias_temp', the derivative w.r.t. the bias.
    KALDI_ASSERT(out_deriv.Stride() == out_deriv.NumCols() &&
                 out_deriv.NumCols() == OutputDim());
    CuSubMatrix<BaseFloat> out_deriv_reshaped(
        out_deriv.Data(), out_num_rows * sub_frames_per_frame_,
        num_filters_out_, num_filters_out_);
    bias_temp.AddRowSumMat(1.0, out_deriv_reshaped);
  }

  CuMatrix<BaseFloat> params_temp(linear_params_.NumRows(),
                                  linear_params_.NumCols() + 1);
  params_temp.CopyColFromVec(bias_temp, linear_params_.NumCols());
  CuSubMatrix<BaseFloat> linear_params_temp(
      params_temp, 0, linear_params_.NumRows(),
      0, linear_params_.NumCols());

  const std::vector<Operation>  &operations = indexes.operations;
  for (size_t i = 0, size = operations.size(); i < size; i++) {
    const Operation &op = operations[i];
    CuSubMatrix<BaseFloat> out_deriv_part(out_deriv, 0, out_num_rows,
                                          op.output_start_col,
                                          op.output_num_cols);
    CuSubMatrix<BaseFloat> in_value_part(in_value,
                                         op.input_start_row, out_num_rows,
                                         op.input_start_col, op.input_num_cols);
    CuSubMatrix<BaseFloat> params_part(linear_params_temp,
                                       0, linear_params_.NumRows(),
                                       op.params_start_col, op.input_num_cols);
    params_part.AddMatMat(1.0, out_deriv_part, kTrans,
                          in_value_part, kNoTrans, 1.0);
  }

  // the precondition-directions code outputs a scalar that
  // must be multiplied by its output (this saves one
  // CUDA operation internally).
  // We don't bother applying this scale before doing the other
  // dimenson of natural gradient, because although it's not
  // invariant to scalar multiplication of the input if the
  // scalars are different across iterations, the scalars
  // will be pretty similar on different iterations
  BaseFloat scale1, scale2;
  preconditioner_in_.PreconditionDirections(&params_temp, NULL,
                                            &scale1);


  CuMatrix<BaseFloat> params_temp_transpose(params_temp, kTrans);
  preconditioner_out_.PreconditionDirections(&params_temp_transpose,
                                             NULL, &scale2);

  linear_params_.AddMat(
      learning_rate_ * scale1 * scale2,
      params_temp_transpose.RowRange(0, linear_params_.NumCols()),
      kTrans);

  bias_params_.AddVec(learning_rate_ * scale1 * scale2,
                      params_temp_transpose.Row(linear_params_.NumCols()));
}




} // namespace nnet3
} // namespace kaldi

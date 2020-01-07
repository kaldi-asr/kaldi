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
    use_natural_gradient_(true) { }

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
  stream << UpdatableComponent::Info() << ' ' << model_.Info();
  PrintParameterStats(stream, "filter-params", linear_params_);
  PrintParameterStats(stream, "bias-params", bias_params_, true);
  stream << ", num-params=" << NumParameters()
         << ", max-memory-mb=" << max_memory_mb_
         << ", use-natural-gradient=" << use_natural_gradient_;
  if (use_natural_gradient_) {
    stream << ", num-minibatches-history="
           << preconditioner_in_.GetNumMinibatchesHistory()
           << ", rank-in=" << preconditioner_in_.GetRank()
           << ", rank-out=" << preconditioner_out_.GetRank()
           << ", alpha=" << preconditioner_in_.GetAlpha();
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
  std::string height_offsets, time_offsets, required_time_offsets = "undef",
      offsets;

  bool ok = cfl->GetValue("num-filters-in", &model_.num_filters_in) &&
      cfl->GetValue("num-filters-out", &model_.num_filters_out) &&
      cfl->GetValue("height-in", &model_.height_in) &&
      cfl->GetValue("height-out", &model_.height_out);
  if (!ok) {
    KALDI_ERR << "Bad initializer: expected all the values "
        "num-filters-in, num-filters-out, height-in, height-out, "
        "to be defined: "
              << cfl->WholeLine();
  }
  // some optional structural configs.
  cfl->GetValue("required-time-offsets", &required_time_offsets);
  cfl->GetValue("height-subsample-out", &model_.height_subsample_out);
  cfl->GetValue("max-memory-mb", &max_memory_mb_);
  KALDI_ASSERT(max_memory_mb_ > 0.0);

  { // This block sets up model_.offsets.
    model_.offsets.clear();
    if (cfl->GetValue("offsets", &offsets)) {
      // init from offsets, like "-1,-1;-1,0;-1,1;0,-1;...;1,1"
      std::vector<std::string> splits;
      SplitStringToVector(offsets, ";", false, &splits);
      for (size_t i = 0; i < splits.size(); i++) {
        std::vector<int32> int_pair;
        if (!SplitStringToIntegers(splits[i], ",", false, &int_pair) ||
            int_pair.size() != 2)
          KALDI_ERR << "Bad config value offsets=" << offsets;
        time_height_convolution::ConvolutionModel::Offset offset;
        offset.time_offset = int_pair[0];
        offset.height_offset = int_pair[1];
        model_.offsets.push_back(offset);
      }
      std::sort(model_.offsets.begin(), model_.offsets.end());
      if (!IsSortedAndUniq(model_.offsets) || model_.offsets.empty())
        KALDI_ERR << "Error in offsets: probably repeated offset.  "
            "offsets=" << offsets;
    } else if (cfl->GetValue("height-offsets", &height_offsets) &&
               cfl->GetValue("time-offsets", &time_offsets)) {
      std::vector<int32> height_offsets_vec,
          time_offsets_vec;
      if (!SplitStringToIntegers(height_offsets, ",", false,
                                 &height_offsets_vec) ||
          !SplitStringToIntegers(time_offsets, ",", false,
                                 &time_offsets_vec)) {
        KALDI_ERR << "Formatting problem in time-offsets or height-offsets: "
                  << cfl->WholeLine();
      }
      if (height_offsets_vec.empty() || !IsSortedAndUniq(height_offsets_vec) ||
          time_offsets_vec.empty() || !IsSortedAndUniq(time_offsets_vec)) {
        KALDI_ERR << "time-offsets and height-offsets must be nonempty, "
            "sorted and unique.";
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
    } else {
      KALDI_ERR << "Expected either 'offsets', or both 'height-offsets' and "
          "'time-offsets', to be defined: " << cfl->WholeLine();
    }
  }

  if (model_.offsets.empty())
    KALDI_ERR << "Something went wrong setting offsets: " << cfl->WholeLine();


  {  // This block sets model_.required_time_offsets.
    std::vector<int32> required_time_offsets_vec;
    if (required_time_offsets == "undef") {
      // it defaults to all the time offsets that were used.
      std::set<int32> required_time_offsets;
      for (size_t i = 0; i < model_.offsets.size(); i++)
        required_time_offsets_vec.push_back(model_.offsets[i].time_offset);
      SortAndUniq(&required_time_offsets_vec);
    } else {
      if (!SplitStringToIntegers(required_time_offsets, ",", false,
                                 &required_time_offsets_vec) ||
          required_time_offsets_vec.empty() ||
          !IsSortedAndUniq(required_time_offsets_vec)) {
        KALDI_ERR << "Formatting problem in required-time-offsets: "
                << cfl->WholeLine();
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
  int32 rank_out = -1, rank_in = -1;
  BaseFloat alpha_out = 4.0, alpha_in = 4.0,
      num_minibatches_history = 4.0;
  cfl->GetValue("use-natural-gradient", &use_natural_gradient_);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("alpha-in", &alpha_in);
  cfl->GetValue("alpha-out", &alpha_out);
  cfl->GetValue("num-minibatches-history", &num_minibatches_history);

  int32 dim_in = linear_params_.NumCols() + 1,
      dim_out = linear_params_.NumRows();
  if (rank_in < 0)
    rank_in = std::min<int32>(80, (dim_in + 1) / 2);
  preconditioner_in_.SetRank(rank_in);
  if (rank_out < 0)
    rank_out = std::min<int32>(80, (dim_out + 1) / 2);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumMinibatchesHistory(num_minibatches_history);
  preconditioner_out_.SetNumMinibatchesHistory(num_minibatches_history);

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

  CuVector<BaseFloat> bias_deriv(bias_params_.Dim());

  { // this block computes 'bias_deriv', the derivative w.r.t. the bias.
    KALDI_ASSERT(out_deriv.Stride() == out_deriv.NumCols() &&
                 out_deriv.NumCols() ==
                 model_.height_out * model_.num_filters_out);
    CuSubMatrix<BaseFloat> out_deriv_reshaped(
        out_deriv.Data(), out_deriv.NumRows() * model_.height_out,
        model_.num_filters_out, model_.num_filters_out);
    bias_deriv.AddRowSumMat(1.0, out_deriv_reshaped);
  }

  CuMatrix<BaseFloat> params_deriv(linear_params_.NumRows(),
                                  linear_params_.NumCols() + 1);
  params_deriv.CopyColFromVec(bias_deriv, linear_params_.NumCols());


  CuSubMatrix<BaseFloat> linear_params_deriv(
      params_deriv, 0, linear_params_.NumRows(),
      0, linear_params_.NumCols());

  ConvolveBackwardParams(indexes.computation, in_value, out_deriv,
                         1.0, &linear_params_deriv);

  // the precondition-directions code outputs a scalar that
  // must be multiplied by its output (this saves one
  // CUDA operation internally).
  // We don't bother applying this scale before doing the other
  // dimenson of natural gradient, because although it's not
  // invariant to scalar multiplication of the input if the
  // scalars are different across iterations, the scalars
  // will be pretty similar on different iterations
  BaseFloat scale1, scale2;
  preconditioner_in_.PreconditionDirections(&params_deriv, &scale1);


  CuMatrix<BaseFloat> params_deriv_transpose(params_deriv, kTrans);
  preconditioner_out_.PreconditionDirections(&params_deriv_transpose, &scale2);

  linear_params_.AddMat(
      learning_rate_ * scale1 * scale2,
      params_deriv_transpose.RowRange(0, linear_params_.NumCols()),
      kTrans);

  bias_params_.AddVec(learning_rate_ * scale1 * scale2,
                      params_deriv_transpose.Row(linear_params_.NumCols()));
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
  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank();
  BaseFloat alpha_in = preconditioner_in_.GetAlpha(),
      alpha_out = preconditioner_out_.GetAlpha(),
      num_minibatches_history = preconditioner_in_.GetNumMinibatchesHistory();
  WriteToken(os, binary, "<NumMinibatchesHistory>");
  WriteBasicType(os, binary, num_minibatches_history);
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
  int32 rank_in,  rank_out;
  BaseFloat alpha_in, alpha_out,
      num_minibatches_history;
  ExpectToken(is, binary, "<NumMinibatchesHistory>");
  ReadBasicType(is, binary, &num_minibatches_history);
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
  preconditioner_in_.SetNumMinibatchesHistory(num_minibatches_history);
  preconditioner_out_.SetNumMinibatchesHistory(num_minibatches_history);
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

void TimeHeightConvolutionComponent::FreezeNaturalGradient(bool freeze) {
  preconditioner_in_.Freeze(freeze);
  preconditioner_out_.Freeze(freeze);
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

void TimeHeightConvolutionComponent::ConsolidateMemory() {
  OnlineNaturalGradient temp_in(preconditioner_in_);
  preconditioner_in_.Swap(&temp_in);
  OnlineNaturalGradient temp_out(preconditioner_out_);
  preconditioner_out_.Swap(&temp_out);
}

} // namespace nnet3
} // namespace kaldi

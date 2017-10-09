// nnet3/nnet-combine.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-combine.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetCombiner::NnetCombiner(const NnetCombineConfig &config,
                           int32 num_nnets,
                           const std::vector<NnetExample> &egs,
                           const Nnet &first_nnet):
    config_(config),
    egs_(egs),
    nnet_(first_nnet),
    num_real_input_nnets_(num_nnets),
    nnet_params_(std::min(num_nnets, config_.max_effective_inputs),
                 NumParameters(first_nnet)),
    tot_input_weighting_(nnet_params_.NumRows()) {

  if (config_.sum_to_one_penalty != 0.0 &&
      config_.enforce_sum_to_one) {
    KALDI_WARN << "--sum-to-one-penalty=" << config_.sum_to_one_penalty
              << " is nonzero, so setting --enforce-sum-to-one=false.";
    config_.enforce_sum_to_one = false;
  }
  SubVector<BaseFloat> first_params(nnet_params_, 0);
  VectorizeNnet(nnet_, &first_params);
  tot_input_weighting_(0) += 1.0;
  num_nnets_provided_ = 1;
  ComputeUpdatableComponentDims();
  NnetComputeProbOptions compute_prob_opts;
  compute_prob_opts.compute_deriv = true;
  prob_computer_ = new NnetComputeProb(compute_prob_opts, nnet_);
}

void NnetCombiner::ComputeUpdatableComponentDims(){
  updatable_component_dims_.clear();
  for (int32 c = 0; c < nnet_.NumComponents(); c++) {
    Component *comp = nnet_.GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      // For now all updatable components inherit from class UpdatableComponent.
      // If that changes in future, we will change this code.
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
            "UpdatableComponent; change this code.";
      updatable_component_dims_.push_back(uc->NumParameters());
    }
  }
}

void NnetCombiner::AcceptNnet(const Nnet &nnet) {
  KALDI_ASSERT(num_nnets_provided_ < num_real_input_nnets_ &&
               "You called AcceptNnet too many times.");
  int32 num_effective_nnets = nnet_params_.NumRows();
  if (num_effective_nnets == num_real_input_nnets_) {
    SubVector<BaseFloat> this_params(nnet_params_, num_nnets_provided_);
    VectorizeNnet(nnet, &this_params);
    tot_input_weighting_(num_nnets_provided_) += 1.0;
  } else {
    // this_index is a kind of warped index, mapping the range
    // 0 ... num_real_inputs_nnets_ - 1 onto the range
    // 0 ... num_effective_nnets - 1.  View the index as falling in
    // between two integer indexes and determining weighting factors.
    // we could view this as triangular bins.
    BaseFloat this_index = num_nnets_provided_ * (num_effective_nnets - 1)
        / static_cast<BaseFloat>(num_real_input_nnets_ - 1);
    int32 lower_index = std::floor(this_index),
        upper_index = lower_index + 1;
    BaseFloat remaining_part = this_index - lower_index,
        lower_weight = 1.0 - remaining_part,
        upper_weight = remaining_part;
    KALDI_ASSERT(lower_index >= 0 && upper_index <= num_effective_nnets &&
                 lower_weight >= 0.0 && upper_weight >= 0.0 &&
                 lower_weight <= 1.0 && upper_weight <= 1.0);
    Vector<BaseFloat> vec(nnet_params_.NumCols(), kUndefined);
    VectorizeNnet(nnet, &vec);
    nnet_params_.Row(lower_index).AddVec(lower_weight, vec);
    tot_input_weighting_(lower_index) += lower_weight;
    if (upper_index == num_effective_nnets) {
      KALDI_ASSERT(upper_weight < 0.1);
    } else {
      nnet_params_.Row(upper_index).AddVec(upper_weight, vec);
      tot_input_weighting_(upper_index) += upper_weight;
    }
  }
  num_nnets_provided_++;
}

void NnetCombiner::FinishPreprocessingInput() {
  KALDI_ASSERT(num_nnets_provided_ == num_real_input_nnets_ &&
               "You did not call AcceptInput() enough times.");
  int32 num_effective_nnets = nnet_params_.NumRows();
  for (int32 i = 0; i < num_effective_nnets; i++) {
    BaseFloat tot_weight = tot_input_weighting_(i);
    KALDI_ASSERT(tot_weight > 0.0);  // Or would be a coding error.
    // Rescale so this row is like a weighted average instead of
    // a weighted sum.
    if (tot_weight != 1.0)
      nnet_params_.Row(i).Scale(1.0 / tot_weight);
  }
}

void NnetCombiner::Combine() {
  FinishPreprocessingInput();

  if (!SelfTestDerivatives()) {
    KALDI_LOG << "Self-testing model derivatives since parameter-derivatives "
        "self-test failed.";
    SelfTestModelDerivatives();
  }

  int32 dim = ParameterDim();
  LbfgsOptions lbfgs_options;
  lbfgs_options.minimize = false; // We're maximizing.
  lbfgs_options.m = dim; // Store the same number of vectors as the dimension
                         // itself, so this is BFGS.
  lbfgs_options.first_step_impr = config_.initial_impr;

  Vector<double> params(dim), deriv(dim);
  double objf, initial_objf;
  GetInitialParameters(&params);


  OptimizeLbfgs<double> lbfgs(params, lbfgs_options);

  for (int32 i = 0; i < config_.num_iters; i++) {
    params.CopyFromVec(lbfgs.GetProposedValue());
    objf = ComputeObjfAndDerivFromParameters(params, &deriv);
    KALDI_VLOG(2) << "Iteration " << i << " params = " << params
                  << ", objf = " << objf << ", deriv = " << deriv;
    if (i == 0) initial_objf = objf;
    lbfgs.DoStep(objf, deriv);
  }

  if (!config_.sum_to_one_penalty) {
    KALDI_LOG << "Combining nnets, objective function changed from "
              << initial_objf << " to " << objf;
  } else {
    Vector<double> weights(WeightDim());
    GetWeights(params, &weights);
    bool print_weights = true;
    double penalty = GetSumToOnePenalty(weights, NULL, print_weights);
    // note: initial_objf has no penalty term because it summed exactly
    // to one.
    KALDI_LOG << "Combining nnets, objective function changed from "
              << initial_objf << " to " << objf << " = "
              << (objf - penalty) << " + " << penalty;
  }


  // must recompute nnet_ if "params" is not exactly equal to the
  // final params that LB
  Vector<double> final_params(dim);
  final_params.CopyFromVec(lbfgs.GetValue(&objf));
  if (!params.ApproxEqual(final_params, 0.0)) {
    // the following call makes sure that nnet_ corresponds to the parameters
    // in "params".
    ComputeObjfAndDerivFromParameters(final_params, &deriv);
  }
  PrintParams(final_params);

}

void NnetCombiner::PrintParams(const VectorBase<double> &params) const {
  Vector<double> weights(WeightDim()), normalized_weights(WeightDim());
  GetWeights(params, &weights);
  GetNormalizedWeights(weights, &normalized_weights);
  int32 num_models = nnet_params_.NumRows(),
      num_uc = NumUpdatableComponents();

  if (config_.separate_weights_per_component) {
    std::vector<std::string> updatable_component_names;
    for (int32 c = 0; c < nnet_.NumComponents(); c++) {
      const Component *comp = nnet_.GetComponent(c);
      if (comp->Properties() & kUpdatableComponent)
        updatable_component_names.push_back(nnet_.GetComponentName(c));
    }
    KALDI_ASSERT(static_cast<int32>(updatable_component_names.size()) ==
                 NumUpdatableComponents());
    for (int32 uc = 0; uc < num_uc; uc++) {
      std::ostringstream os;
      os.width(20);
      os << std::left << updatable_component_names[uc] << ": ";
      os.width(9);
      os.precision(4);
      for (int32 m = 0; m < num_models; m++) {
        int32 index = m * num_uc + uc;
        os << " " << std::left << normalized_weights(index);
      }
      KALDI_LOG << "Weights for " << os.str();
    }
  } else {
    int32 c = 0;  // arbitrarily chosen; they'll all be the same.
    std::ostringstream os;
    os.width(9);
    os.precision(4);
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      os << " " <<  std::left << normalized_weights(index);
    }
    KALDI_LOG << "Model weights are: " << os.str();
  }
  int32 num_effective_nnets = nnet_params_.NumRows();
  if (num_effective_nnets != num_real_input_nnets_)
    KALDI_LOG << "Above, only " << num_effective_nnets << " weights were "
              "printed due to the the --max-effective-inputs option; "
              "there were " << num_real_input_nnets_ << " actual input nnets. "
              "Each weight corresponds to a weighted average over a range of "
              "nnets in the sequence (with triangular bins)";
}

bool NnetCombiner::SelfTestDerivatives() {
  int32 num_tests = 2;  // more properly, this is the number of dimensions in a
                        // single test.
  double delta = 0.001;
  int32 dim = ParameterDim();

  Vector<double> params(dim), deriv(dim);
  Vector<double> predicted_changes(num_tests),
      observed_changes(num_tests);

  GetInitialParameters(&params);
  double initial_objf = ComputeObjfAndDerivFromParameters(params,
                                                             &deriv);
  for (int32 i = 0; i < num_tests; i++) {
    Vector<double> new_deriv(dim), offset(dim), new_params(params);
    offset.SetRandn();
    new_params.AddVec(delta, offset);
    double new_objf = ComputeObjfAndDerivFromParameters(new_params,
                                                           &new_deriv);
    // for predicted changes, interpolate old and new derivs.
    predicted_changes(i) =
        0.5 * VecVec(new_params, deriv) -  0.5 * VecVec(params, deriv) +
        0.5 * VecVec(new_params, new_deriv) - 0.5 * VecVec(params, new_deriv);
    observed_changes(i) = new_objf - initial_objf;
  }
  double threshold = 0.1;
  KALDI_LOG << "predicted_changes = " << predicted_changes;
  KALDI_LOG << "observed_changes = " << observed_changes;
  if (!ApproxEqual(predicted_changes, observed_changes, threshold)) {
    KALDI_WARN << "Derivatives self-test failed.";
    return false;
  } else {
    return true;
  }
}


void NnetCombiner::SelfTestModelDerivatives() {
  int32 num_tests = 3;  // more properly, this is the number of dimensions in a
                        // single test.
  int32 dim = ParameterDim();

  Vector<double> params(dim), deriv(dim);
  Vector<double> predicted_changes(num_tests),
      observed_changes(num_tests);

  GetInitialParameters(&params);
  Vector<double> weights(WeightDim()), normalized_weights(WeightDim());
  Vector<BaseFloat> nnet_params(NnetParameterDim(), kUndefined),
      nnet_deriv(NnetParameterDim(), kUndefined);
  GetWeights(params, &weights);
  GetNormalizedWeights(weights, &normalized_weights);
  GetNnetParameters(normalized_weights, &nnet_params);

  double initial_objf = ComputeObjfAndDerivFromNnet(nnet_params,
                                                       &nnet_deriv);

  double delta = 0.002 * std::sqrt(VecVec(nnet_params, nnet_params) /
                                   NnetParameterDim());


  for (int32 i = 0; i < num_tests; i++) {
    Vector<BaseFloat> new_nnet_deriv(NnetParameterDim()),
        offset(NnetParameterDim()), new_nnet_params(nnet_params);
    offset.SetRandn();
    new_nnet_params.AddVec(delta, offset);
    double new_objf = ComputeObjfAndDerivFromNnet(new_nnet_params,
                                                     &new_nnet_deriv);
    // for predicted changes, interpolate old and new derivs.
    predicted_changes(i) =
        0.5 * VecVec(new_nnet_params, nnet_deriv) -
        0.5 * VecVec(nnet_params, nnet_deriv) +
        0.5 * VecVec(new_nnet_params, new_nnet_deriv) -
        0.5 * VecVec(nnet_params, new_nnet_deriv);
    observed_changes(i) = new_objf - initial_objf;
  }
  double threshold = 0.1;
  KALDI_LOG << "model-derivatives: predicted_changes = " << predicted_changes;
  KALDI_LOG << "model-derivatives: observed_changes = " << observed_changes;
  if (!ApproxEqual(predicted_changes, observed_changes, threshold))
    KALDI_WARN << "Model derivatives self-test failed.";
}




int32 NnetCombiner::ParameterDim() const {
  if (config_.separate_weights_per_component)
    return NumUpdatableComponents() * nnet_params_.NumRows();
  else
    return nnet_params_.NumRows();
}


void NnetCombiner::GetInitialParameters(VectorBase<double> *params) const {
  KALDI_ASSERT(params->Dim() == ParameterDim());
  params->Set(1.0 / nnet_params_.NumRows());
  if (config_.enforce_positive_weights) {
    // we enforce positive weights by treating the params as the log of the
    // actual weight.
    params->ApplyLog();
  }
}

void NnetCombiner::GetWeights(const VectorBase<double> &params,
                              VectorBase<double> *weights) const {
  KALDI_ASSERT(weights->Dim() == WeightDim());
  if (config_.separate_weights_per_component) {
    weights->CopyFromVec(params);
  } else {
    int32 nc = NumUpdatableComponents();
    // have one parameter per row of nnet_params_, and need to repeat
    // the weight for the different components.
    for (int32 n = 0; n < nnet_params_.NumRows(); n++) {
      for (int32 c = 0; c < nc; c++)
        (*weights)(n * nc + c) = params(n);
    }
  }
  // we enforce positive weights by having the weights be the exponential of the
  // corresponding parameters.
  if (config_.enforce_positive_weights)
    weights->ApplyExp();
}


void NnetCombiner::GetParamsDeriv(const VectorBase<double> &weights,
                                  const VectorBase<double> &weights_deriv,
                                  VectorBase<double> *param_deriv) {
  KALDI_ASSERT(weights.Dim() == WeightDim() &&
               param_deriv->Dim() == ParameterDim());
  Vector<double> preexp_weights_deriv(weights_deriv);
  if (config_.enforce_positive_weights) {
    // to enforce positive weights we first compute weights (call these
    // preexp_weights) and then take exponential.  Note, d/dx exp(x) = exp(x).
    // So the derivative w.r.t. the preexp_weights equals the derivative
    // w.r.t. the weights, times the weights.
    preexp_weights_deriv.MulElements(weights);
  }
  if (config_.separate_weights_per_component) {
    param_deriv->CopyFromVec(preexp_weights_deriv);
  } else {
    int32 nc = NumUpdatableComponents();
    param_deriv->SetZero();
    for (int32 n = 0; n < nnet_params_.NumRows(); n++)
      for (int32 c = 0; c < nc; c++)
        (*param_deriv)(n) += preexp_weights_deriv(n * nc + c);
  }
}


double NnetCombiner::GetSumToOnePenalty(
    const VectorBase<double> &weights,
    VectorBase<double> *weights_penalty_deriv,
    bool print_weights) const {

  KALDI_ASSERT(config_.sum_to_one_penalty >= 0.0);
  double penalty = config_.sum_to_one_penalty;
  if (penalty == 0.0) {
    weights_penalty_deriv->SetZero();
    return 0.0;
  }
  double ans = 0.0;
  int32 num_uc = NumUpdatableComponents(),
    num_models = nnet_params_.NumRows();
  Vector<double> tot_weights(num_uc);
  std::ostringstream tot_weight_info;
  for (int32 c = 0; c < num_uc; c++) {
    double this_total_weight = 0.0;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      double this_weight = weights(index);
      this_total_weight += this_weight;
    }
    tot_weights(c) = this_total_weight;
    // this_total_weight_deriv is the derivative of the penalty
    // term w.r.t. this component's total weight.
    double this_total_weight_deriv;
    if (config_.enforce_positive_weights) {
      // if config_.enforce_positive_weights is true, then we choose to
      // formulate the penalty in a slightly different way.. this solves the
      // problem that with the formulation in the 'else' below, if for some
      // reason the total weight is << 1.0, the deriv w.r.t. the actual
      // parameters gets tiny [because weight = exp(params)].
      double log_total = log(this_total_weight);
      ans += -0.5 * penalty * log_total * log_total;
      double log_total_deriv = -1.0 * penalty * log_total;
      this_total_weight_deriv = log_total_deriv / this_total_weight;
    } else {
      ans += -0.5 * penalty *
             (this_total_weight - 1.0) * (this_total_weight - 1.0);
      this_total_weight_deriv = penalty * (1.0 - this_total_weight);

    }
    if (weights_penalty_deriv != NULL) {
      KALDI_ASSERT(weights.Dim() == weights_penalty_deriv->Dim());
      for (int32 m = 0; m < num_models; m++) {
        int32 index = m * num_uc + c;
        (*weights_penalty_deriv)(index) = this_total_weight_deriv;
      }
    }
  }
  if (print_weights) {
    Vector<BaseFloat> tot_weights_float(tot_weights);
    KALDI_LOG << "Total weights per component: "
              << PrintVectorPerUpdatableComponent(nnet_,
                                                  tot_weights_float);
  }
  return ans;
}


void NnetCombiner::GetNnetParameters(const Vector<double> &weights,
                                     VectorBase<BaseFloat> *nnet_params) const {
  KALDI_ASSERT(nnet_params->Dim() == nnet_params_.NumCols());
  nnet_params->SetZero();
  int32 num_uc = NumUpdatableComponents(),
      num_models = nnet_params_.NumRows();
  for (int32 m = 0; m < num_models; m++) {
    const SubVector<BaseFloat> src_params(nnet_params_, m);
    int32 dim_offset = 0;
    for (int32 c = 0; c < num_uc; c++) {
      int32 index = m * num_uc + c;
      BaseFloat weight = weights(index);
      int32 dim = updatable_component_dims_[c];
      const SubVector<BaseFloat> src_component_params(src_params, dim_offset,
                                                      dim);
      SubVector<BaseFloat> dest_component_params(*nnet_params, dim_offset, dim);
      dest_component_params.AddVec(weight, src_component_params);
      dim_offset += dim;
    }
    KALDI_ASSERT(dim_offset == nnet_params_.NumCols());
  }
}

// compare GetNnetParameters.
void NnetCombiner::GetWeightsDeriv(
    const VectorBase<BaseFloat> &nnet_params_deriv,
    VectorBase<double> *weights_deriv) {
  KALDI_ASSERT(nnet_params_deriv.Dim() == nnet_params_.NumCols() &&
               weights_deriv->Dim() == WeightDim());
  int32 num_uc = NumUpdatableComponents(),
      num_models = nnet_params_.NumRows();
  for (int32 m = 0; m < num_models; m++) {
    const SubVector<BaseFloat> src_params(nnet_params_, m);
    int32 dim_offset = 0;
    for (int32 c = 0; c < num_uc; c++) {
      int32 index = m * num_uc + c;
      int32 dim = updatable_component_dims_[c];
      const SubVector<BaseFloat> src_component_params(src_params, dim_offset,
                                                      dim);
      const SubVector<BaseFloat> component_params_deriv(nnet_params_deriv,
                                                        dim_offset, dim);
      (*weights_deriv)(index) = VecVec(src_component_params,
                                       component_params_deriv);
      dim_offset += dim;
    }
    KALDI_ASSERT(dim_offset == nnet_params_.NumCols());
  }
}

double NnetCombiner::ComputeObjfAndDerivFromNnet(
    VectorBase<BaseFloat> &nnet_params,
    VectorBase<BaseFloat> *nnet_params_deriv) {
  BaseFloat sum = nnet_params.Sum();
  // inf/nan parameters->return -inf objective.
  if (!(sum == sum && sum - sum == 0))
    return -std::numeric_limits<double>::infinity();
  // Set nnet to have these params.
  UnVectorizeNnet(nnet_params, &nnet_);

  prob_computer_->Reset();
  std::vector<NnetExample>::const_iterator iter = egs_.begin(),
                                            end = egs_.end();
  for (; iter != end; ++iter)
    prob_computer_->Compute(*iter);
  double tot_weights,
    tot_objf = prob_computer_->GetTotalObjective(&tot_weights);
  KALDI_ASSERT(tot_weights > 0.0);
  const Nnet &deriv = prob_computer_->GetDeriv();
  VectorizeNnet(deriv, nnet_params_deriv);
  // we prefer to deal with normalized objective functions.
  nnet_params_deriv->Scale(1.0 / tot_weights);
  return tot_objf / tot_weights;
}


double NnetCombiner::ComputeObjfAndDerivFromParameters(
    VectorBase<double> &params,
    VectorBase<double> *params_deriv) {
  Vector<double> weights(WeightDim()), normalized_weights(WeightDim()),
      weights_sum_to_one_penalty_deriv(WeightDim()),
      normalized_weights_deriv(WeightDim()), weights_deriv(WeightDim());
  Vector<BaseFloat>
      nnet_params(NnetParameterDim(), kUndefined),
      nnet_params_deriv(NnetParameterDim(), kUndefined);
  GetWeights(params, &weights);
  double ans = GetSumToOnePenalty(weights, &weights_sum_to_one_penalty_deriv);
  GetNormalizedWeights(weights, &normalized_weights);
  GetNnetParameters(normalized_weights, &nnet_params);
  ans += ComputeObjfAndDerivFromNnet(nnet_params, &nnet_params_deriv);
  if (ans != ans || ans - ans != 0) // NaN or inf
    return ans;  // No point computing derivative
  GetWeightsDeriv(nnet_params_deriv, &normalized_weights_deriv);
  GetUnnormalizedWeightsDeriv(weights, normalized_weights_deriv,
                              &weights_deriv);
  weights_deriv.AddVec(1.0, weights_sum_to_one_penalty_deriv);
  GetParamsDeriv(weights, weights_deriv, params_deriv);
  return ans;
}


// enforces the constraint that the weights for each component must sum to one,
// if necessary.
void NnetCombiner::GetNormalizedWeights(
    const VectorBase<double> &unnorm_weights,
    VectorBase<double> *norm_weights) const {
  if (!config_.enforce_sum_to_one) {
    norm_weights->CopyFromVec(unnorm_weights);
    return;
  }
  int32 num_uc = NumUpdatableComponents(),
      num_models = nnet_params_.NumRows();
  for (int32 c = 0; c < num_uc; c++) {
    double sum = 0.0;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      sum += unnorm_weights(index);
    }
    double inv_sum = 1.0 / sum;  // if it's NaN then it's OK, we'll get NaN
                                    // weights and eventually -inf objective.
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      (*norm_weights)(index) = unnorm_weights(index) * inv_sum;
    }
  }
}

void NnetCombiner::GetUnnormalizedWeightsDeriv(
    const VectorBase<double> &unnorm_weights,
    const VectorBase<double> &norm_weights_deriv,
    VectorBase<double> *unnorm_weights_deriv) {
  if (!config_.enforce_sum_to_one) {
    unnorm_weights_deriv->CopyFromVec(norm_weights_deriv);
    return;
  }
  int32 num_uc = NumUpdatableComponents(),
      num_models = nnet_params_.NumRows();
  for (int32 c = 0; c < num_uc; c++) {
    double sum = 0.0;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      sum += unnorm_weights(index);
    }
    double inv_sum = 1.0 / sum;
    double inv_sum_deriv = 0.0;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      // in the forward direction, we'd do:
      // (*norm_weights)(index) = unnorm_weights(index) * inv_sum;
      (*unnorm_weights_deriv)(index) = inv_sum * norm_weights_deriv(index);
      inv_sum_deriv += norm_weights_deriv(index) * unnorm_weights(index);
    }
    // note: d/dx (1/x) = -1/x^2
    double sum_deriv = -1.0 * inv_sum_deriv * inv_sum * inv_sum;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      (*unnorm_weights_deriv)(index) += sum_deriv;
    }
  }
}




} // namespace nnet3
} // namespace kaldi

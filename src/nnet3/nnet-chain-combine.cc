// nnet3/nnet-chain-combine.cc

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

#include "nnet3/nnet-chain-combine.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

NnetChainCombiner::NnetChainCombiner(const NnetCombineConfig &combine_config,
                                     const chain::ChainTrainingOptions &chain_config,
                                     int32 num_nnets,
                                     const std::vector<NnetChainExample> &egs,
                                     const fst::StdVectorFst &den_fst,
                                     const Nnet &first_nnet):
    combine_config_(combine_config),
    chain_config_(chain_config),
    egs_(egs),
    den_fst_(den_fst),
    nnet_(first_nnet),
    num_real_input_nnets_(num_nnets),
    nnet_params_(std::min(num_nnets, combine_config_.max_effective_inputs),
                 NumParameters(first_nnet)),
    tot_input_weighting_(nnet_params_.NumRows()) {
  SetDropoutProportion(0, &nnet_);
  SubVector<BaseFloat> first_params(nnet_params_, 0);
  VectorizeNnet(nnet_, &first_params);
  tot_input_weighting_(0) += 1.0;
  num_nnets_provided_ = 1;
  ComputeUpdatableComponentDims();
  NnetComputeProbOptions compute_prob_opts;
  compute_prob_opts.compute_deriv = true;
  prob_computer_ = new NnetChainComputeProb(compute_prob_opts, chain_config_, den_fst_, nnet_);
}

void NnetChainCombiner::ComputeUpdatableComponentDims(){
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

void NnetChainCombiner::AcceptNnet(const Nnet &nnet) {
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

void NnetChainCombiner::FinishPreprocessingInput() {
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

void NnetChainCombiner::Combine() {
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
  lbfgs_options.first_step_impr = combine_config_.initial_impr;

  Vector<BaseFloat> params(dim), deriv(dim);
  BaseFloat objf, initial_objf;
  GetInitialParameters(&params);


  OptimizeLbfgs<BaseFloat> lbfgs(params, lbfgs_options);

  for (int32 i = 0; i < combine_config_.num_iters; i++) {
    params.CopyFromVec(lbfgs.GetProposedValue());
    objf = ComputeObjfAndDerivFromParameters(params, &deriv);
    KALDI_VLOG(2) << "Iteration " << i << " params = " << params
                  << ", objf = " << objf << ", deriv = " << deriv;
    if (i == 0) initial_objf = objf;
    lbfgs.DoStep(objf, deriv);
  }

  KALDI_LOG << "Combining nnets, objective function changed from "
            << initial_objf << " to " << objf;

  // must recompute nnet_ if "params" is not exactly equal to the
  // final params that LB
  Vector<BaseFloat> final_params(dim);
  final_params.CopyFromVec(lbfgs.GetValue(&objf));
  if (!params.ApproxEqual(final_params, 0.0)) {
    // the following call makes sure that nnet_ corresponds to the parameters
    // in "params".
    ComputeObjfAndDerivFromParameters(final_params, &deriv);
  }
  PrintParams(final_params);
}


void NnetChainCombiner::PrintParams(const VectorBase<BaseFloat> &params) const {

  Vector<BaseFloat> weights(params.Dim()), normalized_weights(params.Dim());
  GetWeights(params, &weights);
  GetNormalizedWeights(weights, &normalized_weights);
  int32 num_models = nnet_params_.NumRows(),
      num_uc = NumUpdatableComponents();

  if (combine_config_.separate_weights_per_component) {
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
              "printed due to the the --num-effective-nnets option; "
              "there were " << num_real_input_nnets_ << " actual input nnets. "
              "Each weight corresponds to a weighted average over a range of "
              "nnets in the sequence (with triangular bins)";
}

bool NnetChainCombiner::SelfTestDerivatives() {
  int32 num_tests = 2;  // more properly, this is the number of dimensions in a
                        // single test.
  BaseFloat delta = 0.001;
  int32 dim = ParameterDim();

  Vector<BaseFloat> params(dim), deriv(dim);
  Vector<BaseFloat> predicted_changes(num_tests),
      observed_changes(num_tests);

  GetInitialParameters(&params);
  BaseFloat initial_objf = ComputeObjfAndDerivFromParameters(params,
                                                             &deriv);
  for (int32 i = 0; i < num_tests; i++) {
    Vector<BaseFloat> new_deriv(dim), offset(dim), new_params(params);
    offset.SetRandn();
    new_params.AddVec(delta, offset);
    BaseFloat new_objf = ComputeObjfAndDerivFromParameters(new_params,
                                                           &new_deriv);
    // for predicted changes, interpolate old and new derivs.
    predicted_changes(i) =
        0.5 * VecVec(new_params, deriv) -  0.5 * VecVec(params, deriv) +
        0.5 * VecVec(new_params, new_deriv) - 0.5 * VecVec(params, new_deriv);
    observed_changes(i) = new_objf - initial_objf;
  }
  BaseFloat threshold = 0.1;
  KALDI_LOG << "predicted_changes = " << predicted_changes;
  KALDI_LOG << "observed_changes = " << observed_changes;
  if (!ApproxEqual(predicted_changes, observed_changes, threshold)) {
    KALDI_WARN << "Derivatives self-test failed.";
    return false;
  } else {
    return true;
  }
}


void NnetChainCombiner::SelfTestModelDerivatives() {
  int32 num_tests = 3;  // more properly, this is the number of dimensions in a
                        // single test.
  int32 dim = ParameterDim();

  Vector<BaseFloat> params(dim), deriv(dim);
  Vector<BaseFloat> predicted_changes(num_tests),
      observed_changes(num_tests);

  GetInitialParameters(&params);
  Vector<BaseFloat> weights(WeightDim()), normalized_weights(WeightDim()),
      nnet_params(NnetParameterDim(), kUndefined),
      nnet_deriv(NnetParameterDim(), kUndefined);
  GetWeights(params, &weights);
  GetNormalizedWeights(weights, &normalized_weights);
  GetNnetParameters(normalized_weights, &nnet_params);

  BaseFloat initial_objf = ComputeObjfAndDerivFromNnet(nnet_params,
                                                       &nnet_deriv);

  BaseFloat delta = 0.002 * std::sqrt(VecVec(nnet_params, nnet_params) /
                                      NnetParameterDim());


  for (int32 i = 0; i < num_tests; i++) {
    Vector<BaseFloat> new_nnet_deriv(NnetParameterDim()),
        offset(NnetParameterDim()), new_nnet_params(nnet_params);
    offset.SetRandn();
    new_nnet_params.AddVec(delta, offset);
    BaseFloat new_objf = ComputeObjfAndDerivFromNnet(new_nnet_params,
                                                     &new_nnet_deriv);
    // for predicted changes, interpolate old and new derivs.
    predicted_changes(i) =
        0.5 * VecVec(new_nnet_params, nnet_deriv) -
        0.5 * VecVec(nnet_params, nnet_deriv) +
        0.5 * VecVec(new_nnet_params, new_nnet_deriv) -
        0.5 * VecVec(nnet_params, new_nnet_deriv);
    observed_changes(i) = new_objf - initial_objf;
  }
  BaseFloat threshold = 0.1;
  KALDI_LOG << "model-derivatives: predicted_changes = " << predicted_changes;
  KALDI_LOG << "model-derivatives: observed_changes = " << observed_changes;
  if (!ApproxEqual(predicted_changes, observed_changes, threshold))
    KALDI_WARN << "Model derivatives self-test failed.";
}




int32 NnetChainCombiner::ParameterDim() const {
  if (combine_config_.separate_weights_per_component)
    return NumUpdatableComponents() * nnet_params_.NumRows();
  else
    return nnet_params_.NumRows();
}


void NnetChainCombiner::GetInitialParameters(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == ParameterDim());
  params->Set(1.0 / nnet_params_.NumRows());
  if (combine_config_.enforce_positive_weights) {
    // we enforce positive weights by treating the params as the log of the
    // actual weight.
    params->ApplyLog();
  }
}

void NnetChainCombiner::GetWeights(const VectorBase<BaseFloat> &params,
                              VectorBase<BaseFloat> *weights) const {
  KALDI_ASSERT(weights->Dim() == WeightDim());
  if (combine_config_.separate_weights_per_component) {
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
  if (combine_config_.enforce_positive_weights)
    weights->ApplyExp();
}


void NnetChainCombiner::GetParamsDeriv(const VectorBase<BaseFloat> &weights,
                                  const VectorBase<BaseFloat> &weights_deriv,
                                  VectorBase<BaseFloat> *param_deriv) {
  KALDI_ASSERT(weights.Dim() == WeightDim() &&
               param_deriv->Dim() == ParameterDim());
  Vector<BaseFloat> preexp_weights_deriv(weights_deriv);
  if (combine_config_.enforce_positive_weights) {
    // to enforce positive weights we first compute weights (call these
    // preexp_weights) and then take exponential.  Note, d/dx exp(x) = exp(x).
    // So the derivative w.r.t. the preexp_weights equals the derivative
    // w.r.t. the weights, times the weights.
    preexp_weights_deriv.MulElements(weights);
  }
  if (combine_config_.separate_weights_per_component) {
    param_deriv->CopyFromVec(preexp_weights_deriv);
  } else {
    int32 nc = NumUpdatableComponents();
    param_deriv->SetZero();
    for (int32 n = 0; n < nnet_params_.NumRows(); n++)
      for (int32 c = 0; c < nc; c++)
        (*param_deriv)(n) += preexp_weights_deriv(n * nc + c);
  }
}


void NnetChainCombiner::GetNnetParameters(const Vector<BaseFloat> &weights,
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
void NnetChainCombiner::GetWeightsDeriv(
    const VectorBase<BaseFloat> &nnet_params_deriv,
    VectorBase<BaseFloat> *weights_deriv) {
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

double NnetChainCombiner::ComputeObjfAndDerivFromNnet(
    VectorBase<BaseFloat> &nnet_params,
    VectorBase<BaseFloat> *nnet_params_deriv) {
  BaseFloat sum = nnet_params.Sum();
  // inf/nan parameters->return -inf objective.
  if (!(sum == sum && sum - sum == 0))
    return -std::numeric_limits<double>::infinity();
  // Set nnet to have these params.
  UnVectorizeNnet(nnet_params, &nnet_);

  prob_computer_->Reset();
  std::vector<NnetChainExample>::const_iterator iter = egs_.begin(),
                                                end = egs_.end();
  for (; iter != end; ++iter)
    prob_computer_->Compute(*iter);
  const ChainObjectiveInfo *objf_info =
      prob_computer_->GetObjective("output");
  if (objf_info == NULL)
    KALDI_ERR << "Error getting objective info (unsuitable egs?)";
  KALDI_ASSERT(objf_info->tot_weight > 0.0);
  const Nnet &deriv = prob_computer_->GetDeriv();
  VectorizeNnet(deriv, nnet_params_deriv);
  // we prefer to deal with normalized objective functions.
  nnet_params_deriv->Scale(1.0 / objf_info->tot_weight);
  return (objf_info->tot_like + objf_info->tot_l2_term) / objf_info->tot_weight;
}


double NnetChainCombiner::ComputeObjfAndDerivFromParameters(
    VectorBase<BaseFloat> &params,
    VectorBase<BaseFloat> *params_deriv) {
  Vector<BaseFloat> weights(WeightDim()), normalized_weights(WeightDim()),
      nnet_params(NnetParameterDim(), kUndefined),
      nnet_params_deriv(NnetParameterDim(), kUndefined),
      normalized_weights_deriv(WeightDim()), weights_deriv(WeightDim());
  GetWeights(params, &weights);
  GetNormalizedWeights(weights, &normalized_weights);
  GetNnetParameters(normalized_weights, &nnet_params);
  double ans = ComputeObjfAndDerivFromNnet(nnet_params, &nnet_params_deriv);
  if (ans != ans || ans - ans != 0) // NaN or inf
    return ans;  // No point computing derivative
  GetWeightsDeriv(nnet_params_deriv, &normalized_weights_deriv);
  GetUnnormalizedWeightsDeriv(weights, normalized_weights_deriv,
                              &weights_deriv);
  GetParamsDeriv(weights, weights_deriv, params_deriv);
  return ans;
}


// enforces the constraint that the weights for each component must sum to one.
void NnetChainCombiner::GetNormalizedWeights(
    const VectorBase<BaseFloat> &unnorm_weights,
    VectorBase<BaseFloat> *norm_weights) const {
  if (!combine_config_.enforce_sum_to_one) {
    norm_weights->CopyFromVec(unnorm_weights);
    return;
  }
  int32 num_uc = NumUpdatableComponents(),
      num_models = nnet_params_.NumRows();
  for (int32 c = 0; c < num_uc; c++) {
    BaseFloat sum = 0.0;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      sum += unnorm_weights(index);
    }
    BaseFloat inv_sum = 1.0 / sum;  // if it's NaN then it's OK, we'll get NaN
                                    // weights and eventually -inf objective.
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      (*norm_weights)(index) = unnorm_weights(index) * inv_sum;
    }
  }
}

void NnetChainCombiner::GetUnnormalizedWeightsDeriv(
    const VectorBase<BaseFloat> &unnorm_weights,
    const VectorBase<BaseFloat> &norm_weights_deriv,
    VectorBase<BaseFloat> *unnorm_weights_deriv) {
  if (!combine_config_.enforce_sum_to_one) {
    unnorm_weights_deriv->CopyFromVec(norm_weights_deriv);
    return;
  }
  int32 num_uc = NumUpdatableComponents(),
      num_models = nnet_params_.NumRows();
  for (int32 c = 0; c < num_uc; c++) {
    BaseFloat sum = 0.0;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      sum += unnorm_weights(index);
    }
    BaseFloat inv_sum = 1.0 / sum;
    BaseFloat inv_sum_deriv = 0.0;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      // in the forward direction, we'd do:
      // (*norm_weights)(index) = unnorm_weights(index) * inv_sum;
      (*unnorm_weights_deriv)(index) = inv_sum * norm_weights_deriv(index);
      inv_sum_deriv += norm_weights_deriv(index) * unnorm_weights(index);
    }
    // note: d/dx (1/x) = -1/x^2
    BaseFloat sum_deriv = -1.0 * inv_sum_deriv * inv_sum * inv_sum;
    for (int32 m = 0; m < num_models; m++) {
      int32 index = m * num_uc + c;
      (*unnorm_weights_deriv)(index) += sum_deriv;
    }
  }
}




} // namespace nnet3
} // namespace kaldi

// nnet3/nnet-chain-combine.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_CHAIN_COMBINE_H_
#define KALDI_NNET3_NNET_CHAIN_COMBINE_H_

#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-compute.h"
#include "util/parse-options.h"
#include "itf/options-itf.h"
#include "nnet3/nnet-combine.h"
#include "nnet3/nnet-chain-diagnostics.h"


namespace kaldi {
namespace nnet3 {

// we re-use NnetCombineConfig from nnet-combine.h

/*
  You should use this class as follows:
    - Call the constructor, giving it the egs and the first nnet.
    - Call AcceptNnet to provide all the other nnets.  (the nnets will
      be stored in a matrix in CPU memory, to avoid filing up GPU memory).
    - Call Combine()
    - Get the resultant nnet with GetNnet().
 */
class NnetChainCombiner {
 public:
  /// Caution: this object retains a const reference to the "egs", so don't
  /// delete them until it goes out of scope.
  NnetChainCombiner(const NnetCombineConfig &nnet_config,
                    const chain::ChainTrainingOptions &chain_config,
                    int32 num_nnets,
                    const std::vector<NnetChainExample> &egs,
                    const fst::StdVectorFst &den_fst,
                    const Nnet &first_nnet);

  /// You should call this function num_nnets-1 times after calling
  /// the constructor, to provide the remaining nnets.
  void AcceptNnet(const Nnet &nnet);

  void Combine();

  const Nnet &GetNnet() const { return nnet_; }

  ~NnetChainCombiner() { delete prob_computer_; }
 private:
  const NnetCombineConfig &combine_config_;
  const chain::ChainTrainingOptions &chain_config_;

  const std::vector<NnetChainExample> &egs_;

  const fst::StdVectorFst &den_fst_;

  Nnet nnet_;  // The current neural network.

  NnetChainComputeProb *prob_computer_;

  std::vector<int32> updatable_component_dims_;  // dimension of each updatable
                                                 // component.

  int32 num_real_input_nnets_;  // number of actual nnet inputs.

  int32 num_nnets_provided_;  // keeps track of the number of calls to AcceptNnet().

  // nnet_params_ are the parameters of the "effective input"
  // neural nets; they will often be the same as the real inputs,
  // but if num_real_input_nnets_ > config_.num_effective_nnets, they
  // will be weighted combinations.
  Matrix<BaseFloat> nnet_params_;

  // This vector has the same dimension as nnet_params_.NumRows(),
  // and helps us normalize so each row of nnet_params correspondss to
  // a weighted average of its inputs.
  Vector<BaseFloat> tot_input_weighting_;

  // returns the parameter dimension, i.e. the dimension of the parameters that
  // we are optimizing.  This depends on the config, the number of updatable
  // components and nnet_params_.NumRows(); it will never exceed the number of
  // updatable components times nnet_params_.NumRows().
  int32 ParameterDim() const;

  int32 NumUpdatableComponents() const {
    return updatable_component_dims_.size();
  }
  // returns the weight dimension.
  int32 WeightDim() const {
    return nnet_params_.NumRows() * NumUpdatableComponents();
  }

  int32 NnetParameterDim() const { return nnet_params_.NumCols(); }

  // Computes the initial parameters.  The parameters are the underlying thing
  // that we optimize; their dimension equals ParameterDim().  They are not the same
  // thing as the nnet parameters.
  void GetInitialParameters(VectorBase<BaseFloat> *params) const;

  // Tests that derivatives are accurate.  Prints warning and returns false if not.
  bool SelfTestDerivatives();

  // Tests that model derivatives are accurate.  Just prints warning if not.
  void SelfTestModelDerivatives();


  // prints the parameters via logging statements.
  void PrintParams(const VectorBase<BaseFloat> &params) const;

  // This function computes the objective function (and its derivative, if the objective
  // function is finite) at the given value of the parameters (the parameters we're optimizing,
  // i.e. the combination weights; not the nnet parameters.  This function calls most of the
  // functions below.
  double ComputeObjfAndDerivFromParameters(
      VectorBase<BaseFloat> &params,
      VectorBase<BaseFloat> *params_deriv);


  // Computes the weights from the parameters in a config-dependent way.  The
  // weight dimension is always (the number of updatable components times
  // nnet_params_.NumRows()).
  void GetWeights(const VectorBase<BaseFloat> &params,
                  VectorBase<BaseFloat> *weights) const;

  // Given the raw weights: if config_.enforce_sum_to_one, then compute weights
  // with sum-to-one constrint per component included; else just copy input to
  // output.
  void GetNormalizedWeights(const VectorBase<BaseFloat> &unnorm_weights,
                            VectorBase<BaseFloat> *norm_weights) const;

  // Computes the nnet-parameter vector from the normalized weights and
  // nnet_params_, as a vector.  (See the functions Vectorize() and
  // UnVectorize() for how they relate to the nnet's components' parameters).
  void GetNnetParameters(const Vector<BaseFloat> &normalized_weights,
                         VectorBase<BaseFloat> *nnet_params) const;

  // This function computes the objective function (and its derivative, if the objective
  // function is finite) at the given value of nnet parameters.  This involves the
  // nnet computation.
  double ComputeObjfAndDerivFromNnet(VectorBase<BaseFloat> &nnet_params,
                                     VectorBase<BaseFloat> *nnet_params_deriv);

  // Given an objective-function derivative with respect to the nnet parameters,
  // computes the derivative with respect to the (normalized) weights.
  void GetWeightsDeriv(const VectorBase<BaseFloat> &nnet_params_deriv,
                       VectorBase<BaseFloat> *normalized_weights_deriv);


  // Computes the derivative w.r.t. the unnormalized weights, by propagating
  // through the normalization operation.
  // If config_.enforce_sum_to_one == false, just copies norm_weights_deriv to
  // unnorm_weights_deriv.
  void GetUnnormalizedWeightsDeriv(const VectorBase<BaseFloat> &unnorm_weights,
                                   const VectorBase<BaseFloat> &norm_weights_deriv,
                                   VectorBase<BaseFloat> *unnorm_weights_deriv);


  // Given a derivative w.r.t. the weights, outputs a derivative w.r.t.
  // the params
  void GetParamsDeriv(const VectorBase<BaseFloat> &weights,
                      const VectorBase<BaseFloat> &weight_deriv,
                      VectorBase<BaseFloat> *param_deriv);

  void ComputeUpdatableComponentDims();
  void FinishPreprocessingInput();

};



} // namespace nnet3
} // namespace kaldi

#endif

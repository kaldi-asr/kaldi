// nnet2/nnet-lbfgs.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-lbfgs.h"

namespace kaldi {
namespace nnet2 {


Nnet *GetPreconditioner(const Nnet &nnet) {
  Nnet *ans = new Nnet(nnet);
  bool is_gradient = true;
  ans->SetZero(is_gradient);
  for (int32 c = 0; c < ans->NumComponents(); c++) {
    AffineComponent *ac = dynamic_cast<AffineComponent*>(
        &(ans->GetComponent(c)));
    if (ac != NULL) {
      AffineComponentA *ac_new = new AffineComponentA(*ac);
      ans->SetComponent(c, ac_new);
      ac_new->InitializeScatter();
    }
  }
  return ans;
}

void PreconditionNnet(const PreconditionConfig &config,
                      Nnet *preconditioner,
                      Nnet *nnet) {
  KALDI_ASSERT(preconditioner->NumComponents() == nnet->NumComponents());
  for (int32 c = 0; c < preconditioner->NumComponents(); c++) {
    AffineComponent *ac =
        dynamic_cast<AffineComponent*>(&(nnet->GetComponent(c)));
    if (ac != NULL) {
      AffineComponentA *pac =
          dynamic_cast<AffineComponentA*>(&(preconditioner->GetComponent(c)));
      KALDI_ASSERT(pac != NULL &&
                   "Incorrect input to PreconditionNnet (not a preconditioner?)");
      pac->Precondition(config, ac);
    }
  }
}

void NnetLbfgsTrainer::Initialize(Nnet *nnet_in) {

  if (config_.precondition_config.do_precondition) {
    nnet_precondition_ = GetPreconditioner(*nnet_in);
    
    initial_objf_ = ComputeNnetGradient(*nnet_in, egs_, config_.minibatch_size,
                                        nnet_precondition_);
    
  } else {
    nnet_precondition_ = NULL;
  }
  nnet_ = nnet_in;
  params_.Resize(nnet_->GetParameterDim());
  
  CopyParamsOrGradientFromNnet(*nnet_, &params_);

  LbfgsOptions opts;
  opts.m = config_.lbfgs_dim;
  opts.first_step_impr = config_.initial_impr;
  opts.minimize = false; // We're maximizing.
  
  lbfgs_ = new OptimizeLbfgs<BaseFloat>(params_, opts);
}


void NnetLbfgsTrainer::CopyParamsOrGradientFromNnet(const Nnet &nnet,
                                                    VectorBase<BaseFloat> *params) {
  KALDI_ASSERT(nnet.GetParameterDim() == params->Dim());
  if (nnet_precondition_ != NULL) {
    Nnet nnet_temp(nnet);
    for (int32 c = 0; c < nnet.NumComponents(); c++) {
      AffineComponentA *aca = dynamic_cast<AffineComponentA*>(
          &(nnet_precondition_->GetComponent(c)));
      AffineComponent *aca_temp = dynamic_cast<AffineComponent*>(
          &(nnet_temp.GetComponent(c)));
      if (aca != NULL)
        aca->Transform(config_.precondition_config,
                       true, // "forward"-- we're transforming into normalized space.
                       aca_temp);
    }
    nnet_temp.Vectorize(params);
  } else {
    nnet.Vectorize(params);
  }
}

void NnetLbfgsTrainer::CopyParamsOrGradientToNnet(const VectorBase<BaseFloat> &params,
                                                  Nnet *nnet) {
  KALDI_ASSERT(nnet->GetParameterDim() == params.Dim());
  nnet->UnVectorize(params);
  if (nnet_precondition_ != NULL) {
    for (int32 c = 0; c < nnet->NumComponents(); c++) {
      AffineComponentA *aca = dynamic_cast<AffineComponentA*>(
          &(nnet_precondition_->GetComponent(c)));
      AffineComponent *aca_nnet = dynamic_cast<AffineComponent*>(
          &(nnet->GetComponent(c)));
      if (aca != NULL)
        aca->Transform(config_.precondition_config,
                       false, // "forward == false"-- we're transforming from normalized to
                       aca_nnet); // canonical space.
    }
  } 
}

void NnetLbfgsTrainer::Train(Nnet *nnet_in) {
  Initialize(nnet_in);

  BaseFloat initial_objf;
  for (int32 iter = 0; iter < config_.lbfgs_num_iters; iter++) {
    // Note: if we use --verbose=2, debugging info will be printed
    // out from the L-BFGS code.

    Vector<BaseFloat> gradient(params_.Dim());
    BaseFloat objf = GetObjfAndGradient(params_, &gradient);
    KALDI_LOG << "Iteration " << iter << " of L-BFGS, objf is "
              << objf;
    if (iter == 0)
      initial_objf = objf;
    lbfgs_->DoStep(objf, gradient);
    params_.CopyFromVec(lbfgs_->GetProposedValue()); 
  }
  BaseFloat cur_objf;
  params_.CopyFromVec(lbfgs_->GetValue(&cur_objf));
  KALDI_LOG << "After " << config_.lbfgs_num_iters << " iterations of L-BFGS, "
            << " objective function improved from " << initial_objf << " to "
            << cur_objf;
  CopyParamsOrGradientToNnet(params_, nnet_in);
}

BaseFloat NnetLbfgsTrainer::GetObjfAndGradient(
    const VectorBase<BaseFloat> &cur_value,
    VectorBase<BaseFloat> *gradient) {
  Nnet nnet(*nnet_), nnet_gradient(*nnet_);
  CopyParamsOrGradientToNnet(cur_value, &nnet);
  bool is_gradient = true;
  nnet_gradient.SetZero(is_gradient);
  BaseFloat objf = ComputeNnetGradient(nnet, egs_, config_.minibatch_size,
                                       &nnet_gradient);
  CopyParamsOrGradientFromNnet(nnet_gradient, gradient);
  gradient->Scale(1.0 / egs_.size());
  return objf;
}

NnetLbfgsTrainer::~NnetLbfgsTrainer() {
  delete nnet_precondition_;
  delete lbfgs_;
}


} // namespace nnet2
} // namespace kaldi

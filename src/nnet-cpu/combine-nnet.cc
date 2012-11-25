// nnet/combine-nnet.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/combine-nnet.h"

namespace kaldi {

static int32 NumUpdatableComponents(const Nnet &nnet) {
  int32 ans = 0;
  for (int32 i = 0; i < nnet.NumComponents(); i++)
    if (dynamic_cast<const UpdatableComponent*>(&(nnet.GetComponent(i))) != NULL)
      ans++;
  return ans;
}


static void ScaleNnet(const Vector<BaseFloat> &scale_params,
                      Nnet *nnet) {
  int32 i = 0;
  for (int32 j = 0; j < nnet->NumComponents(); j++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(nnet->GetComponent(j)));
    if (uc != NULL) {
      uc->Scale(scale_params(i));
      i++;
    }
  }
  KALDI_ASSERT(i == scale_params.Dim());
}


static BaseFloat ComputeObjfAndGradient(
    const std::vector<NnetTrainingExample> &validation_set,
    const Vector<double> &log_scale_params,
    const Nnet &nnet,
    Vector<double> *gradient) {
  Vector<BaseFloat> scale_params(log_scale_params);
  scale_params.ApplyExp();
  Nnet nnet_scaled(nnet);
  ScaleNnet(scale_params, &nnet_scaled);
  
  Nnet nnet_gradient(nnet);
  bool is_gradient = true;
  nnet_gradient.SetZero(is_gradient);

  // note: "ans" is normalized by the total weight of validation frames.
  int32 batch_size = 1024;
  BaseFloat ans = ComputeNnetGradient(nnet_scaled,
                                      validation_set,
                                      batch_size,
                                      &nnet_gradient);

  BaseFloat tot_count = TotalNnetTrainingWeight(validation_set);
  int32 i = 0; // index into log_scale_params.
  for (int32 j = 0; j < nnet_scaled.NumComponents(); j++) {
    const UpdatableComponent *uc =
        dynamic_cast<const UpdatableComponent*>(&(nnet.GetComponent(j))),
        *uc_gradient =
        dynamic_cast<const UpdatableComponent*>(&(nnet_gradient.GetComponent(j)));
    if (uc != NULL) {
      BaseFloat dotprod = uc->DotProduct(*uc_gradient) / tot_count;
      (*gradient)(i) = dotprod * scale_params(i); // gradient w.r.t log of scaling factor.
      // We multiply by scale_params(i) to take into account d/dx exp(x); "gradient"
      // is the gradient w.r.t. the log of the scale_params.
      i++;
    }
  }
  KALDI_ASSERT(i == log_scale_params.Dim());
  return ans;
}
                                   

void ShrinkNnet(const NnetShrinkConfig &shrink_config,
                const std::vector<NnetTrainingExample> &validation_set,
                Nnet *nnet) {

  int32 dim = NumUpdatableComponents(*nnet);
  KALDI_ASSERT(dim > 0);
  Vector<double> log_scale(dim), gradient(dim),
      diag_approx_2nd_deriv(dim); // will be zero.
  
  // Get initial gradient.
  double objf = ComputeObjfAndGradient(validation_set, log_scale,
                                       *nnet,
                                       &gradient),
      initial_objf = objf;

  LbfgsOptions lbfgs_options;
  lbfgs_options.minimize = false; // We're maximizing.
  lbfgs_options.m = dim; // Store the same number of vectors as the dimension
  // itself, so this is BFGS.
  
  // set diag_approx_2nd_deriv (should be negative, we're maximizing).
  // We set this on the assumption that we want the change in log-scale
  // to have absolute value initial_change=0.05, which seems reasonable.
  for (int32 d = 0; d < dim; d++) {
    diag_approx_2nd_deriv(d) =
        -std::min(static_cast<double>(shrink_config.min_initial_2nd_deriv),
                  std::abs(gradient(d)) / shrink_config.initial_change);
  }

  OptimizeLbfgs<double> lbfgs(log_scale,
                              lbfgs_options);
  
  for (int32 i = 0; i < shrink_config.num_bfgs_iters; i++) {
    KALDI_VLOG(2) << "log-scale = " << log_scale << ", objf = " << objf
                  << ", gradient = " << gradient;

    log_scale.CopyFromVec(lbfgs.GetProposedValue());
    objf = ComputeObjfAndGradient(validation_set, log_scale,
                                  *nnet,
                                  &gradient);

    lbfgs.DoStep(objf, gradient);
  }

  log_scale.CopyFromVec(lbfgs.GetValue(&objf));

  Vector<BaseFloat> scale(log_scale);
  scale.ApplyExp();
  KALDI_LOG << "Shrinking nnet, validation objf per frame changed from "
            << initial_objf << " to " << objf << ", scale factors per layer are "
            << scale;
  ScaleNnet(scale, nnet);
}
 
  
} // namespace

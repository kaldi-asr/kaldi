// nnet2/shrink-nnet.cc

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

#include "nnet2/shrink-nnet.h"

namespace kaldi {
namespace nnet2 {

static BaseFloat ComputeObjfAndGradient(
    const std::vector<NnetExample> &validation_set,
    const Vector<double> &log_scale_params,
    const Nnet &nnet,
    Vector<double> *gradient) {
  Vector<BaseFloat> scale_params(log_scale_params);
  scale_params.ApplyExp();
  Nnet nnet_scaled(nnet);
  nnet_scaled.ScaleComponents(scale_params);
  
  Nnet nnet_gradient(nnet);
  bool is_gradient = true;
  nnet_gradient.SetZero(is_gradient);

  // note: "ans" is normalized by the total weight of validation frames.
  int32 batch_size = 1024;
  BaseFloat ans = ComputeNnetGradient(nnet_scaled,
                                      validation_set,
                                      batch_size,
                                      &nnet_gradient);

  BaseFloat tot_count = validation_set.size();
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
                const std::vector<NnetExample> &validation_set,
                Nnet *nnet) {

  int32 dim = nnet->NumUpdatableComponents();
  KALDI_ASSERT(dim > 0);
  Vector<double> log_scale(dim), gradient(dim); // will be zero.
  
  // Get initial gradient.
  double objf, initial_objf;


  LbfgsOptions lbfgs_options;
  lbfgs_options.minimize = false; // We're maximizing.
  lbfgs_options.m = dim; // Store the same number of vectors as the dimension
  // itself, so this is BFGS.
  lbfgs_options.first_step_length = shrink_config.initial_step;
  
  OptimizeLbfgs<double> lbfgs(log_scale,
                              lbfgs_options);
  
  for (int32 i = 0; i < shrink_config.num_bfgs_iters; i++) {
    log_scale.CopyFromVec(lbfgs.GetProposedValue());
    objf = ComputeObjfAndGradient(validation_set, log_scale,
                                  *nnet,
                                  &gradient);

    KALDI_VLOG(2) << "log-scale = " << log_scale << ", objf = " << objf
                  << ", gradient = " << gradient;
    if (i == 0) initial_objf = objf;

    lbfgs.DoStep(objf, gradient);
  }

  log_scale.CopyFromVec(lbfgs.GetValue(&objf));

  Vector<BaseFloat> scale(log_scale);
  scale.ApplyExp();
  KALDI_LOG << "Shrinking nnet, validation objf per frame changed from "
            << initial_objf << " to " << objf << ", scale factors per layer are "
            << scale;
  nnet->ScaleComponents(scale);
}
 
  
} // namespace nnet2
} // namespace kaldi

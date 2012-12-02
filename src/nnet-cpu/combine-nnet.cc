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


// Here, "scale_params" is in blocks, with the first block
// corresponding to nnets[0].
static void CombineNnets(const Vector<BaseFloat> &scale_params,
                         const std::vector<Nnet> &nnets,
                         Nnet *dest) {
  int32 num_nnets = nnets.size();
  KALDI_ASSERT(num_nnets >= 1);
  int32 num_uc = nnets[0].NumUpdatableComponents();
  KALDI_ASSERT(num_nnets * nnets[0].NumUpdatableComponents());
  
  
  *dest = nnets[0];
  SubVector<BaseFloat> scale_params0(scale_params, 0, num_uc);
  dest->ScaleComponents(scale_params0);
  for (int32 n = 1; n < num_nnets; n++) {
    SubVector<BaseFloat> scale_params_n(scale_params, n * num_uc, num_uc);
    dest->AddNnet(scale_params_n, nnets[n]);
  }
}


// This function chooses from among the neural nets, the one
// which has the best validation set objective function.
static void GetInitialScaleParams(
    const std::vector<NnetTrainingExample> &validation_set,
    const std::vector<Nnet> &nnets,
    Vector<double> *scale_params) {
  int32 minibatch_size = 1024;
  KALDI_ASSERT(!nnets.empty());
  BaseFloat tot_weight = TotalNnetTrainingWeight(validation_set);
  int32 best_n = -1;
  BaseFloat best_objf;
  Vector<BaseFloat> objfs(nnets.size());
  for (int32 n = 0; n < static_cast<int32>(nnets.size()); n++) {
    BaseFloat objf = ComputeNnetObjf(nnets[n], validation_set,
                                     minibatch_size) / tot_weight;
    
    if (n == 0 || objf > best_objf) {
      best_objf = objf;
      best_n = n;
    }
    objfs(n) = objf;
  }
  KALDI_LOG << "Objective functions for the source neural nets are " << objfs;
  int32 num_uc = nnets[0].NumUpdatableComponents();

  { // Now try a version where all the neural nets have the same weight.
    scale_params->Resize(num_uc * nnets.size());
    scale_params->Set(1.0 / nnets.size());
    Nnet average_nnet;
    Vector<BaseFloat> scale_params_float(*scale_params);
    CombineNnets(scale_params_float, nnets, &average_nnet);
    BaseFloat objf = ComputeNnetObjf(average_nnet, validation_set,
                                     minibatch_size) / tot_weight;
    KALDI_LOG << "Objf with all neural nets averaged is "
              << objf;
    if (objf > best_objf) {
      KALDI_LOG << "Initializing with all neural nets averaged.";
      return;
    }
  }

  KALDI_LOG << "Using neural net with index " << best_n
            << ", objective function was " << best_objf;
      
  // At this point we're using the best of the individual neural nets.
  scale_params->Set(0.0);

  // Set the block of parameters corresponding to the "best" of the
  // source neural nets to
  SubVector<double> best_block(*scale_params, num_uc * best_n, num_uc);
  best_block.Set(1.0);
}
    
static BaseFloat ComputeObjfAndGradient(
    const std::vector<NnetTrainingExample> &validation_set,
    const Vector<double> &scale_params,
    const std::vector<Nnet> &nnets,
    Vector<double> *gradient) {

  Vector<BaseFloat> scale_params_float(scale_params);

  Nnet nnet_combined;
  CombineNnets(scale_params_float, nnets, &nnet_combined);
  
  Nnet nnet_gradient(nnet_combined);
  bool is_gradient = true;
  nnet_gradient.SetZero(is_gradient);
  
  // note: "ans" is normalized by the total weight of validation frames.
  int32 batch_size = 1024;
  BaseFloat ans = ComputeNnetGradient(nnet_combined,
                                      validation_set,
                                      batch_size,
                                      &nnet_gradient);

  BaseFloat tot_count = TotalNnetTrainingWeight(validation_set);
  int32 i = 0; // index into scale_params.
  for (int32 n = 0; n < static_cast<int32>(nnets.size()); n++) {
    for (int32 j = 0; j < nnet_combined.NumComponents(); j++) {
      const UpdatableComponent *uc =
          dynamic_cast<const UpdatableComponent*>(&(nnets[n].GetComponent(j))),
          *uc_gradient =
          dynamic_cast<const UpdatableComponent*>(&(nnet_gradient.GetComponent(j)));
      if (uc != NULL) {
        BaseFloat dotprod = uc->DotProduct(*uc_gradient) / tot_count;
        (*gradient)(i) = dotprod; 
        i++;
      }
    }
  }
  KALDI_ASSERT(i == scale_params.Dim());
  return ans;
}
                                   

void CombineNnets(const NnetCombineConfig &combine_config,
                  const std::vector<NnetTrainingExample> &validation_set,
                  const std::vector<Nnet> &nnets,
                  Nnet *nnet_out) {

  Vector<double> scale_params;

  GetInitialScaleParams(validation_set,
                        nnets,
                        &scale_params);

  int32 dim = scale_params.Dim();
  KALDI_ASSERT(dim > 0);
  Vector<double> gradient(dim);
  
  double objf, initial_objf;

  LbfgsOptions lbfgs_options;
  lbfgs_options.minimize = false; // We're maximizing.
  lbfgs_options.m = dim; // Store the same number of vectors as the dimension
  // itself, so this is BFGS.
  lbfgs_options.first_step_length = combine_config.initial_step;
  
  OptimizeLbfgs<double> lbfgs(scale_params,
                              lbfgs_options);
  
  for (int32 i = 0; i < combine_config.num_bfgs_iters; i++) {    
    scale_params.CopyFromVec(lbfgs.GetProposedValue());
    objf = ComputeObjfAndGradient(validation_set,
                                  scale_params,
                                  nnets,
                                  &gradient);

    KALDI_VLOG(2) << "Iteration " << i << " scale-params = " << scale_params
                  << ", objf = " << objf << ", gradient = " << gradient;
    
    if (i == 0) initial_objf = objf;
    
    lbfgs.DoStep(objf, gradient);
  }

  scale_params.CopyFromVec(lbfgs.GetValue(&objf));

  Vector<BaseFloat> scale_params_float(scale_params);

  KALDI_LOG << "Combining nnets, validation objf per frame changed from "
            << initial_objf << " to " << objf << ", scale factors are "
            << scale_params_float;
  CombineNnets(scale_params_float, nnets, nnet_out);
}
 
  
} // namespace

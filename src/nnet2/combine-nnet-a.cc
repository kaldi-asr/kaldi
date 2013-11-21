// nnet2/combine-nnet-a.cc

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

#include "nnet2/combine-nnet-a.h"

namespace kaldi {
namespace nnet2 {

/*
  This function gets the "update direction".  The vector "nnets" is
  interpreted as (old-nnet new-nnet1 net-nnet2 ... new-nnetN), and
  the "update direction" is the average of the new nnets, minus the
  old nnet.
*/
static void GetUpdateDirection(const std::vector<Nnet> &nnets,
                               Nnet *direction) {
  KALDI_ASSERT(nnets.size() > 1);
  int32 num_new_nnets = nnets.size() - 1;
  Vector<BaseFloat> scales(nnets[0].NumUpdatableComponents());

  scales.Set(1.0 / num_new_nnets);
  
  *direction = nnets[1];
  direction->ScaleComponents(scales); // first of the new nnets.
  for (int32 n = 2; n < 1 + num_new_nnets; n++)
    direction->AddNnet(scales, nnets[n]);
  // now "direction" is the average of the new nnets.  Subtract
  // the old nnet's parameters.
  scales.Set(-1.0);
  direction->AddNnet(scales, nnets[0]);
}

/// Sets "dest" to orig_nnet plus "direction", with
/// each updatable component of "direction" first scaled by
/// the appropriate scale.
static void AddDirection(const Nnet &orig_nnet,
                         const Nnet &direction,
                         const VectorBase<BaseFloat> &scales,
                         Nnet *dest) {
  *dest = orig_nnet;
  dest->AddNnet(scales, direction);
}


static BaseFloat ComputeObjfAndGradient(
    const std::vector<NnetExample> &validation_set,
    const Vector<double> &scale_params,
    const Nnet &orig_nnet,
    const Nnet &direction,
    Vector<double> *gradient) {
  
  Vector<BaseFloat> scale_params_float(scale_params);

  Nnet nnet_combined;
  AddDirection(orig_nnet, direction, scale_params_float, &nnet_combined);
  
  Nnet nnet_gradient(nnet_combined);
  bool is_gradient = true;
  nnet_gradient.SetZero(is_gradient);
  
  // note: "ans" is normalized by the total weight of validation frames.
  int32 batch_size = 1024;
  BaseFloat ans = ComputeNnetGradient(nnet_combined,
                                      validation_set,
                                      batch_size,
                                      &nnet_gradient);

  BaseFloat tot_count = validation_set.size();
  int32 i = 0; // index into scale_params.
  for (int32 j = 0; j < nnet_combined.NumComponents(); j++) {
    const UpdatableComponent *uc_direction =
        dynamic_cast<const UpdatableComponent*>(&(direction.GetComponent(j))),
        *uc_gradient =
        dynamic_cast<const UpdatableComponent*>(&(nnet_gradient.GetComponent(j)));
    if (uc_direction != NULL) {
      BaseFloat dotprod = uc_direction->DotProduct(*uc_gradient) / tot_count;
      (*gradient)(i) = dotprod; 
      i++;
    }
  }
  KALDI_ASSERT(i == scale_params.Dim());
  return ans;
}
                                   

void CombineNnetsA(const NnetCombineAconfig &config,
                   const std::vector<NnetExample> &validation_set,
                   const std::vector<Nnet> &nnets,
                   Nnet *nnet_out) {

  Nnet direction; // the update direction = avg(nnets[1 ... N]) - nnets[0].
  GetUpdateDirection(nnets, &direction);
  
  Vector<double> scale_params(nnets[0].NumUpdatableComponents()); // initial
  // scale on "direction".

  int32 dim = scale_params.Dim();
  KALDI_ASSERT(dim > 0);
  Vector<double> gradient(dim);
  
  double objf, initial_objf, zero_objf;

  // Compute objf at zero; we don't actually need this gradient.
  zero_objf = ComputeObjfAndGradient(validation_set,
                                     scale_params,
                                     nnets[0],
                                     direction,
                                     &gradient);
  KALDI_LOG << "Objective function at old parameters is "
            << zero_objf;
  
  scale_params.Set(1.0); // start optimization from the average of the parameters.

  LbfgsOptions lbfgs_options;
  lbfgs_options.minimize = false; // We're maximizing.
  lbfgs_options.m = dim; // Store the same number of vectors as the dimension
  // itself, so this is BFGS.
  lbfgs_options.first_step_length = config.initial_step;
  
  OptimizeLbfgs<double> lbfgs(scale_params,
                              lbfgs_options);
  
  for (int32 i = 0; i < config.num_bfgs_iters; i++) {    
    scale_params.CopyFromVec(lbfgs.GetProposedValue());
    objf = ComputeObjfAndGradient(validation_set,
                                  scale_params,
                                  nnets[0],
                                  direction,
                                  &gradient);

    KALDI_VLOG(2) << "Iteration " << i << " scale-params = " << scale_params
                  << ", objf = " << objf << ", gradient = " << gradient;
    
    if (i == 0) initial_objf = objf;    
    lbfgs.DoStep(objf, gradient);
  }

  scale_params.CopyFromVec(lbfgs.GetValue(&objf));

  KALDI_LOG << "Combining nnets, after BFGS, validation objf per frame changed from "
            << zero_objf << " (no change), or " << initial_objf << " (default change), "
            << " to " << objf << "; scale factors on update direction are "
            << scale_params;

  BaseFloat objf_change = objf - zero_objf;
  KALDI_ASSERT(objf_change >= 0.0); // This is guaranteed by the L-BFGS code.

  if (objf_change < config.valid_impr_thresh) {
    // We'll overshoot.  To have a smooth transition between the two regimes, if
    // objf_change is close to valid_impr_thresh we don't overshoot as far.
    BaseFloat overshoot = config.overshoot,
        overshoot_max = config.valid_impr_thresh / objf_change; // >= 1.0.
    if (overshoot_max < overshoot) {
      KALDI_LOG << "Limiting overshoot from " << overshoot << " to " << overshoot_max
                << " since the objf-impr " << objf_change << " is close to "
                << "--valid-impr-thresh=" << config.valid_impr_thresh;
      overshoot = overshoot_max;
    }
    KALDI_ASSERT(overshoot < 2.0 && "--valid-impr-thresh must be < 2.0 or "
                 "it will lead to instability.");
    scale_params.Scale(overshoot);

    BaseFloat optimized_objf = objf;
    objf = ComputeObjfAndGradient(validation_set,
                                  scale_params,
                                  nnets[0],
                                  direction,
                                  &gradient);

    KALDI_LOG << "Combining nnets, after overshooting, validation objf changed "
              << "to " << objf << ".  Note: (zero, start, optimized) objfs were "
              << zero_objf << ", " << initial_objf << ", " << optimized_objf;
    if (objf < zero_objf) {
      // Note: this should not happen according to a quadratic approximation, and we
      // expect this branch to be taken only rarely if at all.
      KALDI_WARN << "After overshooting, objf was worse than not updating; not doing the "
                 << "overshoot. ";
     scale_params.Scale(1.0 / overshoot);
    }
  } // Else don't do the "overshoot" stuff.
  
  Vector<BaseFloat> scale_params_float(scale_params);
  // Output to "nnet_out":
  AddDirection(nnets[0], direction, scale_params_float, nnet_out);

  // Now update the neural net learning rates.
  int32 i = 0;
  for (int32 j = 0; j < nnet_out->NumComponents(); j++) {
    UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(nnet_out->GetComponent(j)));
    if (uc != NULL) {
      BaseFloat step_length = scale_params(i), factor = step_length;
      // Our basic rule is to update the learning rate by multiplying it
      // by "step_lenght", but this is subject to certain limits.
      if (factor < config.min_learning_rate_factor)
        factor = config.min_learning_rate_factor;
      if (factor > config.max_learning_rate_factor)
        factor = config.max_learning_rate_factor;
      BaseFloat new_learning_rate = factor * uc->LearningRate();
      if (new_learning_rate < config.min_learning_rate)
        new_learning_rate = config.min_learning_rate;
      KALDI_LOG << "For component " << j << ", step length was " << step_length
                << ", updating learning rate by factor " << factor << ", changing "
                << "learning rate from " << uc->LearningRate() << " to "
                << new_learning_rate;
      uc->SetLearningRate(new_learning_rate);
      i++;
    }
  }
}
 
  
} // namespace nnet2
} // namespace kaldi

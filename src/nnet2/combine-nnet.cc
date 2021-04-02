// nnet2/combine-nnet.cc

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

#include "nnet2/combine-nnet.h"

namespace kaldi {
namespace nnet2 {


// Here, "scale_params" is in blocks, with the first block
// corresponding to nnets[0].
static void CombineNnets(const Vector<BaseFloat> &scale_params,
                         const std::vector<Nnet> &nnets,
                         Nnet *dest) {
  int32 num_nnets = nnets.size();
  KALDI_ASSERT(num_nnets >= 1);
  int32 num_uc = nnets[0].NumUpdatableComponents();
  KALDI_ASSERT(nnets[0].NumUpdatableComponents() >= 1);


  *dest = nnets[0];
  SubVector<BaseFloat> scale_params0(scale_params, 0, num_uc);
  dest->ScaleComponents(scale_params0);
  for (int32 n = 1; n < num_nnets; n++) {
    SubVector<BaseFloat> scale_params_n(scale_params, n * num_uc, num_uc);
    dest->AddNnet(scale_params_n, nnets[n]);
  }
}

/// Returns an integer saying which model to use:
/// either 0 ... num-models - 1 for the best individual model,
/// or (#models) for the average of all of them.
static int32 GetInitialModel(
    const std::vector<NnetExample> &validation_set,
    const std::vector<Nnet> &nnets) {
  int32 minibatch_size = 1024;
  int32 num_nnets = static_cast<int32>(nnets.size());
  KALDI_ASSERT(!nnets.empty());
  BaseFloat tot_frames = validation_set.size();
  int32 best_n = -1;
  BaseFloat best_objf = -std::numeric_limits<BaseFloat>::infinity();
  Vector<BaseFloat> objfs(nnets.size());
  for (int32 n = 0; n < num_nnets; n++) {
    BaseFloat objf = ComputeNnetObjf(nnets[n], validation_set,
                                     minibatch_size) / tot_frames;

    if (n == 0 || objf > best_objf) {
      best_objf = objf;
      best_n = n;
    }
    objfs(n) = objf;
  }
  KALDI_LOG << "Objective functions for the source neural nets are " << objfs;

  int32 num_uc = nnets[0].NumUpdatableComponents();

  { // Now try a version where all the neural nets have the same weight.
    Vector<BaseFloat> scale_params(num_uc * num_nnets);
    scale_params.Set(1.0 / num_nnets);
    Nnet average_nnet;
    CombineNnets(scale_params, nnets, &average_nnet);
    BaseFloat objf = ComputeNnetObjf(average_nnet, validation_set,
                                     minibatch_size) / tot_frames;
    KALDI_LOG << "Objf with all neural nets averaged is " << objf;
    if (objf > best_objf) {
      return num_nnets;
    } else {
      return best_n;
    }
  }
}

// This function chooses from among the neural nets, the one
// which has the best validation set objective function.
static void GetInitialScaleParams(
    const NnetCombineConfig &combine_config,
    const std::vector<NnetExample> &validation_set,
    const std::vector<Nnet> &nnets,
    Vector<double> *scale_params) {

  int32 initial_model = combine_config.initial_model,
      num_nnets = static_cast<int32>(nnets.size());
  if (initial_model < 0 || initial_model > num_nnets)
    initial_model = GetInitialModel(validation_set, nnets);

  KALDI_ASSERT(initial_model >= 0 && initial_model <= num_nnets);
  int32 num_uc = nnets[0].NumUpdatableComponents();

  scale_params->Resize(num_uc * num_nnets);
  if (initial_model < num_nnets) {
    KALDI_LOG << "Initializing with neural net with index " << initial_model;
    // At this point we're using the best of the individual neural nets.
    scale_params->Set(0.0);

    // Set the block of parameters corresponding to the "best" of the
    // source neural nets to
    SubVector<double> best_block(*scale_params, num_uc * initial_model, num_uc);
    best_block.Set(1.0);
  } else { // initial_model == num_nnets
    KALDI_LOG << "Initializing with all neural nets averaged.";
    scale_params->Set(1.0 / num_nnets);
  }
}




static double ComputeObjfAndGradient(
    const std::vector<NnetExample> &validation_set,
    const Vector<double> &scale_params,
    const std::vector<Nnet> &nnets,
    bool debug,
    Vector<double> *gradient) {

  Vector<BaseFloat> scale_params_float(scale_params);

  Nnet nnet_combined;
  CombineNnets(scale_params_float, nnets, &nnet_combined);

  Nnet nnet_gradient(nnet_combined);
  bool is_gradient = true;
  nnet_gradient.SetZero(is_gradient);

  // note: "ans" is normalized by the total weight of validation frames.
  int32 batch_size = 1024;
  double ans = ComputeNnetGradient(nnet_combined,
                                   validation_set,
                                   batch_size,
                                   &nnet_gradient);

  double tot_frames = validation_set.size();
  if (gradient != NULL) {
    int32 i = 0; // index into scale_params.
    for (int32 n = 0; n < static_cast<int32>(nnets.size()); n++) {
      for (int32 j = 0; j < nnet_combined.NumComponents(); j++) {
        const UpdatableComponent *uc =
            dynamic_cast<const UpdatableComponent*>(&(nnets[n].GetComponent(j))),
            *uc_gradient =
            dynamic_cast<const UpdatableComponent*>(&(nnet_gradient.GetComponent(j)));
        if (uc != NULL) {
          double dotprod = uc->DotProduct(*uc_gradient) / tot_frames;
          (*gradient)(i) = dotprod;
          i++;
        }
      }
    }
    KALDI_ASSERT(i == scale_params.Dim());
  }

  if (debug) {
    KALDI_LOG << "Double-checking gradient computation";

    Vector<BaseFloat> manual_gradient(scale_params.Dim());
    for (int32 i = 0; i < scale_params.Dim(); i++) {
      double delta = 1.0e-04, fg = fabs((*gradient)(i));
      if (fg < 1.0e-07) fg = 1.0e-07;
      if (fg * delta < 1.0e-05)
        delta = 1.0e-05 / fg;

      Vector<double> scale_params_temp(scale_params);
      scale_params_temp(i) += delta;
      double new_ans = ComputeObjfAndGradient(validation_set,
                                              scale_params_temp,
                                              nnets,
                                              false,
                                              NULL);
      manual_gradient(i) = (new_ans - ans) / delta;
    }
    KALDI_LOG << "Manually computed gradient is " << manual_gradient;
    KALDI_LOG << "Gradient we computed is " << *gradient;
  }

  return ans;
}


void CombineNnets(const NnetCombineConfig &combine_config,
                  const std::vector<NnetExample> &validation_set,
                  const std::vector<Nnet> &nnets,
                  Nnet *nnet_out) {

  Vector<double> scale_params;

  GetInitialScaleParams(combine_config,
                        validation_set,
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
  lbfgs_options.first_step_impr = combine_config.initial_impr;

  OptimizeLbfgs<double> lbfgs(scale_params,
                              lbfgs_options);

  for (int32 i = 0; i < combine_config.num_bfgs_iters; i++) {
    scale_params.CopyFromVec(lbfgs.GetProposedValue());
    objf = ComputeObjfAndGradient(validation_set,
                                  scale_params,
                                  nnets,
                                  combine_config.test_gradient,
                                  &gradient);

    KALDI_VLOG(2) << "Iteration " << i << " scale-params = " << scale_params
                  << ", objf = " << objf << ", gradient = " << gradient;

    if (i == 0) initial_objf = objf;

    lbfgs.DoStep(objf, gradient);
  }

  scale_params.CopyFromVec(lbfgs.GetValue(&objf));

  Vector<BaseFloat> scale_params_float(scale_params);

  KALDI_LOG << "Combining nnets, validation objf per frame changed from "
            << initial_objf << " to " << objf;

  Matrix<BaseFloat> scale_params_mat(nnets.size(),
                                     nnets[0].NumUpdatableComponents());
  scale_params_mat.CopyRowsFromVec(scale_params_float);
  KALDI_LOG << "Final scale factors are " << scale_params_mat;

  CombineNnets(scale_params_float, nnets, nnet_out);
}


} // namespace nnet2
} // namespace kaldi

// nnet2/combine-nnet-a.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_COMBINE_NNET_A_H_
#define KALDI_NNET2_COMBINE_NNET_A_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "util/parse-options.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {

struct NnetCombineAconfig {
  int32 num_bfgs_iters; // The dimension is small (the number of layers)
  // so we do BFGS.  Note: this num-iters is really the number of function
  // evaluations.
  
  BaseFloat initial_step;

  BaseFloat valid_impr_thresh;
  BaseFloat overshoot;

  BaseFloat min_learning_rate_factor; // 0.5 by default;
  BaseFloat max_learning_rate_factor; // 2.0 by default.
  BaseFloat min_learning_rate; // 0.0001 by default; we don't allow learning rate to go below
  // this, mainly because it would lead to roundoff problems.
  
  NnetCombineAconfig(): num_bfgs_iters(15), initial_step(0.1),
                        valid_impr_thresh(0.5), overshoot(1.8),
                        min_learning_rate_factor(0.5),
                        max_learning_rate_factor(2.0),
                        min_learning_rate(0.0001) { }
  
  void Register(OptionsItf *po) {
    po->Register("num-bfgs-iters", &num_bfgs_iters, "Maximum number of function "
                 "evaluations for BFGS to use when optimizing combination weights");
    po->Register("initial-step", &initial_step, "Parameter in the optimization, "
                 "used to set the initial step length; the default value should be "
                 "suitable.");
    po->Register("num-bfgs-iters", &num_bfgs_iters, "Maximum number of function "
                 "evaluations for BFGS to use when optimizing combination weights");
    po->Register("valid-impr-thresh", &valid_impr_thresh, "Threshold of improvement "
                 "in validation-set objective function for one iteratin; below this, "
                 "we start using the \"overshoot\" mechanism to keep learning rates high.");
    po->Register("overshoot", &overshoot, "Factor by which we overshoot the step "
                 "size obtained by BFGS; only applies when validation set impr is less "
                 "than valid-impr-thresh.");
    po->Register("max-learning-rate-factor", &max_learning_rate_factor,
                 "Maximum factor by which to increase the learning rate for any layer.");
    po->Register("min-learning-rate-factor", &min_learning_rate_factor,
                 "Minimum factor by which to increase the learning rate for any layer.");
    po->Register("min-learning-rate", &min_learning_rate,
                 "Floor on the automatically updated learning rates");
  }  
};

void CombineNnetsA(const NnetCombineAconfig &combine_config,
                   const std::vector<NnetExample> &validation_set,
                   const std::vector<Nnet> &nnets_in,
                   Nnet *nnet_out);
  


} // namespace nnet2
} // namespace kaldi

#endif

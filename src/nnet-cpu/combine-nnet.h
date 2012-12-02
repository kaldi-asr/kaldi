// nnet-cpu/combine-nnet.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_COMBINE_NNET_H_
#define KALDI_NNET_CPU_COMBINE_NNET_H_

#include "nnet-cpu/nnet-update.h"
#include "nnet-cpu/nnet-compute.h"
#include "util/parse-options.h"

namespace kaldi {

/** Configuration class that controls neural net combination, where we combine a
    number of neural nets, trying to find for each layer the optimal weighted
    combination of the different neural-net parameters.
 */
struct NnetCombineConfig {
  int32 num_bfgs_iters; // The dimension is small (e.g. 3 to 5 times the
  // number of neural nets we were given, e.g. 10) so we do
  // BFGS.  We actually implement this as L-BFGS but setting the number of
  // vectors to be the same as the dimension of the space.  Note: this
  // num-iters is in reality the number of function evaluations.
  
  BaseFloat initial_step;
  BaseFloat min_objf_change;
  
  NnetCombineConfig(): num_bfgs_iters(30), initial_step(0.1),
                       min_objf_change(1.0e-05) { }
  
  void Register(ParseOptions *po) {
    po->Register("num-bfgs-iters", &num_bfgs_iters, "Maximum number of function "
                 "evaluations for BFGS to use when optimizing combination weights");
    po->Register("initial-step", &initial_step, "Parameter in the optimization, "
                 "used to set the initial step length; the default value should be "
                 "suitable.");
    po->Register("min-step-length", &min_objf_change, "Objective function "
                 "change (averaged over several iterations), that controls early "
                 "termination of BFGS.");
  }  
};

void CombineNnets(const NnetCombineConfig &combine_config,
                  const std::vector<NnetTrainingExample> &validation_set,
                  const std::vector<Nnet> &nnets_in,
                  Nnet *nnet_out);
  


} // namespace

#endif

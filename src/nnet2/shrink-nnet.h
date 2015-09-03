// nnet2/shrink-nnet.h

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

#ifndef KALDI_NNET2_SHRINK_NNET_H_
#define KALDI_NNET2_SHRINK_NNET_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {

/** Configuration class that controls neural net "shrinkage" which is actually a
    scaling on the parameters of each of the updatable layers.
 */
struct NnetShrinkConfig {
  int32 num_bfgs_iters; // The dimension is small (e.g. 3 to 5) so we do
  // BFGS.  We actually implement this as L-BFGS but setting the number of
  // vectors to be the same as the dimension of the space.  Note: this
  // num-iters is in reality the number of function evaluations.

  BaseFloat initial_step;
  
  NnetShrinkConfig(): num_bfgs_iters(10), initial_step(0.1) { }
  void Register(OptionsItf *opts) {
    opts->Register("num-bfgs-iters", &num_bfgs_iters, "Number of iterations of "
                   "BFGS to use when optimizing shrinkage parameters");
    opts->Register("initial-step", &initial_step, "Parameter in the optimization, "
                   "used to set the initial step length");
  }  
};

void ShrinkNnet(const NnetShrinkConfig &shrink_config,
                const std::vector<NnetExample> &validation_set,
                Nnet *nnet);
  


} // namespace nnet2
} // namespace kaldi

#endif

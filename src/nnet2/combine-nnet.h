// nnet2/combine-nnet.h

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

#ifndef KALDI_NNET2_COMBINE_NNET_H_
#define KALDI_NNET2_COMBINE_NNET_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "util/parse-options.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {

/** Configuration class that controls neural net combination, where we combine a
    number of neural nets, trying to find for each layer the optimal weighted
    combination of the different neural-net parameters.
 */
struct NnetCombineConfig {
  int32 initial_model; // If provided, the index of the initial model to start
  // the optimization from.
  int32 num_bfgs_iters; // The dimension is small (e.g. 3 to 5 times the
  // number of neural nets we were given, e.g. 10) so we do
  // BFGS.  We actually implement this as L-BFGS but setting the number of
  // vectors to be the same as the dimension of the space.  Note: this
  // num-iters is in reality the number of function evaluations.
  
  BaseFloat initial_impr;
  bool test_gradient;
  NnetCombineConfig(): initial_model(-1), num_bfgs_iters(30),
                       initial_impr(0.01),
                       test_gradient(false) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("initial-model", &initial_model, "Specifies where to start the "
                   "optimization from.  If 0 ... #models-1, then specifies the model; "
                   "if #models, then the average of all inputs; otherwise, chosen "
                   "automatically from the previous options.");
    opts->Register("num-bfgs-iters", &num_bfgs_iters, "Maximum number of function "
                   "evaluations for BFGS to use when optimizing combination weights");
    opts->Register("initial-impr", &initial_impr, "Amount of objective-function change "
                   "we aim for on the first iteration.");
    opts->Register("test-gradient", &test_gradient, "If true, activate code that "
                   "tests the gradient is accurate.");
  }  
};

void CombineNnets(const NnetCombineConfig &combine_config,
                  const std::vector<NnetExample> &validation_set,
                  const std::vector<Nnet> &nnets_in,
                  Nnet *nnet_out);
  


} // namespace nnet2
} // namespace kaldi

#endif

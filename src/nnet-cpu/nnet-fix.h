// nnet-cpu/nnet-fix.h

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

#ifndef KALDI_NNET_CPU_NNET_FIX_H_
#define KALDI_NNET_CPU_NNET_FIX_H_

#include "nnet-cpu/nnet-nnet.h"

namespace kaldi {
namespace nnet2 {

/* This header provides a function FixNnet(), and associated config, which
   is responsible for fixing certain pathologies in a neural network during
   training: it identifies neurons whose parameters are getting so large that
   they are maxing out the sigmoid, and scales down those parameters by a
   specified factor.  It also identifies neurons that have the opposite pathology
   that they are just in the linear part of the sigmoid, and it scales up
   their parameters.
*/

struct NnetFixConfig {
  BaseFloat min_average_deriv; // Minimum average derivative that we allow,
  // as a proportion of the maximum derivative of the nonlinearity (1.0 for tanh, 0.25 for sigmoid).
  // If average derivative is less, we scale up the parameters.
  BaseFloat max_average_deriv; // Maximum average derivative that we allow,
  // also expressed relative to the maximum derivative of the nonlinearity.
  BaseFloat parameter_factor; // Factor (>1.0) by which we change the parameters if
  // the exceed the bounds above.

  NnetFixConfig(): min_average_deriv(0.1), max_average_deriv(0.75),
                   parameter_factor(2.0) { }
  void Register(OptionsItf *po) {
    po->Register("min-average-deriv", &min_average_deriv, "Miniumum derivative, "
                 "averaged over the training data, that we allow for a nonlinearity,"
                 "expressed relative to the maximum derivative of the nonlinearity,"
                 "i.e. 1.0 for tanh or 0.25 for sigmoid.");
    po->Register("max-average-deriv", &max_average_deriv, "Maximum derivative, "
                 "averaged over the training data, that we allow for the nonlinearity "
                 "associated with one neuron.");
    po->Register("parameter-factor", &parameter_factor, "Maximum factor by which we change "
                 "the set of parameters associated with a neuron.");
  }
};

void FixNnet(const NnetFixConfig &config, Nnet *nnet);

} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET_CPU_NNET_FIX_H_

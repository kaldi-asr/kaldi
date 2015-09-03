// nnet2/rescale-nnet.h

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

#ifndef KALDI_NNET2_RESCALE_NNET_H_
#define KALDI_NNET2_RESCALE_NNET_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "itf/options-itf.h"

// Neural net rescaling is a rescaling of the parameters of the various layers
// of a neural net, done so as to match certain specified statistics on the
// average derivative of the sigmoid, measured on sample data.  This relates to
// how "saturated" the sigmoid is.

namespace kaldi {
namespace nnet2 {


struct NnetRescaleConfig {
  BaseFloat target_avg_deriv;
  BaseFloat target_first_layer_avg_deriv;
  BaseFloat target_last_layer_avg_deriv;

  // These are relatively unimportant; for now they have no
  // command line options.
  BaseFloat num_iters;
  BaseFloat delta;
  BaseFloat max_change; // maximum change on any one iteration (to
  // ensure stability).
  BaseFloat min_change; // minimum change on any one iteration (controls
  // termination
  
  NnetRescaleConfig(): target_avg_deriv(0.2),
                       target_first_layer_avg_deriv(0.3),
                       target_last_layer_avg_deriv(0.1),
                       num_iters(10),
                       delta(0.01),
                       max_change(0.2), min_change(1.0e-05) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("target-avg-deriv", &target_avg_deriv, "Target average derivative "
                   "for hidden layers that are the not the first or last hidden layer "
                   "(as fraction of maximum derivative of the nonlinearity)");
    opts->Register("target-first-layer-avg-deriv", &target_first_layer_avg_deriv,
                   "Target average derivative for the first hidden layer"
                   "(as fraction of maximum derivative of the nonlinearity)");
    opts->Register("target-last-layer-avg-deriv", &target_last_layer_avg_deriv,
                   "Target average derivative for the last hidden layer, if "
                   "#hid-layers > 1"
                   "(as fraction of maximum derivative of the nonlinearity)");
  }  
};

void RescaleNnet(const NnetRescaleConfig &rescale_config,
                 const std::vector<NnetExample> &examples,
                 Nnet *nnet);
  


} // namespace nnet2
} // namespace kaldi

#endif

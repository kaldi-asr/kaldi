// nnet2/widen-nnet.h

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_WIDEN_NNET_H_
#define KALDI_NNET2_WIDEN_NNET_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {

/** Configuration class that controls neural net "widening", which means increasing
    the dimension of the hidden layers of an already-trained neural net.
 */
struct NnetWidenConfig {
  int32 hidden_layer_dim;
  BaseFloat param_stddev_factor;
  BaseFloat bias_stddev;
  
  NnetWidenConfig(): hidden_layer_dim(-1),
                     param_stddev_factor(1.0),
                     bias_stddev(0.5) { }

  void Register(OptionsItf *opts) {
    opts->Register("hidden-layer-dim", &hidden_layer_dim, "[required option]: "
                   "target dimension of hidden layers");
    opts->Register("param-stddev-factor", &param_stddev_factor, "Factor in "
                   "standard deviation of linear parameters of new part of "
                   "transform (multiply by 1/sqrt of input-dim)");
    opts->Register("bias-stddev", &bias_stddev, "Standard deviation of added "
                   "bias parameters");
  }  
};

/**
   This function widens a neural network by increasing the hidden-layer
   dimensions to the target. */

void WidenNnet(const NnetWidenConfig &widen_config,
               Nnet *nnet);
  


} // namespace nnet2
} // namespace kaldi

#endif

// nnet/nnet-trnopts.h

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

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

#ifndef KALDI_NNET_NNET_TRNOPTS_H_
#define KALDI_NNET_NNET_TRNOPTS_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet1 {


struct NnetTrainOptions {
  // option declaration
  BaseFloat learn_rate;
  BaseFloat momentum;
  BaseFloat l2_penalty;
  BaseFloat l1_penalty;

  // default values
  NnetTrainOptions():
    learn_rate(0.008),
    momentum(0.0),
    l2_penalty(0.0),
    l1_penalty(0.0)
  { }

  // register options
  void Register(OptionsItf *opts) {
    opts->Register("learn-rate", &learn_rate, "Learning rate");
    opts->Register("momentum", &momentum, "Momentum");
    opts->Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    opts->Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");
  }

  // print for debug purposes
  friend std::ostream& operator<<(std::ostream& os, const NnetTrainOptions& opts) {
    os << "NnetTrainOptions : "
       << "learn_rate" << opts.learn_rate << ", "
       << "momentum" << opts.momentum << ", "
       << "l2_penalty" << opts.l2_penalty << ", "
       << "l1_penalty" << opts.l1_penalty;
    return os;
  }
};


struct RbmTrainOptions {
  // option declaration
  BaseFloat learn_rate;
  BaseFloat momentum;
  BaseFloat momentum_max;
  int32 momentum_steps;
  int32 momentum_step_period;
  BaseFloat l2_penalty;

  // default values
  RbmTrainOptions():
    learn_rate(0.4),
    momentum(0.5),
    momentum_max(0.9),
    momentum_steps(40),
    momentum_step_period(500000),
    // 500000 * 40 = 55h of linear increase of momentum
    l2_penalty(0.0002)
  { }

  // register options
  void Register(OptionsItf *opts) {
    opts->Register("learn-rate", &learn_rate, "Learning rate");

    opts->Register("momentum", &momentum,
                   "Initial momentum for linear scheduling");
    opts->Register("momentum-max", &momentum_max,
                   "Final momentum for linear scheduling");
    opts->Register("momentum-steps", &momentum_steps,
                   "Number of steps of linear momentum scheduling");
    opts->Register("momentum-step-period", &momentum_step_period,
                   "Number of datapoints per single momentum increase step");

    opts->Register("l2-penalty", &l2_penalty,
                   "L2 penalty (weight decay, increases mixing-rate)");
  }

  // print for debug purposes
  friend std::ostream& operator<<(std::ostream& os, const RbmTrainOptions& opts) {
    os << "RbmTrainOptions : "
       << "learn_rate" << opts.learn_rate << ", "
       << "momentum" << opts.momentum << ", "
       << "momentum_max" << opts.momentum_max << ", "
       << "momentum_steps" << opts.momentum_steps << ", "
       << "momentum_step_period" << opts.momentum_step_period << ", "
       << "l2_penalty" << opts.l2_penalty;
    return os;
  }
};  // struct RbmTrainOptions

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_TRNOPTS_H_

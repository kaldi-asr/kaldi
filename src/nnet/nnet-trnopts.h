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
#include "util/text-utils.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet1 {


struct NnetTrainOptions {
  // option declaration
  BaseFloat learn_rate;
  BaseFloat momentum;
  BaseFloat l2_penalty;
  BaseFloat l1_penalty;
  bool  freeze_update;
  int32 parallel_level;
  // default values
  NnetTrainOptions() : learn_rate(0.008),
                       momentum(0.0),
                       l2_penalty(0.0),
                       l1_penalty(0.0),
                       freeze_update(false),
                       parallel_level(-1)
                       { }
  // register options
  void Register(OptionsItf *opts) {
    opts->Register("learn-rate", &learn_rate, "Learning rate");
    opts->Register("momentum", &momentum, "Momentum");
    opts->Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    opts->Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");
    opts->Register("freeze-update", &freeze_update, "disable updating the main nnet");
    opts->Register("parallel-level", &parallel_level, "indicate at which hidden layer subnets are attached, indexed from 1");
  }
  // print for debug purposes
  friend std::ostream& operator<<(std::ostream& os, const NnetTrainOptions& opts) {
    os << "RbmTrainOptions : "
       << "learn_rate" << opts.learn_rate << ", "
       << "momentum" << opts.momentum << ", "
       << "l2_penalty" << opts.l2_penalty << ", "
       << "l1_penalty" << opts.l1_penalty << ", "
       << "freeze_update" << opts.freeze_update << ", "
       << "parallel_level" << opts.parallel_level;
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
  RbmTrainOptions() : learn_rate(0.4),
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

    opts->Register("momentum", &momentum, "Initial momentum for linear scheduling");
    opts->Register("momentum-max", &momentum_max, "Final momentum for linear scheduling");
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
};
struct ParallelNnetOptions {
  // int32 parallel_level_sub;
  bool parallel_freeze_update;
  std::string parallel_feature;
  std::string parallel_utt2spk;
  std::string parallel_feature_transform;
  std::string parallel_net;
  std::string parallel_update;
  int32 parallel_nnet_level;
  ParallelNnetOptions() : parallel_freeze_update(false),
                          parallel_feature(""),
                          parallel_utt2spk(""),
                          parallel_feature_transform(""),
                          parallel_net(""),
                          parallel_update(""),
                          parallel_nnet_level(-1)
                          { }
  void Register(OptionsItf *opts) {
    // opts->Register("parallel-level", &parallel_level, "indicate which layer to insert the sub nnet, starting from 1");
    opts->Register("parallel-freeze-update", &parallel_freeze_update, "determine if parameter is updatable");
    opts->Register("parallel-feature", &parallel_feature, "feature reader specifiers, splitted with \';\' mark");
    opts->Register("parallel-utt2spk", &parallel_utt2spk, "utt2spk specifiers, splitted with \';\' mark");
    opts->Register("parallel-feature-transform", &parallel_feature_transform, "feature transform files, splitted with \';\' mark");
    opts->Register("parallel-net", &parallel_net, "input sub net specifiers, splitted with \';\' mark");
    opts->Register("parallel-update", &parallel_update, "updated sub net specifiers, splitted with \';\' mark");
    opts->Register("parallel-nnet-level", &parallel_nnet_level, "indicate at which hidden layer subnets are attached, indexed from 1");
  } 
  friend std::ostream & operator<<(std::ostream& os, const ParallelNnetOptions opts) {
    os << "ParallelOptions: "
       << "parallel_freeze_update " << opts.parallel_freeze_update << ", "
       << "parallel_feature " << opts.parallel_feature << ", "
       << "parallel_utt2spk " << opts.parallel_utt2spk << ", "
       << "parallel_feature_transform " << opts.parallel_feature_transform << ", "
       << "parallel_net " << opts.parallel_net << ", "
       << "parallel_update " << opts.parallel_update << std::endl << ", "
       << "parallel_nnet_level " << opts.parallel_nnet_level << std::endl;
    return os;
  }
};


}//namespace nnet1
}//namespace kaldi

#endif

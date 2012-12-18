// nnet-cpu/nnet-lbfgs.h

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

#ifndef KALDI_NNET_CPU_NNET_LBFGS_H_
#define KALDI_NNET_CPU_NNET_LBFGS_H_

#include "nnet-cpu/nnet-update.h"
#include "nnet-cpu/nnet-compute.h"
#include "util/parse-options.h"

namespace kaldi {

// Note:the num-samples is determined by what you pipe in.
struct NnetLbfgsTrainerConfig {
  PreconditionConfig precondition_config;
  int32 minibatch_size;
  int32 num_iters; // more precisely, the number of function evaluations.
  BaseFloat initial_impr;

  NnetLbfgsTrainerConfig(): minibatch_size(1024), num_iters(20),
                            initial_impr(0.1) { }

  void Register(ParseOptions *po) {
    precondition_config.Register(po);
    po->Register("minibatch-size", &minibatch_size, "Size of minibatches used to "
                 "compute gradient information (only affects speed)");
    po->Register("num-iters", &num_iters, "Number of function evaluations to do "
                 "in L-BFGS");
    po->Register("initial-impr", &initial_impr, "Improvement in objective "
                 "function per frame to aim for on initial iteration.");
  };
};

class NnetLbfgsTrainer {
 public:
  NnetLbfgsTrainer(const NnetLbfgsTrainerConfig &config): config_(config) { }

  void AddTrainingExample(const NnetTrainingExample &eg) { egs_.push_back(eg); }
  
  void Train(Nnet *nnet);
  
 private:
  const NnetLbfgsTrainerConfig &config_;
  std::vector<NnetTrainingExample> egs_;  
};



} // namespace

#endif

// nnet3/nnet-chain-training.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_CHAIN_TRAINING_H_
#define KALDI_NNET3_NNET_CHAIN_TRAINING_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-training.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"

namespace kaldi {
namespace nnet3 {

struct NnetChainTrainingOptions {
  NnetTrainerOptions nnet_config;
  chain::ChainTrainingOptions chain_config;
  bool apply_deriv_weights;
  NnetChainTrainingOptions(): apply_deriv_weights(true) { }

  void Register(OptionsItf *opts) {
    nnet_config.Register(opts);
    chain_config.Register(opts);
    opts->Register("apply-deriv-weights", &apply_deriv_weights,
                   "If true, apply the per-frame derivative weights stored with "
                   "the example");
  }
};


/**
   This class is for single-threaded training of neural nets using the 'chain'
   model.
*/
class NnetChainTrainer {
 public:
  NnetChainTrainer(const NnetChainTrainingOptions &config,
                   const fst::StdVectorFst &den_fst,
                   Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetChainExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  ~NnetChainTrainer();
 private:
  // The internal function for doing one step of conventional SGD training.
  void TrainInternal(const NnetChainExample &eg,
                     const NnetComputation &computation);

  // The internal function for doing one step of backstitch training. Depending
  // on whether is_backstitch_step1 is true, It could be either the first
  // (backward) step, or the second (forward) step of backstitch.
  void TrainInternalBackstitch(const NnetChainExample &eg,
                               const NnetComputation &computation,
                               bool is_backstitch_step1);

  void ProcessOutputs(bool is_backstitch_step2, const NnetChainExample &eg,
                      NnetComputer *computer);

  const NnetChainTrainingOptions opts_;

  chain::DenominatorGraph den_graph_;
  Nnet *nnet_;
  Nnet *delta_nnet_;  // stores the change to the parameters on each training
                      // iteration.
  CachingOptimizingCompiler compiler_;

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.
  int32 num_minibatches_processed_;

  // stats for max-change.
  MaxChangeStats max_change_stats_;

  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher> objf_info_;

  // This value is used in backstitch training when we need to ensure
  // consistent dropout masks.  It's set to a value derived from rand()
  // when the class is initialized.
  int32 srand_seed_;
};


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAIN_TRAINING_H_

// nnet3/nnet-training.h

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

#ifndef KALDI_NNET3_NNET_TRAINING_H_
#define KALDI_NNET3_NNET_TRAINING_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

struct NnetXentTrainerOptions {
  bool zero_component_stats;
  bool store_component_stats;
  int32 print_interval;
  bool debug_computation;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  NnetXentTrainerOptions():
      zero_component_stats(true),
      store_component_stats(false),
      print_interval(100),
      debug_computation(false) { }
  void Register(OptionsItf *opts) {
    opts->Register("store-component-stats", &store_component_stats,
                   "If true, store activations and derivatives for nonlinear "
                   "components during training.");
    opts->Register("zero-component-stats", &zero_component_stats,
                   "If both this and --store-component-stats are true, then "
                   "the component stats are zeroed before training.");
    opts->Register("print-interval", &print_interval, "Interval (measured in "
                   "minibatches) after which we print out objective function "
                   "during training\n");
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);


    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};

// This struct is used in multiple nnet training classes for keeping
// track of objective function values.
struct ObjectiveFunctionInfo {
  int32 current_phase;

  double tot_weight;
  double tot_like;

  double tot_weight_this_phase;
  double tot_like_this_phase;

  ObjectiveFunctionInfo():
      current_phase(0),
      tot_weight(0.0), tot_like(0.0),
      tot_weight_this_phase(0.0), tot_like_this_phase(0.0) { }

  // This function updates the stats and, if the phase has just changed,
  // prints a message indicating progress.  The phase equals
  // minibatch_counter / minibatches_per_phase.
  void UpdateStats(const std::string &output_name,
                   int32 minibatches_per_phase,
                   int32 minibatch_counter,
                   BaseFloat this_minibatch_weight,
                   BaseFloat this_minibatch_tot_like);
  // Prints stats for the current phase.
  void PrintStatsForThisPhase(const std::string &output_name,
                              int32 minibatches_per_phase) const;
  // Prints total stats, and returns true if it was nonzero.
  bool PrintTotalStats(const std::string &output_name) const;
};


/** This class is for single-threaded cross-entropy training of neural nets.
    Note: because we put a "logsoftmax" component in the nnet, the actual
    objective function becomes linear at the output, but the printed messages
    reflect the fact that it's the cross-entropy objective.

    Something that we should do in the future is to make it possible to have
    two different threads, one for the compilation, and one for the computation.
    This would only improve efficiency in the cases where the structure of the
    input example was different each time, which isn't what we expect to see in
    speech-recognition training.  (If the structure is the same each time,
    the CachingOptimizingCompiler notices this and uses the computation from
    last time).
 */
class NnetXentTrainer {
 public:
  NnetXentTrainer(const NnetXentTrainerOptions &config,
                  Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;
 private:
  void ProcessOutputs(const NnetExample &eg,
                      NnetComputer *computer);
  
  const NnetXentTrainerOptions config_;
  Nnet *nnet_;
  CachingOptimizingCompiler compiler_;

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.  
  int32 num_minibatches_processed_;
    
  unordered_map<std::string, ObjectiveFunctionInfo> objf_info_;
  

};




} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_TRAINING_H_

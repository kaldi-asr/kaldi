// nnet3/nnet-discriminative-training.h

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//           2014-2015   Vimal Manohar

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

#ifndef KALDI_NNET3_NNET_DISCRIMINATIVE_TRAINING_H_
#define KALDI_NNET3_NNET_DISCRIMINATIVE_TRAINING_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-discriminative-example.h"
#include "nnet3/nnet-training.h"
#include "nnet3/discriminative-training.h"

namespace kaldi {
namespace nnet3 {

struct NnetDiscriminativeOptions {
  NnetTrainerOptions nnet_config;
  discriminative::DiscriminativeOptions discriminative_config;

  bool apply_deriv_weights;

  NnetDiscriminativeOptions(): apply_deriv_weights(true) { }

  void Register(OptionsItf *opts) {
    nnet_config.Register(opts);
    discriminative_config.Register(opts);
    opts->Register("apply-deriv-weights", &apply_deriv_weights,
                   "If true, apply the per-frame derivative weights stored with "
                   "the example.");
  }
};

// This struct is used in multiple nnet training classes for keeping
// track of objective function values.
// Also see struct AccuracyInfo, in nnet-diagnostics.h.
struct DiscriminativeObjectiveFunctionInfo {
  int32 current_phase;

  discriminative::DiscriminativeObjectiveInfo stats;
  discriminative::DiscriminativeObjectiveInfo stats_this_phase;

  DiscriminativeObjectiveFunctionInfo():
      current_phase(0) { }

  // This function updates the stats and, if the phase has just changed,
  // prints a message indicating progress.  The phase equals
  // minibatch_counter / minibatches_per_phase.  Its only function is to
  // control how frequently we print logging messages.
  void UpdateStats(const std::string &output_name,
                   const std::string &criterion,
                   int32 minibatches_per_phase,
                   int32 minibatch_counter,
                   discriminative::DiscriminativeObjectiveInfo stats);

  // Prints stats for the current phase.
  void PrintStatsForThisPhase(const std::string &output_name,
                              const std::string &criterion,
                              int32 minibatches_per_phase) const;
  // Prints total stats, and returns true if total stats' weight was nonzero.
  bool PrintTotalStats(const std::string &output_name,
                       const std::string &criterion) const;
};


/**
   This class is for single-threaded discriminative training of neural nets 
*/
class NnetDiscriminativeTrainer {
 public:
  NnetDiscriminativeTrainer(const NnetDiscriminativeOptions &config,
                            const TransitionModel &tmodel,
                            const VectorBase<BaseFloat> &priors,
                            Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetDiscriminativeExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  ~NnetDiscriminativeTrainer();
 private:
  void ProcessOutputs(const NnetDiscriminativeExample &eg,
                      NnetComputer *computer);

  const NnetDiscriminativeOptions opts_;

  const TransitionModel &tmodel_;
  CuVector<BaseFloat> log_priors_;
  
  Nnet *nnet_;

  Nnet *delta_nnet_;  // Only used if momentum != 0.0.  nnet representing
                      // accumulated parameter-change (we'd call this
                      // gradient_nnet_, but due to natural-gradient update,
                      // it's better to consider it as a delta-parameter nnet.
  CachingOptimizingCompiler compiler_;

  int32 num_minibatches_processed_;

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.
  unordered_map<std::string, DiscriminativeObjectiveFunctionInfo, StringHasher> objf_info_;
};


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_DISCRIMINATIVE_TRAINING_H_


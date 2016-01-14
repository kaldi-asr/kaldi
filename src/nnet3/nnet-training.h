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

struct NnetTrainerOptions {
  bool zero_component_stats;
  bool store_component_stats;
  int32 print_interval;
  bool debug_computation;
  BaseFloat momentum;
  BaseFloat max_param_change;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  NnetTrainerOptions():
      zero_component_stats(true),
      store_component_stats(true),
      print_interval(100),
      debug_computation(false),
      momentum(0.0),
      max_param_change(2.0) { }
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
    opts->Register("max-param-change", &max_param_change, "The maximum change in"
                   "parameters allowed per minibatch, measured in Frobenius norm "
                   "over the entire model (change will be clipped to this value)");
    opts->Register("momentum", &momentum, "momentum constant to apply during "
                   "training (help stabilize update).  e.g. 0.9.  Note: we "
                   "automatically multiply the learning rate by (1-momenum) "
                   "so that the 'effective' learning rate is the same as "
                   "before (because momentum would normally increase the "
                   "effective learning rate by 1/(1-momentum))");

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
// Also see struct AccuracyInfo, in nnet-diagnostics.h.
struct ObjectiveFunctionInfo {
  int32 current_phase;

  double tot_weight;
  double tot_objf;
  double tot_aux_objf;  // An 'auxiliary' objective function that is optional-
                        // may be used when things like regularization are being
                        // used.

  double tot_weight_this_phase;
  double tot_objf_this_phase;
  double tot_aux_objf_this_phase;

  ObjectiveFunctionInfo():
      current_phase(0),
      tot_weight(0.0), tot_objf(0.0), tot_aux_objf(0.0),
      tot_weight_this_phase(0.0), tot_objf_this_phase(0.0),
      tot_aux_objf_this_phase(0.0) { }

  // This function updates the stats and, if the phase has just changed,
  // prints a message indicating progress.  The phase equals
  // minibatch_counter / minibatches_per_phase.  Its only function is to
  // control how frequently we print logging messages.
  void UpdateStats(const std::string &output_name,
                   int32 minibatches_per_phase,
                   int32 minibatch_counter,
                   BaseFloat this_minibatch_weight,
                   BaseFloat this_minibatch_tot_objf,
                   BaseFloat this_minibatch_tot_aux_objf = 0.0);

  // Prints stats for the current phase.
  void PrintStatsForThisPhase(const std::string &output_name,
                              int32 minibatches_per_phase) const;
  // Prints total stats, and returns true if total stats' weight was nonzero.
  bool PrintTotalStats(const std::string &output_name) const;
};


/** This class is for single-threaded training of neural nets using
    standard objective functions such as cross-entropy (implemented with
    logsoftmax nonlinearity and a linear objective function) and quadratic loss.

    Something that we should do in the future is to make it possible to have
    two different threads, one for the compilation, and one for the computation.
    This would only improve efficiency in the cases where the structure of the
    input example was different each time, which isn't what we expect to see in
    speech-recognition training.  (If the structure is the same each time,
    the CachingOptimizingCompiler notices this and uses the computation from
    last time).
 */
class NnetTrainer {
 public:
  NnetTrainer(const NnetTrainerOptions &config,
              Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  ~NnetTrainer();
 private:
  void ProcessOutputs(const NnetExample &eg,
                      NnetComputer *computer);

  const NnetTrainerOptions config_;
  Nnet *nnet_;
  Nnet *delta_nnet_;  // Only used if momentum != 0.0.  nnet representing
                      // accumulated parameter-change (we'd call this
                      // gradient_nnet_, but due to natural-gradient update,
                      // it's better to consider it as a delta-parameter nnet.
  CachingOptimizingCompiler compiler_;

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.
  int32 num_minibatches_processed_;

  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher> objf_info_;
};

/**
   This function computes the objective function, and if supply_deriv = true,
   supplies its derivative to the NnetComputation object.
   See also the function ComputeAccuracy(), declared in nnet-diagnostics.h.

  @param [in]  supervision   A GeneralMatrix, typically derived from a NnetExample,
                             containing the supervision posteriors or features.
  @param [in] objective_type The objective function type: kLinear = output *
                             supervision, or kQuadratic = -0.5 * (output -
                             supervision)^2.  kLinear is used for softmax
                             objectives; the network contains a LogSoftmax
                             layer which correctly normalizes its output.
  @param [in] output_name    The name of the output node (e.g. "output"), used to
                             look up the output in the NnetComputer object.

  @param [in] supply_deriv   If this is true, this function will compute the
                             derivative of the objective function and supply it
                             to the network using the function
                             NnetComputer::AcceptOutputDeriv
  @param [in,out] computer   The NnetComputer object, from which we get the
                             output using GetOutput and to which we may supply
                             the derivatives using AcceptOutputDeriv.
  @param [out] tot_weight    The total weight of the training examples.  In the
                             kLinear case, this is the sum of the supervision
                             matrix; in the kQuadratic case, it is the number of
                             rows of the supervision matrix.  In order to make
                             it possible to weight samples with quadratic
                             objective functions, we may at some point make it
                             possible for the supervision matrix to have an
                             extra column containing weights.  At the moment,
                             this is not supported.
  @param [out] tot_objf      The total objective function; divide this by the
                             tot_weight to get the normalized objective function.
*/
void ComputeObjectiveFunction(const GeneralMatrix &supervision,
                              ObjectiveType objective_type,
                              const std::string &output_name,
                              bool supply_deriv,
                              NnetComputer *computer,
                              BaseFloat *tot_weight,
                              BaseFloat *tot_objf);



} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_TRAINING_H_

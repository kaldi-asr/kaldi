// nnet3/nnet-diagnostics.h

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

#ifndef KALDI_NNET3_NNET_DIAGNOSTICS_H_
#define KALDI_NNET3_NNET_DIAGNOSTICS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-training.h"

namespace kaldi {
namespace nnet3 {


struct SimpleObjectiveInfo {
  double tot_weight;
  double tot_objective;
  SimpleObjectiveInfo(): tot_weight(0.0),
                         tot_objective(0.0) { }

};


struct NnetComputeProbOptions {
  bool debug_computation;
  bool compute_deriv;
  bool compute_accuracy;
  // note: the component stats, if stored, will be stored in the derivative nnet
  // (c.f. GetDeriv()) if compute_deriv is true; otherwise, you should use the
  // constructor of NnetComputeProb that takes a pointer to the nnet, and the
  // stats will be stored there.
  bool store_component_stats;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  CachingOptimizingCompilerOptions compiler_config;
  NnetComputeProbOptions():
      debug_computation(false),
      compute_deriv(false),
      compute_accuracy(true),
      store_component_stats(false) { }
  void Register(OptionsItf *opts) {
    // compute_deriv is not included in the command line options
    // because it's not relevant for nnet3-compute-prob.
    // store_component_stats is not included in the command line
    // options because it's not relevant for nnet3-compute-prob.
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");
    opts->Register("compute-accuracy", &compute_accuracy, "If true, compute "
                   "accuracy values as well as objective functions");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);
    // register the compiler options with the prefix "compiler".
    ParseOptions compiler_opts("compiler", opts);
    compiler_config.Register(&compiler_opts);
    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};


/** This class is for computing cross-entropy and accuracy values in a neural
    network, for diagnostics.
    Note: because we put a "logsoftmax" component in the nnet, the actual
    objective function becomes linear at the output, but the printed messages
    reflect the fact that it's the cross-entropy objective.
 */
class NnetComputeProb {
 public:
  // does not store a reference to 'config' but does store one to 'nnet'.
  NnetComputeProb(const NnetComputeProbOptions &config,
                  const Nnet &nnet);

  // This version of the constructor may only be called if
  // config.store_component_stats == true and config.compute_deriv == false;
  // it means it will store the component stats in 'nnet'.  In this
  // case you should call ZeroComponentStats(nnet) first if you want
  // the stats to be zeroed first.
  NnetComputeProb(const NnetComputeProbOptions &config,
                  Nnet *nnet);


  // Reset the likelihood stats, and the derivative stats (if computed).
  void Reset();

  // compute objective on one minibatch.
  void Compute(const NnetExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // returns the objective-function info for this output name (e.g. "output"),
  // or NULL if there is no such info.
  const SimpleObjectiveInfo *GetObjective(const std::string &output_name) const;

  // This function returns the total objective over all output nodes recorded here, and
  // outputs to 'tot_weight' the total weight (typically the number of frames)
  // corresponding to it.
  double GetTotalObjective(double *tot_weight) const;

  // if config.compute_deriv == true, returns a reference to the
  // computed derivative.  Otherwise crashes.
  const Nnet &GetDeriv() const;

  ~NnetComputeProb();
 private:
  void ProcessOutputs(const NnetExample &eg,
                      NnetComputer *computer);

  NnetComputeProbOptions config_;
  const Nnet &nnet_;

  bool deriv_nnet_owned_;
  Nnet *deriv_nnet_;
  CachingOptimizingCompiler compiler_;

  // this is only for diagnostics.
  int32 num_minibatches_processed_;

  unordered_map<std::string, SimpleObjectiveInfo, StringHasher> objf_info_;

  unordered_map<std::string, SimpleObjectiveInfo, StringHasher> accuracy_info_;
};


/**
   This function computes the frame accuracy for this minibatch.  It interprets
   the supervision information in "supervision" as labels or soft labels; it
   picks the maximum element in each row and treats that as the label for
   purposes of computing the accuracy (in situations where you would care about
   the accuracy, there will normally be just one nonzero label).  The
   hypothesized labels are computed by taking the neural net output (supplied as
   a CuMatrix), and finding the maximum element in each row.
   See also the function ComputeObjectiveFunction, declared in nnet-training.h.

   @param [in] supervision  The supervision information (no elements may be
                     negative); only the maximum in each row matters (although
                     we expect that usually there will be just one nonzero
                     element in each row); and the sum of each row is
                     interpreted as a weighting factor (although we expect that
                     this sum will usually be one).

  @param [in] nnet_output   The neural net output must have the same dimensions
                     as the supervision.  Only the index of the maximum value in
                     each row matters.  Ties will be broken in an unspecified
                     way.
   @param [out] tot_weight  The sum of the values in the supervision matrix
   @param [out] tot_accuracy  The total accuracy, equal to the sum over all row
                     indexes r such that the maximum column index of row r of
                     supervision and nnet_output is the same, of the sum of the
                     r'th row of supervision (i.e. the row's weight).

*/
void ComputeAccuracy(const GeneralMatrix &supervision,
                     const CuMatrixBase<BaseFloat> &nnet_output,
                     BaseFloat *tot_weight,
                     BaseFloat *tot_accuracy);


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_DIAGNOSTICS_H_

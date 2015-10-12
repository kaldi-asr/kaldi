// nnet3/nnet-cctc-diagnostics.h

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

#ifndef KALDI_NNET3_NNET_CCTC_DIAGNOSTICS_H_
#define KALDI_NNET3_NNET_CCTC_DIAGNOSTICS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-cctc-example.h"
#include "nnet3/nnet-diagnostics.h"
#include "nnet3/nnet-training.h"

namespace kaldi {
namespace nnet3 {

struct NnetCctcComputeProbOptions {
  bool debug_computation;
  bool compute_deriv;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  // cctc_training_opts contains normalizing_weight,
  // which affects the objective function as well
  // as the derivatives.
  ctc::CctcTrainingOptions cctc_training_config;
  NnetCctcComputeProbOptions():
      debug_computation(false),
      compute_deriv(false) { }
  void Register(OptionsItf *opts) {
    // compute_deriv is not included in the command line options
    // because it's not relevant for nnet3-ctc-compute-prob.
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");
    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);
    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
    cctc_training_config.Register(opts);
  }
};


/** This class is for computing objective-function values in a nnet3+ctc
    setup, for diagnostics.
 */
class NnetCctcComputeProb {
 public:
  // does not store a reference to 'config' but does store one to 'nnet'.
  NnetCctcComputeProb(const NnetCctcComputeProbOptions &config,
                      const ctc::CctcTransitionModel &trans_model,
                      const Nnet &nnet);

  // Reset the likelihood stats, and the derivative stats (if computed).
  void Reset();

  // compute objective on one minibatch.
  void Compute(const NnetCctcExample &cctc_eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;
  
  // returns the objective-function info for this output name (e.g. "output"),
  // or NULL if there is no such info.
  const SimpleObjectiveInfo *GetObjective(const std::string &output_name) const;

  // if config.compute_deriv == true, returns a reference to the
  // computed derivative.  Otherwise crashes.
  const Nnet &GetDeriv() const;

  ~NnetCctcComputeProb();
 private:
  void ProcessOutputs(const NnetCctcExample &cctc_eg,
                      NnetComputer *computer);

  NnetCctcComputeProbOptions config_;
  const ctc::CctcTransitionModel trans_model_;
  CuMatrix<BaseFloat> cu_weights_;  // weights derived from trans_model_.
  const Nnet &nnet_;

  Nnet *deriv_nnet_;
  CachingOptimizingCompiler compiler_;

  // this is only for diagnostics.
  int32 num_minibatches_processed_;

  unordered_map<std::string, SimpleObjectiveInfo, StringHasher> objf_info_;

};




} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CCTC_DIAGNOSTICS_H_

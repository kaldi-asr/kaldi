// nnet3/nnet-discriminative-diagnostics.h

// Copyright    2012-2015  Johns Hopkins University (author: Daniel Povey)
// Copyright    2014-2015  Vimal Manohar

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

#ifndef KALDI_NNET3_NNET_DISCRIMINATIVE_DIAGNOSTICS_H_
#define KALDI_NNET3_NNET_DISCRIMINATIVE_DIAGNOSTICS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-discriminative-example.h"
#include "nnet3/nnet-diagnostics.h"
#include "nnet3/discriminative-training.h"

namespace kaldi {
namespace nnet3 {

/** This class is for computing objective-function values in a nnet3 
    discriminative training, for diagnostics.  It also supports computing model
    derivatives.
 */
class NnetDiscriminativeComputeObjf {
 public:
  // does not store a reference to 'config' but does store one to 'nnet'.
  NnetDiscriminativeComputeObjf(const NnetComputeProbOptions &nnet_config,
      const discriminative::DiscriminativeOptions &discriminative_config,
      const TransitionModel &tmodel,
      const VectorBase<BaseFloat> &priors,
      const Nnet &nnet);

  // Reset the likelihood stats, and the derivative stats (if computed).
  void Reset();

  // compute objective on one minibatch.
  void Compute(const NnetDiscriminativeExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // returns the objective-function info for this output name (e.g. "output"),
  // or NULL if there is no such info.
  const discriminative::DiscriminativeObjectiveInfo *GetObjective(
      const std::string &output_name) const;

  // if config.compute_deriv == true, returns a reference to the
  // computed derivative.  Otherwise crashes.
  const Nnet &GetDeriv() const;
  
  ~NnetDiscriminativeComputeObjf();
 private:
  void ProcessOutputs(const NnetDiscriminativeExample &eg,
                      NnetComputer *computer);

  NnetComputeProbOptions nnet_config_;

  discriminative::DiscriminativeOptions discriminative_config_;
  const TransitionModel &tmodel_;
  CuVector<BaseFloat> log_priors_;
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;
  Nnet *deriv_nnet_;
  int32 num_minibatches_processed_;  // this is only for diagnostics

  unordered_map<std::string, discriminative::DiscriminativeObjectiveInfo, StringHasher> objf_info_;
};

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_DISCRIMINATIVE_DIAGNOSTICS_H_


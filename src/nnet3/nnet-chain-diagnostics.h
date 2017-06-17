// nnet3/nnet-chain-diagnostics.h

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

#ifndef KALDI_NNET3_NNET_CHAIN_DIAGNOSTICS_H_
#define KALDI_NNET3_NNET_CHAIN_DIAGNOSTICS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-diagnostics.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"

namespace kaldi {
namespace nnet3 {


struct ChainObjectiveInfo {
  double tot_weight;
  double tot_like;
  double tot_l2_term;
  ChainObjectiveInfo(): tot_weight(0.0),
                        tot_like(0.0),
                        tot_l2_term(0.0) { }
};


/** This class is for computing objective-function values in a nnet3+chain
    setup, for diagnostics.  It also supports computing model derivatives.
    Note: if the --xent-regularization option is nonzero, the cross-entropy
    objective will be computed, and displayed when you call PrintTotalStats(),
    but it will not contribute to model derivatives (there is no code to compute
    the regularized objective function, and anyway it's not clear that we really
    need this regularization in the combination phase).
 */
class NnetChainComputeProb {
 public:
  // does not store a reference to 'config' but does store one to 'nnet'.
  NnetChainComputeProb(const NnetComputeProbOptions &nnet_config,
                       const chain::ChainTrainingOptions &chain_config,
                       const fst::StdVectorFst &den_fst,
                       const Nnet &nnet);

  // This version of the constructor may only be called if
  // nnet_config.store_component_stats == true and nnet_config.compute_deriv ==
  // false; it means it will store the component stats in 'nnet'.  In this case
  // you should call ZeroComponentStats(nnet) first if you want the stats to be
  // zeroed first.
  NnetChainComputeProb(const NnetComputeProbOptions &nnet_config,
                       const chain::ChainTrainingOptions &chain_config,
                       const fst::StdVectorFst &den_fst,
                       Nnet *nnet);


  // Reset the likelihood stats, and the derivative stats (if computed).
  void Reset();

  // compute objective on one minibatch.
  void Compute(const NnetChainExample &chain_eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // returns the objective-function info for this output name (e.g. "output"),
  // or NULL if there is no such info.
  const ChainObjectiveInfo *GetObjective(const std::string &output_name) const;

  // if config.compute_deriv == true, returns a reference to the
  // computed derivative.  Otherwise crashes.
  const Nnet &GetDeriv() const;

  ~NnetChainComputeProb();
 private:
  void ProcessOutputs(const NnetChainExample &chain_eg,
                      NnetComputer *computer);

  NnetComputeProbOptions nnet_config_;
  chain::ChainTrainingOptions chain_config_;
  chain::DenominatorGraph den_graph_;
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;
  bool deriv_nnet_owned_;
  Nnet *deriv_nnet_;
  int32 num_minibatches_processed_;  // this is only for diagnostics

  unordered_map<std::string, ChainObjectiveInfo, StringHasher> objf_info_;

};

/// This function zeros the stored component-level stats in the nnet using
/// ZeroComponentStats(), then recomputes them with the supplied egs.  It
/// affects batch-norm, for instance.  See also the version of RecomputeStats
/// declared in nnet-utils.h.
void RecomputeStats(const std::vector<NnetChainExample> &egs,
                    const chain::ChainTrainingOptions &chain_config,
                    const fst::StdVectorFst &den_fst,
                    Nnet *nnet);



} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CHAIN_DIAGNOSTICS_H_

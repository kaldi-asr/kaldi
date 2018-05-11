// xvector/nnet-xvector-diagnostics.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)
// Copyright    2016  Pegah Ghahremani

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

#ifndef KALDI_XVECTOR_NNET_XVECTOR_DIAGNOSTICS_H_
#define KALDI_XVECTOR_NNET_XVECTOR_DIAGNOSTICS_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-training.h"
#include "xvector/nnet-xvector-training.h"
#include "xvector/xvector.h"

namespace kaldi {
namespace nnet3 {



/** This class is for computing cross-entropy values in a neural
    network with xvector as output and unsupervised objective, for diagnostics.
    Note: because we put a "logsoftmax" component in the nnet, the actual
    objective function becomes linear at the output, but the printed messages
    reflect the fact that it's the cross-entropy objective.

    TODO: In future we plan to check that the same values are returned whether
    we run the computation with or without optimization.
 */
class NnetXvectorComputeProb {
 public:
  // does not store a reference to 'config' but does store one to 'nnet'.
  NnetXvectorComputeProb(const NnetComputeProbOptions &config,
                  const Nnet &nnet);

  // Reset the likelihood stats, and the derivative stats (if computed).
  void Reset();

  // compute objective on one minibatch.
  void Compute(const NnetExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;


  // returns the objective-function info for this output name (e.g. "output"),
  // or NULL if there is no such info.
  const SimpleObjectiveInfo *GetObjective(const std::string &output_name) const;

  // if config.compute_deriv == true, returns a reference to the
  // computed derivative.  Otherwise crashes.
  const Nnet &GetDeriv() const;

  ~NnetXvectorComputeProb();
 private:
  void ProcessOutputs(NnetComputer *computer);
  // Computes the accuracy for this minibatch.
  void ComputeAccuracy(const CuMatrixBase<BaseFloat> &raw_scores,
                       BaseFloat *tot_accuracy_out);
  NnetComputeProbOptions config_;
  const Nnet &nnet_;

  Nnet *deriv_nnet_;
  CachingOptimizingCompiler compiler_;

  // this is only for diagnostics.
  int32 num_minibatches_processed_;

  unordered_map<std::string, SimpleObjectiveInfo, StringHasher> objf_info_;
  unordered_map<std::string, SimpleObjectiveInfo, StringHasher> acc_info_;

};


} // namespace nnet3
} // namespace kaldi

#endif // KALDI_XVECTOR_NNET_XVECTOR_DIAGNOSTICS_H_

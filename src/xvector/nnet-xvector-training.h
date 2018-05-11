// xvector/nnet-xvector-training.h

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)
//              2016  Xiaohui Zhang
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

#ifndef KALDI_XVECTOR_NNET_XVECTOR_TRAINING_H_
#define KALDI_XVECTOR_NNET_XVECTOR_TRAINING_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-example-utils.h"
#include "xvector/xvector.h"
#include "nnet3/nnet-training.h"

namespace kaldi {
namespace nnet3 {


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
class NnetXvectorTrainer {
 public:
  NnetXvectorTrainer(const NnetTrainerOptions &config,
              Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  ~NnetXvectorTrainer();
 private:
  void ProcessOutputs(NnetComputer *computer);

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



void GetComputationRequestXvector(const Nnet &nnet,
                                  const NnetExample &eg,
                                  bool need_model_derivative,
                                  bool store_component_stats,
                                  ComputationRequest *request);
} // namespace nnet3
} // namespace kaldi

#endif //

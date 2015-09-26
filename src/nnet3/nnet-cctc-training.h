// nnet3/nnet-cctc-training.h

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

#ifndef KALDI_NNET3_NNET_CCTC_TRAINING_H_
#define KALDI_NNET3_NNET_CCTC_TRAINING_H_

#include "nnet3/nnet-example.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-cctc-example.h"
#include "nnet3/nnet-training.h"

namespace kaldi {
namespace nnet3 {

struct NnetCctcTrainerOptions: public NnetTrainerOptions {
  ctc::CctcTrainingOptions cctc_training_config;

  void Register(OptionsItf *opts) {
    NnetTrainerOptions::Register(opts);
    cctc_training_config.Register(opts);
  }
};



/** This class is for single-threaded training of neural nets using
    CCTC.
*/
class NnetCctcTrainer {
 public:
  NnetCctcTrainer(const NnetCctcTrainerOptions &config,
                  const ctc::CctcTransitionModel &trans_model,
                  Nnet *nnet);

  // train on one minibatch.
  void Train(const NnetCctcExample &eg);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  ~NnetCctcTrainer();
 private:
  void ProcessOutputs(const NnetCctcExample &eg,
                      NnetComputer *computer);

  const NnetCctcTrainerOptions config_;
  const ctc::CctcTransitionModel &trans_model_;
  CuMatrix<BaseFloat> cu_weights_;  // derived from trans_model_.
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

#endif // KALDI_NNET3_NNET_CCTC_TRAINING_H_

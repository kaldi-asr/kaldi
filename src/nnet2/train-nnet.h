// nnet2/train-nnet.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_TRAIN_NNET_H_
#define KALDI_NNET2_TRAIN_NNET_H_

#include "nnet2/nnet-update.h"
#include "nnet2/nnet-compute.h"
#include "itf/options-itf.h"

namespace kaldi {
namespace nnet2 {


struct NnetSimpleTrainerConfig {
  int32 minibatch_size;
  int32 minibatches_per_phase;
  
  NnetSimpleTrainerConfig(): minibatch_size(500),
                             minibatches_per_phase(50) { }
  
  void Register (OptionsItf *po) {
    po->Register("minibatch-size", &minibatch_size,
                 "Number of samples per minibatch of training data.");
    po->Register("minibatches-per-phase", &minibatches_per_phase,
                 "Number of minibatches to wait before printing training-set "
                 "objective.");
  }  
};


// Class NnetSimpleTrainer doesn't do much apart from batching up the
// input into minibatches and giving it to the neural net code 
// to call Update(), which will typically do stochastic gradient
// descent.  It also reports training-set objective-function values.
// It takes in the training examples through the call
// "TrainOnExample()".
class NnetSimpleTrainer {
 public:
  NnetSimpleTrainer(const NnetSimpleTrainerConfig &config,
                    Nnet *nnet);
  
  /// TrainOnExample will take the example and add it to a buffer;
  /// if we've reached the minibatch size it will do the training.
  void TrainOnExample(const NnetExample &value);

  ~NnetSimpleTrainer();
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(NnetSimpleTrainer);
  
  void TrainOneMinibatch();
  
  // The following function is called by TrainOneMinibatch() when we enter a new
  // phase.  A phase is just a certain number of epochs, and now matters only
  // for diagnostics (originally it meant something more).
  void BeginNewPhase(bool first_time);
  
  // Things we were given in the initializer:
  NnetSimpleTrainerConfig config_;

  Nnet *nnet_; // the nnet we're training.

  // State information:
  int32 num_phases_;
  int32 minibatches_seen_this_phase_;
  std::vector<NnetExample> buffer_;

  double logprob_this_phase_; // Needed for accumulating train log-prob on each phase.
  double weight_this_phase_; // count corresponding to the above.
  
  double logprob_total_;
  double weight_total_;
};



} // namespace nnet2
} // namespace kaldi

#endif

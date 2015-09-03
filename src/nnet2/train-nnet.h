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
  
  void Register (OptionsItf *opts) {
    opts->Register("minibatch-size", &minibatch_size,
                   "Number of samples per minibatch of training data.");
    opts->Register("minibatches-per-phase", &minibatches_per_phase,
                   "Number of minibatches to wait before printing training-set "
                   "objective.");
  }  
};


/// Train on all the examples it can read from the reader.  This does training
/// in a single thread, but it uses a separate thread to read in the examples
/// and format the input data on the CPU; this saves us time when using GPUs.
/// Returns the number of examples processed.
/// Outputs to tot_weight and tot_logprob_per_frame, if non-NULL, the total
/// weight of the examples (typically equal to the number of examples) and the
/// total logprob objective function.
int64 TrainNnetSimple(const NnetSimpleTrainerConfig &config,
                      Nnet *nnet,
                      SequentialNnetExampleReader *reader,
                      double *tot_weight = NULL,
                      double *tot_logprob = NULL);

} // namespace nnet2
} // namespace kaldi

#endif

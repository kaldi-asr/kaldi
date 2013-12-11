// nnet2/nnet-limit-rank.h

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

#ifndef KALDI_NNET2_NNET_LIMIT_RANK_H_
#define KALDI_NNET2_NNET_LIMIT_RANK_H_

#include "nnet2/nnet-nnet.h"
#include "util/table-types.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-thread.h"
#include "nnet2/nnet-update.h"

namespace kaldi {
namespace nnet2 {

struct NnetLimitRankOpts {
  int32 num_threads;
  BaseFloat parameter_proportion;
  
  NnetLimitRankOpts(): num_threads(1), parameter_proportion(0.75) { }

  void Register(OptionsItf *po) {
    po->Register("num-threads", &num_threads, "Number of threads used for "
                 "rank-limiting operation; note, will never use more than "
                 "#layers.");
    po->Register("parameter-proportion", &parameter_proportion, "Proportion of "
                 "dimension of each transform to limit the rank to.");
  }  
};


/// This function limits the rank of each affine transform in the
/// neural net, by zeroing out the smallest singular values.  The number of
/// singular values to zero out is determined on a layer by layer basis, using
/// "parameter_proportion" to set the proportion of parameters to remove.
void LimitRankParallel(const NnetLimitRankOpts &opts,
                       Nnet *nnet);


/// Also see the function LimitRankOfLastLayer in class Nnet.                            


} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_LIMIT_RANK_H_

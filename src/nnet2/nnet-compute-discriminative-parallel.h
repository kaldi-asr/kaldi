// nnet2/nnet-compute-discriminative-parallel.h

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_PARALLEL_H_
#define KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_PARALLEL_H_

#include "nnet2/am-nnet.h"
#include "nnet2/nnet-example.h"
#include "hmm/transition-model.h"
#include "nnet2/nnet-compute-discriminative.h"

namespace kaldi {
namespace nnet2 {

/* This header provides a multi-threaded version of the discriminative training
   code (this is for a CPU-based, instead of GPU-based, setup).
   Note: we expect that "nnet_to_update" will be the same as "&(am_nnet.GetNnet())"
*/

void NnetDiscriminativeUpdateParallel(
    const AmNnet &am_nnet,
    const TransitionModel &tmodel,
    const NnetDiscriminativeUpdateOptions &opts,
    int32 num_threads,
    SequentialDiscriminativeNnetExampleReader *example_reader,
    Nnet *nnet_to_update,
    NnetDiscriminativeStats *stats);


} // namespace nnet2
} // namespace kaldi

#endif //  KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_PARALLEL_H_

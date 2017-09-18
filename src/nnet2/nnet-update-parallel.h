// nnet2/nnet-update-parallel.h

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

#ifndef KALDI_NNET2_NNET_UPDATE_PARALLEL_H_
#define KALDI_NNET2_NNET_UPDATE_PARALLEL_H_

#include "nnet2/nnet-nnet.h"
#include "util/table-types.h"
#include "util/kaldi-semaphore.h"
#include "util/kaldi-thread.h"
#include "itf/options-itf.h"
#include "nnet2/nnet-update.h"

namespace kaldi {
namespace nnet2 {


/// This function is similar to "DoBackprop" in nnet-update.h
/// This function computes the objective function and either updates the model
/// or computes parameter gradients.  It returns the cross-entropy objective
/// function summed over all samples, weighted, and the total weight of
/// the samples (typically the same as the #frames) into total_weight.
/// It is mostly a wrapper for
/// a class NnetUpdater that's defined in nnet-update.cc, but we
/// don't want to expose that complexity at this level.
/// Note: this function 
/// If &nnet == nnet_to_update, it assumes we're doing SGD and does
/// something like Hogwild; otherwise it assumes we're computing a
/// gradient and it sums up the gradients.
/// The return value is the total log-prob summed over the #frames. It also
/// outputs the #frames into "num_frames".
double DoBackpropParallel(const Nnet &nnet,
                          int32 minibatch_size,
                          SequentialNnetExampleReader *example_reader,
                          double *tot_weight,
                          Nnet *nnet_to_update);


/// This version of DoBackpropParallel takes a vector of examples, and will
/// typically be used to compute the exact gradient. 
double DoBackpropParallel(const Nnet &nnet,
                          int32 minibatch_size,
                          int32 num_threads,
                          const std::vector<NnetExample> &examples,
                          double *num_frames,
                          Nnet *nnet_to_update);



/// This is basically to clarify the fact that DoBackpropParallel will
/// also work with nnet_to_update == NULL, and will compute the objf.
/// Both versions of the function will support it, but this
/// version (that takes a vector) is currently the only one we need
/// to do this with.
inline double ComputeNnetObjfParallel(
    const Nnet &nnet,
    int32 minibatch_size,
    int32 num_threads,
    const std::vector<NnetExample> &examples,
    double *num_frames) {
  return DoBackpropParallel(nnet, minibatch_size, num_threads,
                            examples, num_frames, NULL);
}





} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_UPDATE_PARALLEL_H_

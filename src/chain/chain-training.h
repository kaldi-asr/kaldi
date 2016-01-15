// chain/chain-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_TRAINING_H_
#define KALDI_CHAIN_CHAIN_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-den-graph.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {


struct ChainTrainingOptions {
  // Currently empty.

  ChainTrainingOptions() { }

  void Register(OptionsItf *opts) {
  }
};


/**
   This function does both the numerator and denominator parts of the 'chain'
   computation in one call.

   @param [in] opts        Struct containing options
   @param [in] den_graph   The denominator graph, derived from denominator fst.
   @param [in] supervision  The supervision object, containing the supervision
                            paths and constraints on the alignment as an FST
   @param [in] nnet_output  The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).
   @param [out] objf       The [num - den] objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.
   @param [out] nnet_output_deriv  The derivative of the objective function w.r.t.
                           the neural-net output.  Only written to if non-NULL.
                           You don't have to zero this before passing to this function,
                           we zero it internally.
*/
void ComputeChainObjfAndDeriv(const ChainTrainingOptions &opts,
                              const DenominatorGraph &den_graph,
                              const Supervision &supervision,
                              const CuMatrixBase<BaseFloat> &nnet_output,
                              BaseFloat *tot_objf,
                              BaseFloat *tot_weight,
                              CuMatrixBase<BaseFloat> *nnet_output_deriv);


}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_TRAINING_H_


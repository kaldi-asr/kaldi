// nnet2/nnet-compute-discriminative.h

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

#ifndef KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_H_
#define KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_H_

#include "nnet2/am-nnet.h"
#include "nnet2/nnet-example.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace nnet2 {

/* This header provides functionality for doing model updates, and computing
   gradients, using discriminative objective functions (MPFE, SMBR, MMI).
   We use the DiscriminativeNnetExample defined in nnet-example.h.
*/

struct NnetDiscriminativeUpdateOptions {
  std::string criterion; // "mmi" or "mpfe" or "smbr"
  BaseFloat acoustic_scale; // e.g. 0.1
  bool drop_frames; // for MMI, true if we ignore frames where alignment
                    // pdf-id is not in the lattice.
  bool one_silence_class;  // Affects MPE/SMBR>
  BaseFloat boost; // for MMI, boosting factor (would be Boosted MMI)... e.g. 0.1.

  std::string silence_phones_str; // colon-separated list of integer ids of silence phones,
                                  // for MPE/SMBR only.

  NnetDiscriminativeUpdateOptions(): criterion("smbr"), acoustic_scale(0.1),
                                     drop_frames(false),
                                     one_silence_class(false),
                                     boost(0.0) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("criterion", &criterion, "Criterion, 'mmi'|'mpfe'|'smbr', "
                   "determines the objective function to use.  Should match "
                   "option used when we created the examples.");
    opts->Register("acoustic-scale", &acoustic_scale, "Weighting factor to "
                   "apply to acoustic likelihoods.");
    opts->Register("drop-frames", &drop_frames, "For MMI, if true we drop frames "
                   "with no overlap of num and den frames");
    opts->Register("boost", &boost, "Boosting factor for boosted MMI (e.g. 0.1)");
    opts->Register("one-silence-class", &one_silence_class, "If true, newer "
                   "behavior which will tend to reduce insertions.");
    opts->Register("silence-phones", &silence_phones_str,
                   "For MPFE or SMBR, colon-separated list of integer ids of "
                   "silence phones, e.g. 1:2:3");
    
  }
};


struct NnetDiscriminativeStats {
  double tot_t; // total number of frames
  double tot_t_weighted; // total number of frames times weight.
  double tot_num_count; // total count of numerator posterior (should be
                        // identical to denominator-posterior count, so we don't
                        // separately compute that).
  double tot_num_objf;  // for MMI, the (weighted) numerator likelihood; for
                        // SMBR/MPFE, 0.
  double tot_den_objf;  // for MMI, the (weighted) denominator likelihood; for
                        // SMBR/MPFE, the objective function.
  NnetDiscriminativeStats() { std::memset(this, 0, sizeof(*this)); }
  void Print(std::string criterion); // const NnetDiscriminativeUpdateOptions &opts);
  void Add(const NnetDiscriminativeStats &other);
};

/** Does the neural net computation, lattice forward-backward, and backprop,
    for either the MMI, MPFE or SMBR objective functions.
    If nnet_to_update == &(am_nnet.GetNnet()), then this does stochastic
    gradient descent, otherwise (assuming you have called SetZero(true)
    on *nnet_to_update) it will compute the gradient on this data.
    If nnet_to_update_ == NULL, no backpropagation is done.
    
    Note: we ignore any existing acoustic score in the lattice of "eg".

    For display purposes you should normalize the sum of this return value by
    dividing by the sum over the examples, of the number of frames
    (num_ali.size()) times the weight.

    Something you need to be careful with is that the occupation counts and the
    derivative are, following tradition, missing a factor equal to the acoustic
    scale.  So you need to multiply them by that scale if you plan to do
    something like L-BFGS in which you look at both the derivatives and function
    values.  */

void NnetDiscriminativeUpdate(const AmNnet &am_nnet,
                              const TransitionModel &tmodel,
                              const NnetDiscriminativeUpdateOptions &opts,
                              const DiscriminativeNnetExample &eg,
                              Nnet *nnet_to_update,
                              NnetDiscriminativeStats *stats);


} // namespace nnet2
} // namespace kaldi

#endif // KALDI_NNET2_NNET_COMPUTE_DISCRIMINATIVE_H_

// nnet3/nnet-example-functions.h

// Copyright 2015       Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_EXAMPLE_FUNCTIONS_H_
#define KALDI_NNET3_NNET_EXAMPLE_FUNCTIONS_H_

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {

/// Aggregates a bunch of examples into a single example (typically for a
/// minibatch).  Assigns different "n" indexes to different elements of "in".
/// Output is not in compressed format.
void AggregateExamples(const std::vector<NnetExample> &in,
                       NnetExample *out);


/// Selects a particular frame from the supervision information in the examples,
/// discarding any others present.
void SelectSupervisionFrame(int32 t, NnetExample *eg);


/// Shifts the time-index t of everything in the "eg" by adding "t_offset" to
/// all "t" values.  This might be useful in things like clockwork RNNs that are
/// not invariant to time-shifts, to ensure that we see different shifts of each
/// example during training.  "exclude_names" is a vector of names of nnet
/// inputs that we avoid shifting the "t" values of-- normally it will contain
/// just the single string "ivector" because we always leave t=0 for any
/// ivector.
void ShiftTime(int32 t_offset,
               const std::vector<std::string> &exclude_names,
               NnetExample *eg);



} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_EXAMPLE_FUNCTIONS_H_

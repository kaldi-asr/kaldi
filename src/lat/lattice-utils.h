// lat/lattice-utils.h

// Copyright 2009-2011   Saarland University
// Author: Arnab Ghoshal

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


#ifndef KALDI_LAT_LATTICE_UTILS_H_
#define KALDI_LAT_LATTICE_UTILS_H_

#include <vector>

#include "fstext/fstext-lib.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

/// This function iterates over the states of a topologically sorted lattice
/// and counts the time instance corresponding to each state. The times are
/// returned in a vector of integers 'times' which is resized to have a size
/// equal to the number of states in the lattice. The function also returns
/// the maximum time in the lattice.
int32 LatticeStateTimes(const Lattice &lat, std::vector<int32> *times);

/// This function does the forward-backward over lattices and computes the
/// posterior probabilities of the arcs. It returns the total log-probability
/// of the lattice.
BaseFloat LatticeForwardBackward(const Lattice &lat, Posterior *arc_post);

}  // namespace kaldi

#endif  // KALDI_LAT_LATTICE_UTILS_H_

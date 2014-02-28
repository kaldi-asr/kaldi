// lat/minimize-lattice.h

// Copyright 2013        Johns Hopkins University (Author: Daniel Povey)
//           2014        Guoguo Chen

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


#ifndef KALDI_LAT_MINIMIZE_LATTICE_H_
#define KALDI_LAT_MINIMIZE_LATTICE_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace fst {


/// This function minimizes the compact lattice.  It is to be called after
/// determinization (see ./determinize-lattice-pruned.h) and pushing
/// (see ./push-lattice.h).  If the lattice is not determinized and pushed this
/// function will not combine as many states as it could, but it won't crash.
/// Returns true on success, and false if it failed due to topological sorting
/// failing.
template<class Weight, class IntType>
bool MinimizeCompactLattice(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat,
    float delta = fst::kDelta);



}  // namespace fst

#endif  // KALDI_LAT_PUSH_LATTICE_H_

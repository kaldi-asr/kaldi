// lat/kaldi-lattice.h

// Copyright 2009-2011  Microsoft Corporation

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


#ifndef KALDI_LAT_KALDI_LATTICE_H_
#define KALDI_LAT_KALDI_LATTICE_H_

#include "fstext/fstext-lib.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"


namespace kaldi {
// will import some things above...

typedef fst::LatticeWeightTpl<BaseFloat> LatticeWeight;

// careful: kaldi::int32 is not always the same C type as fst::int32
typedef fst::CompactLatticeWeightTpl<LatticeWeight, int32> CompactLatticeWeight;

typedef fst::CompactLatticeWeightCommonDivisorTpl<LatticeWeight, int32>
  CompactLatticeWeightCommonDivisor;

typedef fst::ArcTpl<LatticeWeight> LatticeArc;

typedef fst::ArcTpl<CompactLatticeWeight> CompactLatticeArc;


} // namespace kaldi

#endif  // KALDI_LAT_KALDI_LATTICE_H_

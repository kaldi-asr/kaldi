// lat/minimize-lattice-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


#include "lat/kaldi-lattice.h"
#include "lat/minimize-lattice.h"
#include "lat/push-lattice.h"
#include "fstext/rand-fst.h"


namespace kaldi {
using namespace fst;

CompactLattice *RandDeterministicCompactLattice() {
  RandFstOptions opts;
  opts.acyclic = true;
  while (1) {
    Lattice *fst = fst::RandPairFst<LatticeArc>(opts);
    CompactLattice *cfst = new CompactLattice;
    if (!DeterminizeLattice(*fst, cfst)) {
      delete fst;
      delete cfst;
      KALDI_WARN << "Determinization failed, trying again.";
    } else {
      delete fst;
      return cfst;
    }
  }
}

void TestMinimizeCompactLattice() {
  CompactLattice *clat = RandDeterministicCompactLattice();
  CompactLattice clat2(*clat);
  BaseFloat delta = (Rand() % 2 == 0 ? 1.0 : 1.0e-05);

  // Minimization will only work well on determinized and pushed lattices.
  PushCompactLatticeStrings(&clat2);
  PushCompactLatticeWeights(&clat2);
  
  MinimizeCompactLattice(&clat2, delta);
  KALDI_ASSERT(fst::RandEquivalent(*clat, clat2, 5, delta, Rand(), 10));
  
  delete clat;
}


} // end namespace kaldi

int main() {
  using namespace kaldi;
  using kaldi::int32;
  SetVerboseLevel(4);
  for (int32 i = 0; i < 1000; i++) {
    TestMinimizeCompactLattice();
  }
  KALDI_LOG << "Success.";
}

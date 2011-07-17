// util/kaldi-lattice-test.cc

// Copyright 2009-2011  Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "base/kaldi-math.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {
LatticeWeight RandomLatticeWeight() {
  if(rand() % 3 == 0) {
    return LatticeWeight::Zero();
  } else {
    return LatticeWeight( 100 * RandGauss(), 100 * RandGauss());
  }
}

CompactLatticeWeight RandomCompactLatticeWeight() {
  LatticeWeight w = RandomLatticeWeight();
  if(w == LatticeWeight::Zero()) {
    return CompactLatticeWeight(w, std::vector<int32>());
  } else {
    int32 len = rand() % 4;
    std::vector<int32> str;
    for(int32 i = 0; i < len; i++)
      str.push_back(rand() % 10 + 1);
    return CompactLatticeWeight(w, str);
  }
}

}

namespace kaldi {

main() {
}

}  // end namespace kaldi.



int main() {
  using namespace kaldi;
  return 0;
}


// feat/feature-fbank-test.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek

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


#include <iostream>

#include "feat/feature-fbank.h"
#include "base/kaldi-math.h"
#include "matrix/kaldi-matrix-inl.h"
#include "feat/wave-reader.h"

using namespace kaldi;




int main() {
  try {
    for (int i = 0; i < 5; i++)
      UnitTestFeat();
    std::cout << "Tests succeeded.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}

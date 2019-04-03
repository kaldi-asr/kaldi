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



/**
 */
static void UnitTestSimple() {
  std::cout << "=== UnitTestSimple() ===\n";

  Vector<BaseFloat> v(100000);
  Matrix<BaseFloat> m;

  // init with noise
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = (abs( i * 433024253 ) % 65535) - (65535 / 2);
  }

  std::cout << "<<<=== Just make sure it runs... Nothing is compared\n";
  // the parametrization object
  FbankOptions op;
  // trying to have same opts as baseline.
  op.frame_opts.window_type = "rectangular";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.mel_opts.low_freq = 0.0;
  op.use_energy = true;

  Fbank fbank(op);
  // use default parameters

  // compute fbanks.
  fbank.Compute(v, 1.0, &m);

  // possibly dump
  //   std::cout << "== Output features == \n" << m;
  std::cout << "Test passed :)\n\n";
}



static void UnitTestFeat() {
  UnitTestSimple();
}




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



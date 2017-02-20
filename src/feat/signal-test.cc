// feat/signal-test.cc

// Copyright 2015  Tom Ko

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/signal.h"

namespace kaldi {

void UnitTestBlockConvolution() {
  for (int32 i = 0; i < 5; i++) {
    int32 signal_length = 400000 + Rand() % 40000;
    int32 filter_length = 1000 + Rand() % 100;
    Vector<BaseFloat> signal(signal_length);
    Vector<BaseFloat> filter(filter_length);
    signal.SetRandn();
    filter.SetRandn();
    Vector<BaseFloat> signal_test(signal);
    FFTbasedConvolveSignals(filter, &signal_test);
    FFTbasedBlockConvolveSignals(filter, &signal);
    AssertEqual(signal, signal_test, 0.000001 * signal.Dim());
  }
}

void UnitTestConvolution() {
  for (int32 i = 0; i < 5; i++) {
    int32 signal_length = 4000 + Rand() % 400;
    int32 filter_length = 10 + Rand() % 10;
    Vector<BaseFloat> signal(signal_length);
    Vector<BaseFloat> filter(filter_length);
    signal.SetRandn();
    filter.SetRandn();
    Vector<BaseFloat> signal_test(signal);
    ConvolveSignals(filter, &signal_test);
    FFTbasedBlockConvolveSignals(filter, &signal);
    AssertEqual(signal, signal_test, 0.0001 * signal.Dim());
  }
}
}

int main() {
  using namespace kaldi;
  UnitTestBlockConvolution();
  UnitTestConvolution();
  KALDI_LOG << "Tests succeeded.";

}

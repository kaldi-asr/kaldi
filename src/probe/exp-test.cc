// probe/exp-test.cc

// Copyright 2014  Yandex LLC (Author: Ilya Edrenkin)

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

// Read Makefile.slow_expf. This test must be compiled with -O0.

#include <iostream>
#include <cmath>
#include "base/timer.h"

int main() {
  int test_iter = 300000;

  // Make sure that the CPU bumps its clock to full speed: run the first loop
  // without timing. Then increase the sample iteration count exponentially
  // until the loop takes at least 10ms. We run this loop 1/4 of the number of
  // actual test iterations and call both exp() and expf(), so that the overall
  // test run will take 20 to 60 ms, to ensure a sensibly measurable result.
  for (bool first = true; ; first=false) {
    kaldi::Timer timer;
    for(int i = 0; i < test_iter; i += 4) {
      (void)exp((double)(i & 0x0F));
      (void)expf((double)(i & 0x0F));
    }
    double time = timer.Elapsed();
    if (first) continue;
    if (time > 0.01) break;
    test_iter *= 3;
  }

  kaldi::Timer exp_timer;
  for(int i = 0; i < test_iter; ++i) {
    (void)exp((double)(i & 0x0F));
  }
  double exp_time = exp_timer.Elapsed();

  kaldi::Timer expf_timer;
  for(int i = 0; i < test_iter; ++i) {
    (void)expf((double)(i & 0x0F));
  }
  double expf_time = expf_timer.Elapsed();

  double ratio = expf_time / exp_time;
  if (ratio < 1.1) {
    // Often exp() and expf() perform very similarly, so we will replace expf()
    // by exp() only if there is at least 10% difference.
    return 0;
  }

  std::cerr << ("WARNING: slow expf() detected. expf() is slower than exp() "
                "by the factor of ") << ratio << "\n";
  return 1;
}

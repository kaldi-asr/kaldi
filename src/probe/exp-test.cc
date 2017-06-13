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

#include <iostream>
#include <cmath>
#include "base/timer.h"

#define SAMPLE 100000

int main() { 
  float dummy = 0.0;
  kaldi::Timer exp_timer;
  for(int i = 0; i < SAMPLE; ++i) {
    dummy += exp((double)(i % 10));
  }
  double exp_time = exp_timer.Elapsed();

  kaldi::Timer expf_timer;
  for(int i = 0; i < SAMPLE; ++i) {
    dummy += expf((double)(i % 10));
  }
  double expf_time = expf_timer.Elapsed();
  
  // Often exp() and expf() perform very similarly, 
  // so we will replace expf() by exp() only if there is at least 10% difference 
  if (expf_time < exp_time * 1.1) { 
    return 0;
  } else {
    std::cerr << "exp() time: " << exp_time << std::endl;
    std::cerr << "expf() time: " << expf_time << std::endl;
    return 1;
  }
  
  std::cerr << dummy << std::endl; // No complaint about the unused variable
}

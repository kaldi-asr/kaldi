// nnet/nnet-test.cc

// Copyright 2010  Karel Vesely

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

#include "base/kaldi-common.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"

using namespace kaldi;

// :TODO:
static void UnitTestSomething() {
  KALDI_ERR << "Unimeplemented";
}


static void UnitTestNnet() {
  try {
    UnitTestSomething();
  } catch (const std::exception &e) {
    std::cerr << e.what();
  }
}


int main() {
  UnitTestNnet();
}

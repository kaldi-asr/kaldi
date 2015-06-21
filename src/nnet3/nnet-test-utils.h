// nnet3/nnet-test-utils.h

// Copyright   2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_NNET_TEST_UTILS_H_
#define KALDI_NNET3_NNET_NNET_TEST_UTILS_H_

#include "nnet3/nnet-nnet.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>

namespace kaldi {
namespace nnet3 {

/** @file
    This file contains various routines that are useful in test code.
*/
struct NnetGenerationConfig {
  bool allow_recursion;
  bool allow_clockwork;
  bool allow_multiple_inputs;
  bool allow_multiple_outputs;

  NnetGenerationConfig():
      allow_recursion(true), allow_clockwork(true) { }
};

/// GenerateConfigSequence generates a sequence of at least one config files,
/// output as strings, where the first in the sequence is the initial nnet,
/// and the remaining ones may do things like add layers.
void GenerateConfigSequence(
    const NnetGenerationConfig &opts,
    std::vector<std::string> *configs);



} // namespace nnet3
} // namespace kaldi

#endif

// nnet3/nnet-test-utils.h

// Copyright   2015  Johns Hopkins University (author: Daniel Povey)
// Copyright   2016  Daniel Galvez
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

#ifndef KALDI_NNET3_NNET_TEST_UTILS_H_
#define KALDI_NNET3_NNET_TEST_UTILS_H_

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {

/** @file
    This file contains various routines that are useful in test code.
*/
struct NnetGenerationOptions {
  bool allow_context;
  bool allow_nonlinearity;
  bool allow_recursion;
  bool allow_clockwork;
  bool allow_multiple_inputs;
  bool allow_multiple_outputs;
  bool allow_final_nonlinearity;
  // if set to a value >0, the output-dim of the network
  // will be set to this value.
  int32 output_dim;

  NnetGenerationOptions():
      allow_context(true),
      allow_nonlinearity(true),
      allow_recursion(true),
      allow_clockwork(true),
      allow_multiple_inputs(true),
      allow_multiple_outputs(false),
      allow_final_nonlinearity(true),
      output_dim(-1) { }
};

/** Generates a sequence of at least one config files, output as strings, where
    the first in the sequence is the initial nnet, and the remaining ones may do
    things like add layers.  */
void GenerateConfigSequence(const NnetGenerationOptions &opts,
                            std::vector<std::string> *configs);

/// Generate a config string with a composite component composed only
/// of block affine, repeated affine, and natural gradient repeated affine
/// components.
void GenerateConfigSequenceCompositeBlock(const NnetGenerationOptions &opts,
                                          std::vector<std::string> *configs);

/**  This function computes an example computation request, for testing purposes.
     The "Simple" in the name means that it currently only supports neural nets
     that satisfy IsSimple(nnet) (defined in nnet-utils.h).
     If there are 2 inputs, the "input" will be first, followed by "ivector". */
void ComputeExampleComputationRequestSimple(
    const Nnet &nnet,
    ComputationRequest *request,
    std::vector<Matrix<BaseFloat> > *inputs);

Component *GenerateRandomSimpleComponent();


/** Used for testing that the updatable parameters in two networks are the same.
    May crash if structure differs.  Prints warning and returns false if
    parameters differ.  E.g. set threshold to 1.0e-05 (it's a relative
    threshold, applied per component). */
bool NnetParametersAreIdentical(const Nnet &nnet1,
                                const Nnet &nnet2,
                                BaseFloat threshold);


/** Low-level function that generates an nnet training example.  By "simple" we
    mean there is one output named "output", an input named "input", and
    possibly also an input named "ivector" (this will be assumed absent if
    ivector_dim <= 0).  This function generates exactly "left_context" or
    "right_context" frames of context on the left and right respectively. */
void GenerateSimpleNnetTrainingExample(
    int32 num_supervised_frames,    
    int32 left_context,
    int32 right_context,
    int32 input_dim,
    int32 output_dim,
    int32 ivector_dim,
    NnetExample *example);


/// Returns true if the examples are approximately equal (only intended to be
/// used in testing).
bool ExampleApproxEqual(const NnetExample &eg1,
                        const NnetExample &eg2,
                        BaseFloat delta);

} // namespace nnet3
} // namespace kaldi

#endif

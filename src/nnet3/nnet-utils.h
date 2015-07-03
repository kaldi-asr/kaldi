// nnet3/nnet-utils.h

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

#ifndef KALDI_NNET3_NNET_UTILS_H_
#define KALDI_NNET3_NNET_UTILS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/nnet-descriptor.h"

#include "nnet-computation-graph.h"

namespace kaldi {
namespace nnet3 {


/// \file nnet-utils.h
/// This file contains some miscellaneous functions dealing with class Nnet.

/// Given an nnet and a computation request, this function works out which
/// requested outputs in the computation request are computable; it outputs this
/// information as a vector "is_computable" indexed by the same indexes as
/// request.outputs.
/// It does this by executing some of the early stages of compilation.
void EvaluateComputationRequest(
    const Nnet &nnet,
    const ComputationRequest &request,
    std::vector<std::vector<bool> > *is_computable);


/// returns the number of output nodes of this nnet.
int32 NumOutputNodes(const Nnet &nnet);

/// returns the number of input nodes of this nnet.
int32 NumInputNodes(const Nnet &nnet);


/// This function returns true if the nnet has the following properties:
///  It has one output, called "output".
///  It has an input called "input", and possibly an extra input called
///    "ivector", but no other inputs.
///  There are probably some other properties that we really ought to
///  be checking, and we may add more later on.
bool IsSimpleNnet(const Nnet &nnet);


/// ComputeNnetContext computes the left-context and right-context of a nnet.
/// The nnet must satisfy IsSimpleNnet(nnet).
///
/// It does this by constructing a ComputationRequest with a certain number of inputs
/// available, outputs can be computed..  It does the same after shifting the time
/// index of the output to all values 0, 1, ... n-1, where n is the output
/// of Modulus(nnet).   Then it returns the largest left context and the largest
/// right context that it infers from any of these computation requests.
void ComputeSimpleNnetContext(const Nnet &nnet,
                              int32 *left_context,
                              int32 *right_context);


} // namespace nnet3
} // namespace kaldi

#endif

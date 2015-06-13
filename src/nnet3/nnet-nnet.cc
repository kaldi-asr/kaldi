// nnet3/nnet-nnet.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-nnet.h"

namespace kaldi {
namespace nnet3 {

// returns dimension that this node outputs.
int32 NetworkNode::Dim(const Nnet &nnet) const {
  int32 ans;
  switch (node_type) {
    case kInput: case kDimRange:
      ans = dim;
      break;
    case kDescriptor:
      ans = descriptor.Dim(nnet);
      break;
    case kComponentOutput:
      ans = nnet.GetComponent(u.component_index)->OutputDim();
      break;
    default:
      KALDI_ERR << "Invalid node type.";
  }
  KALDI_ASSERT(ans > 0);
  return ans;
}




} // namespace nnet3
} // namespace kaldi

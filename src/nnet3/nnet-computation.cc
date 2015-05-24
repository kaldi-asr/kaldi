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
#include "nnet3/nnet-computation.h"

namespace kaldi {
namespace nnet3 {

bool ComputationRequest::NeedDerivatives() const {
  if (need_model_derivative)
    return true;
  for (size_t i = 0; i < inputs.size(); i++)
    if (inputs[i].has_deriv)  // derivative requested for this input
      return true;
  return false;
}

NnetComputation::~NnetComputation() {
  for (size_t i = 0; i < component_precomputed_indexes.size(); i++)
    delete component_precomputed_indexes[i];
}

void NnetComputation::ComputeCudaIndexes() {
  indexes_cuda.resize(indexes.size());

  for (size_t i = 0; i < indexes.size(); i++)
    indexes_cuda[i].CopyFromVec(indexes[i]);

  std::vector<bool> need_cuda(indexes_multi.size(), false);
  for (int32 c = 0; c < commands.size(); c++) {
    if (commands[c].command_type == kAddRowRanges) {
      int32 indexes_multi_index = commands[c].arg3;
      KALDI_ASSERT(static_cast<size_t>(indexes_multi_index) < need_cuda.size());
      need_cuda[indexes_multi_index] = true;
    }
  }
  KALDI_ASSERT(sizeof(Int32Pair) == sizeof(std::pair<int32,int32>));
  indexes_multi_cuda.resize(indexes_multi.size());
  for (int32 i = 0; i < indexes_multi.size(); i++) {
    if (need_cuda[i]) {
      const std::vector<std::pair<int32,int32> > *input = &(indexes_multi[i]);
      const std::vector<Int32Pair> *input_cast =
          reinterpret_cast<const std::vector<Int32Pair> *>(input);
      // note: the indexes for CUDA use can't very easily use STL types due to
      // the interface of CUDA being plain C.
      indexes_multi_cuda[i].CopyFromVec(*input_cast);
    }
  }
}

} // namespace nnet3
} // namespace kaldi

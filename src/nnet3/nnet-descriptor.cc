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
#include "nnet3/nnet-descriptor.h"

namespace kaldi {
namespace nnet3 {


void InputDescriptor::GetInputCindexes(
    const Index &index,
    std::vector<Cindex> *required_indexes) const {
  required_indexes->clear();
  std::vector<SumDescriptor>::const_iterator sum_iter = parts.begin(),
      sum_end = parts.end();
  for (; sum_iter != sum_end; ++sum_iter) {
    const SumDescriptor &sum_descriptor = *sum_iter;
    std::vector<ForwardingDescriptor>::const_iterator
        forwarding_iter = sum_descriptor.terms.begin(),
        forwarding_end = sum_descriptor.terms.end();
    for (; forwarding_iter != forwarding_end; ++forwarding_iter) {
      const ForwardindDescriptor &forwarding_desc = *forwarding_iter;
      required_indexes->push_back(forwarding_desc.MapToInput(index));
    }
  }
}

Cindex ForwardingDescriptor::MapToInput(const Index &output) {
  return impl_->MapToInput(output);
}

void SimpleForwardingDescriptor::ComputeDependencies(
    std::vector<int32> *node_indexes) {
  node_indexes->push_back(src_node_);  
}


} // namespace nnet3
} // namespace kaldi

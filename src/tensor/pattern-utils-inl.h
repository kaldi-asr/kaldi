// tensor/pattern-utils-inl.h

//  Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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


// Do not include this header directly; it is only to be included by pattern-utils.h.

#ifndef KALDI_TENSOR_TENSOR_PATTERN_UTILS_INL_H_
#define KALDI_TENSOR_TENSOR_PATTERN_UTILS_INL_H_ 1


namespace kaldi {
namespace tensor {

// See pattern-utils.h for documentation.
inline bool ContainsNegativeStride(const Pattern &pattern) {
  // 2048 is 1 << 11; 11th bit in code is set if code indicates negative stride.
  if (pattern.code >= 0 && (pattern.code | 2048) != 0)
    return true;
  int32 num_axes = pattern.num_axes;
  for (int32 raxis = 0; raxis < num_axes; raxis++)
    if (pattern.strides[raxis] < 0)
      return true;
  return false;
}


}  // namespace tensor
}  // namespace kaldi

#endif KALDI_TENSOR_TENSOR_PATTERN_UTILS_INL_H_

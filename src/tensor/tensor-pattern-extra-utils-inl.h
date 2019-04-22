// tensor/tensor-pattern-extra-utils-inl.h

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

#ifndef KALDI_TENSOR_TENSOR_PATTERN_EXTRA_UTILS_INL_H_
#define KALDI_TENSOR_TENSOR_PATTERN_EXTRA_UTILS_INL_H_ 1

// This file is only to be included by tensor-pattern-extra-utils.h; do not include it
// directly.



namespace kaldi {
namespace tensor {

inline void ComputeMinAndMaxMindex(const TensorPattern *pattern,
                                   int64 *min_mindex,
                                   int64 *max_mindex) {
  KALDI_PARANOID_ASSERT(IsValid(pattern));
  int32 num_axes = pattern.num_axes;
  if (ContainsNegativeStride(pattern.code)) {
    // The if-statement above may be read as "if either pattern.code is -1 or it
    // indicates that `pattern` contains a negative stride.  That is, at this
    // point we know that `pattern` *might* contain a negative stride.
    int64 min_mindex_sum = 0, max_mindex_sum = 0;
    for (int32 raxis = 0; raxis < num_axes; raxis++) {
      int64 prod (pattern.dims[raxis] - 1) *
          static_cast<int64>(pattern.strides[raxis]);
      if (pattern.strides[raxis] > 0) max_mindex_sum += prod;
      else min_mindex_sum += prod;
    }
    *min_mindex = min_mindex_sum;
    *max_mindex = max_mindex_sum;
  } else {
    // This is a faster branch of the code that can assume all strides are
    // positive.
    *min_mindex = 0;
    int64 max_mindex_sum = 0;
    for (int32 raxis = 0; raxis < num_axes; raxis++)
      max_mindex_sum += (pattern.dims[raxis] - 1) *
          static_cast<int64>(pattern.strides[raxis]);
    *max_mindex = max_mindex_sum;
  }
}


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_PATTERN_EXTRA_UTILS_INL_H_

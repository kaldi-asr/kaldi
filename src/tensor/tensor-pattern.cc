// tensor/tensor-pattern.cc

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#include "tensor/tensor-pattern.h"


namespace kaldi {
namespace tensor {

bool TensorPattern::Check() {
  if (num_axes < 0 || num_axes >= KALDI_TENSOR_MAX_DIM)
    return false;
  for (int32 axis = 0; axis < num_axes; axis++) {
    int32 dim = dims[axis], stride = strides[axis];
    // All dims must be positive.  (We have no concept of
    // an empty tensor, you would use NULL, or None, to represent
    // that.
    if (dim <= 0)
      return false;
    // If dim == 1, stride must be zero.  Otherwise, stride must be nonzero.
    if (dim == 1) {
      if (stride != 0) return false;
    } else {
      if (stride == 0) return false;
    }
  }

  {
    // Now check for potential overlap.  We take all the axes with dim != 1 and
    // sort them from greatest to least stride, and check that for each i,
    // abs(strides[i]) >= dims[i+1] * abs(strides[i+1]).
    std::pair<int32, int32> dims_abs_strides [KALDI_TENSOR_MAX_DIM];
    int32 num_nontrivial_axes = 0;
    for (int32 i = 0; i < num_axes; i++) {
      if(dims[i] != 1) {
        dims_abs_strides[num_nontrivial_axes].first = dims[i];
        dims_abs_strides[num_nontrivial_axes].second = std::abs(strides[i]);
        num_nontrivial_axes++;
      }
    }
    // We want to sort on strides from greatest to least, so use '>' not
    // '<' as the comparator.
    std::sort(dims_abs_strides, dims_abs_strides + num_nontrivial_axes,
              [](const std::pair<int32, int32> &a, const std::pair<int32, int32> &b) {
                return a.second > b.second
                    });
    for (int32 i = 0; i < num_nontrivial_axes; i++) {
      // if (abs(strides[i]) < dims[i+1] * abs(strides[i+1])) return false;
      if (dims_abs_strides[i].second <
          dims_abs_strides[i+1).first * dims_abs_strides[i+1).second)
        return false;
    }
  }
  return true;
}


void TensorPatternProperties::UpdateProperties(const TensorPattern &pattern) {
  KALDI_PARANOID_ASSERT(pattern.Check());
  int32 num_axes = pattern.num_axes;
  int64 dim_prod = 1;
  bool c_strides = true;
  // 'element_range' is the distance (in elements) between the
  // first and last elements of the array.
  int64 element_range = 0;
  for (int32 i = num_axes - 1; i >= 0; i--) {
    int32 dim = pattern.dims[i], stride = pattern.strides[i];
    if (dim != 1) {
      if (pattern.strides[i] != dim_prod)
        c_strides = false;
      element_range += std::abs(static_cast<int64>(stride) *
                                static_cast<int64>(dim - 1));
    }
    dim_prod *= dim;
  }
  this->num_elements = dim_prod;
  this->has_c_strides = c_strides;
  if (has_c_strides) {
    KALDI_PARANOID_ASSERT(element_range + 1 == num_elements);
    this->is_contiguous = true;
  } else {
    KALDI_PARANOID_ASSERT(element_range < num_elements);
    this->is_contiguous = (element_range + 1 == num_elements);
  }
}


}  // namespace kaldi
}  // namespace tensor

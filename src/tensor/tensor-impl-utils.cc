// tensor/tensor-impl-utils.cc

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

#include "tensor/tensor-impl-utils.h"


namespace kaldi {
namespace tensor {


void Slice(int32 axis, int32 start, int32 end, TensorImpl *t) {
  int32 num_axes = t->num_axes;
  int32 raxis = (axis >= 0 ? num_axes - 1 - axis : - 1 - axis);
  if (static_cast<uint32>(raxis) >= static_cast<uint32>(num_axes)) {
    KALDI_ERR << "Axis out of range: " << axis << ", num-axes = "
              << num_axes;
  }
  int32 dim = t->dims[raxis], stride = t->strides[raxis];
  if (end <= start || start < 0 || end > dim) {
    KALDI_ERR << "Slice() parameters out of range: start,end = "
              << start << "," << end << ", dim = " << dim;
  }
  AddToPointer(stride * static_cast<int64>(start), t);

  int32 new_dim = end - start;
  t->dims[raxis] = new_dim;
  if (new_dim == 1)
    t->strides[raxis] = 0;
}





}  // namespace kaldi
}  // namespace tensor

// tensor/tensor-impl.h

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

#ifndef KALDI_TENSOR_TENSOR_IMPL_H_
#define KALDI_TENSOR_TENSOR_IMPL_H_ 1

#include "tensor/tensor-common.h"
#include "tensor/tensor-pattern.h"

namespace kaldi {
namespace tensor {


/**
   TensorImpl is the core part of a Tensor, without the wrapping code and
   storage management in Tensor.h.  Most of the core implementation deals
   directly with TensorImpl to avoid the overhead of shared_ptr management
   and the need to deal with accessors and the like, but TensorImpl
   is intended for use in the tensor/ directory, to implement Tensor
   internals, and not for users of this library.
*/
struct TensorImpl {
  TensorPattern pattern;
  DataType dtype;
  Device device;
  void *data{nullptr};

  // Returns true if this TensorImpl is valid, false otherwise.  It is an
  // implied requirement of functions operating on TensorImpl's, that all
  // TensorImpl's that are provided input or input+output arguments to functions
  // must have IsValid() == true.
  bool IsValid() { return pattern.IsValid(true) && data != nullptr; }
};

// Metadata for a Tensor.  It's occasionally convenient to have this
// in a struct (it's the same as a Tensor without the 'data' pointer.
struct TensorMeta {
  TensorPattern pattern;
  DataType dtype;
  Device device;
};


void Compatible(const TensorImpl &a, TensorImpl &b



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_IMPL_H_

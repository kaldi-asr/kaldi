// tensor/tensor.h

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

#ifndef KALDI_TENSOR_TENSOR_UTILS_H_
#define KALDI_TENSOR_TENSOR_UTILS_H_ 1

#include "tensor/tensor.h"

namespace kaldi {
namespace tensor {


/**
   SubTensor is used in our implementation when we want to create temporaries
   that are easier to manipulate outside the Tensor class, and that don't have
   the overhead of managing the std::shared_ptr.  The idea is that a SubTensor
   will be constructed temporarily from a longer-living underlying Tensor.
 */
struct SubTensor {
  TensorPattern pattern;
  DataType dtype;
  Device device;
  void *data;

  // Constructor from Tensor.
  explicit SubTensor(const Tensor &tensor);
};


// Used in checking function arguments, this function will
// crash and print a statck trace if Tensor a and b have different
// Dtype() or different Device().
void CheckDeviceAndDtype(const Tensor &a, const Tensor &b);

// Used in checking function arguments, this function will
// crash and print a statck trace if Tensor a, b and c have different
// Dtype() or different Device().
void CheckDeviceAndDtype(const Tensor &a, const Tensor &b, const Tensor &c);



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_H_

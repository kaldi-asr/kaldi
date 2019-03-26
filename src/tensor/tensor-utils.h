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




// Used in checking function arguments, this function will
// crash and print a statck trace if Tensor a and b have different
// Dtype() or different Device().
void CheckDeviceAndDtype(const TensorImpl &a, const TensorImpl &b);

// Used in checking function arguments, this function will
// crash and print a statck trace if Tensor a, b and c have different
// Dtype() or different Device().
void CheckDeviceAndDtype(const TensorImpl &a, const TensorImpl &b, const TensorImpl &c);


/**
   This function allocates the appropriate storage for the Tensor described
   in 'impl', and sets is 'data' pointer to the allocated memory address.
   It returns the address a newly allocated Storage object which manages
   the memory location; you will probably want to construct a
   std::unique_ptr<Storage> from this so that when it goes out of scope,
   the memory will be freed.

      @param [in,out] impl   The TensorImpl object we are allocating for.
                      Any previous value of impl->data is overwritten.
                      It is required that that the product of dims in
                      impl->pattern be nonzero (i.e. that the pattern
                      is initialized to a valid value), and that its
                      dtype and device values be set.
      @return         Returns a newly allocated Storage object that
                      manages this memory block.  When it is freed,
                      the memory block will be deallocated using a
                      method appropriate for the device.

   This function throws on error.  See also AllocateTensorShared().  This
   function is used by class Tensor, but also by various implementation
   functions (called with TensorImpl) where we need to allocate temporaries.
   We don't construct a full-fledged Tensor because we don't want the
   overhead of managing any shared_ptr's.
 */
Storage *AllocateTensor(TensorImpl *impl);


/**
   This function is as AllocateTensor(), except that the Storage
   object returned is allocated via std::make_shared (which involves
   just one heap allocation, as opposed to two if you constructed
   the shared_ptr from the Storage* pointer).  See the documentation
   for AllocateTensor() for more details.
 */
std::shared_ptr<Storage> AllocateTensorShared(TensorImpl *impl);


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_H_

// tensor/tensor-impl-utils.h

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

#ifndef KALDI_TENSOR_IMPL_UTILS_H_
#define KALDI_TENSOR_IMPL_UTILS_H_ 1

#include "tensor/tensor-impl.h"
#include "tensor/tensor-patterns-utils.h"


/**
   This header contains basic linear-algebra and copying types of operations
   on TensorImpl objects.  See also tensor-impl-nonlinearly
 */

namespace kaldi {
namespace tensor {


// This function returns true if a and b have the same dtype
// and device.  See also Broadcastable().
inline bool Compatible(const TensorImpl &a, const TensorImpl &b);


// This function returns true if a and b have the same dtype
// and device; equivalent to Compatible(a, b) && Compatible(b, c).
inline bool Compatible(const TensorImpl &a, const TensorImpl &b,
                       const TensorImpl &c);


/**
   This function allocates the appropriate storage for the Tensor described
   in 'impl', and sets is 'data' pointer to the allocated memory address.
   It returns the address a newly allocated Storage object which manages
   the memory location; you will probably want to construct a
   std::unique_ptr<Storage> from this so that when it goes out of scope,
   the memory will be freed.

      @param [in,out] impl   The TensorImpl object we are allocating for.
                      Any previous value of impl->data is ignored and
                      overwritten.
                      It is required that that the product of dims in
                      impl->pattern be nonzero (i.e. that the pattern
                      is initialized to a valid value), and that its
                      dtype and device values be set.
      @return         Returns a newly allocated Storage object that
                      manages this memory block.  When this object is deleted,
                      the memory block will be deallocated using a
                      method appropriate for the device.

   This function throws on error.

   See also AllocateTensorDataShared().
 */
Storage *AllocateTensorData(TensorImpl *impl);


/**
   This function is as AllocateTensor(), except that the Storage
   object returned is allocated via std::make_shared (which involves
   just one heap allocation, as opposed to two if you constructed
   the shared_ptr from the Storage* pointer).  See the documentation
   for AllocateTensor() for more details.
 */
std::shared_ptr<Storage> AllocateTensorDataShared(TensorImpl *impl);



/**
   Modifies 't' in-place by inserting an axis with (dim=1,stride=0) at the
   specified position.  Updates the code.

   A negative axis-index i is interpreted (like PyTorch) as (num_axes + 1 - i).

   Showing just the dims in the tensor for some examples:

\verbatim
    Unsqueeze({3,4}, 0)  -> {1,3,4}
    Unsqueeze({3,4}, 1)  -> {3,1,4}
    Unsqueeze({3,4}, 2)  -> {3,4,1}
    Unsqueeze({3,4}, -1)  -> {3,4,1}
    Unsqueeze({3,4}, -2)  -> {3,1,4}
\endverbatim
 */
inline void Unsqueeze(TensorImpl *t, int32 axis) {
  Unsqueeze(&(t->pattern), axis);
}


/**
   Modifies 't' in-place by removing an axis with (dim=1,stride=0) from the
   specified position.  It is an error if 't' did not initially contain
   such an axis.  This function updates the code.  See also the same-named
   function that operates on TensorPattern.

   Showing just the dims in the tensor for an example:

\verbatim
    Squeeze({1,3,4}, 0)  -> {3,4}
    Squeeze({3,1,4}, 1)  -> {3,4}
    Squeeze({3,1,4}, 2)  -> [error]
\endverbatim
 */
inline void Squeeze(TensorImpl *t, int32 axis) {
  Squeeze(&(t->pattern), axis));
}






}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_IMPL_UTILS_H_

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




//  This function moves the 'data' pointer stored in 't' by adding
//  a number of elements equal to 'offset'.  It casts it to the
// type specified in t->dtype so the memory address changes by
// the right amount.
inline void AddToPointer(int64 offset, TensorImpl *t) {
  switch(t->dtype) {
    case kFloatDtype:
      t->data = static_cast<void*>(static_cast<float>(t->data) + offset);
      return;
    case kDoubleDtype:
      t->data = static_cast<void*>(static_cast<double>(t->data) + offset);
      return;
    default:
      KALDI_ERR << "Unknown data type";
  }
}


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
    Squeeze(0, {1,3,4})  -> {3,4}
    Squeeze(1, {3,1,4})  -> {3,4}
    Squeeze(2, {3,1,4})  -> [error]
\endverbatim
 */
void Squeeze(int32 axis, TensorImpl *t);
  Squeeze(&(t->pattern), axis));
}


/** Transpose the two specified axes of a TensorImpl

    @param [in] axis1  First axis to be transposed; must be in range
                       `[-t->NumAxes(), t->NumAxes() - 1]`,
                       with negative axis being interpreted as an offset
                       from t->NumAxes().
    @param [in] axis2  Second axis to be transposed; must be in range
                       `[-t->NumAxes(), t->NumAxes() - 1]`.
                       If identical to axis1, nothing will be done.
    @param [in,out] t    TensorImpl whose axes are to be transposed.
 */
inline void Transpose(int32 axis1, int32 axis2, TensorImpl *t) {
  Transpose(axis1, axis2, &(tensor->pattern));
}



/**
   This is like PyTorch's slice() / narrow() functions.
   It selects a range of dimensions on one of the axes.  It is similar to
   indexing with a range in Python, like A[10:20].

      @param [in] axis   Axis on which to possibly reduce the dimensionality;
                         require -t->NumAxes() <= axis < t->NumAxes(), with
                         negative axis interpreted as an offset from t->NumAxes().
      @param [in] start  Starting index; must be in range [0, t->Dim(axis) - 1]
      @param [in] end    Ending index; must be in the range [start + 1, t->Dim(axis)]
      @param [in,out] t  TensorImpl whose metadata is to be modified.  Its num_axes
                         is not changed by this function (unlike Select()).

   See also: the other overloaded version of Slice() which accepts the 'step'
   parameter; and Select(), which also reduces the num-axes.
 */
void Slice(int32 axis, int32 start, int32 end, TensorImpl *t);


/**
   This is a version of Slice() which also takes a 'step' argument to support
   things like taking every other element.  See the documentation for the other
   Slice() for more context.   This is related to indexing with a range
   in Python: for example, A[0:6:2], selecting elements [0, 2, 4] of A.

      @param [in] axis   Axis on which to possibly reduce the dimensionality;
                         require -t->NumAxes() <= axis < t->NumAxes(), with
                         negative axis interpreted as an offset from t->NumAxes().
      @param [in] start  Starting index; must be in range [0, t->Dim(axis) - 1]
      @param [in] end    Ending index.  If `step > 0` must be in the range
                         [start + 1, t->Dim(axis)]; if step  < 0, must be
                         in the range [start - 1, -1].
      @param [in] step   Nonzero number that indicates the subsampling of elements
                         (and possible axis flipping).
      @param [in,out] t  TensorImpl whose metadata is to be modified.  Its num_axes
                         is not changed by this function (unlike Select()).

   See the other version of Slice(), and Select().
 */
void Slice(int32 axis, int32 start, int32 end, int32 step, TensorImpl *t);


/**
   Select one element from an axis of TensorImpl 't', reducing t->NumAxes() by
   one.

       @param [in] axis Axis from which to select an element; require
                         -t->NumAxes() <= axis < t->NumAxes(), with negative
                         axis interpreted as an offset from t->NumAxes().
       @param [in] index  Index in t to select; must be in range
                          [0, t->Dim(axis) - 1].
       @param [in,out]  t   TensorImpl whose metadata is to be modified.
 */
void Select(int32 axis, int32 index, TensorImpl *t);




}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_IMPL_UTILS_H_

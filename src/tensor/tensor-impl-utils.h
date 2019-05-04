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
   This header contains mostly functions for usage by other code in the
   framework, that operate on Tensors; see tensor-functions.h for more
   user-facing functions.
*/
namespace kaldi {
namespace tensor {


/**
  This function returns true if a and b have the same dtype
  and device.  See also Broadcastable().
*/
inline bool Compatible(const TensorImpl &a, const TensorImpl &b);


/*
  This function returns true if a, b and c have the same dtype
  and device; equivalent to Compatible(a, b) && Compatible(b, c).
*/
inline bool Compatible(const TensorImpl &a, const TensorImpl &b,
                       const TensorImpl &c);



/**
  This function returns true if the patterns of a and b are broadcastable.
  See similar function in tensor-pattern-utils.h for more information.
*/
inline bool Broadcastable(const TensorImpl &a, const TensorImpl &b,
                          bool b_non_reducing = false);

/**
  This function returns true if the patterns of a, b and c are broadcastable.
  See similar function in tensor-pattern-utils.h for more information.
*/
inline bool Broadcastable(const TensorImpl &a, const TensorImpl &b,
                          const TensorImpl &c, bool c_non_reducing = false);


/**
   This function creates the appropriate storage object for the Tensor described
   in 'impl', and sets impl->storage to that value.  Due to lazy allocation (see
   "Lazy allocation" in glossary in tensor.h) the underlying memory won't be
   allocated, but the meta-information is set up.

      @param [in,out] impl   The TensorImpl object we are allocating for.
                      Any previous value of impl->storage is ignored and
                      overwritten.  Must satisfy impl->IsValid(false).
      @return         Returns a newly allocated Storage object that
                      manages this memory block.  When this object is deleted,
                      the memory block will be deallocated using a
                      method appropriate for the device.

   This function throws on error.

   See also AllocateTensorDataShared().
 */
void CreateTensorStorage(TensorImpl *impl);


/**
   Returns true if the provided TensorImpl covers the whole of the
   allocated storage region, i.e. if every byte of the storage region
   is accessible through `impl`.
 */
bool IsWhole(const TensorImpl &impl);


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
   function that operates on Pattern.

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
void Slice(int32 axis, int32 start, int32 end, const TensorImpl &src,
           TensorImpl *dest);


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
void Slice(int32 axis, int32 start, int32 end, int32 step,
           const TensorImpl &src, TensorImpl *dest);


/**
   Copy metadata from one TensorImpl to another, while modifying it
   by selecting one index from a specified axis of a TensorImpl `t`, reducing
   the num_axes by one.

       @param [in] axis Axis from which to select an element; require
                         `-t->NumAxes() <= axis < t->NumAxes()`, with negative
                         axis interpreted as an offset from t->NumAxes().
       @param [in] index  Index in t to select; must be in range
                          [0, t->Dim(axis) - 1].
       @param [in] src    TensorImpl which is to be copied
       @param [out] dest  TensorImpl which we are copying to.  It is allowed
                          to be the same object as 'src'.
*/
void Select(int32 axis, int32 index, const TensorImpl &src,
            TensorImpl *dest);


/**


 */
inline void RegisterTensorChange(const TensorImpl &impl) {
  if (DebugMode()) {
    impl.storage_->GetChangeTracker()->RecordChange(
        SizeOf(impl.dtype), impl.pattern);
  }
}

inline int64 NumElements(const TensorImpl &a) {
  return NumElements(a.pattern);
}



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_IMPL_UTILS_H_

// tensor/tensor-functions.h

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

#ifndef KALDI_TENSOR_FUNCTIONS_H_
#define KALDI_TENSOR_FUNCTIONS_H_ 1

#include "tensor/tensor.h"

namespace kaldi {
namespace tensor {

// This file contains functions that operate on Tensors in various ways.  To
// avoid class Tensor blowing up hugely, we implement these things outside
// class Tensor.


// Note: we use the distinction between references and pointers the same way as
// you might expect from Google-style-guide code, to reflect which Tensors'
// contents are changed (so a pointer argument might have its contents changed.
// But these are in most cases pointers to const Tensors; they can be
// donst because the metadata is not changed, even if the data is.


// Sets all elements of the tensor to zero.
void SetZero(const Tensor *tensor);

// Sets all elements of the tensor to value f (cast to whatever type this Tensor
// has).
void Set(float f, const Tensor *tensor);


/** Transpose the two specified axes of a Tensor

    @param [in] axis1  First axis to be transposed; must be in range
                       `[-t->NumAxes(), t->NumAxes() - 1]`,
                       with negative axis being interpreted as an offset
                       from t->NumAxes().
    @param [in] axis2  Second axis to be transposed; must be in range
                       `[-t->NumAxes(), t->NumAxes() - 1]`.
                       If identical to axis1, nothing will be done.
    @param [in,out] t     Tensor whose axes are to be transposed.
 */
inline void Transpose(int32 axis1, int32 axis2, Tensor *t) {
  Transpose(axis1, axis2, &(t->impl_));
}

/**
   Copy the data from tensor 'src' to tensor 'dest', allowing broadcasting
   (so a dim of src can be 1 while the corresponding dim of 'dest' is >1).
   Requires Broadcastable(src, *dest, true).

   Does not require that the Dtype() or Device() of src and dest be the same
   (i.e. does not require Compatible(src, *dest)).  This is the only way in
   which Copy() is more general than Add(); otherwise, what Copy() does is a
   strict subset of what Add(1.0, 0.0, ...)  can do.
*/
void Copy(const Tensor &src, const Tensor *dest);



/**
   Template used to implement unary functions such as Log, Relu, and
   so on (this avoids boilerplate).

   Implements dest = F(src), where the F is applied elementwise.

     @param [in] src  Source Tensor
     @param [out] dest  Destination Tensor.  We require
                       SameDim(src, *dest).  May be the same
                       Tensor as 'src' (but must not partially
                       overlap in memory with 'src').

 */
template <UnaryFunctionEnum F>
void UnaryFunctionTpl(const Tensor &src, const Tensor *dest);


/*
   Implements *dest = exp(src), applied elementwise.

     @param [in] src  Source Tensor
     @param [out] dest  Destination Tensor.  We require
                       SameDim(src, *dest).  May be the same
                       Tensor as 'src' (but must not partially
                       overlap in memory with 'src').
 */
inline void Exp(const Tensor &src, const Tensor *dest) {
  UnaryFunctionTpl<kUnaryFunctionExp>(src, dest);
}

// TODO: other unary function wrappers.



/**
   Template used to implement binary functions such as division,
   taking to a power, max, min.

   Implements c = F(a, b), where F is some function of two scalars
   that returns a scalar.

     @param [in]  a  First source Tensor
     @param [in]  b  Second source Tensor
     @param [out] c  Destination Tensor.
                   We require Broadcastable(a, b, c, true).
*/
template <BinaryFunctionEnum F>
void BinaryFunctionTpl(const Tensor &a, Tensor &b, const Tensor *c);



/*
   Implements c = a / b, applied elementwise.

     @param [in] a  First source Tensor
     @param [in] b  Second source Tensor
     @param [out] c   Destination Tensor.  We require Broadcastable(a, b, c, true).
                    'c' does not have to be initialized on entry and is allowed
                    to be the same Tensor as one of a or b.
 */
inline void Div(const Tensor &a, Tensor &b, const Tensor *c) {
  BinaryFunctionTpl<kBinaryFunctionDivide>(a, b, c);
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
      @param [in,out] t  Tensor whose metadata is to be modified.  Its NumAxes()
                         is not changed by this function (unlike Select()).

   See also: the other overloaded version of Slice() which accepts the 'step'
   parameter; and Select(), which also reduces the num-axes.
 */
inline void Slice(int32 axis, int32 start, int32 end, Tensor *t) {
  Slice(axis, start, end, &(t->impl_));
}


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
      @param [in,out] t  Tensor whose metadata is to be modified.  Its NumAxes()
                         is not changed by this function (unlike Select()).

   See the other version of Slice(), and Select().
 */
inline void Slice(int32 axis, int32 start, int32 end, int32 step, Tensor *t) {
  Slice(axis, start, end, stride, &(t->impl_));
}


/**
   Select one element from an axis of Tensor 't', reducing t->NumAxes() by
   one.

       @param [in] axis Axis from which to select an element; require
                         -t->NumAxes() <= axis < t->NumAxes(), with negative
                         axis interpreted as an offset from t->NumAxes().
       @param [in] index  Index in t to select; must be in range
                         [0, t->Dim(axis) - 1].
       @param [in,out]  t   Tensor whose metadata is to be modified.
 */
inline void Select(int32 axis, int32 index, Tensor *t) {
  Select(axis, index, &(t->impl_));
}





/**
   Does

      dest := alpha * src  +  beta * dest

   while supporting broadcasting and summation, as dictated by the shapes
   of src and dest.  If beta == 0, guarantees that NaN's or inf's will
   not be propagated from the original data in 'dest' (so it works with
   uninitialized 'dest' if beta == 0).

   Requires Broadcastable(src, *dest) and Compatible(src, *dest).
   If src and dest have an integer Dtype, alpha and beta will
   be cast to integers before the operation.
*/
void Add(float alpha, float beta, const Tensor &src, const Tensor *dest);

/**
  If possible, modifies the Tensor metadata to have the requested
  dimensions.

  The semantics are based on those of PyTorch's "view" or NumPy's
  "reshape", except we try to be more accepting regarding the
  acceptable striding of the input (see below).

  Consider a Tensor 'a' has "C"-style strides.  Then this function will return
  Tensor (say, 'b') that interprets the raw data of 'a' as an array with
  "C"-style strides but with dimensions 'dims'.  (The product of 'dims' must
  equal src.NumElements()).

  Now consider a Tensor 'a2' that does not have "C"-style strides but
  has the same elements as 'a' in the sense that a(i,j,k) == a2(i,j,k).
  Then, *if possible*, this function will return a matrix b2 with
  the same elements as b, e.g. b2(i,j,k) == b(i,j,k).  Of course, whether
  this is possible depends on the details of the strides involved.

  This function returns NULL if such a tensor could not be constructed.  In that
  case,

     @param   [in] dims  The dimensions that we want The tensor to have at
                       exit; its product must equal t->NumElements().
     @param   [in,out] t   The Tensor whose metadata is to be changed

     @return  Returns true if it was possible to construct such a view, and
              false otherwise.  If t->HasCStrides() is true at entry,
              this function will never return false.  If this function returns
              false, you will likely want to construct a temporary Tensor from t
              with the same dimensions but "C"-style strides (see the
              constructor of Tensor that accepts the 'dims' parameter), and copy
              the data from t to that new Tensor.  You may then call View() on
              the temporary Tensor, which is guaranteed to succeed.

     Example:
<code>
    Tensor a({90}, kFloatDtype, kCpuDevice);
    Tensor b(a);
    bool ans = View({9,5,2}, &b);
    KALDI_ASSERT(ans);
</code>
 */
bool View(ArrayRef<int32> dims, Tensor *t);


/**
   Attempts to modify a Tensor to contain a new view of its data, in which the
   axes numbered axis1 and axis1 + 1 are merged.  This is just a special case of
   View().

   For example, if 't' is a Tensor with dims (3,4,5) and you call
   MergeAxes(1, &t), this funtion will merge axes 1 and 2 and t will, at
   exit, have shape (3 20), with elements arranged in 4 blocks of 5
   elements each (i.e. axis 1 having the higher stride).

       @param [in] axis1  The index of the first of the two axes which
                          this function will attempt to merge.  Must
                          be less than t->NumAxes() - 1.
       @param [out] t     The Tensor to be modified; on success this
                          will be a Tensor with axes merged as requested,
                          sharing the data of 'src'.  On failure, it will
                          not be changed.
       @return            Returns true on success, false if the axes could
                          not be merged.  It returns true if and only if
                        `t->Stride(axis1 + 1)==t->Stride(axis1)*t->Dim(axis1)`

     Example:
<code>
    Tensor a({3,4,5}, kFloatDtype, kCpuDevice);
    MergeAxes(0, &a);  // a now has dims {12,5}.
</code>
 */
bool MergeAxes(int32 axis1, Tensor *t);

/**
   Modifies a Tensor by splitting the axis numbered 'axis' into
   multiple axes as supplied in the 'dims' array.
   The interpretation will be as for a "C" array; so, for instance,
   if the dimensions of 'src' were (10,12) and you called
   `SplitAxis(src, 1, 3, 4)` resulting in a Tensor of dimensions
   (10,3,4), the indexes along the original axis of dimension 12 would be
   interpreted as 3 blocks of size 4.  (This is the normal semantics
   of things like NumPy's reshape or PyTorch's view.)

      @param [in] axis  The index of the axis to be split; must
                       satisfy `0 <= axis < src.Dims().`
      @param [in] dims  The dimensions desired in the axes to
                        replace axis 'axis'.  Their product must
                        equal the value of `t->Dim(axis)` at
                        entry.
      param [in,out] t   Tensor whose metadata is to be modified
   Example:
<code>
  Tensor a({10,3}, kFloatDtype, kCpuDevice);
  SplitAxis(0, {2,5}, &a);  // a now has dims {2,5,3}.
</code>
*/
void SplitAxis(int32 axis, ArrayRef<int32> dims, Tensor *t);






/**
   Does:

    `c := alpha (a * b)  +  beta c`

   where '*' is elementwise multiplication subject to broadcasting rules.  This
   supports reducing operations, and is the underlying implementation used in
   things like matrix-matrix or matrix-vector product.

   @param [in] alpha  Value that scales a * b
   @param [in] beta   Value that scales the initial value of c
   @param [in] a      First input tensor
   @param [in] b      Second input tensor
   @param [out] c     Tensor to be added to.  We require Broadcastable(a, b, c).
                      Either its data must be initialized to a known
                      value (if beta != 0) or it must be known to not contain NaN (if
                      beta == 0).   We require BroadcastCompatible(a, b, c, true).
                      'c' is const because its metadata is not changed; it is
                      a pointer as a hint to the user that its data is changed.
 */
void AddProduct(float alpha, float beta,
                const Tensor &a, const Tensor &b, const Tensor *c);





}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_FUNCTIONS_H_

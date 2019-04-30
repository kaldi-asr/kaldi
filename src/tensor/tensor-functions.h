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
   Scales each element of the Tensor `dest` by the scalar alpha.
   Equivalent to a special case of CopyScaled() where src and dest
   are the same.
*/
void Scale(Scalar alpha, const Tensor *dest);


/**
   Copy `src` to `dest` with broadcasting and possibly summation depending on
   the dims.  Equivalent to a special case of Add() with `alpha == 1.0` and
   `beta == 0.0.`

   Formally equivalent to the following; for the notation, the most relevant
   glossary entries in tensor-pattern.h are "Dereferencing a memory-index" and
   "Memory-index-tuple-set of a Pattern-tuple".
       (1)  For each memory-index `m` in `dest`, do: `*m = 0.0`
       (2)  For each memory-index-tuple `(m_src, m_dest)` in the memory-index-tuple-set
            `M(src, dest)`, do: `*m_dest += *m_src`.

     @param [in] src     Source Tensor.
     @param [out] dest   Destination Tensor.  Must satisfy
                        `BroadcastableAndCompatible(src, *dest) && !Overlap(src, *dest)`
 */
void Copy(const Tensor &src, const Tensor *dest);

/**
   Copy with a scale, `dest := src * alpha`, where the scale is a
   user-supplied scalar constant.
   This copying may involve broadcasting and/or summation depending on the dims.
   Equivalent to a special case of Add() with `beta == 1.0`.

   Formally equivalent to the following; for the notation, the most relevant
   glossary entries in tensor-pattern.h are "Dereferencing a memory-index" and
   "Memory-index-tuple-set of a Pattern-tuple".
       (1)  For each memory-index `m` in `dest`, do: `*m = 0.0`
       (2)  For each memory-index-tuple `(m_src, m_dest)` in the memory-index-tuple-set
            `M(src, dest)`, do: `*m_dest += alpha * *m_src`.

     @param [in]  alpha   Scale used in the operation
     @param [in]  src     Source Tensor.
     @param [out] dest   Destination Tensor.  Must satisfy
                        `BroadcastableAndCompatible(src, *dest) &&
                         !Overlap(src, *dest) || Identical(src, *dest))`
 */
void CopyScaled(Scalar alpha, const Tensor &src, const Tensor *dest);

/**
   Copy with a scale, where the scale is a Tensor that the user asserts has only
   one element.  (E.g. a previously computed scalar value).

   This copying may involve broadcasting and/or summation depending on the dims.
   Equivalent to a special case of Add() with `beta == 1.0`.

   Formally equivalent to the following; for the notation, the most relevant
   glossary entries in tensor-pattern.h are "Dereferencing a memory-index" and
   "Memory-index-tuple-set of a Pattern-tuple".
       (1)  For each memory-index `m` in `dest`, do: `*m = 0.0`
       (2)  For each memory-index-tuple `(m_src, m_dest)` in the memory-index-tuple-set
            `M(src, dest)`, do: `*m_dest += alpha * *m_src`.

     @param [in]  alpha   Scale used in the operation, supplied as a Tensor.
     @param [in]  src     Source Tensor.
     @param [out] dest   Destination Tensor.  Must satisfy
                        `BroadcastableAndCompatible(alpha, src, *dest) &&
                         !Overlap(src, *dest) || Identical(src, *dest))`

 */
void CopyScaled(const Tensor &alpha, const Tensor &src, const Tensor *dest);

/**
   Does

       dest := alpha * src  +  beta * dest

   while supporting broadcasting and summation, as dictated by the shapes
   of src and dest.  If beta == 0, guarantees that NaN's or inf's will
   not be propagated from the original data in 'dest' (so it works with
   uninitialized 'dest' if beta == 0).

   Requires `Broadcastable(src, *dest), Compatible(src, *dest)` and
   `Overlap(src, *dest) || Identical(src, *dest)`.  [Note: in the
   case where `Identical(src, *dest)`, i.e. they are the same Tensor
   with the same memory, you could also use Scale().

      @param [in] alpha  Scale on 'src'
      @param [in] beta   Scale on 'dest'
      @param [in] src    Source Tensor, to be added to 'dest'
      @param [in,out] dest  Destination Tensor.  Must satisfy
                     `BroadcastableAndCompatible(src, *dest) &&
                     !Overlap(src, *dest) || Identical(src, *dest))`,
*/
void AddTo(Scalar alpha, Scalar beta, const Tensor &src, const Tensor *dest);


/**
   Does

       dest := alpha * src  +  beta * dest

   while supporting broadcasting and summation, as dictated by the shapes
   of src and dest.  If beta == 0, guarantees that NaN's or inf's will
   not be propagated from the original data in 'dest' (so it works with
   uninitialized 'dest' if beta == 0).

   Requires `Broadcastable(src, *dest)`, alpha and beta
   to have one element each, all arcs be Compatible() with each other,
   `Overlap(src, *dest) || Identical(src, *dest)`, and for neither alpha
   nor beta to overlap with src or dest. [Note: in the
   case where `Identical(src, *dest)`, i.e. they are the same Tensor
   with the same memory, you could also use Scale().

      @param [in] alpha  Scale on 'src', supplied as a Tensor; must
                         have
      @param [in] beta   Scale on 'dest'
      @param [in] src    Source Tensor, to be added to 'dest'
      @param [in,out] dest  Destination Tensor.  Must satisfy
                     `BroadcastableAndCompatible(src, *dest) &&
                     !Overlap(src, *dest) || Identical(src, *dest))`,
*/
void AddTo(const Tensor &alpha, const Tensor &beta,
           const Tensor &src, const Tensor *dest);





/**
  If possible, creates a new Tensor that has the requested dimensions,
  as a 'view' of the provided Tensor; else returns NULL.  (For
  explanation of the return type, see "Optional Tensor" in glossary
  in tensor.h.)

  The quick way to describe the semantics is: first, in the case where
  'src' is laid out as a contiguous "C"-style array (w.r.t. the
  public axis numbering), return a Tensor that's also a contiguous
  "C"-style array looking at the same memory, with the provided
  dims.  Then generalize this concept to when 'src' isn't laid out
  as a "C"-style array, to preserve the same relationship between
  the index-tuples that index "src" and the returned Tensor.

  We can desribe this more precisely as follows: Consider the index-tuple-set
  I(src) of the pattern `src`; and let list(I(src)) be that set considered as a
  list sorted according to (the natural ordering c.f. "Natural order of
  index-tuples").  Let I(dest) be the index-tuple-set of a Pattern with the
  provided dimensions `dims`, and let list(I(dest)) be that set considered as an
  ordered list as above.  Extend the notion of indexing a Pattern
  (c.f. "Indexing a Pattern") to accept, and return, ordered lists in the
  obvious way.  Then this function attempts to return a pointer to a TensorImpl
  sharing the same storage as 'src', having a Pattern with the provided dims
  `dims` satisfying dest[list(I(dest))] = src[list(I(src))] if such a Pattern
  exists; and if that is not possible, returns NULL.


     @param   [in] src  The source Tensor that we are attempting to
                        construct a view of
     @param   [in] dims  The dimensions requested of the destination
                        Tensor.  Must be list of positive integers of size
                        not exceeding KALDI_TENSOR_MAX_DIM, whose product
                        equals NumElements(src).  The order is according
                        to the public numbering of axes.
     @return            Returns a `shared_ptr<TensorImpl>` of the constructed
                        view, or NULL if that was not possible.

# TODO: check that the following is valid.
<code>
    Tensor a({90}, kFloatDtype, kCpuDevice);
    Tensor v = View(a, {9,5,2});  // Tensor constructor will crash if
                                  // View returned NULL
</code>
 */
std::shared_ptr<TensorImpl> View(const Tensor &src, ArrayRef<int32> dims);


/**
   Attempts to create a Tensor containing a new view of the data in the source
   Tensor in which the axes numbered
   (axis1, axis1+1, ... axis1+num_axes_to_merge-1) are merged.  This is
   a special case of View(), provided for convenience.  For explanation of
   the return type, search for "Optional Tensor" in tensor.h.

   This attempt will only succeed if
   `src.Stride(axis1) == src.Stride(axis1 + 1) * src.Dim(axis1 + 1)`, i.e.
   if the two axes were laid out like a "C"-style array.

   More formally, we can express the relationship as follows.  Suppose this
   function returns a Tensor called `dest`; and write d = src.Dim(axis1).
   For an index-tuple i in I(src) [c.f.: "Index-tuple-set of a Pattern" in
   tensor-pattern.h], split up its indexes as:
      i = j + k + l
   where '+' in this context means appending the tuples, and 'k' corresponds
   to the range of axes (axis1, axis1+1, ... axis1+num_axes_to_merge-1).
   Let K be the set of such k values encountered from splitting up each
   i in I(src) this way, and let f be a function from tuples to integers
   that maps list(K) to a sequence of consecutive integers starting from
   zero (search for "list:" in tensor-pattern.h for explanation).
   Let g be a function from tuples to possibly-shorter tuples that
   maps j + k + l to j + (f(k),) + l, here using Python-like notation to
   interpret (x,) as a tuple with a single element x and "+" meaning appending.
   Then this function returns a Tensor sharing the same storage as `src`
   and with a Pattern such that dest[g(i)] = src[i] for all i in I(src) and
   I(dest) = g(I(src)).

      @param [in] src  Source Tensor which we are attempting
                      to construct a view of
      @param [in] axis1  Axis-index, in the public numbering.
                      Must satisfy 0 < axis1 and
                      axis1 + num_axes_to_merge <= src.NumAxes().
                      The axes axis1 and axis1 + 1 will be merged.
      @param [in] num_axes_to_merge   Default: 2.  Must be >= 1;
                      if 1, the returned Tensor will be the same
                      as 'src'.
      @return         Returns a new TensorImpl that can be used to
                      construct a Tensor with the axes merged
                      as requested, or NULL if that was not possible.
<code>
    Tensor a({3,4,5}, kFloatDtype, kCpuDevice);
    Tensor b = MergeAxes(0, &a);  // a now has dims {12,5}.
</code>
 */
std::shared_ptr<TensorImpl> MergeAxes(const Tensor &src, int32 axis1,
                                      int32 num_axes_to_merge = 2);

/**
   Modifies a Tensor by splitting the axis numbered `axis` into
   multiple axes as supplied in the `dims` array.
   The interpretation will be as for a "C"-style array; so, for instance,
   if the dimensions of `src` were (10,12) and you called
   `SplitAxis(src, 1, 3, 4)` resulting in a Tensor of dimensions
   (10,3,4), the indexes along the original axis of dimension 12 would be
   interpreted as 3 blocks of size 4.  (This is the normal semantics
   of things like NumPy's reshape or PyTorch's view.)  Note:
   the strides in the returned Tensor will be negative if the stride
   of axis `axis` of `src` was negative.

   More formally the relationship is as follows (most readers will want to skip
   this).  Let `dims` be the vector of dims supplied; let I(dims) be the
   memory-index-set of a Pattern with dimensions equal to `dims`; let
   list(I(dims)) be that set ordered as in the natural ordering (c.f. "Natural
   order of index-tuples" in tensor-pattern.h), and let f(i) be the function
   from index-tuple to integers that when applied to list(I(dims)), produces a
   sequence of consecutive integers starting from zero.  Let g be the
   function from index-tuples to index-tuples that when applied on an
   index-tuple i = (j, k, l), produces something like i = (j, k1, k2, k3, l)
   where the tuple (k1,k2,k3) = f^{-1}(k), where of course f^{-1} is the inverse
   function of f.  Then this function returns a Tensor `dest` sharing the same
   storage as `src`, such that dest[g(i)] = src[i] for i in I(src) and
   I(dest) = g(I(src))
   (Relevant glossary entries in tensor-pattern.h to understand the notation
   include "Index-tuple-set of a Pattern" and "Indexing a Pattern").

      @param [in] src   The source Tensor whose axis is to be split
      @param [in] axis  The index of the axis to be split; must
                        satisfy `0 <= axis < src.Dims().`
      @param [in] dims  The dimensions desired in the axes that
                        replace axis 'axis'.  Their product must
                        equal `src.Dim(axis)`.
      @return           Returns a Tensor whose axis is split as
                        requested.

  Example:
<code>
  Tensor a({10,3}, kFloatDtype, kCpuDevice);
  Tensor b = SplitAxis(a, 0, {2,5};  // b has dims {2,5,3}.
</code>
*/
Tensor SplitAxis(const Tensor &src, int32 axis, ArrayRef<int32> dims);






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

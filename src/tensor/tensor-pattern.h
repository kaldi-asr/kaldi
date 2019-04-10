// tensor/tensor-pattern.h

//  Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_TENSOR_TENSOR_PATTERN_H_
#define KALDI_TENSOR_TENSOR_PATTERN_H_ 1

#include "tensor/tensor-common.h"
#include <limits>

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {



/*
  GLOSSARY

    Axis:             An axis is an index into the `dims` or `strides` of a
                      Tensor.  For example, if we had a Tensor with dims=[5 6 7],
                      axis 0 would have dim=5 and axis 2 would have dim=7.
                      Some other toolkits use the word 'dimension' for this concept,
                      but we avoid that usage because it is ambiguous; we consistently
                      use the word 'axis' for this concept.

    Broadcasting:     A convention whereby for an operation on Tensors that would
                      normally be required to have the same dimension, it's
                      acceptable for, on some axis, one Tensor to have `dim = n`
                      with `n != 1` and the other to have `dim = 1`.  I.e., two dims can be
                      different as long as one of them is 1.  Most operations will
                      take place as if the Tensor with `dim = 1` had been extended
                      to `dim = n` by making identical copies.  However, if it is
                      the output Tensor that has `dim = 1`, there would be summation
                      or possibly some other appropriate reduction instead of making
                      copies.  This is different from other toolkits (the fact that
                      we extend the concept of broadcasting to encompass summation).
                      See also: PyTorch-style broadcasting, extended indexing.

    Broadcastable:   See documentation for function Broadcastable() in pattern-utils.h.
                     Briefly, two Patterns are broadcastable if their dims (padded
                     as necessary on the left by 1's to make them the same size)
                     are, for each axis, either the same or one of them is 1.
                     So for example, comparing ([ 3 4 ], [4]), we first
                     pad on the left to get ([3 4], [1 4]); then we say they
                     are broadcastable because 4 == 4 and in the remaining axis,
                     one of the dimensions is 1.

    Canonical form:  A TensorPattern is in canonical form if all axes that could be combined
                     (without affecting its memory-index-set, obviously) have been
                     combined, there are no trivial axes, all strides are positive,
                     and the axes are sorted in increasing order of stride.
                     (Note: this is in the private numbering; in the public numbering
                     this means decreasing order of stride, which is consistent
                     with "C" strides).  See CanonicalizePattern().

    Contiguous:      A Pattern is contiguous if its memory-index-set forms a contiguous
                     range of integers (no gaps).  This is different from the PyTorch
                     definition of 'contiguous', which also requires C-style strides.

    Dims vector of a Pattern: The vector of dimension of a Pattern: e.g. [] for
                    a Pattern with num_axes = 1 or [2 3] for a Pattern with
                    num-axes = 2.  Note: whenever we display dims vectors in
                    square brackets as opposed to curly, it implies we are
                    displaying them in the public numbering.

    Dims vector of a Pattern-tuple:  The dims vector of a Pattern-tuple is
                    formed by taking the dims-vectors of each Pattern in the
                    tuple, extending them on the left with 1's as necessary
                    to make the the same size, then taking the largest
                    dim on each axis (i.e. the one that is not equal to 1,
                    if they are different).  For example, for a Pattern-tuple
                    of Patterns whose dims-vectors were ([4 1 5], [6 1], [5]),
                    the dims-vector of the tuple would be [4 6 5].

    Extended indexing:  A convention whereby if we have a Tensor with, say,
                      `dims = [5 1]`, we can index that Tensor with an index-tuple
                      that:
                       (1) may have nonzero index values in any axis
                          with dim=1, so `index_tuple = [4 100]` would be a valid
                          index for this Tensor in extended indexing.
                       (2) may have more elements than the Tensor's num-axes; the
                         Tensor is implicitly extended with extra axes on the left
                         with dim=1.  This is related to PyTorch-style broadcasting.

    Index-tuple:      A tuple of integers used as an index into a Tensor.  Must
                      have at least as many elements as the Tensor's num_axes
                      (see Extended indexing).  Elements of such tuples may
                      not be negative.

    (valid Index-tuple) An index-tuple is *valid for a pattern* if it may be
                      used to index that Pattern, allowing extended indexing.
                      This is true if, after padding the index-tuple with 0's
                      on the left and padding the Pattern's dims-vector with
                      1's on the left as needed to make them the same size,
                      for each axis, if the element of the index-tuple is
                      i and the element of the dims-vector is d, i >= 0
                      and either i < d or d == 1.

    Index-tuple-set of a Pattern: The index-tuple-set of a Pattern is the set
                      of valid index-tuples assuming we are not allowing extended
                      indexing.  For example, for a Tensor with `dims = [2]`, the
                      set of valid index-tuples would be `{ (0), (1) }`; for
                      a Tensor with `dims = [2 2]` the set of valid index-tuples
                      is `{ (0,0), (0,1), (1,0), (1,1) }`.

    Index-tuple-set of a Pattern-tuple:  The index-tuple-set of a Pattern-tuple is
                      the index-tuple-set that you would obtain for a Pattern whose
                      dims equal the dims-vector of that Pattern-tuple.
                      See "dims-vector of a Pattern-tuple" for explanation of what
                      that is.

    Memory region:    A region of memory that will have been allocated with malloc()
                      or some equivalent (or obtained from some memory-management
                      code, in the case of GPU memory).  Objects of type `Storage`
                      are responsible for allocating and deleting memory regions.

    Memory-pointer:   A void* pointer to the start of a memory region.

    Memory-index:     A scalar (int64) index into a memory region viewed as a
                      linear array.  For example, for a Tensor of floats, we'd cast
                      the address of the memory-pointer to `float*` and then use
                      the memory-index as an index into that array.  For a
                      Pattern p and an index-tuple i that is valid for p, we have
                      a memory-index m = p[i], which is equal to the
                      pattern's offset plus the sum over all axes of the product of the
                      element of the index-tuple times the corresponding axis's
                      stride.  (Note: this becomes much easier to compute and
                      explain in the private numbering, because no left-padding
                      has to be done explicitly).

    Memory-index-tuple:  A tuple of Memory-indexes.  This concept is used in connection
                      with Pattern-tuples.  For a pattern-tuple q = (p1, p2, p3)
                      and an index-tuple i, we may write q[i] = (p1[i], p2[i] p3[i]),
                      where expressions like p1[i] evaluate to a memory-index.

    Offset:           The memory-index of the element with index-tuple = (all zeros)
                      of a Tensor.  Offsets will always be >= 0 because they are to
                      be used as an index into a memory-region, and negative
                      index would be outside that region.

    Pattern:          An object representing the dims, strides and offset of a Tensor.
                      (see struct TensorPattern).  Mathematically the Pattern
                      has a number of axes, its `num_axes`, and for each axis
                      from 0 <= axis < num_axes, it has a dimension and stride.


                      write these as dim[axis] and


                        Mathematical

                      We ensure this by requiring a slightly stronger property,
                      namely:

    Pattern-tuple:    A pattern-tuple of a tuple of Patterns, say:  (pattern1, pattern2);
                      we require the patterns in the tuple to be broadcastable, meaning,
                      for example: Broadcastable(pattern1, pattern2).


    An object of type TensorPattern, representing the dims, strides
                      and offset of a Tensor.

    Public numbering: The numbering of axes used in the public interface of class
                      Tensor.  We use the index `axis` when in the public numbering.
                      We use square brackets when describing dims or strides ordered
                      in the public numbering, e.g. dims=[3 4].

    Private numbering:  The reversed numbering of axes in struct TensorPattern.
                      For an axis numbered `axis` in the public numbering, its
                      reversed axis index is `raxis = num_axes - 1 - axis`.
                      This reversal makes PyTorch-style broadcasting easier.
                      We use curly brackets when describing dims or strides
                      ordered in the private numbering, e.g. dims={4,3}; this
                      is supposed to call to mind a C++ brace-initializer.

    PyTorch-style broadcasting:  We use this name to refer to the fact that in
                      PyTorch, if an operation is done on two Tensors with
                      dims=[5 6] and dims=[6], the second one would be interpreted
                      as having dims=[1 6].  That is: we pad with 1's on the left.

    Trivial axis:     An axis of a Pattern for which dim=1 and stride=0.

    Memory-index-set of a Pattern:
                      The set of all memory-indexes obtained by indexing
                      the pattern with all index-tuples in the index-tuple-set
                      of the Pattern.  The size of this set is the same as the
                      size of the index-tuple-set (by the uniqueness property).

    Memory-index-tuple-set of a Pattern-tuple:
                      The set of all memory-index-tuples obtained by indexing
                      the Patterns in the tuple with all members of the
                      index-tuple-set of the Pattern-tuple.  See "memory-index-tuple"
                      and "index-tuple-set of a Pattern-tuple" for more information.

    Stride:           A stride is the distance, in elements, between successive
                      elements of a Tensor along a particular dimension.
                      For example, a Tensor with one axis having dim=3 and
                      stride=2 would have its elements laid out in memory
                      as:  `[ element0  xxx   element1  xxx  element2 ]`,
                      where `xxx` means an element that is not part of the
                      Tensor.  Axes with dimension=1 always have stride=0
                      in this toolkit.  Tensors with negative strides may be created,
                      although they will be copied to temporaries with
                      positive stride in linear algebra operations where
                      necessary (since most BLAS implementations do not support
                      negative stride).

   Valid pattern:    A pattern is valid if it satisfies the following properties.

                      0 <= num_axe

   Uniqueness property:  A property that we require of Patterns (and hence of
                      Tensors), that ensures that no two distinct index-tuples in the
                      index-tuple-set of a Pattern may map to the same memory-index.
                      The property is: that if the axes are sorted in increasing order
                      of abs(stride), for each `0 <= axis < num_axes - 1` we have
                      `abs(strides[axis+1]) >= abs(strides[axis]) * dims[axis]`.

 */


/*
  This struct stores the dimension and strides of a Tensor.

  Below we describe the the properties that a TensorPattern is required to have.

  These properties are stricter than some other frameworks, such as PyTorch,
  which allow the users to manually add dimensions with stride 0 and dim > 1 so
  that a lower-dimensional quantity can masquerade as one with a higher
  dimension.  We require that it never be possible to access the same memory
  location using two different tuples of indexes.  We also don't allow zero dims
  (i.e. a Tensor that is initialized must not have num_elemnts==0).  If you want
  an empty Tensor, just use a null pointer.  In addition, we require that the
  stride equal zero for any axis that has dim = 1.

  Our requirements on a TensorPattern are:

    0 <= num_axes <= KALDI_TENSOR_MAX_DIM.

    for 0 <= i < num_axes:
       dims[i] > 0
       if dims[i] == 1, then strides[i] = 0.
       if dims[i] != 1, then strides[i] != 0

    for num_axes <= i < KALDI_TENSOR_MAX_DIM:
       dims[i] == 1
       strides[i] == 0

    offset >= 0

    ... plus the uniqueness property.

  Note: in the public interface of class Tensor, if you ask for Dim(i) it will
  return pattern.dims[pattern.num_axes - i], i.e.  the order is reversed.  In
  the "public numbering" we use the variable name 'axis' to describe the axis
  index, and in the "private numbering" we use the variable name 'raxis' (the
  'r' means reversed).  This reversal makes it much easier to implement
  PyTorch-style broadcasting where in an operation on Tensors of dims,
  say, (3,4) and (4), the (4) is interpreted as (1,4).

  The uniqueness property requires that we must not be able to access the same
  memory location via two different tuples of indexes).  Recause testing this
  property exactly would be difficult in general without bringing in concepts
  from number theory, we test a slightly stronger version of it that covers all
  cases we are likely to encounter.  This is that, if we take all the axes with
  dim != 1 and sort them from greatest to least stride, then for each i,
  abs(strides[i]) >= dims[i+1] * abs(strides[i+1]).
*/
struct TensorPattern {
  int32 num_axes;
  int32 dims[KALDI_TENSOR_MAX_DIM];     // the dims in reversed order, indexed
                                        // by 'raxis' (reversed axis)
  int32 strides[KALDI_TENSOR_MAX_DIM];  // the strides in reversed order,
                                        // indexed by 'raxis' (reversed axis)
  int32 code;  // pattern code; see ComputePatternCode() in tensor-pattern-utils.h
               // for details.  It is the responsibility of the user to keep
               // this updated (i.e. don't change dims or strides without updating
               // 'code').
  int64 offset;  // Offset of the element with all-zero indexes
                 // from the start of the originally allocated memory
                 // region

  // Returns true if the TensorPattern is valid, I.e. that it satifies all the
  // properties mentioned above.
  //
  //  @param [in] check_code   If true, the check includes verifying that the
  //                        'code' has the value it should (c.f. GetPatternCode()).
  //  @return     Returns true if valid, false if not valid.
  bool IsValid(bool check_code = true);

  // This comparator induces a total ordering on valid TensorPatterns.
  // It is a lexical comparison on the offset, num_axes, dims and strides.
  // (The code does not need to be compared because it is a function of the dims
  // and strides).
  bool operator < (const TensorPattern &other) const;
};


// We may later get rid of this struct and just have functions to get
// these properties.
struct TensorPatternProperties {
  // Below are cached properties that are derived from the underlying data in
  // struct TensorPattern.

  // The number of elements in the Tensor, which equals the product
  // of dims[0] .. dims[num_axes - 1].  Will always be >0.
  int64 num_elements;


  // Binary code describing the pattern, see GetPatternCode() in
  // tensor-pattern-utils.h.
  int32 code;

  // is_contiguous means that the data form a contiguous block in memory; it is
  // not the same as PyTorch's is_contiguous which is a stronger condition,
  // implying also that the strides are as for a `C-style` array.
  bool is_contiguous;

  // has_c_strides means that the strides of all axes i with dim[i] != 1,
  // equal the product of all later-numbered dims, i.e.
  // \f$ strides[i] = \prod_{j>i} dim[j] \f$, or `strides[i] = 0` if
  // dim[i] == 1 (since we use the convention that axes with dim=1 always
  // have stride=0.
  // has_c_strides is the equivalent of PyTorch's is_contiguous.
  // this->has_c_strides implies this->is_contiguous.
  bool has_c_strides;

  // Sets the members of *this to be the properties of pattern 'pattern'.
  // Ignores the previously existing values of *this.
  void UpdateProperties(const TensorPattern &pattern);
};



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_PATTERN_H_

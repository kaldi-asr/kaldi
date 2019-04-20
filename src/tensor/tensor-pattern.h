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

    Axis:             An axis is one of the (dim, stride) pairs that form part
                      of a TensorPattern, and we often use the word "axis"
                      to refer to the index of the axis, as in, for example,
                      in a Tensor with dims=[5 6 7], axis 0 has dim=5 and
                      axis 2 has dim=7.  See also axis-index and raxis-index,
                      which are more precise terms for the index of the axis
                      and clearly disambiguate the numbering used (public
                      numbering, or reversed private numbering).
                      Caution: some other toolkits use the word 'dimension' where
                      we use 'axis', but we avoid that usage because it is
                      ambiguous.

    Axis-index:       An axis-index of a Pattern or Tensor (sometimes just "axis" for short,
                      especially in code) is an index in the range [0, num_axes - 1]
                      that identifies an axis in the public numbering (see "Public numbering").
                      See also: Raxis-index.

    Axis-sorting property: search below for [Valid Pattern], point (vi), for the main
                      definition.
          [Axis-sorting property of an axis-index]:
                      There is another sense in which we use the term
                      'axis-sorting property': for a Pattern whose axes are sorted
                      from least to greatest abs(stride) [in the private numbering],
                      we say that "the axis-sorting property holds for axis-index i
                      of that Pattern" if:
                                 dim(i) * abs(stride(i)) <= abs(stride(i+1)).


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

    Canonical form:  A TensorPattern is in canonical form if all pairs of axes that
                     could be combined (without affecting its memory-index-set)
                     have been combined, where there are no trivial axes, all
                     strides are positive, and the axes are sorted in increasing
                     order of stride.  (Note: this is in the private numbering;
                     in the public numbering this means decreasing order of
                     stride, which is consistent with "C" strides).  See
                     CanonicalizePattern().

    Contiguous:      A Pattern is contiguous if its memory-index-set forms a contiguous
                     range of integers (no gaps).  This is different from the PyTorch
                     definition of 'contiguous', which also requires C-style strides.

    Dims-vector of a Pattern: The vector of dimension of a Pattern: e.g. [] for
                    a Pattern with num_axes = 1 or [2 3] for a Pattern with
                    num-axes = 2.  Note: whenever we display dims vectors in
                    square brackets as opposed to curly, it implies we are
                    displaying them in the public numbering.

    Dims-vector of a Pattern-tuple:  The dims vector of a Pattern-tuple is
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
                       (1) may have nonzero index values in any axis for which
                          with dim=1, so `index_tuple = [4 100]` would be a valid
                          index for this Tensor in extended indexing.
                       (2) may have more elements than the Tensor's num-axes; the
                         Tensor is implicitly extended with extra axes on the left
                         (in the public numbering) / the right (in the private
                         numbering) with dim=1.  See also: PyTorch-style broadcasting.

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
                      (see struct TensorPattern).  The Pattern has
                      an 'offset' which is the memory-index of the element of the Tensor
                      whose index-tuple is all zeros; the Pattern also
                      has a number of axes, `0 <= num_axes < KALDI_TENSOR_MAX_AXES`,
                      and for each axis from 0 <= axis < num_axes, it has a dimension
                      dim(axis) and stride(axis).

                      Search below for 'Valid Pattern' for properties a Pattern must
                      (in most circumstances) satisfy.


    Pattern-tuple:    A pattern-tuple of a tuple of Patterns, say:  (pattern1, pattern2);
                      we require the patterns in the tuple to be broadcastable, meaning,
                      for example: Broadcastable(pattern1, pattern2).


    An object of type TensorPattern, representing the dims, strides
                      and offset of a Tensor.

    Public numbering: The numbering of axes used in the public interface of class
                      Tensor.  We use the index `axis` when in the public numbering.
                      We use square brackets when describing dims or strides ordered
                      in the public numbering, e.g. dims=[3 4].
                      See also: axis-index

    Private numbering:  The reversed numbering of axes in struct TensorPattern.
                      For an axis numbered `axis` in the public numbering, its
                      reversed axis index is `raxis = num_axes - 1 - axis`.
                      This reversal makes PyTorch-style broadcasting easier.
                      We use curly brackets when describing dims or strides
                      ordered in the private numbering, e.g. dims={4,3}; this
                      is supposed to call to mind a C++ brace-initializer.
                      See also: raxis-index

    PyTorch-style broadcasting:  We use this name to refer to the fact that in
                      PyTorch, if an operation is done on two Tensors with
                      dims=[5 6] and dims=[6], the second one would be interpreted
                      as having dims=[1 6].  That is: we pad with 1's on the left.

    Raxis-index:      We use the term "raxis-index", often just "raxis" for short,
                      to mean the index of an axis in the reversed, private numbering.
                      This would usually be in the range [0, num_axes - 1] for
                      a Pattern with `num_axes` axes, but for broadcasting purposes,
                      if we are doing an operation between Tensors of different
                      numbers of axes we may often use larger raxis values for the Tensor
                      of smaller num_axes (see PyTorch-style broadcasting).

    Set-equivalent:   Two Patterns are set-equivalent if their memory-index-sets
                      are identical.

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

    Linear property:
                      Consider Patterns P and Q with the property that the
                      memory-index-set of P is a subset of the memory-index-set of
                      Q.  If i is an index-tuple, let P(i) be the map from
                      i to a memory-index, and let
                            \f$   Q^{-1}(m)   \f$
                      be the function that maps a memory-index m in the memory-index-set
                      of Q to the index-tuple i in the index-tuple-set of Q such
                      that Q(i) = m.  Then we say that P is linear in Q if
                      for all index-tuples i and j such that i, j and i + j are
                      in the index-tuple-set of P,
                      \f$  Q^{-1}(P(i)) + Q^{-1}(P(j)) = Q^{-1}(P(i+j)) \f$.
                      [Transitivity]
                      It is easy to show that the linear property is transitive;
                      that is if P is linear in Q and Q is linear in R, then
                      P is linear in R.

    Regularity property:   This is a property of Patterns that is relevant when reducing
                      Patterns to a common set of strides.

                      We formulate the regularity property to only apply for
                      Patterns which are valid-- and which have positive strides in increasing order; these
                      the stipulation on having postive, sorted strides
                      is for convenience, since we happen to need it only for
                      that case and it's easier to formulate in that case.

                      For the regularity property to apply, a Pattern must also
                      be valid-- (see its own glossary entry).

                      A Pattern is regular if, in addition to satisfying the
                      properties mentioned above, for each axis-index
                      0 <= i < num_axes - 1,
                      there is an integer k with i < k <= num_axes, such that:
                        (i) Either k == num_axes, or dim(i) * stride(i) <= stride(k),
                      and
                        (ii) For all j with i < j < k, stride(i) divides stride(j)
                            exactly and dim(j) = 1.
                      [Note: the condition that dim(j) == 1 will anyway be true if
                      the Pattern has the uniqueness property.]

                      The reader may notice that if we were to restrict
                      k to equal i + 1, then
                      this would be equivalent to the axis-sorting property
                      (property (v)) plus the requirement that the strides be
                      positive and sorted.

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

   Uniqueness property:  A property of a Pattern that no two different index-tuples,
                      when used to index the Pattern, generate the same memory-index.
                      The axis-sorting property is sufficient, but not necessary,
                      to ensure the uniqueness property.  (The uniqueness property
                      is probably not so easy to test for efficiently in the general
                      case; at least, we have not found a way).

    Valid Pattern:
                     A valid Pattern must be as follows.  Think of this as the mathematical definition;
                     see the declaration of struct TensorPattern for additional details about how
                     it is stored.

                          (i) The num_axes must satisfy 0 <= num_axes < KALDI_TENSOR_MAX_DIM
                          (ii) The offset must be >= 0.
                          (iii) the dims must all be >0.
                          (iv) the strides must be nonzero (but not necessarily positive) for axes with
                                dim != 1.
                          (v) the axis-sorting property.   This property is sufficient, but not
                              necessary, to ensure the uniqueness property.  It requires that
                              when the axes are sorted from least to greatest value of abs(stride),
                              for each axis-index 0 <= i < num_axes - 1:
                                    dim(i) * abs(stride(i)) <= abs(stride(i+1)).
                              (Note: this property doesn't require that the axes be sorted that
                              way; if you need that, search for "Canonical form").
                          (vi) the strides must be zero for axes with dim=1.


     Valid- Pattern:
                      A Pattern is valid- if it satisfies properties (i) through (v) of
                      a valid Pattern (i.e. it may have nonzero strides for axes with dim=1).
                      A valid pattern is also valid-.
     Valid-- Pattern:
                      A Pattern is valid-- if it satisfies properties (i) through (iv) of
                      a valid Pattern.  A pattern that is valid or valid- is also valid--.
 */


/*
  This struct stores the dimension and strides of a Tensor.

  Below we describe the the properties that a TensorPattern is required to have.
  Most of them are described in the glossary in the entry for "Valid Pattern",
  but there are a couple more that have to do with the specifics of how we
  store things in this struct.

  These properties are stricter than some other frameworks, such as PyTorch,
  which allow the users to manually add dimensions with stride 0 and dim > 1 so
  that a lower-dimensional quantity can masquerade as one with a higher
  dimension.  (This framework allows the same kinds of operations, they are just
  not done by the same mechanism).   We
  also don't allow zero dims (i.e. a Tensor that is initialized must not have
  num_elemnts==0).  If you want an empty Tensor, just use a null pointer.  In
  addition, we require that the stride equal zero for any axis that has dim = 1.
  There is also the "axis-sorting" property (see its glossary entry for more info).

  Our requirements of a TensorPattern are:

    0 <= num_axes <= KALDI_TENSOR_MAX_DIM.

    for 0 <= i < num_axes:
       dims[i] > 0
       if dims[i] == 1, then strides[i] = 0.
       if dims[i] != 1, then strides[i] != 0

    for num_axes <= i < KALDI_TENSOR_MAX_DIM:
       dims[i] == 1
       strides[i] == 0

    offset >= 0

    The axis-sorting property (see property (v) in "Valid Pattern" above)

  Note: in the public interface of class Tensor, if you ask for Dim(i) it will
  return pattern.dims[pattern.num_axes - i], i.e. the interface uses the public
  numbering, while the axes are physically stored using the reversed "private
  numbering".   This reversal makes it much easier to implement
  PyTorch-style broadcasting where in an operation on Tensors of dims,
  say, (3,4) and (4), the (4) is interpreted as (1,4).
*/
struct TensorPattern {
  int32 num_axes;
  int32 dims[KALDI_TENSOR_MAX_DIM];     // the dims in reversed order, indexed
                                        // by 'raxis' (reversed axis)
  int32 strides[KALDI_TENSOR_MAX_DIM];  // the strides in reversed order,
                                        // indexed by 'raxis' (reversed axis)
  int32 code;  // pattern code; see ComputePatternCode() in tensor-pattern-utils.h
               // for details.  If this is negative then it means it has not been
               // computed.  In a valid TensorPattern the code will always be either
               // negative or up-to-date.
  int64 offset;  // Offset of the element with all-zero indexes
                 // from the start of the originally allocated memory
                 // region

  // Returns true if the TensorPattern is valid.  This includes all the
  // mathematical conditions on a valid Pattern (search above for "Valid
  // Pattern"), plus extra conditions related to struct TensorPattern,
  // namely: dims and strides with index >= num_axes should be
  // 1 and 0 respectively; and the code should either be -1 or or
  // be the same as ComputePatternCode() returns on this pattern.
  // See also IsCanonical() in tensor-pattern-utils.h.
  bool IsValid();

  // This comparator induces a total ordering on valid TensorPatterns.  It is a
  // lexical comparison on the offset, num_axes, dims and strides.  (The code
  // does not need to be compared because, if not -1, it is a function of the
  // dims and strides).
  bool operator < (const TensorPattern &other) const;
};


/// Returns a string representing a Pattern, of the form:
/// "offset=a dims=[b c d] strides=[e f g]"; this is for debugging
/// purposes.
std::string PatternAsString(const TensorPattern &pattern);

/// Returns a string representing the dims of a Pattern, something like
/// "[10 20 100]"
std::string DimsAsString(const TensorPattern &pattern);

/// Returns a string representing the strides of a Pattern, something like
/// "[1 10 200]"
std::string StridesAsString(const TensorPattern &pattern);



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
  // TODO: see if this is even needed; it may not be.
  bool is_contiguous;

  // has_c_strides means that the strides of all axes i with dim[i] != 1,
  // equal the product of all later-numbered dims, i.e.
  // \f$ strides[i] = \prod_{j>i} dim[j] \f$, or `strides[i] = 0` if
  // dim[i] == 1 (since we use the convention that axes with dim=1 always
  // have stride=0.
  // has_c_strides is the equivalent of PyTorch's is_contiguous.
  // this->has_c_strides implies this->is_contiguous.
  // TODO: see if this is even needed; it may not be.
  bool has_c_strides;

  // Sets the members of *this to be the properties of pattern 'pattern'.
  // Ignores the previously existing values of *this.
  void UpdateProperties(const TensorPattern &pattern);
};



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_PATTERN_H_

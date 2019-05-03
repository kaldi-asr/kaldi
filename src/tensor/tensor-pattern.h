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


namespace kaldi {
namespace tensor {



/*
  PATTERN GLOSSARY   (note: see also TENSOR GLOSSARY in tensor.h)

    Axis:             An axis is one of the (dim, stride) pairs that form part
                      of a TensorPattern.  We will sometimes use the word "axis"
                      to refer to the integer index of the axis, as in, for example,
                      in a Tensor with dims=[5 6 7], axis 0 has dim=5 and
                      axis 2 has dim=7; but this should more precisely
                      be called axis-index or raxis-index (see their own
                      glossary entries; they respectively use the public
                      numbering, or reversed private numbering).  To describe
                      the number of axes of a Tensor, we use the term "num-axes" /
                      "number of axes".

    Axis-index:       An axis-index of a Pattern or Tensor (sometimes just "axis" for short,
                      especially in code) is an index that identifies an axis in the
                      public (see "Public numbering").  A valid axis-index for a Pattern
                      with `num_axes` axes is in the range [0, num_axes - 1].

                      For an axis-index i, the corresponding raxis-index (c.f. "Raxis-index:"
                      or "Private numbering:") would be num_axes - 1 - i.

                      See also "Eaxis-index" for where we allow negative axis-indexes
                      as offsets from the end.

    axis-dominance property: search below for [Valid Pattern], point (vi), for the main
                      definition.
          [axis-dominance property of an axis-index]:
                      There is another sense in which we use the term
                      'axis-dominance property': for a Pattern whose axes are sorted
                      from least to greatest abs(stride) [in the private numbering],
                      we say that "the axis-dominance property holds for axis-index i
                      of that Pattern" if:
                                 dim(i) * abs(stride(i)) <= abs(stride(i+1)).


    Broadcasting:    A convention whereby for an operation on Tensors that would
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

    Compact:         A Pattern is compact if its memory-index-set forms a contiguous
                     range of integers (no gaps).  (We don't call this "contiguous"
                     because PyTorch uses the same word with a different meaning).

    Default strides:  The default strides for a pattern with provided dimensions are:
                     of course, zero for any axis with dim=1; and otherwise (describing
                     it in the public numbering of axes), each axis's stride is
                     the product of the later-numbered axes' dims.  It corresponds
                     to the strides of a "C" array.
                     This is the policy that we will use when constructing new
                     Tensors if only the dims are provided, which is why we call these
                     the default strides.
                     A Pattern having default strides is equivalent to its having
                     normalized strides and also being compact.

                     See also: Normalized strides, Compact.

    Dereferencing a memory-index:
                     Sometimes in formal explanations of algorithms we will use notation
                     `*m` meaning, for a memory-index `m`, the location that it points to
                     in the relevant storage region; we will assume that it is obvious
                     from the context which storage region.   See also: "Storage region"

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

    Disjoint Patterns:  When we speak of disjoint Patterns we mean that
                    their memory-index-sets are disjoint; see memory-index-set.

    Eaxis-index / extended axis-index:
                      We use the term Eaxis-index, or in code, eaxis_index, to
                      mean an axis-index in the public numbering (c.f.:
                      Axis-index) but where negative values are allowed, as in
                      Python.  Negative values are interpreted as offsets from
                      the num_axes of the Pattern in question, so for instance
                      -1 would correspond to num_axes - 1.  Valid eaxis-indexes
                      would be in the range [-num_axes, num_axes - 1].  See
                      also: Axis-index, Raxis-index.

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

    Index:            If this word is used unqualified in the context of a Pattern
                      or tensor it will generally mean an integer that's part of an
                      index-tuple, and is being used to index a particular axis of
                      a Pattern.  For example, on an axis where the Pattern's dimension
                      is `dim`, a valid index i would be in the range 0 <= i < dim.

    Index-tuple:      A tuple of integers used as an index into a Tensor.  Must
                      have at least as many elements as the Tensor's num_axes
                      (see Extended indexing).  Elements of such tuples may
                      not be negative.  The elements of an index-tuple are in
                      the same order as the axes, and in some cases it may
                      be necessary to disambiguate whether we are referring
                      to the public numbering or the private numbering of the
                      axes.

    [Valid Index-tuple]: An index-tuple is *valid for a pattern* if it may be
                      used to index that Pattern, allowing extended indexing.
                      (see "Extended indexing" for details).

    Indexing a Pattern:  For a pattern `p` and an index-tuple `i` that is valid
                       for the pattern (see: "Valid Index-tuple"), we write
                      `p[i] = m` meaning that when indexing a pattern `p`
                      with index-tuple `i` we get memory-index `m`.
                      `m` is of coure the sum of the pattern's offset plus
                      the sum over all axis-indexes, of the element of the index-tuple
                      multiplied by the Pattern's stride for that axis.

    Index-tuple-set of a Pattern: The index-tuple-set I(p) of a Pattern p is the
                      set of valid index-tuples assuming we are not allowing extended
                      indexing.  For example, for a Tensor with `dims = [2]`, the
                      set of valid index-tuples would be `{ (0), (1) }`; for
                      a Tensor with `dims = [2 2]` the set of valid index-tuples
                      is `{ (0,0), (0,1), (1,0), (1,1) }`.

    Index-tuple-set of a Pattern-tuple:  The index-tuple-set I(P, Q) of a Pattern-tuple
                      (P, Q) is the index-tuple-set that you would obtain for a
                      Pattern whose dims equal the dims-vector of that
                      Pattern-tuple.  See "dims-vector of a Pattern-tuple" for
                      explanation of what that is.  View I(P, Q) as simply
                      shorthand for I((P, Q)).

    Justified:        We say that a Pattern is justified if least (i.e. most
                      negative) memory-index in its memory-index-set is zero.  For
                      Patterns with nonnegative strides, this is equivalent to
                      its offset being zero.

    Memory region:    A region of memory that will have been allocated with malloc()
                      or some equivalent (or obtained from some memory-management
                      code, in the case of GPU memory).  Objects of type `Storage`
                      are responsible for allocating and deleting memory regions.

    Memory-pointer:   A void* pointer to the start of a memory region.

    Memory-index: (abbr: mindex)
                      An integer (int64) index into a memory region viewed as a
                      linear array.  For example, for a Tensor of floats, we'd
                      cast the address of the memory-pointer to `float*` and
                      then use the memory-index as an index into that array.  In
                      code, this may be called 'mindex.'  For a Pattern p and an
                      index-tuple i that is valid for p, we have a memory-index
                      m = p[i], which is equal to the pattern's offset plus the
                      sum over all axes of the product of the element of the
                      index-tuple times the corresponding axis's stride.

    Memory-index-tuple:  A tuple of Memory-indexes.  This concept is used in connection
                      with Pattern-tuples.  For a pattern-tuple q = (p1, p2, p3)
                      and an index-tuple i, we may write q[i] = (p1[i], p2[i] p3[i]),
                      where expressions like p1[i] evaluate to a memory-index.

    Natural order of index-tuples: Suppose we have a set of index-tuples, all with
                    the same num-axes / length of tuple.  What we call the
                    "natural order" (this is just a convenient name, it does not
                    imply any objective naturalnesss) is a total order on
                    index-tuples that corresponds to interpreting the
                    index-tuples as indexes into a "C"-style array (in the
                    public numbering of axes) or a Fortran-style one (in the
                    private one) and comparing the memory addresses.  In
                    the public numbering this order is the same as lexical
                    order, e.g. ([0 0], [0 1], [1 0], [1 1]); in the private
                    numbering it is lexical order but starting from the right,
                    not the left.
       [list:]      Given a set S of index-tuples, we will sometimes write
                    list(S) to mean a list of index-tuples with the same
                    elements as S, ordered in the natural order.

    Num-axes:        The number of axes that a Tensor has.  This is a number in the
                     range [0, KALDI_TENSOR_MAX_DIM], i.e. 0 through 6.

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


    Pattern-tuple:    A pattern-tuple of a tuple of Patterns, say:  (P, Q),
                      where the patterns in the tuple are broadcastable, meaning,
                      for example: Broadcastable(P, Q).


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
                      Note: whenever we refer to broadcasting we include this feature;
                      this glossary entry exists just to explain it, not to claim
                      that we have two different versions of broadcasting.

    Raxis-index:      We use the term "raxis-index", often just "raxis" for short,
                      to mean the index of an axis in the reversed, private numbering.
                      This would usually be in the range [0, num_axes - 1] for
                      a Pattern with `num_axes` axes, but for broadcasting purposes,
                      if we are doing an operation between Tensors of different
                      numbers of axes we may often use larger raxis values for the Tensor
                      of smaller num_axes (see PyTorch-style broadcasting).

    Set-equivalent:   Two Patterns are set-equivalent if their memory-index-sets
                      are identical.

    Trivial axis:     An axis of a Pattern for which dim=1.  Such axes will have
                      stride=0 if the Pattern is valid.

    Memory-index-set of a Pattern:
                      The memory-index-set M(p) of a Pattern p is
                      the set of all memory-indexes obtained by indexing
                      the pattern with all index-tuples in the index-tuple-set
                      I(p) of the Pattern.  By extending the notion of indexing
                      a Pattern (c.f. "Indexing a Pattern") to take set
                      arguments, this could be written as M(p) = p[I(p)].  Note:
                      by the uniqueness property, we always have |M(p)| = |I(p)|
                      for a valid Pattern, i.e. the sizes of the sets are the
                      same.

    Memory-index-tuple-set of a Pattern-tuple:
                      The set of all memory-index-tuples M(P, Q) obtained by indexing
                      the Patterns in the tuple (P, Q) with all members of the
                      index-tuple-set of the Pattern-tuple.  See "memory-index-tuple"
                      and "index-tuple-set of a Pattern-tuple" for more information.
                      View the notation M(P, Q) as shorthand for M((P, Q)).

    Normalized strides:  We say that a Pattern has normalized strides if the
                      strides are all nonnegative and the nonzero strides
                      are in strictly increasing order in the private numbering
                      (hence strictly decreasing in the public numbering).

                      See also: Default strides (which is a stronger property).

    Linear property:
                      This is a slightly technical property used in certain
                      proofs involving patterns.
                      Consider patterns P and Q with the property that the
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
                      this would be equivalent to the axis-dominance property
                      (property (v)) plus the requirement that the strides be
                      positive and sorted.

    Storage region:   A Tensor, in addition to a Pattern, has a storage region
                      that can be though of as a pointer (say, to float) which
                      we index with a memory-index: say, p[m], if s is the
                      pointer and m is the memory-index.  See storage.h.
                      See also "Dereferencing a memory-index".

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
                      The axis-dominance property is sufficient, but not necessary,
                      to ensure the uniqueness property.  (The uniqueness property
                      is probably not so easy to test for efficiently in the general
                      case).

    Valid Pattern:
                     A valid Pattern must be as follows.  Think of this as the mathematical definition;
                     see the declaration of struct TensorPattern for additional details about how
                     it is stored.

                          (i) The num_axes must satisfy 0 <= num_axes < KALDI_TENSOR_MAX_DIM
                          (ii) The offset must be >= 0.
                          (iii) the dims must all be >0.
                          (iv) the strides must be nonzero (but not necessarily positive) for axes with
                                dim != 1.
                          (v) the axis-dominance property.   This property is sufficient, but not
                              necessary, to ensure the uniqueness property.  It requires that
                              when the axes are sorted from least to greatest value of abs(stride),
                              for each axis-index 0 <= i < num_axes - 1:
                                    dim(i) * abs(stride(i)) <= abs(stride(i+1)).
                              (Note: this property doesn't require that the axes be sorted that
                              way; if you need that, search for "Canonical form").
                          (vi) the strides must be zero for axes with dim=1.


     Valid-1 Pattern:
                      A Pattern is valid-1 (read as: valid minus one) if it
                      satisfies properties (i) through (v) of a valid Pattern
                      (i.e. it may have nonzero strides for axes with dim=1).  A
                      valid pattern is also valid-1.
     Valid-2 Pattern:
                      A Pattern is valid-2 (read as valid minus two) if it
                      satisfies properties (i) through (iv) of a valid Pattern.
                      A pattern that is valid or valid-1 is also valid-2.
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
  There is also the "axis-dominance" property (see its glossary entry for more info).

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

    The axis-dominance property (see property (v) in "Valid Pattern" above)

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
  int64 offset;  // Offset of the element with all-zero indexes
                 // from the start of the originally allocated memory
                 // region

  int32 code;  // pattern code; see ComputePatternCode() in tensor-pattern-utils.h
               // for details.  If this is negative then it means it has not been
               // computed.  In a valid TensorPattern the code will always be either
               // negative or up-to-date.

  int32 properties;  // More occasionally-needed properties.  This is similar to
                     // OpenFst's notion of properties, where we compute them
                     // only on demand.  In a valid TensorPattern the properties
                     // will always be accurate, but see "Accurate properties"
                     // in glossary above for definition (it can be zero).

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


  // Equality operator on TensorPattern.  Compares the num_axes, offset, and
  // dims and strides indexed [0... num_axes-1].  (In patterns that satisfy IsValid(),
  // the remaining dims and strides would be 1 and 0 respectively, so checking
  // the is pointless).
  bool operator == (const TensorPattern &other) const;

  // Assignment operator (copies all members).
  bool operator = (const TensorPattern &other) const;
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



/**
   Returns a hash value for hashing TensorPattern.  Depends on num_axes,
   offset, and dims and strides indexed [0... num_axes-1].  pattern does
   not have to be valid.
 */
size_t GetHash(const TensorPattern &pattern);

// C++ hashing object for TensorPattern
struct TensorPatternHasher {
  size_t operator (const TensorPattern &pattern) { return GetHash(pattern); }
};

// C++ hashing object for TensorPattern*; requires the pointer
// be non-NULL and to point to a TensorPattern.
struct TensorPatternPtrHasher {
  size_t operator (TensorPattern *pattern) { return GetHash(*pattern); }
};

struct TensorPatternPtrEqual {
  size_t operator (TensorPattern *pattern1,
                   TensorPattern *pattern2) {
    *pattern1 == *pattern2;
  }
};



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_PATTERN_H_

// tensor/tensor-pattern-utils.h

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


#include "tensor/tensor-common.h"
#include "tensor/tensor-pattern.h"
#include "tensor/array-ref.h"

// This header includes various functions operating on Patterns.
// See also tensor-pattern-extra-utils.h which contains the
// more obscure and less user-facing functions.

namespace kaldi {
namespace tensor {


/**
   This function returns a code that compactly says whether each axis
   has dim = 1 or dim != 1.  For purposes of the code generated, the number
   of axes does not matter.  The lower-order KALDI_TENSOR_MAX_DIM bits
   of the code might potentially be set; the rest will be zero.

   The rightmost (least significant) bit corresponds to the last-numbered axis,
   equivalent to raxis (reversed axis-index) == 0.

   Note that non of the example `dims` vectors below have any leading
   (dim=1) axes, because they wouldn't affect the code.

   The examples below will use c++14 binary literals, although
   the code doesn't use them.  In the notation below, in dims vectors,
   x is a stand-in for 'any number greater than 1'.

    0b00000000  0x00  dims=(), a scalar
    0b00000001  0x01  dims=(x)
    0b00000010  0x02  dims=(x,1)
    0b00000011  0x03  dims=(x,x)

    etc.

  See also GetPatternCode(), which includes the same information but
  also stride-related information.
 */
int32 GetDimsCode(const TensorPattern &pattern);


enum PatternEnum {
  kPatternContainsNegativeStride = 2048
  // e.g.:
  // bool contains_negative_stride =
  //     (pattern.code | kPatternContainsNegativeStride) != 0;
};

// Returns true if the pattern code indicates that the pattern contains a
// negative stride.
inline bool ContainsNegativeStride(int32 pattern_code) {
  return (pattern_code | kPatternContainsNegativeStride) != 0;
}

// Returns true if the pattern code indicates that the raxis
// numbered 'raxis' (the r refers to the backwards numbering used
// in 'pattern') is 'trivial' (meaning: dim=1, stride=0).
inline bool AxisIsTrivial(int32 pattern_code, int32 raxis) {
  return (pattern_code | 1 << raxis) == 0;
}


/**
   This function removes trivial axes (i.e. axes with dim=1) from 'pattern'.
   Although in a valid pattern axes with dim=1 must have stride=0
   and vice versa, this function does not check that property; it simply
   removes axes with dim=1, reducing num_axes appropriately.

     @param [in,out] pattern   Pattern to be modified.  Any axes with dim=1
                         will be removed and the num_axes reduced.  Will be
                         valid at output if it was valid at input, or even if
                         it was valid at input in all but property (iv),
                         that strides must be zero for axes with dim=1.
                         CAUTION: the code of 'pattern' is *not* updated.
 */
void RemoveTrivialAxes(TensorPattern *pattern);


/**
   This function returns a code that compactly represents the same information
   as GetDimsCode() [i.e. which axes had dim != 1], but also encodes which axis,
   if any, had stride=1, and has a bit that says whether any axis had negative
   stride.  (No two axes can have stride=1, due to a combination of the fact
   that dim=1 implies stride=0, and the the uniqueness rule; search in
   tensor-pattern.h).

   Let
      n = 0 if no axis had stride=1, otherwise:
      n = 1 + the raxis index which had stride=1.

    (raxis is the axis index when accessing the axes in reversed order, as
     stored in pattern.dims and pattern.strides).

   For example if the strides were [10,3,1] we would have
   n = 1; i if the strides were [10,1,3] we would have n = 2.

   IMPORTANT NOTE ON ORDERING: lists of dims or strides in square
   brackets, like [1,2], are in the non-reversed ordering as exposed
   by the Tensor API.

   The value 'n' occupies the bits starting from 8 in the returned code,
   i.e. bits 8,9,10 (counting from the right, i.e. from the least to
   most significant).

   Bit 11 is 1 if any of the strides were negative, and zero otherwise.
   None of the example bit-patterns below have this bit set.  The
   underlying BLAS in most cases does not support negative strides so
   we deal with it by copying the data to a temporary with positive
   strides.

   The low-order KALDI_TENSOR_MAX_DIM bits are as returned by GetDimsCode().

   The explanation below will use c++14 binary literals (like 0b010101), although the code
   doesn't use them as we compile as c++11; we show the corresponding hex codes which
   are used in the code (and anyway easier to parse).

   In the notation below, in dims vectors, x or X is a stand-in for 'any number
   not equal to 1', and upper-case X indicates that the axis has stride=1.  In
   the example `dims` vectors below, we don't put any leading `dim=1` axes,
   because they would not affect the code generated.  The list of numbers
   in square brackets [] below may be interpreted as the sequence of dims for the
   Tensor, in the non-reversed ordering that the Tensor API exposes.

   The ' at the 8th bit is to make the bit-string easier to parse.

    0b000'00000000  0x000  dims=[], a scalar
    0b000'00000001  0x001  dims=[x], a vector with a stride
    0b001'00000001  0x101  dims=[X], a vector
    0b000'00000010  0x002  dims=[x,1], a vector with a stride
    0b010'00000010  0x202  dims=[X,1], a vector
    0b000'00000011  0x003  dims=[x,x], a matrix with a stride
    0b001'00000011  0x103  dims=[x,X], a matrix
    0b010'00000011  0x203  dims=[X,x], a transposed matrix
    0b000'00000100  0x008  dims=[x,1,1], a vector with a stride
    0b011'00000100  0x308  dims=[X,1,1], a vector
    0b010'00000110  0x20B  dims=[x,X,1], a matrix
    0b011'00000110  0x30B  dims=[X,x,1], a transposed matrix
    0b000'00000110  0x10B  dims=[x,x,1], a matrix with column stride
    0b001'00000101  0x109  dims=[x,1,X], a matrix
    0b011'00000101  0x309  dims=[X,1,x], a transposed matrix
    0b000'00000101  0x009  dims=[x,1,x], a matrix with column stride

    ...
 */
int32 ComputePatternCode(const TensorPattern &pattern);


inline int32 CombineCodes(int32 code1, int32 code2) {
  return (code1 << 12) | code2;
}

inline int64 CombineCodes(int32 code1, int32 code2, int32 code3) {
  return (static_cast<int64>(code1) << 24) |
      static_cast<int64>(code2 << 12) |
      static_cast<int64>(code3);
}


/**
   Modifies 'p' in-place by inserting an axis with (dim=1,stride=0) at the
   specified position specified in the reversed numbering physically used
   in the pattern.  Updates p->code.

   Showing just the dims in the pattern (in the order physically present in the
   dims array), for some examples:

\verbatim
    UnsqueezeR({3,4}, 0)  -> {1,3,4}
    UnsqueezeR({3,4}, 1)  -> {3,1,4}
    UnsqueezeR({3,4}, 2)  -> {3,4,1}
\endverbatim

     @param [in]    raxis   The index at which the extra axis is to appear.
                            We require 0 <= raxis <= p->num_axes.
     @param [in,out] p      The pattern to which we are adding an axis.
                            Will have its num_axes increased by 1
                            at exit, possibly its dims and strides
                            arrays changed, and its code updated.
 */
void UnsqueezeR(int32 raxis, TensorPattern *p);


/**
   Modifies 'p' in-place by inserting an axis with (dim=1,stride=0) at the
   specified axis-index (numbered in the public numbering).
   Equivalent to PyTorch's unsqueeze(), including its behavior with
   negative axis indexes (axis < 0 is interpreted as to num_axes + 1 - axis).

   Showing just the dims in the pattern, in the non-reversed order as
   exported by the API, some examples are:

\verbatim
    Unsqueeze([6,5], 0) -> [1,6,5]
    Unsqueeze([3,4], 1) -> [3,1,4]
    Unsqueeze([9,10], 2) -> [9,10,1]
    Unsqueeze([9,10], -1) -> [9,10,1]
\endverbatim

     @param [in]    axis   The index at which the extra axis is to appear.
                           We require -p->num_axes - 1 <= raxis <= p->num_axes
                           The large allowable range is because negative
                           axes are permitted, e.g. -1 means insert a new
                           axis after the last existing axis.
     @param [in,out] p      The pattern to which we are adding an axis.
                            Will have its num_axes increased by 1
                            at exit, possibly its dims and strides
                            arrays changed, and its code updated.
 */
inline void Unsqueeze(int32 axis, TensorPattern *p) {
  if (axis < 0) UnsqueezeR(1 - axis, p);
  else UnsqueezeR(p->num_axes - axis, p);
}

/**
   Modifies 'p' in-place by removing an axis with dim=1 from the specified
   position (in the reversed numbering physically used in the pattern).  Updates
   p->code.  It is an error if 'p' did not, on entry, contain an axis with dim=1
   as position 'raxis' in the array.


   Modifies 'p' in-place by removing an axis with dim=1 from the
   specified position specified in the reversed numbering physically used in the
   pattern.  Updates p->code.  It is an error if 'p' did not initially contain
   an axis with dim=1 at position 'raxis' in the array.

   This function updates p->code.

   In the example below we show the dims in the order they appear in the
   physical array:
\verbatim
   SqueezeR(0, {1,3,4})  -> {3,4}
   SqueezeR(1, {5,1,7})  -> {5,7}
   SqueezeR(2, {8,1,9})  -> [error]
\endverbatim
     @param [in]    raxis   The reversed-order axis to be squeezed.
                            We require 0 <= raxis < p->num_axes and
                            p->dims[raxis] == 1.
     @param [in,out] p      The pattern from which we are removing an
                            axis.  Will have its num_axes reduced by 1
                            at exit, possibly its dims and strides
                            arrays changed, and its 'code' updated.
*/
void SqueezeR(int32 raxis, TensorPattern *p);


/**
   Modifies 'p' in-place by removing an axis with dim=1 (hence stride=0)
   located at the specified axis (as numbered in the public numbering).
   Equivalent to PyTorch's squeeze(), including its behavior with
   negative axis indexes; axis < 0 is interpreted as to num_axes - axis,
   i.e. the last axis.  It is an error if 'p' did not, on entry,
   contain an axis with dim=1 at position 'axis' (in the public numbering).

   Showing just the dims in the pattern, in the non-reversed order as
   exported by the API, some examples are:
\verbatim
    Squeeze([1,6,5], 0) -> [6,5]
    Squeeze([3,1,4], 1) -> [3,4]
    Squeeze([9,1,10], 2) -> error
    Squeeze([7,1], -1) -> [7]
\endverbatim

     @param [in]    axis    The index at which the extra axis is to appear.
                            We require -p->num_axes <= axis < p->num_axes
                            (negative axes are permitted, interpreted
                            as an offset from p->num_axes).
                            We require that the specified axis have
                            dim=1.
     @param [in,out] p      The pattern from which we are removing an
                            axis.  Will have its num_axes reduced by 1
                            at exit, possibly its dims and strides
                            arrays changed, and its 'code' updated.
 */
inline void Squeeze(int32 axis, TensorPattern *p) {
  if (axis < 0) SqueezeR(1 - axis, p);
  else SqueezeR(p->num_axes - 1 - axis, p);
}



/** Transpose the two specified axes of a TensorPattern

    @param [in] axis1  First axis to be transposed; must be in range
                       `[-p->num_axes, p->num_axes - 1]`,
                       with negative axis being interpreted as an offset
                       from p->num_axes.  This axis-index is in the
                       public numbering, not the reversed numbering
                       physically used in 'pattern'.
    @param [in] axis2  Second axis to be transposed; must be in range
                       `[-p->num_axes, t->num_axes - 1]`.
                       If identical to axis1, nothing will be done.
    @param [in,out] p  TensorPattern whose axes are to be transposed.
                       p->code is updated.
 */
void Transpose(int32 axis1, int32 axis2, TensorPattern *p) {
  Transpose(axis1, axis2, &(tensor->pattern));
}




/**
   Modifies 'p' in-place by removing an axis with dim=1 (hence stride=0)
   located at the specified axis (as numbered in the public numbering).
   Equivalent to PyTorch's squeeze(), including its behavior with
   negative axis indexes; axis < 0 is interpreted as to num_axes - axis,
   i.e. the last axis.  It is an error if 'p' did not, on entry,
   contain an axis with dim=1 at position 'axis' (in the public numbering).

   Showing just the dims in the pattern, in the non-reversed order as
   exported by the API, some examples are:
\verbatim
    Squeeze([1,6,5], 0) -> [6,5]
    Squeeze([3,1,4], 1) -> [3,4]
    Squeeze([9,1,10], 2) -> error
    Squeeze([7,1], -1) -> [7]
\endverbatim

     @param [in]    axis    The index at which the extra axis is to appear.
                            We require -p->num_axes <= axis < p->num_axes
                            (negative axes are permitted, interpreted
                            as an offset from p->num_axes).
                            We require that the specified axis have
                            dim=1.
     @param [in,out] p      The pattern from which we are removing an
                            axis.  Will have its num_axes reduced by 1
                            at exit, possibly its dims and strides
                            arrays changed, and its 'code' updated.
 */
inline void Squeeze(int32 axis, TensorPattern *p) {
  if (axis < 0) SqueezeR(1 - axis, p);
  else SqueezeR(p->num_axes - 1 - axis, p);
}

bool Broadcastable(const TensorPattern &a, const TensorPattern &b,
                   bool b_non_reducing = false);


/**  This function returns true if the dimensions of tensor patterns
     a, b and c are broadcastable in the PyTorch sense (meaning;
     after padding their dims on the left with ones to make them
     have the same num-axes, corresponding dimensions are either
     identical or 1).  See the version of Broadcastable() above
     for more information.

       @param [in] a  The dimensions of the first Tensor
       @param [in] b  The dimensions of the second Tensor
       @param [in] c  The dimensions of the third Tensor
       @param [in] c_non_reducing   If true, then we do not allow a dim of
                      c to be 1 while corresponding dims of a or b
                      are > 1.
 */
bool Broadcastable(const TensorPattern &a, const TensorPattern &b,
                   const TensorPattern &c, bool c_non_reducing = false);



/**
   Returns true if the 'dims' vectors of a and b are the same.
   Does not require the number of axes to be the same, so effectively
   it's testing that the dims are the same after padding on the left
   with dim=1 (here referring to the public, non-reversed numbering
   of the dims).

   This is a stronger condition than Broadcastable(a, b).
 */
bool SameDim(const TensorPattern &a, const TensorPattern &b);


/**
   Returns true if the 'dims' vectors of a, b and c are all the same.
   Does not require the number of axes to be the same, so effectively
   it's testing that the dims are the same after padding on the left
   with dim=1 (here referring to the public, non-reversed numbering
   of the dims).

   This is a stronger condition than Broadcastable(a, b, c).
 */
bool SameDim(const TensorPattern &a, const TensorPattern &b,
             const TensorPattern &c);


/**
   Compresses a TensorPattern by removing or combining as many axes as possible.
   This version is suitable for operations that do not rely on any kind
   of structure, such as zeroing or nonlinearities; the only equivalence
   maintained is equivalence of the set of memory locations covered.
   The order of the (dim,stride) pairs in the input does not affect the
   output.  The output (dim,stride) pairs will be ordered from
   greatest to least stride (note: all output strides will be positive).

      @param [in,out]  pattern   The pattern to be compressed

   Examples are below, where we write a TensorPattern as

   `{{dim1,dim2,..}, {stride1,stride2,..} [,offset] }`

   (the offset is written only if nonzero).

   (the curly braces in our notation imply that we are referring to the reversed
   ordering physically used in 'pattern', but actually this doesn't affect
   anything since the order of axes does not matter here as long as it is constent.

\verbatim
   Input pattern             Output pattern
     {{10},{1}}               {{10},{1}}
    {{3,4},{4,1}}             {{12},{1}}
    {{4,3},{1,4}}             {{12},{1}}
    {{9},{-1},8}                {{9},{1}}    // offset reduced by 8.
   {{2,3,4},{100,4,1}}        {{2,12},{100,1}}
\endverbatim
 */
void CompressOnePattern(TensorPattern *pattern);


/**
   Sorts the axes in 'pattern' from most negative to most positive
   stride value
   (negative to positive in the reversed numbering physically present in
   'pattern'; would be positive to negative in the public API).  Useful in
   testing equivalence of patterns, as CompressOnePattern() followed by
   SortAxes() leads to a normalized form.

     @param [in,out]  The pattern whose axes are to be sorted
                   from most negative to most positive stride (in the
                   physical ordering).
 */
void SortAxes(TensorPattern *pattern);


/**
   This version of SortAxes() sorts the axes in 'patterns' (which must be
   nonempty and all have the same number of axes), by ordering them from the
   most negative stride value in patterns[0] to the most positive stride value
   in patterns[0] (using the other patterns to disambiguate the order only in case
   of ties, which could only happen if some strides were zero).

     @param [in,out]  The patterns whose axes are to be sorted.  All
                    will have their axes subject to the same permutation.
                    The ordering is based on the strides of patterns[0],
                    but using the strides of later numbered patterns in
                    case of ties.
 */
void SortAxes(ArrayRef<TensorPattern*> patterns);

/**
  Multiplies all strides and the offset in 'pattern' by 'scale', which must be >
  0.  For now, will just crash if this causes integer overflow.

  This function is used in the memory-locking code if the same storage
  location is accessed using different dtypes (unlikely).
 */
void ScaleStridesAndOffset(int32 scale, TensorPattern *pattern);


// Used when we need an unordered_map containing TensorPattern.
class PatternHasher {
  size_t operator () (const TensorPattern &pattern) const;
};


/**
   Canonicalizes the pattern 'pattern' by calling CompressOnePattern() and
   then SortAxes().  The modified pattern will cover the same set of
   memory locations as the original one.
 */
void CanonicalizePattern(TensorPattern *pattern);


/*
  CompressTwoPatterns() is a special case of CompressPatterns() where there
  are exactly two patterns to be jointly compressed.  See documentation of
  CompressPatterns() for explanation.
 */
void CompressTwoPatterns(TensorPattern *a,
                         TensorPattern *b);


/**
   Compresses one or more TensorPattern by removing or combining as many axes as
   possible.  See the documentation for CompressOnePattern() to understand the
   basic concept of compressing a single TensorPattern to a pattern with possibly
   fewer axes (and maybe with negative strides converted to positive),
   which covers the same set of memory locations as the original Tensor.

   The difference with just calling CompressOnePattern() several times is
   that CompressPatterns() preserves the relationships between the tensors.

   Firstly, we require that all pairs of TensorPattern in 'patterns' be
   broadcastable: that is, Broadcastable(p1, p2) would hold for any
   p1, p2 in 'patterns'.  In the explanation below we will use a
   'permissive indexing' convention whereby if a Tensor has an axis
   with dim,stride (0, 1), we allow it to be indexed by any value
   (not just zero), so that all the tensors represented can accept the
   same set of index tuples.  Suppose for example that there are three
   patterns, p1, p2, p3, in 'patterns', with 4 axes.  Let max_axes
   larger of the num-axes of p1, p2 or p3, and let
   x = (i, j, k, l) be an index tuple that would be valid for a tensor
   with that many axes.  Each such x, when used as an index into p1, p2
   and p3 with 'permissive indexing' as mentioned above, will
   give us a tuple of memory-offsets (o1, o2, o3); o1, o2 and o3 are indexes
   into the respective data pointers.  Ranging over the set of index-tuples
   x, we get a set of memory-offset tuples; call this set S_in,
   and call the set that we would get if doing the same procedure
   on the output tensors (with their possibly changed num-axes), be
   S_out.  Let us represent the 'data_offset' output of this function
   as (in this case) a 3-tuple o.  Then the invariant that this
   function needs to satisfy is that:

        `S_in = S_out + o`

   (this equates two sets of 3-tuples, in our example) where we interpret the '+
   o' as adding to each element of the set.  The '+ o' above would only be
   necessary if any strides were negated; it is a tuple containing offsets, in
   elements, to be added to the data pointers of the respective output tensors.


      @param [in,out] patterns   An nonempty array of the patterns
                         to be jointly compressed.

      @return  Returns true if it made any change to the patterns,
               false if they were unchanged.  If false, the
               data_offsets will be set to zero.

 Examples are below, where we write a TensorPattern as
 `{{dim1,dim2,..}, {stride1,stride2,..}}`.

\verbatim
    src1                src2              dest1,offset1       dest2,offset2
  {{10},{1}}           {{10},{1}}        {{10},{1}},0        {{10},{1}},0  # no-op
  {{8},{1}}            {{1},{0}}         {{8},{1}},0         {{1},{0}},0   # no-op
  {{7},{-1}}           {{7},{1}}         {{7},{1}},-6         {{7},{-1}},6 # flip sign
 {{3,4},{4,1}}        {{3,4},{4,1}}      {{12},{1}},0         {{12},{1}},0 # combine dims
 {{3,4},{4,1}}        {{3,1},{4,0}}      {{3,4},{4,1}}        {{3,1},{4,0}} # can't combine, would be incompatible
 {{3,4},{4,1}}        {{1,1},{0,0}}      {{12},{1}}           {{1},{0}}    # combine
\endverbatim
 */
bool CompressPatterns(ArrayRef<TensorPattern*> patterns);

/**
   Compresses a TensorPattern by removing or combining as many axes as possible,
   while respecting certain invariances that are relevant when constructing
   'views' ('view' is PyTorch terminology; the NumPy equivalent is 'reshape').
   The "C" in the function name refers to C-style arrays.

    This function removes axes with dim=1.

   This function combines successive axes if the relationship of their
   dims and strides is what you would expect in a "C"-style array
   when the axes are listed in their non-reversed ordering (i.e.
   as exposed by class Tensor).


   Suppose that in pattern 'p' we had two successive axes physically numbered
   raxis, raxis+1, with p->dims[raxis] > 1 and p->dims[raxis+1] > 1
   and p->strides[raxis + 1] == p->strides[raxis] * p->dims[raxis],
   then this function will merge them into a single axis with dimension
   the product of the two dimensions..

    TODO...

   finish this if it turns out to be needed for something.


   with dims and
   strides (dim_a, dim_b) and (stride_a, stride_b), with dim_a > 1 and
   dim_b > 1.  If stride_a == stride_b * dim_b, then this function
   will merge them into a single axis with dimension (dim_a * dim_b)
   and stride stride_b.   (However, they won't be merged if it would
   result in a dimension exceeding the range of int32).

   The output pattern 'dest' is what you get if you keep applying the
   rules above until no further change is made.

   Examples are below, where we write a TensorPattern as
  `   {{dim1,dim2,..}, {stride1,stride2,..}}`.
\verbatim
   Input pattern             Output pattern
     {{10},{1}}               {{10},{1}}
    {{5,1},{1,1}}             {{5},{1}}
    {{9},{-1}}                {{9},{-1}}
   {2,3,4},{100,4,1}        {{2,12},{100,1}}
   {2,3,4},{100,-4,-1}        {{2,12},{100,-1}}
\endverbatim
 */
void CompressPatternC(TensorPattern *p);



/**
   Creates a TensorPattern corresponding to a requested 'view' of the matrix.
   ('view' is PyTorch terminology; the NumPy equivalent is 'reshape').

   The PyTorch/NumPy semantics are (I believe) as follows: Firstly, a view
   can/should only be created for a tensor whose layout in memory is as for a
   "C" array; suppose that the shape of array a is (9, 8), a "C" layout would
   imply strides of (8, 1).  A 'view' of this array simply implies interpreting
   the same block of memory as a "C" array with some other sequence of
   dimensions, say (3, 3, 8) or (8, 9) or (1, 72); any sequence whose product
   matches the number of elements in "a".

   Our semantics of "view" is the same as that of PyTorch/NumPy except that we
   impose fewer constraints on what strides the input Tensor cmay have.  Let the
   'view' of the array 'a' be 'b'.  As long as it is possible to find a tensor
   pattern for 'b' that would lead to the same relationship between the elements
   of 'a' and 'b' as what you would get by asking for the same "view" in
   PyTorch/NumPy assuming 'a' had had "C"-style strides (viewed in terms of
   indexed elements of and b, without regard to the physical memory layout), we
   allow it.


   Notes on implementation (glossing over ones in 'dims' which are easy to
   handle as a special case): we would first call CompressPattern on
   'pattern_in'.  Then we would attempt to find a correspondence with
   the dimensions of this compressed pattern and a partition of the
   sequence 'dims'.  For example, suppose the compressed pattern
   is (100, 9) and dims is (50, 2, 3, 3), then the partition would
   be (50, 2), (3, 3).  If this is not possible (e.g. if dims
   had been (30,10,3) instead), we return false.

   @param [in]  pattern_in   The input pattern for which we are trying to
                          find an alternative view
   @param [in]  dims  The sequence of dimensions corresponding to the
                      desired view.  Its product must be the same as the
                      product of pattern_in.dims.
   @param [out] pattern_out  The output pattern, if we were
                      successful (otherwise undefined).  Its 'dims'
                      will be the same as 'dims'.
   @return           Returns true on success (i.e. such a view existed),
                     and false otherwise.  This function will never return
                     false if 'pattern_in' had strides as for a "C" array
                     (i.e., if its properties' has_c_strides was true).

 */
bool CreateViewPattern(const TensorPattern &pattern_in,
                       ArrayRef<int32> dims,
                       TensorPattern *pattern_out);

/**
   Returns true if there is overlap between pattern1 and pattern2,
   meaning that pattern1's memory-index-set and pattern2's
   memory-index-set have nonempty intersection.
 */
bool PatternsOverlap(const TensorPattern &pattern1,
                     const TensorPattern &pattern2);

/**
   Returns true if pattern2's memory-index-set is a subset of pattern1's
   memory-index-set.  See glossary in tensor-pattern.h for explanation of
   memory-index-set.
 */
bool PatternIncludes(const TensorPattern &pattern1,
                     const TensorPattern &pattern2);


/**
   Returns true if the two patterns are equivalent in the sense that their
   memory-index-sets are the same.  See glossary in tensor-pattern.h for
   explanation.
 */
bool PatternsEquivalent(const TensorPattern &pattern1,
                        const TensorPattern &pattern2);


/**
   Outputs the memory-index-set corresponding to the pattern
   'pattern' to 's'.   See glossary in tensor-pattern.h for
   definitions.  This is strictly to be used in debugging
   code, as it is extremely inefficient.

      @param [in] pattern  The input pattern
      @param [out] s   The memory-index-set
 */
bool ToMemoryIndexSet(const TensorPattern &pattern,
                      std::unordered_set<int64> *s);



/**
   Outputs the memory-index-tuple-set corresponding to the pattern 'pattern' to
   's' (see tensor-pattern.h for definition).  For storage in 's', each tuple is
   converted into a single integer by a hashing function that should keep
   distinct tuples separate as long as the memory-indexes were not huge.  (We
   may output the actual tuples at some point in the future if they are ever
   needed).  This function is strictly to be used in debugging code, as it is
   extremely inefficient.

      @param [in] pattern  The input pattern
      @param [out] s   The memory-index-set
 */
bool ToMemoryIndexTupleSet(const ArrayRef<TensorPattern*>  patterns,
                           std::unordered_set<int64> *s);


/**
   Returns true if the two pattern-tuples are equivalent in the sense
   that their memory-index-tuple-sets are the same.  See glossary
   in tensor-pattern.h for explanation.
 */
bool PatternTuplesEquivalent(const ArrayRef<const TensorPattern*> &patterns1,
                             const ArrayRef<const TensorPattern*> &patterns2);


/**
   Class TensorPatternRebaser is an object that converts TensorPattern
   when memory layouts change.  The main use-case is when a base Variable
   (c.f. variable.h for definition) has a TensorPattern that is not
   contiguous (see tensor-pattern.h for definition of 'contiguous'), and
   its gradient Tensor is allocated contiguously.  This class is
   needed to convert patterns for Variables into patterns for their
   corresponding gradients.

   We make it an object rather than a function in order to avoid repetition when
   multiple patterns need to be rebased.
 */
class TensorPatternRebaser {

  /*
    Constructor.
       @param [in] src_pattern  The pattern that we are converting *from*,
                              e.g. the pattern of a Variable whose gradient
                              has a different layout from itself.
       @param [in] dest_pattern  The pattern that we are converting *to*.
                              Must have the same num_axes and the same dims
                              as 'src_pattern'.

    Let t be a valid index-tuple for src_pattern/dest_pattern, determined
    by their 'dims' and 'num_axes'.  Using t to index src_pattern and
    dest_pattern gives memory-indexes:
       m_src = src_pattern[t]
       m_dest = dest_pattern[t]
    View this object as a function from memory-indexes to memory-indexes
    (m_src -> m_dest), whose domain is the memory-index-set of src_pattern
    and whose range is the memory-index-set of dest_pattern.

    The purpose of this object is to modify patterns in a way that maps
    their memory-indexes with the same function.
  */
  TensorPatternRebaser(const TensorPattern &src_pattern,
                       const TensorPattern &dest_pattern);


  /**
     This function attempts to modify pattern->offset and pattern->strides in a
     way that does the mapping of memory-indexes m_src -> m_dest that is implied
     by the src_pattern and dest_pattern passed to the constructor.  That is,
     for any index-tuple t valid for 'pattern', the memory-index `pattern[t]`
     evaluated before and after calling this function gets mapped according
     to the function (m_src -> m_dest) mentioned in our documentation for
     the constructor.

     @param [in,out]  pattern  The pattern to be rebased.  Must, at entry,
                          satisfy `PatternIncludes(src_pattern, *pattern)`,
                          where `src_pattern` was the pattern passed to the
                          constructor.  On success (i.e. if this function
                          returns true), the condition
                          `PatternIncludes(dest_pattern, *pattern)` will
                          be satisfied.  On failure, the contents of
                          'pattern' is undefined.

     @return  Returns true if the conversion was possible.
   */
  bool Rebase(TensorPattern *pattern);

  private:

  // TODO: remove src_pattern_ and dest_pattern_ once everything
  // is debugged.  They are copies of the src_pattern and dest_pattern
  // passed to the constructor.
  TensorPattern src_pattern_;
  TensorPattern dest_pattern_;

  // If needs_conversion_ is false, it means the patterns don't need any conversion
  // at all (this is an optimization).
  bool needs_conversion_;

  // The 'offset' value of src_pattern_compressed (i.e. the src_pattern passed
  // to the constructor, which has been jointly compressed and normalized with
  // dest_pattern (to make all src_strides positive).
  int64 src_offset_;
  // The 'offset' value of dest_pattern_compressed
  int64 dest_offset_;

  // num_axes_ is the number of axes, not in the original src_pattern /
  // dest_pattern but after the two patterns have been jointly compressed and
  // then sorted from smallest to greatest stride in src_pattern.
  // src_strides_ are the resulting strides from src_pattern_compressed, and
  // dest_strides_ are the resulting strides from dest_pattern_compressed.

  // dest_pattern_ are the strides of the thus-modified src_pattern and
  // dest_pattern.  As an optimization, if src_strides and dest_strides end up
  // being the same, we set num_axes to zero and skip modifying the strides when
  // CompressPattern() is called.

  // Note: all of src_strides_[0] .. src_strides_[num_axes_ - 1] will be greater
  // than zero.  We can guarantee this because src_pattern and dest_pattern as
  // passed to the constructor had the same dims, so any axes with dim=1 would
  // have had dim=1 for both src and dest, hence they would have been removed by
  // CompressPatterns(), hence no strides would be zero after
  // CompressPatterns(); and CompressPatterns() normalizes the signs of the
  // strides so the first one (i.e. src_pattern) has positive strides.
  int32 num_axes_;
  int32 src_strides_[KALDI_TENSOR_MAX_DIM];
  int32 dest_strides_[KALDI_TENSOR_MAX_DIM];

  // The basic algorithm in Convert() is:
  //  First, add offset_ to its offset.
  //   Then:
  //     For each nontrivial axis of 'pattern', we are going to modify
  //     its stride as needed.
  //     Let that stride be `stride`, and the corresponding dim `dim`.
  //     Let `pstride = abs(stride)` be the absolute value of the stride
  //     (we'll modify that, and then restore the sign.
  //     positive.
  //



  // Converts a memory-index from the src to dest pattern.  This is applying,
  // to a single arbitrary memory-index m_src, the mapping (m_src -> m_dest);
  // see the comments above for explanation of this notation.
  // It is required that m >= 0 (otherwise it would not have been inside
  // the source pattern).
  int64 ConvertMemoryIndex(int64 m);

};


}  // namespace tensor
}  // namespace kaldi

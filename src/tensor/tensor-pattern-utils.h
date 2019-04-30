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


#ifndef KALDI_TENSOR_TENSOR_PATTERN_UTILS_H_
#define KALDI_TENSOR_TENSOR_PATTERN_UTILS_H_ 1


#include "tensor/tensor-common.h"
#include "tensor/tensor-pattern.h"
#include "tensor/array-ref.h"

// This header includes various functions operating on Patterns.
// See also tensor-pattern-extra-utils.h which contains the
// more obscure and less user-facing functions.

namespace kaldi {
namespace tensor {


// Returns true if the pattern code indicates that the pattern contains a
// negative stride.
inline bool ContainsNegativeStride(int32 pattern_code) {
  // 2048 is 1 << 11; 11th bit in code is set if code indicates negative stride.
  return (pattern_code | 2048) != 0;
}


/**
   This function converts an eaxis-index into an raxis-index, with no error
   checking (you would normally check afterward that the raxis-index is in the
   correct range).  Find "Eaxis-index:" and "Raxis-index:" in tensor-pattern.h,
   but basically and eaxis-index is an axis-index in the public numbering where
   we allow negative values to mean offsets from the end.
 */
inline int32 EaxisToRaxis(int32 eaxis, int32 num_axes) {
  return (eaxis < 0 ? 1 - eaxis : num_axes - 1 - eaxis);
}

/**
   Returns true if the pattern code indicates that the pattern contains a
   negative stride.  Caution: will return true if pattern_code was -1, so if you
   call this on a code on a valid Pattern where the code might be -1, all it
   means is that the Pattern "might" contain a negative stride.

     @param [in] pattern  The input pattern.  Must be valid;
                          return status is undefined otherwise.
     @return         Returns true if either the pattern's code was
                     -1 (meaning: not known), or if the code
                     indicates that a negative stride was present.
*/
inline bool PatternMightContainNegativeStride(
    const TensorPattern &pattern) {
  // 2048 is 1 << 11; 11th bit in code is set if code indicates negative stride.
  return (pattern.code | 2048) != 0;
}


/**
   Returns true if the pattern contains a negative stride.
   See tensor-pattern-utils-inl.h for implementation.

      @param [in] pattern   Input pattern.  Must be valid;
                            return status is undefined otherwise.
                            TODO: if we need this to work for, e.g.
                            valid- or valid-- patterns, find
                            the exact conditions.
      @return     Returns true if the pattern contained at
                  least one negative stride, false otherwise.
 */
inline bool ContainsNegativeStride(const Pattern &pattern);


// Returns true if the pattern code indicates that the raxis
// numbered 'raxis' (the r refers to the backwards numbering used
// in 'pattern') is 'trivial' (meaning: dim=1, stride=0).
inline bool AxisIsTrivial(int32 pattern_code, int32 raxis) {
  return (pattern_code | 1 << raxis) == 0;
}



/**
   This function copies pattern_in to pattern_out while removing
   trivial axes (i.e. axes with dim=1), reducing num_axes appropriately.

     @param [in] pattern_in   Input pattern.
     @param [out] pattern_out Output pattern; may not point to pattern_in.
                        At exit it will be the same as pattern_in except any
                        axes with dim=1 will have been removed and the num_axes
                        reduced.  Will be valid at output if pattern_in was
                        valid-1 at input.
*/
void RemoveTrivialAxes(const TensorPattern &pattern_in,
                       TensorPattern *pattern_out);


/**
   This function removes trivial axes (i.e. axes with dim=1) from 'pattern'.
   This version works in-place.

     @param [in,out] pattern   Pattern to be modified.  Any axes with dim=1
                         will be removed and the num_axes reduced.  Will be
                         valid at output if it was valid-1 at input.
 */
void RemoveTrivialAxes(TensorPattern *pattern);


/**
   This function returns a code that compactly represents information about
   which axes had dim != 1; which axis, if any, had stride == 1; and
   whether any axis had stride < 0.

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
   (None of the example bit-patterns below have this bit set.)  The
   underlying BLAS in most cases does not support negative strides so
   we deal with it by copying the data to a temporary with positive
   strides.

   The low-order KALDI_TENSOR_MAX_DIM bits have a 1 corresponding to
   axes where dim != 1, and a 0 if dim == 1 for that axis.  Axis zero
   in the private numbering (equal to the highest-numbered axis in the
   public numbering) is the rightmost/lowest-order of these bits.

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
   Copies a TensorPattern from `src` to `dest` while modifying it by inserting
   an axis with (dim=1,stride=0) at position `raxis` (specified in the
   private numbering).

     @param [in]    raxis   The index at which the extra axis is to appear.
                            We require 0 <= raxis <= p->num_axes.
     @param [in]    src    The source pattern.  Must be valid and have
                           NumAxes() < KALDI_TENSOR_MAX_DIM.
     @param [out]   dest   The destination pattern.  Is allowed to be the same
                           object as `src`.  Will be valid at exit if src
                           was valid at entry (which this function may not
                           check).
 */
void UnsqueezeR(int32 raxis, const TensorPattern &src, TensorPattern *dest);


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

     @param [in]    eaxis   The axis-index at which the extra axis is to appear,
                           with negatives allowed (see: "Eaxis-index" in glossary
                           in tensor-pattern.h).
     @param [in,out] p      The pattern to which we are adding an axis.
                            Will have its num_axes increased by 1
                            at exit, possibly its dims and strides
                            arrays changed, and its code updated.
 */
inline void Unsqueeze(int32 eaxis, TensorPattern *p) {
  UnsqueezeR(EaxisToRaxis(eaxis, p->num_axes));
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



/** Transpose the two specified axes (specified in the private/reversed
    numbering) of a TensorPattern.

    @param [in] raxis1  First axis to be transposed; must be in range
                        `[0, p->num_axes - 1]`
    @param [in] raxis2  Second axis to be transposed; must be in range
                        `[0, p->num_axes - 1]`
                        If identical to axis1, nothing will be done.
    @param [in,out] p  TensorPattern whose axes are to be transposed.
 */
void TransposeR(int32 raxis1, int32 raxis2, TensorPattern *p);


/** Transpose the two specified axes (specified in the private/reversed
    numbering) of a TensorPattern.

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
void TransposeR(int32 raxis1, int32 raxis2, TensorPattern *p);



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

/**  This function returns true if the dimensions of tensor patterns
     a, b and c are broadcastable in the PyTorch sense (meaning;
     after padding their dims on the left with ones to make them
     have the same num-axes, corresponding dimensions are either
     identical or 1).  The previous sentence is written in terms
     of the public numbering; in the private numbering it just means
     for each index `raxis` into the dims vector,
     either `a.dims[raxis] == b.dims[raxis]`, or one of them si 1.

       @param [in] a  The pattern of the first Tensor
       @param [in] b  The pattern of the second Tensor
       @param [in] b_non_reducing   If true, then we do not allow a dim of
                      b to be 1 while corresponding dim of a is >1.
       @return  Returns true if a and b are broadcastable (with
                an additional constraint that `a.dims[i] <= b.dims[i]` if
                `b_non_reducing == true`.
 */
bool Broadcastable(const TensorPattern &a, const TensorPattern &b,
                   bool b_non_reducing = false);


/**  This function returns true if the dimensions of tensor patterns
     a, b and c are broadcastable in the PyTorch sense, which is
     the same as
     `Broadcastable(a, b) && Broadcastable(b, c) && Broadcastable(a, c)`.
     See the 2-argument version of Broadcastable for more information.

       @param [in] a  The pattern of the first Tensor
       @param [in] b  The pattern of the second Tensor
       @param [in] c  The pattern of the third Tensor
       @param [in] c_non_reducing   If true, then we do not allow a dim of
                      c to be 1 while corresponding dims of a or b
                      are > 1.
       @return  Returns true if a, b and c are broadcastable (with
                an additional constraint that
                `max(a.dims[i], b.dims[i]) <= c.dims[i]` if
                `c_non_reducing == true`).

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
   This version is suitable for operations that do not rely on any kind of
   structure, such as zeroing or nonlinearities; the only equivalence maintained
   is equivalence of the set of memory locations covered (the memory-index-set).
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
   Sorts the axes in 'pattern' from most negative to most positive stride
   in private numbering, equivalent to sorting from most positive to
   most negative stride in public numbering.

   TODO: decide whether to change this to sort on abs(stride), or
   maybe create another version that does sort on abs(stride), if there
   are situations where this turns out to be useful.

     @param [in,out]  The pattern whose axes are to be sorted
                   from most negative to most positive stride (in the
                   physical ordering).
 */
void SortAxes(TensorPattern *pattern);


// TODO: document this.
inline void CanonicalizePattern(TensorPattern *pattern) {
  CompressOnePattern(pattern);
  SortAxes(pattern);
}

// TODO: document this.  This will later be replaced with
// a more efficient version.
inline void CanonicalizePattern(contst TensorPattern &pattern_in,
                                TensorPattern *pattern_out) {
  *pattern_out = pattern_in;
  CanonicalizePattern(pattern_out);
}

/**
   This pattern checks that 'pattern' is valid and in canonical form (see
   glossary for the meaning).  CanonicalizePattern() will modify a valid pattern
   to put it in canonical form.
 */
bool IsCanonical(const TensorPattern &pattern);


/**
   Returns the number of elements in the pattern, computed as the
   product of the dims.  ('pattern' is expected to either be valid or
   to at least satisfy the uniqueness property for this to actually give
   the number of elements, but this is not checked).
*/
int64 NumElements(const TensorPattern &pattern);


/**
   This version of SortAxes() sorts the axes in 'patterns' (which must be
   nonempty and all have the same number of axes), by ordering them from the
   most negative stride value in patterns[0] to the most positive stride value
   in patterns[0], using the strides in the other patterns to disambiguate the
   order only in case of ties (which could only happen if some strides were
   zero), and then the dims in the same order if the strides are all the same
   (the strides would only be the same if they were zero, if the patterns were
   valid).  Roughly, it's a lexical order on the (strides, then dims) of the
   patterns.  Note: the most-negative-to-most-positive ordering is in terms of
   the private, `raxis` numbering; it would be most-positive-to-most-negative in
   the public numbering.

   TODO: work out what the ordering should be; should it really be negative-to-
   positive, or based on abs(stride), and do we need disambiguation with the
   dims?

   TODO: do we even need this??

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

  This function is used in the memory-locking code if the same storage location
  is accessed using different dtypes (which is unlikely).
 */
void ScaleStridesAndOffset(int32 scale, TensorPattern *pattern);



/// Hashing object, used when we need an unordered_map containing TensorPattern.
class PatternHasher {
  size_t operator () (const TensorPattern &pattern) const;
};


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

   In technical terms (and you will have to follow definitions several deep
   in the glossary to find all the definitions), this operation
   preserves the memory-index-tuple-set of the Pattern-tuple, and
   also the memory-index-set of each of the Patterns (we have to specify
   the part after "and" to disallow swapping the Patterns).

   Note: while the first Pattern will have no negative strides at output,
   the others may.

      @param [in,out] patterns   An nonempty array of the patterns
                         to be jointly compressed.

      @return  Returns true if it made any change to the patterns,
               false if they were unchanged.

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
   while preserving the memory-index-set of the pattern (see glossary for
   explanation), and also while respecting certain invariances that are relevant
   when constructing 'views' ('view' is PyTorch terminology; the NumPy
   equivalent is 'reshape').  The "C" in the function name refers to C-style
   arrays.  Basically what this function does is a highly restricted subset
   of what CompressOnePattern() does.

   This function removes axes with dim=1.

   This function combines successive axes if the relationship of their
   dims and strides is what you would expect in a "C"-style array
   when the axes are listed in their non-reversed ordering (i.e.
   as exposed by class Tensor).

   Suppose that in pattern 'p' we had two successive axes physically numbered
   raxis, raxis+1, with p->dims[raxis] > 1 and p->dims[raxis+1] > 1
   and p->strides[raxis + 1] == p->strides[raxis] * p->dims[raxis],
   then this function will merge them into a single axis whose dimension
   is the product of the dimensions of the two original axes.
   (However, they won't be merged if it would
   result in a dimension exceeding the range of int32).

   TODO...  finish this if it turns out to be needed for something.
   I'm not sure if it will be.


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
   @return           Returns true on success (i.e. such a view could be
                     created), and false otherwise.  This function will
                     never return false if 'pattern_in' had strides as
                     for a "C" array (i.e., if HasCStrides(pettern_in)
                     returns true).

 */
bool CreateViewPattern(const TensorPattern &pattern_in,
                       ArrayRef<int32> dims,
                       TensorPattern *pattern_out);


/**
   This is like PyTorch's slice() / narrow() functions.
   It selects a range of dimensions on one of the axes.  It is similar to
   indexing with a range in Python, like A[10:20].

      @param [in] eaxis  Eaxis-index (see glossary in tensor-pattern.h) on which
                         to possibly reduce the dimensionality.
      @param [in] start  Starting index; must be in range [0, t->Dim(eaxis) - 1]
      @param [in] end    Ending index; must be in the range [start + 1, t->Dim(eaxis)]
      @param [in,out] pattern  TensorPattern to be modified.  Will be valid at
                         exit if it was valid at entry.

   See also: the other overloaded version of Slice() which accepts the 'step'
   parameter; and Select(), which is similar but also reduces the num-axes.
 */
void Slice(int32 eaxis, int32 start, int32 end, TensorPattern *pattern);



/**
   Copy one Pattern to another while modifying it by by selecting one index from
   a specified axis (specified in the public numbering), of a TensorImpl `t`,
   reducing the num_axes by one.

       @param [in] eaxis Eaxis-index (see glossary in tensor-pattern.h) on which
                         to possibly reduce the dimensionality.
       @param [in] index Index to select; must be in range
                         [0, t->Dim(eaxis) - 1].
       @param [in,out] src   TensorPattern which is to be copied; must be valid,
                         but we don't guarantee to check this.
       @param [out] dest TensorPattern which we are copying to and modifying.
                         It is allowed to be the same object as 'src'.
                         Will be valid if src was valid.
*/
void Select(int32 eaxis, int32 index,
            const TensorPattern &src, TensorPattern *dest);


/**
   This function returns true if 'pattern' has the same strides
   as 'C' array with the same dimensions would have.  (Note:
   we are referring here to the public numbering of the axes).
   For example, an array of dims [3, 4, 5], if it were
   a "C" array, would have strides of [20, 5, 1].  As a special
   case, since our Patterns use stride=0 for axes with dim=1,
   we treat that zero as a wildcard; that is, if there
   is a stride value for which the array would have "C" strides
   then we'll return true.

     @param [in] pattern  The pattern we are checking.  It is expected
                     to satisfy Valid(pattern), but this function does not
                     check this.

     @return  Returns true if this pattern has 'C' strides, and
              false otherwise.   (See note above about axes
              with dim=1).
*/
void HasCStrides(const TensorPattern &pattern);

/**
   Returns true if there is overlap between pattern1 and pattern2,
   meaning that pattern1's memory-index-set and pattern2's
   memory-index-set have nonempty intersection.
 */
bool PatternsOverlap(const TensorPattern &pattern1,
                     const TensorPattern &pattern2);

/**
   Returns true if the memory-index-set of this pattern forms a contiguous
   range, otherwise false.  (Note: this is not the same as PyTorch's notion of
   contiguous; see HasCStrides()).  Caution: the interface may later be changed
   to allow caching of this property in the 'properties' field.
*/
bool IsContiguous(const TensorPattern &pattern);


/**
   Returns true if the lowest memory-index of 'pattern' is zero (see
   "Justified" in glossary in pattern.h.
   (see also: ComputeMinAndMaxMindex()).
*/
bool IsJustified(const TensorPattern &pattern);


/**
   This is the same is IsContiguous(pattern) &&
   StartsFromZero(pattern).
*/
bool IsContiguousAndJustified(const TensorPattern &pattern);




}  // namespace tensor
}  // namespace kaldi


#include "tensor/tensor-pattern-utils-inl.h"

#endif KALDI_TENSOR_TENSOR_PATTERN_UTILS_H_

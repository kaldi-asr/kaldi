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

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {



/**
   This function returns a code that compactly represents the number of axes in
   'pattern' and whether the dimension for each axis is greater than 1.  This is
   used in fast lookup of which algorithm to choose.

   The 3 least significant bits of 'pattern' represent the number of dims.
   From there, in order from least to most significant, is a 1 for
   each dimension that is >1, or a zero if that dimension is 1.

   The examples below will use c++14 binary literals, although
   the code doesn't use them.  In the notation below, in dims vectors,
   x is a stand-in for 'any number greater than 1'.  Note that
   the order of the numbers in the 'dims' vectors below and the
   corresponding bit positions saying whether they are >1, are
   in opposite directions (left to right vs. right to left).

    0b00000000  0x00  dims=(), a scalar
    0b00000001  0x01  dims=(1)
    0b00001001  0x09  dims=(x)
    0b00000010  0x02  dims=(1,1)
    0b00001010  0x0A  dims=(1,2)
    0b00010010  0x01  dims=(x,1)
    0b00011010  0x1A  dims=(x,x)
    0b00000011  0x03  dims=(1,1,1)
    0b00001011  0x0B  dims=(1,1,x)
    0b00010011  0x13  dims=(1,x,1)
    0b00100011  0x23  dims=(x,1,1)
    0b00101011  0x2B  dims=(x,1,x)
    0b00111011  0x3B  dims=(x,x,x)

  See also GetPatternCode(), which includes the same information but
  also stride-related information.
 */
int32 GetDimsCode(const TensorPattern &pattern);


/**
   This function returns a code that compactly represents:
     - the num-axes in 'pattern'
     - which dims in 'pattern' were >1
     - which stride in 'pattern', if any, was equal to 1.  (It's not
       possible for more than one stride to equal 1 in a valid Tensor).

   The low-order 9 bits are as returned by GetDimsCode(), and are
   explained in its documentation.

   The next 3 bits are a number that is: 0 if no dims had stride=1;
   otherwise, one plus the axis that had stride=1.  The maximum
   number of bits in the output is 12.

   The explanation below will use c++14 binary literals, although the code
   doesn't use them.  In the notation below, in dims vectors, X or x is a
   stand-in for 'any number greater than 1', and upper-case x indicates that the
   axis has stride=1.  Note that the order of the numbers in the 'dims' vectors
   below and the corresponding bit positions saying whether they are >1, are in
   opposite directions (left to right vs. right to left).
   The ' at the 8th bit is to make the bit-string easier to parse (but notice
   that the number representing the stride information starts from the 9th bit).

    0b0000'00000000  0x000  dims=(), a scalar
    0b0000'00000001  0x001  dims=(1)
    0b0000'00001001  0x009  dims=(x)
    0b0010'00001001  0x209  dims=(x)
    0b0000'00000010  0x002  dims=(1,1)
    0b0000'00001010  0x00A  dims=(1,x)
    0b0010'00001010  0x20A  dims=(1,X)
    0b0000'00010010  0x001  dims=(x,1)
    0b0000'00010010  0x001  dims=(x,1)
    0b0000'00011010  0x01A  dims=(x,x)
    0b0010'00011010  0x21A  dims=(x,X)
    0b0100'00011010  0x41A  dims=(X,x)
    0b0000'00000011  0x003  dims=(1,1,1)
    0b0000'00001011  0x00B  dims=(1,1,x)
    0b0010'00001011  0x00B  dims=(1,1,X)

    0b00010011  0x13  dims=(1,2,1)
    0b00100011  0x23  dims=(2,1,1)
    0b00101011  0x2B  dims=(2,1,2)
    0b00111011  0x3B  dims=(2,2,2)


 */
int32 GetPatternCode(const TensorPattern &pattern);


/**
   This function returns true if the two 'dims' vectors supplied are
   broadcastable in the PyTorch sense.  (Note that they are required to be valid
   'dims' vectors, which means that all elements are >0).

   The rule as is as follows.  Firstly, if a and b are not the same length, then
   insert 1's at the beginning of the shorter one to make them the same length
   'num_axes'.  Then for each axis 0 <= i < num_axes, if a[i] != b[i]
   and neither of them equals 1, a and b are not broadcast compatible.
   If the test above did not fail for any axis i, a and b are broadcast
   compatible.

     @param [in] a  The dimensions of the first Tensor
     @param [in] b  The dimensions of the second Tensor
     @param [in] b_non_reducing   If you set this flag to true, then
                    we impose a more stringent test for compatibility,
                    namely: in the test above, after padding to
                    the same length, we also fail if any
                    a[i] > b[i], i.e. we don't allow b[i] to be 1
                    and a[i] to be greater than 1.
                    The reason for the name is that this is a
                    condition we would need to impose for non-reducing
                    operations on b (such as:  b += a).

        @return  Returns true if a and b are broadcast-compatible
                 AND `(!b_non_reducing || b >= a).`, interpreting >= as
                 'all elements are >, after left-padding with ones`.
 */
bool BroadcastCompatible(ArrayRef<int32> a, ArrayRef<int32> b,
                         bool b_non_reducing = false);


/**  This function returns true if the dimensions of tensor patterns
     a and b are broadcastable in the PyTorch sense; this is
     a convenience wrapper that calls BroadcastCompatible on their
     dims turned into ArrayRef<int32>.

     See the documentation for the other overloaded version of this
     function, which is more detailed.  And see also Broadcastable()
 */
bool BroadcastCompatible(const TensorPattern &a, const TensorPattern &b,
                         bool b_non_reducing = false);


/**  This function returns true if the dimensions of tensor patterns
     a, b and c are broadcastable in the PyTorch sense (meaning;
     after padding their dims on the left with ones to make them
     have the same num-axes, corresponding dimensions are either
     identical or 1).

     See the first version of BroadcastCompatible (accepting ArrayRef's)
     for an explanation of its meaning.  Briefly: left-pad with ones;.
     then all dims must be the same, except that 1's are allowed
     even if different from dims greater than one.

       @param [in] a  The dimensions of the first Tensor
       @param [in] b  The dimensions of the second Tensor
       @param [in] c  The dimensions of the third Tensor
       @param [in] c_non_reducing   If true, then we do not allow a dim of
                      c to be 1 while corresponding dims of a or b
                      are > 1.
 */
bool BroadcastCompatible(const TensorPattern &a, const TensorPattern &b,
                         const TensorPattern &c, bool c_non_reducing = false);


/**
   Pads the axes of a and b with leading axes with (dim=1, stride=0),
   as required so that their NumAxes() is the same.  For instance,
   if their respective dimensions were (10,5) and (8,10,5), after
   padding they would be (1,10,5) and (8,10,5) respectively.

      @param [in,out] a  First TensorPattern to be padded
      @param [in,out] b  Second TensorPattern to be padded
 */
void PadAxes(TensorPattern *a, TensorPattern *b);

/**
   Pads the axes of a and b with leading axes with (dim=1, stride=0),
   as required so that their NumAxes() are all the same.

      @param [in,out] a  First TensorPattern to be padded
      @param [in,out] b  Second TensorPattern to be padded
      @param [in,out] c  Second TensorPattern to be padded
 */
void PadAxes(TensorPattern *a, TensorPattern *b, TensorPattern *c);


/**
   Broadcastable(a, b) returns true if a.num_axes == b.num_axes and
   for each 0 <= i < a.num_axes, either a.dims[i]  == b.dims[i]
   or one of the two dims equals 1.  This is a stricter condition
   than BroadcastCompatible because it requires the same number of
   axes.
 */
bool Broadcastable(const TensorPattern &a, const TensorPattern &b);


/**
   Broadcastable(a, b, c) is equivalent to
   Broadcastable(a, b) && Broadastable(b, c) && Broadcastable(a, c);
   is provided for speed and convenience.

   This condition is expected to be satisfied when you are doing, say, a binary
   operation on a and b with the output in c.  Any axis i on which c.dims[i] ==
   1 and one or both of a.dims[i] and b.dims[i] is not 1, would be interpreted
   as some kind of reduction, depending on the context; probably summation.
   Thus, with suitable dim=1 axes inserted, matrix multiplication can be
   interpreted as elementwise multiplication with summation.

   For instance, suppose we are multiplying a matrix with dims
   (m, n) by a matrix with dims (n, k), producing an output
   of dimension (m, k).  This could be interpreted as elementwise
   multiplication with reduction, with dimensions:

     a = (m, n, 1), b = (1, n, k), c = (m, 1, k).

   (The extra dims are just used for our internal meta-manipulation, they would
   never be exposed to the user).  Of course, we would eventually want to turn
   this into a conventional invocation of something like BLAS matrix-multiply;
   but viewing it in this way will, in more complex cases, allow us to spot
   opportunities for combining some tensor dims that would otherwise
   be hard to spot except on a case-by-case basis.
 */
bool Broadcastable(const TensorPattern &a, const TensorPattern &b,
                   const TensorPattern &c)



/**
   Compresses a TensorPattern by removing or combining as many axes as possible.
   This version is suitable for operations that do not rely on any kind
   of structure, such as zeroing or nonlinearities; the only equivalence
   maintained is equivalence of the set of memory locations covered.
   The order of the (dim,stride) pairs in the input does not affect the
   output.  The output (dim,stride) pairs will be ordered from
   greatest to least stride (note: all output strides will be positive).

      @param [in]  src   The pattern to be compressed
      @param [in]  src_properties  Properties of 'src'; required to
                          be accurate (behavior is undefined otherwise,
                          e.g. if you provide some other pattern's properties).
      @param [out] dest   A simplified-as-much-as-possible pattern that
                          covers the same set of memory locations as 'src' (when
                          combined with the offset below).  'dest' will
                          contain only nonnegative strides.
      @param [out] data_offset  A number that we would have to add to
                          the data pointer of the source Tensor so
                          that 'dest' would cover the same set of
                          elements.  It will always be zero if 'src'
                          was free of negative strides.
   Examples are below, where we write a TensorPattern as
    `{{dim1,dim2,..}, {stride1,stride2,..}}`.

\verbatim
   Input pattern             Output pattern            Output offset
     {{10},{1}}               {{10},{1}}                  0
    {{3,4},{4,1}}             {{12},{1}}                  0
    {{4,3},{1,4}}             {{12},{1}}                  0
    {{9},{-1}}                {{9},{1}}                  -8
   {2,3,4},{100,4,1}        {{2,12},{100,1}}              0
\endverbatim
 */
void CompressOnePattern(TensorPattern *pattern,
                        int64 *data_offset);

/*
  Compress two TensorPatterns by combining axes (and possibly
  flipping the sign of their strides and changing the data offset)
  The type of compression involved is the same as for CompressOnePattern
  (meaning we are doing some kind of operation that doesn't care about
  the structure, such as an element-by-element nonlinearity).

  The difference from calling CompressOnePattern() twice is that this function
  needs to preserve the relationship between the tensors whose pattern is src1
  and src2.  Suppose that a tensor with pattern src3 was the result of this
  elementwise operation satisfying Broadcastable(src1, src2, src3); there is
  only one such pattern.  Let x be a tuple which would be a valid index for the
  tensor with pattern src3.  Let us use an extended indexing convention
  whereby if an axis of src1 or src2 has dimension 1, we allow that axis to be
  indexed by any value, which would not affect the memory location because the
  stride is zero.  Then each such tuple x leads to a different pair of memory
  locations (p1, p2) in the tensors corresponding to patterns src1, src2.  The
  invariance that this function must preserve is that the set of memory-location
  pairs (p1, p2) must be the same in the output tensors (with their
  appropriately moved data pointers), as in the input tensors.

  What this means in practice is that we need to do the same operations on src1
  and src2.  For example, if flipping the sign of an axis of src1 we would have
  to flip that of src2, and if merging two axes of src1 we would have to merge
  the same two axes of src2.

    @param [in] src1  The first source pattern.
    @param [in] src2  The second source pattern.
                      We require Broadcastable(src1,src2) == true.
    @param [out] dest1  Compressed pattern out corresponding to src1.  Will
                     be free of negative strides (but dest2 might not be).
    @param [out] dest_offset1  Data offset that we'd need to add to src1's
                     data pointer before using the pattern 'dest1'
    @param [out] dest1  Compressed pattern out corresponding to src2.
                     Might not be free of negative strides if some dimensions
                     of src1/src2 had strides of opposite sign.
    @param [out] dest_offset1  Data offset that we'd need to add to src1's
                     data pointer before using the pattern 'dest1'


 */
void CompressTwoPatterns(const TensorPattern &src1,
                         const TensorPattern &src2,
                         TensorPattern *dest1,
                         int64 *data_offset1,
                         TensorPattern *dest2,
                         int64 *data_offset2);


/**
   Compresses one or more TensorPattern by removing or combining as many axes as
   possible.  See the documentation for CompressOnePattern() to understand the
   basic concept of compressing a single TensorPattern to a pattern with possibly
   fewer axes (and maybe with negative strides converted to positive),
   which covers the same set of memory locations as the original Tensor.

   The difference with just calling CompressOnePattern() several times is
   that this preserves the relationships between the tensors.

   Firstly, we require that all pairs of TensorPattern in 'patterns' be
   broadcastable: that is, Broadcastable(p1, p2) would hold for any
   p1, p2 in 'patterns'.  In the explanation below we will use a
   'permissive indexing' convention whereby if a Tensor has an axis
   with dim,stride (0, 1), we allow it to be indexed by any value
   (not just zero), so that all the tensors represented can accept the
   same set of index tuples.  Suppose for example that there are three
   patterns, p1, p2, p3, in 'patterns', with 4 axes.  Let max_dim
   be the 'combined' dimension, which contains the max of the dims
   of the corresponding axes of p1,p2,p3, and let
   x = (i, j, k, l) be an index tuple that would be valid for a tensor
   of dim max_dim.  Each such x, when used as an index into p1, p2
   and p3 with 'permissive indexing' as mentioned above, will
   will give us a tuple of memory-offsets (o1, o2, o3) (indexes
   into the respective data pointers).  Ranging over the set of such
   x, we get a set of memory-offset tuples; call this set S_in,
   and call the set that we would get if doing the same procedure
   on the output tensors (with their possibly changed num-axes), be
   S_out.  Let us represent the 'data_offset' output of this function
   as (in this case) a 3-tuple o.  Then the invariant that this
   function needs to satisfy is that:
        `S_in = S_out + o`
   where we interpret the '+ o' as adding to each element of the set.
   Interpret the above as: one set of 3-tuples == another set of 3-tuples.

   Of course, the 3 tensors and 4 axes mentioned here is just an example.

      @param [in,out] patterns   An array of 1 <= size <= 4 of the patterns
                         to be jointly compressed.
      @param [out]  data_offsets  Pointer to an array of the same size
                        as patterns, which on output will contain
                        offsets to be added to the data pointers.

   Examples are below, where we write a TensorPattern as
    `{{dim1,dim2,..}, {stride1,stride2,..}}`.

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
void CompressPatterns(ArrayRef<TensorPattern> patterns,
                      int64_t *data_offsets);

/**
   Compresses a TensorPattern by removing or combining as many axes as possible,
   while respecting certain invariances that are relevant when constructing
   'views' ('view' is PyTorch terminology; the NumPy equivalent is 'reshape').
   The "C" in the function name refers to C-style arrays.

    This function removes axes with dim=1.

    This function combines successive axes if the relationship of their
    dims and strides is what you would expect in a "C"-style array.
    Suppose that in 'src' we had two successive axes with dims and
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
void CompressPatternC(const TensorPattern &src,
                      const TensorPatternProperties &src_properties,
                      TensorPattern *dest);


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




};


}
}

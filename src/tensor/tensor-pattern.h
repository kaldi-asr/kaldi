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

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {


// This enum with one value is a trick to allow you to
// emulate indexing schemes like, say, A[10:].
// In C++ you'd do A(all,10).
enum RangeEnum { all };

/**
   struct Range represents an integer or a range of integers (e.g. as used in
   indexing).  It emulates Python's range().

   There are various possibilities of what Range can contain, enumerated below.
   Be careful: we use {a,b,c} to denote the actual class members, not the
   arguments to constructors, which mimic the arguments of expressions with colons
   in Python's indexing with ':'

   For purposes of explanation I will assume we are indexing a 1-dimensional
   array a, but this struct is also used for multi-dimensional indexing.

   Examples are below (showing members {begin,end,step}, where inf means
   std::numeric_limits<int64>::max()):


   Literal contents     Python equivalent,     How obtained             Elements of array
   of Range struct      indexing array a     using constructors           you would get

    {0,inf,1}          a[:], a[0:]          Range(all), Range(0,all)    all of them

    {0,10,2}           a[:10:2], a[0:10:2]   Range(0,10,2)             [0,2,4,8]

    {0,-1,1}           a[:-1], a[0:-1]       Range(0,-1)                all but the last

    {10,2,-1}          a[10:2:-1]           Range(10,2,-1)              [10,9,...3]

    {inf,inf,-1}        a[::-1]             Range(all,all,-1)            all, reversed order

    {-3,-2,1}          a[-3:-2]            Range(-3,-2)             third-from-last element only

    {10,0,inf}         a[10]              10 (implicit; RangeExt constructor)    the 10th element, removing axis


*/
struct Range {
  int32 begin;
  int32 end;
  int32 step;

  static inline int32 inf() { return std::numeric_limits<int32>::max(); }

  // The default constructor leaves the range undefined.
  Range() { }

  Range(RangeEnum): begin(0), end(inf()), step(1) { }

  explicit Range(int32 end): begin(0), end(end), step(1) { }

  Range(int32 begin, int32 end, int32 step = 1):
      begin(begin), end(end), step(1) { }

  Range(int32 begin, RangeEnum, int32 step = 1):
      begin(begin), end(inf()), step(step) { }

  Range(RangeEnum, int32 end, int32 step = 1):
      begin(inf()), end(end), step(step) { }

  Range(RangeEnum, RangeEnum, int32 step = 1):
      begin(inf()), end(inf()), step(step) { }
};

/**
  struct RangeExt is used in situations, such as indexing, where what we have
  might be a Range (like, in numpy, indexing with something that has a colon);
  or it might simply be an integer.  There are no new members.  The reason we
  don't just make this an additional constructor of Range is that we want it
  so if someone does Range(10) it is interpreted as 0 through 9, but if
  you do just 10 it means the index 10.  You can't have an explicit and
  implicit constructor taking the same type: hence this child class.

  Note that numpy's A[1] is not the same as A[1:2] because the former returns a
  tensor with one fewer axes.
*/
struct RangeExt: public Range {
  RangeExt(Range r): Range(r) { }

  // implicit
  RangeExt(int32 index):
      Range(index, 0, inf());
};


/**
   This function, used in indexing operations, takes a Range that may have, say,
   negative 'end' or end equal to Range::inf(), and turns it into actual
   numbers with begin and end both in the range [0,dim].  So, for instance, if
   the range had `end = -1`, it would be turned into `dim - 1`; or if `end` was
   Range::inf(), it would be interpreted as `dim`.

   Raises an exception the resulting range is empty.
 */
void MakeRangeExplicit(int32 dim, Range *range);


/*
  This struct stores the dimension and strides of a Tensor.

  The main thing to watch out for is that the dimensions of 'dims' and 'strides'
  to look at is not 0 ... num_axes, but KALDI_TENSOR_MAX_DIM - num_axes
  ... KALDI_TENSOR_MAX_DIM - 1.  The last dimension is always located at
  KALDI_TENSOR_MAX_DIM - 1, i.e. the dims and strides are always
  right-justified.  In addition, for unused axes, we always maintain dim=1 and
  stride=0. This happens to be quite convenient due to the standard broadcasting
  rules in things like PyTorch.

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

    for 0 <= i < KALDI_TENSOR_MAX_DIM
       dims[i] > 0
       if i < KALDI_TENSOR_MAX_DIM - num_axes, then dims[i] = 1.
       if dims[i] = 1, then strides[i] = 0.
       if dims[i] != 1, then strides[i] != 0

    ... plus the uniqueness property.

  Note: in the public interface of class Tensor, if you ask for
  dim(i) it will return pattern.dims[KALDI_TENSOR_MAX_DIM - num_axes + i].


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
  int32 dims[KALDI_TENSOR_MAX_DIM];
  int32 strides[KALDI_TENSOR_MAX_DIM];
  int32 code;  // pattern code; see GetPatternCode() in tensor-pattern-utils.h
               // for details.

  // We may later add methods to this.

  // Checks that the TensorPattern is valid, assuming it is part of a Tensor.
  // I.e. that it satifies all the properties mentioned above.
  // Returns true if valid, false if not valid.
  bool Check();
};

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


#endif  // KALDI_TENSOR_TENSOR_COMMON_H_

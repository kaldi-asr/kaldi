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
  int64_t begin;
  int64_t end;
  int64_t step;

  static inline int64_t inf() { return std::numeric_limits<int64_t>::max(); }

  // The default constructor leaves the range undefined.
  Range() { }

  Range(RangeEnum): begin(0), end(inf()), step(1) { }

  explicit Range(int64_t end): begin(0), end(end), step(1) { }

  Range(int64_t begin, int64_t end, int64_t step = 1):
      begin(begin), end(end), step(1) { }

  Range(int64_t begin, RangeEnum, int64_t step = 1):
      begin(begin), end(inf()), step(step) { }

  Range(RangeEnum, int64_t end, int64_t step = 1):
      begin(inf), end(end), step(step) { }

  Range(RangeEnum, RangeEnum, int64_t step = 1):
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
  RangeExt(int64_t index):
      Range(index, 0, inf());
};

/**
enum IndexingType{
  kIndexingTypeRange,
  kIndexingTypeNumber,
  kIndexingTypeTensor
};

// This struct is used when indexing with mixed types. Ror
// example:
// Tensor a(...), b(...);
// Tensor c = a(1, b, Range(0,-1), Range(all));

struct IndexingArg {
  IndexingArg(const Tensor &tensor);
  IndexingArg(int64_t index);
  IndexingArg(Range range);

  IndexingType itype;
  int64_t index;
  std::shared_ptr<Tensor> tensor {nullptr};
  Range range;
  };*/

/**
   This function, used in indexing operations, takes a Range that may have, say,
   negative 'end' or end equal to Range::inf(), and turns it into actual
   numbers with begin and end both in the range [0,dim].  So, for instance, if
   the range had `end = -1`, it would be turned into `dim - 1`; or if `end` was
   Range::inf(), it would be interpreted as `dim`.

   Raises an exception the resulting range is empty.
 */
void MakeRangeExplicit(Range *range, int64_t dim);



/*
  This struct stores the dimension and strides of a Tensor.  The following
  describes the properties that a TensorPattern will always have.

  These properties are stricter than some other frameworks, such as PyTorch,
  which allow the users to manually add dimensions with stride 0 and dim > 1 so
  that a lower-dimensional quantity can masquerade as one with a higher
  dimension.  We require that it never be possible to access the same memory
  location using two different tuples of indexes.  We also don't allow zero dims
  (i.e. a tensor must not be empty); if you want an empty Tensor, just use a
  null pointer.  In addition, require that the stride equal zero for any
  axis that has dim = 1.

  Our requirements on a TensorPattern are:

    0 <= num_axes <= 5
    for 0 <= axis < num_axes:
       dims[i] > 0
       if dims[i] = 1, then strides[i] = 0.
       if dims[i] != 1, then strides[i] != 0
    ... plus the uniqueness property.

  The uniqueness property means that we must not be able to access
  the same memory location via two different tuples of indexes).
  Recause testing this property exactly would be difficult in general
  without bringing in number theory, we test a slightly stronger version
  of it that covers all cases we are likely to encounter.
*/
struct TensorPattern {
  int64_t num_axes;
  int64_t dims[KALDI_TENSOR_MAX_DIM];
  int64_t strides[KALDI_TENSOR_MAX_DIM];
  // We may later add methods to this.

  // Checks that the TensorPattern is valid, assuming it is part of a Tensor.
  // I.e. that it satifies all the properties mentioned above.
  bool Check();
};

struct TensorPatternProperties {
  // Below are cached properties that are derived from the underlying data in
  // struct TensorPattern.

  // The number of elements in the Tensor, which equals the product
  // of dims[0] .. dims[num_axes - 1].  Will always be >0.
  int64_t num_elements;

  // is_contiguous means that the data form a contiguous block in memory; it is
  // not the same as PyTorch's is_contiguous which is a stronger condition,
  // implying also that the strides are as for a `C-style` array.
  bool is_contiguous;

  // has_c_strides means that the strides are as if this was a "c"-style
  // multidimensional array, meaning that (using Python wrap-around indexing
  // conventions as if strides was an array of dimension 'num_axes'),
  // strides[-1] == 1, strides[-1] == dims[-1], strides[-2] = dims[-1] *
  // dims[-1], and so on.  This is the same as PyTorch's is_contiguous.
  bool has_c_strides;

  void UpdateProperties(const TensorPattern &pattern);
};


/**
   Compresses a TensorPattern by removing or combining as many dimensions
   as possible.  This version is suitable for 'flat' operations that do
   not rely on any kind of structure, such as zeroing or nonlinearities.

      @param [in]   src   The pattern to be compressed
      @param [in]  src_properties  Properties of 'src'; required to
                          be accurate (behavior is undefined otherwise,
                          e.g. if you provide some other pattern's properties).
      @param [out] dest   A simplified-as-much-as-possible pattern that
                          covers the same set of elements as 'src' (when
                          combined with the offset below).  'dest' will
                          be free of negative strides.
      @param [out] data_offset  A number that we would have to add to
                          the data pointer of the source Tensor so
                          that 'dest' would cover the same set of
                          elements.  It will always be zero if 'src'
                          was free of negative strides.

   Examples are below, where we write a TensorPattern as
  `   {{dim1,dim2,..}, {stride1,stride2,..}}

   Input pattern             Output pattern            Output offset
     {{10},{1}}               {{10},{1}}                  0
    {{3,4},{4,1}}             {{12},{1}}                  0
    {{9},{-1}}                {{9},{1}}                  -8
   {2,3,4},{100,4,1}        {{2,12},{100,1}}              0


 */
void CompressPatternFlat(const TensorPattern &src,
                         const TensorPatternProperties &src_properties,
                         TensorPattern *dest,
                         int64_t *data_offset)

/**

 */
void CompressPattern(const TensorPattern &src,
                     const TensorPatternProperties &src_properties,
                     TensorPattern *dest);




/**


 */
bool CreateViewPattern(const TensorPattern &pattern_in,
                       const TensorPatternProperties &properties_in,
                       ArrayRef<int64_t> dims,
                       TensorPattern *pattern_out,
                       TensorPatternProperties *properties_out);



};


}
}

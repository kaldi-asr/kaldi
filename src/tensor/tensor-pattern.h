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
  // conventions as if strides was an array with 'num_axes' axes),
  // strides[-1] == 1, strides[-1] == dims[-1], strides[-2] = dims[-1] *
  // dims[-1], and so on.  This is the same as PyTorch's is_contiguous.
  // this->has_c_strides implies this->is_contiguous.
  bool has_c_strides;

  void UpdateProperties(const TensorPattern &pattern);
};


/**
   Compresses a TensorPattern by removing or combining as many axes as possible.
   This version is suitable for 'flat' operations that do not rely on any kind
   of structure, such as zeroing or nonlinearities; the only equivalence
   maintained is equivalence of the set of memory locations covered.
   The order of the (dim,stride) pairs in the input does not affect the
   output.  The output (dim,stride) pairs will be ordered from
   greatest to least stride (note: all output strides will be positive).

      @param [in]   src   The pattern to be compressed
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
void CompressPatternFlat(const TensorPattern &src,
                         const TensorPatternProperties &src_properties,
                         TensorPattern *dest,
                         int64_t *data_offset);

/*
  Compress two TensorPatterns by combining axes (and possibly
  flipping the sign of their strides and changing the data offset)
  The type of compression involved is the same as for CompressPatternFlat
  (meaning we are doing some kind of operation that doesn't care about
a  the structure, such as an element-by-element nonlinearity).

  The difference from calling CompressPatternFlat() twice is that this function
  is only allowed to do the same operation to src1 and src2, e.g. if combining
  two axes of src1 we would also have to combine the same two axes of src2.

    @param [in] src1  The first source pattern.
    @param [in] src2  The second source pattern.  The assumption is that src1
                     and src2 are participating in some kind of operation like
                     copying, or elementwise multiplication.  The patterns
                     must satisfy src1.NumAxes() == src2.NumAxes(), and
                     for each axis, either src1.dims[axis] == src2.dims[axis],
                     or one of those two dimensions equals 1 (so there would be
                     some kind of broadcasting).  The
    @param [out] dest1  Compressed pattern out corresponding to src1.  Will
                     be free of negative strides (but dest2 might not be).
    @param [out] dest_offset1  Data offset that we'd need to add to src1's
                     data pointer before using the pattern 'dest1'
    @param [out] dest1  Compressed pattern out corresponding to src2.
                     Might not be free of negative strides if some dimensions
                     of src1/src2 had strides of opposite sign.
    @param [out] dest_offset1  Data offset that we'd need to add to src1's
                     data pointer before using the pattern 'dest1'

  TODO: examples
 */
void CompressPatternsFlat(const TensorPattern &src1,
                          const TensorPattern &src2,
                          TensorPattern *dest1,
                          int64_t *data_offset1
                          TensorPattern *dest2,
                          int64_t *data_offset2);


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
    and stride stride_b.

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
                       ArrayRef<int64_t> dims,
                       TensorPattern *pattern_out);




};


}
}

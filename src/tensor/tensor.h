/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {


// Similar to llvm/PyTorch's ArrayRef, this is a lightweight way to store an
// array (zero or more elements of type T).  The array is not owned here; it
// will generally be unsafe to use an ArrayRef as other than a local variable.
template <typename T>
struct ArrayRef {
  // Note:
  uint64_t size;
  int64_t *data;

  // Lots to do here.

  inline int64_t operator (uint64_t i) const {
    KALDI_ASSERT(i < size);
    return data[i];
  }

  // cast to std::vector; for cases where you might need to
  // change the contents or keep it more permanently.
  operator std::vector<int64_t> () const;
};


enum {
  kCpuDevice = 0,
  kCudaDevice = 1
} DeviceType;

// We may later add a device number (like which GPU we are using),
// once we support multiple GPUs.
struct Device {
  DeviceType device_type;
  // operator ==, probably, maybe constructors.
};


// 'Storage' contains a single allocated region (on CPU or GPU, according
// to 'device').
struct Storage {
  void *data;
  size_t num_bytes;
  Device device;

  // Note: will throw if allocation fails (for now).
  Storage(Device device, size_t num_bytes);

  // Destructor deallocates 'data'.  For now there is no
  // concept of a custom allocator or an allocator object, we just use our CuDevice stuff for cuda
  // allocation and posix_memalign for CPU allocation (obviously we need
  // to make sure 'data' is aligned in most specific way we might need).
  // in future we might choose
  // to add that.
  ~Storage();
};


enum DataType {
  // We will of course later extend this with many more types, including
  // integer types and half-precision floats.
  kFloatDtype = 0,
  kDoubleDtype = 1
};

enum StridePolicy {
  kCstrides = 0,
  kCopyStridesIfContiguous = 1
};

#define KALDI_TENSOR_MAX_DIM 5



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

    {inf,inf,1}        a[::-1]             Range(all,all,-1)            all, reversed order

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
};

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
  describes the properties that a Tensor will always have (note: we
  also use TensorDim inside implementation code in ways such that these
  properties do not all hold).

  These properties are stricter than some other frameworks, such as PyTorch,
  which allow the users to manually add dimensions with stride 0 and dim > 1 so
  that a lower-dimensional quantity can masquerade as one with a higher
  dimension.  We require that it never be possible to access the same memory
  location using two different tuples of indexes.  We also don't allow zero dims
  (i.e. a tensor must not be empty); if you want an empty Tensor, just use a
  null pointer.  In addition, require that the stride equal zero for any
  axis that has dim = 1.

  Our requirements on a TensorDim are:

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

struct TensorDim {
  uint64_t num_axes;
  int64_t dims[KALDI_TENSOR_MAX_DIM];
  int64_t strides[KALDI_TENSOR_MAX_DIM];
  // We may later add methods to this.

  // Checks that the TensorDim is valid, assuming it is part of a Tensor.
  // I.e. that it satifies all the properties mentioned above.
  bool Check();
};

struct TensorDimProperties {
  // Below are cached properties that are derived from the underlying data in
  // struct TensorDim.

  // The number of elements in the Tensor, which equals the product
  // of dims[0] .. dims[num_axes - 1].  Must be >0.
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

  void UpdateProperties(const TensorDim &dim);
};


/**
   A Tensor is a multi-dimensional array (up to 5 dimensions) of types such as
   float or double (and eventually ints).  Multiple Tensors may point to data
   allocated from the same Storage.  Class Tensor contains enough elements that
   it makes sense most of the time to pass it around by reference (Tensor&) or
   by pointer (e.g. Tensor* or std::shared_pointer<Tensor>).  This is unlike
   in PyTorch where there is a separate TensorImpl class and Tensor really just
   contains a pointer to it.

   Most of the operations that you would do on a Tensor (like addition,
   multiplication and so on) are declared out-of-line in tensor-functions.h.
 */
class Tensor {
 public:
  /// Return the number of axes (a number in {0,1,2,3,4}).  In mathematical
  // contexts, this is sometimes known as the rank of the tensor, or sometimes
  // even its dimension, but these terms are ambiguous so we avoid them, and use
  // the terms 'number of axes' or 'axis' throughout.
  inline int64_t NumAxes() const { return dim_.num_axes; }

  // Return reference to the struct containing the dimension and
  // stride info.
  const TensorDim &DimAndStrides() const { return dim_; }

  // Return an array containing dimensions of the tensor; equivalent to
  // .shape in PyTorch.  Dims().size() will equal NumAxes().
  inline ArrayRef<int64_t> Dims() const { return ArrayRef{dim_.num_axes, dim_.dims}; }

  // Returns the dimension on this axis, a number >= 1.  Result is
  // undefined if axis < 0 or axis >= NumAxes().
  inline int64_t Dim(int64_t axis) const { return dim_.dims[axis]; }

  // Returns an array containing the strides of the tensor.
  // Strides().size() will equal NumAxes().
  inline ArrayRef<int64_t> Strides() const { return ArrayRef{dim_.num_axes, dim_.strides}; }

  // Returns the stride on this axis.  Will be zero if the corresponding
  // dimension is 1, and otherwise nonzero (but not necessarily positive).
  inline int64_t Stride(int64_t axis) const { return dim_.strides[axis]; }

  // Returns the number of elements in the Tensor; must be > 0.
  inline int64_t NumElements() const { return derived_.num_elements; }

  // Returns true if the data forms a contiguous block in memory.
  // (not the same as 'contiguous()' in PyTorch, which also requires
  // that the strides be 'C'-style.
  inline bool IsContiguous() const { return derived_.is_contiguous; }


  // Returns true if the strides for this array are what you would
  // expect if you were to construct a Tensor from this->Dims();
  // this means "C"-style strides, except that any axis with dimension=1
  // has its stride set to zero.  This is our equivalent of PyTorch's
  // contiguous().
  inline bool HasNormalStrides() const { return derived_.has_c_strides; }

  // Return the data type.
  DataType Dtype() const { return dtype_; }

  // Indexing operators.  All of these return Tensors which reference the same
  // underlying data as the original Tensor.  We could have done this with just
  // a single indexing operator taking 5 args of type RangeExt defaulting to
  // `all`, but we provide separate versions for each num-args for efficiency.
  // You can provide an int64_t where RangeExt is expected; it will be
  // converted to a special struct of type Range. See the documentation for type
  // Range, and the table which it contains.  If a is a Tensor with 1 axis, a(0)
  // will return a scalar Tensor (0 axes
  //
  // Any of these indexing operators can operate on Tensors with more axes;
  // trailing axes will be left alone.

  // this operator () taking int64_t is only provided in the one-arg case as a
  // convenience; in any case, RangeExt can be constructed from int64_t with the
  // same effect.
  Tensor operator () (int64_t i0) const;
  Tensor operator () (RangeExt s0) const;
  Tensor operator () (RangeExt s0, RangeExt s1) const;
  Tensor operator () (RangeExt s0, RangeExt s1, RangeExt s2) const;
  Tensor operator () (RangeExt s0, RangeExt s1, RangeExt s2,
                      RangeExt s3) const;
  // A particularly complicated example showing what is possible:
  // Tensor a(...);
  // Tensor b = a(all,10,Range(0,5),Range(all,all,-1),all)
  Tensor operator () (RangeExt s0, RangeExt s1, RangeExt s2,
                      RangeExt s3, RangeExt s4) const;


  // For a scalar Tensor (NumAxes() == 0) returns the item, cast to
  // float (if it was not already float); throws if NumAxes() > 0.
  explicit operator float() const;
  // For a scalar Tensor (NumAxes() == 0) returns the item, cast to
  // double (if it was not already double); throws if NumAxes() > 0.
  explicit operator double() const;
  // For a scalar Tensor (NumAxes() == 0) returns the item, cast to
  // int64_t (if it was not already int64_t); throws if NumAxes() > 0.
  explicit operator int64_t() const;


  // For a Tensor storing floats, returns the data pointer cast to float;
  // otherwise, throws.  (note: this is const only as it doesn't change the
  // Tensor meta-info, but you could change the data using the pointer).
  explicit operator float* () const;
  // For a Tensor storing doubles, returns the data pointer cast to float;
  // otherwise, throws.  (note: this is const only as it doesn't change the
  // Tensor meta-info, but you could change the data using the pointer).
  explicit operator double* () const;



  // Assignment operation which sets all elements to a constant.  Valid
  // for Tensors of any floating point type.
  const Tensor & operator = (float f);

  // Transpose the two axes by swapping their dims and strides without changing
  // the underlying data in memory.  This modifies *this;
  void Transpose(int64_t axis1 = 0, int64_t axis2 = 1);


  // Copy constructor that copies the metadata while sharing the underlying
  // data.
  Tensor (const Tensor &other) = default;

  // Move assignment.  Does not copy the data.
  Tensor(Tensor &&other);

  // Copy data from tensor 'other'.  Requires this Dims() and other.Dims()
  // be compatible, meaning that they are the same, except it's OK for
  // a dim of 'other' to be 1 and a dim of *this to be >1 (we will
  // broadcast, i.e. copy).
  void CopyData(const Tensor &other);

  // Construct a Tensor with the supplied dimensions; and if set_zero is true,
  // zero it.
  Tensor(ArrayRef<int64_t> dims, bool set_zero = false);

  // Construct a Tensor with
  Tensor(TensorDim &dim, StridePolicy policy, bool set_zero = false);



 private:
  // The tensor dim and strides.
  TensorDim dim_;
  // Cached properties that depend on dim_.
  TensorDimProperties derived_;
  // The data-type of this tensor.
  DataType dtype_;

  // The raw data pointer.  Will be cast to a pointer of the appropriate
  // type before indexing.
  void *data_;

  // The storage region where the data resides.  data_ does not necessarily
  // equal storage_->data; it may be more than that, e.g. if this is a view
  // to part of another Tensor.
  std::shared_ptr<Storage> storage_;
};




/*
  This is the 'gradient information' that class Variable stores for a Tensor
  when it is initialized with requires_grad = true (or is a result of
  an operation on Variables one of which had requires_grad = true).
  This does not give you access to the underlying Variables; doing it
  like this makes reference counting easier (no loops).  The GradFunc
  will store any pointers to the original Variable that it may have
  needed.

  Users will rarely need to interact directly with this struct directly.
 */
struct TensorGrad {
  // The gradients corresponding to the input variables, which
  // we may need to update.  Some subset of these may be nullptr,
  // corresponding to input Variables for which no gradient
  // was required.
  std::vector<std::shared_ptr<TensorGrad> > inputs;

  // is_view is
  bool is_view{false};

  // The device we
  Device device;

  // The dimension of the Tensor for which this is the gradient.  Used
  // to set up 'grad' when needed.
  TensorDim dim;

  // 'offset' is only inspected if this is a view; it is the offset
  // (in elements) from the
  // 'inputs' will just contain one member, which is the gradient for the source
  // Variable, and we use 'dim' and 'offset' to construct the sub-tensor).
  int64_t offset;

  // This stores the gradient (if we already have one), or nullptr if not.
  std::unique_ptr<Variable> grad{nullptr};

};


/**
   class Variable is the same as class Tensor but augmented with autograd
   machinery.  The overall design is quite similar to PyTorch, and the C++
   is similar to flashlight.  If you are only familiar with PyTorch's
   python frontend, class Variable is equivalent to af.tensor.
 */
class Variable {
  using GradFunc = std::function<
    void(std::vector<Variable>& inputs, Variable &grad_output)>;
  using GradHook = std::function<void(Variable *grad)>;




};

typedef std::unique_ptr<Storage>




};

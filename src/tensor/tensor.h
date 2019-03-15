/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {


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


enum {
  kFloatDtype = 0,
  kDoubleDtype = 1
} DataType;

#define KALDI_TENSOR_MAX_DIM 5



/*
  This struct stores the dimension and strides of a Tensor.  The following
  describes the properties that a Tensor will always have (note: we
  also use TensorDim inside implementation code in ways such that these
  properties do not all hold).

  These properties are stricter than some other frameworks, such as PyTorch,
  which allow the users to manually add dimensions with stride 0 (and dim>1) so
  that a lower-dimensional quantity can masquerade as one with a higher
  dimension.  We require that it never be possible to access the same
  memory location using two different tuples of indexes.  We also
  don't allow zero dims (i.e. a tensor must not be empty); if you want an
  empty Tensor, just use a null pointer.

    0 <= num_axes <= 5
    for 0 <= axis < num_axes:
       dims[i] > 0

  The strides may take any value, including zero or negative, as long as the
  uniqueness property is satisfied (i.e. must not be possible to access the
  same memory location using two different tuples of indices.

*/

struct TensorDim {

  int64_t num_axes;
  int64_t dims[KALDI_TENSOR_MAX_DIM];
  int64_t strides[KALDI_TENSOR_MAX_DIM];
  // We may later add methods to this.

  // Checks that the TensorDim is valid, assuming it is part of a Tensor.
  // I.e. that it satifies the properties mentioned above.
  bool Check();
};

struct TensorDimProperties {
  // Below are cached properties that depend on a TensorDim.

  // The number of elements in the Tensor, which equals the product
  // of dims[0] .. dims[num_axes - 1].  Must always be >0.
  int64_t num_elements;

  // is_contiguous means that the data form a contiguous block in memory; it is
  // not the same as PyTorch's is_contiguous which is a stronger condition; our
  // has_expected_strides is equivalent to that.
  bool is_contiguous;

  // has_expected_strides means that the strides are as if this was a "c"-style
  // multidimensional array, meaning that (using Python wrap-around indexing
  // conventions as if strides was an array of dimension 'num_axes'),
  // strides[-1] == 1, strides[-1] == dims[-1], strides[-2] = dims[-1] *
  // dims[-1], and so on.  This is the same as PyTorch's is_contiguous.
  bool has_expected_strides;

  void UpdateProperties(const TensorDim &dim);
};



class Tensor {
 public:
  //  ...

 private:
  // The tensor dim and strides.
  TensorDim dim_;
  // Cached properties that depend on dim_.
  TensorDimProperties derived_;
  // The data-type of this tensor.
  DataType dtype_;

  // The raw data pointer
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
  std::unique_ptr<Tensor> grad{nullptr};


};


class Variable {
    using GradFunc = std::function<
      void(std::vector<Variable>& inputs, const Variable& grad_output)>;


};

typedef std::unique_ptr<Storage>




};

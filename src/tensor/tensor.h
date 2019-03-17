/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {

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
  inline int64_t NumAxes() const { return pattern_.num_axes; }

  // Return reference to the struct containing the dimension and
  // stride info.
  const TensorPattern &DimAndStrides() const { return pattern_; }

  // Return an array containing dimensions of the tensor; equivalent to
  // .shape in PyTorch.  Dims().size() will equal NumAxes().
  inline ArrayRef<int64_t> Dims() const { return ArrayRef{pattern_.num_axes, pattern_.dims}; }

  // Returns the dimension on this axis, a number >= 1.  Result is
  // undefined if axis < 0 or axis >= NumAxes().
  inline int64_t Dim(int64_t axis) const { return pattern_.dims[axis]; }

  // Returns an array containing the strides of the tensor.
  // Strides().size() will equal NumAxes().
  inline ArrayRef<int64_t> Strides() const { return ArrayRef{pattern_.num_axes, pattern_.strides}; }

  // Returns the stride on this axis.  Will be zero if the corresponding
  // dimension is 1, and otherwise nonzero (but not necessarily positive).
  inline int64_t Stride(int64_t axis) const { return pattern_.strides[axis]; }

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

  /**
     Construct a new Tensor with freshly allocated underlying data with
     the data type, device and dimension the same as `other`.

       @param [in]  other  The tensor that we are taking metadata from (we
                    are not sharing its underlying data).
       @param [in]  sp   The stride policy; if kCopyStrides then we use
                       strides with the same sign and size-order as
                       `other`, while filling in any gaps if `other`
                       was not contiguous, if kCstrides then we use
                       "C" style strides for any dimensions != 1.
       @param [in]  ip   The data initialize policy

     The strides will not be the same as 'other' if other.IsContiguous() ==
     false, but the ordering of the strides (smaller vs. larger) and their
     signs will remain the same.
  */
  Tensor(const Tensor &other, StridePolicy sp, InitializePolicy ip);



  /** Construct a Tensor with freshly allocated data.
       @param [in] dims    The dimensions of the tensor (zero to 5
                    positive integers).
       @param [in] dtype   The data type to use
       @param [in] device  The device to put the data on
       @param [in] set_zero   If true, set the tensor to zero.  If false,
                        the contents will be undefined.
   */
  Tensor(ArrayRef<int64_t> dims, DataType dtype, Device device,
         bool set_zero = false);

  /**
     Construct a Tensor with the dimensions and strides provided.  This differs
     from the constructor taking `ArrayRef<int64_t> dims` in that it will use
     the strides in `pattern` (except that if the data in `pattern` is not
     contiguous, it will make it contiguous by filling in any gaps).  This means
     that, for example, if you use this constructor on a 2-dimensional Tensor
     that has been transposed and thus has a column-major layout, the resulting
     Tensor will also have a column-major layout.

       @param [in] pattern  The dimension and stride information that
                  this tensor should match (although we will fill gaps
                  to make it contiguous)
       @param [in] dtype   The data type to use
       @param [in] device  The device to put the data on
       @param [in] set_zero   If true, set the data to zero.  If false,
                        the contents will be undefined.

  */
  Tensor(TensorPattern &pattern, DataType dtype, Device device,
         InitializePolicy p);


 private:
  // The tensor dim and strides.
  TensorPattern pattern_;
  // Cached properties that depend on pattern_.
  TensorPatternProperties derived_;
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
  TensorPattern dim;

  // 'offset' is only inspected if this is a view; it is the offset
  // (in elements) from the
  // 'inputs' will just contain one member, which is the gradient for the source
  // Variable, and we use 'dim' and 'offset' to construct the sub-tensor).
  int64_t offset;

  // This stores the gradient (if we already have one), or nullptr if not.
  std::unique_ptr<Variable> grad{nullptr};

};


/**
   class Variable is somewhat like class Tensor but augmented with autograd
   machinery.  Because autograd requires a rather 'functional' way of doing
   things (i.e. is not super friendly to in-place operations), the functions
   that operate on class Variable will tend to be ones that return something,
   rather than in-place operations.

   The overall design is quite similar to PyTorch, and the structure
   of the the C++ code is similar to flashlight.  If you are only familiar with
   PyTorch's python frontend, class Variable is rougtly equivalent to what they
   expose as af.tensor.
 */
class Variable {
  using GradFunc = std::function<
    void(const std::vector<Variable>& inputs, TensorGrad *grad_output)>;
  using GradHook = std::function<void(TensorGrad *grad)>;



  /** Constructor from a Tensor.
       @param [in] data  Pointer to the source Tensor
       @param [in] requires_grad    If requires_grad argument is true,
                the gradient w.r.t. this Variable will be computed if and when
                you call Backward() on a Variable that depends on it.
                The same as requires_grad in PyTorch.
  */
  Variable(const std::shared_ptr<Tensor> &data, bool requires_grad);



  /**
   * Creates a Variable which wraps the array and inputs specified
   * @param[in] data array to the stored in the Variable
   * @param[in] inputs a vector specifying inputs for this Variable
   * @param[in] gradFunc function specifying how to calculate gradient of the
   * input Variables
   */
  Variable(std::shared_ptr<Tensor> &data, std::vector<Variable> inputs,
           GradFunc gradFunc);



};

typedef std::unique_ptr<Storage>




};

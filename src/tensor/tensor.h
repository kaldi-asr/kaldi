// tensor/tensor.h

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_TENSOR_TENSOR_H_
#define KALDI_TENSOR_TENSOR_H_ 1

#include "tensor/tensor-common.h"
#include "tensor/tensor-pattern.h"
#include "tensor/tensor-impl.h"
#include "tensor/storage.h"

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

  inline bool Initialized() { return data_ != NULL; }

  /// Return the number of axes (a number in {0,1,2,3,4}).  In mathematical
  // contexts, this is sometimes known as the rank of the tensor, or sometimes
  // even its dimension, but these terms are ambiguous so we avoid them, and use
  // the terms 'number of axes' or 'axis' throughout.
  // Caution: the numbering of axes in the Tensor interface is different
  // than in TensorImpl::pattern.  Here they are numbered from zero;
  // in TensorImpl::pattern they are shifted to the right so
  // the last axis is KALDI_TENSOR_MAX_DIM - 1.
  inline int32 NumAxes() const { return impl_.pattern.num_axes; }

  const TensorImpl &Impl() { return impl_; }

  const TensorMeta &Meta() { return reinterpret_cast<TensorMeta&>(impl_); }

  // Return reference to the struct containing the dimension and
  // stride info.
  const TensorPattern &Pattern() const { return impl_.pattern; }

  // Return a vector containing dimensions of the tensor; equivalent to
  // .shape in PyTorch.  Dims().size() will equal NumAxes().
  // This cannot return a const reference because the
  // dims are stored internally in reversed order.
  std::vector<int32> Dims() const;

  // Return a vector containing the strides of the tensor.
  // Strides().size() will equal NumAxes().
  std::vector<int32> Strides() const;


  // Returns the dimension on the supplied axis
  //  @param [in] axis  Axis on which dimension is required, with
  //                    -NumAxes() <= axis < NumAxes(); negative axis
  //                    is interpreted as an offset from NumAxes().
  //  @return        Returns the dimension on this axis, a number >= 1.
  inline int32 Dim(int32 axis) const { return impl_.Dim(axis); }

  // Returns the stride on the supplied axis (using the public axis numbering)
  //  @param [in] axis  Axis on which stride is required, with
  //                    -NumAxes() <= axis < NumAxes(); negative axis
  //                    is interpreted as an offset from NumAxes().
  //  @return          Returns the stride on this axis, which will be 0 if
  //                   Dim(axis) == 1, and otherwise nonzero.
  inline int32 Stride(int32 axis) const { return impl_.Stride(axis); }

  // Returns the number of elements in the Tensor; will be > 0,
  // and will equal the product of Dims().
  int64 NumElements() const;

  // Returns true if the data forms a contiguous block in memory.
  // (not the same as 'contiguous()' in PyTorch, which also requires
  // that the strides be 'C'-style; for that, see HasCStrides().
  bool IsContiguous() const;

  // Returns true if the strides for this array are what you would
  // expect if you were to construct a Tensor from this->Dims();
  // this means "C"-style strides, except that any axis with dimension=1
  // has its stride set to zero.  This is our equivalent of PyTorch's
  // contiguous().
  bool HasCStrides() const;

  // Return the data type.
  DataType Dtype() const { return dtype_; }

  /**
     Indexing operator taking one arg.  Returns a Tensor referencing
     the same underlying data as this Tensor.


     You can provide an int32 where RangeExt is expected; it will be
     converted to a special struct of type Range. See the documentation for type
     Range, and the table which it contains.
     will return a scalar Tensor (0 axes

     Any of these indexing operators can operate on Tensors with more axes;
     trailing axes will be left alone.

  // this operator () taking int32 is only provided in the one-arg case as a
  // convenience; in any case, RangeExt can be constructed from int32 with the
  // same effect.

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


  // For a Tensor with NumElements() == 1, returns the element, cast to float
  explicit operator float() const;
  // For a Tensor with NumElements() == 1, returns the element, cast to double
  explicit operator double() const;
  // For a Tensor with NumElements() == 1, returns the element, cast to int32
  explicit operator int32() const;

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
  // Negative axes are allowed, and interpreted as NumAxes() - axis.
  void Transpose(int32 axis1 = 0, int32 axis2 = 1);


  // Constructor which does not really initialize the Tensor.  impl_.pattern,
  // derived_ and dtype_ may contain nonsense.
  Tensor(): data_(NULL) { }

  // Copy constructor that copies the metadata while sharing the underlying
  // data.
  Tensor (const Tensor &other) = default;

  // Move assignment.
  Tensor(Tensor &&other): impl_(other.impl_) { storage.swap(other.storage_); }

  /**
     Construct a new Tensor with freshly allocated underlying data with
     the data type, device and dimensions the same as `other`.  The strides
     will be the same order as 'other' if sp == kCopyStrides.

       @param [in]  meta  The metadata we are copying the dims, device,
                       dtype and possibly strides from
       @param [in]  sp   The stride policy; if kCopyStrideOrder then we use
                       strides with the same sign and size-order as
                       `other`, while filling in any gaps if `other`
                       was not contiguous, if kCstrides then we use
                       "C" style strides, i.e. we ignore the stride
                       order of the source.  (Of course, we set strides
                       to zero for any axes with `dim=1`, as required by our
                       framework).
       @param [in]  ip   The data initialization policy
  */
  Tensor(const Meta &meta,
         StridePolicy sp);


  /** Construct a Tensor with freshly allocated data.
       @param [in] dims    The dimensions of the tensor (zero to 5
                    positive integers).
       @param [in] dtype   The data type to use
       @param [in] device  The device to put the data on

       Example:  `Tensor a({3,4,5}, kDoubleDtype, kCpuDevice);`
   */
  Tensor(ArrayRef<int32> dims, DataType dtype, Device device);

  /** Construct a Tensor with freshly allocated data, and device ==
      `GetDefaultDevice().`.

       @param [in] dims    The dimensions of the tensor (zero to 5
                    positive integers).
       @param [in] dtype   The data type to use

       Example:  `Tensor a({3,4,5}, kDoubleDtype);`
   */
  Tensor(ArrayRef<int32> dims, DataType dtype);

  /** Construct a Tensor with freshly allocated data, data type ==
      `GetDefaultDtype()`,

       @param [in] dims    The dimensions of the tensor (zero to 5
                    positive integers).
       @param [in] device  The device to put the data on

       Example:  `Tensor a({3,4,5}, kCpuDevice);`
   */
  Tensor(ArrayRef<int32> dims, Device device);


  /** Construct a Tensor with freshly allocated data, data type ==
      `GetDefaultDtype()`, and device == GetDefaultDevice().

       @param [in] dims    The dimensions of the tensor (zero to 5
                    positive integers).
       @param [in] device  The device to put the data on

       Example:  `Tensor a({3,4,5}, kCpuDevice);`
   */
  Tensor(ArrayRef<int32> dims);



  /**
     Construct a Tensor with the dimensions and strides provided.  This differs
     from the constructor taking `ArrayRef<int32> dims` in that it will use
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

  /**
     Construct a Tensor from the metadata in 'meta'.  Requires
     that meta.pattern be contiguous (meaning: literally contiguous,
     not the PyTorch meaning which is a stronger condition).
     ??Possibly we could make it similar to the constructor above
       and have it just make it contiguous if it was not.??


       @param [in] meta  Struct containing the metadata specifying
                     the Tensor's pattern, data-type and device

                     ;pattern  The dimension and stride information that
                  this tensor should match (although we will fill gaps
                  to make it contiguous)
       @param [in] dtype   The data type to use
       @param [in] device  The device to put the data on
       @param [in] set_zero   If true, set the data to zero.  If false,
                        the contents will be undefined.

  */
  Tensor(TensorMeta &meta, InitializePolicy p);


  /**
     This constructor takes the 'impl' and 'storage' provided and returns
     a Tensor containing them.  Intended for special-purpose code such
     as when we wrap arrays from external frameworks.
   */
  Tensor(const TensorImpl &impl, std::shared_ptr<Storage> storage);

 private:
  // This object contains the num-axes, dims, strides and data pointer, plus
  // cached properties.
  TensorImpl impl_;


  // The storage region where the data resides.  data_ does not necessarily
  // equal storage_->data; it may be more than that, e.g. if this is a view
  // to part of another Tensor.
  std::shared_ptr<Storage> storage_;
};




}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_H_

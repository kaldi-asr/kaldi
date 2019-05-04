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


/*
   TENSOR GLOSSARY

    Base Variable:  A Variable that is not a view into another Variable,
             but has been created directly from a Tensor (or via Detach()).
             Each Variable has a base Variable; a base Variable's
             base Variable is itself.  See also: "View Variable".

    Invalidated:  if some data used in backprop needs to have been unchanged since
              a particular tick (as recorded in an Op), but it has been changed
              since then, we say that it has been invalided.  This is an error,
              but it will only be detected in debug mode.  In effect we store a
              record of what time (in ticks) data last changed at the
              individual-element level (e.g. per float), via the ChangeTracker
              object that is attached to the Storage object.  It's done in a
              structured way, not via a huge boolean array.  This means that the
              change-tracking mechanism is not defeated by doing Detach() or by
              constructing multiple Variables from the same Tensor.

    In-place operation: An operation that modifies a Variable, such as adding
              to it after it has been created.  This notion is not particularly
              meaningful in this framework, since in a sense all operations
              are in-place operations; conceptually, the creation of a Variable
              is seen as separate from an operation that sets it to some value,
              and in-place operations are thus not "special".

    Lazy allocation:  We do not allocate memory as soon as a Tensor is created,
              but wait until an operation is done on it.  This makes it easier
              to implement backprop with views of Tensors, because we can
              construct views of Tensors whose memory has not been allocated yet.
              The code for this happens in class Storage (see storage.h).  We
              can also repeat this trick: on a base Variable, you can call
              ZeroDeallocating(), which conceptually zeroes the Variable, but
              does it by freeing the underlying data.  This enables the autograd
              graph to be re-used without leaving too many things allocated.

     Leaf Variable:  A leaf Variable is a Variable that you create directly
             by wrapping a Tensor (or by calling .Detach()).  A leaf
             Variable is always a base Variable.

     Node:   A node in the autograd graph (Ops correspond-- roughly-- to the
             edges in that graph).  There is a node for each tracked base variable.
             [See also: Tracked; Base Variable].

     Op:     (see op.h)  An operation on a Tensor (e.g. addition, multiplication, etc.),
             including in-place operations.  Each Node in the autograd graph stores
             a list of Ops that operated on that base Variable or some sub-part of
             it.  However, if an Op modified two Nodes we need to call its Backprop()
             only once; after figuring out which Ops need to be done, we call their
             Backprop() in reverse order of their ticks (see: Tick).

    Op-input-node:  Relative to a particular Op, a Node is an Op-input-node if
            it is attached to at least one Variable that is an input of that Op,
            but is not attached to any Variable that is an output of that Op.
            An Op-input-node may not also be an Op-output-node (they are disjoint
            sets).

    Op-output-node: Relative to a particular Op, a Node is an Op-output-node
            if it is attached to any Variable that is an output of that Op
            (i.e. that is modified by that Op).

    Optional Tensor:  In situations where we might have a Tensor and might
            not, we use a raw std::shared_ptr<TensorImpl>.  (A Tensor wraps
            a std::shared_ptr<TensorImpl> that is known not to be NULL).
            Note: we don't allow a Tensor to have zero dim, so we can't
            use that representation when the Tensor isn't really there.

    Tick:   a tick is the value of a global 64-bit time counter that we increment
            every time we mutate a Tensor; see GetTick(), and
            Op::GetTimestamp().  When we create Ops for backpropagation of
            derivatives, we record the tick at which the Op was created, for
            purposes of checking for invalidation (see: "Invalidated"), and
            also of ordering Ops during backprop.

   Tracked:  We say a Variable is tracked if gradient-tracking is
             enabled for it.  This will be the case if it is
             a leaf Variable constructed with requires_grad = true,
             or a non-leaf Variable that has been created or changed
             by an operation that depended on a tracked Variable.
             A non-tracked Variable can become tracked but not vice
             versa.  The granularity of being tracked is at the
            "base variable" level.

   Underlying / memory underlying: For a Tensor or Variable a, the "memory
             underlying a" means the part of computer memory, accessible through
             the storage object, that is covered by the pattern of a.

   View Variable:  A View Variable is any variable that is not a base
            variable.  Such variables will be views of base Variables that have
            been created from them by some operation such as slicing
            (e.g. taking row or column ranges).

    Whole Tensor:  A whole Tensor is a Tensor through which one can
            access every byte of the storage region underlying it.
            W.r.t. the notation in tensor-pattern.h (and using words
            that describe Patterns to describe Tensors having those patterns),
            this is equivalent to saying that Tensor is compact and
            justified, and the size of its memory-index-set times the
            bytes per element for its data-type equals the number of bytes
            allocated in the storage region.


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

  /// Return the number of axes (a number in {0,1,2,3,4,5,6}).  In mathematical
  // contexts, this is sometimes known as the rank of the tensor, or sometimes
  // even its dimension, but these terms are ambiguous so we avoid them, and use
  // the terms 'number of axes' or 'axis' throughout.
  // Caution: the numbering of axes in the Tensor interface is different
  // than in TensorImpl::pattern.  Here they are numbered from zero;
  // in TensorImpl::pattern they are shifted to the right so
  // the last axis is KALDI_TENSOR_MAX_DIM - 1.
  inline int32 NumAxes() const { return impl_.pattern.num_axes; }

  const TensorImpl &Impl() const { return impl_; }

  const TensorMeta &Meta() const { return reinterpret_cast<TensorMeta&>(impl_); }

  // Return reference to the struct containing the dimension and
  // stride info.
  const TensorPattern &Pattern() const { return impl_.pattern; }

  // Return a vector containing dimensions of the tensor; equivalent to
  // .shape in PyTorch.  Dims().size() will equal NumAxes().
  // This cannot return some kind of reference because the
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
  // TODO: see if this needs to be cached.
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
     will be the same order as 'other' if sp == kCopyStrideOrder.

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
         StridePolicy sp): impl_(new TensorImpl(meta, sp)) { }


  /** Construct a Tensor with freshly allocated, uninitialized data.

       @param [in] dims    The dimensions of the tensor, up to
                     KALDI_TENSOR_MAX_DIM positive integers.
       @param [in] opts    Options regarding data-type and device;
                           see examples below.
    Example (note: the braces are braced-initializer-lists)
<code>
   Tensor a({3,4});
   Tensor b({}, kDoubleDtype);
   Tensor c({5,6,7}, kCpuDevice);
   Tensor d({1,2}, {kDoubleDtype, kCpuDevice});
</code>
  */
  inline Tensor(ArrayRef<int32> dims,
                TensorOptions opts = TensorOptions()):
      impl_(new TensorImpl(meta, opts)) { }



  /**
     Construct a Tensor from the metadata in 'meta'.  Requires
     that meta.pattern be contiguous (meaning: literally contiguous,
     not the PyTorch meaning which is a stronger condition).

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


  // Move assignment.  TODO: check whether this really does move on the
  // shared_ptr.
  Tensor(Tensor &&other): impl_(other.impl_) { }

  /**
     Constructor from TensorImpl.  Will often be used by framework code; not
     intended for use by users.
   */
  Tensor(const std::shared_ptr<const TensorImpl> &impl);

  /**
     Move-constructor version of constructor from TensorImpl.  Will often be
     used by framework code; not intended for use by users.  TODO: check that
     this really does move.
  */
  Tensor(const std::shared_ptr<const TensorImpl> &&impl): impl_(impl) { }


 private:

  // It might seem odd that we contain a shared_ptr to *const* TensorImpl.
  // What is const here is the meta-information, not the underlying data
  // (e.g. the floats).  The reason for this decision is mostly so that class
  // Variable can store Tensors and shared_ptr's to TensorImpl and not
  // worry about the meta-information pointed to by those pointers being
  // unexpectedly changed.  The idea is, whenever you need to change this
  // meta-info, you reallocate;  things that need to manipulate meta-info
  // and don't want to reallocate can work directly with TensorImpl which
  // is a lower-level, less safe interface intended for the developers of
  // this toolkit.
  //
  // Note: the difference between a Tensor and a simple std::shared_ptr<const
  // TensorImpl> is that in the Tensor the pointer is guaranteed to be non-NULL.
  // We use the shared_ptr where it could be NULL, e.g. in Variables for the
  // grad (since it might not have been set up).
  std::shared_ptr<const TensorImpl> impl_;
};



/**
   This is to be used when you know that 'impl' is non-NULL and you want to
   treat it as a Tensor.  You should view the type `std::shared_ptr<const
   TensorImpl>` as "might be Tensor, might be NULL".
*/
inline Tensor &AsTensor(std::shared_ptr<const TensorImpl> &impl) {
  return reinterpret_cast<Tensor&>(impl);
}







}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_H_

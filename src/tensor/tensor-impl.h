// tensor/tensor-impl.h

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

#ifndef KALDI_TENSOR_TENSOR_IMPL_H_
#define KALDI_TENSOR_TENSOR_IMPL_H_ 1

#include "tensor/tensor-common.h"
#include "tensor/tensor-pattern.h"

namespace kaldi {
namespace tensor {

// Metadata for a Tensor.  It's occasionally convenient to have this
// in a struct (it's the same as a Tensor without the 'data' pointer.
// The members must stay in sync with the corresponding members of
// TensorImpl, as we have code that does reinterpret_cast on
// these types.  (We don't use base-classing as it would make the code
// harder to read).
struct TensorMeta {
  TensorPattern pattern;
  DataType dtype;
  Device device;
};

/**
   TensorImpl is the core part of a Tensor, without the wrapping code and
   storage management in Tensor.h.  Most of the core implementation deals
   directly with TensorImpl to avoid the overhead of shared_ptr management
   and the need to deal with accessors and the like, but TensorImpl
   is intended for use in the tensor/ directory, to implement Tensor
   internals, and not for users of this library.
*/
struct TensorImpl {
  TensorPattern pattern;
  DataType dtype;
  Device device;
  std::shared_ptr<Storage> storage;  // 'storage' points to a shared Storage object
                                     // that contains (or eventually will contain,
                                     // due to lazy allocation) the actual data
                                     // pointer.

  inline int32 NumAxes() { return pattern.num_axes; }

  // Returns the dimension on the supplied axis, using the public axis
  // numbering, with negative index interpreted as an offset from the end.
  //
  //  @param [in] eaxis  Eaxis-index (see definition in tensor-pattern.h)
  //                    Require -NumAxes() <= eaxis < NumAxes().
  //  @return        Returns the dimension on this axis, a number >= 1.
  inline int32 Dim(int32 eaxis);

  // Returns the stride (== distance between successive elements) on the
  // supplied axis, using the public axis numbering, with negative index
  // interpreted as an offset from the end.
  //
  //  @param [in] eaxis  Eaxis-index (see definition in tensor-pattern.h)
  //                    Require -NumAxes() <= eaxis < NumAxes().
  //  @return          Returns the stride on this axis, which will be 0 if
  //                   Dim(axis) == 1, and otherwise nonzero.
  inline int32 Stride(int32 axis);


  // Returns the data pointer corresponding to the element whose index
  // is all zeros.  [TODO: maybe have overloads of this for different types.]
  // CAUTION: this function may allocate the data if it has not yet been
  // allocated.
  inline void* GetData() const;




  /**
    Returns true if this TensorImpl is valid, false otherwise.

       @param [in] check_storage   You can set this to false to disable
                     checks related to the `storage` element (that
                     it's non-NULL and covers the memory range used
                     by the pattern.
       @return   Return true if the TensorImpl is valid (requires
                pattern.Valid(), plus checks on dtype and device,
                plus checks on the storage object if check_storage == true.
  */
  bool IsValid(bool check_storage = true) const;


  /**
     This is to be called by users if they are about to do an operation on this
     Tensor which writes to its underlying memory but does not read from it.
     It gives the framework a free pass to not zero the part of memory covered
     by this Tensor, even if it was instructed to zero the entire storage
     region upon allocation.  Note: calling this will cause the storage region
     to be allocated if it was not already allocated, so only call this
     if you are about to actually use the data for something.

     This function is const, like most operations on TensorImpl, because it doesn't
     change the metadata, only (possibly) the Storage object.
  */
  inline void AllowUndefined() const { storage->AllowUndefined(*this); }

  const TensorMeta &Meta() const {
    return reinterpret_cast<const TensorMeta&>(*this);
  }

  // Note: a copy constructor for TensorImpl might not be needed as we store
  // shared_ptrs to it and just reuse the same object.

  // Constructor that is used when copying the meta-info from one source
  // but the storage from another; this version does move-construction
  // on 'storage'.
  TensorImpl(const TensorMeta &meta,
             const std::shared_ptr<Storage> &storage);

  // Constructor that is used when copying the meta-info from one source
  // but the storage from another; this version does move-construction
  // on 'storage'.
  TensorImpl(const TensorMeta &meta,
             std::shared_ptr<Storage> &&storage);

  // Constructor that copies the meta-info provided; if create_storage
  // == true it creates the storage reason, else leaves it NULL.
  TensorImpl(const TensorMeta &meta,
             bool create_storage = true);

  /**
     Initializes a TensorImpl with the provided dimensions, creating a new
     storage object for it.  The strides will be as for a "C" array; see
     "Default strides:" in tensor-pattern.h.

        @param [in] dims  The dimensions for each axis (in the public
                       numbering).  All elements must be nonnegative,
                       and we require `0 <= dims.size < KALDI_TENSOR_MAX_DIM`.
        @param [in] opts  Options class to set device and dtype;
                          see examples below
<code>
   TensorImpl *t = new TensorImpl({10,20}),
       *u = new TensorImpl({9}, {kGpuDevice});
       *v = new TensorImpl({9}, {kDoubleDtype, kGpuDevice});
</code>
  */
  TensorImpl(ArrayRef<int32> dims,
             TensorOptions opts = TensorOptions());

  /**
    This constructor initializes a TensorImpl with dtype, device and dims taken
    from an existing TensorImpl, but a new storage object, and strides
    determined by the StridePolicy provided.

       @param [in] meta  Meta-info of another TensorImpl; the num_axes,
                        dims, dtype and device will be taken from here
                        and the strides may be inspected, depending
                        on `sp`.
       @param [in] sp   Stride policy (briefly as follows; see more by
                      declaration of StridePolicy in tensor-common.h).
                      kKeepStrideOrder -> use the same order of abs(stride) as
                                          in 'meta'
                      kNormalized -> use normalized strides (see definition
                       in tensor-pattern.h); basically, the normal order we'd use
                       for a new Tensor.
                      kCopyStrides -> use the exact strides from the source
                       pattern.
  */
  TensorImpl(const TensorMeta &meta,
             StridePolicy sp);

  // Default constructor
  TensorImpl() { }

};


inline int32 TensorImpl::Dim(int32 eaxis) {
  int32 raxis = EaxisToRaxis(eaxis);
  if (raxis >= pattern.num_axes)
    KALDI_ERR << "Invalid axis given to Dim(): "
              << eaxis << ", num_axes = "
              << pattern.num_axes;
  return pattern.dims[num_axes];
}



}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_IMPL_H_

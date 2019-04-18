// tensor/storage.h

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

#ifndef KALDI_TENSOR_STORAGE_H_
#define KALDI_TENSOR_STORAGE_H_ 1

#include <functional>
#include "tensor/tensor-common.h"


namespace kaldi {
namespace tensor {

struct StorageExtras;

// 'Storage' contains a single allocated region (on CPU or GPU, according
// to 'device').
class Storage {
 public:
  // This returns a reference to the object held in this->locker if it is
  // non-NULL; otherwise it allocates one and returns that.
  ChangeTracker &GetChangeTracker();

  inline bool Allocated() {  return (data != NULL);  }

  // TODO: we may need a mechanism to automatically zero data when it is
  // allocated, we have to figure out the right level to do this at.
  inline void *Data() {
    if (data) {
      return data;
    } else {
      Allocate();
      return data;
    }
  }


  /**
     Creates a Storage object for device 'device' with size 'num_bytes'.
     The actual data will not be allocated until someone calls this->Data().

       @param [in] device  The device on which the data is to be allocated
       @param [in] num_bytes  The number of bytes to be allocated; must be >0.
  */
  Storage(Device device, size_t num_bytes);

  /**
     This constructor is intended for use with data allocated by code outside
     this codebase (for instance in external toolkits).

          @param [in] device  The device on which this data exists
          @param [in] data    Pointer to the data to be held
          @param [in] num_bytes  The number of bytes held in this region
                              (does not have to be exact, but should be
                              at least the number of bytes in the part of
                              this memory block that is going to be accessed
                              through this Storage object.
          @param [in] deallocator A std::function, which, if not nullptr,
                              will be invoked in
   */

  Storage(Device device,
          void *data,
          size_t num_bytes,
          DeallocatorFunc deallocator):
      data(NULL), num_bytes(0),
      device(device),
      deallocator(deallocator) { }

  // Returns true if the data has already been allocated.  I am hoping that it
  // will never be necessary to call this.
  bool IsAllocated();

  // Deallocates the data.  This is user-callable because our autograd mechanism
  // deletes the underlying data of gradients that are no longer needed, while
  // keeping around the metadata in cases where it is instructed to retain the
  // autograd graph.  Conceptually we think of this as simply zeroing the
  // relevant gradients, since any data that is deallocated is implicitly
  // treated as zero.
  // Calling this is an error if a deallocator function was provided
  // to the constructor of this object.
  void Deallocate();

  // Destructor that frees any data held.
  ~Storage();

 private:

  // Allocate the data.  It is an error to call this if data_ != NULL.
  void Allocate();

  // 'data_' is either 'nullptr' or the actual data pointer.  Due to lazy allocation,
  // the 'data' pointer will remain NULL until it is actually needed.  Lazy
  // allocation makes it much easier to set up the autograd graph without
  // allocating the memory for the gradients.
  void *data_;

  // num_bytes is the number of bytes in the region we have allocated
  // (or are going to allocate).
  size_t num_bytes;

  // the device the data is located on (or is to be located on).
  Device device;

  // contains some extra, less-often-used fields
  std::unique_ptr<StorageExtras> extras;

};



// struct StorageExtras contains what (conceptually) are some rarely-needed
// extra fields of class Storage; we store them separately, holding a
// possibly-NULL pointer to struct StorageExtras, to reduce the size of struct
// Storage in the normal case.
struct StorageExtras {
  using DeallocatorFunc = std::function<void()>;

  // 'tracker' is used in debug mode to detect when data that might be
  // required in the backprop phase is invalidated.
  std::unique_ptr<ChangeTracker> tracker;

  // 'deallocator' is to be used with external toolkits, for example, to
  // decrease the refcount.  In normal cases it will be nullptr.
  // If non-NULL, it will be invoked when we want to deallocate the
  // storage object.
  DeallocatorFunc deallocator;
};




}  // namespace tensor
}  // namespace kaldi

#endif  // KALDI_TENSOR_STORAGE_H_

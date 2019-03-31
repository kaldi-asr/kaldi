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


// 'Storage' contains a single allocated region (on CPU or GPU, according
// to 'device').
struct Storage {
  using DeallocatorFunc = std::function<void()>;

  void *data;
  size_t num_bytes;
  Device device;
  // 'deallocator' to be used with external toolkits, for example, to decrease
  // the refcount
  DeallocatorFunc deallocator;

  // 'device' and 'deallocator' have default constructors.
  Storage(): data(NULL), num_bytes(0) { }

  // This constructor tries to allocate the requested data on the specified
  // device.  It will throw if allocation fails (for now).
  Storage(Device device, size_t num_bytes);


  Storage(Device device, DeallocatorFunc deallocator):
      data(NULL), num_bytes(0),
      device(device),
      deallocator(deallocator) { }

  // If 'deallocator' is non-NULL (only true for external-to-Kaldi tensors such
  // as NumPy), the destructor calls it; otherwise it deallocates 'data' (the
  // method of deallocation depends on the device pointer 'device'.
  ~Storage();
};





}  // namespace tensor
}  // namespace kaldi

#endif  // KALDI_TENSOR_STORAGE_H_

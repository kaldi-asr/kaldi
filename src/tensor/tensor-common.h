// tensor/tensor-common.h

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

#ifndef KALDI_TENSOR_TENSOR_COMMON_H_
#define KALDI_TENSOR_TENSOR_COMMON_H_ 1

#include <cstdint>
#include <vector>

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {

typedef int64_t int64;
typedef uint64_t uint64;
typedef int32_t int32;
typedef uint32_t uint32;



enum {
  kCpuDevice = 0,
  kCudaDevice = 1
} DeviceType;


// We may later add a device number (like which GPU we are using),
// once we support multiple GPUs.
struct Device {
  DeviceType device_type;

  Device(): device_type(kCpuDevice) { }
  Device(DeviceType t): device_type(t) { }

  std::string ToString() const;

  // TODO: operator ==
  // maybe in future we'll make a way to set the default device.
};


enum DataType {
  // We will of course later extend this with many more types, including
  // integer types and half-precision floats.
  kFloatDtype = 0,
  kDoubleDtype = 1
};



/// Enumeration that says what strides we should choose when allocating
/// A Tensor.
enum StridePolicy {
  kCopyStrides,  // means: copy the strides from the source Tensor, preserving
                 //  their signs and relative ordering (but filling in gaps if
                 //  the source Tensor's data was not contiguous.
  kCstrides      // means: strides for dimensions that are != 1 are ordered from
                 // greatest to smallest as in a "C" array.  Per our policy,
                 // any dimension that is 1 will have a zero stride.
};

/// Enumeration that says whether to zero a freshly initialized Tensor.
enum InitializePolicy {
  kZeroData,
  kUninitialized
};

/// This enumeration with one value is used in the constructor of Tensor,
/// so if you do:
///  `Tensor a;  Tensor b(a, kUntrackedStorage);`
/// it will not copy the 'storage' pointer like it normallly would.
/// This is useful as an optimization that avoids atomics with
/// std::shared_ptr, for temporary Tensors in situations where we
/// know the Tensor we are copying from is not going out of scope
/// for the lifetime of the temporary.
enum TensorStorageEnum {
  kUntrackedStorage
};



// In practice we don't expect user-owned tensors with dims greater than 5 to
// exist, but there are certain manipulations we do when simplifying matrix
// multiplications that temporarily add an extra dimension, and it's most
// convenient to just increase the maximum.
#define KALDI_TENSOR_MAX_DIM 6


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_COMMON_H_

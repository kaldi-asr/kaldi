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
#include <string>

/**
   This is some notes on plans for kaldi10 tensor stuff, nothing is fully fleshed out.
*/

namespace kaldi {
namespace tensor {

typedef int64_t int64;
typedef uint64_t uint64;
typedef int32_t int32;
typedef uint32_t uint32;



enum DeviceType {
  kCpuDevice = 0,
  kCudaDevice = 1
};


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


Device GetDefaultDevice();
void SetDefaultDevice(Device device);

class WithDeviceAs {
  // Example:
  // {
  //   WithDeviceAs(kCudaDevice);
  //   // code in this block uses this default.
  // }
 public:
  inline WithDeviceAs(Device device):
      prev_default_(GetDefaultDevice()) {
    SetDefaultDevice(device);
  }
  ~WithDeviceAs() { SetDefaultDevice(prev_default_); }

 private:
  Device prev_default_;
};



enum DataType {
  // We will of course later extend this with many more types, including
  // integer types and half-precision floats.
  kFloatDtype = 0,
  kDoubleDtype = 1
};


aDataType GetDefaultDtype();
void SetDefaultDtype(DataType dtype);

class WithDtypeAs {
  // Example:
  // {
  //   WithDtypeAs(kDoubleDtype);
  //   // code in this block uses this default.
  // }
 public:
  inline WithDtypeAs(DataType dtype):
      prev_default_(GetDefaultDtype()) {
    SetDefaultDtype(dtype);
  }
  ~WithDtypeAs() { SetDefaultDtype(prev_default_); }

 private:
  DataType prev_default_;
};




/// Enumeration that says what strides we should choose when allocating
/// A Tensor.
enum StridePolicy {
  kCopyStrideOrder,  // means: copy the size-ordering of the strides from the
                     // source Tensor (they will all be positive even of some of
                     // the source Tensor's strides were negative).
  kCstrides      // means: strides for dimensions that are != 1 are ordered from
                 // greatest to smallest as in a "C" array.  Per our policy,
                 // any dimension that is 1 will have a zero stride.

  // We may later add options for Fortran-style striding and for the sign of the
  // source Tensor's strides, as well as their order, to be copied.
};

/// Enumeration that says whether to zero a freshly initialized Tensor.
enum InitializePolicy {
  kZeroData,
  kUninitialized
};



/// This enumeration value lists the unary functions that we might
/// want to apply to Tensors; it exists so that much of the glue
/// code can be templated.
enum UnaryFunctionEnum {
  kUnaryFunctionExp,
  kUnaryFunctionLog,
  kUnaryFunctionRelu,
  kUnaryFunctionInvert,
  kUnaryFunctionSquare
  // TODO: add more.
};



/// This enumeration value lists the binary functions that we might
/// want to apply to Tensors; it exists so that much of the glue
/// code can be templated.  (Note: multiplication is not counted
/// here; that is a special case as it will genearlly go to BLAS).
enum BinaryFunctionEnum {
  kBinaryFunctionAdd,
  kBinaryFunctionDivide,
  kBinaryFunctionMax,
  kBinaryFunctionMin
};



// In practice we don't expect user-owned tensors with dims greater than 5 to
// exist, but there are certain manipulations we do when simplifying matrix
// multiplications that temporarily add an extra dimension, and it's most
// convenient to just increase the maximum.
#define KALDI_TENSOR_MAX_DIM 6


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_COMMON_H_

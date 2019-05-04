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


enum DataType {
  // We will of course later extend this with many more types, including
  // integer types and half-precision floats.
  kDefaultDtype = 0,
  // kDefaultDtype means the type used when not specified; it's user definable
  // via SetDefaultDtype.
  kFloatDtype = 1,
  kDoubleDtype = 2,
};



inline int32 SizeOf(DataType dtype) {
  switch(dtype) {
    case 0: return 4;
    case 1: return 8;
    case 2: KALDI_ERR << "Invalid data-type " << int32(dtype); return 0;
  }
}


/// Enumeration that says what strides we should choose when allocating
/// A Tensor.
enum StridePolicy {
  kKeepStrideOrder,  // Means: keep the size-ordering of the strides from the
                     // source Tensor (but the chosen strides will all be
                     // positive even of some of the source Tensor's strides
                     // were negative).
  kNormalized    // Means: strides for dimensions that are != 1 are ordered from
                 // greatest to smallest as in a "C" array in the public
                 // numbering, or smallest to greatest in the private numbering.
                 // Per our policy, any dimension that is 1 will be given a zero stride.
                 // C.f. "Normalized strides" in tensor-pattern.h
  kCopyStrides   // Means: use the exact strides provided.
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



// In practice we don't expect user-owned tensors with num-axes greater than 5
// to exist, but there are certain manipulations we do when simplifying matrix
// multiplications that temporarily add an extra dimension, and it's most
// convenient to just increase the maximum.
#define KALDI_TENSOR_MAX_AXES 6


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_COMMON_H_

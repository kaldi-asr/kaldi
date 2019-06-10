// tensor/linear-special-ops.h

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

#ifndef KALDI_TENSOR_LINEAR_SPECIAL_OPS_H_
#define KALDI_TENSOR_LINEAR_SPECIAL_OPS_H_ 1

#include "tensor/tensor.h"


// This Ops are more specialized forms of the Ops declared in linear-ops; these
// correspond to more specific combinations of Tensor shapes.
// Just the template declarations are here; the overrides for CPU and
// GPU are in linear-cpu-ops.h and linear-gpu-ops.h.
namespace kaldi {
namespace tensor {


/**
   Operation doing a += b with a and b scalar.

   a and b may not point to the same data.

   Template parameter T is the datatype concerned (say, T = float)
   D is the DeviceType enum, kCpuDevice or kGpuDevice.

   Will be specialized for CPU and GPU in linear-cpu-ops.h and linear-gpu-ops.h
*/
template <class T, DeviceType D>
class ScalarPlusEqScalarOp;


/**
   Operation doing a += b with a and b possibly-strided vectors.

   a and b may not overlap.

   Template parameter T is the datatype concerned (say, T = float)
   D is the DeviceType enum, kCpuDevice or kGpuDevice.

   Will be specialized for CPU and GPU in linear-cpu-ops.h and linear-gpu-ops.h
*/
template <class T, DeviceType D>
class StvectorPlusEqStvectorOp;


/**
   Operation doing a += b with a a vector and b a scalar.  (I.e. add
   a constant elementwise to a vector).

   May not be used if a and b overlap.

   Template parameter T is the datatype concerned (say, T = float)
   D is the DeviceType enum, kCpuDevice or kGpuDevice.

   Will be specialized for CPU and GPU in linear-cpu-ops.h and linear-gpu-ops.h
*/
template <class T, DeviceType D>
class StvectorPlusEqScalarOp;

}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR__LINEAR_OPS_H_

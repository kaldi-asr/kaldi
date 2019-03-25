// tensor/tensor-impl-functions.cc

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

#include "tensor/tensor-pattern.h"
#include "tensor/cpu-impl-linear.h"


namespace kaldi {
namespace tensor {


inline static void AddProductScalar3(
    float alpha, float beta,
    const TensorImpl &a, const TensorImpl &b, const TensorImpl *c) {
  switch (a.device.device_type) {
    case kCpuDevice:
      AddProductScalar3Cpu(alpha, beta, a, b, c);
      return;
#ifdef HAVE_CUDA
    case kGpuDevice:
      AddProductScalar3Gpu(alpha, beta, a, b, c);
      return;
#endif
    default:
      KALDI_ERR << "Unsupported device type " << a.ToString();
  }
}


void AddProduct(float alpha, float beta,
                const TensorImpl &a, const TensorImpl &b, const TensorImpl *c){
  CheckDeviceAndDtype(a, b, *c);
  KALDI_PARANOID_ASSERT(a.pattern.code <= b.pattern.code);

  int64 combined_code = CombineCodes(a.pattern.code, b.pattern.code,
                                     c->pattern.code);

  /*
    The case-statement values in the switch statement below may be
    interpreted in groups of 3 hex characters, are 0xAAABBBCCC,
    pertaining to Tensors a, b and c respectively.  See
    GetPatternCode() in tensor-pattern-utils.h for documentation on
    the meanings of the values:
   */
  switch(combined_code) {
    case 0x000000000:
      // scalar * scalar -> scalar
      AddProductScalar3(a, b, c);
      return;
    case 0x000101101:
      // scalar * vector -> vector
      AddProductScalarVector2(a, b, c);
      return;
    case 0x101101101:
      // vector .* vector -> vector
      AddProductVector3(a, b, c);
      return;
    default:
      break;

  }

  // If we reached this point, it means we could
  // not handle this request with any of the basic operations above.
  // Something is a little differ


  SubTensor a_temp(a), b_temp(b), c_temp(*c);

  PadAxes(&(a.pattern), &(b.pattern), &(c.pattern));

  CompressPatterns({&a_temp, &b_temp, &c_temp});
}



}  // namespace kaldi
}  // namespace tensor

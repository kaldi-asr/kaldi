// tensor/tensor-functions.cc

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


namespace kaldi {
namespace tensor {

void AddProductReducing(float alpha, float beta,
                        const Tensor &a, const Tensor &b, Tensor *c){
  CheckDeviceAndDtype(a, b, *c);

  int32 a_pcode = a.PatternCode(), b_pcode = b.PatternCode(),
      c_pcode = c->PatternCode();
  int64 combined_pcode = (int64(a_pcode) << 24) + b_pcode << 12 + c_pcode;

  // Each group of 3 hex numbers describes on of the argument Tensors,
  // so it's 0xAAABBBCCC.
  //
  switch (combined_pcode) {

    case 0x000000000:
      // scalar multiplication




  }


  SubTensor a_temp(a), b_temp(b), c_temp(*c);

  PadAxes(&(a.pattern), &(b.pattern), &(c.pattern));

  CompressPatterns({&a_temp, &b_temp, &c_temp});
}



}  // namespace kaldi
}  // namespace tensor

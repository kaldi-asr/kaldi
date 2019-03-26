// tensor/cpu-impl.cc

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

#include "tensor/cpu-impl.h"


namespace kaldi {
namespace tensor {


template <typename Real>
inline static void AddProductScalar3CPU(
    float alpha, float beta,
    const TensorImpl &a, const TensorImpl &b, const TensorImpl *c) {
  Real *a_data = static_cast<Real*>(a->data),
      *b_data = static_cast<Real*>(b->data),
      *c_data = static_cast<Real*>(c->data);
  if (beta != 0.0) {
    *c_data = (beta * *c_data) + alpha * (*a_data + *b_data);
  } else {  // don't propagate NaN
    *c_data = alpha * (*a_data + *b_data);
  }
}


void AddProductScalar3CPU(
    float alpha, float beta,
    const TensorImpl &a, const TensorImpl &b, const TensorImpl *c) {
  if (c.dtype == kFloatDtype) {
    AddProductScalar3CPU<float>(a, b, c);
  } else if (c.dtype == kDoubleDtype) {
    AddProductScalar3CPU<double>(a, b, c);
  } else {
    KALDI_ERR << "Data type not supported for this operation";
  }
}




}  // namespace tensor
}  // namespace kaldi

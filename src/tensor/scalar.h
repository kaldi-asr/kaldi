// tensor/scalar.h

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

#ifndef KALDI_TENSOR_TENSOR_H_
#define KALDI_TENSOR_TENSOR_H_ 1

#include "tensor/tensor-common.h"
#include "tensor/pattern.h"
#include "tensor/tensor-impl.h"
#include "tensor/storage.h"




namespace kaldi {
namespace tensor {


/**
   Scalar is how we wrap user-supplied constant scalar value.  Right now this
   basically wraps a double, but for future extensibility to ints, complex
   numbers and so on, we make it a class.
*/
class Scalar {
 public:
  Scalar(float f): value_(f) { }
  Scalar(double d): value_(d) { }


  float operator float() const (return value_);
  float operator double() const (return value_);
  // DataType Dtype() { return dtype_; }
 private:
  double value_;
  // DataType dtype_;
};


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_H_

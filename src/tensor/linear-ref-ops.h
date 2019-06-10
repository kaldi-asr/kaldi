// tensor/linear-ref-ops.h

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

#ifndef KALDI_TENSOR_LINEAR_REF_OPS_H_
#define KALDI_TENSOR_LINEAR_REF_OPS_H_ 1

#include "tensor/tensor.h"
#include "tensor/op.h"


// This header contains the "reference version" of linear Ops;
// this is the very simple, not-efficient version that runs on
// CPU when we run in "reference mode" (or when we encounter
// some combination that can't be run using our normal BLAS-based
// speciailized Ops).
namespace kaldi {
namespace tensor {

// Corresponds to the command a += b.

template <typename Real>
class PlusEqRefOp: public Op {
  PlusEqRefOp(const Tensor &a, const Tensor &b):
      a_(a), b_(b) {
    KALDI_ASSERT(!Overlap(a, b) && Compatible(a, b));

  }

  int32 Properties() { return 0 ; }  // Not concrete.

  Op *Copy() const override {
    return new PlusEqRefOp(a, b_);
  }

  void Do() const override {
    RecordUse(a_, kReadWrite);
    RecordUse(b_, kRead);
    Do(a_.GetData<Real>(), b_.GetData<Real>,
       KALDI_TENSOR_MAX_DIM - 1);
  }

  private:

  void Do(Real *a, Real *b, int32 raxis) {
    int32 dim = std::max<int32>(a_.dims[raxis], b_.dims[raxis]),
        a_stride = a_.strides[raxis], b_stride = b_.strides[raxis];
    if (raxis == 0) {
      for (int32 i = 0; i < dim; i++) {
        a[i * a_stride] += b[i * b_stride];
      }
    } else {
      for (int32 i = 0; i < dim; i++) {
        Do(a + i * a_stride, b + i * b_stride, raxis - 1);
      }
    }
  }

  Tensor a_;
  Tensor b_;
};


}
}


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_LINEAR_REF_OPS_H_

// util/tensor-pattern-utils-test.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "tensor/tensor-pattern.h"
#include "tensor/tensor-pattern-utils.h"
#include "base/kaldi-math.h"


namespace kaldi {
namespace tensor {

// We may later move this function to somewhere more permanent.
void GenerateRandomPattern(Pattern *pattern) {

  int32 num_axes = RandInt(0, KALDI_TENSOR_MAX_DIM);

  // the 'cur_stride' stuff is a mechanism for generating strides that
  // will satisfy the 'uniqueness' rule; we'll later randomize the
  // order of axes.
  int32 cur_stride = 1;
  for (int32 raxis = 0; raxis < num_axes; raxis++) {
    int32 dim = RandInt(1, 10);
    pattern->dims[raxis] = dim;
    if (dim > 1) {
      cur_stride *= RandInt(1, 3);
      pattern->strides[raxis] = cur_stride;
      cur_stride *= dim;
    } else {
      pattern->strides[raxis] = 0;
    }
  }

  for (int32 i = 0; i <= num_axes; i++) {
    int32 raxis1 = RandInt(0, num_axes - 1),
        raxis2 = RandInt(0, num_axes - 1);
    if (raxis1 != raxis2) {
      std::swap(pattern->dims[raxis1], pattern->dims[raxis2]);
      std::swap(pattern->strides[raxis1], pattern->strides[raxis2]);
    }
  }
  for (int32 raxis = num_axes; raxis < KALDI_TENSOR_MAX_DIM; raxis++) {
    pattern->dims[raxis] = 1;
    pattern->strides[raxis] = 0;
  }
  pattern->code = ComputePatternCode(*pattern);
  if (RandInt(0, 1) == 0) {
    KALDI_ASSERT(pattern->Check());
  } else {
    KALDI_ASSERT(pattern->Check(true));
  }
}


void UnitTestGenRandomPattern() {
  Pattern p;
  for (int32 i = 0; i < 100; i++) {
    GenerateRandomPattern(&p);
  }
}

}  // namespace kaldi
}  // namespace tensor

int main(int argc, const char** argv) {
  using namespace kaldi;
  using namespace kaldi::tensor;
  UnitTestGenRandomPattern();
  return 0;
}

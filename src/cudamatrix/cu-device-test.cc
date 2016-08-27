// cudamatrix/cu-device-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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


#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

using namespace kaldi;


namespace kaldi {


template<typename Real>
std::string NameOf() {
  return (sizeof(Real) == 8 ? "<double>" : "<float>");
}

template<typename Real> void TestCuMatrixResize(int32 size_multiple) {
  int32 num_matrices = 256;
  BaseFloat time_in_secs = 0.2;

  std::vector<std::pair<int32, int32> > sizes(num_matrices);

  for (int32 i = 0; i < num_matrices; i++) {
    int32 num_rows = RandInt(1, 10);
    num_rows *= num_rows;
    num_rows *= size_multiple;
    int32 num_cols = RandInt(1, 10);
    num_cols *= num_cols;
    num_cols *= size_multiple;
    sizes[i].first = num_rows;
    sizes[i].second = num_rows;
  }

  std::vector<CuMatrix<BaseFloat> > matrices(num_matrices);

  Timer tim;
  size_t num_floats_processed = 0;
  for (;tim.Elapsed() < time_in_secs; ) {
    int32 matrix = RandInt(0, num_matrices - 1);
    if (matrices[matrix].NumRows() == 0) {
      int32 num_rows = sizes[matrix].first,
          num_cols = sizes[matrix].second;
      matrices[matrix].Resize(num_rows, num_cols, kUndefined);
      num_floats_processed += num_rows * num_cols;
    } else {
      matrices[matrix].Resize(0, 0);
    }
  }

  BaseFloat gflops = num_floats_processed / (tim.Elapsed() * 1.0e+09);

  KALDI_LOG << "For CuMatrix::Resize" << NameOf<Real>() << ", for size_multiple = "
            << size_multiple << ", speed was " << gflops << " gigaflops.";
}

template <typename Real>
void CudaMatrixResizeTest() {
  std::vector<int32> sizes;
  sizes.push_back(1);
  sizes.push_back(2);
  sizes.push_back(4);
  sizes.push_back(8);
  sizes.push_back(16);
  //sizes.push_back(24);
  //sizes.push_back(32);
  //sizes.push_back(40);

  int32 ns = sizes.size();
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixResize<Real>(sizes[s]);
}


} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    kaldi::CudaMatrixResizeTest<float>();
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaMatrixResizeTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaMatrixResizeTest<double>();
#endif
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  std::cout << "Tests succeeded.\n";
}

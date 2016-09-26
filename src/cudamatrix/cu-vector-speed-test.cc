// cudamatrix/cu-vector-speed-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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
#include "cudamatrix/cu-math.h"

using namespace kaldi;


namespace kaldi {

template<typename Real>
std::string NameOf() {
  return (sizeof(Real) == 8 ? "<double>" : "<float>");
}

template<typename Real> void TestCuVectorSoftmax(int32 dim) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> M(dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    M.ApplySoftMax();
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::Softmax" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuVectorSum(int32 dim) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> M(dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    M.Sum();
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::Sum" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real, typename OtherReal> void TestCuVectorCopyFromVec(int32 dim) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> M(dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    CuVector<OtherReal> v(dim);
    v.CopyFromVec(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::CopyFromVec" << NameOf<Real>() << " to "
            <<  NameOf<OtherReal>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


#if HAVE_CUDA == 1
// This test choose the min length of vectors to be reduced on GPU.
// Smaller vector will be copied to RAM and reduced on CPU.
template<typename Real> void TestCuVectorSumChooseMinLength() {
  BaseFloat time_in_secs = 0.02;
  for (int dim = 100; dim < 1000000; dim = dim * 1.5 + 1 ) {
    CuVector<Real> M(dim);
    BaseFloat gflops, gflops_cpu;
    Real result = 0, result_cpu = 0;
    M.SetRandn();
    {
      Timer tim;
      int32 iter = 0;
      for (; tim.Elapsed() < time_in_secs; iter++) {
        // Force GPU reduction
        int dimBlock = CU1DBLOCK;
        int dimGrid = n_blocks(M.Dim(), dimBlock);
        if (dimGrid > 256) {
          dimGrid = 256;
        }
        CuVector<Real> ans(dimGrid, kUndefined);
        cuda_vec_sum(dimGrid, dimBlock, M.Data(), ans.Data(), M.Dim(), 1);
        CU_SAFE_CALL(cudaGetLastError());
        Vector<Real> ans_cpu(ans);
        result = ans_cpu.Sum();
      }

      BaseFloat fdim = dim;
      gflops = (fdim * iter) / (tim.Elapsed() * 1.0e+09);
    }
    {
      Timer tim;
      int32 iter = 0;
      for (; tim.Elapsed() < time_in_secs; iter++) {
        Vector<Real> M_cpu(M);
        result_cpu = M_cpu.Sum();
      }

      BaseFloat fdim = dim;
      gflops_cpu = (fdim * iter) / (tim.Elapsed() * 1.0e+09);
    }
    KALDI_LOG << "CuVector::Sum" << NameOf<Real>() << ", dim: " << dim
              << ", speed: GPU " << (gflops > gflops_cpu ? ">" : "<")
              << " CPU, GPU speed: " << gflops << " Gflops. CPU speed: "
              << gflops_cpu << " Gflops. Result diff: " << (result - result_cpu);
  }
}
#endif

template<typename Real> void TestCuVectorVecVecOne(int32 dim) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> M(dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    CuVector<Real> ones(dim);
    ones.Set(1.0);
    VecVec(M, ones);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::VecVecOne" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}




template<typename Real> void TestCuVectorAddDiagMatMat(int32 dim,
                                                       MatrixTransposeType transN,
                                                       MatrixTransposeType transO) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> v(dim);
  v.SetRandn();
  CuMatrix<Real> N(dim, dim), O(dim, dim);
  N.SetRandn();
  O.SetRandn();

  Timer tim;
  int32 iter = 0;

  for (;tim.Elapsed() < time_in_secs; iter++) {
    v.AddDiagMatMat(1.0, N, transN, O, transO, 1.0);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::AddDiagMatMat" << NameOf<Real>()
            << (transN == kNoTrans ? "[no-trans],":"[trans],")
            << (transO == kNoTrans ? "[no-trans],":"[trans],")
            << " for dim = "<< dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuVectorAddDiagMat2(int32 dim, MatrixTransposeType trans) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> v(dim);
  v.SetRandn();
  CuMatrix<Real> N(dim, dim);
  N.SetRandn();

  Timer tim;
  int32 iter = 0;

  for (;tim.Elapsed() < time_in_secs; iter++) {
    v.AddDiagMat2(1.0, N, trans, 0.0);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::AddDiagMat2" << NameOf<Real>()
            << (trans == kTrans ? "[trans]" : "[no-trans]") << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuVectorAddRowSumMat(int32 dim, MatrixTransposeType trans) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> v(dim);
  v.SetRandn();
  CuMatrix<Real> N(dim, dim);
  N.SetRandn();

  Timer tim;
  int32 iter = 0;

  for (;tim.Elapsed() < time_in_secs; iter++) {
    v.AddRowSumMat(1.0, N, 0.5);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::AddRowSumMat" << NameOf<Real>()
            << (trans == kTrans ? "[trans]" : "[no-trans]") << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuVectorAddColSumMat(int32 dim, MatrixTransposeType trans) {
  BaseFloat time_in_secs = 0.02;
  CuVector<Real> v(dim);
  v.SetRandn();
  CuMatrix<Real> N(dim, dim);
  N.SetRandn();

  Timer tim;
  int32 iter = 0;

  for (;tim.Elapsed() < time_in_secs; iter++) {
    v.AddColSumMat(1.0, N, 0.5);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuVector::AddColSumMat" << NameOf<Real>()
            << (trans == kTrans ? "[trans]" : "[no-trans]") << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void CudaVectorSpeedTest() {
  std::vector<int32> sizes;
  sizes.push_back(16);
  sizes.push_back(32);
  sizes.push_back(64);
  sizes.push_back(128);
  sizes.push_back(256);
  sizes.push_back(1024);
  int32 ns = sizes.size();
  for (int32 s = 0; s < ns; s++)
    TestCuVectorSoftmax<Real>(sizes[s]);
#if HAVE_CUDA == 1
  TestCuVectorSumChooseMinLength<Real>();
#endif
  for (int32 s = 0; s < ns; s++)
    TestCuVectorSum<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuVectorVecVecOne<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuVectorCopyFromVec<Real, float>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuVectorCopyFromVec<Real, double>(sizes[s]);
  for (int32 s = 0; s < ns; s++) {
    TestCuVectorAddDiagMatMat<Real>(sizes[s], kNoTrans, kNoTrans);
    TestCuVectorAddDiagMatMat<Real>(sizes[s], kNoTrans, kTrans);
    TestCuVectorAddDiagMatMat<Real>(sizes[s], kTrans, kNoTrans);
    TestCuVectorAddDiagMatMat<Real>(sizes[s], kTrans, kTrans);
  }
  for (int32 s = 0; s < ns; s++) {
    TestCuVectorAddDiagMat2<Real>(sizes[s], kNoTrans);
    TestCuVectorAddDiagMat2<Real>(sizes[s], kTrans);
  }
  for (int32 s = 0; s < ns; s++) {
    TestCuVectorAddRowSumMat<Real>(sizes[s], kNoTrans);
    TestCuVectorAddRowSumMat<Real>(sizes[s], kTrans);
  }
  for (int32 s = 0; s < ns; s++) {
    TestCuVectorAddColSumMat<Real>(sizes[s], kNoTrans);
    TestCuVectorAddColSumMat<Real>(sizes[s], kTrans);
  }

}


} // namespace kaldi


int main() {
    //Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId("yes"); //-2 .. automatic selection
#endif

    kaldi::CudaVectorSpeedTest<float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported()) {
    kaldi::CudaVectorSpeedTest<double>();
  } else {
    KALDI_WARN << "Double precision not supported";
  }
#else
  kaldi::CudaVectorSpeedTest<double>();
#endif
  std::cout << "Tests succeeded.\n";
}


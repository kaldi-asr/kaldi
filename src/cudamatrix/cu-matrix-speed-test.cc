// cudamatrix/cu-matrix-speed-test.cc

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
    
template<typename Real> void TestCuMatrixMatMat(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim), N(dim, dim), O(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    O.AddMatMat(1.0, M, kNoTrans, N, kNoTrans, 0.0);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddMatMat" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixSigmoid(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.Sigmoid(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Sigmoid" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixSoftmax(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(256, dim), N(256, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.ApplySoftMaxPerRow(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Softmax" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixTraceMatMat(int32 dim) {
  for (int32 n = 0; n < 2; n++) {
    MatrixTransposeType trans = (n == 0 ? kNoTrans : kTrans);
    BaseFloat time_in_secs = 0.08;
  
    CuMatrix<Real> M(dim, dim), N(dim, dim);
    M.SetRandn();
    N.SetRandn();
    Timer tim;
    int32 iter = 0;
    for (;tim.Elapsed() < time_in_secs; iter++) {
      TraceMatMat(M, N, trans);
    }
    BaseFloat fdim = dim;
    BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For CuMatrix::TraceMatMat" << NameOf<Real>() 
              << (trans == kTrans ? " [transposed]" : "") << ", for dim = "
              << dim << ", speed was " << gflops << " gigaflops.";
  }
}

template<typename Real> void TestCuMatrixCopyLowerToUpper(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyLowerToUpper();
  }
  CuMatrix<Real> M2(M, kTrans);
  AssertEqual(M, M2);
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyLowerToUpper" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixCopyUpperToLower(int32 dim) {
  BaseFloat time_in_secs = 0.05;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyUpperToLower();
  }
  CuMatrix<Real> M2(M, kTrans);
  AssertEqual(M, M2);
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyUpperToLower" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void CudaMatrixSpeedTest() {
  std::vector<int32> sizes;
  sizes.push_back(16);
  sizes.push_back(128);
  sizes.push_back(256);
  sizes.push_back(1024);
  int32 ns = sizes.size();
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixMatMat<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSigmoid<Real>(sizes[s]);

  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSoftmax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixTraceMatMat<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyLowerToUpper<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyUpperToLower<Real>(sizes[s]);
}


} // namespace kaldi


int main() {
    //Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId("yes"); //-2 .. automatic selection
#endif

    kaldi::CudaMatrixSpeedTest<float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported()) {
    kaldi::CudaMatrixSpeedTest<double>();
  } else {
    KALDI_WARN << "Double precision not supported";
  }
#else
  kaldi::CudaMatrixSpeedTest<double>();
#endif
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  std::cout << "Tests succeeded.\n";
}

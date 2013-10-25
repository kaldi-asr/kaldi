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
#include "cudamatrix/cu-sp-matrix.h"

using namespace kaldi;


namespace kaldi {

template<typename Real>
std::string NameOf() {
  return (sizeof(Real) == 8 ? "<double>" : "<float>");
}

template<typename Real>
static void UnitTestCuSpMatrixInvert(int32 dim) {
  BaseFloat time_in_secs = 0.5;
  int32 iter = 0;
  Timer tim;
  CuSpMatrix<Real> A(dim);
  A.SetRandn();
  for (;tim.Elapsed() < time_in_secs; iter++) {
    KALDI_ASSERT(A.Trace() != 0.0); // true with probability 1...
    CuSpMatrix<Real> B(A);

    if (iter  > 0) {
      B.Invert();
    } else { // do some more testing...
    
      CuMatrix<Real> D(A);
      A.AddMat2(1.0, D, kTrans, 1.0);
      A.AddToDiag(0.1 * dim);

      CuMatrix<Real> C(B);
      B.AddMat2(1.0, C, kTrans, 1.0);
      B.AddToDiag(0.1 * dim);
    
      A.Invert();
      B.Invert();
    
      SpMatrix<Real> E(dim);
      B.CopyToSp(&E);

      SpMatrix<Real> A2(A);
      AssertEqual(A2, E);
    }
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuSpMatrix::Invert" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}



template<typename Real>
static void UnitTestCuSpMatrixCopyFromMat(int32 dim, SpCopyType copy_type) {
  BaseFloat time_in_secs = 0.1;
  int32 iter = 0;
  Timer tim;
  CuMatrix<Real> A(dim, dim);
  CuSpMatrix<Real> S(dim);

  for (;tim.Elapsed() < time_in_secs; iter++) {
    S.CopyFromMat(A, copy_type);
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuSpMatrix::CopyFromMat" << NameOf<Real>()
            << ", with copy-type "
            <<(copy_type == kTakeLower ? "kTakeLower" :
               (copy_type == kTakeUpper ? "kTakeUpper" :
                "kTakeMeanAndCheck")) << " and dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}



template<typename Real>
static void UnitTestCuMatrixApproxInvert(int32 dim) {
  BaseFloat time_in_secs = 0.5;
  int32 iter = 0;

  // Get random orthogonal matrix.
  Matrix<Real> Q_cpu(dim, dim);

  Q_cpu.SetRandn();
  for (int32 r = 0; r < dim; r++) {
    for (int32 s = 0; s < r; s++)
      Q_cpu.Row(r).AddVec(-1.0 * VecVec(Q_cpu.Row(r), Q_cpu.Row(s)), Q_cpu.Row(s));
    Q_cpu.Row(r).Scale(1.0 / Q_cpu.Row(r).Norm(2.0));
  }
  CuMatrix<Real> Q(Q_cpu);
  
  CuVector<Real> s(dim);
  Real eig_range = 50.0; // factor of 50 on eigenvalues.. this affects the speed.
  Real first_eig = 0.001 + RandUniform() * 5.0;
  for (int32 r = 0; r < dim; r++)
    s(r) = first_eig * exp(r * log(eig_range) / dim);

  s.ApplyPow(0.5);
  Q.MulColsVec(s);
  CuSpMatrix<Real> A(dim);
  A.AddMat2(1.0, Q, kNoTrans, 0.0);


  Timer tim;
  
  for (;tim.Elapsed() < time_in_secs; iter++) {  
    CuSpMatrix<Real> Atmp(A);
    Atmp.InvertPosDefApprox(0.1);
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuSpMatrix::InvertPosDefApprox" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}  

template<typename Real> void CuSpMatrixSpeedTest() {
  std::vector<int32> sizes;
  sizes.push_back(16);
  sizes.push_back(128);
  sizes.push_back(256);
  sizes.push_back(1024);
  int32 ns = sizes.size();

  for (int32 s = 0; s < ns; s++) {
    UnitTestCuSpMatrixInvert<Real>(sizes[s]);
    UnitTestCuMatrixApproxInvert<Real>(sizes[s]);
    UnitTestCuSpMatrixCopyFromMat<Real>(sizes[s], kTakeLower);
    UnitTestCuSpMatrixCopyFromMat<Real>(sizes[s], kTakeUpper);
    UnitTestCuSpMatrixCopyFromMat<Real>(sizes[s], kTakeMean);
  }
}


} // namespace kaldi


int main() {
    //Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId("yes"); //-2 .. automatic selection
#endif

    kaldi::CuSpMatrixSpeedTest<float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported()) {
    kaldi::CuSpMatrixSpeedTest<double>();
  } else {
    KALDI_WARN << "Double precision not supported";
  }
#else
  kaldi::CuSpMatrixSpeedTest<double>();
#endif
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  std::cout << "Tests succeeded.\n";
}

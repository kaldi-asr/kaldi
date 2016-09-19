// cudamatrix/cu-tp-matrix-test.cc
//
// Copyright 2013  Ehsan Variani
//                 Lucas Ondel

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
//
// UnitTests for testing cu-sp-matrix.h methods.
//

#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sp-matrix.h"

using namespace kaldi;

namespace kaldi {


template<typename Real>
static void AssertEqual(const CuPackedMatrix<Real> &A,
                        const CuPackedMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j) - B(i, j))
                   < tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j)))));
}

template<typename Real>
static void AssertEqual(const PackedMatrix<Real> &A,
                        const PackedMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j) - B(i, j))
                   < tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j)))));
}

template<typename Real>
static void AssertEqual(const PackedMatrix<Real> &A,
                        const CuPackedMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j) - B(i, j))
                   < tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j)))));
}



/*
 * Unit Tests
 */
template<typename Real>
static void UnitTestCuTpMatrixInvert() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + Rand() % 10;
    
    TpMatrix<Real> A(dim);
    A.SetRandn();
    CuTpMatrix<Real> B(A);
    
    AssertEqual<Real>(A, B, 0.005);
    A.Invert();
    B.Invert();
    AssertEqual<Real>(A, B, 0.005);
  }
}

template<typename Real>
static void UnitTestCuTpMatrixCopyFromTp() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + Rand() % 10;
    
    TpMatrix<Real> A(dim);
    A.SetRandn();
    CuTpMatrix<Real> B(dim);
    B.CopyFromTp(A);
    CuTpMatrix<Real> C(dim);
    C.CopyFromTp(B);
    
    AssertEqual<Real>(A, B);
    AssertEqual<Real>(B, C);
  }
}

template<typename Real>
static void UnitTestCuTpMatrixCopyFromMat() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixTransposeType trans = (i % 2 == 0 ? kNoTrans : kTrans);

    MatrixIndexT dim = 10*i + Rand() % 5;
    CuMatrix<Real> A(dim, dim);
    A.SetRandn();
    Matrix<Real> A2(A);
    
    CuTpMatrix<Real> B(dim);
    B.CopyFromMat(A, trans);
    TpMatrix<Real> B2(dim);
    B2.CopyFromMat(A2, trans);
    TpMatrix<Real> B3(B);
    AssertEqual(B2, B3);
    KALDI_ASSERT(B3.Trace() != 0);
  }
}



template<typename Real>
static void UnitTestCuTpMatrixCholesky() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 1 + Rand() % 10;
    if (i > 4) {
      dim += 32 * (Rand() % 5);
    }

    Matrix<Real> M(dim, dim + 2);
    M.SetRandn();
    SpMatrix<Real> A(dim);
    A.AddMat2(1.0, M, kNoTrans, 0.0); // sets A to random almost-surely +ve
                                      // definite matrix.
    CuSpMatrix<Real> B(A);

    TpMatrix<Real> C(dim);
    C.SetRandn();
    CuTpMatrix<Real> D(C);
    C.Cholesky(A);
    D.Cholesky(B);

    AssertEqual<Real>(C, D);
  }
}

template<class Real>
static void UnitTestCuTpMatrixIO() {
  for (int32 i = 0; i < 3; i++) {
    int32 dimM = Rand() % 255 + 10;
    if (i % 5 == 0) { dimM = 0; }
    CuTpMatrix<Real> mat(dimM);
    mat.SetRandn();
    std::ostringstream os;
    bool binary = (i % 4 < 2);
    mat.Write(os, binary);

    CuTpMatrix<Real> mat2;
    std::istringstream is(os.str());
    mat2.Read(is, binary);
    AssertEqual(mat, mat2);
  }
}

template<typename Real> void CudaTpMatrixUnitTest() {
  UnitTestCuTpMatrixIO<Real>();
  UnitTestCuTpMatrixInvert<Real>();
  UnitTestCuTpMatrixCopyFromTp<Real>();
  UnitTestCuTpMatrixCholesky<Real>();
  UnitTestCuTpMatrixCopyFromMat<Real>();
}

} // namespace kaldi


int main() {
  using namespace kaldi;


  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("yes"); // -2 .. automatic selection
#endif
    kaldi::CudaTpMatrixUnitTest<float>();
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaTpMatrixUnitTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaTpMatrixUnitTest<double>();
#endif
  
    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

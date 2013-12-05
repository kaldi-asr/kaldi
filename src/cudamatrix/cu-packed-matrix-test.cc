// cudamatrix/cu-sp-matrix-test.cc
//
// Copyright 2013  Ehsan Variani
//                 Lucas Ondel

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

// UnitTests for testing cu-sp-matrix.h methods.
//

#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-math.h"

using namespace kaldi;

namespace kaldi {

/*
 * INITIALIZERS
 */

/*
 * ASSERTS
 */
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
static void AssertDiagEqual(const PackedMatrix<Real> &A,
                        const CuPackedMatrix<Real> &B,
                        float value,
                        float tol = 0.001) {
  for (MatrixIndexT i = 0; i < A.NumRows(); i++) {
    KALDI_ASSERT(std::abs((A(i, i)+value) - B(i, i))  
                 < tol * std::max(1.0, (double) (std::abs(A(i, i)) + std::abs(B(i, i) + value))));
  }
}
template<typename Real>
static void AssertDiagEqual(const PackedMatrix<Real> &A,
                        const PackedMatrix<Real> &B,
                        float value,
                        float tol = 0.001) {
  for (MatrixIndexT i = 0; i < A.NumRows(); i++) {
    KALDI_ASSERT(std::abs((A(i, i)+value) - B(i, i))  
                 < tol * std::max(1.0, (double) (std::abs(A(i, i)) + std::abs(B(i, i) + value))));
  }
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

template<typename Real>
static bool ApproxEqual(const PackedMatrix<Real> &A,
                        const PackedMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  PackedMatrix<Real> diff(A);
  diff.AddPacked(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min()),
      d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}

/*
 * Unit Tests
 */
template<typename Real>
static void UnitTestCuPackedMatrixConstructor() { 
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;

    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);
    CuPackedMatrix<Real> C(B);
    AssertEqual(B, C);
  }
}

template<typename Real>
static void UnitTestCuPackedMatrixCopy() { 
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);

    CuPackedMatrix<Real> C(dim);
    C.CopyFromPacked(A);
    CuPackedMatrix<Real> D(dim);
    D.CopyFromPacked(B);
    AssertEqual(C, D);

    PackedMatrix<Real> E(dim);
    D.CopyToPacked(&E);
    AssertEqual(A, E);
  }
}

template<typename Real>
static void UnitTestCuPackedMatrixTrace() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);
    
    AssertEqual(A.Trace(), B.Trace());
  }
}

template<typename Real>
static void UnitTestCuPackedMatrixScale() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);

    Real scale_factor = 23.5896223;
    A.Scale(scale_factor); 
    B.Scale(scale_factor);
    AssertEqual(A, B);
  }
}

template<typename Real>
static void UnitTestCuPackedMatrixScaleDiag() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);

    Real scale_factor = 23.5896223;
    A.ScaleDiag(scale_factor); 
    B.ScaleDiag(scale_factor);
    AssertEqual(A, B);
  }
}



template<typename Real>
static void UnitTestCuPackedMatrixAddToDiag() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    PackedMatrix<Real> A(dim);
    A.SetRandn();
    CuPackedMatrix<Real> B(A);

    Real value = rand() % 50;
    B.AddToDiag(value); 
    
    AssertDiagEqual(A, B, value);
  }
}

template<typename Real>
static void UnitTestCuPackedMatrixSetUnit() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    CuPackedMatrix<Real> A(dim);
    A.SetUnit();
    
    for (MatrixIndexT i = 0; i < A.NumRows(); i++) {
      for (MatrixIndexT j = 0; j < A.NumRows(); j++) {
        if (i != j) { 
          KALDI_ASSERT(A(i, j) == 0);
        } else {
          KALDI_ASSERT(A(i, j) == 1.0);
        }
      }
    } 
  }
}


template<typename Real> void CudaPackedMatrixUnitTest() {
  UnitTestCuPackedMatrixConstructor<Real>();
  //UnitTestCuPackedMatrixCopy<Real>();
  UnitTestCuPackedMatrixTrace<Real>();
  UnitTestCuPackedMatrixScale<Real>();
  UnitTestCuPackedMatrixAddToDiag<Real>();
  UnitTestCuPackedMatrixSetUnit<Real>();
}

} // namespace kaldi


int main() {
  using namespace kaldi;
#if HAVE_CUDA == 1
  // Select the GPU
  CuDevice::Instantiate().SelectGpuId("yes");
#endif
  kaldi::CudaPackedMatrixUnitTest<float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported()) {
    kaldi::CudaPackedMatrixUnitTest<double>();
  } else {
    KALDI_WARN << "Double precision not supported";
  }
#else
  kaldi::CudaPackedMatrixUnitTest<double>();
#endif
  
  KALDI_LOG << "Tests succeeded";
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

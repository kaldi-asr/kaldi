// cudamatrix/cu-sp-matrix-test.cc
//
// Copyright 2013  Ehsan Variani
//                 Lucas Ondel

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
//
// UnitTests for testing cu-sp-matrix.h methods.
//

#include <iostream>
#include <vector>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "cu-device.h"
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
template<class Real>
static void AssertEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0; i < A.Dim(); i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}

template<class Real>
static bool ApproxEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0; i < A.Dim(); i++)
    if (std::abs(A(i)-B(i)) > tol) return false;
  return true;
}

template<class Real>
static void AssertEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  for (MatrixIndexT i = 0; i < A.NumRows(); i++)
    for (MatrixIndexT j = 0; j <= i ; j++)
{
// std::cout<<A(i, j) <<' '<< B(i, j)<<' '<<tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j))))<<std::endl;


      KALDI_ASSERT(std::abs(A(i, j) - B(i, j))
                   < tol * std::max(1.0, (double) (std::abs(A(i, j)) + std::abs(B(i, j)))));
}
}
template<class Real>
static bool ApproxEqual(const SpMatrix<Real> &A,
                        const SpMatrix<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  SpMatrix<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min()),
      d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}

/*
 * Unit Tests
 */
template<class Real>
static void UnitTestCuSpMatrixConstructor() { 
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;

    Matrix<Real> A(dim, dim);
    A.SetRandn();
    SpMatrix<Real> B(A, kTakeLower);

    CuMatrix<Real> C(A);
    CuSpMatrix<Real> D(C, kTakeLower);

    SpMatrix<Real> E(dim);
    D.CopyToSp(&E);

    SpMatrix<Real> F(D);
    
    AssertEqual(F, B);
     //added by hxu, to test copy from SpMatrix to CuSpMatrix

    AssertEqual(B, E);
  }
}




template<class Real>
static void UnitTestCuSpMatrixOperator() {
  SpMatrix<Real> A(100);
  A.SetRandn();

  CuSpMatrix<Real> B(100);
  B.CopyFromSp(A);

  for (MatrixIndexT i = 0; i < A.NumRows(); i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      KALDI_ASSERT(std::abs(A(i, j) - B(i, j)) < 0.0001);
  }
}

template<class Real>
static void UnitTestCuSpMatrixAddToDiag() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10*i;
    SpMatrix<Real> A(dim);
    A.SetRandn();
    CuSpMatrix<Real> B(A);
    
    Matrix<Real> D(A);
    A.AddToDiag(i);

    CuMatrix<Real> C(B);
    B.AddToDiag(i);
    
    SpMatrix<Real> E(dim);
    B.CopyToSp(&E);
    
    AssertEqual(A, E);    
  }
}

template<class Real>
static void UnitTestCuSpMatrixInvert() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10*i;
    SpMatrix<Real> A(dim);
    A.SetRandn();
    CuSpMatrix<Real> B(A);
    
    Matrix<Real> D(A);
    A.AddMat2(1.0, D, kTrans, 1.0);
    A.AddToDiag(i);

    CuMatrix<Real> C(B);
    B.AddMat2(1.0, C, kTrans, 1.0);
    B.AddToDiag(i);
    
    A.Invert();
    B.Invert();
    
    SpMatrix<Real> E(dim);
    B.CopyToSp(&E);
    
    AssertEqual(A, E);    
  }
}

// TODO (variani) : fails for dim = 0 
template<class Real>
static void UnitTestCuSpMatrixAddVec2() {
  for (int32 i = 0; i < 50; i++) {
    MatrixIndexT dim = 1 + rand() % 200;
    SpMatrix<Real> A(dim);
    A.SetRandn();
    CuSpMatrix<Real> B(A);
    
    Vector<Real> C(dim);
    C.SetRandn();
    CuVector<Real> D(C);
    Real alpha = RandGauss();

    A.AddVec2(alpha, C);
    B.AddVec2(alpha, D);

    SpMatrix<Real> E(dim);
    B.CopyToSp(&E);

    AssertEqual(A, E);
  }
}

template<class Real>
static void UnitTestCuSpMatrixAddMat2() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim_row = 15 * i + rand() % 10;
    MatrixIndexT dim_col = 7 *i + rand() % 10;
    Matrix<Real> A(dim_row, dim_col);
    A.SetRandn();
    CuMatrix<Real> B(A);

    SpMatrix<Real> C(dim_col);
    C.SetRandn();
    CuSpMatrix<Real> D(C);

    const Real alpha = 2.0;
    const Real beta = 3.0;

    C.AddMat2(alpha, A, kTrans, beta);
    D.AddMat2(alpha, B, kTrans, beta);

    SpMatrix<Real> E(dim_col);
    D.CopyToSp(&E);

    AssertEqual(C, E);
  }
}

template<class Real>
static void UnitTestCuSpMatrixAddSp() {
  for (MatrixIndexT i = 1; i < 50; i++) {
    MatrixIndexT dim = 7 * i + rand() % 10;
    
    SpMatrix<Real> A(dim);
    A.SetRandn();
    CuSpMatrix<Real> B(A);

    SpMatrix<Real> C(dim);
    C.SetRandn();
    const CuSpMatrix<Real> D(C);

    const Real alpha = 2.0;
    
    A.AddSp(alpha, C);
    B.AddSp(alpha, D);

    SpMatrix<Real> E(dim);
    B.CopyToSp(&E);

    AssertEqual(A, E);
  }
}

template<class Real, class OtherReal>
static void UnitTestCuSpMatrixTraceSpSp() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    SpMatrix<Real> A(dim);
    A.SetRandn();
    const CuSpMatrix<Real> B(A);
    SpMatrix<OtherReal> C(dim);
    C.SetRandn();
    const CuSpMatrix<OtherReal> D(C);


#ifdef KALDI_PARANOID
    KALDI_ASSERT(TraceSpSp(A, C), TraceSpSp(B, D));
#endif
  }
}

template<class Real, class OtherReal>
static void UnitTestCuSpMatrixAddSp() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    SpMatrix<Real> A(dim);
    A.SetRandn();
    const CuSpMatrix<Real> B(A);
    SpMatrix<OtherReal> C(dim);
    C.SetRandn();
    const CuSpMatrix<OtherReal> D(C);
    
    A.AddSp(1.0, C);
    B.AddSp(1.0, D);
    
    AssertEqual(A, B);

  }
}

template<class Real> void CudaSpMatrixUnitTest() {
  UnitTestCuSpMatrixConstructor<Real>();
  UnitTestCuSpMatrixOperator<Real>();
  UnitTestCuSpMatrixInvert<Real>();
  UnitTestCuSpMatrixAddVec2<Real>();
  UnitTestCuSpMatrixAddMat2<Real>();
  UnitTestCuSpMatrixAddSp<Real>();
  UnitTestCuSpMatrixAddToDiag<Real>();
}

template<class Real, class OtherReal> void CudaSpMatrixUnitTest() {
  UnitTestCuSpMatrixTraceSpSp<Real, OtherReal>();

}

} // namespace kaldi


int main() {
  using namespace kaldi;
  // Select the GPU
#if HAVE_CUDA == 1
  kaldi::int32 use_gpu_id = -2;
  CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif
  //  std::cout<<"here\n";

  kaldi::CudaSpMatrixUnitTest<float>();
  kaldi::CudaSpMatrixUnitTest<float, float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported()) {
    kaldi::CudaSpMatrixUnitTest<double>();
    kaldi::CudaSpMatrixUnitTest<float, double>();
    kaldi::CudaSpMatrixUnitTest<double, float>();
    kaldi::CudaSpMatrixUnitTest<double, double>();
  } else {
    KALDI_WARN << "Double precision not supported";
  }
#else
  kaldi::CudaSpMatrixUnitTest<float, double>();
  kaldi::CudaSpMatrixUnitTest<double, float>();
  kaldi::CudaSpMatrixUnitTest<double, double>();
#endif
  KALDI_LOG << "Tests succeeded";
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

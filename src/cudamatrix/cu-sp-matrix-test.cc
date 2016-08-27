// cudamatrix/cu-sp-matrix-test.cc
//
// Copyright 2013  Ehsan Variani
//                 Lucas Ondel
//                 Johns Hopkins University (author: Daniel Povey)

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
//
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
 * Unit Tests
 */
template<typename Real>
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

    KALDI_ASSERT(!B.IsUnit());
    B.SetZero();
    B.SetDiag(1.0);
    KALDI_ASSERT(B.IsUnit());
  }
}

template<typename Real>
static void UnitTestCuSpMatrixApproxEqual() {

  for (int32 i = 0; i < 10; i++) {
    int32 dim = 1 + Rand() % 10;
    SpMatrix<Real> A(dim), B(dim);
    A.SetRandn();
    B.SetRandn();
    BaseFloat threshold = 0.01;
    for (int32 j = 0; j < 20; j++, threshold *= 1.3) {
      bool b1 = A.ApproxEqual(B, threshold);
      SpMatrix<Real> diff(A);
      diff.AddSp(-1.0, B);
      bool b2 = (diff.FrobeniusNorm() < threshold * std::max(A.FrobeniusNorm(),
                                                             B.FrobeniusNorm()));
      KALDI_ASSERT(b1 == b2);
    }
  }
  
}



template<typename Real>
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

template<typename Real>
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


template<typename Real>
static void UnitTestCuSpMatrixCopyFromMat() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    SpCopyType copy_type = (i % 3 == 0 ? kTakeMean :
                            (i % 3 == 1 ? kTakeLower : kTakeUpper));
    MatrixIndexT dim = 10 * i + Rand() % 5;
    CuMatrix<Real> A(dim, dim);
    A.SetRandn();
    Matrix<Real> A2(A);
    
    CuSpMatrix<Real> B(A, copy_type);
    SpMatrix<Real> B2(A2, copy_type);
    SpMatrix<Real> B3(B);
    if (!ApproxEqual(B2, B3) ) {
      KALDI_ERR << "Matrices differ, A = " << A << ", B2 = " << B2 << ", B3(CUDA) = " << B3;
    }
    KALDI_ASSERT(B3.Trace() != 0);
  }
}

template<typename Real>
static void UnitTestCuSpMatrixInvert() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10*i + Rand() % 5;
    CuSpMatrix<Real> A(dim);
    A.SetRandn();
    KALDI_ASSERT(A.Trace() != 0.0); // true with probability 1...
    SpMatrix<Real> B(A);
    
    CuMatrix<Real> D(A);
    A.AddMat2(1.0, D, kTrans, 1.0);
    A.AddToDiag(i);

    Matrix<Real> C(B);
    B.AddMat2(1.0, C, kTrans, 1.0);
    B.AddToDiag(i);

    CuSpMatrix<Real> Acopy(A);
    A.Invert();
    B.Invert();
    
    SpMatrix<Real> A2(A);
    AssertEqual(A2, B);

    CuMatrix<Real> I(dim, dim);
    I.AddMatMat(1.0, CuMatrix<Real>(Acopy), kNoTrans, CuMatrix<Real>(A), kNoTrans, 0.0);
    KALDI_ASSERT(I.IsUnit(0.01));
  }
}

// TODO (variani) : fails for dim = 0 
template<typename Real>
static void UnitTestCuSpMatrixAddVec2() {
  for (int32 i = 0; i < 50; i++) {
    MatrixIndexT dim = 1 + Rand() % 200;
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

template<typename Real>
static void UnitTestCuSpMatrixAddMat2() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim_row = 15 * i + Rand() % 10;
    MatrixIndexT dim_col = 7 *i + Rand() % 10;
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

template<typename Real>
static void UnitTestCuSpMatrixAddSp() {
  for (MatrixIndexT i = 1; i < 50; i++) {
    MatrixIndexT dim = 7 * i + Rand() % 10;
    
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

template<typename Real, typename OtherReal>
static void UnitTestCuSpMatrixTraceSpSp() {
  for (MatrixIndexT i = 1; i < 2; i++) {
    MatrixIndexT dim = 100 + Rand() % 255;
    
    SpMatrix<Real> A(dim);
    A.SetRandn();
    const CuSpMatrix<Real> B(A);
    SpMatrix<OtherReal> C(dim);
    C.SetRandn();
    const CuSpMatrix<OtherReal> D(C);

    Real t1 = TraceSpSp(A, C), t2 = TraceSpSp(B, D);
    KALDI_ASSERT(ApproxEqual(t1, t2));
  }
}


template<typename Real>
void UnitTestCuSpMatrixSetUnit() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 100 * i + Rand() % 255;
    if (i % 5 == 0) dim = 0;
    CuSpMatrix<Real> S1(dim), S2(dim), S4(dim);
    S1.SetRandn();
    S2.SetRandn();
    S4.SetRandn();
    SpMatrix<Real> S3(dim);
    S3.SetUnit();
    S1.SetUnit();
    S2.SetZero();
    S2.SetDiag(1.0);
    S4.SetZero();
    S4.AddToDiag(0.4);
    S4.AddToDiag(0.6);
    CuSpMatrix<Real> cu_S3(S3);
    KALDI_LOG << "S1 norm is " << S1.FrobeniusNorm();
    KALDI_LOG << "S2 norm is " << S2.FrobeniusNorm();
    KALDI_LOG << "S3 norm is " << S3.FrobeniusNorm();
    AssertEqual(S1, cu_S3);
    AssertEqual(S2, cu_S3);
    AssertEqual(S4, cu_S3);
  }
}
   
template<class Real>
static void UnitTestCuSpMatrixIO() {
  for (int32 i = 0; i < 10; i++) {
    int32 dimM = Rand() % 255;
    if (i % 5 == 0) { dimM = 0; }
    CuSpMatrix<Real> mat(dimM);
    mat.SetRandn();
    std::ostringstream os;
    bool binary = (i % 4 < 2);
    mat.Write(os, binary);

    CuSpMatrix<Real> mat2;
    std::istringstream is(os.str());
    mat2.Read(is, binary);
    AssertEqual(mat, mat2);
  }
}




template<typename Real, typename OtherReal>
static void UnitTestCuSpMatrixAddSp() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 100 * i + Rand() % 255;
    
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

template<typename Real> void CudaSpMatrixUnitTest() {
  UnitTestCuSpMatrixIO<Real>();
  UnitTestCuSpMatrixConstructor<Real>();
  UnitTestCuSpMatrixOperator<Real>();
  UnitTestCuSpMatrixApproxEqual<Real>();
  UnitTestCuSpMatrixInvert<Real>();
  UnitTestCuSpMatrixCopyFromMat<Real>();
  UnitTestCuSpMatrixAddVec2<Real>();
  UnitTestCuSpMatrixAddMat2<Real>();
  UnitTestCuSpMatrixAddSp<Real>();
  UnitTestCuSpMatrixAddToDiag<Real>();
  UnitTestCuSpMatrixSetUnit<Real>();
}

template<typename Real, typename OtherReal> void CudaSpMatrixUnitTest() {
  UnitTestCuSpMatrixTraceSpSp<Real, OtherReal>();

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

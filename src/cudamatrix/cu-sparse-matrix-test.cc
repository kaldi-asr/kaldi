// cudamatrix/cu-sparse-matrix-test.cc

// Copyright 2015      Guoguo Chen

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
#include "cudamatrix/cu-matrix-lib.h"

using namespace kaldi;


namespace kaldi {

template <typename Real>
static void UnitTestCuSparseMatrixTraceMatSmat() {
  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    CuMatrix<Real> M1(row, col);
    CuMatrix<Real> M2(col, row);
    CuMatrix<Real> M3(row, col);
    M1.SetRandn();
    M2.SetRandn();
    M3.SetRandn();

    SparseMatrix<Real> Smat1(row, col);
    SparseMatrix<Real> Smat2(col, row);
    Smat1.SetRandn(0.8);
    Smat2.SetRandn(0.8);
    CuSparseMatrix<Real> CuSmat1(Smat1);
    CuSparseMatrix<Real> CuSmat2(Smat2);

    CuSmat1.CopyToMat(&M1);
    CuSmat2.CopyToMat(&M2);

    Real trace1 = TraceMatMat(M3, M1, kTrans);
    Real trace2 = TraceMatSmat(M3, CuSmat1, kTrans);
    AssertEqual(trace1, trace2, 0.00001);

    trace1 = TraceMatMat(M3, M2, kNoTrans);
    trace2 = TraceMatSmat(M3, CuSmat2, kNoTrans);
    AssertEqual(trace1, trace2, 0.00001);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixSum() {
  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    CuMatrix<Real> M(row, col);
    M.SetRandn();

    SparseMatrix<Real> Smat(row, col);
    Smat.SetRandn(0.8);
    CuSparseMatrix<Real> CuSmat(Smat);

    CuSmat.CopyToMat(&M);

    Real sum1 = CuSmat.Sum();
    Real sum2 = M.Sum();
    AssertEqual(sum1, sum2, 0.00001);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixFrobeniusNorm() {
  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    CuMatrix<Real> M(row, col);
    M.SetRandn();

    SparseMatrix<Real> Smat(row, col);
    Smat.SetRandn(0.8);
    CuSparseMatrix<Real> CuSmat(Smat);

    CuSmat.CopyToMat(&M);

    Real sum1 = CuSmat.FrobeniusNorm();
    Real sum2 = M.FrobeniusNorm();
    AssertEqual(sum1, sum2, 0.00001);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixCopyToSmat() {
  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    SparseMatrix<Real> Smat1(row, col);
    Smat1.SetRandn(0.8);
    CuSparseMatrix<Real> CuSmat1(Smat1);

    SparseMatrix<Real> Smat2(col, row);
    CuSmat1.CopyToSmat(&Smat2);
    CuSparseMatrix<Real> CuSmat2(Smat2);

    CuMatrix<Real> M1(row, col);
    CuMatrix<Real> M2(row, col);

    CuSmat1.CopyToMat(&M1);
    CuSmat2.CopyToMat(&M2);

    AssertEqual(M1, M2, 0.00001);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixSwap() {
  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    CuMatrix<Real> M1, M2, M3, M4;

    SparseMatrix<Real> Smat1(row, col);
    Smat1.SetRandn(0.8);
    SparseMatrix<Real> Smat2 = Smat1;
    M2.Resize(Smat2.NumRows(), Smat2.NumCols());
    CuSparseMatrix<Real> tmp1(Smat2);
    tmp1.CopyToMat(&M2);

    SparseMatrix<Real> Smat3(col, row);
    Smat3.SetRandn(0.5);
    CuSparseMatrix<Real> CuSmat1(Smat3);
    CuSparseMatrix<Real> CuSmat2 = CuSmat1;
    M4.Resize(CuSmat2.NumRows(), CuSmat2.NumCols());
    CuSmat2.CopyToMat(&M4);

    CuSmat1.Swap(&Smat1);

    M1.Resize(Smat1.NumRows(), Smat1.NumCols());
    M3.Resize(CuSmat1.NumRows(), CuSmat1.NumCols());
    CuSparseMatrix<Real> tmp2(Smat1);
    tmp2.CopyToMat(&M1);
    CuSmat1.CopyToMat(&M3);

    AssertEqual(M3, M2, 0.00001);
    AssertEqual(M1, M4, 0.00001);

    CuSmat2.Swap(&CuSmat1);

    M1.Resize(CuSmat1.NumRows(), CuSmat1.NumCols());
    CuSmat1.CopyToMat(&M1);
    M2.Resize(CuSmat2.NumRows(), CuSmat2.NumCols());
    CuSmat2.CopyToMat(&M2);

    AssertEqual(M3, M2, 0.00001);
    AssertEqual(M1, M4, 0.00001);

  }
}

template <typename Real>
void CudaSparseMatrixUnitTest() {
  UnitTestCuSparseMatrixTraceMatSmat<Real>();
  UnitTestCuSparseMatrixSum<Real>();
  UnitTestCuSparseMatrixFrobeniusNorm<Real>();
  UnitTestCuSparseMatrixCopyToSmat<Real>();
  UnitTestCuSparseMatrixSwap<Real>();
}


} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    kaldi::CudaSparseMatrixUnitTest<float>();

#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaSparseMatrixUnitTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaSparseMatrixUnitTest<double>();
#endif

    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
  SetVerboseLevel(4);
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

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

#include <vector>
#include <algorithm>
#include <cstdlib>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix-lib.h"

namespace kaldi {

template <typename Real>
static void UnitTestCuSparseMatrixTraceMatSmat() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    CuMatrix<Real> mat1(row, col);
    CuMatrix<Real> mat2(col, row);
    CuMatrix<Real> mat3(row, col);
    mat1.SetRandn();
    mat2.SetRandn();
    mat3.SetRandn();

    SparseMatrix<Real> smat1(row, col);
    SparseMatrix<Real> smat2(col, row);
    smat1.SetRandn(0.8);
    smat2.SetRandn(0.8);
    CuSparseMatrix<Real> cu_smat1(smat1);
    CuSparseMatrix<Real> cu_smat2(smat2);

    cu_smat1.CopyToMat(&mat1);
    cu_smat2.CopyToMat(&mat2);

    Real trace1 = TraceMatMat(mat3, mat1, kTrans);
    Real trace2 = TraceMatSmat(mat3, cu_smat1, kTrans);
    AssertEqual(trace1, trace2, 0.00001);

    trace1 = TraceMatMat(mat3, mat2, kNoTrans);
    trace2 = TraceMatSmat(mat3, cu_smat2, kNoTrans);
    AssertEqual(trace1, trace2, 0.00001);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixSum() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    CuMatrix<Real> mat(row, col);
    mat.SetRandn();

    SparseMatrix<Real> smat(row, col);
    smat.SetRandn(0.8);
    CuSparseMatrix<Real> cu_smat(smat);

    cu_smat.CopyToMat(&mat);

    Real sum1 = cu_smat.Sum();
    Real sum2 = mat.Sum();
    KALDI_ASSERT(fabs(sum1 - sum2) < 1.0e-05);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixFrobeniusNorm() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    CuMatrix<Real> mat(row, col);
    mat.SetRandn();

    SparseMatrix<Real> smat(row, col);
    smat.SetRandn(0.8);
    CuSparseMatrix<Real> cu_smat(smat);

    cu_smat.CopyToMat(&mat);

    Real sum1 = cu_smat.FrobeniusNorm();
    Real sum2 = mat.FrobeniusNorm();
    AssertEqual(sum1, sum2, 0.00001);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixCopyToSmat() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    SparseMatrix<Real> smat1(row, col);
    smat1.SetRandn(0.8);
    CuSparseMatrix<Real> cu_smat1(smat1);

    SparseMatrix<Real> smat2(col, row);
    cu_smat1.CopyToSmat(&smat2);
    CuSparseMatrix<Real> cu_smat2(smat2);

    CuMatrix<Real> mat1(row, col);
    CuMatrix<Real> mat2(row, col);

    cu_smat1.CopyToMat(&mat1);
    cu_smat2.CopyToMat(&mat2);

    AssertEqual(mat1, mat2, 0.00001);
  }
}

template <typename Real>
static void UnitTestCuSparseMatrixSwap() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    CuMatrix<Real> mat1, mat2, mat3, mat4;

    SparseMatrix<Real> smat1(row, col);
    smat1.SetRandn(0.8);
    SparseMatrix<Real> smat2 = smat1;
    mat2.Resize(smat2.NumRows(), smat2.NumCols());
    CuSparseMatrix<Real> tmp1(smat2);
    tmp1.CopyToMat(&mat2);

    SparseMatrix<Real> smat3(col, row);
    smat3.SetRandn(0.5);
    CuSparseMatrix<Real> cu_smat1(smat3);
    CuSparseMatrix<Real> cu_smat2 = cu_smat1;
    mat4.Resize(cu_smat2.NumRows(), cu_smat2.NumCols());
    cu_smat2.CopyToMat(&mat4);

    cu_smat1.Swap(&smat1);

    mat1.Resize(smat1.NumRows(), smat1.NumCols());
    mat3.Resize(cu_smat1.NumRows(), cu_smat1.NumCols());
    CuSparseMatrix<Real> tmp2(smat1);
    tmp2.CopyToMat(&mat1);
    cu_smat1.CopyToMat(&mat3);

    AssertEqual(mat3, mat2, 0.00001);
    AssertEqual(mat1, mat4, 0.00001);

    cu_smat2.Swap(&cu_smat1);

    mat1.Resize(cu_smat1.NumRows(), cu_smat1.NumCols());
    cu_smat1.CopyToMat(&mat1);
    mat2.Resize(cu_smat2.NumRows(), cu_smat2.NumCols());
    cu_smat2.CopyToMat(&mat2);

    AssertEqual(mat3, mat2, 0.00001);
    AssertEqual(mat1, mat4, 0.00001);
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


}  // namespace kaldi


int main() {
  for (kaldi::int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    kaldi::CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      kaldi::CuDevice::Instantiate().SelectGpuId("no");
    else
      kaldi::CuDevice::Instantiate().SelectGpuId("yes");
#endif

    kaldi::CudaSparseMatrixUnitTest<float>();

#if HAVE_CUDA == 1
    if (kaldi::CuDevice::Instantiate().DoublePrecisionSupported()) {
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
  kaldi::SetVerboseLevel(4);
#if HAVE_CUDA == 1
  kaldi::CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

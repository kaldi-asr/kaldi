// cudamatrix/cu-sparse-matrix-test.cc

// Copyright 2015      Guoguo Chen
//           2017      Shiyin Kang

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

using namespace kaldi;

namespace kaldi {

template<typename Real>
static void UnitTestCuSparseMatrixConstructFromIndexes() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    MatrixTransposeType trans = (i % 2 == 0 ? kNoTrans : kTrans);

    std::vector<int32> idx(row);
    Vector<Real> weights(row);
    for (int i = 0; i < row; ++i) {
      idx[i] = Rand() % col;
      weights(i) = Rand() % 100;
    }
    CuArray<int32> cuidx(idx);
    CuVector<Real> cuweights(weights);

    // construct smat
    SparseMatrix<Real> smat1(idx, col, trans);
    CuSparseMatrix<Real> cusmat1(cuidx, col, trans);

    // copy to mat
    Matrix<Real> mat1(smat1.NumRows(), smat1.NumCols());
    smat1.CopyToMat(&mat1);
    CuMatrix<Real> cumat1(cusmat1.NumRows(), cusmat1.NumCols());
    cusmat1.CopyToMat(&cumat1);

    // compare
    Matrix<Real> mat1ver2(cumat1);
    AssertEqual(mat1, mat1ver2, 0.00001);

    // construct smat with weights
    SparseMatrix<Real> smat2(idx, weights, col, trans);
    CuSparseMatrix<Real> cusmat2(cuidx, cuweights, col, trans);

    // copy to mat
    Matrix<Real> mat2(smat2.NumRows(), smat2.NumCols());
    smat2.CopyToMat(&mat2);
    CuMatrix<Real> cumat2(cusmat2.NumRows(), cusmat2.NumCols());
    cusmat2.CopyToMat(&cumat2);

    // compare
    Matrix<Real> mat2ver2(cumat2);
    AssertEqual(mat2, mat2ver2, 0.00001);
  }
}


template <typename Real>
static void UnitTestCuSparseMatrixSelectRowsAndTranspose() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    SparseMatrix<Real> sp_input(row, col);
    sp_input.SetRandn(0.8);
    CuSparseMatrix<Real> cusp_input(sp_input);

    std::vector<int32> idx(row + col);
    for (auto && e : idx) {
      e = Rand() % row;
    }
    CuArray<int32> cu_idx(idx);

    // select on cpu
    SparseMatrix<Real> sp_selected2;
    sp_selected2.SelectRows(idx, sp_input);
    // select on gpu
    CuSparseMatrix<Real> cusp_selected;
    cusp_selected.SelectRows(cu_idx, cusp_input);

    // transpose by CopyToMat
    CuSparseMatrix<Real> cusp_selected2(sp_selected2);
    CuMatrix<Real> cu_selected2_trans(cusp_selected2.NumCols(),
                                      cusp_selected2.NumRows());
    cusp_selected2.CopyToMat(&cu_selected2_trans, kTrans);
    // transpose by Constructor
    CuSparseMatrix<Real> cusp_selected_trans(cusp_selected, kTrans);
    CuMatrix<Real> cu_selected_trans(cusp_selected_trans.NumRows(),
                                     cusp_selected_trans.NumCols());
    cusp_selected_trans.CopyToMat(&cu_selected_trans);

    CuMatrix<Real> cu_selected(row+col, col);
    CuMatrix<Real> cu_selected2(row+col, col);
    cusp_selected.CopyToMat(&cu_selected);
    cusp_selected2.CopyToMat(&cu_selected2);

    AssertEqual(cu_selected, cu_selected2, 0.00001);
    AssertEqual(cu_selected_trans, cu_selected2_trans, 0.00001);
  }
}

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
    CuSparseMatrix<Real> cu_smat1x(smat1);
    CuSparseMatrix<Real> cu_smat1(cu_smat1x);  // Test constructor.
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
    KALDI_ASSERT(fabs(sum1 - sum2) < 1.0e-04);
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
  UnitTestCuSparseMatrixConstructFromIndexes<Real>();
  UnitTestCuSparseMatrixSelectRowsAndTranspose<Real>();
  UnitTestCuSparseMatrixTraceMatSmat<Real>();
  UnitTestCuSparseMatrixSum<Real>();
  UnitTestCuSparseMatrixFrobeniusNorm<Real>();
  UnitTestCuSparseMatrixCopyToSmat<Real>();
  UnitTestCuSparseMatrixSwap<Real>();
}


}  // namespace kaldi


int main() {
  int32 loop = 0;
#if HAVE_CUDA == 1
  for (; loop < 2; loop++) {
    CuDevice::Instantiate().SetDebugStrideMode(true);
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
    SetVerboseLevel(4);
#if HAVE_CUDA == 1
  }
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

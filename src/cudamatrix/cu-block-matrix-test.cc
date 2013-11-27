// cudamatrix/cu-block-matrix-test.cc

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
#include "cudamatrix/cu-matrix-lib.h"

using namespace kaldi;


namespace kaldi {

/*
 * ASSERTS
 */
template<typename Real> 
static void AssertEqual(const MatrixBase<Real> &A,
                        const MatrixBase<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++) {
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) <= tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
  }
}


template<typename Real> 
static void AssertEqual(const CuMatrixBase<Real> &A,
                        const CuMatrixBase<Real> &B,
                        float tol = 0.001) {
  Real Anorm = A.FrobeniusNorm(), Bnorm = B.FrobeniusNorm();
  CuMatrix<Real> diff(A);
  diff.AddMat(-1.0, B);
  Real diff_norm = diff.FrobeniusNorm();
  if (diff_norm > tol * 0.5 * (Anorm + Bnorm)) {
    KALDI_LOG << "A = " << A;
    KALDI_LOG << "B = " << B;
    KALDI_ERR << "Matrices differ, " << diff_norm << " > " << tol << " * 0.5 *  ( "
              << Anorm << " + " << Bnorm << " ). ";
  }
}


template<typename Real> 
static void AssertEqual(const CuBlockMatrix<Real> &A,
                        const CuBlockMatrix<Real> &B,
                        float tol = 0.001) {
  CuMatrix<Real> Acopy(A), Bcopy(B);
  AssertEqual(Acopy, Bcopy, tol);
}


template<typename Real> 
static bool ApproxEqual(const CuBlockMatrix<Real> &A,
                        const CuBlockMatrix<Real> &B,
                        float tol = 0.001) {
  CuMatrix<Real> Acopy(A), Bcopy(B);
  return Acopy.ApproxEqual(Bcopy, tol);
}





template<class Real>
static void UnitTestCuBlockMatrixIO() {
  for (int32 i = 0; i < 10; i++) {
    int32 num_blocks = rand() % 5;
    std::vector<CuMatrix<Real> > data(num_blocks);
    for (int32 b = 0; b < num_blocks; b++) {
      int32 dimM = 100 + rand() % 255, dimN = 10 + rand() % 20;
      if (b % 2 == 0) std::swap(dimM, dimN);
      data[b].Resize(dimM, dimN);
      data[b].SetRandn();
    }
    CuBlockMatrix<Real> B(data);

    std::ostringstream os;
    bool binary = (i % 4 < 2);
    B.Write(os, binary);

    CuBlockMatrix<Real> B2;
    std::istringstream is(os.str());
    B2.Read(is, binary);

    CuMatrix<Real> mat(B), mat2(B2);
    AssertEqual(mat, mat2);
    if (!data.empty())
      KALDI_ASSERT(mat.Sum() != 0.0);
  }
}



template<class Real>
static void UnitTestCuBlockMatrixAddMatBlock() {
  for (int32 i = 0; i < 20; i++) {
    int32 num_blocks = rand() % 5;
    std::vector<CuMatrix<Real> > data(num_blocks);
    for (int32 b = 0; b < num_blocks; b++) {
      int32 dimM = 100 + rand() % 255, dimN = 10 + rand() % 20;
      // early failures will have small dim for easier eyeballing.
      if (b % 2 == 0) std::swap(dimM, dimN);
      data[b].Resize(dimM, dimN);
      data[b].SetRandn();
    }
    CuBlockMatrix<Real> B(data);
    int32 B_num_rows = B.NumRows(), B_num_cols = B.NumCols();
    // will do X += A B

    MatrixTransposeType transB = (i % 2 == 1 ? kTrans : kNoTrans),
        transA = (i % 3 == 1 ? kTrans : kNoTrans);
    if (transB == kTrans) std::swap(B_num_rows, B_num_cols);
    
    int32 X_num_rows = 100 + rand() % 255, X_num_cols = B_num_cols,
        A_num_rows = X_num_rows, A_num_cols = B_num_rows;
    if (data.size() == 0) { X_num_rows = 0; A_num_rows = 0; }
    if (transA == kTrans) std::swap(A_num_rows, A_num_cols);

    Real alpha = 2.0, beta = -1.0;
    CuMatrix<Real> X(X_num_rows, X_num_cols);
    X.SetRandn();
    CuMatrix<Real> A(A_num_rows, A_num_cols);
    A.SetRandn();

    CuMatrix<Real> Xcopy(X), Bcopy(B), Xorig(X), Aorig(A);
    Xcopy.AddMatMat(alpha, A, transA, Bcopy, transB, beta);
    X.AddMatBlock(alpha, A, transA, B, transB, beta);

    AssertEqual(X, Xcopy);
  }
}


template<class Real>
static void UnitTestCuBlockMatrixAddMatMat() {
  for (int32 i = 0; i < 20; i++) {
    int32 num_blocks = rand() % 5;
    std::vector<CuMatrix<Real> > data(num_blocks);
    for (int32 b = 0; b < num_blocks; b++) {
      int32 dimM = 100 + rand() % 255, dimN = 10 + rand() % 20;
      if (i == 0) { dimM = 1; dimN = 1; }
      // early failures will have small dim for easier eyeballing.
      if (b % 2 == 0) std::swap(dimM, dimN);
      data[b].Resize(dimM, dimN);
      data[b].SetRandn();
    }    
    
    CuBlockMatrix<Real> B(data);
    int32 B_num_rows = B.NumRows(), B_num_cols = B.NumCols();
    // will do B += C D

    int32 C_num_rows = B_num_rows, C_num_cols = 100 + rand() % 255;
    if (C_num_rows == 0) C_num_cols = 0;
    int32 D_num_rows = C_num_cols, D_num_cols = B_num_cols;

    MatrixTransposeType transC = (i % 2 == 1 ? kTrans : kNoTrans),
        transD = (i % 3 == 1 ? kTrans : kNoTrans);
    if (transC == kTrans) std::swap(C_num_rows, C_num_cols);
    if (transD == kTrans) std::swap(D_num_rows, D_num_cols);

    CuMatrix<Real> C(C_num_rows, C_num_cols), D(D_num_rows, D_num_cols);
    C.SetRandn();
    D.SetRandn();
    
    CuMatrix<Real> Bmat(B);

    Real alpha = 2.0, beta = -1.0;

    CuBlockMatrix<Real> Bcopy(B);

    B.AddMatMat(alpha, C, transC, D, transD, beta);
    
    Bmat.AddMatMat(alpha, C, transC, D, transD, beta);


    // Now check that the block-structured part of Bmat is the
    // same as B.
    Bcopy.CopyFromMat(Bmat); // copy block-structured part from Bmat to Bcopy.

    if (!ApproxEqual(B, Bcopy)) {
      KALDI_WARN << "CuBlockMatrixTest failure, please report to maintainers: Bcopy = "
                 << Bcopy << ", B = " << B << ", C = " << C << ", D = " << D
                 << ", Bmat = " << B << " transD = " << transD << ", transC = "
                 << transC;
      KALDI_ERR << "Please give this log to the maintainers.";
    }
    KALDI_ASSERT(Bmat.Sum() != 0 || B_num_rows == 0);
  }
}


template<typename Real> void CuBlockMatrixUnitTest() {
  UnitTestCuBlockMatrixIO<Real>();
  UnitTestCuBlockMatrixAddMatBlock<Real>();
  UnitTestCuBlockMatrixAddMatMat<Real>();
}


} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("yes"); // -2 .. automatic selection
#endif

    kaldi::CuBlockMatrixUnitTest<float>();
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CuBlockMatrixUnitTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CuBlockMatrixUnitTest<double>();
#endif
    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.\n";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.\n";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

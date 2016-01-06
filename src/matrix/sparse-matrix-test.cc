// matrix/sparse-matrix-test.cc

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

#include "matrix/matrix-lib.h"
#include "util/stl-utils.h"

namespace kaldi {

template <typename Real>
void UnitTestSparseVectorSum() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT dim = 10 + Rand() % 40;

    SparseVector<Real> svec(dim);
    svec.SetRandn(0.8);

    Vector<Real> vec(dim);
    vec.SetRandn();
    svec.CopyElementsToVec(&vec);

    Real sum1 = svec.Sum();
    Real sum2 = vec.Sum();
    AssertEqual(sum1, sum2, 0.00001);
  }
}

template <typename Real>
void UnitTestSparseVectorAddToVec() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT dim = 10 + Rand() % 40;

    SparseVector<Real> svec(dim);
    svec.SetRandn(0.8);

    Vector<Real> vec(dim);
    vec.SetRandn();
    svec.CopyElementsToVec(&vec);

    Vector<Real> other_vec1(dim);
    other_vec1.SetRandn();
    Vector<Real> other_vec2 = other_vec1;

    svec.AddToVec(0.7, &other_vec1);
    other_vec2.AddVec(0.7, vec);
    AssertEqual(other_vec1, other_vec2, 0.00001);
  }
}

template <typename Real>
void UnitTestSparseVectorMax() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT dim = 10 + Rand() % 40;
    if (RandInt(0, 3) == 0)
      dim = RandInt(1, 5);

    SparseVector<Real> svec(dim);
    if (RandInt(0, 3) != 0)
      svec.SetRandn(0.8);

    Vector<Real> vec(dim);
    vec.SetRandn();
    svec.CopyElementsToVec(&vec);

    int32 index1, index2;
    Real max1, max2;

    max1 = svec.Max(&index1);
    max2 = vec.Max(&index2);

    AssertEqual(max1, max2, 0.00001);
    AssertEqual(index1, index2, 0.00001);
  }
}

template <typename Real>
void UnitTestSparseVectorVecSvec() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT dim = 10 + Rand() % 40;

    SparseVector<Real> svec(dim);
    svec.SetRandn(0.8);

    Vector<Real> vec(dim);
    vec.SetRandn();
    svec.CopyElementsToVec(&vec);

    Vector<Real> other_vec(dim);
    other_vec.SetRandn();

    Real product1 = VecSvec(other_vec, svec);
    Real product2 = VecVec(other_vec, vec);

    KALDI_ASSERT(fabs(product1 - product2) < 1.0e-04);
  }
}

template <typename Real>
void UnitTestSparseMatrixSum() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    SparseMatrix<Real> smat(row, col);
    smat.SetRandn(0.8);

    Matrix<Real> mat(row, col);
    mat.SetRandn();
    smat.CopyToMat(&mat);

    Real sum1 = smat.Sum();
    Real sum2 = mat.Sum();
    AssertEqual(sum1, sum2, 0.00001);
  }
}

template <typename Real>
void UnitTestSparseMatrixFrobeniusNorm() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    SparseMatrix<Real> smat(row, col);
    smat.SetRandn(0.8);

    Matrix<Real> mat(row, col);
    mat.SetRandn();
    smat.CopyToMat(&mat);

    Real norm1 = smat.FrobeniusNorm();
    Real norm2 = mat.FrobeniusNorm();
    AssertEqual(norm1, norm2, 0.00001);
  }
}

template <typename Real>
void UnitTestSparseMatrixAddToMat() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    SparseMatrix<Real> smat(row, col);
    smat.SetRandn(0.8);

    Matrix<Real> mat(row, col);
    mat.SetRandn();
    smat.CopyToMat(&mat);

    Matrix<Real> other_mat1(row, col);
    other_mat1.SetRandn();
    Matrix<Real> other_mat2 = other_mat1;

    smat.AddToMat(0.7, &other_mat1);
    other_mat2.AddMat(0.7, mat);
    AssertEqual(other_mat1, other_mat2, 0.00001);
  }
}

template <typename Real>
void UnitTestSparseMatrixTraceMatSmat() {
  for (int32 i = 0; i < 10; i++) {
    MatrixIndexT row = 10 + Rand() % 40;
    MatrixIndexT col = 10 + Rand() % 50;

    Matrix<Real> mat1(row, col);
    Matrix<Real> mat2(col, row);
    Matrix<Real> mat3(row, col);
    mat1.SetRandn();
    mat2.SetRandn();
    mat3.SetRandn();

    SparseMatrix<Real> smat1(row, col);
    SparseMatrix<Real> smat2(col, row);
    smat1.SetRandn(0.8);
    smat2.SetRandn(0.8);

    smat1.CopyToMat(&mat1);
    smat2.CopyToMat(&mat2);

    Real trace1 = TraceMatMat(mat3, mat1, kTrans);
    Real trace2 = TraceMatSmat(mat3, smat1, kTrans);
    AssertEqual(trace1, trace2, 0.00001);

    trace1 = TraceMatMat(mat3, mat2, kNoTrans);
    trace2 = TraceMatSmat(mat3, smat2, kNoTrans);
    AssertEqual(trace1, trace2, 0.00001);
  }
}

template <typename Real>
void SparseMatrixUnitTest() {
  // SparseVector
  UnitTestSparseVectorSum<Real>();
  UnitTestSparseVectorAddToVec<Real>();
  UnitTestSparseVectorMax<Real>();
  UnitTestSparseVectorVecSvec<Real>();

  // SparseMatrix
  UnitTestSparseMatrixSum<Real>();
  UnitTestSparseMatrixFrobeniusNorm<Real>();
  UnitTestSparseMatrixAddToMat<Real>();
  UnitTestSparseMatrixTraceMatSmat<Real>();
}

}  // namespace kaldi

int main() {
  kaldi::SetVerboseLevel(5);
  kaldi::SparseMatrixUnitTest<float>();
  kaldi::SparseMatrixUnitTest<double>();
  KALDI_LOG << "Tests succeeded.";
  return 0;
}

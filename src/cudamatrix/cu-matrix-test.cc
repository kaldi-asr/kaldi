// cudamatrix/cu-matrix-test.cc

// Copyright 2010  Karel Vesely
//           2013  Lucas Ondel
//           2013  Johns Hopkins University (author: Daniel Povey)
//           2013  Hainan Xu
//           2013  Xiaohui Zhang
//           2013  Johns Hopkins University (author: Guoguo Chen)

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
 * INITIALIZERS
 */
template<typename Real>
static void InitRand(VectorBase<Real> *v) {
  for (MatrixIndexT i = 0; i < v->Dim(); i++)
    (*v)(i) = RandGauss();
}



template<typename Real>
static void InitRand(MatrixBase<Real> *M) {
  do {
    for (MatrixIndexT i = 0;i < M->NumRows();i++)
      for (MatrixIndexT j = 0;j < M->NumCols();j++)
        (*M)(i, j) = RandGauss();
  } while (M->NumRows() != 0 && M->Cond() > 100);
}



template<typename Real>
static void RandZeroToOneMatrix(MatrixBase<Real>* mat) {
  for(int32 r=0; r<mat->NumRows(); r++)
    for(int32 c=0; c<mat->NumCols(); c++)
      (*mat)(r,c) = RandUniform();
}


/*
 * Unit tests
 */

template<typename Real>
static void UnitTestCuMatrixTraceMatMat() {
  for (int32 i = 0; i < 2; i++) {
    int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
    CuMatrix<Real> A(M, N);
    A.SetRandn();
    if (i % 2 == 1) {
      CuMatrix<Real> B(M, N);
      B.SetRandn();
      Real r1 = TraceMatMat(A, B, kTrans),
          r2 = TraceMatMat(Matrix<Real>(A), Matrix<Real>(B), kTrans),
          r3 = TraceMatMat(Matrix<Real>(A), Matrix<Real>(B, kTrans), kNoTrans);
      Matrix<Real> X(B, kTrans);
      KALDI_LOG << "Xsum = " << X.Sum();
      Matrix<Real> Y(B, kTrans);
      KALDI_LOG << "Ysum = " << Y.Sum();
      KALDI_LOG << "Bsum = " << B.Sum();
      KALDI_ASSERT(ApproxEqual(r1, r2));
      KALDI_ASSERT(ApproxEqual(r2, r3));
    } else {
      CuMatrix<Real> B(N, M);
      B.SetRandn();
      Real r1 = TraceMatMat(A, B, kNoTrans),
          r2 = TraceMatMat(Matrix<Real>(A), Matrix<Real>(B), kNoTrans),
          r3 = TraceMatMat(Matrix<Real>(A), Matrix<Real>(B, kTrans), kTrans);
      KALDI_ASSERT(ApproxEqual(r1, r2));
      KALDI_ASSERT(ApproxEqual(r2, r3));
    }
  }
}


template<typename Real>
static void UnitTestCuCholesky() {
  for (int32 i = 0; i < 2; i++) {
    int32 M = 1 + Rand() % 10, N = M + 5;

    CuMatrix<Real> A(M, N);
    A.SetRandn();
    CuMatrix<Real> S(M, M);
    // SymAddMat2 only copies lower triangle.
    // it's OK- Cholesky only reads the lower triangle.
    S.SymAddMat2(1.0, A, kNoTrans, 0.0);

    CuMatrix<Real> C(S);
    C.Cholesky();

    CuMatrix<Real> S2(M, M);
    S2.AddMatMat(1.0, C, kNoTrans, C, kTrans, 0.0);
    S.CopyLowerToUpper();
    KALDI_ASSERT(S.ApproxEqual(S2));
  }
}







/*
 * CuMatrix
 */
template<typename Real>
static void UnitTestCuMatrixApplyLog() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  Matrix<Real> H(M, N);
  H.SetRandn();
  H.MulElements(H); // make numbers positive

  CuMatrix<Real> D(H);

  D.ApplyLog();
  H.ApplyLog();

  Matrix<Real> H2(D);

  AssertEqual(H,H2);
}


/*
 * CuMatrix
 */
template<typename Real>
static void UnitTestCuMatrixApplyExp() {
  int32 M = 10 + Rand() % 20, N = 10 + Rand() % 20;
  Matrix<Real> H(M, N);
  H.SetRandn();
  H.MulElements(H); // make numbers positive

  CuMatrix<Real> D(H);

  D.ApplyExp();
  H.ApplyExp();

  Matrix<Real> H2(D);

  AssertEqual(H,H2);
}



template<typename Real>
static void UnitTestCuMatrixSigmoid() {
  for (int32 i = 0; i < 2; i++) {
    int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
    Matrix<Real> H(M, N);
    H.SetRandn();
    H.MulElements(H); // make numbers positive

    CuMatrix<Real> D(H);
    CuMatrix<Real> E(M, N);

    E.Sigmoid(D);
    H.Sigmoid(H);

    Matrix<Real> H2(E);

    AssertEqual(H, H2);
  }
}

template<typename Real>
static void UnitTestCuMatrixScale() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  Matrix<Real> H(M, N);
  H.SetRandn();

  BaseFloat scale = -1 + (0.33 * (Rand() % 5));
  CuMatrix<Real> D(H);
  D.Scale(scale);
  H.Scale(scale);
  Matrix<Real> E(D);

  AssertEqual(H, E);
}

template<typename Real>
static void UnitTestCuMatrixAdd() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  Matrix<Real> H(M, N);
  H.SetRandn();

  BaseFloat offset = -1 + (0.33 * (Rand() % 5));
  CuMatrix<Real> D(H);
  D.Add(offset);
  H.Add(offset);
  Matrix<Real> E(D);

  AssertEqual(H, E);
}


template<typename Real>
static void UnitTestCuMatrixSoftHinge() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  Matrix<Real> H(M, N);
  H.SetRandn();
  H.MulElements(H); // make numbers positive

  CuMatrix<Real> D(H);
  CuMatrix<Real> E(M, N);

  E.SoftHinge(D);
  H.SoftHinge(H);

  Matrix<Real> H2(E);

  AssertEqual(H,H2);
}

template<typename Real>
static void UnitTestCuMatrixGroupPnorm() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  Real power[] = { 1.4, 1.6, 0.1234, 2.123, 0, 1, 2,
      std::numeric_limits<Real>::infinity() };
  for (int32 K = 5; K < 7; K++) {
    for (int32 i = 0; i < 2 * sizeof(power) / sizeof(Real); ++i) {
      Real p = power[i / 2];
      int32 N_src = N * K;
      Matrix<Real> H_src(M, N_src);
      H_src.SetRandn();
      if (i % 2 == 0)
        H_src.ApplyFloor(0.0); // will put some zeros in the matrix.. harder to
                               // do derivatives.
      Matrix<Real> H(M, N);
      H.GroupPnorm(H_src, p);
      CuMatrix<Real> D(H_src);
      CuMatrix<Real> E(M, N);
      E.GroupPnorm(D, p);
      Matrix<Real> H2(E);
      AssertEqual(H, H2);
    }
  }
}

template<typename Real>
static void UnitTestCuMatrixGroupMax() {
  int32 M = 100 + Rand() % 200, N = 100 + Rand() % 200;
  // M = 256; N = 256;
  for (int32 K = 5; K < 7; K++) {
    int32 N_src = N * K;
    Matrix<Real> H_src(M, N_src);
    H_src.SetRandn();
    if (rand () % 2 == 0)
      H_src.ApplyFloor(0.0); // will put some zeros in the matrix.. harder to
                             // do derivatives.
    Matrix<Real> H(M, N);
    H.GroupMax(H_src);
    CuMatrix<Real> D(H_src);
    CuMatrix<Real> E(M, N);
    E.GroupMax(D);
    Matrix<Real> H2(E);
    AssertEqual(H,H2);
  }
}

template<typename Real>
static void UnitTestCuMatrixSet() {
  for (int32 i = 0; i < 2; i++) {
    BaseFloat value= 0.333;
    int32 dimM = 10 + Rand() % 600, dimN = 10 + Rand() % 400;
    CuMatrix<Real> m1(dimM, dimN);
    Matrix<Real> m2(dimM, dimN);
    m1.Set(value);
    m2.Set(value);
    Matrix<Real> m3(m1);
    AssertEqual(m2, m3);
  }
}


template<typename Real>
static void UnitTestCuMatrixApplyPow() {

  for (int32 i = 0; i < 2; i++) {
    BaseFloat pow = 0.5 * (Rand() % 6);

    Matrix<Real> H(10 + Rand() % 60, 10 + Rand() % 20);
    H.SetRandn();
    H.Row(0).Set(0.0);
    if (i == 2) { Matrix<Real> tmp(H, kTrans); H = tmp; }

    if (pow != 1.0 && pow != 2.0 && pow != 3.0)
      H.MulElements(H); //make numbers positive

    CuMatrix<Real> cH(H);

    cH.ApplyPow(pow);

    H.ApplyPow(pow);
    Matrix<Real> H2(cH);
    AssertEqual(H, H2);
  }
}

template<typename Real>
static void UnitTestCuMatrixApplyPowAbs() {

  for (int32 i = 0; i < 2; i++) {
    BaseFloat pow = 0.5 * (Rand() % 6);

    Matrix<Real> H(10 + Rand() % 60, 10 + Rand() % 20);
    H.SetRandn();
    H.Row(0).Set(0.0);
    if (i == 2) { Matrix<Real> tmp(H, kTrans); H = tmp; }

    CuMatrix<Real> cH(H);

    cH.ApplyPowAbs(pow, true);

    H.ApplyPowAbs(pow, true);
    Matrix<Real> H2(cH);
    AssertEqual(H, H2);
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyRowsFromVec() {
  for (int32 p = 0; p < 2; p++) {
    int32 num_rows = 100 + Rand() % 255, num_cols;
    if (p <= 2) num_cols = 128;
    else if (p <= 4) num_cols = 256;
    else num_cols = 100 + Rand() % 200;

    int32 vec_dim;
    if (p % 2 == 0) vec_dim = num_cols;
    else vec_dim = num_cols * num_rows;

    CuVector<Real> cu_vec(vec_dim);
    cu_vec.SetRandn();
    Vector<Real> vec(cu_vec);

    CuMatrix<Real> cu_mat(num_rows, num_cols);
    cu_mat.CopyRowsFromVec(cu_vec);
    Matrix<Real> mat(num_rows, num_cols);
    mat.CopyRowsFromVec(vec);

    Matrix<Real> mat2(cu_mat);
    AssertEqual(mat, mat2);
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyColsFromVec() {
  for (int32 p = 0; p < 2; p++) {
    int32 num_rows = 100 + Rand() % 255;
    int32 num_cols = 100 + Rand() % 200;

    int32 vec_dim;
    if (p % 2 == 0) vec_dim = num_rows;
    else vec_dim = num_cols * num_rows;

    CuVector<Real> cu_vec(vec_dim);
    cu_vec.SetRandn();
    Vector<Real> vec(cu_vec);

    CuMatrix<Real> cu_mat(num_rows, num_cols);
    cu_mat.CopyColsFromVec(cu_vec);
    Matrix<Real> mat(num_rows, num_cols);
    mat.CopyColsFromVec(vec);

    Matrix<Real> mat2(cu_mat);
    AssertEqual(mat, mat2);
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyRows() {
  for (int32 p = 0; p < 2; p++) {
    MatrixIndexT num_rows1 = 10 + Rand() % 10,
        num_rows2 = 10 + Rand() % 10,
        num_cols = 10 + Rand() % 10;
    CuMatrix<Real> M(num_rows1, num_cols);
    M.SetRandn();

    CuMatrix<Real> N1(num_rows2, num_cols),
        N2(num_rows2, num_cols), O(num_rows2, num_cols);
    std::vector<int32> reorder(num_rows2);
    std::vector<const Real*> reorder_src(num_rows2, NULL);
    for (int32 i = 0; i < num_rows2; i++) {
      reorder[i] = -1 + (Rand() % (num_rows1 + 1));
      if (reorder[i] != -1) {
        reorder_src[i] = M.RowData(reorder[i]);
      }
    }

    CuArray<int32> reorder_cuda(reorder);
    CuArray<const Real*> reorder_src_cuda(reorder_src);
    N1.CopyRows(M, reorder_cuda);
    N2.CopyRows(reorder_src_cuda);

    for (int32 i = 0; i < num_rows2; i++)
      for (int32 j = 0; j < num_cols; j++)
        if (reorder[i] < 0) O(i, j) = 0;
        else O(i, j) = M(reorder[i], j);

    AssertEqual(N1, O);
    AssertEqual(N2, O);
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyToRows() {
  for (int32 p = 0; p < 2; p++) {
    MatrixIndexT num_rows1 = 10 + Rand() % 10,
        num_rows2 = 10 + Rand() % 10,
        num_cols = 10 + Rand() % 10;
    CuMatrix<Real> M(num_rows1, num_cols);
    M.SetRandn();

    CuMatrix<Real> N(num_rows2, num_cols), O(num_rows2, num_cols);
    std::vector<Real*> reorder_dst(num_rows1, NULL);
    unordered_map<MatrixIndexT, bool> used_index;
    for (int32 i = 0; i < num_rows1; i++) {
      MatrixIndexT index = -1 + (Rand() % (num_rows2 + 1));
      if (used_index.find(index) == used_index.end()) {
        used_index[index] = true;
      } else {
        index = -1;
      }
      if (index != -1) {
        reorder_dst[i] = N.RowData(index);
        for (int32 j = 0; j < num_cols; j++)
          O(index, j) = M(i, j);
      }
    }

    CuArray<Real*> reorder_dst_cuda(reorder_dst);
    M.CopyToRows(reorder_dst_cuda);

    AssertEqual(N, O);
  }
}


template<typename Real>
static void UnitTestCuMatrixAddRows() {
  for (int32 p = 0; p < 2; p++) {
    MatrixIndexT num_rows1 = 10 + Rand() % 10,
        num_rows2 = 10 + Rand() % 10,
        num_cols = 10 + Rand() % 10;
    CuMatrix<Real> M(num_rows1, num_cols);
    M.SetRandn();

    CuMatrix<Real> N1(num_rows2, num_cols),
        N2(num_rows2, num_cols), O(num_rows2, num_cols);
    std::vector<int32> reorder(num_rows2);
    std::vector<const Real*> reorder_src(num_rows2, NULL);
    for (int32 i = 0; i < num_rows2; i++) {
      reorder[i] = -1 + (Rand() % (num_rows1 + 1));
      if (reorder[i] != -1)
        reorder_src[i] = M.RowData(reorder[i]);
    }

    Real alpha =
        static_cast<Real>((Rand() % num_rows2)) / static_cast<Real>(num_rows1);

    CuArray<int32> reorder_cuda(reorder);
    CuArray<const Real*> reorder_src_cuda(reorder_src);
    N1.AddRows(alpha, M, reorder_cuda);
    N2.AddRows(alpha, reorder_src_cuda);

    for (int32 i = 0; i < num_rows2; i++) {
      if (reorder[i] != -1) {
        for (int32 j = 0; j < num_cols; j++) {
          O(i, j) += alpha * M(reorder[i], j);
        }
      }
    }

    AssertEqual(N1, O);
    AssertEqual(N2, O);
  }
}


template<typename Real>
static void UnitTestCuMatrixAddToRows() {
  for (int32 p = 0; p < 2; p++) {
    MatrixIndexT num_rows1 = 10 + Rand() % 10,
        num_rows2 = 10 + Rand() % 10,
        num_cols = 10 + Rand() % 10;
    CuMatrix<Real> M(num_rows1, num_cols);
    M.SetRandn();

    Real alpha =
        static_cast<Real>((Rand() % num_rows2)) / static_cast<Real>(num_rows1);

    CuMatrix<Real> N(num_rows2, num_cols), O(num_rows2, num_cols);
    std::vector<Real*> reorder_dst(num_rows1, NULL);
    unordered_map<MatrixIndexT, bool> used_index;
    for (int32 i = 0; i < num_rows1; i++) {
      MatrixIndexT index = -1 + (Rand() % (num_rows2 + 1));
      if (used_index.find(index) == used_index.end()) {
        used_index[index] = true;
      } else {
        index = -1;
      }
      if (index != -1) {
        reorder_dst[i] = N.RowData(index);
        for (int32 j = 0; j < num_cols; j++)
          O(index, j) += alpha * M(i, j);
      }
    }

    CuArray<Real*> reorder_dst_cuda(reorder_dst);
    M.AddToRows(alpha, reorder_dst_cuda);

    AssertEqual(N, O);
  }
}


template<typename Real>
void UnitTestCuMatrixCopyCross() {
  for (int32 i = 0; i < 2; i++) {
    int32 M = 100 + Rand() % 255, N = 100 + Rand() % 255;
    if (Rand() % 3 == 0) { M = 0; N = 0; }
    CuMatrix<Real> mat1(M, N);
    mat1.SetRandn();
    if (i % 2 == 0) {
      CuMatrix<float> mat2(M, N);
      mat2.CopyFromMat(mat1);
      CuMatrix<Real> mat3(M, N);
      mat3.CopyFromMat(mat2);
      AssertEqual(mat1, mat3);
    } else {
      CuMatrix<float> mat2(N, M);
      mat2.CopyFromMat(mat1, kTrans);
      CuMatrix<Real> mat3(M, N);
      mat3.CopyFromMat(mat2, kTrans);
      AssertEqual(mat1, mat3);
    }
  }
}

template<typename Real> void UnitTestCuMatrixCopyCross2() {
  for (int32 i = 0; i < 2; i++) {
    int32 M = 100 + Rand() % 255, N = 100 + Rand() % 255;
    if (Rand() % 3 == 0) { M = 0; N = 0; }
    CuMatrix<Real> mat1(M, N);
    mat1.SetRandn();
    Matrix<float> mat2(M, N);
    mat2.CopyFromMat(mat1);
    CuMatrix<Real> mat3(M, N);
    mat3.CopyFromMat(mat2);
    AssertEqual(mat1, mat3);
  }
}

template<typename Real>
static void UnitTestCuMatrixSumColumnRanges() {
  for (int32 p = 0; p < 2; p++) {
    MatrixIndexT num_cols1 = 10 + Rand() % 10,
        num_cols2 = 10 + Rand() % 10,
        num_rows = 10 + Rand() % 10;
    Matrix<Real> src(num_rows, num_cols1);
    Matrix<Real> dst(num_rows, num_cols2);
    std::vector<Int32Pair> indices(num_cols2);
    for (int32 i = 0; i < num_cols2; i++) {
      indices[i].first = Rand() % num_cols1;
      int32 headroom = num_cols1 - indices[i].first,
        size = (Rand() % headroom) + 1;
      indices[i].second = indices[i].first + size;
      KALDI_ASSERT(indices[i].second >= indices[i].first &&
                   indices[i].second <= num_cols1 &&
                   indices[i].first >= 0);
      // In the test we allow second == first.
    }
    src.SetRandn();
    // Simple computation:
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      for (MatrixIndexT j = 0; j < num_cols2; j++) {
        int32 start = indices[j].first, end = indices[j].second;
        Real sum = 0.0;
        for (MatrixIndexT j2 = start; j2 < end; j2++)
          sum += src(i, j2);
        dst(i, j) = sum;
      }
    }
    CuMatrix<Real> cu_src(src);
    CuMatrix<Real> cu_dst(num_rows, num_cols2, kUndefined);
    CuArray<Int32Pair> indices_tmp(indices);
    cu_dst.SumColumnRanges(cu_src, indices_tmp);
    Matrix<Real> dst2(cu_dst);
    AssertEqual(dst, dst2);
  }
}


template<typename Real>
static void UnitTestCuMatrixAddRowRanges() {
  for (int32 p = 0; p < 10; p++) {
    MatrixIndexT num_rows1 = 10 + Rand() % 10,
        num_rows2 = 10 + Rand() % 10,
        num_cols = 10 + Rand() % 10;
    Matrix<Real> src(num_rows1, num_cols); src.SetRandn();
    Matrix<Real> dst(num_rows2, num_cols); dst.SetRandn();

    // Computes the indexes.
    std::vector<Int32Pair> indexes(num_rows2);
    for (MatrixIndexT i = 0; i < num_rows2; i++) {
      indexes[i].first = Rand() % num_rows1;
      int32 headroom = num_rows1 - indexes[i].first,
            size = (Rand() % headroom) + 1;
      indexes[i].second = indexes[i].first + size;
      KALDI_ASSERT(indexes[i].second >= indexes[i].first &&
                   indexes[i].second <= num_rows1 &&
                   indexes[i].first >= 0);
    }
    // Computes reference matrix.
    Matrix<Real> dst1(dst);
    for (MatrixIndexT i = 0; i < num_rows2; i++) {
      int32 start = indexes[i].first, end = indexes[i].second;
      for (MatrixIndexT j = 0; j < num_cols; j++) {
        for (MatrixIndexT i2 = start; i2 < end; i2++)
          dst1(i, j) += src(i2, j);
      }
    }

    CuMatrix<Real> cu_src(src);
    CuMatrix<Real> cu_dst(dst);
    CuArray<Int32Pair> cu_indexes(indexes);
    cu_dst.AddRowRanges(cu_src, cu_indexes);
    Matrix<Real> dst2(cu_dst);
    AssertEqual(dst1, dst2);
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyCols() {
  for (int32 p = 0; p < 2; p++) {
    MatrixIndexT num_cols1 = 10 + Rand() % 10,
        num_cols2 = 10 + Rand() % 10,
        num_rows = 10 + Rand() % 10;
    CuMatrix<Real> M(num_rows, num_cols1);
    M.SetRandn();

    CuMatrix<Real> N(num_rows, num_cols2), O(num_rows, num_cols2);
    std::vector<int32> reorder(num_cols2);
    for (int32 i = 0; i < num_cols2; i++)
      reorder[i] = -1 + (Rand() % (num_cols1 + 1));

    CuArray<int32> reorder_gpu(reorder);
    N.CopyCols(M, reorder_gpu);

    for (int32 i = 0; i < num_rows; i++)
      for (int32 j = 0; j < num_cols2; j++)
        if (reorder[j] < 0) O(i, j) = 0;
        else O(i, j) = M(i, reorder[j]);
    AssertEqual(N, O);
  }
}


template<typename Real>
static void UnitTestCuMatrixAddCols() {
  for (int32 p = 0; p < 2; p++) {
    MatrixIndexT num_cols1 = 10 + Rand() % 10,
        num_cols2 = 10 + Rand() % 10,
        num_rows = 10 + Rand() % 10;
    CuMatrix<Real> M(num_rows, num_cols1);
    M.SetRandn();

    CuMatrix<Real> N(num_rows, num_cols2), O(num_rows, num_cols2);
    std::vector<int32> reorder(num_cols2);
    for (int32 i = 0; i < num_cols2; i++)
      reorder[i] = -1 + (Rand() % (num_cols1 + 1));

    CuArray<int32> reorder_gpu(reorder);
    N.AddCols(M, reorder_gpu);

    for (int32 i = 0; i < num_rows; i++)
      for (int32 j = 0; j < num_cols2; j++)
        if (reorder[j] < 0) O(i, j) = 0;
        else O(i, j) = M(i, reorder[j]);
    AssertEqual(N, O);
  }
}


template<typename Real>
static void UnitTestCuMatrixApplyFloor() {

  for (int32 i = 0; i < 3; i++) {
    BaseFloat floor = 0.33 * (Rand() % 6);

    Matrix<Real> H(10 + Rand() % 600, 10 + Rand() % 20);
    H.SetRandn();
    if (i == 2) { Matrix<Real> tmp(H, kTrans); H = tmp; }

    CuMatrix<Real> cH(H);

    cH.ApplyFloor(floor);

    H.ApplyFloor(floor);
    Matrix<Real> H2(cH);

    AssertEqual(H, H2);
  }
}

template<typename Real>
static void UnitTestCuMatrixApplyCeiling() {

  for (int32 i = 0; i < 3; i++) {
    BaseFloat ceiling = 0.33 * (Rand() % 6);

    Matrix<Real> H(10 + Rand() % 600, 10 + Rand() % 20);
    H.SetRandn();
    if (i == 2) { Matrix<Real> tmp(H,kTrans); H = tmp; }

    CuMatrix<Real> cH(H);

    cH.ApplyCeiling(ceiling);

    H.ApplyCeiling(ceiling);
    Matrix<Real> H2(cH);

    AssertEqual(H, H2);
  }
}

template<typename Real>
static void UnitTestCuMatrixApplyHeaviside() {

  for (int32 i = 0; i < 1; i++) {
    Matrix<Real> H(10 + Rand() % 60, 10 + Rand() % 20);
    H.SetRandn();
    H.Row(0).Set(0.0);
    if (i == 2) { Matrix<Real> tmp(H, kTrans); H = tmp; }


    CuMatrix<Real> cH(H);

    cH.ApplyHeaviside();
    H.ApplyHeaviside();
    Matrix<Real> H2(cH);
    AssertEqual(H, H2);
  }
}


template<typename Real>
static void UnitTestCuMatrixHeaviside() {

  for (int32 i = 0; i < 1; i++) {
    Matrix<Real> H(10 + Rand() % 60, 10 + Rand() % 20);
    H.SetRandn();
    H.Row(0).Set(0.0);
    if (i == 2) { Matrix<Real> tmp(H, kTrans); H = tmp; }

    CuMatrix<Real> cH(H);
    CuMatrix<Real> cH2(H.NumRows(), H.NumCols(), kUndefined);
    cH2.Heaviside(cH);
    H.ApplyHeaviside();
    Matrix<Real> H2(cH2);
    AssertEqual(H, H2);
  }
}


template<typename Real>
static void UnitTestCuMatrixMulElements() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT dimM = 100 + Rand() % 256, dimN = 100 + Rand() % 256;

    Matrix<Real> Ha(dimM, dimN);
    Matrix<Real> Hb(dimM, dimN);
    Ha.SetRandn();
    Hb.SetRandn();

    CuMatrix<Real> Da(dimM, dimN);
    CuMatrix<Real> Db(dimM, dimN);
    Da.CopyFromMat(Ha);
    Db.CopyFromMat(Hb);

    Da.MulElements(Db);
    Ha.MulElements(Hb);

    Matrix<Real> Ha2(dimM, dimN);
    Da.CopyToMat(&Ha2);

    AssertEqual(Ha,Ha2);
  }
}

template<typename Real>
static void UnitTestCuMatrixDivElements() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT dimM = 100 + Rand() % 256, dimN = 100 + Rand() % 256;

    Matrix<Real> Ha(dimM, dimN);
    Matrix<Real> Hb(dimM, dimN);
    Ha.SetRandn();
    Hb.SetRandn();

    CuMatrix<Real> Da(dimM, dimN);
    CuMatrix<Real> Db(dimM, dimN);
    Da.CopyFromMat(Ha);
    Db.CopyFromMat(Hb);

    Da.DivElements(Db);
    Ha.DivElements(Hb);

    Matrix<Real> Ha2(dimM, dimN);
    Da.CopyToMat(&Ha2);

    AssertEqual(Ha,Ha2);
  }
}

template<typename Real>
static void UnitTestCuMatrixMax() {
  Matrix<Real> Ha(100,100);
  Matrix<Real> Hb(100,100);
  Ha.SetRandn();
  Hb.SetRandn();

  CuMatrix<Real> Da(100,100);
  CuMatrix<Real> Db(100,100);
  Da.CopyFromMat(Ha);
  Db.CopyFromMat(Hb);

  Da.Max(Db);
  Ha.Max(Hb);

  Matrix<Real> Ha2(100,100);
  Da.CopyToMat(&Ha2);

  AssertEqual(Ha,Ha2);
}

template<typename Real>
static void UnitTestCuMatrixMin() {
  Matrix<Real> Ha(100,100);
  Matrix<Real> Hb(100,100);
  Ha.SetRandn();
  Hb.SetRandn();

  CuMatrix<Real> Da(100,100);
  CuMatrix<Real> Db(100,100);
  Da.CopyFromMat(Ha);
  Db.CopyFromMat(Hb);

  Da.Min(Db);
  Ha.Min(Hb);

  Matrix<Real> Ha2(100,100);
  Da.CopyToMat(&Ha2);

  AssertEqual(Ha, Ha2);
}



template<typename Real>
static void UnitTestCuMatrixMulColsVec() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(99);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(100,99);
  CuVector<Real> Dv(99);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dm.MulColsVec(Dv);
  Hm.MulColsVec(Hv);

  Matrix<Real> Hm2(100,99);
  Dm.CopyToMat(&Hm2);

  AssertEqual(Hm,Hm2);
}



template<typename Real>
static void UnitTestCuMatrixMulRowsVec() {
  for (int32 i = 0; i < 2; i++) {
    int32 dimM = 100 + Rand() % 200, dimN = 100 + Rand() % 200;
   // int32 dimM = 256, dimN = 256;
    Matrix<Real> Hm(dimM, dimN);
    Vector<Real> Hv(dimM);
    Hm.SetRandn();
    InitRand(&Hv);

    CuMatrix<Real> Dm(dimM, dimN);
    CuVector<Real> Dv(dimM);
    Dm.CopyFromMat(Hm);
    Dv.CopyFromVec(Hv);

    Dm.MulRowsVec(Dv);
    Hm.MulRowsVec(Hv);

    Matrix<Real> Hm2(dimM, dimN);
    Dm.CopyToMat(&Hm2);

    AssertEqual(Hm,Hm2);
  }
}

template<typename Real>
static void UnitTestCuMatrixMulRowsGroupMat() {
  for (int32 i = 0; i < 2; i++) {
    int32 dimM = 100 + Rand() % 200, dimNs = 100 + Rand() % 200;
    int32 group_size = 1 + Rand() % 10;
    //int32 group_size = 1;
    int32 dimN = group_size * dimNs;
    Matrix<Real> Hm(dimM, dimN);
    Matrix<Real> Hs(dimM, dimNs);
    Hm.SetRandn();
    Hs.SetRandn();

    CuMatrix<Real> Dm(dimM, dimN);
    CuMatrix<Real> Ds(dimM, dimNs);
    Dm.CopyFromMat(Hm);
    Ds.CopyFromMat(Hs);

    Dm.MulRowsGroupMat(Ds);
    Hm.MulRowsGroupMat(Hs);

    Matrix<Real> Hm2(dimM, dimN);
    Dm.CopyToMat(&Hm2);
    AssertEqual(Hm,Hm2);
  }
}

template<typename Real>
static void UnitTestCuMatrixDiffGroupPnorm() {
  Real p[] = { 1.234, 2.345, 1, 2, std::numeric_limits<Real>::infinity() };
  for (int i = 0; i < 2 * sizeof(p) / sizeof(Real); i++) {
    int32 dimM = 100 + Rand() % 200, dimNs = 100 + Rand() % 200;
    int32 group_size = 1 + Rand() % 10;
    BaseFloat power = p[i / 2];
    int32 dimN = group_size * dimNs;
    Matrix<Real> Hiv(dimM, dimN);
    Matrix<Real> Hov(dimM, dimNs);
    Matrix<Real> Hid(dimM, dimN);
    Matrix<Real> Hod(dimM, dimNs);
    Hiv.SetRandn();
    Hod.SetRandn();
    if (i % 2 == 0)
      Hiv.ApplyFloor(0.0); // will put some zeros in the matrix.. harder to
                           // do derivatives.
    Hov.GroupPnorm(Hiv, power);
    CuMatrix<Real> Div(dimM, dimN);
    CuMatrix<Real> Dov(dimM, dimNs);
    CuMatrix<Real> Did(dimM, dimN);
    CuMatrix<Real> Dod(dimM, dimNs);
    Div.CopyFromMat(Hiv);
    Dod.CopyFromMat(Hod);
    Dov.CopyFromMat(Hov);

    // GPU
    Did.DiffGroupPnorm(Div, Dov, Dod, power);

    // CPU
    Hid.GroupPnormDeriv(Hiv, Hov, power);
    Hid.MulRowsGroupMat(Hod);

    Matrix<Real> Hid2(dimM, dimN);
    Did.CopyToMat(&Hid2);
    AssertEqual(Hid, Hid2);
  }
}


template<typename Real>
static void UnitTestCuMatrixGroupMaxDeriv() {
  int32 dimM = 100 + Rand() % 200, dimNs = 100 + Rand() % 200;
  int32 group_size = 1 + Rand() % 10;
  // int32 dimM = 256, dimNs = 2;
  // int32 group_size = 2;
  int32 dimN = group_size * dimNs;
  Matrix<Real> Hm(dimM, dimN);
  Matrix<Real> Hr(dimM, dimN);
  Matrix<Real> Hs(dimM, dimNs);
  Hs.SetRandn();
  if (rand () % 2 == 0)
    Hm.ApplyFloor(0.0); // will put some zeros in the matrix.. harder to
                        // do derivatives.
  Hs.GroupMax(Hm);

  CuMatrix<Real> Dm(dimM, dimN);
  CuMatrix<Real> Dr(dimM, dimN);
  CuMatrix<Real> Ds(dimM, dimNs);
  Dm.CopyFromMat(Hm);
  Dr.CopyFromMat(Hr);
  Ds.CopyFromMat(Hs);

  // KALDI_LOG << "Hr " << Hr << " Dr " << Dr << "Ds" << Ds << " Hs " << Hs ;
  Dr.GroupMaxDeriv(Dm, Ds);
  Hr.GroupMaxDeriv(Hm, Hs);

  // KALDI_LOG << "Hr " << Hr << " Dr " << Dr << "Ds" << Ds << " Hs " << Hs ;
  Matrix<Real> Hr2(dimM, dimN);
  Dr.CopyToMat(&Hr2);
  AssertEqual(Hr,Hr2);
}

template<typename Real> static void UnitTestCuMatrixAddDiagVecMat() {
  for (int p = 0; p < 4; p++) {
    MatrixIndexT dimM = 100 + Rand() % 255, dimN = 100 + Rand() % 255;
    //MatrixIndexT dimM = 10 + Rand() % 2, dimN = 10 + Rand() % 2;
    Real alpha = 0.43243, beta = 1.423;
    CuMatrix<Real> M(dimM, dimN), N(dimM, dimN);
    M.SetRandn();
    N.SetRandn();
    MatrixTransposeType trans = (p % 2 == 0 ? kNoTrans : kTrans);
    if (trans == kTrans)
      N.Transpose();

    KALDI_ASSERT(M.Sum() != 0.0);
    KALDI_ASSERT(N.Sum() != 0.0);

    CuVector<Real> V(dimM);
    V.SetRandn();

    KALDI_ASSERT(V.Sum() != 0.0);

    CuMatrix<Real> Mcheck(M);

    for (int32 r = 0; r < dimM; r++) {
      CuSubVector<Real> Mcheckrow(Mcheck, r);
      CuVector<Real> Nrow(dimN);
      if (trans == kTrans) Nrow.CopyColFromMat(N, r);
      else Nrow.CopyFromVec(N.Row(r));
      Mcheckrow.Scale(beta);
      Mcheckrow.AddVec(alpha * V(r), Nrow);
    }

    M.AddDiagVecMat(alpha, V, N, trans, beta);
    AssertEqual(M, Mcheck);
    KALDI_ASSERT(M.Sum() != 0.0);
  }
}

template<typename Real> static void UnitTestCuMatrixAddMatDiagVec() {
  // M <- alpha * N[^T] * diag(v) + beta * M
  for (int p = 0; p < 2; p++) {
    MatrixIndexT dimM = 100 + Rand() % 255, dimN = 100 + Rand() % 255;
    Real alpha = 0.43243, beta = 1.423;

    CuMatrix<Real> M(dimM, dimN), N(dimM, dimN), buf(dimM, dimN);
    M.SetRandn();
    N.SetRandn();
    buf.CopyFromMat(N);
    MatrixTransposeType trans = (p % 2 == 0 ? kNoTrans : kTrans);
    if (trans == kTrans)
      N.Transpose();

    CuVector<Real> V(dimN);
    V.SetRandn();

    CuMatrix<Real> Mcheck(M);
    Mcheck.Scale(beta);
    buf.MulColsVec(V);
    Mcheck.AddMat(alpha, buf, kNoTrans);

    M.AddMatDiagVec(alpha, N, trans, V, beta);
    AssertEqual(M, Mcheck);
    KALDI_ASSERT(M.Sum() != 0.0);
  }
}

template<typename Real> static void UnitTestCuMatrixAddMatMatElements() {
  // M <- alpha *(A .* B) + beta * M
  MatrixIndexT dimM = 100 + Rand() % 255, dimN = 100 + Rand() % 255;
  Real alpha = 0.43243, beta = 1.423;
  CuMatrix<Real> M(dimM, dimN), A(dimM, dimN), B(dimM, dimN), buf(dimM, dimN);
  M.SetRandn();
  A.SetRandn();
  B.SetRandn();

  CuMatrix<Real> Mcheck(M);
  buf.CopyFromMat(A); buf.MulElements(B);
  Mcheck.Scale(beta); Mcheck.AddMat(alpha, buf, kNoTrans);

  M.AddMatMatElements(alpha, A, B, beta);
  AssertEqual(M, Mcheck);
  KALDI_ASSERT(M.Sum() != 0.0);
}

template<typename Real> static void UnitTestCuMatrixSetMatMatDivMat() {
  // M = a * b / c (by element; when c = 0, M = a)
  MatrixIndexT dimM = 100 + Rand() % 255, dimN = 100 + Rand() % 255;
  CuMatrix<Real> M(dimM, dimN), A(dimM, dimN), B(dimM, dimN), C(dimM, dimN);
  CuMatrix<Real> ref(dimM, dimN);
  M.SetRandn();
  A.SetRandn();
  B.SetRandn();
  C.SetRandn();

  M.SetMatMatDivMat(A,B,C);
  ref.AddMatMatElements(1.0, A, B, 0.0);
  ref.DivElements(C);
  AssertEqual(M, ref);

  C.SetZero();
  M.SetMatMatDivMat(A,B,C);
  AssertEqual(M, A);
}

template<typename Real>
static void UnitTestCuMatrixDivRowsVec() {
  MatrixIndexT dimM = 1000, dimN = 5;
  Matrix<Real> Hm(dimM, dimN);
  Vector<Real> Hv(dimM);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(dimM, dimN);
  CuVector<Real> Dv(dimM);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dm.DivRowsVec(Dv);
  Hv.InvertElements();
  Hm.MulRowsVec(Hv);

  Matrix<Real> Hm2(dimM, dimN);
  Dm.CopyToMat(&Hm2);

  AssertEqual(Hm, Hm2);
}



template<typename Real>
static void UnitTestCuMatrixAddMat() {
  Matrix<Real> Ha(100,100);
  Matrix<Real> Hb(100,100);
  Ha.SetRandn();
  Hb.SetRandn();

  CuMatrix<Real> Da(100,100);
  CuMatrix<Real> Db(100,100);
  Da.CopyFromMat(Ha);
  Db.CopyFromMat(Hb);

  Da.AddMat(0.5,Db);
  Ha.AddMat(0.5,Hb);

  Matrix<Real> Ha2(100,100);
  Da.CopyToMat(&Ha2);

  AssertEqual(Ha,Ha2);

  //check use with submatrix
  CuMatrix<Real> mat1(10,10,kSetZero);
  mat1.AddMat(1.0,Da.Range(5,10,12,10)); //different stride for mat1,mat2
  CuMatrix<Real> mat2(Da.Range(5,10,12,10));
  AssertEqual(mat1,mat2);

  for (int i = 0; i < 10; i++) {
    int32 N = 5 * (10 + Rand() % 10),  M = 100 + Rand() % 50;
    Matrix<Real> Hc(N,M);
    Matrix<Real> Hd(M,N);
    Hc.SetRandn();
    Hd.SetRandn();

    CuMatrix<Real> Dc(N,M);
    CuMatrix<Real> Dd(M,N);
    Dc.CopyFromMat(Hc);
    Dd.CopyFromMat(Hd);

    Real alpha = 0.5;
    Dc.AddMat(alpha,Dd,kTrans);
    Hc.AddMat(alpha,Hd,kTrans);

    Matrix<Real> Hc2(N,M);
    Dc.CopyToMat(&Hc2);
    AssertEqual(Hc,Hc2);

    // check use with submatrix
    CuMatrix<Real> mat3(N/5,M,kSetZero);
    mat3.AddMat(1.0, Dd.Range(0,M,0,N/5),kTrans);

    CuMatrix<Real> mat4(Dd.Range(0,M,0,N/5),kTrans);
    AssertEqual(mat3,mat4);
  }
}


// this tests the branch of AddMatBlocks() that is taken when
// 'this' has a smaller dimension than 'src' (it sums).
template<typename Real>
static void UnitTestCuMatrixAddMatBlocks1() {
  for (int32 l = 0; l < 5; l++) {
    int32 num_row_blocks = RandInt(1, 10), num_col_blocks = RandInt(1, 20);
    int32 block_rows = RandInt(1, 100), block_cols = RandInt(1, 100);
    BaseFloat alpha = RandInt(3, 10);
    CuMatrix<Real> dst(block_rows, block_cols);
    dst.SetRandn();
    CuMatrix<Real> src(num_row_blocks * block_rows,
                       num_col_blocks * block_cols);
    src.SetRandn();

    CuMatrix<Real> dst_copy(dst);
    for (int32 rb = 0; rb < num_row_blocks; rb++) {
      for (int32 cb = 0; cb < num_col_blocks; cb++) {
        CuSubMatrix<Real> src_part(src,
                                   rb * block_rows, block_rows,
                                   cb * block_cols, block_cols);
        dst_copy.AddMat(alpha, src_part);
      }
    }
    dst.AddMatBlocks(alpha, src);
    AssertEqual(dst, dst_copy);
  }
}

// this is as UnitTestCuMatrixAddMatBlocks1, but tests with transpose.
template<typename Real>
static void UnitTestCuMatrixAddMatBlocks1Trans() {
  for (int32 l = 0; l < 5; l++) {
    int32 num_row_blocks = RandInt(1, 10), num_col_blocks = RandInt(1, 20);
    int32 block_rows = RandInt(1, 100), block_cols = RandInt(1, 100);
    BaseFloat alpha = RandInt(3, 10);
    CuMatrix<Real> dst(block_cols, block_rows);
    dst.SetRandn();
    CuMatrix<Real> src(num_row_blocks * block_rows,
                       num_col_blocks * block_cols);
    src.SetRandn();

    CuMatrix<Real> dst_copy(dst);
    for (int32 rb = 0; rb < num_row_blocks; rb++) {
      for (int32 cb = 0; cb < num_col_blocks; cb++) {
        CuSubMatrix<Real> src_part(src,
                                   rb * block_rows, block_rows,
                                   cb * block_cols, block_cols);
        dst_copy.AddMat(alpha, src_part, kTrans);
      }
    }
    dst.AddMatBlocks(alpha, src, kTrans);
    AssertEqual(dst, dst_copy);
  }
}


// this tests the branch of AddMatBlocks() that is taken when
// 'this' has a larger dimension than 'src'.  In this case, it does
// a broadcasting rather than a summing operation.
template<typename Real>
static void UnitTestCuMatrixAddMatBlocks2() {
  for (int32 l = 0; l < 5; l++) {
    int32 num_row_blocks = RandInt(1, 10), num_col_blocks = RandInt(1, 20);
    int32 block_rows = RandInt(1, 100), block_cols = RandInt(1, 100);
    BaseFloat alpha = RandInt(3, 10);
    CuMatrix<Real> src(block_rows, block_cols);
    src.SetRandn();
    CuMatrix<Real> dst(num_row_blocks * block_rows,
                       num_col_blocks * block_cols);
    src.SetRandn();

    CuMatrix<Real> dst_copy(dst);
    for (int32 rb = 0; rb < num_row_blocks; rb++) {
      for (int32 cb = 0; cb < num_col_blocks; cb++) {
        CuSubMatrix<Real> dst_copy_part(dst_copy,
                                        rb * block_rows, block_rows,
                                        cb * block_cols, block_cols);
        dst_copy_part.AddMat(alpha, src);
      }
    }
    dst.AddMatBlocks(alpha, src);
    AssertEqual(dst, dst_copy);
  }
}




template<typename Real>
static void UnitTestCuMatrixReduceSum() {
  int32 M = 100 + Rand() % 300, N = 100 + Rand() % 300;
  CuMatrix<Real> A(M, N);
  A.SetRandn();
  Matrix<Real> mA(A);
  KALDI_ASSERT(ApproxEqual(mA.Sum(), A.Sum()));
}

template<typename Real>
static void UnitTestCuMatrixReduceMax() {
  int32 M = 100 + Rand() % 300, N = 100 + Rand() % 300;
  CuMatrix<Real> A(M, N);
  A.SetRandn();
  Matrix<Real> mA(A);
  KALDI_ASSERT(ApproxEqual(mA.Max(), A.Max()));
}

template<typename Real>
static void UnitTestCuMatrixReduceMin() {
  int32 M = 100 + Rand() % 300, N = 100 + Rand() % 300;
  CuMatrix<Real> A(M, N);
  A.SetRandn();
  Matrix<Real> mA(A);
  KALDI_ASSERT(ApproxEqual(mA.Min(), A.Min()));
}

template<typename Real>
static void UnitTestCuMatrixAddVecToCols() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(100);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(100,99);
  CuVector<Real> Dv(100);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dm.AddVecToCols(0.5,Dv);
  Hm.AddVecToCols(0.5,Hv);

  Matrix<Real> Hm2(100,99);
  Dm.CopyToMat(&Hm2);

  AssertEqual(Hm,Hm2);
}



template<typename Real>
static void UnitTestCuMatrixAddVecToRows() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(99);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(100,99);
  CuVector<Real> Dv(99);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dm.AddVecToRows(0.5,Dv);
  Hm.AddVecToRows(0.5,Hv);

  Matrix<Real> Hm2(100,99);
  Dm.CopyToMat(&Hm2);

  AssertEqual(Hm,Hm2);
}


template<typename Real>
static void UnitTestCuMatrixSymAddMat2() {
  for (int32 i = 0; i < 2; i++) {
    int32 dimM = 10 + Rand() % 200, dimN = 10 + Rand() % 30;
    if (i == 8) {
      dimM = 0;
      dimN = 0;
    }
    CuMatrix<Real> M(dimM, dimM); // square matrix..
    CuMatrix<Real> N(dimM, dimN);
    M.SetRandn();
    N.SetRandn();
    MatrixTransposeType trans = (i % 2 == 0 ? kTrans : kNoTrans),
        other_trans = (trans == kTrans ? kNoTrans : kTrans);
    if (trans == kTrans) N.Transpose();
    CuMatrix<Real> M2(M);
    Real alpha = 0.3, beta = 1.75432;
    M.SymAddMat2(alpha, N, trans, beta);

    M2.AddMatMat(alpha, N, trans, N, other_trans, beta);

    CuTpMatrix<Real> T1(M), T2(M2);
    CuMatrix<Real> X1(T1), X2(T2); // so we can test equality.
    AssertEqual(X1, X2);
    KALDI_ASSERT(dimM == 0 || X1.Trace() != 0);
  }
}



template<typename Real>
static void UnitTestCuMatrixSymInvertPosDef() {
  for (int32 i = 0; i < 2; i++) {
    int32 dimM = 10 + Rand() % 200, dimN = dimM + 20;
    // dimN > dimM, so will be PSD almost surely.
    if (i == 8) {
      dimM = 0;
      dimN = 0;
    }
    if (i == 0) {
      dimM = 2;
      dimN = 5;
    }
    if (i == 1) {
      dimM = 9;
      dimN = 20;
    }
    CuMatrix<Real> M(dimM, dimM); // square matrix..
    CuMatrix<Real> N(dimM, dimN);
    N.SetRandn();
    MatrixTransposeType trans = (i % 2 == 0 ? kTrans : kNoTrans);
    // MatrixTranposeType other_trans = (trans == kTrans ? kNoTrans : kTrans);

    if (trans == kTrans) N.Transpose();
    CuMatrix<Real> M2(M);
    Real alpha = 0.3, beta = 1.75432;
    M.SymAddMat2(alpha, N, trans, beta);
    // M.AddMatMat(alpha, N, trans, N, other_trans, beta);
    CuSpMatrix<Real> spTemp(M, kTakeLower);
    SpMatrix<Real> S(spTemp);
    S.Invert();
    CuSpMatrix<Real> spTemp2(M, kTakeLower);
    CuMatrix<Real> M_orig(spTemp2);
    M.SymInvertPosDef();
    CuSpMatrix<Real> spTemp3(M, kTakeLower);
    CuMatrix<Real> M_inverted(spTemp3);
    CuMatrix<Real> M_prod(dimM, dimM);
    M_prod.AddMatMat(Real(1.0), M_orig, kNoTrans, M_inverted, kNoTrans, Real(0.0));
    KALDI_ASSERT(M_prod.IsUnit());
    CuSpMatrix<Real> spTemp4(M, kTakeLower);
    SpMatrix<Real> S2(spTemp4);
    KALDI_ASSERT(ApproxEqual(S, S2, (Real)0.1));
    KALDI_ASSERT(dimM == 0 || S.Trace() != 0);
  }
}


template<typename Real>
static void UnitTestCuMatrixAddMatMat() {
  Matrix<Real> Ha(200,100);
  Matrix<Real> Hb(100,200);
  Matrix<Real> Hc1(200,200);
  Matrix<Real> Hc2(100,100);
  Ha.SetRandn();
  Hb.SetRandn();

  CuMatrix<Real> Da(200,100);
  CuMatrix<Real> Db(100,200);
  Da.CopyFromMat(Ha);
  Db.CopyFromMat(Hb);
  CuMatrix<Real> Dc1(200,200);
  CuMatrix<Real> Dc2(100,100);

  Dc1.AddMatMat(0.5f,Da,kNoTrans,Db,kNoTrans,0.0f);
  Dc2.AddMatMat(0.5f,Da,kTrans,Db,kTrans,0.0f);
  Hc1.AddMatMat(0.5f,Ha,kNoTrans,Hb,kNoTrans,0.0f);
  Hc2.AddMatMat(0.5f,Ha,kTrans,Hb,kTrans,0.0f);

  Matrix<Real> Hc1a(200,200);
  Matrix<Real> Hc2a(100,100);
  Dc1.CopyToMat(&Hc1a);
  Dc2.CopyToMat(&Hc2a);

  AssertEqual(Hc1,Hc1a);
  AssertEqual(Hc2,Hc2a);
}


template<typename Real>
static void UnitTestCuMatrixAddVecVec() {
  Vector<Real> x(100);
  Vector<Real> y(200);
  x.SetRandn();
  y.SetRandn();

  CuVector<Real> Cux(100);
  CuVector<Real> Cuy(200);
  Cux.CopyFromVec(x);
  Cuy.CopyFromVec(y);

  Matrix<Real> A(100,200);
  CuMatrix<Real> CuA(100,200);

  A.AddVecVec(0.5f, x, y);
  CuA.AddVecVec(0.5f, Cux, Cuy);
  Matrix<Real> A2(100, 200);
  CuA.CopyToMat(&A2);

  AssertEqual(A,A2);
}


template<typename Real>
static void UnitTestCuMatrixAddMatMatBatched() {
  // Random stride is disabled as AddMatMatBatched requires consistent stride
#if HAVE_CUDA == 1
  bool old_mode = CuDevice::Instantiate().SetDebugStrideMode(false);
#endif
  const int32 batchCount = 10;
  std::vector<Matrix<Real>* > Ha(batchCount), Hb(batchCount), Hc1(batchCount), Hc2(batchCount);
  std::vector<CuMatrix<Real>* > Da(batchCount), Db(batchCount), Dc1(batchCount), Dc2(batchCount);
  std::vector<SubMatrix<Real>* > HA, HB, HC1, HC2;
  std::vector<CuSubMatrix<Real>* > DA, DB, DC1, DC2;

  for (int32 i = 0; i < batchCount; i++) {
    // first create a Matrix intance and then creat a SubMatrix instance from that
    Ha[i] = new Matrix<Real>(200, 100);
    Hb[i] = new Matrix<Real>(100, 200);
    Hc1[i] = new Matrix<Real>(200, 200);
    Hc2[i] = new Matrix<Real>(100, 100);
    Ha[i]->SetRandn();
    Hb[i]->SetRandn();
    HA.push_back(new SubMatrix<Real>(*(Ha[i]), 0, Ha[i]->NumRows(), 0,
                                     Ha[i]->NumCols()));
    HB.push_back(new SubMatrix<Real>(*(Hb[i]), 0, Hb[i]->NumRows(), 0,
                                     Hb[i]->NumCols()));
    HC1.push_back(new SubMatrix<Real>(*(Hc1[i]), 0, Hc1[i]->NumRows(), 0,
                                      Hc1[i]->NumCols()));
    HC2.push_back(new SubMatrix<Real>(*(Hc2[i]), 0, Hc2[i]->NumRows(), 0,
                                      Hc2[i]->NumCols()));

    // first create a CuMatrix intance and then creat a CuSubMatrix instance from that
    Da[i] = new CuMatrix<Real>(200, 100);
    Db[i] = new CuMatrix<Real>(100, 200);
    Dc1[i] = new CuMatrix<Real>(200, 200);
    Dc2[i] = new CuMatrix<Real>(100, 100);
    Da[i]->CopyFromMat(*(Ha[i]));
    Db[i]->CopyFromMat(*(Hb[i]));
    DA.push_back(new CuSubMatrix<Real>(*(Da[i]), 0, Da[i]->NumRows(), 0,
                                       Da[i]->NumCols()));
    DB.push_back(new CuSubMatrix<Real>(*(Db[i]), 0, Db[i]->NumRows(), 0,
                                       Db[i]->NumCols()));
    DC1.push_back(new CuSubMatrix<Real>(*(Dc1[i]), 0, Dc1[i]->NumRows(), 0,
                                        Dc1[i]->NumCols()));
    DC2.push_back(new CuSubMatrix<Real>(*(Dc2[i]), 0, Dc2[i]->NumRows(), 0,
                                        Dc2[i]->NumCols()));
  }

  AddMatMatBatched(static_cast<Real>(0.5f), DC1, DA, kNoTrans, DB, kNoTrans,
                   static_cast<Real>(0.0f));
  AddMatMatBatched(static_cast<Real>(0.5f), DC2, DA, kTrans, DB, kTrans,
                   static_cast<Real>(0.0f));

  // used to store results from DC1 and DC2 for equality check
  Matrix<Real> Hca1(200,200);
  Matrix<Real> Hca2(100,100);

  // equality check
  for (int32 i = 0; i< batchCount; i++) {
    (*HC1[i]).AddMatMat(0.5f, *(HA[i]), kNoTrans, *(HB[i]), kNoTrans, 0.0f);
    (*HC2[i]).AddMatMat(0.5f, *(HA[i]), kTrans, *(HB[i]), kTrans, 0.0f);
    DC1[i]->CopyToMat(&Hca1);
    DC2[i]->CopyToMat(&Hca2);
    AssertEqual(*(HC1[i]), Hca1);
    AssertEqual(*(HC2[i]), Hca2);
    delete Ha[i]; delete Hb[i]; delete Hc1[i]; delete Hc2[i];
    delete HA[i]; delete HB[i]; delete HC1[i]; delete HC2[i];
    delete Da[i]; delete Db[i]; delete Dc1[i]; delete Dc2[i];
    delete DA[i]; delete DB[i]; delete DC1[i]; delete DC2[i];
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().SetDebugStrideMode(old_mode);
#endif
}


template<typename Real>
static void UnitTestCuMatrixAddToDiag() {
  for (int32 i = 0; i < 10; i++) {
    int32 dimM = 100 + Rand() % 200, dimN = 100 + Rand() % 200;
    Matrix<Real> M(dimM, dimN);
    CuMatrix<Real> Mc(M);
    Real alpha = 5.5;
    M.AddToDiag(alpha);
    Mc.AddToDiag(alpha);
    Matrix<Real> M2(Mc);
    AssertEqual(M, M2);
  }
}

template<typename Real>
static void UnitTestCuMatrixAdd2() {
  for (int32 i = 0; i < 10; i++) {
    int32 dimM = 100 + Rand() % 200, dimN = 100 + Rand() % 200;
    Matrix<Real> M(dimM, dimN);
    CuMatrix<Real> Mc(M);
    Real alpha = 5.5;
    M.Add(alpha);
    Mc.Add(alpha);
    Matrix<Real> M2(Mc);
    AssertEqual(M, M2);
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyFromMat() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + Rand() % 10;

    Matrix<Real> A(dim, dim);
    A.SetRandn();
    CuMatrix<Real> E(A);
    CuMatrix<Real> B(dim, dim);
    B.CopyFromMat(E);

    AssertEqual<Real>(B, E);
  }
}

template<typename Real>
static void UnitTestCuMatrixCopyFromTp() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + Rand() % 10;
    TpMatrix<Real> A(dim);
    A.SetRandn();
    CuTpMatrix<Real> E(A);
    Matrix<Real> B(dim, dim);
    CuMatrix<Real> C(dim, dim);
    B.CopyFromTp(A, kNoTrans);
    C.CopyFromTp(E, kNoTrans);
    CuMatrix<Real> D(B);
    AssertEqual<Real>(D, C);
  }
}

template<typename Real>
static void UnitTestCuMatrixAddMatTp() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + Rand() % 10;

    Matrix<Real> A(dim, dim);
    Matrix<Real> B(dim, dim);
    TpMatrix<Real> C(dim);
    A.SetRandn();
    B.SetRandn();
    C.SetRandn();
    CuMatrix<Real> D(A);
    CuMatrix<Real> E(B);
    CuTpMatrix<Real> F(C);

    A.AddMatTp(1.0, B, kNoTrans, C, kNoTrans, 1.0);
    D.AddMatTp(1.0, E, kNoTrans, F, kNoTrans, 1.0);

    CuMatrix<Real> G(A);
    AssertEqual<Real>(G, D);
  }
}


template<typename Real>
static void UnitTestCuMatrixTranspose() {
  for (int32 i = 1; i < 2; i++) {
    MatrixIndexT dimM = 5 * i + Rand() % 10,
        dimN = dimM;
    if (i % 2 == 0) dimN += 5;

    CuMatrix<Real> A(dimM, dimN);
    A.SetRandn();
    CuMatrix<Real> B(A, kTrans);

    Matrix<Real> hA(A);
    Matrix<Real> hB(B);
    hB.Transpose();
    AssertEqual(hA, hB);
  }
}

template<typename Real>
static void UnitTestCuMatrixAddTpMat() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + Rand() % 10;

    Matrix<Real> A(dim, dim);
    Matrix<Real> B(dim, dim);
    TpMatrix<Real> C(dim);
    A.SetRandn();
    B.SetRandn();
    C.SetRandn();
    CuMatrix<Real> D(A);
    CuMatrix<Real> E(B);
    CuTpMatrix<Real> F(C);

    A.AddTpMat(1.0, C, kNoTrans, B, kNoTrans, 1.0);
    D.AddTpMat(1.0, F, kNoTrans, E, kNoTrans, 1.0);

    CuMatrix<Real> G(A);
    AssertEqual<Real>(G, D);
  }
}

/*
 * CuVector unit tests
 */
template<typename Real>
static void UnitTestCuVectorAddVec() {
  Vector<Real> Hv(777);
  Vector<Real> Hw(777);
  InitRand(&Hv);
  InitRand(&Hw);

  CuVector<Real> Dv(777);
  CuVector<Real> Dw(777);
  Dv.CopyFromVec(Hv);
  Dw.CopyFromVec(Hw);

  Dv.AddVec(0.1,Dw,0.9);
  Hv.Scale(0.9);
  Hv.AddVec(0.1,Hw);

  Vector<Real> Hv2(777);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}



template<typename Real>
static void UnitTestCuVectorAddRowSumMat() {
 const int32 X=4321, Y=19;
  Real alpha=0.1, beta=0.7;

  Matrix<Real> Hm(X,Y);
  Vector<Real> Hv(Y);
  Vector<Real> Hv_accu(Y);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(X,Y);
  CuVector<Real> Dv(Y);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dv.AddRowSumMat(alpha,Dm,beta);

  Hv_accu.SetZero();
  Hv_accu.AddRowSumMat(1.0, Hm);
  Hv.Scale(beta);
  Hv.AddVec(alpha,Hv_accu);

  Vector<Real> Hv2(Y);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}



template<typename Real>
static void UnitTestCuVectorAddRowSumMatLarge() {
  Matrix<Real> Hm(1000,990);
  Vector<Real> Hv(990);
  Vector<Real> Hv_accu(990);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(1000,990);
  CuVector<Real> Dv(990);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dv.AddRowSumMat(0.5,Dm,0.7);

  Hv_accu.SetZero();
  Hv_accu.AddRowSumMat(1.0, Hm);
  Hv.Scale(0.7);
  Hv.AddVec(0.5,Hv_accu);

  Vector<Real> Hv2(990);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}



template<typename Real>
static void UnitTestCuVectorAddColSumMat() {
  const int32 X=19, Y=4321;
  Real alpha=0.5, beta=0.7;

  Matrix<Real> Hm(X,Y);
  Vector<Real> Hv(X);
  Vector<Real> Hv_accu(X);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(X,Y);
  CuVector<Real> Dv(X);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dv.AddColSumMat(alpha,Dm,beta);

  Hv_accu.SetZero();
  Hv_accu.AddColSumMat(1.0, Hm);
  Hv.Scale(beta);
  Hv.AddVec(alpha, Hv_accu);

  Vector<Real> Hv2(X);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}

template<typename Real>
static void UnitTestCuSubMatrix() {
  for (int32 iter = 0 ; iter < 10; iter++) {
    int32 M1 = 1 + rand () % 10, M2 = 1 + Rand() % 1, M3 = 1 + Rand() % 10, M = M1 + M2 + M3,
        N1 = 1 + rand () % 10, N2 = 1 + Rand() % 1, N3 = 1 + Rand() % 10, N = N1 + N2 + N3,
        m = Rand() % M2, n = Rand() % N2;
    CuMatrix<Real> mat(M, N);
    mat.SetRandn();
    CuSubMatrix<Real> submat1(mat, M1, M2,
                              N1, N2),
        submat2 = mat.Range(M1, M2, N1, N2);
    Real f1 = mat(M1 + m, N1 + n), f2 = submat1(m, n), f3 = submat2(m, n);
    KALDI_ASSERT(f1 == f2);
    KALDI_ASSERT(f2 == f3);
  }
}



template<typename Real>
static void UnitTestCuVectorAddColSumMatLarge() {
  Matrix<Real> Hm(1000,990);
  Vector<Real> Hv(1000);
  Vector<Real> Hv_accu(1000);
  Hm.SetRandn();
  InitRand(&Hv);

  CuMatrix<Real> Dm(1000,990);
  CuVector<Real> Dv(1000);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dv.AddColSumMat(0.5, Dm, 0.7);

  Hv_accu.SetZero();
  Hv_accu.AddColSumMat(1.0, Hm);
  Hv.Scale(0.7);
  Hv.AddVec(0.5,Hv_accu);

  Vector<Real> Hv2(1000);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}



template<typename Real>
static void UnitTestCuVectorInvertElements() {
  Vector<Real> Hv(777);
  InitRand(&Hv);

  CuVector<Real> Dv(777);
  Dv.CopyFromVec(Hv);

  Dv.InvertElements();
  Hv.InvertElements();

  Vector<Real> Hv2(777);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}

template<typename Real>
static void UnitTestCuMatrixInvertElements() {
  Matrix<Real> Hm(77, 77);
  InitRand(&Hm);

  CuMatrix<Real> Dm(77, 77);
  Dm.CopyFromMat(Hm);

  Dm.InvertElements();
  Hm.InvertElements();

  Matrix<Real> Hm2(77, 77);
  Dm.CopyToMat(&Hm2);

  AssertEqual(Hm,Hm2);
}


template<class Real>
static void UnitTestCuMatrixIO() {
  for (int32 i = 0; i < 10; i++) {
    int32 dimM = 100 + Rand() % 255, dimN = 10 + Rand() % 20;
    if (i % 2 == 0) std::swap(dimM, dimN);
    if (i % 5 == 0) { dimM = 0; dimN = 0; }
    CuMatrix<Real> mat(dimM, dimN);
    mat.SetRandn();
    std::ostringstream os;
    bool binary = (i % 4 < 2);
    mat.Write(os, binary);

    CuMatrix<Real> mat2;
    std::istringstream is(os.str());
    mat2.Read(is, binary);
    AssertEqual(mat, mat2);
  }
}


template<typename Real>
static void UnitTestCuVectorAddTpVec() {
  Vector<Real> Hv(300);
  InitRand(&Hv);
  CuVector<Real> Dv(300);
  Dv.CopyFromVec(Hv);
  Vector<Real> Hv1(300);
  InitRand(&Hv1);
  CuVector<Real> Dv1(300);
  Dv1.CopyFromVec(Hv1);

  TpMatrix<Real> Hm(300);
  Hm.SetRandn();
  CuTpMatrix<Real> Dm(Hm);

  //gpu
  Dv.AddTpVec(1.0,Dm,kNoTrans,Dv1,1.0);
  //cpu
  Hv.AddTpVec(1.0,Hm,kNoTrans,Hv1,1.0);

  Vector<Real> Hv2(300);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}

template<typename Real>
static void UnitTestCuApproxEqual() {
  Real tol = 0.1;
  for (int32 i = 0; i < 2; i++) {
    int32 M = 1 + Rand() % 10, N = 1 + Rand() % 10;
    CuMatrix<Real> A(M, N), B(M, N);
    A.SetRandn();
    B.SetRandn();
    Matrix<Real> diff(A), Bm(B);
    diff.AddMat(-1.0, Bm);
    Real norm = diff.FrobeniusNorm();
    KALDI_ASSERT((norm <= tol * A.FrobeniusNorm()) == (A.ApproxEqual(B, tol)));
    tol *= 2.0;
  }
}

template<typename Real>
static void UnitTestCuVectorMulTp() {
  Vector<Real> Hv(300);
  InitRand(&Hv);
  CuVector<Real> Dv(300);
  Dv.CopyFromVec(Hv);

  TpMatrix<Real> Hm(300);
  Hm.SetRandn();
  CuTpMatrix<Real> Dm(Hm);

  //gpu
  Dv.MulTp(Dm,kNoTrans);
  //cpu
  Hv.MulTp(Hm,kNoTrans);

  Vector<Real> Hv2(300);
  Dv.CopyToVec(&Hv2);

  AssertEqual(Hv,Hv2);
}

template<typename Real, typename OtherReal>
static void UnitTestCuCopy() {
  for (int32 i = 0; i < 10; i++) {
    int32 M = 1 + Rand() % 10, N = 1 + Rand() % 10;
    CuMatrix<Real> A(M, N);
    CuMatrix<OtherReal> B(A, kTrans);
    CuMatrix<Real> C(B, kTrans);
    CuMatrix<Real> D(N, M);
    D.CopyFromMat(C, kTrans);
    CuMatrix<OtherReal> E(N, M);
    E.CopyFromMat(D, kNoTrans);
    CuMatrix<Real> F(M, N);
    F.CopyFromMat(E, kTrans);

    Matrix<OtherReal> G(M, N);
    G.CopyFromMat(F, kNoTrans);
    CuMatrix<Real> H(N, M);
    H.CopyFromMat(G, kTrans);
    Matrix<OtherReal> I(M, N);
    I.CopyFromMat(H, kTrans);
    CuMatrix<Real> J(I, kTrans);
    Matrix<OtherReal> K(J, kTrans);
    CuMatrix<Real> L(K, kNoTrans);

    KALDI_ASSERT(A.ApproxEqual(L));
  }

}

template<typename Real>
static void UnitTestCuSigmoid() {
  Matrix<Real> Hi(100,111);
  Matrix<Real> Ho(100,111);
  Hi.SetRandn();

  CuMatrix<Real> Di(100,111);
  CuMatrix<Real> Do(100,111);
  Di.CopyFromMat(Hi);

  //gpu
  Do.Sigmoid(Di);
  //cpu
  for(MatrixIndexT r=0; r < Hi.NumRows(); r++) {
    for(MatrixIndexT c=0; c < Hi.NumCols(); c++) {
      Ho(r, c) = 1.0/(1.0+exp(-Hi(r, c)));
    }
  }

  Matrix<Real> Ho2(100,111);
  Do.CopyToMat(&Ho2);

  AssertEqual(Ho,Ho2);
}



template<typename Real>
static void UnitTestCuDiffSigmoid() {
  Matrix<Real> Hi(100,111);
  Matrix<Real> Ho(100,111);
  Matrix<Real> Hy(100,111);
  Hi.SetRandn();
  RandZeroToOneMatrix(&Hy);

  CuMatrix<Real> Di(100,111);
  CuMatrix<Real> Do(100,111);
  CuMatrix<Real> Dy(100,111);
  Di.CopyFromMat(Hi);
  Dy.CopyFromMat(Hy);

  //gpu
  Do.DiffSigmoid(Dy, Di);
  //cpu
  for(MatrixIndexT r=0; r<Ho.NumRows(); r++) {
    for(MatrixIndexT c=0; c<Ho.NumCols(); c++) {
      Ho(r, c) = Hy(r, c)*(1.0 - Hy(r, c)) * Hi(r, c);
    }
  }

  Matrix<Real> Ho2(100,111);
  Do.CopyToMat(&Ho2);

  AssertEqual(Ho,Ho2);
}


template<typename Real>
static void UnitTestCuDiffSoftmax() {
  int m = 100, n = 111;
  Matrix<Real> Hi(m, n);
  Matrix<Real> Ho(m, n);
  Matrix<Real> Hy(m, n);
  Hi.SetRandn();
  RandZeroToOneMatrix(&Hy);

  CuMatrix<Real> Di(m, n);
  CuMatrix<Real> Do(m, n);
  CuMatrix<Real> Dy(m, n);
  Di.CopyFromMat(Hi);
  Dy.CopyFromMat(Hy);

  //gpu
  Do.DiffSoftmaxPerRow(Dy, Di);
  //cpu
  {
    const MatrixBase<Real> &P(Hy), &E(Hi);
    MatrixBase<Real> &D(Ho);
    D.CopyFromMat(P);
    D.MulElements(E);
    // At this point, D = P .* E (in matlab notation)
    Vector<Real> pe_vec(D.NumRows()); // For each row i, the dot product (p_t . e_t).
    pe_vec.AddDiagMatMat(1.0, P, kNoTrans, E, kTrans, 0.0);
    D.AddDiagVecMat(-1.0, pe_vec, P, kNoTrans, 1.0); // does D -= diag(pe_vec) * P.
  }

  Matrix<Real> Ho2(m, n);
  Do.CopyToMat(&Ho2);

  AssertEqual(Ho, Ho2);
}


template<typename Real>
static void UnitTestCuDiffLogSoftmax() {
  int m = 100, n = 111;
  Matrix<Real> Hi(m, n);
  Matrix<Real> Ho(m, n);
  Matrix<Real> Hy(m, n);
  Hi.SetRandn();
  RandZeroToOneMatrix(&Hy);

  CuMatrix<Real> Di(m, n);
  CuMatrix<Real> Do(m, n);
  CuMatrix<Real> Dy(m, n);
  Di.CopyFromMat(Hi);
  Dy.CopyFromMat(Hy);

  //gpu
  Do.DiffLogSoftmaxPerRow(Dy, Di);
  //cpu
  {
    const MatrixBase<Real> &Y(Hy), &E(Hi);
    MatrixBase<Real> &D(Ho);
    D.CopyFromMat(Y);
    D.ApplyExp();                           // exp(y)
    Vector<Real> E_sum(D.NumRows());        // Initializes to zero
    E_sum.AddColSumMat(1.0, E);             // Sum(e)
    D.MulRowsVec(E_sum);                    // exp(y) Sum(e)
    D.Scale(-1.0);                          // - exp(y) Sum(e)
    D.AddMat(1.0, E, kNoTrans);             // e - exp(y_i) Sum(e)
  }

  Matrix<Real> Ho2(m, n);
  Do.CopyToMat(&Ho2);

  AssertEqual(Ho, Ho2);
}


template<typename Real>
static void UnitTestCuSoftmax() {

  for (int32 i = 0; i < 2; i++) {
    int row = 10 + Rand() % 40;
    int col = 10 + Rand() % 50;

    Matrix<Real> Hi(row,col);
    Matrix<Real> Ho(row,col);
    Hi.SetRandn();
    Hi.Scale(5.0);

    CuMatrix<Real> Di(row, col);
    CuMatrix<Real> Do(row, col);
    Di.CopyFromMat(Hi);

    //gpu
    Do.ApplySoftMaxPerRow(Di);
    //cpu
    Ho.CopyFromMat(Hi);
    for(MatrixIndexT r=0; r<Ho.NumRows(); r++) {
      Ho.Row(r).ApplySoftMax();
    }

    Matrix<Real> Ho2(Do);

    AssertEqual(Ho,Ho2,0.00001);
  }
}


template<typename Real>
static void UnitTestCuLogSoftmax() {

  for (int32 i = 0; i < 50; i++) {
    int row = 10 + Rand() % 300;
    int col = 10 + Rand() % 300;

    Matrix<Real> Hi(row, col);
    Matrix<Real> Ho(row, col);
    Hi.SetRandn();
    Hi.Scale(5.0);

    CuMatrix<Real> Di(row, col);
    CuMatrix<Real> Do(row, col);
    Di.CopyFromMat(Hi);

    //gpu
    Do.ApplyLogSoftMaxPerRow(Di);
    //cpu
    Ho.CopyFromMat(Hi);
    for(MatrixIndexT r=0; r<Ho.NumRows(); r++) {
      Ho.Row(r).ApplyLogSoftMax();
    }

    Matrix<Real> Ho2(Do);

    AssertEqual(Ho, Ho2, 0.00001);
  }
}


template<typename Real>
static void UnitTestCuFindRowMaxId() {
  for (int32 i = 0; i < 2; i++) {
    int32 dimM = 100 + Rand() % 200, dimN = 100 + Rand() % 200;
    Matrix<Real> Hi(dimM, dimN);
    Hi.SetRandn();

    CuMatrix<Real> Di(dimM, dimN);
    Di.CopyFromMat(Hi);

    std::vector<int32> Hmax(dimM);
    CuArray<int32> Dmax(dimN);

    // on gpu
    Di.FindRowMaxId(&Dmax);

    // on cpu
    for(MatrixIndexT r=0; r<Hi.NumRows(); r++) {
      Real max=-1.0e+20; int32 idx=-1;
      for(MatrixIndexT c=0; c<Hi.NumCols(); c++) {
        if(Hi(r,c) > max) { idx=c; max=Hi(r,c); }
      }
      Hmax[r] = idx;
    }

    std::vector<int32> Hmax2(dimM);
    Dmax.CopyToVec(&Hmax2);

    KALDI_ASSERT(Hmax == Hmax2);
  }
}



template<typename Real>
static void UnitTestCuDiffXent() {
  int32 X=100, Y=111;
  //nnet output / diff
  Matrix<Real> Hi(X,Y);
  RandZeroToOneMatrix(&Hi);
  CuMatrix<Real> Di(X,Y);
  Di.CopyFromMat(Hi);
  //target vector
  std::vector<int32> Htgt(X);
  for(int32 i=0; i<X; i++) {
    Htgt[i] = Rand()%Y;
  }
  CuArray<int32> Dtgt(X);
  Dtgt.CopyFromVec(Htgt);
  //logpost vector
  Vector<Real> Hlogpost(X);
  CuVector<Real> Dlogpost(X);

  //gpu
  Di.DiffXent(Dtgt, &Dlogpost);
  //cpu
  for(MatrixIndexT r=0; r<Hi.NumRows(); r++) {
    int32 col_tgt = Htgt[r];
    Hlogpost(r) = Log(Hi(r, col_tgt));
    Hi(r, col_tgt) -= 1.0;
  }

  Matrix<Real> Hi2(X,Y);
  Di.CopyToMat(&Hi2);
  Vector<Real> Hlogpost2(X);
  Dlogpost.CopyToVec(&Hlogpost2);

  AssertEqual(Hi,Hi2);
  AssertEqual(Hlogpost,Hlogpost2);
}

template<typename Real> void UnitTestCheck() {
  Matrix<Real> Hi(100,111);
  Hi.SetRandn();

  CuMatrix<Real> Di(100,111);
  Di.CopyFromMat(Hi);

  CuMatrix<Real> Dj(Di);
  KALDI_LOG << Dj.NumRows();


}

template<typename Real>
void UnitTestSwapCu2Cu() {
  Matrix<Real> Hi(100,111);
  Hi.SetRandn();
  CuMatrix<Real> Di(100,111);
  Di.CopyFromMat(Hi);

  Matrix<Real> Hi2(110,121);
  Hi2.SetRandn();
  CuMatrix<Real> Di2(110,121);
  Di2.CopyFromMat(Hi2);

  Di.Swap(&Di2);
  Matrix<Real> Hf(Di.NumRows(), Di.NumCols());
  Di.CopyToMat(&Hf);
  Matrix<Real> Hf2(Di2.NumRows(), Di2.NumCols());
  Di2.CopyToMat(&Hf2);
  AssertEqual(Hi,Hf2);
  AssertEqual(Hi2,Hf);
}

template<typename Real>
void UnitTestSwapCu2M() {
  Matrix<Real> Hi(100,111);
  Hi.SetRandn();
  CuMatrix<Real> Di(100,111);
  Di.CopyFromMat(Hi);

  Matrix<Real> Hi2(110,121);
  Hi2.SetRandn();
  Matrix<Real> Di2(110,121);
  Di2.CopyFromMat(Hi2);

  Di.Swap(&Hi2);
  Matrix<Real> Hf(Di.NumRows(), Di.NumCols());
  Di.CopyToMat(&Hf);
  AssertEqual(Di2,Hf);
  AssertEqual(Hi2,Hi);
}


template<typename Real>
void UnitTestCuTanh() {
  Matrix<Real> H(100,110);
  H.SetRandn();
  CuMatrix<Real> D(100,110);
  D.CopyFromMat(H);

  //gpu
  CuMatrix<Real> Di(100,110);
  Di.Tanh(D);
  Matrix<Real> Df(Di.NumRows(), Di.NumCols());
  Di.CopyToMat(&Df);

  //cpu
  Matrix<Real> Hf(H.NumRows(), H.NumCols());
  Hf.Tanh(H);
  AssertEqual(Df,Hf);
}

template<typename Real>
static void UnitTestCuDiffTanh() {
  Matrix<Real> Hi(100,111);
  Matrix<Real> Ho(100,111);
  Matrix<Real> Hy(100,111);
  Hi.SetRandn();
  RandZeroToOneMatrix(&Hy);

  CuMatrix<Real> Di(100,111);
  CuMatrix<Real> Do(100,111);
  CuMatrix<Real> Dy(100,111);
  Di.CopyFromMat(Hi);
  Dy.CopyFromMat(Hy);

  //gpu
  Do.DiffTanh(Dy, Di);
  //cpu
  for(MatrixIndexT r=0; r<Ho.NumRows(); r++) {
    for(MatrixIndexT c=0; c<Ho.NumCols(); c++) {
      Ho(r, c) = (1.0 - Hy(r, c)*Hy(r, c)) * Hi(r, c);
    }
  }

  Matrix<Real> Ho2(100,111);
  Do.CopyToMat(&Ho2);

  AssertEqual(Ho,Ho2);
}

// just need this for testing function below.  Compute n!!
static int32 DoubleFactorial(int32 i) {
  if (i <= 0) { return 1; } else { return i * DoubleFactorial(i - 2); }
}

template <typename Real>
static void UnitTestCuMatrixSetRandn() {

  { // First test consistency when called twice.
    int32 dimM = 100 + Rand() % 200, dimN = 100 + Rand() % 200;
    Matrix<Real> M(dimM, dimN), N(dimM, dimN);
    srand(104);
    M.SetRandn();
    srand(104);
    N.SetRandn();
    AssertEqual(M, N);
  }

  for (int32 i = 0; i < 5; i++) {
    MatrixIndexT rows = 100 + Rand() % 50, cols = 100 + Rand() % 50;
    CuMatrix<Real> M(rows, cols);
    M.SetRandn();

    for (int32 pow = 1; pow < 5; pow++) {
      // test moments 1 through 4 of
      // the distribution.
      CuMatrix<Real> Mpow(M);
      Mpow.ApplyPow(pow);
      Real observed_moment = Mpow.Sum() / (rows * cols);
      // see http://en.wikipedia.org/wiki/Normal_distribution#Moments,
      // note that mu = 0 and sigma = 1.
      Real expected_moment = (pow % 2 == 1 ? 0 : DoubleFactorial(pow - 1));
      Real k = 10.0; // This is just a constant we use to give us some wiggle
                     // room before rejecting the distribution... e.g. 20 sigma,
                     // quite approximately.
      Real allowed_deviation = k * pow / sqrt(static_cast<Real>(rows * cols));
      // give it a bit more wiggle room for higher powers.. this is quite
      // unscientific, it would be better to involve the absolute moments or
      // something like that, and use one of those statistical inequalities,
      // but it involves the gamma function and it's too much hassle to implement.
      Real lower_bound = expected_moment - allowed_deviation,
          upper_bound = expected_moment + allowed_deviation;
      KALDI_ASSERT(observed_moment >= lower_bound && observed_moment <= upper_bound);
    }
  }
}


template <typename Real>
static void UnitTestCuMatrixSetRandUniform() {
  for (int32 i = 0; i < 2; i++) {
    MatrixIndexT rows = 180 + Rand() % 200, cols = 200 + Rand() % 200;
    CuMatrix<Real> M(rows, cols);
    M.SetRandUniform();

    M.Add(-0.5); // we'll be testing the central moments, so
    // center it around zero first.
    // Got these moments from http://mathworld.wolfram.com/UniformDistribution.html
    Vector<Real> central_moments(5);
    central_moments(0) = 0.0;
    central_moments(1) = 0.0;
    central_moments(2) = 1.0 / 12; // times (b - a)^2, which equals 1.
    central_moments(3) = 0.0;
    central_moments(4) = 1.0 / 80; // times (b - a)^4, which equals 1.

    for (int32 pow = 1; pow < central_moments.Dim(); pow++) {
      CuMatrix<Real> Mpow(M);
      Mpow.ApplyPow(pow);
      Real observed_moment = Mpow.Sum() / (rows * cols);
      // see http://en.wikipedia.org/wiki/Normal_distribution#Moments,
      // note that mu = 0 and sigma = 1.
      Real expected_moment = central_moments(pow);
      Real k = 20.0; // This is just a constant we use to give us some wiggle
                     // room before rejecting the distribution... e.g. 10 sigma,
                     // quite approximately.
      Real allowed_deviation = k / sqrt(static_cast<Real>(rows * cols));
      Real lower_bound = expected_moment - allowed_deviation,
          upper_bound = expected_moment + allowed_deviation;
      if (!(observed_moment >= lower_bound && observed_moment <= upper_bound)) {
        KALDI_LOG << "Random matrix is " << M;
        KALDI_ERR << "Bad observed " << pow <<  "'th moment " << observed_moment
                  << ", expected " << expected_moment << ", allowed range "
                  << lower_bound << " to " << upper_bound;
      }
    }
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyLowerToUpper() {
  for (int i = 1; i < 2; ++i) {
    MatrixIndexT dim = 10 * i + Rand() % 4 + (i == 9 ? 255 : 0);
    if (i == 8) dim = 0;
    CuMatrix<Real> A(dim, dim);
    A.SetRandn();
    Matrix<Real> A2(A);
    A.CopyLowerToUpper();
    Matrix<Real> A3(A);
    for (int32 i = 0; i < dim;  i++) {
      for (int32 j = 0; j <= i; j++) {
        KALDI_ASSERT(A3(i, j) == A3(j, i));
        KALDI_ASSERT(A3(i, j) == A2(i, j));
      }
    }
    KALDI_ASSERT(dim == 0 || A3.Trace() != 0);
  }
}

template<typename Real>
static void UnitTestCuMatrixSetZeroAboveDiag() {
  for (int i = 1; i < 2; ++i) {
    MatrixIndexT dim = 10 * i + Rand() % 4 + (i == 9 ? 255 : 0);
    if (i == 8) dim = 0;
    CuMatrix<Real> A(dim, dim);
    A.SetRandn();
    Matrix<Real> A_orig(A);
    A.SetZeroAboveDiag();
    Matrix<Real> A_copy(A);

    for (int32 i = 0; i < dim;  i++) {
      for (int32 j = 0; j < dim; j++) {
        Real aval = A_copy(i, j), aorigval = A_orig(i, j);
        KALDI_ASSERT(aval == (j > i ? 0.0 : aorigval));
      }
    }
  }
}


template<typename Real>
static void UnitTestCuMatrixCopyUpperToLower() {
  for (int i = 1; i < 10; ++i) {
    MatrixIndexT dim = 10 * i + Rand() % 4 + (i == 9 ? 255 : 0);
    if (i == 8) dim = 0;
    CuMatrix<Real> A(dim, dim);
    A.SetRandn();
    Matrix<Real> A2(A);
    A.CopyUpperToLower();
    Matrix<Real> A3(A);
    //KALDI_LOG << "A2 is " << A2 << " A3 is " << A3;
    for (int32 i = 0; i < dim;  i++) {
      for (int32 j = i; j < dim; j++) {
        KALDI_ASSERT(A3(i, j) == A3(j, i));
        KALDI_ASSERT(A3(i, j) == A2(i, j));
      }
    }
    KALDI_ASSERT(dim == 0 || A3.Trace() != 0);
  }
}


template<typename Real>
static void UnitTestCuMatrixObjfDeriv() {
  int32 n_r = 100 + Rand() % 200, n_c = 20 + Rand() % 30;
  CuMatrix<Real> A(n_r, n_c), B(n_r, n_c);
  B.SetRandn();
  B.Add(1.0);
  B.ApplyFloor(1.0e-10);

  std::vector<MatrixElement<Real> > labels;
  for(int i = 0; i < n_r; i++) {
    for(int j = 0; j < n_c; j++) {
      // have approximately one weight per row of the matrix.
      if (Rand() % n_c == 0) {
        A(i, j) = RandUniform();
        MatrixElement<Real> t = {i, j, A(i, j)};
        labels.push_back(t);
      }
    }
  }
  CuMatrix<Real> C(n_r, n_c);
  C.Set(0);
  Real a = 0, b = 0;

  // (sv_labels, logprobs, &tot_objf, &tot_weight)
  C.CompObjfAndDeriv(labels, B, &a, &b);

  KALDI_ASSERT(ApproxEqual(b, A.Sum()));

  Real sum2;  // sum(i, j) A(i, j) log(B(i, j));
  {
    CuMatrix<Real> Bcopy(B);
    Bcopy.ApplyLog();
    sum2 = TraceMatMat(Bcopy, A, kTrans);
  }
  KALDI_ASSERT(ApproxEqual(a, sum2));

  B.InvertElements();
  A.MulElements(B);  // each element of A is now A(i, j) / B(i, j);
  KALDI_ASSERT(ApproxEqual(A, C));
}

template<typename Real>
static void UnitTestCuMatrixAddElements() {
  for (int32 i = 0; i < 2; i++) {
    int32 dimM = 100 + Rand() % 50, dimN = 100 + Rand() % 50;
   // int32 dimM = 256, dimN = 256;
    CuMatrix<Real> H(dimM, dimN);
    H.SetRandn();
    CuMatrix<Real> H_copy(H);
    CuMatrix<Real> M(H);
    int32 num_elements = 100 + Rand() % 10;
    std::vector<MatrixElement<Real> > input;
    std::vector<Int32Pair> input_index;
    Real *input_value = new Real[num_elements];
    BaseFloat scale = -1 + (0.33 * (Rand() % 5));
    for (int32 j = 0; j < num_elements; j++) {
      MatrixIndexT r = Rand() % dimM;
      MatrixIndexT c = Rand() % dimN;
      Int32Pair tmp_pair;
      tmp_pair.first = r;
      tmp_pair.second = c;
      Real offset = -1 + (0.33 * (Rand() % 5));
      M(r, c) += scale * offset;
      MatrixElement<Real> t = {r, c, offset};
      input.push_back(t);
      input_index.push_back(tmp_pair);
      input_value[j] = offset;
    }
    H.AddElements(scale, input);
    CuArray<Int32Pair> cu_input_index(input_index);
    H_copy.AddElements(scale, cu_input_index, input_value);
    delete[] input_value;

    AssertEqual(H, M);
    AssertEqual(H_copy, M);
  }
}

template<typename Real>
static void UnitTestCuMatrixLookup() {
  for (int32 i = 0; i < 2; i++) {
    int32 dimM = 100 + Rand() % 200, dimN = 100 + Rand() % 200;
    CuMatrix<Real> H(dimM, dimN);
    H.SetRandn();

    int32 num_elements = 10 + Rand() % 10;
    std::vector<Int32Pair> indices;
    std::vector<Real> reference;
    std::vector<Real> output;
    output.resize(num_elements);

    // Generates the indices and the reference.
    for (int32 j = 0; j < num_elements; j++) {
      MatrixIndexT r = Rand() % dimM;
      MatrixIndexT c = Rand() % dimN;

      Int32Pair tmp_pair;
      tmp_pair.first = r;
      tmp_pair.second = c;
      indices.push_back(tmp_pair);
      reference.push_back(H(r, c));
    }

    H.Lookup(indices, &(output[0]));

    KALDI_ASSERT(reference == output);
  }
}

template<typename Real>
static void UnitTestCuMatrixEqualElementMask() {
  CuMatrix<Real> m1(10,9), m2(10,9);
  CuMatrix<Real> mask_same, mask_different;
  m1.SetRandUniform(); // U[0,1]
  m2.SetRandUniform(); m2.Add(10.0); // U[10,11]

  m1.EqualElementMask(m1,&mask_same); // all elements ones
  m1.EqualElementMask(m2,&mask_different); // all elements zeros

  //KALDI_LOG << m1 << m2 << mask_same << mask_different;
  KALDI_ASSERT(mask_same.Sum() == 10*9);
  KALDI_ASSERT(mask_different.Sum() == 0.0);

  //check matrices with different strides:
  CuMatrix<Real> m3(m1.Range(1,6,2,6));
  CuMatrix<Real> m4(5,5,kSetZero);
  m1.Range(1,5,2,5).EqualElementMask(m3.Range(0,5,0,5),&m4); // strides 9, 6, 5
  KALDI_ASSERT(m4.Sum() == 25);

}

template<typename Real> void CudaMatrixUnitTest() {
  UnitTestCuMatrixTraceMatMat<Real>();
  UnitTestCuMatrixObjfDeriv<Real>();
  //test CuMatrix<Real> methods by cross-check with Matrix
  UnitTestCuMatrixCopyCross<Real>();
  UnitTestCuMatrixCopyCross2<Real>();
  UnitTestCuMatrixApplyLog<Real>();
  UnitTestCuMatrixApplyExp<Real>();
  UnitTestCuMatrixSetRandn<Real>();
  UnitTestCuMatrixSetRandUniform<Real>();
  UnitTestCuMatrixScale<Real>();
  UnitTestCuMatrixSigmoid<Real>();
  UnitTestCuMatrixSoftHinge<Real>();
  UnitTestCuMatrixApplyPow<Real>();
  UnitTestCuMatrixApplyPowAbs<Real>();
  UnitTestCuMatrixSet<Real>();
  UnitTestCuMatrixAdd<Real>();
  UnitTestCuMatrixApplyFloor<Real>();
  UnitTestCuMatrixApplyCeiling<Real>();
  UnitTestCuMatrixApplyHeaviside<Real>();
  UnitTestCuMatrixHeaviside<Real>();
  UnitTestCuMatrixMulElements<Real>();
  UnitTestCuMatrixDivElements<Real>();
  UnitTestCuMatrixMax<Real>();
  UnitTestCuMatrixMin<Real>();
  UnitTestCuMatrixMulColsVec<Real>();
  UnitTestCuMatrixMulRowsVec<Real>();
  UnitTestCuMatrixDivRowsVec<Real>();
  UnitTestCuMatrixAddMat<Real>();
  UnitTestCuMatrixAddMatBlocks1<Real>();
  UnitTestCuMatrixAddMatBlocks1Trans<Real>();
  UnitTestCuMatrixAddMatBlocks2<Real>();
  UnitTestCuMatrixReduceSum<Real>();
  UnitTestCuMatrixReduceMax<Real>();
  UnitTestCuMatrixReduceMin<Real>();
  UnitTestCuMatrixAddVecToCols<Real>();
  UnitTestCuMatrixAddVecToRows<Real>();
  UnitTestCuMatrixAddMatMat<Real>();
  UnitTestCuMatrixAddVecVec<Real>();
  UnitTestCuMatrixSymAddMat2<Real>();
  UnitTestCuMatrixAddMatMatBatched<Real>();
  UnitTestCuMatrixSymInvertPosDef<Real>();
  UnitTestCuMatrixCopyFromMat<Real>();
  UnitTestCuMatrixCopyFromTp<Real>();
  UnitTestCuMatrixAddMatTp<Real>();
  UnitTestCuMatrixCopyCols<Real>();
  UnitTestCuMatrixAddCols<Real>();
  UnitTestCuMatrixSumColumnRanges<Real>();
  UnitTestCuMatrixCopyRows<Real>();
  UnitTestCuMatrixCopyRowsFromVec<Real>();
  UnitTestCuMatrixCopyColsFromVec<Real>();
  UnitTestCuMatrixCopyToRows<Real>();
  UnitTestCuMatrixAddRows<Real>();
  UnitTestCuMatrixAddToRows<Real>();
  UnitTestCuMatrixAddRowRanges<Real>();
  UnitTestCuMatrixAddTpMat<Real>();
  UnitTestCuMatrixTranspose<Real>();
  UnitTestCuMatrixCopyUpperToLower<Real>();
  UnitTestCuMatrixCopyLowerToUpper<Real>();
  UnitTestCuMatrixSetZeroAboveDiag<Real>();
  UnitTestCuMatrixAddElements<Real>();
  UnitTestCuMatrixLookup<Real>();
  UnitTestCuMatrixEqualElementMask<Real>();
  // test CuVector<Real> methods
  UnitTestCuVectorAddVec<Real>();
  UnitTestCuVectorAddRowSumMat<Real>();
  UnitTestCuVectorAddRowSumMatLarge<Real>();
  UnitTestCuVectorAddColSumMat<Real>();
  UnitTestCuVectorAddColSumMatLarge<Real>();
  UnitTestCuSubMatrix<Real>();
  UnitTestCuMatrixInvertElements<Real>();
  UnitTestCuVectorInvertElements<Real>();
  UnitTestCuMatrixIO<Real>();
  UnitTestCuSigmoid<Real>();
  UnitTestCuApproxEqual<Real>();
  UnitTestCuCopy<Real, float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported())
#endif
    UnitTestCuCopy<Real, double>();
  UnitTestCuMatrixAddToDiag<Real>();
  UnitTestCuMatrixAdd2<Real>();
  UnitTestCuDiffSigmoid<Real>();
  UnitTestCuDiffSoftmax<Real>();
  UnitTestCuDiffLogSoftmax<Real>();
  UnitTestCuMatrixGroupPnorm<Real>();
  UnitTestCuMatrixDiffGroupPnorm<Real>();
  UnitTestCuMatrixGroupMax<Real>();
  UnitTestCuMatrixGroupMaxDeriv<Real>();
  UnitTestCuMatrixMulRowsVec<Real>();
  UnitTestCuMatrixMulRowsGroupMat<Real>();
  UnitTestCuFindRowMaxId<Real>();
  UnitTestCuSoftmax<Real>();
  UnitTestCuLogSoftmax<Real>();
  UnitTestCuDiffXent<Real>();
  UnitTestCheck<Real>();
  UnitTestSwapCu2Cu<Real>();
  UnitTestSwapCu2M<Real>();
  UnitTestCuMatrixAddDiagVecMat<Real>();
  UnitTestCuMatrixAddMatDiagVec<Real>();
  UnitTestCuMatrixAddMatMatElements<Real>();
  UnitTestCuMatrixSetMatMatDivMat<Real>();
  UnitTestCuTanh<Real>();
  UnitTestCuCholesky<Real>();
  UnitTestCuDiffTanh<Real>();
  UnitTestCuVectorAddTpVec<Real>();
  UnitTestCuVectorMulTp<Real>();
}


} // namespace kaldi


int main() {
  int32 loop = 0;
#if HAVE_CUDA == 1
  for (loop = 0; loop < 2; loop++) {
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    kaldi::CudaMatrixUnitTest<float>();

#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaMatrixUnitTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaMatrixUnitTest<double>();
#endif

    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
#if HAVE_CUDA == 1
  } // No for loop if 'HAVE_CUDA != 1',
  SetVerboseLevel(4);
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

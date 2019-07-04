// cudamatrix/cu-matrix-speed-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)
//           2015  Guoguo Chen
//           2017  Shiyin Kang

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
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-sparse-matrix.h"

using namespace kaldi;


namespace kaldi {

template<typename Real>
std::string NameOf() {
  return (sizeof(Real) == 8 ? "<double>" : "<float>");
}

template<typename Real> void TestCuMatrixSum(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  Real result = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    result = M.Sum();
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG<< "For CuMatrix::TestCuMatrixSum" << NameOf<Real>() << ", for dim = "
  << dim << ", speed was " << gflops << " gigaflops, result = " << result;
}

template<typename Real> void TestCuMatrixMax(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  Real result = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    result = M.Max();
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG<< "For CuMatrix::TestCuMatrixMax" << NameOf<Real>() << ", for dim = "
  << dim << ", speed was " << gflops << " gigaflops, result = " << result;
}

template<typename Real> void TestCuMatrixMin(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  Real result = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    result = M.Min();
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG<< "For CuMatrix::TestCuMatrixMin" << NameOf<Real>() << ", for dim = "
  << dim << ", speed was " << gflops << " gigaflops, result = " << result;
}

template<typename Real> void TestCuMatrixDivRowsVec(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  CuVector<Real> V(dim);
  M.SetRandn();
  V.SetRandn();

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.DivRowsVec(V);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG<< "For CuMatrix::DivRowsVec" << NameOf<Real>() << ", for dim = "
  << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixTransposeNS(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim / 2);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.Transpose();
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter / 2) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG<< "For CuMatrix::TransposeNS" << NameOf<Real>() << ", for dim = "
  << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixTransposeS(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.Transpose();
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG<< "For CuMatrix::TransposeS" << NameOf<Real>() << ", for dim = "
  << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixTransposeCross(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<float> Mf(dim / 2, dim), ref(dim, dim / 2);
  CuMatrix<Real> Md(dim, dim / 2);
  Mf.SetRandn();
  ref = Mf;

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    Md.CopyFromMat(Mf, kTrans);
    Mf.CopyFromMat(Md, kTrans);
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG<< "For CuMatrix::TransposeCross" << NameOf<Real>() << ", for dim = "
  << dim << ", speed was " << gflops << " gigaflops.";

  AssertEqual(ref, Mf);
}

template<typename Real> void TestCuMatrixAddMat(int32 dim, int32 num_row_blocks,
                                                int32 num_col_blocks) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> A(dim, dim), B(dim * num_row_blocks, dim * num_col_blocks);
  A.SetRandn();
  B.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    for (int32 i = 0; i < num_row_blocks; i++) {
      for (int32 j = 0; j < num_col_blocks; j++) {
        A.AddMat(0.0, CuSubMatrix<Real>(B, i * dim, dim, j * dim, dim));
      }
    }
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * num_row_blocks * num_col_blocks * iter)
                     / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddMat" << NameOf<Real>() << ", for dim = "
            << dim << "numRowBlocks = "<< num_row_blocks << "numColBlocks = "
            << num_col_blocks << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixAddMatBlocks(int32 dim,
                                                      int32 num_row_blocks,
                                                      int32 num_col_blocks) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> A(dim, dim), B(dim * num_row_blocks, dim * num_col_blocks);
  A.SetRandn();
  B.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    A.AddMatBlocks(0.0, B);
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * num_row_blocks * num_col_blocks * iter)
                     / (tim.Elapsed() * 1.0e+09);
   KALDI_LOG << "For CuMatrix::AddMatBlocks" << NameOf<Real>() << ", for dim = "
             << dim << ", numRowBlocks = "<< num_row_blocks << ", numColBlocks = "
             << num_col_blocks << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixMatMat(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim), O(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    O.AddMatMat(1.0, M, kNoTrans, N, kNoTrans, 0.0);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddMatMat" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixMatMatBatched(int32 dim, int32 batchCount) {
  std::vector<CuMatrix<Real>* > a(batchCount), b(batchCount), c(batchCount);
  std::vector<CuSubMatrix<Real>* > A, B, C;

  for (int32 i = 0; i < batchCount; i++) {
    // first create a Matrix intance and then creat a SubMatrix instance from that
    a[i] = new CuMatrix<Real>(dim, dim);
    b[i] = new CuMatrix<Real>(dim, dim);
    c[i] = new CuMatrix<Real>(dim, dim);
    a[i]->SetRandn();
    b[i]->SetRandn();
    A.push_back(new CuSubMatrix<Real>(*(a[i]), 0, a[i]->NumRows(), 0,
                                      a[i]->NumCols()));
    B.push_back(new CuSubMatrix<Real>(*(b[i]), 0, b[i]->NumRows(), 0,
                                      b[i]->NumCols()));
    C.push_back(new CuSubMatrix<Real>(*(c[i]), 0, c[i]->NumRows(), 0,
                                      c[i]->NumCols()));
  }
  BaseFloat time_in_secs = 0.025;
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    AddMatMatBatched(static_cast<Real>(1.0), C, A, kNoTrans, B, kNoTrans,
                     static_cast<Real>(0.0));
  }
  for (int32 i = 0; i< batchCount; i++) {
    delete a[i]; delete b[i]; delete c[i];
    delete A[i]; delete B[i]; delete C[i];
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter * batchCount) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddMatMatBatched" << NameOf<Real>() << ", for dim = " << dim
            << ", batchSize = " << batchCount << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixAddDiagVecMat(int32 dim, MatrixTransposeType trans) {
  BaseFloat time_in_secs = 0.015;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  CuVector<Real> v(dim);
  M.SetRandn();
  v.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++)
    N.AddDiagVecMat(1.0, v, M, trans, 0.0);

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddDiagVecMat" << NameOf<Real>()
            << (trans == kTrans ? "[trans]" : "[no-trans]")
            << ", for dim = " << dim << ", speed was "
            << gflops << " gigaflops.";
}



template<typename Real> void TestSymInvertPosDef(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim * 2), N(dim, dim);
  M.SetRandn();
  N.SymAddMat2(1.0, M, kNoTrans, 0.0);
  CuMatrix<Real> Ncopy(N);

  int iter = 0;
  Timer tim;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    Ncopy.CopyFromMat(N);
    Ncopy.SymInvertPosDef();
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::TestCuInvertPosDef" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real>
static void TestCuMatrixCompObjfAndDeriv(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  // Previously tested for larger dims, but test was slow.

  int32 n_r = dim, n_c = dim + Rand() % 5;

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

  int iter = 0;
  Timer tim;
  Real a = 0.0, b = 0.0;
  for (;tim.Elapsed() < time_in_secs; iter++)
    C.CompObjfAndDeriv(labels, B, &a, &b);

  BaseFloat gflops = (n_r * n_c * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CompObjfAndDeriv" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";


  // do it one more time for correctness test.
  C.SetZero();
  C.CompObjfAndDeriv(labels, B, &a, &b);

  KALDI_ASSERT(ApproxEqual(b, A.Sum()));

  // repeat the real test.
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
static void TestCuFindRowMaxId(int32 dim) {

  int32 dimM = dim, dimN = dimM + Rand() % 5;

  Matrix<Real> Hi(dimM, dimN);
  Hi.SetRandn();

  CuMatrix<Real> Di(dimM, dimN);
  Di.CopyFromMat(Hi);

  std::vector<int32> Hmax(dimM);
  CuArray<int32> Dmax(dimN);

  BaseFloat time_in_secs = 0.025;
  int iter = 0;
  Timer tim;
  for (;tim.Elapsed() < time_in_secs; iter++)
    Di.FindRowMaxId(&Dmax);


  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::FindRowMaxId" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";


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



template<typename Real> void TestCuMatrixSigmoid(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.Sigmoid(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Sigmoid" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixHeaviside(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.ApplyHeaviside();
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Heaviside" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixMulRowsGroupMat(int32 dim) {
  BaseFloat time_in_secs = 0.025;

  int32 group_size = 5;
  CuMatrix<Real> M(dim, dim * group_size), N(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    M.MulRowsGroupMat(N);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * group_size * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::MulRowsGroupMat" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixDiffSoftmax(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim), L(dim, dim);
  M.SetRandn();
  N.SetRandn();
  L.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    N.DiffSoftmaxPerRow(M, L);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::DiffSoftmaxPerRow" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixDiffLogSoftmax(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim), L(dim, dim);
  M.SetRandn();
  N.SetRandn();
  L.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    N.DiffLogSoftmaxPerRow(M, L);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::DiffLogSoftmaxPerRow" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixSoftmax(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.SoftMaxPerRow(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Softmax" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixLogSoftmax(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++) {
    N.LogSoftMaxPerRow(M);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::LogSoftmax" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixGroupPnorm(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  int32 group_size = 4;
  CuMatrix<Real> M(dim, dim), N(dim, dim / group_size);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++)
    N.GroupPnorm(M, 2.0);

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::GroupPnorm" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixDiffGroupPnorm(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  int32 group_size = 8;
  CuMatrix<Real> iv(dim, dim), ov(dim, dim / group_size);
  CuMatrix<Real> id(dim, dim), od(dim, dim / group_size);
  iv.SetRandn();
  od.SetRandn();
  ov.GroupPnorm(iv, 2.0);
  Timer tim;
  int32 iter = 0;

  for (; tim.Elapsed() < time_in_secs; iter++)
    id.DiffGroupPnorm(iv, ov, od, 2.0);

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::DiffGroupPnorm" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixGroupMax(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  int32 group_size = 4;
  CuMatrix<Real> M(dim, dim), N(dim, dim / group_size);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++)
    N.GroupMax(M);

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::GroupMax" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixGroupMaxAllGroupSizes(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs;) {
    for (int group_size = 1; group_size <= dim; group_size++) {
      if (dim % group_size == 0) {
        CuMatrix<Real> N(dim, dim / group_size, kUndefined);
        N.GroupMax(M);
        iter++;
      }
    }
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::GroupMax (all group sizes)" << NameOf<Real>()
            << ", for dim = " << dim << ", speed was " << gflops
            << " gigaflops.";
}

template<typename Real> void TestCuMatrixGroupMaxDeriv(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  int32 group_size = 4;
  CuMatrix<Real> M(dim, dim), N(dim, dim / group_size), O(dim, dim);
  M.SetRandn();
  N.GroupMax(M);
  Timer tim;
  int32 iter = 0;

  for (;tim.Elapsed() < time_in_secs; iter++)
    O.GroupMaxDeriv(M, N);

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::GroupMaxDeriv" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixTraceMatMat(int32 dim) {
  for (int32 n = 0; n < 2; n++) {
    MatrixTransposeType trans = (n == 0 ? kNoTrans : kTrans);
    BaseFloat time_in_secs = 0.02;

    CuMatrix<Real> M(dim, dim), N(dim, dim);
    M.SetRandn();
    N.SetRandn();
    Timer tim;
    int32 iter = 0;
    for (;tim.Elapsed() < time_in_secs; iter++) {
      TraceMatMat(M, N, trans);
    }
    BaseFloat fdim = dim;
    BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For CuMatrix::TraceMatMat" << NameOf<Real>()
              << (trans == kTrans ? " [transposed]" : "") << ", for dim = "
              << dim << ", speed was " << gflops << " gigaflops.";
  }
}


template<typename Real> void TestCuMatrixCholesky(int32 dim) {
  BaseFloat time_in_secs = 0.025;

  CuMatrix<Real> M(dim, dim);
  M.AddToDiag(100.0);
  Timer tim;
  int32 iter = 0;
  for (;tim.Elapsed() < time_in_secs; iter++)
    M.Cholesky();

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Cholesky" << NameOf<Real>()
            << ", for dim = " << dim << ", speed was " << gflops << " gigaflops.";
}



template<typename Real> void TestCuMatrixCopyLowerToUpper(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyLowerToUpper();
  }
  CuMatrix<Real> M2(M, kTrans);
  AssertEqual(M, M2);
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyLowerToUpper" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixCopyFromTp(int32 dim, MatrixTransposeType trans) {
  BaseFloat time_in_secs = 0.025;
  CuTpMatrix<Real> T(dim);
  T.SetRandn();
  CuMatrix<Real> M(dim, dim);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyFromTp(T, trans);
  }
  TpMatrix<Real> T_cpu(T);
  Matrix<Real> M_cpu(T_cpu, trans);
  Matrix<Real> M2_cpu(M);
  AssertEqual(M_cpu, M2_cpu);

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyFromTp" << (trans == kNoTrans ? "[NoTrans]":"[Trans]")
            << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixCopyFromSp(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuSpMatrix<Real> S(dim);
  S.SetRandn();
  CuMatrix<Real> M(dim, dim);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyFromSp(S);
  }
  SpMatrix<Real> S_cpu(S);
  Matrix<Real> M_cpu(S_cpu);
  Matrix<Real> M2_cpu(M);
  AssertEqual(M_cpu, M2_cpu);

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyFromSp" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}



template<typename Real> void TestCuMatrixCopyUpperToLower(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyUpperToLower();
  }
  CuMatrix<Real> M2(M, kTrans);
  AssertEqual(M, M2);
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyUpperToLower" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}


template<typename Real> void TestCuMatrixResize(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    CuMatrix<Real>M(dim, dim, kUndefined);  // we are testing the allocation and deallocation time.
  }
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::TestCuMatrixResize" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixSetZeroAboveDiag(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim);
  M.SetRandn();
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++)
    M.SetZeroAboveDiag();
  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::SetZeroAboveDiag" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real>
void TestCuMatrixLookup(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  int32 dimM = dim, dimN = dim;
  CuMatrix<Real> H(dimM, dimN);
  H.SetRandn();
  std::vector<Int32Pair> indices;
  std::vector<Real> reference;
  std::vector<Real> output;
  // Generates the indices and the reference.
  int32 num_index = dim * dim;
  output.resize(num_index);
  for (int32 j = 0; j < num_index; j++) {
    MatrixIndexT r = Rand() % dimM;
    MatrixIndexT c = Rand() % dimN;

    Int32Pair tmp_pair;
    tmp_pair.first = r;
    tmp_pair.second = c;
    indices.push_back(tmp_pair);
    reference.push_back(H(r, c));
  }
  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed()< time_in_secs; iter++)
    H.Lookup(indices, &(output[0]));

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::Lookup" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixCopyRows1(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();

  std::vector<int32> reorder(dim);
  for (int32 i = 0; i < dim; i++) {
    reorder[i] = i;
  }
  CuArray<int32> reorder_cuda(reorder);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyRows(N, reorder_cuda);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyRows" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixCopyRows2(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();

  std::vector<const Real*> reorder_src(dim, NULL);
  for (int32 i = 0; i < dim; i++) {
    reorder_src[i] = N.RowData(i);
  }
  CuArray<const Real*> reorder_src_cuda(reorder_src);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyRows(reorder_src_cuda);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyRows" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixCopyToRows(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();

  std::vector<Real*> reorder_dst(dim, NULL);
  for (int32 i = 0; i < dim; i++) {
    reorder_dst[i] = N.RowData(i);
  }
  CuArray<Real*> reorder_dst_cuda(reorder_dst);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.CopyToRows(reorder_dst_cuda);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::CopyToRows" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixAddRows1(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();

  std::vector<int32> reorder(dim);
  for (int32 i = 0; i < dim; i++) {
    reorder[i] = i;
  }
  CuArray<int32> reorder_cuda(reorder);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.AddRows(0.5, N, reorder_cuda);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddRows" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixAddRows2(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();

  std::vector<const Real*> reorder_src(dim, NULL);
  for (int32 i = 0; i < dim; i++) {
    reorder_src[i] = N.RowData(i);
  }
  CuArray<const Real*> reorder_src_cuda(reorder_src);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.AddRows(0.5, reorder_src_cuda);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddRows" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixAddToRows(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();

  std::vector<Real*> reorder_dst(dim, NULL);
  for (int32 i = 0; i < dim; i++) {
    reorder_dst[i] = N.RowData(i);
  }
  CuArray<Real*> reorder_dst_cuda(reorder_dst);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.AddToRows(0.5, reorder_dst_cuda);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddToRows" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuMatrixAddRowRanges(int32 dim) {
  BaseFloat time_in_secs = 0.025;
  CuMatrix<Real> M(dim, dim), N(dim, dim);
  M.SetRandn();
  N.SetRandn();

  std::vector<Int32Pair> indexes(dim);
  for (int32 i = 0; i < dim; i++) {
    indexes[i].first = i;
    indexes[i].second = i + 1;
  }
  CuArray<Int32Pair> indexes_cuda(indexes);

  Timer tim;
  int32 iter = 0;
  for (; tim.Elapsed() < time_in_secs; iter++) {
    M.AddRowRanges(N, indexes_cuda);
  }

  BaseFloat fdim = dim;
  BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
  KALDI_LOG << "For CuMatrix::AddRowRanges" << NameOf<Real>() << ", for dim = "
            << dim << ", speed was " << gflops << " gigaflops.";
}

template<typename Real> void TestCuSparseMatrixTraceMatSmat(int32 dim) {
  for (int32 n = 0; n < 2; n++) {
    MatrixTransposeType trans = (n == 0 ? kNoTrans : kTrans);
    BaseFloat time_in_secs = 0.02;

    CuMatrix<Real> M(dim, dim);
    M.SetRandn();

    std::vector<std::vector<std::pair<MatrixIndexT, Real> > > pairs(dim);
    for (auto && row : pairs) {
      row.push_back( { MatrixIndexT(Rand() % dim), Real(Rand() % dim) });
    }
    SparseMatrix<Real> Ncpu(dim, pairs);
    CuSparseMatrix<Real> N(Ncpu);

    Timer tim;
    int32 iter = 0;
    for (;tim.Elapsed() < time_in_secs; iter++) {
      TraceMatSmat(M, N, trans);
    }
    BaseFloat fdim = dim;
    BaseFloat gflops = (fdim * fdim * iter) / (tim.Elapsed() * 1.0e+09);
    KALDI_LOG << "For CuSparseMatrix::TraceMatSmat" << NameOf<Real>()
              << (trans == kTrans ? " [transposed]" : "") << ", for dim = "
              << dim << ", speed was " << gflops << " gigaflops.";
  }
}


template<typename Real> void CudaMatrixSpeedTest() {
  std::vector<int32> sizes;
  sizes.push_back(16);
  sizes.push_back(32);
  sizes.push_back(64);
  sizes.push_back(128);
  sizes.push_back(256);
  sizes.push_back(512);
  sizes.push_back(1024);
  int32 ns = sizes.size();
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixDivRowsVec<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixResize<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixAddMat<Real>(sizes[s], 3, 3);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixAddMatBlocks<Real>(sizes[s], 3, 3);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixMatMat<Real>(sizes[s]);
  for (int32 s = 0; s + 1 < ns; s++)
    TestCuMatrixMatMatBatched<Real>(sizes[s], 10);
  for (int32 s = 0; s < ns; s++) {
    TestCuMatrixAddDiagVecMat<Real>(sizes[s], kNoTrans);
    TestCuMatrixAddDiagVecMat<Real>(sizes[s], kTrans);
  }
  for (int32 s = 0; s < ns; s++)
    TestSymInvertPosDef<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCholesky<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSigmoid<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixHeaviside<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuFindRowMaxId<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCompObjfAndDeriv<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixMulRowsGroupMat<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSoftmax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixDiffSoftmax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixDiffLogSoftmax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixLogSoftmax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixGroupPnorm<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixDiffGroupPnorm<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixGroupMax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixGroupMaxAllGroupSizes<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixGroupMaxDeriv<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixTraceMatMat<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuSparseMatrixTraceMatSmat<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyLowerToUpper<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyFromTp<Real>(sizes[s], kNoTrans);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyFromTp<Real>(sizes[s], kTrans);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyFromSp<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyUpperToLower<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSetZeroAboveDiag<Real>(sizes[s]);
  for (int32 s = 0; s + 2 < ns; s++)
    TestCuMatrixLookup<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyRows1<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyRows2<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixCopyToRows<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixAddRows1<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixAddRows2<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixAddToRows<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixAddRowRanges<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixTransposeCross<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixTransposeS<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixTransposeNS<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixSum<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixMax<Real>(sizes[s]);
  for (int32 s = 0; s < ns; s++)
    TestCuMatrixMin<Real>(sizes[s]);
}


} // namespace kaldi


int main() {
  SetVerboseLevel(1);
#if HAVE_CUDA == 1
  int32 loop = 0;
  for (loop = 0; loop < 2; loop++) {
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    kaldi::CudaMatrixSpeedTest<float>();
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CudaMatrixSpeedTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CudaMatrixSpeedTest<double>();
#endif
#if HAVE_CUDA == 1
  } // No for loop if 'HAVE_CUDA != 1',
  CuDevice::Instantiate().PrintProfile();
#endif
  KALDI_LOG << "Tests succeeded.";
}

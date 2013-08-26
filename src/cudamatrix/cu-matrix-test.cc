// cudamatrix/cuda-matrix-test.cc

// Copyright 2010  Karel Vesely
//           2013  Lucas Ondel
//           2013  Johns Hopkins University (author: Daniel Povey)

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

using namespace kaldi;


namespace kaldi {

/*
 * INITIALIZERS
 */
template<class Real> 
static void InitRand(VectorBase<Real> *v) {
  for (MatrixIndexT i = 0; i < v->Dim(); i++)
	(*v)(i) = RandGauss();
}



template<class Real> 
static void InitRand(MatrixBase<Real> *M) {
  do {
    for (MatrixIndexT i = 0;i < M->NumRows();i++)
      for (MatrixIndexT j = 0;j < M->NumCols();j++)
        (*M)(i, j) = RandGauss();
  } while (M->NumRows() != 0 && M->Cond() > 100);
}



template<class Real> 
static void RandGaussMatrix(MatrixBase<Real>* mat) {
  for(int32 r=0; r<mat->NumRows(); r++)
    for(int32 c=0; c<mat->NumCols(); c++)
      (*mat)(r,c) = RandGauss();
}



template<class Real> 
static void RandZeroToOneMatrix(MatrixBase<Real>* mat) {
  for(int32 r=0; r<mat->NumRows(); r++)
    for(int32 c=0; c<mat->NumCols(); c++)
      (*mat)(r,c) = RandUniform();
}




/*
 * ASSERTS
 */
template<class Real> 
static void AssertEqual(const MatrixBase<Real> &A,
                        const MatrixBase<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++) {
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
  }
}

template<class Real> 
static void AssertEqual(const CuMatrixBase<Real> &A,
                        const CuMatrixBase<Real> &B,
                        float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++) {
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
  }
}



template<class Real>
static bool ApproxEqual(const MatrixBase<Real> &A,
                        const MatrixBase<Real> &B, Real tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  MatrixBase<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min),
      d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}



template<class Real> 
static void AssertEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i=0; i < A.Dim(); i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}



template<class Real> 
static bool ApproxEqual(VectorBase<Real> &A, VectorBase<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i=0; i < A.Dim(); i++)
    if (std::abs(A(i)-B(i)) > tol) return false;
  return true;
}



static void AssertEqual(std::vector<int32> &A, std::vector<int32> &B) {
  KALDI_ASSERT(A.size() == B.size());
  for (size_t i=0; i < A.size(); i++)
    KALDI_ASSERT(A[i] == B[i]);
}



/*
 * Unit tests
 */

/*
 * CuMatrix
 */
template<class Real> 
static void UnitTestCuMatrixApplyLog() {
  Matrix<Real> H(100,100);
  RandGaussMatrix(&H);
  H.MulElements(H); //make numbers positive

  CuMatrix<Real> D(100,100);
  D.CopyFromMat(H);

  D.ApplyLog();
  H.ApplyLog();

  Matrix<Real> H2(100,100);
  D.CopyToMat(&H2);

  AssertEqual(H,H2);
}


template<class Real> 
static void UnitTestCuMatrixMulElements() {
  Matrix<Real> Ha(100,100);
  Matrix<Real> Hb(100,100);
  RandGaussMatrix(&Ha);
  RandGaussMatrix(&Hb);

  CuMatrix<Real> Da(100,100);
  CuMatrix<Real> Db(100,100);
  Da.CopyFromMat(Ha);
  Db.CopyFromMat(Hb);

  Da.MulElements(Db);
  Ha.MulElements(Hb);

  Matrix<Real> Ha2(100,100);
  Da.CopyToMat(&Ha2);

  AssertEqual(Ha,Ha2);
}



template<class Real> 
static void UnitTestCuMatrixMulColsVec() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(99);
  RandGaussMatrix(&Hm);
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



template<class Real> 
static void UnitTestCuMatrixMulRowsVec() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(100);
  RandGaussMatrix(&Hm);
  InitRand(&Hv);

  CuMatrix<Real> Dm(100,99);
  CuVector<Real> Dv(100);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dm.MulRowsVec(Dv);
  Hm.MulRowsVec(Hv);

  Matrix<Real> Hm2(100,99);
  Dm.CopyToMat(&Hm2);

  AssertEqual(Hm,Hm2);
}



template<class Real> 
static void UnitTestCuMatrixDivRowsVec() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(100);
  RandGaussMatrix(&Hm);
  InitRand(&Hv);

  CuMatrix<Real> Dm(100,99);
  CuVector<Real> Dv(100);
  Dm.CopyFromMat(Hm);
  Dv.CopyFromVec(Hv);

  Dm.DivRowsVec(Dv);
  Hv.InvertElements();
  Hm.MulRowsVec(Hv);

  Matrix<Real> Hm2(100,99);
  Dm.CopyToMat(&Hm2);

  AssertEqual(Hm,Hm2);
}



template<class Real> 
static void UnitTestCuMatrixAddMat() {
  Matrix<Real> Ha(100,100);
  Matrix<Real> Hb(100,100);
  RandGaussMatrix(&Ha);
  RandGaussMatrix(&Hb);

  CuMatrix<Real> Da(100,100);
  CuMatrix<Real> Db(100,100);
  Da.CopyFromMat(Ha);
  Db.CopyFromMat(Hb);

  Da.AddMat(0.5,Db);
  Ha.AddMat(0.5,Hb);

  Matrix<Real> Ha2(100,100);
  Da.CopyToMat(&Ha2);

  AssertEqual(Ha,Ha2);
}



template<class Real> 
static void UnitTestCuMatrixAddVecToCols() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(100);
  RandGaussMatrix(&Hm);
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



template<class Real> 
static void UnitTestCuMatrixAddVecToRows() {
  Matrix<Real> Hm(100,99);
  Vector<Real> Hv(99);
  RandGaussMatrix(&Hm);
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



template<class Real> 
static void UnitTestCuMatrixAddMatMat() {
  Matrix<Real> Ha(200,100);
  Matrix<Real> Hb(100,200);
  Matrix<Real> Hc1(200,200);
  Matrix<Real> Hc2(100,100);
  RandGaussMatrix(&Ha);
  RandGaussMatrix(&Hb);

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

template<class Real>
static void UnitTestCuMatrixCopyFromMat() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
    Matrix<Real> A(dim, dim);
    A.SetRandn();
    CuMatrix<Real> E(A);    
    CuMatrix<Real> B(dim, dim);
    B.CopyFromMat(E);

    AssertEqual<Real>(B, E);
  }
}

template<class Real>
static void UnitTestCuMatrixCopyFromTp() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
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

template<class Real>
static void UnitTestCuMatrixAddMatTp() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
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

template<class Real>
static void UnitTestCuMatrixAddTpMat() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 5 * i + rand() % 10;
    
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
template<class Real> 
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



template<class Real> 
static void UnitTestCuVectorAddRowSumMat() {
  const int32 X=4321, Y=19;
  Real alpha=0.1, beta=0.7;

  Matrix<Real> Hm(X,Y);
  Vector<Real> Hv(Y);
  Vector<Real> Hv_accu(Y);
  RandGaussMatrix(&Hm);
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



template<class Real> 
static void UnitTestCuVectorAddRowSumMatLarge() {
  Matrix<Real> Hm(1000,990);
  Vector<Real> Hv(990);
  Vector<Real> Hv_accu(990);
  RandGaussMatrix(&Hm);
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



template<class Real> 
static void UnitTestCuVectorAddColSumMat() {
  const int32 X=19, Y=4321;
  Real alpha=0.5, beta=0.7;

  Matrix<Real> Hm(X,Y);
  Vector<Real> Hv(X);
  Vector<Real> Hv_accu(X);
  RandGaussMatrix(&Hm);
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

template<class Real> 
static void UnitTestCuSubMatrix() {
  for (int32 iter = 0 ; iter < 10; iter++) {
    int32 M1 = 1 + rand () % 10, M2 = 1 + rand() % 1, M3 = 1 + rand() % 10, M = M1 + M2 + M3,
        N1 = 1 + rand () % 10, N2 = 1 + rand() % 1, N3 = 1 + rand() % 10, N = N1 + N2 + N3,
        m = rand() % M2, n = rand() % N2;
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



template<class Real> 
static void UnitTestCuVectorAddColSumMatLarge() {
  Matrix<Real> Hm(1000,990);
  Vector<Real> Hv(1000);
  Vector<Real> Hv_accu(1000);
  RandGaussMatrix(&Hm);
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



template<class Real> 
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

template<class Real> 
static void UnitTestCuApproxEqual() {
  Real tol = 0.1;
  for (int32 i = 0; i < 10; i++) {
    int32 M = 1 + rand() % 10, N = 1 + rand() % 10;
    CuMatrix<Real> A(M, N), B(M, N);
    A.SetRandn();
    B.SetRandn();
    Matrix<Real> diff(A), Bm(B);
    diff.AddMat(-1.0, Bm);
    Real norm = diff.FrobeniusNorm();
    KALDI_ASSERT( (norm <= tol) == (A.ApproxEqual(B, tol)));
    KALDI_LOG << "Norm is " << norm << ", tol = " << tol;
    tol *= 2.0;
  }
}


template<class Real, class OtherReal> 
static void UnitTestCuCopy() {
  for (int32 i = 0; i < 10; i++) {
    int32 M = 1 + rand() % 10, N = 1 + rand() % 10;
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

template<class Real> 
static void UnitTestCuSigmoid() {
  Matrix<Real> Hi(100,111);
  Matrix<Real> Ho(100,111);
  RandGaussMatrix(&Hi);

  CuMatrix<Real> Di(100,111);
  CuMatrix<Real> Do(100,111);
  Di.CopyFromMat(Hi);

  //gpu
  Do.Sigmoid(Di);
  //cpu
  for(MatrixIndexT r=0; r<Hi.NumRows(); r++) {
    for(MatrixIndexT c=0; c<Hi.NumCols(); c++) {
      Ho(r, c) = 1.0/(1.0+exp(-Hi(r, c)));
    }
  }

  Matrix<Real> Ho2(100,111);
  Do.CopyToMat(&Ho2);

  AssertEqual(Ho,Ho2);
}



template<class Real> 
static void UnitTestCuDiffSigmoid() {
  Matrix<Real> Hi(100,111);
  Matrix<Real> Ho(100,111);
  Matrix<Real> Hy(100,111);
  RandGaussMatrix(&Hi);
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



template<class Real> 
static void UnitTestCuSoftmax() {
  Matrix<Real> Hi(100,111);
  Matrix<Real> Ho(100,111);
  RandGaussMatrix(&Hi);
  
  CuMatrix<Real> Di(100,111);
  CuMatrix<Real> Do(100,111);
  Di.CopyFromMat(Hi);

  //gpu
  Do.ApplySoftMax(Di);
  //cpu
  Ho.CopyFromMat(Hi);
  for(MatrixIndexT r=0; r<Ho.NumRows(); r++) {
    Ho.Row(r).ApplySoftMax();
  }

  Matrix<Real> Ho2(100,111);
  Do.CopyToMat(&Ho2);

  AssertEqual(Ho,Ho2);
}



template<class Real> 
static void UnitTestCuFindRowMaxId() {
  Matrix<Real> Hi(100,111);
  RandGaussMatrix(&Hi);

  CuMatrix<Real> Di(100,111);
  Di.CopyFromMat(Hi);

  std::vector<int32> Hmax(100);
  CuStlVector<int32> Dmax(100);

  //gpu
  Di.FindRowMaxId(&Dmax);

  //cpu
  for(MatrixIndexT r=0; r<Hi.NumRows(); r++) {
    Real max=-1e20; int32 idx=-1;
    for(MatrixIndexT c=0; c<Hi.NumCols(); c++) {
      if(Hi(r,c) > max) { idx=c; max=Hi(r,c); }
    }
    Hmax[r] = idx;
  }

  std::vector<int32> Hmax2(100);
  Dmax.CopyToVec(&Hmax2);

  AssertEqual(Hmax,Hmax2);
}



template<class Real> 
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
    Htgt[i] = rand()%Y;
  }
  CuStlVector<int32> Dtgt(X);
  Dtgt.CopyFromVec(Htgt);
  //logpost vector
  Vector<Real> Hlogpost(X);
  CuVector<Real> Dlogpost(X);
  
  //gpu
  Di.DiffXent(Dtgt, &Dlogpost);
  //cpu
  for(MatrixIndexT r=0; r<Hi.NumRows(); r++) {
    int32 col_tgt = Htgt[r];
    Hlogpost(r) = log(Hi(r, col_tgt));
    Hi(r, col_tgt) -= 1.0;
  }

  Matrix<Real> Hi2(X,Y);
  Di.CopyToMat(&Hi2);
  Vector<Real> Hlogpost2(X);
  Dlogpost.CopyToVec(&Hlogpost2);

  AssertEqual(Hi,Hi2);
  AssertEqual(Hlogpost,Hlogpost2);
}

template<class Real> void UnitTestCheck() {
  Matrix<Real> Hi(100,111);
  RandGaussMatrix(&Hi);

  CuMatrix<Real> Di(100,111);
  Di.CopyFromMat(Hi);

  CuMatrix<Real> Dj(Di);
  KALDI_LOG << Dj.NumRows() << '\n';
 

}

template<class Real>
void UnitTestSwapCu2Cu() {
  Matrix<Real> Hi(100,111);
  RandGaussMatrix(&Hi);
  CuMatrix<Real> Di(100,111);
  Di.CopyFromMat(Hi);

  Matrix<Real> Hi2(110,121);
  RandGaussMatrix(&Hi2);
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

template<class Real>
void UnitTestSwapCu2M() {
  Matrix<Real> Hi(100,111);
  RandGaussMatrix(&Hi);
  CuMatrix<Real> Di(100,111);
  Di.CopyFromMat(Hi);

  Matrix<Real> Hi2(110,121);
  RandGaussMatrix(&Hi2);
  Matrix<Real> Di2(110,121);
  Di2.CopyFromMat(Hi2);

  Di.Swap(&Hi2);
  Matrix<Real> Hf(Di.NumRows(), Di.NumCols());
  Di.CopyToMat(&Hf);
  AssertEqual(Di2,Hf);
  AssertEqual(Hi2,Hi);
}

template<class Real> void CudaMatrixUnitTest() {
  //test CuMatrix<Real> methods by cross-check with Matrix
  UnitTestCuMatrixApplyLog<Real>();
  UnitTestCuMatrixMulElements<Real>();
  UnitTestCuMatrixMulColsVec<Real>();
  UnitTestCuMatrixMulRowsVec<Real>();
  UnitTestCuMatrixDivRowsVec<Real>();
  UnitTestCuMatrixAddMat<Real>();
  UnitTestCuMatrixAddVecToCols<Real>();
  UnitTestCuMatrixAddVecToRows<Real>();
  UnitTestCuMatrixAddMatMat<Real>();
  UnitTestCuMatrixCopyFromMat<Real>();
  UnitTestCuMatrixCopyFromTp<Real>();
  UnitTestCuMatrixAddMatTp<Real>();
  UnitTestCuMatrixAddTpMat<Real>();
  //test CuVector<Real> methods
  UnitTestCuVectorAddVec<Real>();
  UnitTestCuVectorAddRowSumMat<Real>();
  UnitTestCuVectorAddRowSumMatLarge<Real>();
  UnitTestCuVectorAddColSumMat<Real>();
  UnitTestCuVectorAddColSumMatLarge<Real>();
  UnitTestCuSubMatrix<Real>();
  UnitTestCuVectorInvertElements<Real>();


  UnitTestCuSigmoid<Real>();
  UnitTestCuApproxEqual<Real>();
  UnitTestCuCopy<Real, float>();
#if HAVE_CUDA == 1  
  if (CuDevice::Instantiate().DoublePrecisionSupported())
#endif
    UnitTestCuCopy<Real, double>();
  UnitTestCuDiffSigmoid<Real>();
  UnitTestCuFindRowMaxId<Real>();
  UnitTestCuSoftmax<Real>();
  UnitTestCuDiffXent<Real>();

  UnitTestCheck<Real>();

  UnitTestSwapCu2Cu<Real>();
  UnitTestSwapCu2M<Real>();
}


} // namespace kaldi


int main() {
    //Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(-2); //-2 .. automatic selection
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
  std::cout << "Tests succeeded.\n";
}

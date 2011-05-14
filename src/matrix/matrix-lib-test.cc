// matrix/matrix-lib-test.cc
// Copyright 2009-2011   Microsoft Corporation  Mohit Agarwal  Lukas Burget  Ondrej Glembek
//   Arnab Ghoshal  Go Vivace Inc.  Yanmin Qian  Jan Silovsky

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

#include "matrix/matrix-lib.h"

namespace kaldi {

template<class Real> static void InitRand(TpMatrix<Real> &M) {
  // start:
  for (MatrixIndexT i = 0;i < M.NumRows();i++)
	for (MatrixIndexT j = 0;j<=i;j++)
	  M(i, j) = RandGauss();
  // if (M.NumRows() != 0 && M.Cond() > 100) { printf("Condition number of random matrix large %f, trying again (this is normal)\n", (float)M.Cond()); goto start; }
}



template<class Real> static void InitRand(Vector<Real> &v) {
  for (MatrixIndexT i = 0;i < v.Dim();i++)
	v(i) = RandGauss();
}

template<class Real> static void InitRand(VectorBase<Real> &v) {
  for (MatrixIndexT i = 0;i < v.Dim();i++)
	v(i) = RandGauss();
}

template<class Real> static void InitRand(MatrixBase<Real> &M) {
start:
  for (MatrixIndexT i = 0;i < M.NumRows();i++)
    for (MatrixIndexT j = 0;j < M.NumCols();j++)
      M(i, j) = RandGauss();
  if (M.NumRows() != 0 && M.Cond() > 100) { printf("Condition number of random matrix large %f, trying again (this is normal)\n", (float)M.Cond()); goto start; }
}


template<class Real> static void InitRand(SpMatrix<Real> &M) {
 start:
  for (MatrixIndexT i = 0;i < M.NumRows();i++)
	for (MatrixIndexT j = 0;j<=i;j++)
	  M(i, j) = RandGauss();
  if (M.NumRows() != 0 && M.Cond() > 100) { printf("Condition number of random matrix large %f, trying again (this is normal)\n", (float)M.Cond()); goto start; }
}

template<class Real> static void AssertEqual(Matrix<Real> &A, Matrix<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j < A.NumCols();j++) {
	  KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
}

template<class Real> static void AssertEqual(SpMatrix<Real> &A, SpMatrix<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j<=i;j++)
	  KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
}

template<class Real>
static bool ApproxEqual(SpMatrix<Real> &A, SpMatrix<Real> &B, Real tol = 0.001) {

  KALDI_ASSERT(A.NumRows() == B.NumRows());
  SpMatrix<Real> diff(A);
  diff.AddSp(1.0, B);
  Real a = std::max(A.Max(), -A.Min()), b = std::max(B.Max(), -B.Min),
      d = std::max(diff.Max(), -diff.Min());
  return (d <= tol * std::max(a, b));
}

/* was:
template<class Real>
bool ApproxEqual(SpMatrix<Real> &A, SpMatrix<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j<=i;j++)
	  if (std::abs(A(i, j)-B(i, j)) > tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j))))) return false;
  return true;
}
*/

template<class Real> static void AssertEqual(Vector<Real> &A, Vector<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0;i < A.Dim();i++)
    KALDI_ASSERT(std::abs(A(i)-B(i)) < tol);
}

template<class Real> static bool ApproxEqual(Vector<Real> &A, Vector<Real> &B, float tol = 0.001) {
  KALDI_ASSERT(A.Dim() == B.Dim());
  for (MatrixIndexT i = 0;i < A.Dim();i++)
    if (std::abs(A(i)-B(i)) > tol) return false;
  return true;
}


template<class Real> static void AssertEqual(Real a, Real b, float tol = 0.001) {
  KALDI_ASSERT( std::abs(a-b) <= tol*(std::abs(a)+std::abs(b)));
}

template<class Real> static void CholeskyUnitTestTr() {
  for (int32 i = 0; i < 5; i++) {
    int32 dimM = 2 + rand() % 10;
    Matrix<Real> M(dimM, dimM);
    InitRand(M);
    SpMatrix<Real> S(dimM);
    S.AddMat2(1.0, M, kNoTrans, 0.0);
    TpMatrix<Real> C(dimM);
    C.Cholesky(S);
    Matrix<Real> CM(C);
    TpMatrix<Real> Cinv(C);
    Cinv.Invert();
    {
      Matrix<Real> A(C), B(Cinv), AB(dimM, dimM);
      AB.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
      KALDI_ASSERT(AB.IsUnit());
    }
    SpMatrix<Real> S2(dimM);
    S2.AddMat2(1.0, CM, kNoTrans, 0.0);
    AssertEqual(S, S2);
    C.Invert();
    Matrix<Real> CM2(C);
    CM2.Invert();
    SpMatrix<Real> S3(dimM);
    S3.AddMat2(1.0, CM2, kNoTrans, 0.0);
    AssertEqual(S, S3);
  }
}

template<class Real> static void UnitTestAddSp() {
  for (MatrixIndexT i = 0;i< 10;i++) {
    MatrixIndexT dimM = 10+rand()%10;
    SpMatrix<Real> S(dimM);
    InitRand(S);
    Matrix<Real> M(S), N(S);
    N.AddSp(2.0, S);
    M.Scale(3.0);
    AssertEqual(M, N);
  }
}

template<class Real> static void UnitTestSpliceRows() {

  for (MatrixIndexT i = 0;i< 10;i++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;

    Vector<Real> V(dimM*dimN), V10(dimM*dimN);
    Vector<Real> Vs(std::min(dimM, dimN)), Vs10(std::min(dimM, dimN));
    InitRand(V);
    Matrix<Real> M(dimM, dimN);
    M.CopyRowsFromVec(V);
    V10.CopyRowsFromMat(M);
    AssertEqual(V, V10);

    for (MatrixIndexT i = 0;i < dimM;i++)
      for (MatrixIndexT  j = 0;j < dimN;j++)
        KALDI_ASSERT(M(i, j) == V(i*dimN + j));

    {
      Vector<Real> V2(dimM), V3(dimM);
      InitRand(V2);
      MatrixIndexT x;
      M.CopyColFromVec(V2, x = (rand() % dimN));
      V3.CopyColFromMat(M, x);
      AssertEqual(V2, V3);
    }

    {
      Vector<Real> V2(dimN), V3(dimN);
      InitRand(V2);
      MatrixIndexT x;
      M.CopyRowFromVec(V2, x = (rand() % dimM));
      V3.CopyRowFromMat(M, x);
      AssertEqual(V2, V3);
    }

    {
      M.CopyColsFromVec(V);
      V10.CopyColsFromMat(M);
      AssertEqual(V, V10);
    }

    {
      M.CopyDiagFromVec(Vs);
      Vs10.CopyDiagFromMat(M);
      AssertEqual(Vs, Vs10);
    }

  }
}

template<class Real> static void UnitTestRemoveRow() {

  // this is for matrix
  for (MatrixIndexT p = 0;p< 10;p++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;
    Matrix<Real> M(dimM, dimN);
    InitRand(M);
    MatrixIndexT i = rand() % dimM;  // Row to remove.
    Matrix<Real> N(M);
    N.RemoveRow(i);
    for (MatrixIndexT j = 0;j < i;j++) {
      for (MatrixIndexT k = 0;k < dimN;k++) {
        KALDI_ASSERT(M(j, k) == N(j, k));
      }
    }
    for (MatrixIndexT j = i+1;j < dimM;j++) {
      for (MatrixIndexT k = 0;k < dimN;k++) {
        KALDI_ASSERT(M(j, k) == N(j-1, k));
      }
    }
  }

  // this is for vector
  for (MatrixIndexT p = 0;p< 10;p++) {
    MatrixIndexT dimM = 10+rand()%10;
    Vector<Real> V(dimM);
    InitRand(V);
    MatrixIndexT i = rand() % dimM;  // Element to remove.
    Vector<Real> N(V);
    N.RemoveElement(i);
    for (MatrixIndexT j = 0;j < i;j++) {
        KALDI_ASSERT(V(j) == N(j));
    }
    for (MatrixIndexT j = i+1;j < dimM;j++) {
        KALDI_ASSERT(V(j) == N(j-1));
    }
  }

}


template<class Real> static void UnitTestSubvector() {

  Vector<Real> V(100);
  InitRand(V);
  Vector<Real> V2(100);
  for (MatrixIndexT i = 0;i < 10;i++) {
    SubVector<Real> tmp(V, i*10, 10);
    SubVector<Real> tmp2(V2, i*10, 10);
    tmp2.CopyFromVec(tmp);
  }
  AssertEqual(V, V2);
}

template <class Real>
static void UnitTestSimpleForVec() {  // testing some simple operaters on vector

  for (MatrixIndexT i = 0; i < 5; i++) {
    Vector<Real> V(100), V1(100), V2(100), V3(100), V4(100);
    InitRand(V);
    if (i % 2 == 0) {
      V1.SetZero();
      V1.Add(1.0);
    } else {
      V1.Set(1.0);
    }

    V2.CopyFromVec(V);
    V3.CopyFromVec(V1);
    V2.InvertElements();
    V3.DivElemByElem(V);
    V4.CopyFromVec(V3);
    V4.AddVecDivVec(1.0, V1, V, 0.0);
    AssertEqual(V3, V2);
    AssertEqual(V4, V3);
    V4.MulElements(V);
    AssertEqual(V4, V1);
    V3.AddVecVec(1.0, V, V2, 0.0);
    AssertEqual(V3, V1);

    Vector<Real> V5(V), V6(V1), V7(V1), V8(V);
    V5.AddVec(1.0, V);
    V8.Scale(2.0);
    V6.AddVec2(1.0, V);
    V7.AddVecVec(1.0, V, V, 1.0);
    AssertEqual(V6, V7);
    AssertEqual(V5, V8);
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    MatrixIndexT dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN);
    InitRand(M);
    Vector<Real> Vr(dimN), Vc(dimM);
    Vr.AddRowSumMat(M);
    Vc.AddColSumMat(M);

    Vector<Real> V2r(dimN), V2c(dimM);
    for (MatrixIndexT k = 0; k < dimM; k++) {
      V2r.CopyRowFromMat(M, k);
      AssertEqual(V2r.Sum(), Vc(k));
    }
    for (MatrixIndexT j = 0; j < dimN; j++) {
      V2c.CopyColFromMat(M, j);
      AssertEqual(V2c.Sum(), Vr(j));
    }
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    Vector<Real> V(100), V1(100), V2(100);
    InitRand(V);

    V1.CopyFromVec(V);
    V1.ApplyExp();
    V2.Set(exp(V.LogSumExp()));
    V1.DivElemByElem(V2);
    V.ApplySoftMax();
    AssertEqual(V1, V);
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    int dimV = 10 + rand() % 10;
    Real p = 0.5 + RandUniform() * 4.5;
    Vector<Real> V(dimV), V1(dimV), V2(dimV);
    InitRand(V);
    V1.AddVecVec(1.0, V, V, 0.0);  // V1:=V.*V.
    V2.CopyFromVec(V1);
    AssertEqual(V1.Norm(p), V2.Norm(p));
    AssertEqual(sqrt(V1.Sum()), V.Norm(2.0));
  }

  for (MatrixIndexT i = 0; i < 5; i++) {
    int dimV = 10 + rand() % 10;
    Real p = RandUniform() * 1.0e-5;
    Vector<Real> V(dimV);
    V.Set(p);
    KALDI_ASSERT(V.IsZero(p));
    KALDI_ASSERT(!V.IsZero(p*0.9));
  }

  // Test ApplySoftMax() for matrix.
  Matrix<Real> M(10, 10);
  InitRand(M);
  M.ApplySoftMax();
  KALDI_ASSERT( fabs(1.0 - M.Sum()) < 0.01);

}


template<class Real>
static void UnitTestNorm() {  // test some simple norm properties: scaling.  also ApproxEqual test.

  for (MatrixIndexT p = 0; p < 10; p++) {
    Real scalar = RandGauss();
    if (scalar == 0.0) continue;
    if (scalar < 0) scalar *= -1.0;
    MatrixIndexT dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN);
    InitRand(M);
    SpMatrix<Real> S(dimM);
    InitRand(S);
    Vector<Real> V(dimN);
    InitRand(V);

    Real Mnorm = M.FrobeniusNorm(),
        Snorm = S.FrobeniusNorm(),
        Vnorm1 = V.Norm(1.0),
        Vnorm2 = V.Norm(2.0),
        Vnorm3 = V.Norm(3.0);
    M.Scale(scalar);
    S.Scale(scalar);
    V.Scale(scalar);
    KALDI_ASSERT(ApproxEqual(M.FrobeniusNorm(), Mnorm*scalar));
    KALDI_ASSERT(ApproxEqual(S.FrobeniusNorm(), Snorm*scalar));
    KALDI_ASSERT(ApproxEqual(V.Norm(1.0), Vnorm1 * scalar));
    KALDI_ASSERT(ApproxEqual(V.Norm(2.0), Vnorm2 * scalar));
    KALDI_ASSERT(ApproxEqual(V.Norm(3.0), Vnorm3 * scalar));

    KALDI_ASSERT(V.ApproxEqual(V));
    KALDI_ASSERT(M.ApproxEqual(M));
    KALDI_ASSERT(S.ApproxEqual(S));
    SpMatrix<Real> S2(S); S2.Scale(1.1);  KALDI_ASSERT(!S.ApproxEqual(S2));  KALDI_ASSERT(S.ApproxEqual(S2, 0.15));
    Matrix<Real> M2(M); M2.Scale(1.1);  KALDI_ASSERT(!M.ApproxEqual(M2));  KALDI_ASSERT(M.ApproxEqual(M2, 0.15));
    Vector<Real> V2(V); V2.Scale(1.1);  KALDI_ASSERT(!V.ApproxEqual(V2));  KALDI_ASSERT(V.ApproxEqual(V2, 0.15));
  }
}


template<class Real>
static void UnitTestSimpleForMat() {  // test some simple operates on all kinds of matrix

  for (MatrixIndexT p = 0; p < 10; p++) {
    // for FrobeniousNorm() function
    MatrixIndexT dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN);
    InitRand(M);
    {
      Matrix<Real> N(M);
      N.Add(2.0);
      for (MatrixIndexT m = 0; m < dimM; m++)
        for (MatrixIndexT n = 0; n < dimN; n++)
          N(m, n) -= 2.0;
      AssertEqual(M, N);
    }

    Matrix<Real> N(M), M1(M);
    M1.MulElements(M);
    Real tmp1 = sqrt(M1.Sum());
    Real tmp2 = N.FrobeniusNorm();
    KALDI_ASSERT(std::abs(tmp1 - tmp2) < 0.00001);

    // for LargestAbsElem() function
    Vector<Real> V(dimM);
    for (MatrixIndexT i = 0; i < dimM; i++) {
      for (MatrixIndexT j = 0; j < dimN; j++) {
        M(i, j) = std::abs(M(i, j));
      }
      std::sort(M.RowData(i), M.RowData(i) + dimN);
      V(i) = M(i, dimN - 1);
    }
    std::sort(V.Data(), V.Data() + dimM);
    KALDI_ASSERT(std::abs(V(dimM - 1) - N.LargestAbsElem()) < 0.00001);
  }

  SpMatrix<Real> x(3);
  x.SetZero();

  std::stringstream ss;

  ss << "DP 3\n";
  ss << "4.6863" << '\n';
  ss << "3.7062 4.6032" << '\n';
  ss << "3.4160 3.7256  5.2474" << '\n';

  ss >> x;
  KALDI_ASSERT(x.IsPosDef() == true);  // test IsPosDef() function

  TpMatrix<Real> y(3);
  y.SetZero();
  y.Cholesky(x);

  std::cout << "Matrix y is a lower triangular Cholesky decomposition of x:"
      << '\n';
  std::cout << y << '\n';

  // test sp-matrix's LogPosDefDet() function
  Matrix<Real> B(x);
  Real tmp;
  Real *DetSign = &tmp;
  KALDI_ASSERT(std::abs(B.LogDet(DetSign) - x.LogPosDefDet()) < 0.00001);

  for (MatrixIndexT p = 0; p < 10; p++) {  // test for sp and tp matrix's AddSp() and AddTp() function
    MatrixIndexT dimM = 10 + rand() % 10;
    SpMatrix<Real> S(dimM), S1(dimM);
    TpMatrix<Real> T(dimM), T1(dimM);
    InitRand(S);
    InitRand(S1);
    InitRand(T);
    InitRand(T1);
    Matrix<Real> M(S), M1(S1), N(T), N1(T1);

    S.AddSp(1.0, S1);
    T.AddTp(1.0, T1);
    M.AddMat(1.0, M1);
    N.AddMat(1.0, N1);
    Matrix<Real> S2(S);
    Matrix<Real> T2(T);

    AssertEqual(S2, M);
    AssertEqual(T2, N);
  }

  for (MatrixIndexT i = 0; i < 10; i++) {  // test for sp matrix's AddVec2() function
    MatrixIndexT dimM = 10 + rand() % 10;
    SpMatrix<Real> M(dimM);
    Vector<Real> V(dimM);

    InitRand(M);
    SpMatrix<double> Md(M);
    InitRand(V);
    SpMatrix<Real> Sorig(M);
    M.AddVec2(0.5, V);
    Md.AddVec2(static_cast<Real>(0.5), V);
    for (MatrixIndexT i = 0; i < dimM; i++)
      for (MatrixIndexT j = 0; j < dimM; j++) {
        KALDI_ASSERT(std::abs(M(i, j) - (Sorig(i, j)+0.5*V(i)*V(j))) < 0.001);
        KALDI_ASSERT(std::abs(Md(i, j) - (Sorig(i, j)+0.5*V(i)*V(j))) < 0.001);
      }
  }
}


template<class Real> static void UnitTestRow() {

  for (MatrixIndexT p = 0;p< 10;p++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;
    Matrix<Real> M(dimM, dimN);
    InitRand(M);

    MatrixIndexT i = rand() % dimM;  // Row to get.

    Vector<Real> V(dimN);
    V.CopyRowFromMat(M, i);  // get row.
    for (MatrixIndexT k = 0;k < dimN;k++) {
      AssertEqual(M(i, k), V(k));
    }

    MatrixIndexT j = rand() % dimN;  // Col to get.
    Vector<Real> W(dimM);
    W.CopyColFromMat(M, j);  // get row.
    for (MatrixIndexT k = 0;k < dimM;k++) {
      AssertEqual(M(k, j), W(k));
    }

  }
}

template<class Real> static void UnitTestAxpy() {

  for (MatrixIndexT i = 0;i< 10;i++) {
    MatrixIndexT dimM = 10+rand()%10, dimN = 10+rand()%10;
    Matrix<Real> M(dimM, dimN), N(dimM, dimN), O(dimN, dimM);

    InitRand(M); InitRand(N); InitRand(O);
    Matrix<Real> Morig(M);
    M.AddMat(0.5, N);
    for (MatrixIndexT i = 0;i < dimM;i++)
      for (MatrixIndexT j = 0;j < dimN;j++)
        KALDI_ASSERT(std::abs(M(i, j) - (Morig(i, j)+0.5*N(i, j))) < 0.1);
    M.CopyFromMat(Morig);
    M.AddMat(0.5, O, kTrans);
    for (MatrixIndexT i = 0;i < dimM;i++)
      for (MatrixIndexT j = 0;j < dimN;j++)
        KALDI_ASSERT(std::abs(M(i, j) - (Morig(i, j)+0.5*O(j, i))) < 0.1);
    {
      float f = 0.5 * (float) (rand() % 3);
      Matrix<Real> N(dimM, dimM);
      InitRand(N);

      Matrix<Real> N2(N);
      Matrix<Real> N3(N);
      N2.AddMat(f, N2, kTrans);
      N3.AddMat(f, N, kTrans);
      AssertEqual(N2, N3);  // check works same with self as arg.
    }
  }
}


template<class Real> static void UnitTestPower() {
  for (int iter = 0;iter < 5;iter++) {
    // this is for matrix-pow
    int dimM = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimM), N(dimM, dimM);
    InitRand(M);
    N.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);  // N:=M*M^T.
    SpMatrix<Real> S(dimM);
    S.CopyFromMat(N);  // symmetric so should not crash.
    S.ApplyPow(0.5);
    S.ApplyPow(2.0);
    M.CopyFromSp(S);
    AssertEqual(M, N);

    // this is for vector-pow
    int dimV = 10 + rand() % 10;
    Vector<Real> V(dimV), V1(dimV), V2(dimV);
    InitRand(V);
    V1.AddVecVec(1.0, V, V, 0.0);  // V1:=V.*V.
    V2.CopyFromVec(V1);
    V2.ApplyPow(0.5);
    V2.ApplyPow(2.0);
    AssertEqual(V1, V2);
  }
}


template<class Real> static void UnitTestSger() {
  for (int iter = 0;iter < 5;iter++) {
    MatrixIndexT dimM = 10 + rand() % 10;
    MatrixIndexT dimN = 10 + rand() % 10;
    Matrix<Real> M(dimM, dimN), M2(dimM, dimN);
    Vector<Real> v1(dimM); InitRand(v1);
    Vector<Real> v2(dimN); InitRand(v2);
    Vector<double> v1d(v1), v2d(v2);
    M.AddVecVec(1.0, v1, v2);
    M2.AddVecVec(1.0, v1, v2);
    for (MatrixIndexT m = 0;m < dimM;m++)
      for (MatrixIndexT n = 0;n < dimN;n++) {
        KALDI_ASSERT(M(m, n) - v1(m)*v2(n) < 0.01);
        KALDI_ASSERT(M(m, n) - M2(m, n) < 0.01);
      }
  }
}



template<class Real> static void UnitTestDeterminant() {  // also tests matrix axpy and IsZero() and TraceOfProduct{, T}
  for (int iter = 0;iter < 5;iter++) {  // First test the 2 det routines are the same
	int dimM = 10 + rand() % 10;
	Matrix<Real> M(dimM, dimM), N(dimM, dimM);
	InitRand(M);
	N.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);  // N:=M*M^T.
	for (MatrixIndexT i = 0;i < (MatrixIndexT)dimM;i++) N(i, i) += 0.0001;  // Make sure numerically +ve det-- can by chance be almost singular the way we initialized it (I think)
	SpMatrix<Real> S(dimM);
	S.CopyFromMat(N);  // symmetric so should not crash.
	Real logdet = S.LogPosDefDet();
	Real logdet2, logdet3, sign2, sign3;
	logdet2 = N.LogDet(&sign2);
    logdet3 = S.LogDet(&sign3);
	KALDI_ASSERT(sign2 == 1.0 && sign3 == 1.0 && std::abs(logdet2-logdet) < 0.1 && std::abs(logdet2 - logdet3) < 0.1);
	Matrix<Real> tmp(dimM, dimM); tmp.SetZero();
	tmp.AddMat(1.0, N);
	tmp.AddMat(-1.0, N, kTrans);
	// symmetric so tmp should be zero.
	if ( ! tmp.IsZero(1.0e-04)) {
	  printf("KALDI_ERR: matrix is not zero\n");
	  std::cout << tmp;
	  KALDI_ASSERT(0);
	}

	Real a = TraceSpSp(S, S), b = TraceMatMat(N, N), c = TraceMatMat(N, N, kTrans);
	KALDI_ASSERT(std::abs(a-b) < 0.1 && std::abs(b-c) < 0.1);
  }
}


template<class Real> static void UnitTestDeterminantSign() {

  for (int iter = 0;iter < 20;iter++) {  // First test the 2 det routines are the same
	int dimM = 10 + rand() % 10;
	Matrix<Real> M(dimM, dimM), N(dimM, dimM);
	InitRand(M);
	N.AddMatMat(1.0, M, kNoTrans, M, kTrans, 0.0);  // N:=M*M^T.
	for (MatrixIndexT i = 0;i < (MatrixIndexT)dimM;i++) N(i, i) += 0.0001;  // Make sure numerically +ve det-- can by chance be almost singular the way we initialized it (I think)
	SpMatrix<Real> S(dimM);
	S.CopyFromMat(N);  // symmetric so should not crash.
	Real logdet = S.LogPosDefDet();
	Real logdet2, logdet3, sign2, sign3;
	logdet2 = N.LogDet(&sign2);
    logdet3 = S.LogDet(&sign3);
	KALDI_ASSERT(sign2 == 1.0 && sign3 == 1.0 && std::abs(logdet2-logdet) < 0.01 && std::abs(logdet2 - logdet3) < 0.01);

    int num_sign_changes = rand() % 5;
    for (int change = 0; change < num_sign_changes; change++) {
      // Change sign of S's det by flipping one eigenvalue, and N by flipping one row.
      {
        Matrix<Real> M(S);
        Matrix<Real> U(dimM, dimM), Vt(dimM, dimM);
        Vector<Real> s(dimM);
        M.Svd(&s, &U, &Vt);  // SVD: M = U diag(s) Vt
        s(rand() % dimM) *= -1;
        U.MulColsVec(s);
        M.AddMatMat(1.0, U, kNoTrans, Vt, kNoTrans, 0.0);
        S.CopyFromMat(M);
      }
      // change sign of N:
      N.Row(rand() % dimM).Scale(-1.0);
    }

    // add in a scaling factor too.
    Real tmp = 1.0 + ((rand() % 5) * 0.01);
    Real logdet_factor = dimM * log(tmp);
    N.Scale(tmp);
    S.Scale(tmp);

    Real logdet4, logdet5, sign4, sign5;
	logdet4 = N.LogDet(&sign4);
    logdet5 = S.LogDet(&sign5);
    AssertEqual(logdet4, logdet+logdet_factor, 0.01);
    AssertEqual(logdet5, logdet+logdet_factor, 0.01);
    if (num_sign_changes % 2 == 0) {
      KALDI_ASSERT(sign4 == 1);
      KALDI_ASSERT(sign5 == 1);
    } else {
      KALDI_ASSERT(sign4 == -1);
      KALDI_ASSERT(sign5 == -1);
    }
  }
}


template<class Real> static void UnitTestSherman() {
  for (int iter = 0;iter < 1;iter++) {
	MatrixIndexT dimM =10;  // 20 + rand()%10;
	MatrixIndexT dimK =2;  // 20 + rand()%dimM;
	Matrix<Real> A(dimM, dimM), U(dimM, dimK), V(dimM, dimK);
	InitRand(A);
	InitRand(U);
	InitRand(V);
	for (MatrixIndexT i = 0;i < (MatrixIndexT)dimM;i++) A(i, i) += 0.0001;  // Make sure numerically +ve det-- can by chance be almost singular the way we initialized it (I think)

	Matrix<Real> tmpL(dimM, dimM);
	tmpL.AddMatMat(1.0, U, kNoTrans, V, kTrans, 0.0);  // tmpL =U *V.
	tmpL.AddMat(1.0, A);
	tmpL.Invert();

	Matrix<Real> invA(dimM, dimM);
	invA.CopyFromMat(A);
	invA.Invert();


	Matrix<Real> tt(dimK, dimM), I(dimK, dimK);
	tt.AddMatMat(1.0, V, kTrans, invA, kNoTrans, 0.0);  // tt = trans(V) *inv(A)

	Matrix<Real> tt1(dimM, dimK);
	tt1.AddMatMat(1.0, invA, kNoTrans, U, kNoTrans, 0.0);  // tt1=INV A *U
	Matrix<Real> tt2(dimK, dimK);
	tt2.AddMatMat(1.0, V, kTrans, tt1, kNoTrans, 0.0);
	for (MatrixIndexT i = 0;i < dimK;i++)
		for (MatrixIndexT j = 0;j < dimK;j++)
		{
			if (i == j)
				I(i, j) = 1.0;
			else
				I(i, j) = 0.0;
		}
	tt2.AddMat(1.0, I);   // I = identity
	tt2.Invert();

	Matrix<Real> tt3(dimK, dimM);
	tt3.AddMatMat(1.0, tt2, kNoTrans, tt, kNoTrans, 0.0);	// tt = tt*tran(V)
	Matrix<Real> tt4(dimM, dimM);
	tt4.AddMatMat(1.0, U, kNoTrans, tt3, kNoTrans, 0.0);	// tt = U*tt
	Matrix<Real> tt5(dimM, dimM);
	tt5.AddMatMat(1.0, invA, kNoTrans, tt4, kNoTrans, 0.0);	// tt = inv(A)*tt

	Matrix<Real> tmpR(dimM, dimM);
	tmpR.CopyFromMat(invA);
	tmpR.AddMat(-1.0, tt5);
	// printf("#################%f###############################.%f##############", tmpL, tmpR);

	AssertEqual<Real>(tmpL, tmpR);	// checks whether LHS = RHS or not...

  }
}


template<class Real> static void UnitTestTraceProduct() {
  for (int iter = 0;iter < 5;iter++) {  // First test the 2 det routines are the same
	int dimM = 10 + rand() % 10, dimN = 10 + rand() % 10;
	Matrix<Real> M(dimM, dimN), N(dimM, dimN);

	InitRand(M);
	InitRand(N);
	Matrix<Real> Nt(N, kTrans);
	Real a = TraceMatMat(M, Nt), b = TraceMatMat(M, N, kTrans);
	printf("m = %d, n = %d\n", dimM, dimN);
	std::cout << a << " " << b << '\n';
	KALDI_ASSERT(std::abs(a-b) < 0.1);
  }
}

template<class Real> static void UnitTestSvd() {
#ifndef HAVE_ATLAS
  int Base = 3, Rand = 2, Iter = 25;
  for (int iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;
    if (dimM < dimN) std::swap(dimM, dimN);  // Check that rows() >= cols(), dimM>=dimN, as required by JAMA_SVD, and
    // which we ensure inside our Lapack routine for portability to systems with no Lapack.
	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, dimN), Vt(dimN, dimN); Vector<Real> s(dimN);
	InitRand(M);
	if (iter < 2) std::cout << "M " << M;
	Matrix<Real> M2(dimM, dimN); M2.CopyFromMat(M);
	M.LapackGesvd(&s, &U, &Vt);
	if (iter < 2) {
	  std::cout << " s " << s;
	  std::cout << " U " << U;
	  std::cout << " Vt " << Vt;
	}

    Matrix<Real> S(dimN, dimN);
    S.CopyDiagFromVec(s);
	Matrix<Real> Mtmp(dimM, dimN);
    Mtmp.SetZero();
    Mtmp.AddMatMatMat(1.0, U, kNoTrans, S, kNoTrans, Vt, kNoTrans, 0.0);
    AssertEqual(Mtmp, M2);
  }
#endif
}

template<class Real> static void UnitTestSvdBad() {
  int32 N = 20;
  {
    Matrix<Real> M(N, N);
    // M.Set(1591.3614306764898);
    M.Set(1.0);
    M(0, 0) *= 1.000001;
    Matrix<Real> U(N, N), V(N, N);
    Vector<Real> l(N);
    M.Svd(&l, &U, &V);
  }
  SpMatrix<Real> M(N);
  for(int32 i =0; i < N; i++)
    for(int32 j = 0; j <= i; j++)
      M(i, j) = 1591.3614306764898;
  M(0, 0) *= 1.00001;
  M(10, 10) *= 1.00001;
  Matrix<Real> U(N, N);
  Vector<Real> l(N);
  M.SymPosSemiDefEig(&l, &U);
}


template<class Real> static void UnitTestSvdZero() {
  int Base = 3, Rand = 2, Iter = 30;
  for (int iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;  // M>=N.
	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, dimM), Vt(dimN, dimN); Vector<Real> v(std::min(dimM, dimN));
    if (iter%2 == 0) M.SetZero();
    else M.Unit();
	if (iter < 2) std::cout << "M " << M;
	Matrix<Real> M2(dimM, dimN); M2.CopyFromMat(M);
    bool ans = M.Svd(&v, &U, &Vt);
	KALDI_ASSERT(ans);  // make sure works with zero matrix.
  }
}





template<class Real> static void UnitTestSvdNodestroy() {
  int Base = 3, Rand = 2, Iter = 25;
  for (int iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimN = Base + rand() % Rand, dimM =  dimN + rand() % Rand;  // M>=N, as required by JAMA Svd.
    MatrixIndexT minsz = std::min(dimM, dimN);
	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, minsz), Vt(minsz, dimN); Vector<Real> v(minsz);
	InitRand(M);
	if (iter < 2) std::cout << "M " << M;
	M.Svd(&v, &U, &Vt);
	if (iter < 2) {
	  std::cout << " v " << v;
	  std::cout << " U " << U;
	  std::cout << " Vt " << Vt;
	}

    for (MatrixIndexT it = 0;it < 2;it++) {
      Matrix<Real> Mtmp(minsz, minsz);
      for (MatrixIndexT i = 0;i < v.Dim();i++) Mtmp(i, i) = v(i);
      Matrix<Real> Mtmp2(minsz, dimN);
      Mtmp2.AddMatMat(1.0, Mtmp, kNoTrans, Vt, kNoTrans, 0.0);
      Matrix<Real> Mtmp3(dimM, dimN);
      Mtmp3.AddMatMat(1.0, U, kNoTrans, Mtmp2, kNoTrans, 0.0);
      for (MatrixIndexT i = 0;i < Mtmp.NumRows();i++) {
        for (MatrixIndexT j = 0;j < Mtmp.NumCols();j++) {
          KALDI_ASSERT(std::abs(Mtmp3(i, j) - M(i, j)) < 0.0001);
        }
      }

      SortSvd(&v, &U, &Vt);  // and re-check...
    }
  }
}


/*
template<class Real> static void UnitTestSvdVariants() {  // just make sure it doesn't crash if we call it but don't want left or right singular vectors. there are KALDI_ASSERTs inside the Svd.
#ifndef HAVE_ATLAS
  int Base = 10, Rand = 5, Iter = 25;
  for (int iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;
    // if (dimM<dimN) std::swap(dimM, dimN);  // M>=N.
	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, dimM), Vt(dimN, dimN); Vector<Real> v(std::min(dimM, dimN));
	Matrix<Real> Utmp(dimM, 1); Matrix<Real> Vttmp(1, dimN);
	InitRand(M);
	M.Svd(v, U, Vttmp, "A", "N");
	M.Svd(v, Utmp, Vt, "N", "A");
	Matrix<Real> U2(dimM, dimM), Vt2(dimN, dimN); Vector<Real> v2(std::min(dimM, dimN));
	M.Svd(v, U2, Vt2, "A", "A");
	AssertEqual(U, U2); AssertEqual(Vt, Vt2);

  }
#endif
}*/

template<class Real> static void UnitTestSvdJustvec() {  // Making sure gives same answer if we get just the vector, not the eigs.
  int Base = 10, Rand = 5, Iter = 25;
  for (int iter = 0;iter < Iter;iter++) {
	MatrixIndexT dimM = Base + rand() % Rand, dimN =  Base + rand() % Rand;  // M>=N.
    MatrixIndexT minsz = std::min(dimM, dimN);

	Matrix<Real> M(dimM, dimN);
	Matrix<Real> U(dimM, minsz), Vt(minsz, dimN); Vector<Real> v(minsz);
	M.Svd(&v, &U, &Vt);
    Vector<Real> v2(minsz);
    M.Svd(&v2);
    AssertEqual(v, v2);
  }
}

template<class Real> static void UnitTestEigSymmetric() {

  for (int iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10;
    SpMatrix<Real> S(dimM);
    InitRand(S);
    Matrix<Real> M(S);  // copy to regular matrix.
    Matrix<Real> P(dimM, dimM);
    Vector<Real> real_eigs(dimM), imag_eigs(dimM);
    M.Eig(&P, &real_eigs, &imag_eigs);
    KALDI_ASSERT(imag_eigs.Sum() == 0.0);
    // should have M = P D P^T
    Matrix<Real> tmp(P); tmp.MulColsVec(real_eigs);  // P * eigs
    Matrix<Real> M2(dimM, dimM);
    M2.AddMatMat(1.0, tmp, kNoTrans, P, kTrans, 0.0);  // M2 = tmp * Pinv = P * eigs * P^T
    AssertEqual(M, M2);  // check reconstruction worked.
  }
}

template<class Real> static void UnitTestEig() {

  for (int iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 1 + iter;
    /*    if (iter < 10)
      dimM = 1 + rand() % 6;
    else
    dimM = 5 + rand()%10; */
    Matrix<Real> M(dimM, dimM);
    InitRand(M);
    Matrix<Real> P(dimM, dimM);
    Vector<Real> real_eigs(dimM), imag_eigs(dimM);
    M.Eig(&P, &real_eigs, &imag_eigs);

    {  // Check that the eigenvalues match up with the determinant.
      BaseFloat logdet_check = 0.0, logdet = M.LogDet();
      for (MatrixIndexT i = 0; i < dimM ; i++)
        logdet_check += 0.5 * log(real_eigs(i)*real_eigs(i) + imag_eigs(i)*imag_eigs(i));
      AssertEqual(logdet_check, logdet);
    }
    Matrix<Real> Pinv(P);
    Pinv.Invert();
    Matrix<Real> D(dimM, dimM);
    CreateEigenvalueMatrix(real_eigs, imag_eigs, &D);

    // check that M = P D P^{-1}.
    Matrix<Real> tmp(dimM, dimM);
    tmp.AddMatMat(1.0, P, kNoTrans, D, kNoTrans, 0.0);  // tmp = P * D
    Matrix<Real> M2(dimM, dimM);
    M2.AddMatMat(1.0, tmp, kNoTrans, Pinv, kNoTrans, 0.0);  // M2 = tmp * Pinv = P * D * Pinv.

    {  // print out some stuff..
      Matrix<Real> MP(dimM, dimM);
      MP.AddMatMat(1.0, M, kNoTrans, P, kNoTrans, 0.0);
      Matrix<Real> PD(dimM, dimM);
      PD.AddMatMat(1.0, P, kNoTrans, D, kNoTrans, 0.0);

      Matrix<Real> PinvMP(dimM, dimM);
      PinvMP.AddMatMat(1.0, Pinv, kNoTrans, MP, kNoTrans, 0.0);
      AssertEqual(MP, PD);
    }

    AssertEqual(M, M2);  // check reconstruction worked.
  }

}

template<class Real> static void UnitTestMmul() {
  for (int iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10, dimO = 20 + rand()%10;  // dims between 10 and 20.
	// MatrixIndexT dimM = 2, dimN = 3, dimO = 4;
	Matrix<Real> A(dimM, dimN), B(dimN, dimO), C(dimM, dimO);
	InitRand(A);
	InitRand(B);
	//
	// std::cout <<"a = " << A;
	// std::cout<<"B = " << B;
	C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);  // C = A * B.
	//	std::cout << "c = " << C;
	for (MatrixIndexT i = 0;i < dimM;i++) {
	  for (MatrixIndexT j = 0;j < dimO;j++) {
		double sum = 0.0;
		for (MatrixIndexT k = 0;k < dimN;k++) {
		  sum += A(i, k) * B(k, j);
		}
		KALDI_ASSERT(std::abs(sum - C(i, j)) < 0.0001);
	  }
	}
  }
}


template<class Real> static void UnitTestMmulSym() {

  // Test matrix multiplication on symmetric matrices.
  for (int iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10;

	Matrix<Real> A(dimM, dimM), B(dimM, dimM), C(dimM, dimM), tmp(dimM, dimM), tmp2(dimM, dimM);
    SpMatrix<Real> sA(dimM), sB(dimM), sC(dimM), stmp(dimM);

	InitRand(A); InitRand(B); InitRand(C);
    // Make A, B, C symmetric.
    tmp.CopyFromMat(A); A.AddMat(1.0, tmp, kTrans);
    tmp.CopyFromMat(B); B.AddMat(1.0, tmp, kTrans);
    tmp.CopyFromMat(C); C.AddMat(1.0, tmp, kTrans);

    sA.CopyFromMat(A);
    sB.CopyFromMat(B);
    sC.CopyFromMat(C);


    tmp.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);  // tmp = A * B.
    tmp2.AddSpSp(1.0, sA, sB, 0.0);  // tmp = sA*sB.
    AssertEqual(tmp, tmp2);
    tmp2.AddSpSp(1.0, sA, sB, 0.0);  // tmp = sA*sB.
    AssertEqual(tmp, tmp2);
  }
}



template<class Real> static void UnitTestVecmul() {
  for (int iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10;  // dims between 10 and 20.

	Matrix<Real> A(dimM, dimN);
	InitRand(A);
	Vector<Real> x(dimM), y(dimN);
	InitRand(y);


	x.AddMatVec(1.0, A, kNoTrans, y, 0.0);  // x = A * y.
	for (MatrixIndexT i = 0;i < dimM;i++) {
	  double sum = 0.0;
	  for (MatrixIndexT j = 0;j < dimN;j++) {
		sum += A(i, j) * y(j);
	  }
	  KALDI_ASSERT(std::abs(sum - x(i)) < 0.0001);
	}
  }
}

template<class Real> static void UnitTestInverse() {
  for (int iter = 0;iter < 10;iter++) {
	MatrixIndexT dimM = 20 + rand()%10;
	Matrix<Real> A(dimM, dimM), B(dimM, dimM), C(dimM, dimM);
	InitRand(A);
	B.CopyFromMat(A);
    B.Invert();

	C.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);  // C = A * B.


	for (MatrixIndexT i = 0;i < dimM;i++)
	  for (MatrixIndexT j = 0;j < dimM;j++)
		KALDI_ASSERT(std::abs(C(i, j) - (i == j?1.0:0.0)) < 0.1);
  }
}




template<class Real> static void UnitTestMulElements() {
  for (int iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10;
	Matrix<Real> A(dimM, dimN), B(dimM, dimN), C(dimM, dimN);
	InitRand(A);
	InitRand(B);

	C.CopyFromMat(A);
	C.MulElements(B);  // C = A .* B (in Matlab, for example).

	for (MatrixIndexT i = 0;i < dimM;i++)
	  for (MatrixIndexT j = 0;j < dimN;j++)
		KALDI_ASSERT(std::abs(C(i, j) - (A(i, j)*B(i, j))) < 0.0001);
  }
}

template<class Real> static void UnitTestSpLogExp() {
  for (int i = 0; i < 5; i++) {
    MatrixIndexT dimM = 10 + rand() % 10;

    Matrix<Real> M(dimM, dimM); InitRand(M);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, M, kNoTrans, 0.0);  // B = M*M^T -> positive definite (since M nonsingular).

    SpMatrix<Real> B2(B);
    B2.Log();
    B2.Exp();
    AssertEqual(B, B2);

    SpMatrix<Real> B3(B);
    B3.Log();
    B3.Scale(0.5);
    B3.Exp();
    Matrix<Real> sqrt(B3);
    SpMatrix<Real> B4(B.NumRows());
    B4.AddMat2(1.0, sqrt, kNoTrans, 0.0);
    AssertEqual(B, B4);
  }
}

template<class Real> static void UnitTestDotprod() {
  for (int iter = 0;iter < 5;iter++) {
	MatrixIndexT dimM = 200 + rand()%100;
	Vector<Real> v(dimM), w(dimM);

	InitRand(v);
	InitRand(w);
    Vector<double> wd(w);

	Real f = VecVec(w, v), f2 = VecVec(wd, v), f3 = VecVec(v, wd);
	Real sum = 0.0;
	for (MatrixIndexT i = 0;i < dimM;i++) sum += v(i)*w(i);
	KALDI_ASSERT(std::abs(f-sum) < 0.0001);
    KALDI_ASSERT(std::abs(f2-sum) < 0.0001);
    KALDI_ASSERT(std::abs(f3-sum) < 0.0001);
  }
}

template<class Real>
static void UnitTestResize() {
  for (size_t i = 0; i < 10; i++) {
    MatrixIndexT dimM1 = rand() % 10, dimN1 = rand() % 10,
        dimM2 = rand() % 10, dimN2 = rand() % 10;
    if (dimM1*dimN1 == 0) dimM1 = dimN1 = 0;
    if (dimM2*dimN2 == 0) dimM2 = dimN2 = 0;
    for (int j = 0; j < 3; j++) {
      MatrixResizeType resize_type = static_cast<MatrixResizeType>(j);
      Matrix<Real> M(dimM1, dimN1);
      InitRand(M);
      Matrix<Real> Mcopy(M);
      Vector<Real> v(dimM1);
      InitRand(v);
      Vector<Real> vcopy(v);
      SpMatrix<Real> S(dimM1);
      InitRand(S);
      SpMatrix<Real> Scopy(S);
      M.Resize(dimM2, dimN2, resize_type);
      v.Resize(dimM2, resize_type);
      S.Resize(dimM2, resize_type);
      if (resize_type == kSetZero) {
        KALDI_ASSERT(S.IsZero());
        KALDI_ASSERT(v.Sum() == 0.0);
        KALDI_ASSERT(M.IsZero());
      } else if (resize_type == kCopyData) {
        for (MatrixIndexT i = 0; i < dimM2; i++) {
          if (i < dimM1) AssertEqual(v(i), vcopy(i));
          else KALDI_ASSERT(v(i) == 0);
          for (MatrixIndexT j = 0; j < dimN2; j++) {
            if (i < dimM1 && j < dimN1) AssertEqual(M(i, j), Mcopy(i, j));
            else AssertEqual(M(i, j), 0.0);
          }
          for (MatrixIndexT i2 = 0; i2 < dimM2; i2++) {
            if (i < dimM1 && i2 < dimM1) AssertEqual(S(i, i2), Scopy(i, i2));
            else AssertEqual(S(i, i2), 0.0);
          }
        }
      }
    }
  }
}


template<class Real>
static void UnitTestTransposeScatter() {
  for (int iter = 0;iter < 10;iter++) {

	MatrixIndexT dimA = 10 + rand()%3;
	MatrixIndexT dimO = 10 + rand()%3;
    Matrix<Real>   Af(dimA, dimA);
	SpMatrix<Real> Ap(dimA);
	Matrix<Real>   M(dimO, dimA);
    Matrix<Real>   Of(dimO, dimO);
	SpMatrix<Real> Op(dimO);
    Matrix<Real>   A_MT(dimA, dimO);

	for (MatrixIndexT i = 0;i < Ap.NumRows();i++) {
	  for (MatrixIndexT j = 0; j<=i; j++) {
	     Ap(i, j) = RandGauss();
	  }
	}
	for (MatrixIndexT i = 0;i < M.NumRows();i++) {
	  for (MatrixIndexT j = 0; j < M.NumCols(); j++) {
	     M(i, j) = RandGauss();
	  }
	}
/*
   std::stringstream ss("1 2 3");
	ss >> Ap;
	ss.clear();
	ss.str("5 6 7 8 9 10");
	ss >> M;
*/

    Af.CopyFromSp(Ap);
    A_MT.AddMatMat(1.0, Af, kNoTrans, M, kTrans, 0.0);
    Of.AddMatMat(1.0, M, kNoTrans, A_MT, kNoTrans, 0.0);
    Op.AddMat2Sp(1.0, M, kNoTrans, Ap, 0.0);


//    std::cout << "A" << '\n' << Af << '\n';
//    std::cout << "M" << '\n' << M << '\n';
//    std::cout << "Op" << '\n' << Op << '\n';

    for (MatrixIndexT i = 0; i < dimO; i++) {
	  for (MatrixIndexT j = 0; j<=i; j++) {
		KALDI_ASSERT(std::abs(Of(i, j) - Op(i, j)) < 0.0001);
      }
    }

    A_MT.Resize(dimO, dimA);
    A_MT.AddMatMat(1.0, Of, kNoTrans, M, kNoTrans, 0.0);
    Af.AddMatMat(1.0, M, kTrans, A_MT, kNoTrans, 1.0);
    Ap.AddMat2Sp(1.0, M, kTrans, Op, 1.0);

//    std::cout << "Ap" << '\n' << Ap << '\n';
//    std::cout << "Af" << '\n' << Af << '\n';

    for (MatrixIndexT i = 0; i < dimA; i++) {
	  for (MatrixIndexT j = 0; j<=i; j++) {
		KALDI_ASSERT(std::abs(Af(i, j) - Ap(i, j)) < 0.01);
      }
    }
  }
}


template<class Real>
static void UnitTestRankNUpdate() {
  for (int iter = 0;iter < 10;iter++) {
	MatrixIndexT dimA = 10 + rand()%3;
	MatrixIndexT dimO = 10 + rand()%3;
    Matrix<Real>   Af(dimA, dimA);
	SpMatrix<Real> Ap(dimA);
	SpMatrix<Real> Ap2(dimA);
	Matrix<Real>   M(dimO, dimA);
    InitRand(M);
	Matrix<Real>   N(M, kTrans);
    Af.AddMatMat(1.0, M, kTrans, M, kNoTrans, 0.0);
    Ap.AddMat2(1.0, M, kTrans, 0.0);
    Ap2.AddMat2(1.0, N, kNoTrans, 0.0);
    Matrix<Real> Ap_f(Ap);
    Matrix<Real> Ap2_f(Ap2);
    AssertEqual(Ap_f, Af);
    AssertEqual(Ap2_f, Af);
  }
}

template<class Real> static void  UnitTestSpInvert() {
  for (int i = 0;i < 30;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
	SpMatrix<Real> M(dimM);
	for (MatrixIndexT i = 0;i < M.NumRows();i++)
	  for (MatrixIndexT j = 0;j<=i;j++) M(i, j) = RandGauss();
	SpMatrix<Real> N(dimM);
	N.CopyFromSp(M);
    if (rand() % 2 == 0)
      N.Invert();
    else
      N.InvertDouble();
	Matrix<Real> Mm(dimM, dimM), Nm(dimM, dimM), Om(dimM, dimM);
	Mm.CopyFromSp(M); Nm.CopyFromSp(N);
	Om.AddMatMat(1.0, Mm, kNoTrans, Nm, kNoTrans, 0.0);
    KALDI_ASSERT(Om.IsUnit( 0.01*dimM ));
  }
}


template<class Real> static void  UnitTestTpInvert() {
  for (int i = 0;i < 30;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
	TpMatrix<Real> M(dimM);
	for (MatrixIndexT i = 0;i < M.NumRows();i++) {
	  for (MatrixIndexT j = 0;j < i;j++) M(i, j) = RandGauss();
      M(i, i) = 20 * std::max((Real)0.1, (Real) RandGauss());  // make sure invertible by making it diagonally dominant (-ish)
    }
	TpMatrix<Real> N(dimM);
	N.CopyFromTp(M);
	N.Invert();
    TpMatrix<Real> O(dimM);

	Matrix<Real> Mm(dimM, dimM), Nm(dimM, dimM), Om(dimM, dimM);
	Mm.CopyFromTp(M); Nm.CopyFromTp(N);

	Om.AddMatMat(1.0, Mm, kNoTrans, Nm, kNoTrans, 0.0);
    KALDI_ASSERT(Om.IsUnit(0.001));
  }
}


template<class Real> static void  UnitTestLimitCondInvert() {
  for (int i = 0;i < 10;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
    MatrixIndexT dimN = dimM + 1 + rand()%10;

    SpMatrix<Real> B(dimM);
    Matrix<Real> X(dimM, dimN); InitRand(X);
    B.AddMat2(1.0, X, kNoTrans, 0.0);  // B = X*X^T -> positive definite (almost certainly), since N > M.


    SpMatrix<Real> B2(B);
    B2.LimitCond(1.0e+10, true);  // Will invert.

    Matrix<Real> Bf(B), B2f(B2);
    Matrix<Real> I(dimM, dimM); I.AddMatMat(1.0, Bf, kNoTrans, B2f, kNoTrans, 0.0);
    KALDI_ASSERT(I.IsUnit(0.1));
  }
}


template<class Real> static void  UnitTestFloorChol() {
  for (int i = 0;i < 10;i++) {
	MatrixIndexT dimM = 20 + rand()%10;


	MatrixIndexT dimN = 20 + rand()%10;
    Matrix<Real> X(dimM, dimN); InitRand(X);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, X, kNoTrans, 0.0);  // B = X*X^T -> positive semidefinite.

    float alpha = (rand() % 10) + 0.5;
	Matrix<Real> M(dimM, dimM);
    InitRand(M);
    SpMatrix<Real> C(dimM);
    C.AddMat2(1.0, M, kNoTrans, 0.0);  // C:=M*M^T
    InitRand(M);
    C.AddMat2(1.0, M, kNoTrans, 1.0);  // C+=M*M^T (after making new random M)
    if (i%2 == 0)
      C.Scale(0.001);  // so it's not too much bigger than B (or it's trivial)
    SpMatrix<Real> BFloored(B); BFloored.ApplyFloor(C, alpha);


    for (int j = 0;j < 10;j++) {
      Vector<Real> v(dimM);
      InitRand(v);
      Real ip_b = VecSpVec(v, B, v);
      Real ip_a = VecSpVec(v, BFloored, v);
      Real ip_c = alpha * VecSpVec(v, C, v);
      if (i < 3) std::cout << "alpha = " << alpha << ", ip_a = " << ip_a << " ip_b = " << ip_b << " ip_c = " << ip_c <<'\n';
      KALDI_ASSERT(ip_a>=ip_b*0.999 && ip_a>=ip_c*0.999);
    }
  }
}




template<class Real> static void  UnitTestFloorUnit() {
  for (int i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
    MatrixIndexT dimN = 20 + rand()%10;
    float floor = (rand() % 10) - 3;

    Matrix<Real> M(dimM, dimN); InitRand(M);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, M, kNoTrans, 0.0);  // B = M*M^T -> positive semidefinite.

    SpMatrix<Real> BFloored(B); BFloored.ApplyFloor(floor);


    Vector<Real> s(dimM); Matrix<Real> P(dimM, dimM); B.SymPosSemiDefEig(&s, &P);
    Vector<Real> s2(dimM); Matrix<Real> P2(dimM, dimM); BFloored.SymPosSemiDefEig(&s2, &P2);

    KALDI_ASSERT ( (s.Min() > floor && std::abs(s2.Min()-s.Min())<0.01) || std::abs(s2.Min()-floor)<0.01);
  }
}

template<class Real> static void  UnitTestMat2Vec() {
  for (int i = 0; i < 5; i++) {
    MatrixIndexT dimM = 10 + rand() % 10;

    Matrix<Real> M(dimM, dimM); InitRand(M);
    SpMatrix<Real> B(dimM);
    B.AddMat2(1.0, M, kNoTrans, 0.0);  // B = M*M^T -> positive definite (since M nonsingular).

    Matrix<Real> P(dimM, dimM);
    Vector<Real> s(dimM);

    B.SymPosSemiDefEig(&s, &P);
    SpMatrix<Real> B2(dimM);
    B2.CopyFromSp(B);
    B2.Scale(0.25);

    // B2 <-- 2.0*B2 + 0.5 * P * diag(v)  * P^T
    B2.AddMat2Vec(0.5, P, kNoTrans, s, 2.0);  // 2.0 * 0.25 + 0.5 = 1.
    AssertEqual(B, B2);

    SpMatrix<Real> B3(dimM);
    Matrix<Real> PT(P, kTrans);
    B3.AddMat2Vec(1.0, PT, kTrans, s, 0.0);
    AssertEqual(B, B3);
  }
}

template<class Real> static void  UnitTestLimitCond() {
  for (int i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10;
    SpMatrix<Real> B(dimM);
    B(1, 1) = 10000;
    KALDI_ASSERT(B.LimitCond(1000) == (dimM-1));
    KALDI_ASSERT(std::abs(B(2, 2) - 10.0) < 0.01);
    KALDI_ASSERT(std::abs(B(3, 0)) < 0.001);
  }
}

template<class Real> static void  UnitTestSimple() {
  for (int i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%20;
    Matrix<Real> M(dimM, dimN);
    M.SetUnit();
    KALDI_ASSERT(M.IsUnit());
    KALDI_ASSERT(!M.IsZero());
    KALDI_ASSERT(M.IsDiagonal());

    SpMatrix<Real> S(dimM); InitRand(S);
    Matrix<Real> N(S);
    KALDI_ASSERT(!N.IsDiagonal());  // technically could be diagonal, but almost infinitely unlikely.
    KALDI_ASSERT(N.IsSymmetric());
    KALDI_ASSERT(!N.IsUnit());
    KALDI_ASSERT(!N.IsZero());

    M.SetZero();
    KALDI_ASSERT(M.IsZero());
    Vector<Real> V(dimM*dimN); InitRand(V);
    Vector<Real> V2(V), V3(dimM*dimN);
    V2.ApplyExp();
    AssertEqual(V.Sum(), V2.SumLog());
	V3.ApplyLogAndCopy(V2);
    V2.ApplyLog();
    AssertEqual(V, V2);
	AssertEqual(V3, V2);


    KALDI_ASSERT(!S.IsDiagonal());
    KALDI_ASSERT(!S.IsUnit());
    N.SetUnit();
    S.CopyFromMat(N);
    KALDI_ASSERT(S.IsDiagonal());
    KALDI_ASSERT(S.IsUnit());
    N.SetZero();
    S.CopyFromMat(N);
    KALDI_ASSERT(S.IsZero());
    KALDI_ASSERT(S.IsDiagonal());
  }
}

template<class Real> static void UnitTestIoOld() {  // deprecated I/O style.

  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = rand()%10 + 1;
    MatrixIndexT dimN = rand()%10 + 1;
    bool binary = (i%2 == 0);

    if (i == 0) {
      dimM = 0;dimN = 0;  // test case when both are zero.
    }
    Matrix<Real> M(dimM, dimN);
    InitRand(M);
    Matrix<Real> N;
    Vector<Real> v1(dimM);
    InitRand(v1);
    Vector<Real> v2(dimM);

    SpMatrix<Real> S(dimM);
    InitRand(S);
    SpMatrix<Real> T(dimM);

    {
      std::ofstream outs("tmpf", std::ios_base::out |std::ios_base::binary);
      InitKaldiOutputStream(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v1.Write(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v1.Write(outs, binary);
    }

	{
      std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
      bool binary_in;
      InitKaldiInputStream(ins, &binary_in);
      N.Read(ins, binary_in);
      T.Read(ins, binary_in);
      v2.Read(ins, binary_in);
      if (i%2 == 0)
        ((MatrixBase<Real>&)N).Read(ins, binary_in, true);  // add
      else
        N.Read(ins, binary_in, true);
      T.Read(ins, binary_in, true);  // add
      if (i%2 == 0)
        ((VectorBase<Real>&)v2).Read(ins, binary_in, true);  // add
      else
        v2.Read(ins, binary_in, true);
    }
    N.Scale(0.5);
    v2.Scale(0.5);
    T.Scale(0.5);
    AssertEqual(M, N);
    AssertEqual(v1, v2);
    AssertEqual(S, T);
  }
}



template<class Real> static void UnitTestIo () {  // newer I/O test with the kaldi streams.

  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = rand()%10 + 1;
    MatrixIndexT dimN = rand()%10 + 1;
    bool binary = (i%2 == 0);

    if (i == 0) {
      dimM = 0;dimN = 0;  // test case when both are zero.
    }
    Matrix<Real> M(dimM, dimN);
    InitRand(M);
    Matrix<Real> N;
    Vector<Real> v1(dimM);
    InitRand(v1);
    Vector<Real> v2(dimM);

    SpMatrix<Real> S(dimM);
    InitRand(S);
    SpMatrix<Real> T(dimM);

    {
      std::ofstream outs("tmpf", std::ios_base::out |std::ios_base::binary);
      InitKaldiOutputStream(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v1.Write(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v1.Write(outs, binary);
    }

	{
      bool binary_in;
      bool either_way = (i%2 == 0);
      std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
      InitKaldiInputStream(ins, &binary_in);
      N.Resize(0, 0);
      T.Resize(0);
      v2.Resize(0);
      N.Read(ins, binary_in, either_way);
      T.Read(ins, binary_in, either_way);
      v2.Read(ins, binary_in, either_way);
      if (i%2 == 0)
        ((MatrixBase<Real>&)N).Read(ins, binary_in, true);  // add
      else
        N.Read(ins, binary_in, true);
      T.Read(ins, binary_in, true);  // add
      if (i%2 == 0)
        ((VectorBase<Real>&)v2).Read(ins, binary_in, true);  // add
      else
        v2.Read(ins, binary_in, true);
    }
    N.Scale(0.5);
    v2.Scale(0.5);
    T.Scale(0.5);
    AssertEqual(M, N);
    AssertEqual(v1, v2);
    AssertEqual(S, T);
  }
}


template<class Real> static void UnitTestIoCross() {  // across types.

  typedef typename OtherReal<Real>::Real Other;  // e.g. if Real == float, Other == double.
  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = rand()%10 + 1;
    MatrixIndexT dimN = rand()%10 + 1;
    bool binary = (i%2 == 0);
    if (i == 0) {
      dimM = 0;dimN = 0;  // test case when both are zero.
    }
    Matrix<Real> M(dimM, dimN);
    Matrix<Other> MO;
    InitRand(M);
    Matrix<Real> N(dimM, dimN);
    Vector<Real> v(dimM);
    Vector<Other> vO;
    InitRand(v);
    Vector<Real> w(dimM);

    SpMatrix<Real> S(dimM);
    SpMatrix<Other> SO;
    InitRand(S);
    SpMatrix<Real> T(dimM);

    {
      std::ofstream outs("tmpf", std::ios_base::out |std::ios_base::binary);
      InitKaldiOutputStream(outs, binary);

      M.Write(outs, binary);
      S.Write(outs, binary);
      v.Write(outs, binary);
      M.Write(outs, binary);
      S.Write(outs, binary);
      v.Write(outs, binary);
    }
	{
      std::ifstream ins("tmpf", std::ios_base::in | std::ios_base::binary);
      bool binary_in;
      InitKaldiInputStream(ins, &binary_in);

      MO.Read(ins, binary_in);
      SO.Read(ins, binary_in);
      vO.Read(ins, binary_in);
      MO.Read(ins, binary_in, true);
      SO.Read(ins, binary_in, true);
      vO.Read(ins, binary_in, true);
      N.CopyFromMat(MO);
      T.CopyFromSp(SO);
      w.CopyFromVec(vO);
    }
    N.Scale(0.5);
    w.Scale(0.5);
    T.Scale(0.5);
    AssertEqual(M, N);
    AssertEqual(v, w);
    AssertEqual(S, T);
  }
}


template<class Real> static void UnitTestHtkIo() {

  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = rand()%10 + 10;
    MatrixIndexT dimN = rand()%10 + 10;

    HtkHeader hdr;
    hdr.mNSamples = dimM;
    hdr.mSamplePeriod = 10000;  // in funny HTK units-- can set it arbitrarily
    hdr.mSampleSize = sizeof(float)*dimN;
    hdr.mSampleKind = 10;  // don't know what this is.

    Matrix<Real> M(dimM, dimN);
    InitRand(M);

    {
      std::ofstream os("tmpf", std::ios::out|std::ios::binary);
      WriteHtk(os, M, hdr);
    }

    Matrix<Real> N;
    HtkHeader hdr2;
    {
      std::ifstream is("tmpf", std::ios::in|std::ios::binary);
      ReadHtk(is, &N, &hdr2);
    }

    AssertEqual(M, N);
    KALDI_ASSERT(hdr.mNSamples == hdr2.mNSamples);
    KALDI_ASSERT(hdr.mSamplePeriod == hdr2.mSamplePeriod);
    KALDI_ASSERT(hdr.mSampleSize == hdr2.mSampleSize);
    KALDI_ASSERT(hdr.mSampleKind == hdr2.mSampleKind);
  }

}



template<class Real> static void UnitTestRange() {  // Testing SubMatrix class.

  // this is for matrix-range
  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = (rand()%10) + 10;
    MatrixIndexT dimN = (rand()%10) + 10;

    Matrix<Real> M(dimM, dimN);
    InitRand(M);
    MatrixIndexT dimMStart = rand() % 5;
    MatrixIndexT dimNStart = rand() % 5;

    MatrixIndexT dimMEnd = dimMStart + 1 + (rand()%10); if (dimMEnd > dimM) dimMEnd = dimM;
    MatrixIndexT dimNEnd = dimNStart + 1 + (rand()%10); if (dimNEnd > dimN) dimNEnd = dimN;


    SubMatrix<Real> sub(M, dimMStart, dimMEnd-dimMStart, dimNStart, dimNEnd-dimNStart);

    KALDI_ASSERT(sub.Sum() == M.Range(dimMStart, dimMEnd-dimMStart, dimNStart, dimNEnd-dimNStart).Sum());

    for (MatrixIndexT i = dimMStart;i < dimMEnd;i++)
      for (MatrixIndexT j = dimNStart;j < dimNEnd;j++)
        KALDI_ASSERT(M(i, j) == sub(i-dimMStart, j-dimNStart));

    InitRand(sub);

    KALDI_ASSERT(sub.Sum() == M.Range(dimMStart, dimMEnd-dimMStart, dimNStart, dimNEnd-dimNStart).Sum());

    for (MatrixIndexT i = dimMStart;i < dimMEnd;i++)
      for (MatrixIndexT j = dimNStart;j < dimNEnd;j++)
        KALDI_ASSERT(M(i, j) == sub(i-dimMStart, j-dimNStart));
  }

  // this if for vector-range
  for (int i = 0;i < 5;i++) {
    MatrixIndexT length = (rand()%10) + 10;

    Vector<Real> V(length);
    InitRand(V);
    MatrixIndexT lenStart = rand() % 5;

    MatrixIndexT lenEnd = lenStart + 1 + (rand()%10); if (lenEnd > length) lenEnd = length;

    SubVector<Real> sub(V, lenStart, lenEnd-lenStart);

    KALDI_ASSERT(sub.Sum() == V.Range(lenStart, lenEnd-lenStart).Sum());

    for (MatrixIndexT i = lenStart;i < lenEnd;i++)
        KALDI_ASSERT(V(i) == sub(i-lenStart));

    InitRand(sub);

    KALDI_ASSERT(sub.Sum() == V.Range(lenStart, lenEnd-lenStart).Sum());

    for (MatrixIndexT i = lenStart;i < lenEnd;i++)
        KALDI_ASSERT(V(i) == sub(i-lenStart));
  }
}

template<class Real> static void UnitTestScale() {

  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = (rand()%10) + 10;
    MatrixIndexT dimN = (rand()%10) + 10;

    Matrix<Real> M(dimM, dimN);

    Matrix<Real> N(M);
    float f = (float)((rand()%10)-5);
    M.Scale(f);
    KALDI_ASSERT(M.Sum() == f * N.Sum());

    {
      // now test scale_rows
      M.CopyFromMat(N);  // make same.
      Vector<Real> V(dimM);
      InitRand(V);
      M.MulRowsVec(V);
      for (MatrixIndexT i = 0; i < dimM;i++)
        for (MatrixIndexT j = 0;j < dimN;j++)
          KALDI_ASSERT(M(i, j) - N(i, j)*V(i) < 0.0001);
    }

    {
      // now test scale_cols
      M.CopyFromMat(N);  // make same.
      Vector<Real> V(dimN);
      InitRand(V);
      M.MulColsVec(V);
      for (MatrixIndexT i = 0; i < dimM;i++)
        for (MatrixIndexT j = 0;j < dimN;j++)
          KALDI_ASSERT(M(i, j) - N(i, j)*V(j) < 0.0001);
    }

  }
}

template<class Real> static void UnitTestMul() {


  for (MatrixIndexT x = 0;x<=1;x++) {
    MatrixTransposeType trans = (x == 1 ? kTrans: kNoTrans);
    for (int i = 0;i < 5;i++) {
      float alpha = 1.0, beta =0;
      if (i%3 == 0) beta = 0.5;
      if (i%5 == 0) alpha = 0.7;
      MatrixIndexT dimM = (rand()%10) + 10;
      Vector<Real> v(dimM); InitRand(v);
      TpMatrix<Real> T(dimM); InitRand(T);
      Matrix<Real> M(dimM, dimM);
      if (i%2 == 1)
        M.CopyFromTp(T, trans);
      else
        M.CopyFromTp(T, kNoTrans);
      Vector<Real> v2(v);
      Vector<Real> v3(v);
      v2.AddTpVec(alpha, T, trans, v, beta);
      if (i%2 == 1)
        v3.AddMatVec(alpha, M, kNoTrans, v, beta);
      else
        v3.AddMatVec(alpha, M, trans, v, beta);

      v.AddTpVec(alpha, T, trans, v, beta);
      AssertEqual(v2, v3);
      AssertEqual(v, v2);
    }

    for (int i = 0;i < 5;i++) {
      float alpha = 1.0, beta =0;
      if (i%3 == 0) beta = 0.5;
      if (i%5 == 0) alpha = 0.7;

      MatrixIndexT dimM = (rand()%10) + 10;
      Vector<Real> v(dimM); InitRand(v);
      SpMatrix<Real> T(dimM); InitRand(T);
      Matrix<Real> M(T);
      Vector<Real> v2(dimM);
      Vector<Real> v3(dimM);
      v2.AddSpVec(alpha, T, v, beta);
      v3.AddMatVec(alpha, M, i%2 ? kNoTrans : kTrans, v, beta);
      AssertEqual(v2, v3);
    }
  }
}


template<class Real> static void UnitTestInnerProd() {

  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N);
  InitRand(S);
  Vector<Real> v(N);
  InitRand(v);
  Real prod = VecSpVec(v, S, v);
  Real f2=0.0;
  for (MatrixIndexT i = 0; i < N; i++)
    for (MatrixIndexT j = 0; j < N; j++) {
      f2 += v(i) * v(j) * S(i, j);
    }
  AssertEqual(prod, f2);
}

template<class Real> static void UnitTestScaleDiag() {

  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N);
  InitRand(S);
  SpMatrix<Real> S2(S);
  S.ScaleDiag(0.5);
  for (MatrixIndexT i = 0; i  < N; i++) S2(i, i) *= 0.5;
  AssertEqual(S, S2);
}


template<class Real> static void UnitTestTraceSpSpLower() {

  MatrixIndexT N = 1 + rand() % 10;
  SpMatrix<Real> S(N), T(N);
  InitRand(S);
  InitRand(T);

  SpMatrix<Real> Sfast(S);
  Sfast.Scale(2.0);
  Sfast.ScaleDiag(0.5);

  AssertEqual(TraceSpSp(S, T), TraceSpSpLower(Sfast, T));
}


template<class Real> static void UnitTestSolve() {

  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = (rand()%10) + 10;
    MatrixIndexT dimN = dimM - (rand()%3);  // slightly lower-dim.

    SpMatrix<Real> H(dimM);
    Matrix<Real> M(dimM, dimN); InitRand(M);
    H.AddMat2(1.0, M, kNoTrans, 0.0);  // H = M M^T

    Vector<Real> x(dimM);
    InitRand(x);
    Vector<Real> g(dimM);
    InitRand(g);
    Vector<Real> x2(x);
#if defined(_MSC_VER)
    SolveQuadraticProblem(H, g, &x2, (Real)1.0E4, (Real)1.0E-40, "unknown", true);
#else
    SolveQuadraticProblem(H, g, &x2);
#endif
    KALDI_ASSERT(VecVec(x2, g) -0.5* VecSpVec(x2, H, x2) >=
           VecVec(x, g) -0.5* VecSpVec(x, H, x));
    // Check objf not decreased.
  }


  for (int i = 0;i < 5;i++) {
    MatrixIndexT dimM = (rand()%10) + 10;
    MatrixIndexT dimN = dimM - (rand()%3);  // slightly lower-dim.
    MatrixIndexT dimO = (rand()%10) + 10;

    SpMatrix<Real> Q(dimM), SigmaInv(dimO);
    Matrix<Real> Mtmp(dimM, dimN); InitRand(Mtmp);
    Q.AddMat2(1.0, Mtmp, kNoTrans, 0.0);  // H = M M^T

    Matrix<Real> Ntmp(dimO, dimN); InitRand(Ntmp);
    SigmaInv.AddMat2(1.0, Ntmp, kNoTrans, 0.0);  // H = M M^T

    Matrix<Real> M(dimO, dimM), Y(dimO, dimM);
    InitRand(M); InitRand(Y);

    Matrix<Real> M2(M);

#if defined(_MSC_VER)
    SolveQuadraticMatrixProblem(Q, Y, SigmaInv, &M2, (Real)1.0E4, (Real)1.0E-40, "unknown", true);
#else
    SolveQuadraticMatrixProblem(Q, Y, SigmaInv, &M2);
#endif
    {
      Real a1 = TraceMatSpMat(M2, kTrans, SigmaInv, Y, kNoTrans), a2 = TraceMatSpMatSp(M2, kNoTrans, Q, M2, kTrans, SigmaInv),
          b1 = TraceMatSpMat(M, kTrans, SigmaInv, Y, kNoTrans), b2 = TraceMatSpMatSp(M, kNoTrans, Q, M, kTrans, SigmaInv),
        a3 = a1-0.5*a2, b3 = b1-0.5*b2;
      Real a4;
      {
        SpMatrix<Real> MQM(dimO);
        MQM.AddMat2Sp(1.0, M, kNoTrans, Q, 0.0);
        a4 = TraceSpSp(MQM, SigmaInv);
      }
      KALDI_ASSERT(a3 >= b3);
    }
    // Check objf not decreased.
  }
}

template<class Real> static void UnitTestMaxMin() {

  MatrixIndexT M = 1 + rand() % 10, N = 1 + rand() % 10;
  {
    Vector<Real> v(N);
    InitRand(v);
    Real min = 1.0e+10, max = -1.0e+10;
    for (MatrixIndexT i = 0; i< N; i++) {
      min = std::min(min, v(i));
      max = std::max(max, v(i));
    }
    AssertEqual(min, v.Min());
    AssertEqual(max, v.Max());
  }
  {
    SpMatrix<Real> S(N);
    InitRand(S);
    Real min = 1.0e+10, max = -1.0e+10;
    for (MatrixIndexT i = 0; i< N; i++) {
      for (MatrixIndexT j = 0; j <= i; j++) {
        min = std::min(min, S(i, j));
        max = std::max(max, S(i, j));
      }
    }
    AssertEqual(min, S.Min());
    AssertEqual(max, S.Max());
  }
  {
    Matrix<Real> mat(M, N);
    InitRand(mat);
    Real min = 1.0e+10, max = -1.0e+10;
    for (MatrixIndexT i = 0; i< M; i++) {
      for (MatrixIndexT j = 0; j < N; j++) {
        min = std::min(min, mat(i, j));
        max = std::max(max, mat(i, j));
      }
    }
    AssertEqual(min, mat.Min());
    AssertEqual(max, mat.Max());
  }
}

template<class Real>
static bool approx_equal(Real a, Real b) {
  return  ( std::abs(a-b) <= 1.0e-03 * (std::abs(a)+std::abs(b)));
}

template<class Real> static void UnitTestTrace() {

  for (MatrixIndexT i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10, dimO = 20 + rand()%10, dimP = dimM;
    Matrix<Real> A(dimM, dimN), B(dimN, dimO), C(dimO, dimP);
    InitRand(A);     InitRand(B);     InitRand(C);
    Matrix<Real> AT(dimN, dimM), BT(dimO, dimN), CT(dimP, dimO);
    AT.CopyFromMat(A, kTrans); BT.CopyFromMat(B, kTrans); CT.CopyFromMat(C, kTrans);

    Matrix<Real> AB(dimM, dimO);
    AB.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
    Matrix<Real> BC(dimN, dimP);
    BC.AddMatMat(1.0, B, kNoTrans, C, kNoTrans, 0.0);
    Matrix<Real> ABC(dimM, dimP);
    ABC.AddMatMat(1.0, A, kNoTrans, BC, kNoTrans, 0.0);

    Real
      t1 = TraceMat(ABC),
      t2 = ABC.Trace(),
      t3 = TraceMatMat(A, BC),
      t4 = TraceMatMat(AT, BC, kTrans),
      t5 = TraceMatMat(BC, AT, kTrans),
      t6 = TraceMatMatMat(A, kNoTrans, B, kNoTrans, C, kNoTrans),
      t7 = TraceMatMatMat(AT, kTrans, B, kNoTrans, C, kNoTrans),
      t8 = TraceMatMatMat(AT, kTrans, BT, kTrans, C, kNoTrans),
      t9 = TraceMatMatMat(AT, kTrans, BT, kTrans, CT, kTrans);

    Matrix<Real> ABC1(dimM, dimP);  // tests AddMatMatMat.
    ABC1.AddMatMatMat(1.0, A, kNoTrans, B, kNoTrans, C, kNoTrans, 0.0);
    AssertEqual(ABC, ABC1);

    Matrix<Real> ABC2(dimM, dimP);  // tests AddMatMatMat.
    ABC2.AddMatMatMat(0.25, A, kNoTrans, B, kNoTrans, C, kNoTrans, 0.0);
    ABC2.AddMatMatMat(0.25, AT, kTrans, B, kNoTrans, C, kNoTrans, 2.0);  // the extra 1.0 means another 0.25.
    ABC2.AddMatMatMat(0.125, A, kNoTrans, BT, kTrans, C, kNoTrans, 1.0);
    ABC2.AddMatMatMat(0.125, A, kNoTrans, B, kNoTrans, CT, kTrans, 1.0);
    AssertEqual(ABC, ABC2);

    Real tol = 0.001;
    KALDI_ASSERT((std::abs(t1-t2) < tol) && (std::abs(t2-t3) < tol) && (std::abs(t3-t4) < tol)
      && (std::abs(t4-t5) < tol) && (std::abs(t5-t6) < tol) && (std::abs(t6-t7) < tol)
      && (std::abs(t7-t8) < tol) && (std::abs(t8-t9) < tol));
  }

  for (MatrixIndexT i = 0;i < 5;i++) {
	MatrixIndexT dimM = 20 + rand()%10, dimN = 20 + rand()%10;
    SpMatrix<Real> S(dimM), T(dimN);
    InitRand(S); InitRand(T);
    Matrix<Real> M(dimM, dimN), O(dimM, dimN);
    InitRand(M); InitRand(O);
    Matrix<Real> sM(S), tM(T);

    Real x1 = TraceMatMat(tM, tM);
    Real x2 = TraceSpMat(T, tM);
    KALDI_ASSERT(approx_equal(x1, x2) || fabs(x1-x2) < 0.1);

    Real t1 = TraceMatMatMat(M, kNoTrans, tM, kNoTrans, M, kTrans);
    Real t2 = TraceMatSpMat(M, kNoTrans, T, M, kTrans);
    KALDI_ASSERT(approx_equal(t1, t2) || fabs(t1-12) < 0.1);

    Real u1 = TraceMatSpMatSp(M, kNoTrans, T, O, kTrans, S);
    Real u2 = TraceMatMatMatMat(M, kNoTrans, tM, kNoTrans, O, kTrans, sM, kNoTrans);
    KALDI_ASSERT(approx_equal(u1, u2) || fabs(u1-u2) < 0.1);
  }

}


template<class Real> static void UnitTestComplexFt() {

  // Make sure it inverts properly.
  for (MatrixIndexT d = 0; d < 10; d++) {
    MatrixIndexT N = rand() % 100, twoN = 2*N;
    Vector<Real> v(twoN), w(twoN), x(twoN);
    InitRand(v);
    ComplexFt(v, &w, true);
    ComplexFt(w, &x, false);
    if (N>0) x.Scale(1.0/static_cast<Real>(N));
    AssertEqual(v, x);
  }
}

template<class Real> static void UnitTestDct() {

  // Check that DCT matrix is orthogonal (i.e. M^T M = I);
  for (MatrixIndexT i = 0; i < 10; i++) {
    MatrixIndexT N = 1 + rand() % 10;
    Matrix<Real> M(N, N);
    ComputeDctMatrix(&M);
    Matrix<Real> I(N, N);
    I.AddMatMat(1.0, M, kTrans, M, kNoTrans, 0.0);
    KALDI_ASSERT(I.IsUnit());
  }

}
template<class Real> static void UnitTestComplexFft() {

  // Make sure it inverts properly.
  for (MatrixIndexT N_ = 0; N_ < 100; N_+=3) {
    MatrixIndexT N = N_;
    if (N>=95) {
      N = ( rand() % 150);
      N = N*N;  // big number.
    }

    MatrixIndexT twoN = 2*N;
    Vector<Real> v(twoN), w_base(twoN), w_alg(twoN), x_base(twoN), x_alg(twoN);

    InitRand(v);

    if (N< 100) ComplexFt(v, &w_base, true);
    w_alg.CopyFromVec(v);
    ComplexFft(&w_alg, true);
    if (N< 100) AssertEqual(w_base, w_alg, 0.01*N);

    if (N< 100) ComplexFt(w_base, &x_base, false);
    x_alg.CopyFromVec(w_alg);
    ComplexFft(&x_alg, false);

    if (N< 100) AssertEqual(x_base, x_alg, 0.01*N);
    x_alg.Scale(1.0/N);
    AssertEqual(v, x_alg, 0.001*N);
  }
}


template<class Real> static void UnitTestSplitRadixComplexFft() {

  // Make sure it inverts properly.
  for (MatrixIndexT N_ = 0; N_ < 30; N_+=3) {
    MatrixIndexT logn = 1 + rand() % 10;
    MatrixIndexT N = 1 << logn;

    MatrixIndexT twoN = 2*N;
    SplitRadixComplexFft<Real> srfft(N);
    for (MatrixIndexT p = 0; p < 3; p++) {
      Vector<Real> v(twoN), w_base(twoN), w_alg(twoN), x_base(twoN), x_alg(twoN);

      InitRand(v);

      if (N< 100) ComplexFt(v, &w_base, true);
      w_alg.CopyFromVec(v);
      srfft.Compute(w_alg.Data(), true);

      if (N< 100) AssertEqual(w_base, w_alg, 0.01*N);

      if (N< 100) ComplexFt(w_base, &x_base, false);
      x_alg.CopyFromVec(w_alg);
      srfft.Compute(x_alg.Data(), false);

      if (N< 100) AssertEqual(x_base, x_alg, 0.01*N);
      x_alg.Scale(1.0/N);
      AssertEqual(v, x_alg, 0.001*N);
    }
  }
}



template<class Real> static void UnitTestTranspose() {

  Matrix<Real> M(rand() % 5 + 1, rand() % 10 + 1);
  InitRand(M);
  Matrix<Real> N(M, kTrans);
  N.Transpose();
  AssertEqual(M, N);
}
template<class Real> static void UnitTestComplexFft2() {

  // Make sure it inverts properly.
  for (MatrixIndexT pos = 0; pos < 10; pos++) {
    for (MatrixIndexT N_ = 2; N_ < 15; N_+=2) {
      if ( pos < N_) {
        MatrixIndexT N = N_;
        Vector<Real> v(N), vorig(N), v2(N);
        v(pos)  = 1.0;
        vorig.CopyFromVec(v);
        // std::cout << "Original v:\n" << v;
        ComplexFft(&v, true);
        // std::cout << "one fft:\n" << v;
        ComplexFt(vorig, &v2, true);
        // std::cout << "one fft[baseline]:\n" << v2;
        if (!ApproxEqual(v, v2) ) {
          ComplexFft(&vorig, true);
          KALDI_ASSERT(0);
        }
        ComplexFft(&v, false);
        // std::cout << "one more:\n" << v;
        v.Scale(1.0/(N/2));
        if (!ApproxEqual(v, vorig)) {
          ComplexFft(&vorig, true);
          KALDI_ASSERT(0);
        }// AssertEqual(v, vorig);
      }
    }
  }
}


template<class Real> static void UnitTestSplitRadixComplexFft2() {

  // Make sure it inverts properly.
  for (MatrixIndexT p = 0; p < 30; p++) {
    MatrixIndexT logn = 1 + rand() % 10;
    MatrixIndexT N = 1 << logn;
    SplitRadixComplexFft<Real> srfft(N);
    for (MatrixIndexT q = 0; q < 3; q++) {
      Vector<Real> v(N*2), vorig(N*2);
      InitRand(v);
      vorig.CopyFromVec(v);
      srfft.Compute(v.Data(), true);  // forward
      srfft.Compute(v.Data(), false);  // backward
      v.Scale(1.0/N);
      KALDI_ASSERT(ApproxEqual(v, vorig));
    }
  }
}


template<class Real> static void UnitTestRealFft() {

  // First, test RealFftInefficient.
  for (int N_ = 2; N_ < 100; N_ += 6) {
    int N = N_;
    if (N >90) N *= rand() % 60;
    Vector<Real> v(N), w(N), x(N), y(N);
    InitRand(v);
    w.CopyFromVec(v);
    RealFftInefficient(&w, true);
    y.CopyFromVec(v);
    RealFft(&y, true);  // test efficient one.
    // std::cout <<"v = "<<v;
    // std::cout << "Inefficient real fft of v is: "<< w;
    // std::cout << "Efficient real fft of v is: "<< y;
    AssertEqual(w, y, 0.01*N);
    x.CopyFromVec(w);
    RealFftInefficient(&x, false);
    RealFft(&y, false);
    // std::cout << "Inefficient real fft of v twice is: "<< x;
    if (N != 0) x.Scale(1.0/N);
    if (N != 0) y.Scale(1.0/N);
    AssertEqual(v, x, 0.001*N);
    AssertEqual(v, y, 0.001*N);  // ?
  }
}


template<class Real> static void UnitTestSplitRadixRealFft() {

  for (int p = 0; p < 30; p++) {
    int logn = 2 + rand() % 11,
        N = 1 << logn;

    SplitRadixRealFft<Real> srfft(N);
    for (int q = 0; q < 3; q++) {
      Vector<Real> v(N), w(N), x(N), y(N);
      InitRand(v);
      w.CopyFromVec(v);
      RealFftInefficient(&w, true);
      y.CopyFromVec(v);
      srfft.Compute(y.Data(), true);  // test efficient one.
      // std::cout <<"v = "<<v;
      // std::cout << "Inefficient real fft of v is: "<< w;
      // std::cout << "Efficient real fft of v is: "<< y;
      AssertEqual(w, y, 0.01*N);
      x.CopyFromVec(w);
      RealFftInefficient(&x, false);
      srfft.Compute(y.Data(), false);
      // std::cout << "Inefficient real fft of v twice is: "<< x;
      x.Scale(1.0/N);
      y.Scale(1.0/N);
      AssertEqual(v, x, 0.001*N);
      AssertEqual(v, y, 0.001*N);  // ?
    }
  }
}



template<class Real> static void UnitTestRealFftSpeed() {

  // First, test RealFftInefficient.
  std::cout << "starting. ";
  int sz = 512;  // fairly typical size.
  for (int i = 0; i < 3000; i++) {
    if (i % 1000 == 0) std::cout << "done 1000 [ == ten seconds of speech]\n";
    Vector<Real> v(sz);
    RealFft(&v, true);
  }
}

template<class Real> static void UnitTestSplitRadixRealFftSpeed() {

  // First, test RealFftInefficient.
  std::cout << "starting. ";
  int sz = 512;  // fairly typical size.
  SplitRadixRealFft<Real> srfft(sz);
  for (int i = 0; i < 6000; i++) {
    if (i % 1000 == 0) std::cout << "done 1000 [ == ten seconds of speech, split-radix]\n";
    Vector<Real> v(sz);
    srfft.Compute(v.Data(), true);
  }
}

template<class Real>
void UnitTestComplexPower() {
  // This tests a not-really-public function that's used in Matrix::Power().

  for (int32 i = 0; i < 10; i++) {
    Real power = RandGauss();
    Real x = 2.0, y = 0.0;
    bool ans = AttemptComplexPower(&x, &y, power);
    KALDI_ASSERT(ans);
    AssertEqual(std::pow(static_cast<Real>(2.0), power), x);
    AssertEqual(y, 0.0);
  }
  {
    Real x, y;
    x = 0.5; y = -0.3;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(2.21));
    KALDI_ASSERT(ans);
    ans = AttemptComplexPower(&x, &y, static_cast<Real>(1.0/2.21));
    KALDI_ASSERT(ans);
    AssertEqual(x, 0.5);
    AssertEqual(y, -0.3);
  }
  {
    Real x, y;
    x = 0.5; y = -0.3;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(2.0));
    KALDI_ASSERT(ans);
    AssertEqual(x, 0.5*0.5 - 0.3*0.3);
    AssertEqual(y, -0.3*0.5*2.0);
  }

  {
    Real x, y;
    x = 1.0/std::sqrt(2.0); y = -1.0/std::sqrt(2.0);
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(-1.0));
    KALDI_ASSERT(ans);
    AssertEqual(x, 1.0/std::sqrt(2.0));
    AssertEqual(y, 1.0/std::sqrt(2.0));
  }

  {
    Real x, y;
    x = 0.0; y = 0.0;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(-2.0));
    KALDI_ASSERT(!ans);  // zero; negative pow.
  }
  {
    Real x, y;
    x = -2.0; y = 0.0;
    bool ans = AttemptComplexPower(&x, &y, static_cast<Real>(1.5));
    KALDI_ASSERT(!ans);  // negative real case
  }
}
template<class Real>
void UnitTestNonsymmetricPower() {

  for (int iter = 0; iter < 30; iter++) {
    MatrixIndexT dimM = 1 + rand() % 20;
    Matrix<Real> M(dimM, dimM);
    InitRand(M);

    Matrix<Real> MM(dimM, dimM);
    MM.AddMatMat(1.0, M, kNoTrans, M, kNoTrans, 0.0);  // MM = M M.
    Matrix<Real> MMMM(dimM, dimM);
    MMMM.AddMatMat(1.0, MM, kNoTrans, MM, kNoTrans, 0.0);

    Matrix<Real> MM2(MM);
    bool b = MM2.Power(1.0);
    KALDI_ASSERT(b);
    AssertEqual(MM2, MM);
    Matrix<Real> MMMM2(MM);
    b = MMMM2.Power(2.0);
    KALDI_ASSERT(b);
    AssertEqual(MMMM2, MMMM);
  }
  for (int iter = 0; iter < 30; iter++) {
    MatrixIndexT dimM = 1 + rand() % 20;
    Matrix<Real> M(dimM, dimM);
    InitRand(M);

    Matrix<Real> MM(dimM, dimM);
    MM.AddMatMat(1.0, M, kNoTrans, M, kNoTrans, 0.0);  // MM = M M.
    // This ensures there are no real, negative eigenvalues.

    Matrix<Real> MMMM(dimM, dimM);
    MMMM.AddMatMat(1.0, MM, kNoTrans, MM, kNoTrans, 0.0);

    Matrix<Real> MM2(M);
    if (!MM2.Power(2.0)) {  // possibly had negative eigenvalues
      std::cout << "Could not take matrix to power (not an error)\n";
    } else {
      AssertEqual(MM2, MM);
    }
    Matrix<Real> MMMM2(M);
    if (!MMMM2.Power(4.0)) {  // possibly had negative eigenvalues
      std::cout << "Could not take matrix to power (not an error)\n";
    } else {
      AssertEqual(MMMM2, MMMM);
    }
    Matrix<Real> MMMM3(MM);
    if (!MMMM3.Power(2.0)) {
      KALDI_ERR << "Could not take matrix to power (should have been able to)\n";
    } else {
      AssertEqual(MMMM3, MMMM);
    }

    Matrix<Real> MM4(MM);
    if (!MM4.Power(-1.0))
      KALDI_ERR << "Could not take matrix to power (should have been able to)\n";
    MM4.Invert();
    AssertEqual(MM4, MM);
  }
}

void UnitTestAddVecCross() {

  Vector<float> v(5);
  InitRand(v);
  Vector<double> w(5);
  InitRand(w);

  Vector<float> wf(w);

  for (MatrixIndexT i = 0; i < 2; i++) {
    float f = 1.0;
    if (i == 0) f = 2.0;

    {
      Vector<float> sum1(5);
      Vector<double> sum2(5);
      Vector<float> sum3(5);
      sum1.AddVec(f, v); sum1.AddVec(f, w);
      sum2.AddVec(f, v); sum2.AddVec(f, w);
      sum3.AddVec(f, v); sum3.AddVec(f, wf);
      Vector<float> sum2b(sum2);
      AssertEqual(sum1, sum2b);
      AssertEqual(sum1, sum3);
    }

    {
      Vector<float> sum1(5);
      Vector<double> sum2(5);
      Vector<float> sum3(5);
      sum1.AddVec2(f, v); sum1.AddVec2(f, w);
      sum2.AddVec2(f, v); sum2.AddVec2(f, w);
      sum3.AddVec2(f, v); sum3.AddVec2(f, wf);
      Vector<float> sum2b(sum2);
      AssertEqual(sum1, sum2b);
      AssertEqual(sum1, sum3);
    }
  }
}

template<class Real> static void UnitTestMatrixExponential() {

  for (int32 p = 0; p < 10; p++) {
    MatrixIndexT dim = 1 + rand() % 5;
    Matrix<Real> M(dim, dim);
    InitRand(M);
    {  // work out largest eig.
      Real largest_eig = 0.0;
      Vector<Real> real_eigs(dim), imag_eigs(dim);
      M.Eig(NULL, &real_eigs, &imag_eigs);
      for (MatrixIndexT i = 0; i < dim; i++) {
        Real abs_eig =
            std::sqrt(real_eigs(i)*real_eigs(i) + imag_eigs(i)*imag_eigs(i));
        largest_eig = std::max(largest_eig, abs_eig);
      }
      if (largest_eig > 0.5) {  // limit largest eig to 0.5,
        // so Taylor expansion will converge.
        M.Scale(0.5 / largest_eig);
      }
    }

    Matrix<Real> expM(dim, dim);
    Matrix<Real> cur_pow(dim, dim);
    cur_pow.SetUnit();
    Real i_factorial = 1.0;
    for (MatrixIndexT i = 0; i < 52; i++) {  // since max-eig = 0.5 and 2**52 == dbl eps.
      if (i > 0) i_factorial *= i;
      expM.AddMat(1.0/i_factorial, cur_pow);
      Matrix<Real> tmp(dim, dim);
      tmp.AddMatMat(1.0, cur_pow, kNoTrans, M, kNoTrans, 0.0);
      cur_pow.CopyFromMat(tmp);
    }
    Matrix<Real> expM2(dim, dim);
    MatrixExponential<Real> mexp;
    mexp.Compute(M, &expM2);
    AssertEqual(expM, expM2);
  }
}


static void UnitTestMatrixExponentialBackprop() {
  for (int32 p = 0; p < 10; p++) {
    MatrixIndexT dim = 1 + rand() % 5;
    // f is tr(N^T exp(M)).  backpropagating derivative
    // of this function.
    Matrix<double> M(dim, dim), N(dim, dim), delta(dim, dim);
    InitRand(M);
    // { SpMatrix<double> tmp(dim); InitRand(tmp); M.CopyFromSp(tmp); }
    InitRand(N);
    InitRand(delta);
    // { SpMatrix<double> tmp(dim); InitRand(tmp); delta.CopyFromSp(tmp); }
    delta.Scale(0.00001);


    Matrix<double> expM(dim, dim);
    MatrixExponential<double> mexp;
    mexp.Compute(M, &expM);

    Matrix<double> expM2(dim, dim);
    {
      Matrix<double> M2(M);
      M2.AddMat(1.0, delta, kNoTrans);
      MatrixExponential<double> mexp2;
      mexp2.Compute(M2, &expM2);
    }
    double f_before = TraceMatMat(expM, N, kTrans);
    double f_after  = TraceMatMat(expM2, N, kTrans);

    Matrix<double> Mdash(dim, dim);
    mexp.Backprop(N, &Mdash);
    // now Mdash should be df/dM

    double f_diff = f_after - f_before;  // computed using method of differnces
    double f_diff2 = TraceMatMat(Mdash, delta, kTrans);  // computed "analytically"
    std::cout << "f_diff = " << f_diff << "\n";
    std::cout << "f_diff2 = " << f_diff2 << "\n";

    AssertEqual(f_diff, f_diff2);
  }
}
template<class Real>
static void UnitTestPca() {
  // We'll test that we can exactly reconstruct the vectors, if
  // the PCA dim is <= the "real" dim that the vectors live in.
  for (int32 i = 0; i < 10; i++) {
    int32 true_dim = 5 + rand() % 5,
        feat_dim = true_dim + rand() % 5,
        num_points = true_dim + rand() % 5,
        G = std::min(feat_dim,
                     std::min(num_points,
                              static_cast<int32>(true_dim + rand() % 5)));

    Matrix<Real> Proj(feat_dim, true_dim);
    InitRand(Proj);
    Matrix<Real> true_X(num_points, true_dim);
    InitRand(true_X);
    Matrix<Real> X(num_points, feat_dim);
    X.AddMatMat(1.0, true_X, kNoTrans, Proj, kTrans, 0.0);

    Matrix<Real> U(G, feat_dim);
    Matrix<Real> A(num_points, G);
    ComputePca(X, &U, &A, true);
    Matrix<Real> X2(num_points, feat_dim);
    X2.AddMatMat(1.0, A, kNoTrans, U, kNoTrans, 0.0);
    // Check reproduction.
    AssertEqual(X, X2, 0.01);
    // Check basis is orthogonal.
    Matrix<Real> tmp(G, G);
    tmp.AddMatMat(1.0, U, kNoTrans, U, kTrans, 0.0);
    KALDI_ASSERT(tmp.IsUnit(0.01));
  }
}


template<class Real> static void MatrixUnitTest() {
  // UnitTestSvdBad<Real>(); // test bug in Jama SVD code.
  UnitTestResize<Real>();
  UnitTestMatrixExponentialBackprop();
  UnitTestMatrixExponential<Real>();
  UnitTestNonsymmetricPower<Real>();
  UnitTestEigSymmetric<Real>();
  KALDI_LOG << " Point A";
  UnitTestComplexPower<Real>();
  UnitTestEig<Real>();
  // commenting these out for now-- they test the speed, but take a while.
  // UnitTestSplitRadixRealFftSpeed<Real>();
  // UnitTestRealFftSpeed<Real>();   // won't exit!/
  UnitTestComplexFt<Real>();
  KALDI_LOG << " Point B";
  UnitTestComplexFft2<Real>();
  UnitTestComplexFft<Real>();
  UnitTestSplitRadixComplexFft<Real>();
  UnitTestSplitRadixComplexFft2<Real>();
  UnitTestDct<Real>();
  UnitTestRealFft<Real>();
      KALDI_LOG << " Point C";
  UnitTestSplitRadixRealFft<Real>();
  UnitTestSvd<Real>();
  UnitTestSvdNodestroy<Real>();
  UnitTestSvdJustvec<Real>();

  UnitTestSpInvert<Real>();
      KALDI_LOG << " Point D";
  UnitTestTpInvert<Real>();
  UnitTestIo<Real>();
  UnitTestIoOld<Real>();
  UnitTestIoCross<Real>();
  UnitTestHtkIo<Real>();
  UnitTestScale<Real>();
  UnitTestTrace<Real>();
      KALDI_LOG << " Point E";
  CholeskyUnitTestTr<Real>();
  UnitTestAxpy<Real>();
  UnitTestSimple<Real>();
  UnitTestMmul<Real>();
  UnitTestMmulSym<Real>();
  UnitTestVecmul<Real>();
  UnitTestInverse<Real>();
  UnitTestMulElements<Real>();
  UnitTestDotprod<Real>();
  // UnitTestSvdVariants<Real>();
  UnitTestPower<Real>();
  UnitTestDeterminant<Real>();
        KALDI_LOG << " Point F";
  UnitTestDeterminantSign<Real>();
  UnitTestSger<Real>();
  UnitTestPca<Real>();
  UnitTestTraceProduct<Real>();
  UnitTestTransposeScatter<Real>();
  UnitTestRankNUpdate<Real>();
  UnitTestSherman<Real>();
  UnitTestLimitCondInvert<Real>();
        KALDI_LOG << " Point G";
  UnitTestFloorChol<Real>();
  UnitTestFloorUnit<Real>();
  UnitTestLimitCond<Real>();
  UnitTestMat2Vec<Real>();
  UnitTestSpLogExp<Real>();
          KALDI_LOG << " Point H";
  UnitTestSpliceRows<Real>();
  UnitTestAddSp<Real>();
  UnitTestRemoveRow<Real>();
  UnitTestRow<Real>();
  UnitTestSubvector<Real>();
  UnitTestRange<Real>();
  UnitTestSimpleForVec<Real>();
  UnitTestSimpleForMat<Real>();
  UnitTestNorm<Real>();
  UnitTestMul<Real>();
          KALDI_LOG << " Point I";
  UnitTestSolve<Real>();
  UnitTestMaxMin<Real>();
  UnitTestInnerProd<Real>();
  UnitTestScaleDiag<Real>();
            KALDI_LOG << " Point J";
  UnitTestTraceSpSpLower<Real>();
  UnitTestTranspose<Real>();
  UnitTestAddVecCross();
}

}


int main()
{
  kaldi::MatrixUnitTest<float>();
  kaldi::MatrixUnitTest<double>();
  std::cout << "Tests succeeded.\n";
}


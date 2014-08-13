// cudamatrix/cuda-test.cc
//
//
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "base/kaldi-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "cudamatrix/cu-vector.h"
#include <numeric>
#include <time.h>

namespace kaldi {

/*
 * INITIALIZERS
 */ 
template<typename Real>
static void InitRand(SpMatrix<Real> *M) {
  do {
    for (MatrixIndexT i = 0; i < M->NumRows(); i++) {
      for (MatrixIndexT j = 0; j <= i; j++ ) {
	(*M)(i,j) = RandGauss();
      }
    }
  } while (M->NumRows() != 0 && M->Cond() > 100);
}

template<typename Real>
static void InitRand(VectorBase<Real> *v) {
  for (MatrixIndexT i = 0; i < v->Dim(); i++) {
    (*v)(i) = RandGauss();
  }
}

template<typename Real>
static void UnitTestSetZeroAboveDiag() {
  for (MatrixIndexT i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Matrix<Real> A(dim,dim);
    A.SetRandn();
    CuMatrix<Real> B(A);

    B.SetZeroAboveDiag();

    Real sum = 0.0;
    for (MatrixIndexT i = 0; i < dim; i++) {
      for (MatrixIndexT j = i + 1; j < dim; j++)
        sum += A(i,j);
    }
        
    KALDI_LOG << "the upper diaganoal sum for A is : " << sum;
    B.CopyToMat(&A);
    sum = 0.0;
    for (MatrixIndexT i = 0; i < dim; i++) {
      for (MatrixIndexT j = i + 1; j < dim; j++)
        sum += A(i,j);
    }
    KALDI_LOG << "the upper diaganoal sum for B is : " << sum;
  }
}


template<typename Real> static void UnitTestCholesky() {
  for (MatrixIndexT iter = 0; iter < 3; iter++) {
    MatrixIndexT dim = 300 + Rand() %  200;
    // set dimension
    // computing the matrix for cholesky input
    // CuMatrix is cuda matrix class while Matrix is cpu matrix class
    CuMatrix<Real> A(dim, dim);
    Matrix<Real> B(dim, dim);
    Vector<Real> C(dim);
    for (MatrixIndexT i = 0; i < dim; i++) {
      B(i, i) = 1;
      C(i) = 1 + Rand() % 4;
    }
    B.AddVecVec(1.0, C, C);
    // copy the matrix to cudamatrix object
    A.CopyFromMat(B);
    A.CopyToMat(&B);
    //KALDI_LOG << B;
    // doing cholesky
    A.Cholesky();

    Matrix<Real> D(dim,dim);
    A.CopyToMat(&D);
    
    //KALDI_LOG << "D is: " << D;
    Matrix<Real> E(dim,dim);
    E.AddMatMat(1.0, D, kNoTrans, D, kTrans, 0.0);
    // check if the D'D is equal to B or not!
    AssertEqual(B, E);
  }
}

template<typename Real> static void UnitTestTrace() {
  for (MatrixIndexT iter = 1; iter < 18; iter++) {
    MatrixIndexT dim = iter;
    KALDI_LOG << "dim is : " << iter;
    SpMatrix<Real> A(dim);
    A.SetRandn();
    CuSpMatrix<Real> B(A);
    KALDI_LOG << "cpu trace is : " << A.Trace();
    KALDI_LOG << "gpu trace is : " << B.Trace();
  }
  /*
  Vector<Real> tim(100);
  Vector<Real> d(100);
  for (MatrixIndexT iter = 0; iter < 100; iter++) {
    MatrixIndexT dim = 10000 + Rand() % 400;
    Matrix<Real> A(dim,dim);
    A.SetRandn();
    CuMatrix<Real> B(A);
    CuSpMatrix<Real> C(B,kTakeLower);
    clock_t t1 = clock();
    tim(iter) = C.Trace();
    clock_t t2 = clock();
    //tim(iter) = t2 - t1;
    d(iter) = dim;
    KALDI_LOG << tim(iter) << iter;
    KALDI_LOG << d(iter) << iter;
  }
  KALDI_LOG << "tim is " << tim;
  KALDI_LOG << "dim is " << d;
  */
}

template<typename Real> static void UnitInvert() {
  //MatrixIndexT dim = 15 + Rand() %  40;;
  MatrixIndexT dim = 8;
  CuMatrix<Real> A(dim,dim);
  Matrix<Real> B(dim,dim);
  Vector<Real> C(dim);
  for (MatrixIndexT i = 0; i < dim; i++) {
    B(i,i) = 1;
    C(i) = i + 1;
  }
  B.AddVecVec(1.0,C,C);
  CuMatrix<Real> tmp(dim,dim);
  A.CopyFromMat(B);
  //A.Cholesky();
  A.CopyToMat(&B);
  KALDI_LOG << "B is : ";
  KALDI_LOG << B;
  A.SymInvertPosDef();
  Matrix<Real> D(dim,dim);
  A.CopyToMat(&D);
  KALDI_LOG << "D is : ";
  KALDI_LOG << D;
  Matrix<Real> X(dim,dim);
  X.AddMatMat(1,B,kNoTrans,D,kNoTrans,0);
  KALDI_LOG << X;
  //for (MatrixIndexT i = 0; i < dim; i++) {
  //  for (MatrixIndexT j = i+1; j < dim; j++)
  //    D(i,j) = 0;
  //}
  //Matrix<Real> E(dim,dim);
  //E.AddMatMat(1,D,kNoTrans,D,kTrans,0);
  //AssertEqual(B,E);
}

template<typename Real> static void UnitTestInvert() {
  for (MatrixIndexT iter = 0; iter < 3; iter++) {
    MatrixIndexT dim = 500 + Rand() % 400;
    
    KALDI_LOG << "dim is : ";
    KALDI_LOG << dim;
    CuMatrix<Real> A(dim,dim);
    Matrix<Real> B(dim,dim);
    Vector<Real> C(dim);
    for (MatrixIndexT i = 0; i < dim; i++) {
      B(i,i) = 1;
      C(i) = (i/(1.0*dim)) + 1;
    }
    Matrix<Real> Identity(B);
    B.AddVecVec(1.0, C, C);
    // Now we have a positive-definite B (inversion would
    // fail if it were not positive definite). 

    A.CopyFromMat(B);
    
    A.SymInvertPosDef();
    Matrix<Real> D(dim,dim);
    A.CopyToMat(&D);
    
    Matrix<Real> X(dim,dim);
    X.AddMatMat(1.0, B, kNoTrans, D, kNoTrans, 0.0);
    // KALDI_LOG << "X is (should be identity): " << X;
    AssertEqual(Identity, X, (sizeof(Real) == 4 ? 0.1 : 0.001));
  }
}

template<typename Real> static void UnitTestConstructor() {
  MatrixIndexT dim = 8;
  CuMatrix<Real> A(dim,dim);
  Matrix<Real> B(dim,dim);
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <=i; j++)
      B(i,j) = i+j;
    for (MatrixIndexT j = i+1; j < dim; j++)
      B(i,j) = i+j+4;
  }
  KALDI_LOG << "A is : ";
  KALDI_LOG << B;
  A.CopyFromMat(B);
  //CuSpMatrix<Real> C(dim);
  //C.CopyFromMat(A,kTakeLower);
  CuSpMatrix<Real> C(A, kTakeLower);
  SpMatrix<Real> D(dim);
  C.CopyToSp(&D);
  KALDI_LOG << "C is : ";
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }  
}

template<typename Real> static void UnitTestCopySp() {
  // Checking that the various versions of copying                                 
  // matrix to SpMatrix work the same in the symmetric case.                         
  for (MatrixIndexT iter = 0;iter < 5;iter++) {
    int32 dim = 5 + Rand() %  10;
    SpMatrix<Real> A(dim), B(dim);
    A.SetRandn();
    Matrix<Real> C(A);
    //CuMatrix<Real> D(C);
    
    {
      CuMatrix<Real> D2(dim,dim);
      D2.CopyFromMat(C);
      KALDI_LOG << "D2 is " << D2;
      CuSpMatrix<Real> E(D2.NumRows(), kUndefined);
      KALDI_LOG << "D2 is " << D2;
      E.CopyFromMat(D2, kTakeLower);
      KALDI_LOG << "D2 is " << D2;
    }
    
    CuMatrix<Real> D(dim,dim);
    D.CopyFromMat(C);
    KALDI_LOG << "D stride is : " << D.Stride() <<'\n';
    
    CuSpMatrix<Real> E(D,kTakeLower);
    ///CuSpMatrix<Real> E(dim);
    //E.CopyFromMat(D,kTakeLower);
    /*
    KALDI_LOG << D.NumRows();
    //E.CopyFromMat(D, kTakeMean);
    //E(D, kTakeMean);
    //KALDI_LOG << E.NumRows();

    E.CopyToMat(&B);
    AssertEqual(A, B);
    B.SetZero();
    //E.CopyFromMat(D, kTakeLower);
    CuSpMatrix<Real> F(D,kTakeLower);
    //F(D, kTakeLower);
    F.CopyToMat(&B);
    AssertEqual(A, B);
    B.SetZero();
    //E.CopyFromMat(D, kTakeUpper);
    //E(D, kTakeUpper);
    CuSpMatrix<Real> G(D, kTakeUpper);
    G.CopyToMat(&B);
    AssertEqual(A, B);
    */  
  }
  
}

template<typename Real> static void UnitTestCopyFromMat() {
  MatrixIndexT dim = 8;
  CuMatrix<Real> A(dim,dim);
  Matrix<Real> B(dim,dim);
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <=i; j++)
      B(i,j) = i+j;
    for (MatrixIndexT j = i+1; j < dim; j++)
      B(i,j) = i+j+4;
  }
  KALDI_LOG << "A is : ";
  KALDI_LOG << B;
  A.CopyFromMat(B);
  CuSpMatrix<Real> C(dim);
  C.CopyFromMat(A,kTakeLower);
  SpMatrix<Real> D(dim);
  C.CopyToSp(&D);
  KALDI_LOG << "C is : ";
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }
  C.CopyFromMat(A,kTakeUpper);
  C.CopyToSp(&D);
  KALDI_LOG << "C is : ";
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }
  
  C.CopyFromMat(A,kTakeMean);
  C.CopyToSp(&D);
  KALDI_LOG << "C is : ";
  for (MatrixIndexT i = 0; i < dim; i++) {
    for (MatrixIndexT j = 0; j <= i; j++)
      std::cout << D(i,j) << " ";
    std::cout << '\n';
  }
  
  //KALDI_LOG << D;
}

template<typename Real> static void UnitTestMatrix() {
  //operator()
  for (MatrixIndexT iter = 0; iter < 2; iter++) {
    int32 dim1 = 6 + Rand() % 10;
    int32 dim2 = 8 + Rand() % 10;
    Matrix<Real> A(dim1,dim2);
    A.SetRandn();
    CuMatrix<Real> B(A);
    KALDI_ASSERT(A(3, 4) == B(3, 4));
    B(3, 4) = 2.0;
    A(3, 4) = B(3, 4);
    KALDI_ASSERT(A(3, 4) == B(3, 4));

    SpMatrix<Real> As(dim1);
    CuSpMatrix<Real> Bs(As);
    KALDI_ASSERT(As(3, 4) == Bs(3, 4));
    Bs(3, 4) = 2.0;
    if (Rand() % 2 == 0)
      As(3, 4) = Bs(3, 4);
    else
      As(3, 4) = (const_cast<const CuSpMatrix<Real>&>(Bs))(3, 4);
    
    KALDI_ASSERT(As(3, 4) == Bs(3, 4));

    Vector<Real> v(dim1);
    CuVector<Real> w(v);
    KALDI_ASSERT(w(2) == v(2));
    w(2) = 3.0;
    v(2) = w(2);
    KALDI_ASSERT(w(2) == v(2));
  }

  //SetRandn
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim1 = 15 + Rand() % 10;
    int32 dim2 = dim1;//10 + Rand() % 14;
    //KALDI_LOG << "dimension is " << dim1
    //          << " " << dim2 << '\n';
    CuMatrix<Real> A(dim1,dim2);
    A.SetRandn();
    Matrix<Real> A1(dim1,dim2);
    A.CopyToMat(&A1);
    //KALDI_LOG << "gpu sum is: " << A.Sum();
    //KALDI_LOG << "cpu sum is: " << A1.Sum();
  }
}

template<typename Real> static void UnitTestMulTp() {
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 1 + Rand() % 30;
    Vector<Real> v(dim);
    v.SetRandn();
    TpMatrix<Real> M(dim);
    M.SetRandn();
    CuVector<Real> cv(v);
    CuTpMatrix<Real> cM(M);
    
    Vector<Real> v2(dim);
    cv.CopyToVec(&v2);
    AssertEqual(v, v2);
    v.MulTp(M, iter % 2 == 0 ? kTrans:kNoTrans);
    cv.MulTp(cM, iter % 2 == 0 ? kTrans:kNoTrans);
    cv.CopyToVec(&v2);
    // KALDI_LOG << "v is " << v << ", v2 is " << v2;
    AssertEqual(v, v2);
  }
}

template<typename Real> static void UnitTestVector() {
  // Scale
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 24 + Rand() % 10;
    Vector<Real> A(dim);
    A.SetRandn();
    CuVector<Real> B(A);
    Vector<Real> C(dim);
    Real r = 1.43;
    B.Scale(r);
    B.CopyToVec(&C);
    A.Scale(r);
    //KALDI_LOG << A;
    //KALDI_LOG << (A.Scale(r));
    //KALDI_LOG << C;
    AssertEqual(A, C);
  }
  
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + Rand() % 10;
    CuVector<Real> A(dim);
    CuVector<Real> B(dim);
    Vector<Real> A1(dim);
    Vector<Real> B1(dim);
    A.SetRandn();
    B.SetRandn();
    A.CopyToVec(&A1);
    B.CopyToVec(&B1);
    A.MulElements(B);
    A1.MulElements(B1);
    Vector<Real> A2(dim);
    A.CopyToVec(&A2);
    AssertEqual(A1,A2);
  }
  /*
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 72;
    CuVector<Real> A(dim);
    Vector<Real> A1(dim);
    CuMatrix<Real> B(9,8);
    Matrix<Real> B1(9,8);
    B.SetRandn();
    B.CopyToMat(&B1);
    A.CopyRowsFromMat(B);
    A1.CopyRowsFromMat(B1);
    Vector<Real> A2(dim);
    A.CopyToVec(&A2);
    AssertEqual(A1,A2);
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + Rand() % 10;
    CuVector<Real> A(dim);
    A.SetRandn();
    Vector<Real> A1(dim);
    A.CopyToVec(&A1);
    KALDI_LOG << "cpu min is : " << A1.Min();
    KALDI_LOG << "gpu min is : " << A.Min();    
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + Rand() % 10;
    CuVector<Real> A(dim);
    A.SetRandn();
    Vector<Real> A1(dim);
    A.CopyToVec(&A1);
    CuVector<Real> B(dim);
    B.SetRandn();
    Vector<Real> B1(dim);
    B.CopyToVec(&B1);
    CuVector<Real> C(dim);
    C.SetRandn();
    Vector<Real> C1(dim);
    C.CopyToVec(&C1);
    Real alpha = 2;
    Real beta = 3;
    A.AddVecVec(alpha, B, C, beta);
    A1.AddVecVec(alpha,B1,C1,beta);
    Vector<Real> D(dim);
    A.CopyToVec(&D);
    AssertEqual(D,A1);
  }
  
  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim1 = 15 + Rand() % 10;
    int32 dim2 = 10 + Rand() % 10;
    Matrix<Real> A(dim1,dim2);
    for (MatrixIndexT i = 0; i < dim1; i++) {
      for (MatrixIndexT j = 0; j < dim2; j++)
        A(i,j) = i + 2 * j + 1;
    }
    KALDI_LOG << A;
    CuMatrix<Real> B(dim1,dim2);
    B.CopyFromMat(A);
    CuVector<Real> C(dim1);
    C.SetZero();
    Real alpha = 1;
    Real beta = 1;
    C.AddDiagMat2(alpha, B, kNoTrans, beta);
    Vector<Real> D(dim1);
    C.CopyToVec(&D);
    KALDI_LOG << D;
    Vector<Real> E(dim1);
    E.AddDiagMat2(alpha, A, kNoTrans, beta);
    KALDI_LOG << E;
    AssertEqual(D,E);
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim1 = 15 + Rand() % 10;
    int32 dim2 = 10 + Rand() % 10;
    Matrix<Real> A(dim1,dim2);
    for (MatrixIndexT i = 0; i < dim1; i++) {
      for (MatrixIndexT j = 0; j < dim2; j++)
        A(i,j) = i + 2 * j + 1;
    }
    KALDI_LOG << A;
    CuMatrix<Real> B(dim1,dim2);
    B.CopyFromMat(A);
    CuSubVector<Real> C(B,1);
    Vector<Real> D(dim2);
    C.CopyToVec(&D);
    KALDI_LOG << D;
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + Rand() % 10;
    CuVector<Real> A(dim);
    A.SetRandn();
    Vector<Real> A1(dim);
    A.CopyToVec(&A1);
    CuVector<Real> B(dim);
    B.SetRandn();
    Vector<Real> B1(dim);
    B.CopyToVec(&B1);
    Real dot = VecVec(A,B);
    KALDI_LOG << "dot product in gpu: " << dot;
    dot = VecVec(A1,B1);
    KALDI_LOG << "dot product in cpu: " << dot;    
  }

  for (MatrixIndexT iter = 0; iter < 10; iter++) {
    int32 dim = 15 + Rand() % 10;
    CuVector<Real> A(dim);
    Vector<Real> A1(dim);
    for (MatrixIndexT i = 0; i < dim; i++)
      A1(i) = i;
    A.CopyFromVec(A1);
    KALDI_LOG << A(dim-2);
    KALDI_LOG << A1(dim-2);
  }
  */
}

template<typename Real>
static void CuMatrixUnitTest() {
  UnitTestTrace<Real>();
  UnitTestCholesky<Real>();
  UnitTestInvert<Real>();
  UnitInvert<Real>();
  UnitTestCopyFromMat<Real>();
  UnitTestCopySp<Real>();
  UnitTestConstructor<Real>();
  UnitTestVector<Real>();
  UnitTestMulTp<Real>();
  UnitTestMatrix<Real>();
  UnitTestSetZeroAboveDiag<Real>();
}
} //namespace

int main() {
  using namespace kaldi;

  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    kaldi::CuMatrixUnitTest<float>();

#if HAVE_CUDA == 1
    if (!kaldi::CuDevice::Instantiate().DoublePrecisionSupported()) {
      KALDI_WARN << "Double precision not supported, not testing that code";
    } else
#endif
    {
      kaldi::CuMatrixUnitTest<double>();
    }
  }

#if HAVE_CUDA == 1
  kaldi::CuDevice::Instantiate().PrintProfile();
#endif
  
  KALDI_LOG << "Tests succeeded.";
  return 0;
}

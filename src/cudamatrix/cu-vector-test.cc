// cudamatrix/cu-vector-test.cc

// Copyright 2013 Lucas Ondel
//           2013 Johns Hopkins University (author: Daniel Povey)
//           2017 Hossein Hadian, Daniel Galvez

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
#include <cmath>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {

/*
 * INITIALIZERS
 */


/*
 * Unit tests
 */

template<class Real>
static void UnitTestCuVectorIO() {
  for (int32 i = 0; i < 10; i++) {
    int32 dimM = Rand() % 255;
    if (i % 5 == 0) { dimM = 0; }
    CuVector<Real> vec(dimM);
    vec.SetRandn();
    std::ostringstream os;
    bool binary = (i % 4 < 2);
    vec.Write(os, binary);

    CuVector<Real> vec2;
    std::istringstream is(os.str());
    vec2.Read(is, binary);
    AssertEqual(vec, vec2);
  }
}


template<typename Real, typename OtherReal>
static void UnitTestCuVectorCopyFromVec() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Vector<Real> A(dim);
    A.SetRandn();
    CuVector<OtherReal> B(A);
    Vector<Real> C(B);
    CuVector<Real> D(dim);
    D.CopyFromVec(C);
    Vector<OtherReal> E(dim);
    E.CopyFromVec(D);
    CuVector<Real> F(E);
    CuVector<Real> A2(A);
    AssertEqual(F, A2);
  }
}

template<typename Real>
static void UnitTestCuSubVector() {
  for (int32 iter = 0 ; iter < 10; iter++) {
    int32 M1 = 1 + rand () % 10, M2 = 1 + Rand() % 1, M3 = 1 + Rand() % 10, M = M1 + M2 + M3,
        m = Rand() % M2;
    CuVector<Real> vec(M);
    vec.SetRandn();
    CuSubVector<Real> subvec1(vec, M1, M2),
        subvec2 = vec.Range(M1, M2);
    Real f1 = vec(M1 + m), f2 = subvec1(m), f3 = subvec2(m);
    KALDI_ASSERT(f1 == f2);
    KALDI_ASSERT(f2 == f3);
  }
}



template<typename Real>
static void UnitTestCuVectorMulTp() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Vector<Real> A(dim);
    A.SetRandn();
    TpMatrix<Real> B(dim);
    B.SetRandn();

    CuVector<Real> C(A);
    CuTpMatrix<Real> D(B);

    A.MulTp(B, kNoTrans);
    C.MulTp(D, kNoTrans);

    CuVector<Real> E(A);
    AssertEqual(C, E);
  }
}

template<typename Real>
static void UnitTestCuVectorAddTp() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Vector<Real> A(dim);
    A.SetRandn();
    TpMatrix<Real> B(dim);
    B.SetRandn();
    Vector<Real> C(dim);
    C.SetRandn();

    CuVector<Real> D(A);
    CuTpMatrix<Real> E(B);
    CuVector<Real> F(C);

    A.AddTpVec(1.0, B, kNoTrans, C, 1.0);
    D.AddTpVec(1.0, E, kNoTrans, F, 1.0);

    CuVector<Real> G(A);
    AssertEqual(D, G);
  }
}

template<typename Real> void CuVectorUnitTestVecVec() {
  int32 M = 10 + Rand() % 100;
  CuVector<Real> vec1(M), vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  Real prod = 0.0;
  for (int32 i = 0; i < M; i++)
    prod += vec1(i) * vec2(i);
  AssertEqual(prod, VecVec(vec1, vec2));
}

template<typename Real> void CuVectorUnitTestAddVec() {
  int32 M = 10 + Rand() % 100;
  CuVector<Real> vec1(M);
  CuVector<Real> vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  CuVector<Real> vec1_orig(vec1);
  BaseFloat alpha = 0.43243;
  vec1.AddVec(alpha, vec2);

  for (int32 i = 0; i < M; i++)
    AssertEqual(vec1_orig(i) + alpha * vec2(i), vec1(i));
}

template<typename Real> void CuVectorUnitTestAddVecCross() {
  for (int32 i = 0; i < 4; i++) {
    int32 M = 10 + Rand() % 100;
    CuVector<float> vec1(M);
    CuVector<Real> vec2(M);
    vec1.SetRandn();
    vec2.SetRandn();

    if (i == 0) {
      CuVector<Real> vec1_orig(vec1);
      Real alpha = 0.43243;
      vec1.AddVec(alpha, vec2);

      for (int32 i = 0; i < M; i++)
        AssertEqual(vec1_orig(i) + alpha * vec2(i), vec1(i));
    } else {
      CuVector<Real> vec2_orig(vec2);
      Real alpha = 0.43243;
      vec2.AddVec(alpha, vec1);
      for (int32 i = 0; i < M; i++)
        AssertEqual(vec2_orig(i) + alpha * vec1(i), vec2(i));
    }
  }
}

template<typename Real> void CuVectorUnitTestAddVecExtra() {
  int32 M = 10 + Rand() % 100;
  CuVector<Real> vec1(M), vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  CuVector<Real> vec1_orig(vec1);
  BaseFloat alpha = 0.43243, beta = 1.4321;
  vec1.AddVec(alpha, vec2, beta);

  for (int32 i = 0; i < M; i++)
    AssertEqual(beta * vec1_orig(i) + alpha * vec2(i), vec1(i));
}

template<typename Real> void CuVectorUnitTestCopyElements() {
  int32 dim = 10 + Rand() % 100, N = 20 + Rand() % 50;
  CuVector<Real> V(dim);
  V.SetRandn();
  CuVector<Real> V_copy(V);
  for (int n = 0; n < 2; n++) {
    bool transpose = (n == 0);
    CuMatrix<Real> M;
    if (!transpose)
      M.Resize(dim, N, kUndefined);
    else
      M.Resize(N, dim, kUndefined);
    M.SetRandn();
    std::vector<int32> elements(dim);
    for (int32 i = 0; i < dim; i++) {
      int32 j = elements[i] = Rand() % N;
      if (!transpose)
        V_copy(i) = M(i, j);
      else
        V_copy(i) = M(j, i);
    }
    CuArray<int32> cu_elements(elements);
    V.CopyElements(M, transpose ? kTrans : kNoTrans, cu_elements);
    AssertEqual(V, V_copy);
  }
}

template<typename Real> void UnitTestVecMatVec() {
  int32 NR = 10 + Rand() % 100, NC = 20 + Rand() % 100;
  CuVector<Real> v1(NR), v2(NC);
  v1.SetRandn();
  v2.SetRandn();
  CuMatrix<Real> M(NR, NC);
  M.SetRandn();
  Real sum = 0;
  for (int32 i = 0; i < NR; i++)
    for (int32 j = 0; j < NC; j++)
      sum += v1(i) * M(i, j) * v2(j);
  Real result = VecMatVec(v1, M, v2);
  AssertEqual(sum, result);
}

template<typename Real> void CuVectorUnitTestAddRowSumMat() {
  int32 M = 10 + Rand() % 280, N = 10 + Rand() % 20;
  BaseFloat alpha = 10.0143432, beta = 43.4321;
  CuMatrix<Real> mat(N, M);
  mat.SetRandn();
  CuVector<Real> vec(M);
  mat.SetRandn();
  Matrix<Real> mat2(mat);
  Vector<Real> vec2(M);
  vec.AddRowSumMat(alpha, mat, beta);
  vec2.AddRowSumMat(alpha, mat2, beta);
  Vector<Real> vec3(vec);
  AssertEqual(vec2, vec3);
}

template<typename Real> void CuVectorUnitTestAddColSumMat() {
  int32 M = 10 + Rand() % 280, N = 10 + Rand() % 20;
  BaseFloat alpha = 10.0143432, beta = 43.4321;
  CuMatrix<Real> mat(M, N);
  mat.SetRandn();
  CuVector<Real> vec(M);
  mat.SetRandn();
  Matrix<Real> mat2(mat);
  Vector<Real> vec2(M);
  vec.AddColSumMat(alpha, mat, beta);
  vec2.AddColSumMat(alpha, mat2, beta);
  Vector<Real> vec3(vec);
  AssertEqual(vec2, vec3);
}


template<typename Real> void CuVectorUnitTestApproxEqual() {
  int32 M = 10 + Rand() % 100;
  CuVector<Real> vec1(M), vec2(M);
  vec1.SetRandn();
  vec2.SetRandn();
  Real tol = 0.5;
  for (int32 i = 0; i < 10; i++) {
    Real sumsq = 0.0, sumsq_orig = 0.0;
    for (int32 j = 0; j < M; j++) {
      sumsq += (vec1(j) - vec2(j)) * (vec1(j) - vec2(j));
      sumsq_orig += vec1(j) * vec1(j);
    }
    Real rms = sqrt(sumsq), rms_orig = sqrt(sumsq_orig);
    KALDI_ASSERT(vec1.ApproxEqual(vec2, tol) == (rms <= tol * rms_orig));
    tol *= 2.0;
  }
}

template<typename Real> static void UnitTestCuVectorReplaceValue() {
  for (int32 i = 0; i < 5; i++) {
    int32 dim = 100 + Rand() % 200;
    Real orig = 0.1 * (Rand() % 100), changed = 0.1 * (Rand() % 50);
    Vector<Real> vec(dim);
    vec.SetRandn();
    vec(dim / 2) = orig;
    CuVector<Real> vec1(vec);
    vec.ReplaceValue(orig, changed);
    vec1.ReplaceValue(orig, changed);
    Vector<Real> vec2(vec1);
    AssertEqual(vec, vec2);
  }
}

template<typename Real> static void UnitTestCuVectorSum() {
  for (int32 i = 0; i < 200; i++) {
    int32 start_dim = RandInt(1, 500), end_dim = RandInt(1, 500);
    int32 dim = RandInt(10, 12000) + start_dim + end_dim;
    Real quiet_nan = nan("");  // this is from <cmath>.
    Vector<BaseFloat> vec(start_dim + dim + end_dim);
    vec.Range(0, start_dim).Set(quiet_nan);
    vec.Range(start_dim, dim).Set(1.0);
    vec.Range(start_dim + dim, end_dim).Set(quiet_nan);
    BaseFloat sum = vec.Range(start_dim, dim).Sum();
    KALDI_ASSERT(ApproxEqual(sum, dim));
  }
}

template<typename Real> void CuVectorUnitTestInvertElements() {
  // Also tests MulElements();
  int32 M = 256 + Rand() % 100;
  CuVector<Real> vec1(M);
  vec1.SetRandn();
  CuVector<Real> vec2(vec1);
  vec2.InvertElements();
  CuVector<Real> vec3(vec1);
  vec3.MulElements(vec2);
  // vec3 should be all ones.
  Real prod = VecVec(vec3, vec3);
  AssertEqual(prod, static_cast<Real>(M));
}

template<typename Real> void CuVectorUnitTestSum() {
  for (int32 p = 1; p <= 1000000; p *= 2) {
    MatrixIndexT dim = p + Rand() % 500;
    CuVector<Real> A(dim), ones(dim);
    A.SetRandn();
    ones.Set(1.0);

    Real x = VecVec(A, ones);
    Real y = A.Sum();
    Real diff = std::abs(x - y);
    // Note: CuVectorBase<> does not have an ApplyAbs() member
    // function, so we copy back to a host vector for simplicity in
    // this test case.
    Vector<Real> A_host(A);
    A_host.ApplyAbs();
    Real s = A_host.Sum();
    KALDI_ASSERT ( diff <= 1.0e-04 * s);
  }
}

template<typename Real> void CuVectorUnitTestScale() {
  for (int32 i = 0; i < 4; i++) {
    int32 dim = 100 + Rand() % 400;
    CuVector<Real> cu_vec(dim);
    cu_vec.SetRandn();
    Vector<Real> vec(cu_vec);
    BaseFloat scale = 0.333;
    cu_vec.Scale(scale);
    vec.Scale(scale);
    Vector<Real> vec2(cu_vec);
    KALDI_ASSERT(ApproxEqual(vec, vec2));
  }
}

template<typename Real> void CuVectorUnitTestCopyFromMat() {
  int32 M = 100 + Rand() % 255, N = 100 + Rand() % 255;
  CuMatrix<Real> cu_matrix(M, N);
  cu_matrix.SetRandn();
  for(int32 i = 0; i < N; i++) {
    CuVector<Real> vector(M);
    vector.CopyColFromMat(cu_matrix, i);
    for(int32 j = 0; j < M; j++) {
      KALDI_ASSERT(vector(j)==cu_matrix(j, i));
    }
  }
  Matrix<Real> matrix(cu_matrix), matrix2(M, N);
  CuMatrix<Real> matrix3(M, N);

  CuVector<Real> vector(M * N), vector2(M * N);
  vector.CopyRowsFromMat(cu_matrix);
  vector2.CopyRowsFromMat(matrix);
  matrix2.CopyRowsFromVec(vector2);
  matrix3.CopyRowsFromVec(Vector<Real>(vector2));
  Vector<Real> vector3(M * N);
  vector3.CopyRowsFromMat(cu_matrix);


  for(int32 j = 0; j < M*N; j++) {
    if (Rand() % 500 == 0) { // random small subset (it was slow)
      KALDI_ASSERT(vector(j) == cu_matrix(j/N, j%N));
      KALDI_ASSERT(vector2(j) == cu_matrix(j/N, j%N));
      KALDI_ASSERT(vector2(j) == matrix2(j/N, j%N));
      KALDI_ASSERT(vector3(j) == matrix2(j/N, j%N));
      KALDI_ASSERT(vector3(j) == matrix3(j/N, j%N));
    }
  }
}

template<typename Real> void CuVectorUnitTestCopyDiagFromPacked() {
  for (int32 i = 0; i < 5; i++) {
    int32 N = 100 + Rand() % 255;
    CuSpMatrix<Real> S(N);
    S.SetRandn();
    CuVector<Real> V(N, kUndefined);
    V.CopyDiagFromPacked(S);
    SpMatrix<Real> cpu_S(S);
    Vector<Real> cpu_V(N);
    cpu_V.CopyDiagFromPacked(cpu_S);
    Vector<Real> cpu_V2(V);
    KALDI_ASSERT(cpu_V.ApproxEqual(cpu_V2));
  }
}

template<typename Real> void CuVectorUnitTestCopyCross() {
  for (int32 i = 0; i < 10; i++) {
    int32 M = 100 + Rand() % 255;
    if (Rand() % 3 == 0) M = 0;
    CuVector<Real> v1(M);
    v1.SetRandn();
    CuVector<float> v2(M);
    v2.CopyFromVec(v1);
    CuVector<Real> v3(M);
    v3.CopyFromVec(v2);
    AssertEqual(v1, v3);
  }
}

template<typename Real> void CuVectorUnitTestCopyCross2() {
  for (int32 i = 0; i < 10; i++) {
    int32 M = 100 + Rand() % 255;
    if (Rand() % 3 == 0) M = 0;
    CuVector<Real> v1(M);
    v1.SetRandn();
    Vector<float> v2(M);
    v2.CopyFromVec(v1);
    CuVector<Real> v3(M);
    v3.CopyFromVec(v2);
    AssertEqual(v1, v3);
  }
}

template<typename Real> void CuVectorUnitTestCopyDiagFromMat() {
  for (int32 i = 0; i < 5; i++) {
    int32 M = 100 + Rand() % 255, N = M + Rand() % 2;
    Matrix<Real> matrix(M, N);
    if (i % 2 == 0) matrix.Transpose();
    matrix.SetRandn();
    Vector<Real> vector(M, kUndefined);
    vector.CopyDiagFromMat(matrix);

    CuMatrix<Real> cuda_matrix(matrix);
    CuVector<Real> cuda_vector(M, kUndefined);
    cuda_vector.CopyDiagFromMat(cuda_matrix);
    Vector<Real> vector2(cuda_vector);
    AssertEqual(vector, vector2);
    AssertEqual(vector.Sum(), cuda_matrix.Trace(false));
    AssertEqual(cuda_vector.Sum(), matrix.Trace(false));
  }
}


template<typename Real> void CuVectorUnitTestNorm() {
  int32 dim = 2;
  CuVector<Real> cu_vector(dim);
  cu_vector(0) = 1.0;
  cu_vector(1) = -2.0;
  KALDI_ASSERT(ApproxEqual(cu_vector.Norm(1.0), 3.0));
  KALDI_ASSERT(ApproxEqual(cu_vector.Norm(2.0), sqrt(5.0)));
}


template<typename Real> void CuVectorUnitTestMin() {
  for (int32 p = 1; p <= 1000000; p *= 2) {
    int32 dim = p + Rand() % 500;
    CuVector<Real> cu_vector(dim);
    cu_vector.SetRandn();
    Vector<Real> vector(cu_vector);
    Real min1 = cu_vector.Min(), min2 = vector.Min();
    KALDI_ASSERT(min1 == min2);
  }
}


template<typename Real> void CuVectorUnitTestMax() {
  for (int32 p = 1; p <= 1000000; p *= 2) {
    int32 dim = p + Rand() % 500;
    CuVector<Real> cu_vector(dim);
    cu_vector.SetRandn();
    Vector<Real> vector(cu_vector);
    Real max1 = cu_vector.Max(), max2 = vector.Max();
    KALDI_ASSERT(max1 == max2);
  }
}


template<typename Real> void CuVectorUnitTestApplySoftMax() {
  for (int32 i = 0; i < 10; i++) {
    int32 dim = 100 + Rand() % 300;
    //int32 dim = 1024;
    CuVector<Real> cu_vector(dim);
    cu_vector.SetRandn();
    Vector<Real> vector(cu_vector);

    cu_vector.ApplySoftMax();
    vector.ApplySoftMax();
    CuVector<Real> cu_vector2(vector);
    //std::cout<<cu_vector <<"\n"<<cu_vector2<<std::endl;
    AssertEqual(cu_vector, cu_vector2);
  }
}

template<typename Real> void CuVectorUnitTestApplyExp() {
  int32 dim = 100;
  CuVector<Real> vector(dim);
  vector.SetRandn();
  CuVector<Real> vector2(vector);

  vector.ApplyExp();
  for(int32 j = 0; j < dim; j++) {
    //std::cout<<"diff is "<<exp(vector2(j))-vector(j)<<std::endl;;
    KALDI_ASSERT(std::abs(Exp(vector2(j))-vector(j)) < 0.00001);
  }

}

template<typename Real> void CuVectorUnitTestApplyLog() {
  int32 dim = 100;
  CuVector<Real> vector(dim);
  vector.SetRandn();
  for(int32 j = 0; j < dim; j++) {
    if(vector(j) <= 0.0)
      vector(j) = 1.0 - vector(j);
  }

  CuVector<Real> vector2(vector);

  vector.ApplyLog();
  for(int32 j = 0; j < dim; j++) {
    //std::cout<<"diff is "<<exp(vector2(j))-vector(j)<<std::endl;;
    KALDI_ASSERT(std::abs(Log(vector2(j))-vector(j)) < 0.000001 );
  }
}

template<typename Real> void CuVectorUnitTestApplyFloor() {
  for (int32 l = 0; l < 10; l++) {
    int32 dim = 100 + Rand() % 700;
    CuVector<Real> cu_vector(dim);
    cu_vector.SetRandn();

    Vector<Real> vector(cu_vector);
    BaseFloat floor = 0.33 * (-5 + Rand() % 10);
    MatrixIndexT i, j;
    cu_vector.ApplyFloor(floor, &i);
    vector.ApplyFloor(floor, &j);

    CuVector<Real> cu2(vector);

    AssertEqual(cu2, cu_vector);
    if (i != j) {
      KALDI_WARN << "ApplyFloor return code broken...";
    }
    KALDI_ASSERT(i==j);
  }
}

template<typename Real> void CuVectorUnitTestApplyFloorNoCount() {
  for (int32 l = 0; l < 10; l++) {
    int32 dim = 100 + Rand() % 700;
    CuVector<Real> cu_vector1(dim);
    cu_vector1.SetRandn();
    CuVector<Real> cu_vector2(cu_vector1);

    BaseFloat floor = 0.33 * (-5 + Rand() % 10);
    MatrixIndexT dummy_count;
    cu_vector1.ApplyFloor(floor, &dummy_count);
    cu_vector2.ApplyFloor(floor, nullptr);
    AssertEqual(cu_vector1, cu_vector2);
  }
}

template<typename Real> void CuVectorUnitTestApplyCeiling() {
  for (int32 l = 0; l < 10; l++) {
    int32 dim = 100 + Rand() % 700;
    CuVector<Real> cu_vector(dim);
    cu_vector.SetRandn();

    Vector<Real> vector(cu_vector);
    BaseFloat floor = 0.33 * (-5 + Rand() % 10);
    MatrixIndexT i, j;
    cu_vector.ApplyCeiling(floor, &i);
    vector.ApplyCeiling(floor, &j);

    CuVector<Real> cu2(vector);

    AssertEqual(cu2, cu_vector);
    if (i != j) {
      KALDI_WARN << "ApplyCeiling return code broken...";
    }
    KALDI_ASSERT(i==j);
  }
}

template<typename Real> void CuVectorUnitTestApplyCeilingNoCount() {
  for (int32 l = 0; l < 10; l++) {
    int32 dim = 100 + Rand() % 700;
    CuVector<Real> cu_vector1(dim);
    cu_vector1.SetRandn();
    CuVector<Real> cu_vector2(cu_vector1);

    BaseFloat floor = 0.33 * (-5 + Rand() % 10);
    MatrixIndexT dummy_count;
    cu_vector1.ApplyCeiling(floor, &dummy_count);
    cu_vector2.ApplyCeiling(floor, nullptr);
    AssertEqual(cu_vector1, cu_vector2);
  }
}

template<typename Real> void CuVectorUnitTestApplyPow() {
  for (int32 l = 0; l < 10; l++) {
    int32 dim = 100 + Rand() % 700;

    CuVector<Real> cu_vector(dim);
    cu_vector.SetRandn();

    Vector<Real> vector(cu_vector);

    BaseFloat pow = -2 + (Rand() % 5);
    cu_vector.ApplyPow(pow);
    vector.ApplyPow(pow);

    CuVector<Real> cu2(vector);

    AssertEqual(cu2, cu_vector);
  }
}

template<typename Real> void CuVectorUnitTestAddVecVec() {
  int32 dim = 100;
  CuVector<Real> cu_vector(dim);
  cu_vector.SetRandn();
  Vector<Real> vector(cu_vector);

  Real beta = Rand();
  Real alpha = Rand();
  Vector<Real> v(dim), r(dim);
  v.SetRandn(); r.SetRandn();
  CuVector<Real> cuV(v), cuR(r);


  cu_vector.AddVecVec(alpha, cuR, cuV, beta);
  vector.AddVecVec(alpha, r, v, beta);

  CuVector<Real> cu2(vector);
  std::cout<<cu2(0)<<' '<<cu_vector(0)<<std::endl;
  AssertEqual(cu2, cu_vector);
}

template<typename Real> void CuVectorUnitTestAddDiagMat2() {
  for (int p = 0; p < 4; p++) {
    int32 M = 230 + Rand() % 100, N = 230 + Rand() % 100;
    BaseFloat alpha = 0.2 + Rand() % 3, beta = 0.3 + Rand() % 2;
    CuVector<Real> cu_vector(M);
    cu_vector.SetRandn();

    CuMatrix<Real> cu_mat_orig(M, N);
    cu_mat_orig.SetRandn();
    MatrixTransposeType trans = (p % 2 == 0 ? kNoTrans : kTrans);
    CuMatrix<Real> cu_mat(cu_mat_orig, trans);

    Vector<Real> vector(cu_vector);
    Matrix<Real> mat(cu_mat);

    vector.AddDiagMat2(alpha, mat, trans, beta);
    cu_vector.AddDiagMat2(alpha, cu_mat, trans, beta);

    Vector<Real> vector2(cu_vector);
    AssertEqual(vector, vector2);
  }
}

template<typename Real>
static void CuVectorUnitTestAddDiagMatMat() {
  for (MatrixIndexT iter = 0; iter < 4; iter++) {
    BaseFloat alpha = 0.432 + Rand() % 5, beta = 0.043 + Rand() % 2;
    MatrixIndexT dimM = 10 + Rand() % 300,
                 dimN = 5 + Rand() % 300;
    CuVector<Real> v(dimM);
    CuMatrix<Real> M_orig(dimM, dimN), N_orig(dimN, dimM);
    M_orig.SetRandn();
    N_orig.SetRandn();
    MatrixTransposeType transM = (iter % 2 == 0 ? kNoTrans : kTrans);
    MatrixTransposeType transN = ((iter/2) % 2 == 0 ? kNoTrans : kTrans);
    CuMatrix<Real> M(M_orig, transM), N(N_orig, transN);

    v.SetRandn();
    CuVector<Real> w(v);

    w.AddDiagMatMat(alpha, M, transM, N, transN, beta);

    {
      CuVector<Real> w2(v);
      CuMatrix<Real> MN(dimM, dimM);
      MN.AddMatMat(1.0, M, transM, N, transN, 0.0);
      CuVector<Real> d(dimM);
      d.CopyDiagFromMat(MN);
      w2.Scale(beta);
      w2.AddVec(alpha, d);
      AssertEqual(w, w2);
    }
  }
}



template<typename Real> void CuVectorUnitTestAddMatVec() {
  for (int32 i = 0; i < 10; i++) {
    int32 M = 10 + Rand() % 500, N = 10 + Rand() % 400;

    bool transpose = (i % 2 == 0);

    CuVector<Real> src_cu(M);
    src_cu.SetRandn();
    Vector<Real> src(src_cu);

    CuVector<Real> dst_cu(N);
    dst_cu.SetRandn();
    Vector<Real> dst(dst_cu);

    CuMatrix<Real> mat_cu(transpose ? M : N, transpose ? N : M);
    mat_cu.SetRandn();
    Matrix<Real> mat(mat_cu);

    BaseFloat alpha = 0.5 * (Rand() % 10), beta = 0.5 * (Rand() % 10);
    dst_cu.AddMatVec(alpha, mat_cu, transpose ? kTrans : kNoTrans,
                     src_cu, beta);
    dst.AddMatVec(alpha, mat, transpose ? kTrans : kNoTrans,
                  src, beta);
    Vector<Real> dst2(dst_cu);
    AssertEqual(dst, dst2);
  }
}


template<typename Real> void CuVectorUnitTestAddSpVec() {
  for (int32 i = 0; i < 5; i++) {
    int32 M = 100 + Rand() % 256;

    CuVector<Real> src_cu(M);
    src_cu.SetRandn();
    Vector<Real> src(src_cu);

    CuVector<Real> dst_cu(M);
    dst_cu.SetRandn();
    Vector<Real> dst(dst_cu);

    CuSpMatrix<Real> mat_cu(M);
    mat_cu.SetRandn();
    SpMatrix<Real> mat(mat_cu);

    BaseFloat alpha = 0.5 * (Rand() % 5), beta = 0.5 * (Rand() % 5);
    dst_cu.AddSpVec(alpha, mat_cu, src_cu, beta);
    dst.AddSpVec(alpha, mat, src, beta);
    Vector<Real> dst2(dst_cu);
    AssertEqual(dst, dst2);
  }
}



template<typename Real> void CuVectorUnitTest() {
  UnitTestCuVectorCopyFromVec<Real, float>();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().DoublePrecisionSupported())
#endif
  UnitTestCuVectorCopyFromVec<Real, double>();
  UnitTestCuVectorIO<Real>();
  CuVectorUnitTestVecVec<Real>();
  CuVectorUnitTestAddVec<Real>();
  CuVectorUnitTestAddVecCross<Real>();
  CuVectorUnitTestAddVecExtra<Real>();
  CuVectorUnitTestApproxEqual<Real>();
  CuVectorUnitTestScale<Real>();
  CuVectorUnitTestSum<Real>();
  CuVectorUnitTestInvertElements<Real>();
  UnitTestCuVectorSum<Real>();
  CuVectorUnitTestAddRowSumMat<Real>();
  CuVectorUnitTestAddColSumMat<Real>();
  UnitTestCuVectorReplaceValue<Real>();
  UnitTestCuVectorAddTp<Real>();
  UnitTestCuVectorMulTp<Real>();
  UnitTestCuSubVector<Real>();
  CuVectorUnitTestCopyFromMat<Real>();
  CuVectorUnitTestMin<Real>();
  CuVectorUnitTestMax<Real>();
  CuVectorUnitTestApplySoftMax<Real>();
  CuVectorUnitTestCopyDiagFromPacked<Real>();
  CuVectorUnitTestCopyDiagFromMat<Real>();
  CuVectorUnitTestCopyCross<Real>();
  CuVectorUnitTestCopyCross2<Real>();
  CuVectorUnitTestNorm<Real>();
  CuVectorUnitTestApplyExp<Real>();
  CuVectorUnitTestApplyLog<Real>();
  CuVectorUnitTestApplyFloor<Real>();
  CuVectorUnitTestApplyFloorNoCount<Real>();
  CuVectorUnitTestApplyCeilingNoCount<Real>();
  CuVectorUnitTestApplyCeiling<Real>();
  CuVectorUnitTestApplyPow<Real>();
  CuVectorUnitTestAddMatVec<Real>();
  CuVectorUnitTestAddSpVec<Real>();
  CuVectorUnitTestAddVecVec<Real>();
  CuVectorUnitTestAddDiagMat2<Real>();
  CuVectorUnitTestAddDiagMatMat<Real>();
  CuVectorUnitTestCopyElements<Real>();
  UnitTestVecMatVec<Real>();
}


} // namespace kaldi


int main(int argc, char *argv[]) {
  using namespace kaldi;
  SetVerboseLevel(1);
  const char *usage = "Usage: cu-vector-test [options]";

  ParseOptions po(usage);
  std::string use_gpu = "yes";
  po.Register("use-gpu", &use_gpu, "yes|no|optional");
  po.Read(argc, argv);

  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(1);
  }

  int32 loop = 0;
#if HAVE_CUDA == 1
  for (; loop < 2; loop++) {
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    kaldi::CuVectorUnitTest<float>();
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().DoublePrecisionSupported()) {
      kaldi::CuVectorUnitTest<double>();
    } else {
      KALDI_WARN << "Double precision not supported";
    }
#else
    kaldi::CuVectorUnitTest<double>();
#endif

    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
#if HAVE_CUDA == 1
  }
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}

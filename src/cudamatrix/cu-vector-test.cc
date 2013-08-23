// cudamatrix/cuda-matrix-test.cc

// Copyright 2013 Lucas Ondel

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
static void AssertEqual(CuVectorBase<Real> &A, CuVectorBase<Real> &B, float tol = 0.001) {
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
static void UnitTestCuVectorCopyFromVec() {
  for (int32 i = 1; i < 10; i++) {
    MatrixIndexT dim = 10 * i;
    Vector<Real> A(dim);
    A.SetRandn();
    CuVector<Real> B(A);


    CuVector<Real> C(dim);
    CuVector<Real> D(dim);
    C.CopyFromVec(A);
    D.CopyFromVec(B);
    
    Vector<Real> E(dim);
    Vector<Real> F(dim);
    C.CopyToVec(&E);
    D.CopyToVec(&F);

    AssertEqual(C, D);
    AssertEqual(E, F);
  }
}

template<class Real> 
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

template<class Real> 
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

template<class Real> void CuVectorUnitTest() {
  UnitTestCuVectorCopyFromVec<Real>();
  UnitTestCuVectorAddTp<Real>();
  UnitTestCuVectorMulTp<Real>();
}


} // namespace kaldi


int main() {
    //Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(-2); //-2 .. automatic selection
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
  std::cout << "Tests succeeded.\n";
}

// cudamatrix/cu-rand-speed-test.cc

// Copyright 2016  Brno University of Technology (author: Karel Vesely)

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
#include "cudamatrix/cu-rand.h"

using namespace kaldi;


namespace kaldi {

template<typename Real>
std::string NameOf() {
  return (sizeof(Real) == 8 ? "<double>" : "<float>");
}

template <typename T>
std::string ToString(const T& t) {
  std::ostringstream os;
  os << t;
  return os.str();
}

template<typename Real>
std::string MeanVariance(const CuMatrixBase<Real>& m) {
  std::ostringstream os;
  Real mean = m.Sum() / (m.NumRows()*m.NumCols());
  CuMatrix<Real> tmp(m);
  tmp.Add(-mean);
  tmp.ApplyPow(2.0);
  Real var = tmp.Sum() / (tmp.NumRows()*tmp.NumCols());
  return std::string("mean ") + ToString(mean) + ", std-dev " + ToString(std::sqrt(var));
}

template<typename Real>
std::string MeanVariance(const CuVectorBase<Real>& v) {
  std::ostringstream os;
  Real mean = v.Sum() / v.Dim();
  CuVector<Real> tmp(v);
  tmp.Add(-mean);
  tmp.ApplyPow(2.0);
  Real var = tmp.Sum() / tmp.Dim();
  return std::string("mean ") + ToString(mean) + ", std-dev " + ToString(std::sqrt(var));
}


template <typename Real>
void CuRandUniformMatrixSpeedTest(const int32 iter) {
  Timer t;
  CuRand<Real> rand;
  CuMatrix<Real> m(249,1001, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandUniform(&m);
  }
  CuMatrix<Real> m2(256,1024, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandUniform(&m2);
  }
  // flops = number of generated random numbers per second,
  Real flops = iter * (m.NumRows() * m.NumCols() + m2.NumRows() * m2.NumCols()) / t.Elapsed();
  KALDI_LOG << __func__ << NameOf<Real>()
            << " Speed was " << flops << " rand_elems/s. "
            << "(debug " << MeanVariance(m) << ")";
}

template <typename Real>
void CuRandUniformMatrixBaseSpeedTest(const int32 iter) {
  Timer t;
  CuRand<Real> rand;
  CuMatrix<Real> m(249,1001, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandUniform(dynamic_cast<CuMatrixBase<Real>*>(&m));
  }
  CuMatrix<Real> m2(256,1024, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandUniform(dynamic_cast<CuMatrixBase<Real>*>(&m2));
  }
  // flops = number of generated random numbers per second,
  Real flops = iter * (m.NumRows() * m.NumCols() + m2.NumRows() * m2.NumCols()) / t.Elapsed();
  KALDI_LOG << __func__ << NameOf<Real>()
            << " Speed was " << flops << " rand_elems/s. "
            << "(debug " << MeanVariance(m) << ")";
}

template <typename Real>
void CuRandGaussianMatrixSpeedTest(const int32 iter) {
  Timer t;
  CuRand<Real> rand;
  CuMatrix<Real> m(249,1001, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandGaussian(&m);
  }
  CuMatrix<Real> m2(256,1024, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandGaussian(&m2);
  }
  // flops = number of generated random numbers per second,
  Real flops = iter * (m.NumRows() * m.NumCols() + m2.NumRows() * m2.NumCols()) / t.Elapsed();
  KALDI_LOG << __func__ << NameOf<Real>()
            << " Speed was " << flops << " rand_elems/s. "
            << "(debug " << MeanVariance(m) << ")";
}

template <typename Real>
void CuRandGaussianMatrixBaseSpeedTest(const int32 iter) {
  Timer t;
  CuRand<Real> rand;
  CuMatrix<Real> m(249,1001, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandGaussian(dynamic_cast<CuMatrixBase<Real>*>(&m));
  }
  CuMatrix<Real> m2(256,1024, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandGaussian(dynamic_cast<CuMatrixBase<Real>*>(&m2));
  }
  // flops = number of generated random numbers per second,
  Real flops = iter * (m.NumRows() * m.NumCols() + m2.NumRows() * m2.NumCols()) / t.Elapsed();
  KALDI_LOG << __func__ << NameOf<Real>()
            << " Speed was " << flops << " rand_elems/s. "
            << "(debug " << MeanVariance(m) << ")";
}

template <typename Real>
void CuRandUniformVectorSpeedTest(const int32 iter) {
  Timer t;
  CuRand<Real> rand;
  CuVector<Real> v(2011, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandUniform(&v);
  }
  CuVector<Real> v2(2048, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandUniform(&v2);
  }
  // flops = number of generated random numbers per second,
  Real flops = iter * (v.Dim() + v2.Dim()) / t.Elapsed();
  KALDI_LOG << __func__ << NameOf<Real>()
            << " Speed was " << flops << " rand_elems/s. "
            << "(debug " << MeanVariance(v) << ")";
}

template <typename Real>
void CuRandGaussianVectorSpeedTest(const int32 iter) {
  Timer t;
  CuRand<Real> rand;
  CuVector<Real> v(2011, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandGaussian(&v);
  }
  CuVector<Real> v2(2048, kUndefined);
  for (int32 i = 0; i < iter; i++) {
    rand.RandGaussian(&v2);
  }
  // flops = number of generated random numbers per second,
  Real flops = iter * (v.Dim() + v2.Dim()) / t.Elapsed();
  KALDI_LOG << __func__ << NameOf<Real>()
            << " Speed was " << flops << " rand_elems/s. "
            << "(debug " << MeanVariance(v) << ")";
}

} // namespace kaldi


int main() {
  int32 iter = 10; // Be quick on CPU,
#if HAVE_CUDA == 1
  for (int32 loop = 0; loop < 2; loop++) { // NO for loop if 'HAVE_CUDA != 1',
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if ( loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else {
      CuDevice::Instantiate().SelectGpuId("yes");
      iter = 400; // GPUs are faster,
    }
#endif
    Timer t;
    kaldi::CuRandUniformMatrixSpeedTest<float>(iter);
    kaldi::CuRandUniformMatrixBaseSpeedTest<float>(iter);
    kaldi::CuRandUniformVectorSpeedTest<float>(iter);
    kaldi::CuRandGaussianMatrixSpeedTest<float>(iter);
    kaldi::CuRandGaussianMatrixBaseSpeedTest<float>(iter);
    kaldi::CuRandGaussianVectorSpeedTest<float>(iter);
    fprintf(stderr, "---\n");

    kaldi::CuRandUniformMatrixSpeedTest<double>(iter);
    kaldi::CuRandUniformMatrixBaseSpeedTest<double>(iter);
    kaldi::CuRandUniformVectorSpeedTest<double>(iter);
    kaldi::CuRandGaussianMatrixSpeedTest<double>(iter);
    kaldi::CuRandGaussianMatrixBaseSpeedTest<double>(iter);
    kaldi::CuRandGaussianVectorSpeedTest<double>(iter);
    fprintf(stderr, "--- ELAPSED %fs.\n\n", t.Elapsed());
#if HAVE_CUDA == 1
  } // No for loop if 'HAVE_CUDA != 1',
  CuDevice::Instantiate().PrintProfile();
#endif
  KALDI_LOG << "Tests succeeded.";
}

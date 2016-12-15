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

template <typename Real>
void CuRandUniformMatrixSpeedTest() {
  Timer t;
  CuRand<Real> rand;
  CuMatrix<Real> m(249,2011);
  for (int32 i = 0; i < 200; i++) {
    rand.RandUniform(&m);
  }
  KALDI_LOG << __func__ << NameOf<Real>() << " t = " << t.Elapsed() << "s, " << MeanVariance(m);
}

template <typename Real>
void CuRandGaussianMatrixSpeedTest() {
  Timer t;
  CuRand<Real> rand;
  CuMatrix<Real> m(249,2011);
  for (int32 i = 0; i < 200; i++) {
    rand.RandGaussian(&m);
  }
  KALDI_LOG << __func__ << NameOf<Real>() << " t = " << t.Elapsed() << "s, " << MeanVariance(m);
}

template <typename Real>
void CuRandGaussianVectorSpeedTest() {
  Timer t;
  CuRand<Real> rand;
  CuVector<Real> v(2011);
  for (int32 i = 0; i < 200; i++) {
    rand.RandGaussian(&v);
  }
  KALDI_LOG << __func__ << NameOf<Real>() << " t = " << t.Elapsed() << "s";
}

} // namespace kaldi


int main() {
  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    kaldi::CuRandUniformMatrixSpeedTest<float>();
    kaldi::CuRandGaussianMatrixSpeedTest<float>();
    kaldi::CuRandGaussianVectorSpeedTest<float>();
    fprintf(stderr, "---\n");

    kaldi::CuRandUniformMatrixSpeedTest<double>();
    kaldi::CuRandGaussianMatrixSpeedTest<double>();
    kaldi::CuRandGaussianVectorSpeedTest<double>();
    fprintf(stderr, "\n");
  }

#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  std::cout << "Tests succeeded.\n";
}

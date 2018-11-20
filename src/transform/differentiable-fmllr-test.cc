// transform/differentiable-fmllr-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

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
//1 limitations under the License.

#include "transform/differentiable-fmllr.h"

namespace kaldi {
namespace differentiable_transform {



// Test derivatives produced by the Estimator object.
//
void TestCoreFmllrEstimatorDerivs(
    BaseFloat gamma,
    const Matrix<BaseFloat> &G,
    const Matrix<BaseFloat> &K,
    const Matrix<BaseFloat> &A,
    CoreFmllrEstimator *estimator) {
  // TODO.

}


void UnitTestCoreFmllrEstimatorSimple() {
  int32 dim = RandInt(10, 20);
  BaseFloat gamma = RandInt(5, 10);
  Matrix<BaseFloat> G(dim, dim),
      K(dim, dim), A(dim, dim, kUndefined);
  G.AddToDiag(1.234 * gamma);
  K.AddToDiag(0.234 * gamma);
  CoreFmllrEstimatorOptions opts;
  CoreFmllrEstimator estimator(opts, gamma, G, K, &A);
  estimator.Forward();
  KALDI_LOG << "A is " << A;
  KALDI_ASSERT(A.IsUnit(0.01));
  TestCoreFmllrEstimatorDerivs(G, K, A, &estimator);
}


}  // namespace kaldi
}  // namespace differentiable_transform



int main() {
  using namespace kaldi::differentiable_transform;

  UnitTestCoreFmllrEstimatorSimple();
  std::cout << "Test OK.\n";
}

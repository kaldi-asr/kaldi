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



// Test derivatives produced by the Estimator object for K.
void TestCoreFmllrEstimatorKDeriv(
    BaseFloat gamma,
    const Matrix<BaseFloat> &G,
    const Matrix<BaseFloat> &K,
    const Matrix<BaseFloat> &A,
    CoreFmllrEstimator *estimator) {

  int32 num_directions = 4;
  Vector<BaseFloat> expected_changes(num_directions),
      actual_changes(num_directions);

  int32 dim = G.NumRows();
  BaseFloat epsilon = 1.0e-04 * gamma;
  Matrix<BaseFloat> A_deriv(dim, dim);
  // A_deriv defines the objective function: a random linear function in A.
  A_deriv.SetRandn();

  Matrix<BaseFloat> G_deriv(dim, dim),
      K_deriv(dim, dim);
  estimator->Backward(A_deriv, &G_deriv, &K_deriv);

  for (int32 i = 0; i < num_directions; i++) {
    Matrix<BaseFloat> K_new(dim, dim);
    K_new.SetRandn();
    K_new.Scale(epsilon);
    expected_changes(i) = TraceMatMat(K_new, K_deriv, kTrans);
    K_new.AddMat(1.0, K);
    CoreFmllrEstimatorOptions opts;
    Matrix<BaseFloat> A_new(dim, dim);
    CoreFmllrEstimator estimator2(opts, gamma, G, K_new, &A_new);
    estimator2.Forward();
    A_new.AddMat(-1.0, A);
    // compute the change in our random linear objective function defined by
    // A_deriv, that would be produced by taking some small random change in K
    // and computing the A that results from that.
    actual_changes(i) = TraceMatMat(A_new, A_deriv, kTrans);
  }

  if (!expected_changes.ApproxEqual(actual_changes, 0.1)) {
    KALDI_ERR << "Expected and actual changes differ too much: "
               << expected_changes << " vs. "
               << actual_changes;
  }
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
  BaseFloat objf_impr = estimator.Forward();
  KALDI_LOG << "A is " << A;
  KALDI_ASSERT(A.IsUnit(0.01));
  KALDI_ASSERT(fabs(objf_impr) < 0.01);
  for (int32 i = 0; i < 5; i++) {
    TestCoreFmllrEstimatorKDeriv(gamma, G, K, A, &estimator);
    // TestCoreFmllrEstimatorGDeriv(G, K, A, &estimator);
  }
}


}  // namespace kaldi
}  // namespace differentiable_transform



int main() {
  using namespace kaldi::differentiable_transform;

  UnitTestCoreFmllrEstimatorSimple();
  std::cout << "Test OK.\n";
}

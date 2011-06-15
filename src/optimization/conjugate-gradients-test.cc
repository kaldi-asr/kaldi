// optimization/conjugate-gradients-test.cc

// Copyright 2009-2011  Georg Stemer;  Go Vivace Inc.

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

#include "optimization/conjugate-gradients.h"
namespace kaldi {

class QuadraticConjugateGradient: public OptimizableInterface<BaseFloat> {
 public:
  QuadraticConjugateGradient() {
    dim = 1 + rand() % 40;
    Matrix<BaseFloat> N(dim, dim+5);
    N.SetRandn();
    B.Resize(dim, dim);
    B.AddMatMat(1.0, N, kNoTrans, N, kTrans, 0.0);
    a.Resize(dim);
    a.SetRandn();
  }
  virtual void ComputeGradient(const Vector<BaseFloat> &params,
                               Vector<BaseFloat> *gradient_out) {
    if (gradient_out) {
      gradient_out->SetZero();
      gradient_out->AddVec(-1.0, a);  // gradient_out += a
      gradient_out->AddMatVec(1, B, kNoTrans, params, 1.0);
      // gradient_out -= B*params.
    }
  }
  virtual BaseFloat ComputeValue(const Vector<BaseFloat> &params) {
    return(-1.0*VecVec(params, a) +0.5 * VecMatVec(params, B, params));
  }
  size_t dim;
  Matrix<BaseFloat> B;
  Vector<BaseFloat> a;
};

void UnitTestConjugateGradients() {
  bool converged = false;
  for (size_t p = 0; p < 50; p++) {
    QuadraticConjugateGradient qc;
    NCGOptions opts;
    opts.verbose = 0;
    opts.min_iter = 10;
    Vector<BaseFloat> param(qc.dim);
    param.SetRandn();
    //    std::cout << "Initial param: " << param;
    do {
      converged = NonlinearConjugateGradients(opts, &qc, &param);
    } while (!converged);
    //    std::cout << "Optimized param: " << param;
    Vector<BaseFloat> param_optimal(qc.dim);
    Matrix<BaseFloat> Binv(qc.B);
    Binv.Invert();
    // param_optimal = B^{-1} a.
    param_optimal.AddMatVec(1.0, Binv, kNoTrans, qc.a, 0.0);
    std::cout << "Optimized Value: " << qc.ComputeValue(param) <<
        "   Optimal Value: " << qc.ComputeValue(param_optimal);
    Vector<BaseFloat> diff(param);
    diff.AddVec(-1.0, param_optimal);
    //    std::cout << "Diff: " << diff;
    std::cout << " Difference: " << VecVec(diff, diff)/qc.dim << "\n";
    assert(VecVec(diff, diff)/qc.dim < 1e-5);
  }
}
}  // end namespace kaldi

int main() {
  // Removing the test for now; it's failing and we know about it...
  kaldi::UnitTestConjugateGradients();
  std::cout << "Test OK.\n";
}


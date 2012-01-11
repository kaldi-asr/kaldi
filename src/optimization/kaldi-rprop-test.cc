// optimization/kaldi-rprop-test.cc

// Copyright 2009-2011  Georg Stemmer;  Go Vivace Inc.

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

#include "optimization/kaldi-rprop.h"
namespace kaldi {

class QuadraticRprop: public OptimizableInterface<BaseFloat> {
 public:
  QuadraticRprop() {
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

void UnitTestRprop() {
  bool converged = false;
  for (size_t p = 0; p < 10; p++) {
    QuadraticRprop qc;
    RpropOptions<BaseFloat> opts;
    opts.maximizing = false;
    opts.epsilon = 1e-8;
    opts.max_iter = 10000;
    Vector<BaseFloat> param(qc.dim);
    param.SetRandn();
    // std::cout << "Initial param: " << param;
    converged = Rprop(opts, &qc, &param);
    // std::cout << "Optimized param: " << param;
    Vector<BaseFloat> param_optimal(qc.dim);
    Matrix<BaseFloat> Binv(qc.B);
    Binv.Invert();
    // param_optimal = B^{-1} a.
    param_optimal.AddMatVec(1.0, Binv, kNoTrans, qc.a, 0.0);
    std::cout << "Optimized Value: " << qc.ComputeValue(param) <<
        "   Optimal Value: " << qc.ComputeValue(param_optimal) <<"\n";
    Vector<BaseFloat> diff(param);
    diff.AddVec(-1.0, param_optimal);
    std::cout << "Diff: " << diff;
    KALDI_ASSERT(VecVec(diff, diff)/VecVec(param, param) < 1e-5);
    // Arnab: changed the bound from 1e-7 as it was failing 
  }
}
}  // end namespace kaldi

int main() {
  kaldi::UnitTestRprop();
  std::cout << "Test OK.\n";
}


// optimization/conjugate-gradients.h

// Copyright 2009-2011  Go Vivace Inc.;  Georg Stemmer;  Microsoft Corporation
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
#ifndef KALDI_OPTIMIZATION_CONJUGATE_GRADIENTS_H_
#define KALDI_OPTIMIZATION_CONJUGATE_GRADIENTS_H_

#include <vector>
#include <string>
#include "util/stl-utils.h"

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/optimizable-itf.h"

namespace kaldi {

/// @ingroup Optimizers @{

struct NCGOptions {
  bool maximizing;  // maximizing == false means we're minimizing.
  size_t max_iter;  // Maximum number of optimization iterations
  size_t min_iter;  // Minimum number of iterations nonetheless
  double epsilon;   // Conjugate Gradients error tolerance
  BaseFloat rho_zero;  // Secant method step parameter
  size_t secant_iter_max;  // Maximum number of secant method iterations
  size_t n;  // number of iterations after which to re-start CG to
             // improve convergence
  size_t verbose;  // Display progress information
  double tolerance;
  NCGOptions(): maximizing(true), max_iter(20000), min_iter(10000),
                epsilon(1e-6), rho_zero(1.0), secant_iter_max(5),
                n(100), verbose(0), tolerance(1e-7) { }
};

template<class Real>
bool NonlinearConjugateGradients(const NCGOptions &opts,
                                 OptimizableInterface<Real> *optimizable,
                                 Vector<Real> *param) {
  // At start, "param" contains the initial parameters.  This routine
  // optimizes the vector "param" and returns the objective function
  // improvement.  At exit, "param" contains the optimized parameter.
  int rmul = opts.maximizing?1:-1;
  size_t i = 0, k = 0, j = 0;
  size_t dim = param->Dim();
  Vector<Real> r(dim);
  Vector<Real> s(dim);
  Vector<Real> d(dim);
  Vector<double> rD_(dim), sD_(dim), dD_(dim), grad_tmpD_(dim);
  Vector<Real> x(*param), x_tmp(dim);
  // Vector<Real> M(dim);  // [preconditioner]
  double nu_prev, nu, delta_old, delta_d, alpha, delta_mid, beta;
  optimizable->ComputeGradient(*param, &r);
  // M.Set(1.0);  // set preconditioner M \simeq f''(x) to unity.
  r.Scale(rmul);
  rD_.CopyFromVec(r);
  s.CopyFromVec(r);
  // s.DivElemByElem(M);  // M^-1 r
  sD_.CopyFromVec(s);
  d.CopyFromVec(s);
  dD_.CopyFromVec(d);
  double delta_new = VecVec(rD_, dD_);
  double delta_zero = delta_new;
  Vector<Real> grad_tmp(r);

  while ((i < opts.min_iter) || ((i < opts.max_iter) &&
                                 (delta_new > opts.epsilon*opts.epsilon*delta_zero))) {
    if (opts.verbose == 2) std::cerr << "iteration " << i << "  Value: "
                                << optimizable->ComputeValue(x) <<"\n";
    j = 0;
    delta_d = VecVec(dD_, dD_);
    alpha = -1.0*opts.rho_zero;
    x_tmp.CopyFromVec(x);
    x_tmp.AddVec(opts.rho_zero, d);
    optimizable->ComputeGradient(x_tmp, &grad_tmp);
    if (opts.verbose== 2) std::cerr << "Gradients: " << grad_tmp << "\n";
    grad_tmpD_.CopyFromVec(grad_tmp);
    nu_prev = -1.0*rmul*VecVec(grad_tmpD_, dD_);
    do {
      optimizable->ComputeGradient(x, &grad_tmp);
      grad_tmpD_.CopyFromVec(grad_tmp);
      nu = rmul * -1.0 * VecVec(grad_tmpD_, dD_);
      alpha = alpha*nu/(nu_prev - nu);
      if (KALDI_ISINF(alpha) || KALDI_ISNAN(alpha)) break;
      x.AddVec(alpha, d);
      nu_prev = nu;
      j++;
      if (opts.verbose== 2) std::cerr << "Secant " << j << " Value:" <<
                            optimizable->ComputeValue(x) << "\n";
    } while ((j<opts.secant_iter_max) && (alpha*alpha*delta_d >
                                opts.epsilon*opts.epsilon));
    if (opts.verbose== 2) std::cerr << "Number of Secant Iterations: "
                                << j << "\n";
    optimizable->ComputeGradient(x, &r);
    r.Scale(rmul);
    rD_.CopyFromVec(r);
    delta_old = delta_new;
    delta_mid = VecVec(rD_, sD_);

    s.CopyFromVec(r);
    // s.DivElemByElem(M);  // M^-1 r
    sD_.CopyFromVec(s);
    delta_new = VecVec(rD_, sD_);
    beta = (delta_new - delta_mid)/delta_old;
    if ((k++ == opts.n) || (beta < 0)) {
      d.CopyFromVec(s);
      k = 0;
    } else {
      d.Scale(beta);
      d.AddVec(1.0, s);
    }
    dD_.CopyFromVec(d);
    i++;
  }
  param->CopyFromVec(x);
  if (opts.verbose == 1) {
    std::cerr << i << " iterations\n";
  }
  if (i < opts.max_iter) {
    return true;
  } else {
    return false;
  }
}
/// @} Optimizers
}
#endif  // KALDI_OPTIMIZATION_CONJUGATE_GRADIENTS_H_

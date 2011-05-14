// optimization/nonlinear-conjugate-gradients.h

// Copyright 2009-2011  Go Vivace Inc.  Georg Stemmer
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
#ifndef KALDI_OPTIMIZATION_NONLINEAR_CONJUGATE_GRADIENTS_H_
#define KALDI_OPTIMIZATION_NONLINEAR_CONJUGATE_GRADIENTS_H_

#include <vector>
#include <string>
#include "util/stl-utils.h"

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/optimizable-itf.h"

namespace kaldi {

/// @ingroup Optimizers @{
template<class Real>
struct NonlinearConjugateGradientsOptions {
  bool maximizing;  // maximizing == false means
                                       // we're minimizing.
  size_t max_iter;  // Maximum number of optimization iterations
  Real tolerance;   // Conjugate Gradients error tolerance
  Real abstol;    // Absolute tolerance of delta new
  Real alpha_init;  // Secant method step parameter
  size_t secant_iter_max;  // Maximum number of secant method iterations
  size_t n;  // number of iterations after which to re-start CG to
             // improve convergence
  size_t verbose;  // Display progress information
  NonlinearConjugateGradientsOptions(): maximizing(false), max_iter(1000),
                                        tolerance(1e-5), abstol(1e-15),
                                        alpha_init(1.0e-10), secant_iter_max(10),
                                        n(100), verbose(0) { }
};

template<class Real>
static void Abs(Vector<Real> & p) {
  for (size_t i = 0; i < p.Dim(); i++) {
    if (p(i) < 0)
      p(i) = static_cast<Real>(-1.0) * p(i);
  }
}

template<class Real>
bool NonlinearConjugateGradients(const NonlinearConjugateGradientsOptions<Real>
                                 &opts, OptimizableInterface<Real> *optimizable,
                                 Vector<Real> *param) {
  // At start, "param" contains the initial parameters.  This routine
  // optimizes the vector "param" and returns the objective function
  // improvement.  At exit, "param" contains the optimized parameter.
  BaseFloat rmul = opts.maximizing?1.0:-1.0;
  size_t j = 0, k = 0, do_backtrack, num_iter = 0;
  size_t dim = param->Dim();
  Vector<Real> r(dim);
  Vector<Real> x_tmp(dim), x0(dim);
  Vector<Real> M(dim);
  Vector<Real> s(dim);
  Vector<Real> direction(dim);
  Vector<Real> old_dx(dim), dx(dim), olddx(dim);
  Vector<Real> x(*param);
  Real eta_prev, eta, eta_zero, delta_old, alpha, delta_mid, beta, cond_d;
  Real gamma = 10.0, pre_alpha, c1 = 1e-4, c2 = 1e-1;
  Real old_alpha, old_obj, obj, obj0, delta_x;
  bool converged = false;
  optimizable->ComputeGradient(*param, &r);
  /// Calculate the pre-conditioner M = f''(x). Here I arbitrarily
  /// set it to unity, but carry on with the math. this will need to
  /// be fixed in the future. One option is the diagonal entries of
  /// the derivative. then do M^{-1}
  //  Matrix<Real> hessian(dim, dim);
  //  hessian.SetUnit();
  //  s.AddMatVec(1.0, hessian, kNoTrans, r, 0.0);
  cond_d = opts.tolerance*10;
  x.CopyFromVec(*param);
  //  x_tmp.CopyFromVec(x);
  //  x_tmp.Add(cond_d);
  optimizable->ComputeGradient(x, &dx);
  //  dx.CopyFromVec(M);
  obj = optimizable->ComputeValue(x);
  //  M.AddVec(-1.0, r);
  //  M.Scale(1.0/cond_d);
  //  Abs(M);
  //  if (opts.verbose) std::cerr << "M: " << M;
  // M has now been calculated as a vector.
  r.Scale(rmul);
  s.CopyFromVec(r);
  //  s.DivElemByElem(M);  // M^-1 r
  direction.CopyFromVec(s);
  Real delta_new = VecVec(r, direction);
  Real delta_zero = delta_new;
  Vector<Real> dx0(r);
  //  x_tmp.CopyFromVec(x);

  while ((num_iter < opts.max_iter) &&
         (std::fabs(delta_new) > opts.tolerance*opts.tolerance*delta_zero) &&
         std::fabs(delta_new) > opts.abstol) {
    if (opts.verbose) std::cerr << "iteration " << num_iter << "  Value: "
                                << obj <<"\n";
    num_iter++; k++;  // k is j in the matlab code
    //    delta_d = VecVec(direction, direction);
    alpha = (alpha < opts.alpha_init)?opts.alpha_init:alpha;
    // Begin Line Search -[alpha, obj, dx, ogc] =
    // f(x, obj, dx, d, ogfun, ogparams, 'alpha0', alpha, 'verbose', verbose, lsparams{:})
    x0.CopyFromVec(x); obj0 = obj; dx0.CopyFromVec(dx);

    // Do the line search in the next loop
    //    if (opts.verbose) std::cerr << "Gradients: " << dx0 << "\n";
    eta_zero = eta_prev = VecVec(dx0, direction);
    if (KALDI_ISINF(alpha) || KALDI_ISNAN(alpha)) alpha = 1e-10;
    // FindNonZeroAlpha -- If alpha is too small, find a non-zero change alpha
    delta_x = 0;
    while (delta_x == 0) {
      x_tmp.CopyFromVec(x);
      x_tmp.AddVec(alpha, direction);
      x_tmp.AddVec(-1.0, x);
      delta_x = VecVec(x_tmp, x_tmp);
      if (delta_x == 0) { alpha = gamma*alpha; }
    }
    x_tmp.CopyFromVec(x);
    x_tmp.AddVec(alpha, direction);
    obj = optimizable->ComputeValue(x_tmp);
    optimizable->ComputeGradient(x_tmp, &dx);
    // saveAlpha
    old_alpha = alpha; old_dx.CopyFromVec(dx); old_obj = obj;
    // backtrack - make sure current step leads to decrease in objective
    // Ensure either Armijo lower objective or infitsmall alpha
    do_backtrack = 0;
    pre_alpha = alpha;
    x_tmp.CopyFromVec(x);
    x_tmp.AddVec(alpha, direction);
    x_tmp.AddVec(-1.0, x);
    delta_x = VecVec(x_tmp, x_tmp);
    while ((obj > (obj0 + c1*alpha*eta_zero)) & (delta_x > 0)) {
      if ((old_alpha > alpha/gamma) & (old_obj < obj0 + c1*old_alpha*eta_zero)) {
        // restore alpha
        alpha = old_alpha; dx.CopyFromVec(old_dx); obj = old_obj;
        do_backtrack = 1;  // i.e. we did the backtrack
        break;
      }
      alpha = alpha/gamma;
      x_tmp.CopyFromVec(x);
      x_tmp.AddVec(alpha, direction);
      optimizable->ComputeGradient(x_tmp, &dx);
      obj = optimizable->ComputeValue(x_tmp);
      // prepare for the next loop
      x_tmp.CopyFromVec(x);
      x_tmp.AddVec(alpha, direction);
      x_tmp.AddVec(-1.0, x);
      delta_x = VecVec(x_tmp, x_tmp);
    }
    // Backtracking finished
    beta = alpha;
    eta = VecVec(dx, direction);
    //    alpha = alpha*eta/(eta_prev - eta);
    //    x.AddVec(alpha, direction);
    j = 0;  // j is same as i here. number of secant iterations
    x_tmp.CopyFromVec(x);
    x_tmp.AddVec(alpha, direction);
    x_tmp.AddVec(-1.0, x);
    delta_x = VecVec(x_tmp, x_tmp);  // don't forget the same math at
                                   // the end of the loop
    while ((std::fabs(eta) > c2*std::fabs(eta_zero)) &(j<opts.secant_iter_max) &
           (obj<=obj0 + 1e-12) & (eta_prev != eta) & (delta_x > 0)) {

      beta = eta*beta/(eta_prev - eta);
      // saveAlpha
      old_alpha = alpha; olddx.CopyFromVec(dx); old_obj = obj;
      alpha += beta;
      if (alpha < 0) alpha = 1;  // This is bad
      eta_prev = eta;
      j++;
      x_tmp.CopyFromVec(x);
      x_tmp.AddVec(alpha, direction);
      optimizable->ComputeGradient(x_tmp, &dx);
      obj = optimizable->ComputeValue(x_tmp);
      eta = VecVec(dx, direction);
      if (opts.verbose > 1) std::cerr << "Secant " << j << " Value:" <<
                            obj << "\n";
    }
    if (opts.verbose > 1) std::cerr << "Number of Secant Iterations: "
                                << j << "\n";
    // FindNonZeroAlpha -- If alpha is too small, find a non-zero change alpha
    delta_x = 0;
    while (delta_x == 0) {
      x_tmp.CopyFromVec(x);
      x_tmp.AddVec(alpha, direction);
      x_tmp.AddVec(-1.0, x);
      delta_x = VecVec(x_tmp, x_tmp);
      if (delta_x == 0) { alpha = gamma*alpha; }
    }
    //    obj = optimizable->ComputeValue(x_tmp);
    //    optimizable->ComputeGradient(x_tmp, &dx);
    // backtrack - make sure current step leads to decrease in objective
    // Ensure either Armijo lower objective or infitsmall alpha
    // backtrack - make sure current step leads to decrease in objective
    // Ensure either Armijo lower objective or infitsmall alpha
    do_backtrack = 0;
    pre_alpha = alpha;
    x_tmp.CopyFromVec(x);
    x_tmp.AddVec(alpha, direction);
    x_tmp.AddVec(-1.0, x);
    delta_x = VecVec(x_tmp, x_tmp);
    while ((obj > (obj0 + c1*alpha*eta_zero)) & (delta_x > 0)) {
      if ((old_alpha > alpha/gamma) & (old_obj < obj0 + c1*old_alpha*eta_zero)) {
        // restore alpha
        alpha = old_alpha; dx.CopyFromVec(old_dx); obj = old_obj;
        do_backtrack = 1;  // i.e. we did the backtrack
        break;
      }
      alpha = alpha/gamma;
      x_tmp.CopyFromVec(x);
      x_tmp.AddVec(alpha, direction);
      optimizable->ComputeGradient(x_tmp, &dx);
      obj = optimizable->ComputeValue(x_tmp);
      // prepare for the next loop
      x_tmp.CopyFromVec(x);
      x_tmp.AddVec(alpha, direction);
      x_tmp.AddVec(-1.0, x);
      delta_x = VecVec(x_tmp, x_tmp);
    }
    // Backtracking finished
    // CheckConditions
    if (obj >= obj0) { std::cerr <<
          "Finished linesearch without decreasing the objective\n";
    }
    converged = false;
    if (eta_zero == eta) {
      converged = true;
      std::cerr << "No change in directional derivative\n";
    }
    if (delta_x == 0) {
      converged = true;
      std::cerr << " No change in Position!!!\n";
    }
    if (converged && num_iter > 10) break;
    // END OF CheckConditions and LINE SEARCH
    //    x.CopyFromVec(x_tmp);  // update x since alpha has been found
    x.AddVec(alpha, direction);  // Update x
    // Compute M = f''(x) if possible. Then s = M^-1 * r
    //    x_tmp.CopyFromVec(x);
    //    x_tmp.Add(cond_d);
    //    optimizable->ComputeGradient(x_tmp, &M);
    //    M.AddVec(-1.0, r);
    //    M.Scale(1.0/cond_d);
    //    Abs(M);
    // M has now been calculated as a vector.
    r.CopyFromVec(dx);
    r.Scale(rmul);
    delta_old = delta_new;
    s.CopyFromVec(r);
    //    s.DivElemByElem(M);  // M^-1 r
    delta_mid = VecVec(r, s);
    delta_new = VecVec(r, s);
    beta = (delta_new - delta_mid)/delta_old;
    if ((delta_mid/delta_new >= 0.1) || VecVec(direction, dx) >= 0) {
          direction.CopyFromVec(s);
          k = 0;
          } else {
          direction.Scale(beta);
          direction.AddVec(1.0, s);
          }
  }
  param->CopyFromVec(x);
  //  if (opts.verbose) {
    std::cerr << num_iter << " iterations\n";
    //  }
  if ((num_iter < opts.max_iter)) {
    return true;
  } else {
    return false;
  }
}
/// @} Optimizers
}
#endif  // KALDI_OPTIMIZATION_NONLINEAR_CONJUGATE_GRADIENTS_H_

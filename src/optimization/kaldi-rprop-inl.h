// optimization/kaldi-rprop-inl.h

// Copyright 2009-2011  Georg Stemmer
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

#ifndef KALDI_OPTIMIZATION_KALDI_RPROP_INL_H_
#define KALDI_OPTIMIZATION_KALDI_RPROP_INL_H_
namespace kaldi {

template<class Real>
bool Rprop(const RpropOptions<Real> &opts,
           OptimizableInterface<Real> *optimizable,
           Vector<Real> *param) {
  size_t iter = 0;
  bool converged = false;
  Real last_value;
  Real cur_value;
  Real mul_factor = opts.maximizing? 1.0 : -1.0;
  size_t dim = param->Dim();

  // current gradient
  Vector<Real> grad_current(dim);

  // gradient from the previous time frame
  Vector<Real> grad_last(dim);

  // current step size
  Vector<Real> delta(dim);

  grad_last.SetZero();
  for (size_t i = 0; i < dim; i++) {
    delta(i) = opts.gamma_init;
  }

  // compute current valuelihood
  cur_value = optimizable->ComputeValue(*param);

  /*std::cerr << "Iteration " << iter << " : Value: " << cur_value << '\n';
param->Scale(1.0/1e5);
   cur_value = optimizable->ComputeValue(*param);
 std::cerr << "Iteration " << iter << " : Value: " << cur_value << '\n';
 exit(1);*/
  while ((iter < opts.max_iter)&&(!converged)) {
    last_value = cur_value;

    if (opts.verbose && (iter % opts.conv_check_interval == 0)) {
      std::cerr << "Iteration " << iter << " : Value: "
                << cur_value << '\n';
    }

    // compute current gradient
    optimizable->ComputeGradient(*param, &grad_current);

    for (size_t i = 0; i < dim; i++) {
      Real sgn = grad_current(i) * grad_last(i);

      // if the current and the previous gradient point in the
      // same direction...
      if (sgn > 0) {
        // increase the stepsize as long as it is below gamma_max
        if (delta(i)*opts.eta_inc < opts.gamma_max) {
          delta(i) = delta(i)*opts.eta_inc;
        } else {
          delta(i) = opts.gamma_max;
        }
        // compute the new params as a sum of the old params and
        // the stepsize
        Real sgn2 = -1.0;
        if (grad_current(i) > 0) {
          sgn2 = 1.0;
        } else {
          sgn2 = -1.0;
        }
        param->operator()(i) =
            param->operator()(i) + mul_factor * delta(i) * sgn2;
        grad_last(i) = grad_current(i);
      } else {
        // if the old and the current gradient point in different
        // directions
        if (sgn < 0) {
          // backtrack the last weight
          Real sgn2 = -1.0;
          if (grad_current(i) > 0) {
            sgn2 = 1.0;
          } else {
            sgn2 = -1.0;
          }
          param->operator()(i) =
              param->operator()(i) + mul_factor * delta(i) * sgn2;
          // decrease stepsize until it reaches
          // gamma_min
          if (delta(i)*opts.eta_dec < opts.gamma_min) {
            delta(i) = opts.gamma_min;
          } else {
            delta(i) = delta(i)*opts.eta_dec;
          }
          // ensure that in the next iteration the new delta
          // will be applied
          grad_last(i) = 0.0;
        } else {
          // one of the two gradients is 0.
          // do not change stepsize
          Real sgn2 = -1.0;
          if (grad_current(i) > 0) {
            sgn2 = 1.0;
          } else {
            sgn2 = -1.0;
          }
          param->operator()(i) =
              param->operator()(i) + mul_factor * sgn2 * delta(i);
          grad_last(i) = grad_current(i);
        }
      }
    }
    // compute current value
    if (iter % opts.conv_check_interval == 0) {
      try {
        cur_value = optimizable->ComputeValue(*param);
      } catch(const std::runtime_error &) {
        KALDI_WARN << "Problems in value computation (ignoring value)\n";
        cur_value = last_value;
        last_value = last_value - 2*opts.epsilon;
      }
      // check for convergence
      if ((iter > 2)&&(iter >= opts.min_iter)) {
        if (std::abs(cur_value - last_value) < opts.epsilon) {
          converged = true;
        }
      }
    }
    ++iter;
  }
  return converged;
}
}

#endif  // KALDI_OPTIMIZATION_KALDI_RPROP_INL_H_

// optimization/kaldi-rprop.h

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
#ifndef KALDI_OPTIMIZATION_KALDI_RPROP_H_
#define KALDI_OPTIMIZATION_KALDI_RPROP_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/optimizable-itf.h"

namespace kaldi {

/// @defgroup Rprop Rprop
/// @{
/// @ingroup Optimizers
/// The Resilient Propagation (Rprop) optimization algorithm

/// Options for the Rprop algorithm
template<class Real>
struct RpropOptions {
  /// maximizing == false means we're minimizing.
  bool maximizing;

  /// maximum number of optimization iterations
  size_t max_iter;

  /// minimum number of optimization iterations
  size_t min_iter;

  /// check for convergence only every n-th step
  /// (the value of the optimizable object will
  /// only be evaluated every n-th step, so
  /// computation will be faster for large n)
  size_t conv_check_interval;

  /// error tolerance
  Real epsilon;

  /// factor for increasing stepsize
  Real eta_inc;

  /// factor for decreasing stepsize
  Real eta_dec;

  /// maximum stepsize
  Real gamma_max;

  /// minimum stepsize
  Real gamma_min;

  /// initial stepsize
  Real gamma_init;

  /// reasonable defaults
  /// as they can be found in the literature
  RpropOptions(): maximizing(true), max_iter(1500),
                  min_iter(10),
                  conv_check_interval(100),
                  epsilon(1e-10), eta_inc(1.2), eta_dec(0.5),
                  gamma_max(50), gamma_min(1e-10), gamma_init(0.1) { }
};

/// Resilient Propagation Algorithm (Rprop)
/// This function implements the improved version
/// of Rprop ("Rprop with weight backtracking")
/// as it is described in the following articles:
/// Igel, Huesken: Empirical Evaluation of the Improved
/// RPROP Learning Algorithms, Neurocomputing 50 (2003),
/// pp. 105-123.
/// and
/// Riedmiller, Braun: A Direct Adaptive Method for
/// Faster Backpropagation Learning: the RPROP Algorithm,
/// in E.H. Ruspini (Ed.), Proc. of the IEEE International
/// Conference on Neural Networks, New York, 1993,
/// pp. 586-591.
/// "param" contains the initial parameters.
/// The routine optimizes the vector "param" and
///  At exit, "param" contains the optimal parameter.
template<class Real>
bool Rprop(const RpropOptions<Real> &opts,
           OptimizableInterface<Real> *optimizable,
           Vector<Real> *param);

/// @} Optimizers
}

// we need to include the implementation
#include "optimization/kaldi-rprop-inl.h"


#endif  // KALDI_OPTIMIZATION_KALDI_RPROP_H_

// optimization/optimization-common.h

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


#ifndef KALDI_OPTIMIZATION_OPTIMIZATION_COMMON_H_
#define KALDI_OPTIMIZATION_OPTIMIZATION_COMMON_H_

namespace kaldi {
/// @defgroup Optimizers Optimizers
/// @{
/// kaldi Optimization Algorithms
/// These functions can be used for numerical optimization

enum {
  /// use Rprop (Resilient Propagation) algorithm
  kRprop = 0,

  /// use Conjugate Gradient algorithm
  kConjugateGradients = 1
};

/// Flags to determine
/// the selected optimization algorithm
typedef int16 OptimizationMethodType;

/// }@
}  // End namespace kaldi


#endif  // KALDI_OPTIMIZATION_OPTIMIZATION_COMMON_H_

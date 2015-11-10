// feat/mel-computations.h

// Copyright 2009-2011  Phonexia s.r.o.;  Microsoft Corporation

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

#ifndef KALDI_MATRIX_TOEPLITZ_H_
#define KALDI_MATRIX_TOEPLITZ_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

namespace kaldi {
/// \addtogroup matrix_group
/// @{

template<typename Real>
void toeplitz_solve(const Vector<Real> &r, const Vector<Real> &c, const Vector<Real> &y, Vector<Real> *x, Real tol_factor = 1000);

template<typename Real>
void make_toeplitz_matrix(const Vector<Real> &r, Matrix<Real> *rmat);

template<typename Real>
void make_nonsym_toeplitz_matrix(const Vector<Real> &r, const Vector<Real> &c,  Matrix<Real> *rmat);

/// @} end of "addtogroup matrix_group".

}  // namespace kaldi

#endif

// matrix/sp-matrix-inl.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_MATRIX_SP_MATRIX_INL_H_
#define KALDI_MATRIX_SP_MATRIX_INL_H_

#include "matrix/tp-matrix.h"

namespace kaldi {

// All the lines in this file seem to be declaring template specializations.
// These tell the compiler that we'll implement the templated function
// separately for the different template arguments (float, double).

template<>
double SolveQuadraticProblem(const SpMatrix<double> &H, const VectorBase<double> &g,
                             VectorBase<double> *x, double K, double eps,
                             const char *debug_str, bool optimizeDelta);

template<>
float SolveQuadraticProblem(const SpMatrix<float> &H, const VectorBase<float> &g,
                            VectorBase<float> *x, float K, float eps,
                            const char *debug_str, bool optimizeDelta);

template<>
float SolveQuadraticMatrixProblem(const SpMatrix<float> &Q,
                                  const MatrixBase<float> &Y,
                                  const SpMatrix<float> &SigmaInv,
                                  MatrixBase<float> *M,
                                  float K, float eps, const char *debug_str,
                                  bool optimizeDelta);

template<>
double SolveQuadraticMatrixProblem(const SpMatrix<double> &Q,
                                  const MatrixBase<double> &Y,
                                  const SpMatrix<double> &SigmaInv,
                                  MatrixBase<double> *M,
                                  double K, double eps, const char *debug_str,
                                  bool optimizeDelta);

template<>
float SolveDoubleQuadraticMatrixProblem(const MatrixBase<float> &G,
                                        const SpMatrix<float> &P1,
                                        const SpMatrix<float> &P2,
                                        const SpMatrix<float> &Q1,
                                        const SpMatrix<float> &Q2,
                                        MatrixBase<float> *M,
                                        float K, float eps,
                                        const char *debug_str);

template<>
double SolveDoubleQuadraticMatrixProblem(const MatrixBase<double> &G,
                                         const SpMatrix<double> &P1,
                                         const SpMatrix<double> &P2,
                                         const SpMatrix<double> &Q1,
                                         const SpMatrix<double> &Q2,
                                         MatrixBase<double> *M,
                                         double K, double eps,
                                         const char *debug_str);


}  // namespace kaldi


#endif  // KALDI_MATRIX_SP_MATRIX_INL_H_

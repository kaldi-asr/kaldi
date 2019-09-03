// matrix/matrix-functions.h

// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.;  Jan Silovsky;
//                      Yanmin Qian;   1991 Henrique (Rico) Malvar (*)
//
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
//
// (*) incorporates, with permission, FFT code from his book
// "Signal Processing with Lapped Transforms", Artech, 1992.



#ifndef KALDI_MATRIX_MATRIX_FUNCTIONS_H_
#define KALDI_MATRIX_MATRIX_FUNCTIONS_H_

#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

/// @addtogroup matrix_funcs_misc
/// @{


template<typename Real> void ComputeDctMatrix(Matrix<Real> *M);


/// ComplexMul implements, inline, the complex multiplication b *= a.
template<typename Real> inline void ComplexMul(const Real &a_re, const Real &a_im,
                                               Real *b_re, Real *b_im);

/// ComplexMul implements, inline, the complex operation c += (a * b).
template<typename Real> inline void ComplexAddProduct(const Real &a_re, const Real &a_im,
                                                      const Real &b_re, const Real &b_im,
                                                      Real *c_re, Real *c_im);


/// ComplexImExp implements a <-- exp(i x), inline.
template<typename Real> inline void ComplexImExp(Real x, Real *a_re, Real *a_im);



/**
    ComputePCA does a PCA computation, using either outer products
    or inner products, whichever is more efficient.  Let D be
    the dimension of the data points, N be the number of data
    points, and G be the PCA dimension we want to retain.  We assume
    G <= N and G <= D.

    @param X [in]  An N x D matrix.  Each row of X is a point x_i.
    @param U [out] A G x D matrix.  Each row of U is a basis element u_i.
    @param A [out] An N x D matrix, or NULL.  Each row of A is a set of coefficients
         in the basis for a point x_i, so A(i, g) is the coefficient of u_i
         in x_i.
    @param print_eigs [in] If true, prints out diagnostic information about the
         eigenvalues.
    @param exact [in] If true, does the exact computation; if false, does
         a much faster (but almost exact) computation based on the Lanczos
         method.
*/

template<typename Real>
void ComputePca(const MatrixBase<Real> &X,
                MatrixBase<Real> *U,
                MatrixBase<Real> *A,
                bool print_eigs = false,
                bool exact = true);



// This function does: *plus += max(0, a b^T),
// *minus += max(0, -(a b^T)).
template<typename Real>
void AddOuterProductPlusMinus(Real alpha,
                              const VectorBase<Real> &a,
                              const VectorBase<Real> &b,
                              MatrixBase<Real> *plus,
                              MatrixBase<Real> *minus);

template<typename Real1, typename Real2>
inline void AssertSameDim(const MatrixBase<Real1> &mat1, const MatrixBase<Real2> &mat2) {
  KALDI_ASSERT(mat1.NumRows() == mat2.NumRows()
               && mat1.NumCols() == mat2.NumCols());
}


/// @} end of "addtogroup matrix_funcs_misc"

} // end namespace kaldi

#include "matrix/matrix-functions-inl.h"


#endif

// cblasext/cblas-extensions.h

// Copyright 2012-2019  Johns Hopkins University (author: Daniel Povey);
//                      Haihua Xu; Wei Shi

// See ../../COPYING for clarification regarding multiple authors
//
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
#ifndef KALDI_MATRIX_CBLAS_EXTENSIONS_H_
#define KALDI_MATRIX_CBLAS_EXTENSIONS_H_ 1


#include "cblasext/kaldi-blas.h"
#include "cblasext/cblas-wrappers.h"

// In directories other than this directory, this file is intended to mostly be
// included from .cc files, not from headers, since it includes cblas headers
// (via kaldi-blas.h) and those can be quite polluting.

// This file contains templated wrappers for CBLAS functions, which enable C++
// code calling these functions to be templated.
namespace kaldi {



// This has the same interface as cblas_Xgemv, i.e. it does y = alpha M x + beta y;
// it is just specialized for the case where the vector 'x' has a lot of zeros.
template<typename Real>
void cblasext_Xgemv_sparsevec(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                              KaldiBlasInt num_cols, Real alpha, const Real *Mdata,
                              KaldiBlasInt stride, const Real *xdata,
                              KaldiBlasInt incX, Real beta, Real *ydata,
                              KaldiBlasInt incY);



/**
   Does, elementwise for 0 <= i < dim,
     b[i] *= a[i].
*/
template <typename Real>
void cblasext_mul_elements_vec(
    const KaldiBlasInt dim,
    const Real *a,
    Real *b);


/**
   Does b *=  where a and b are matrices of the same dimension.
   Does not currently support transpose.

   Requires that a and b do not overlap (but this is not checked).
*/
template <typename Real>
void cblasext_mul_elements_mat(
    const Real *Adata,
    KaldiBlasInt a_num_rows, KaldiBlasInt a_num_cols, KaldiBlasInt a_stride,
    Real *Bdata,
    KaldiBlasInt b_stride);

/**
   For matrices A and B (possibly with column strides as well as
   row strides): if transB = false, compute
      tr(A B) = \sum_{i,j} A(i, j) B(j, i)
   or if transB = true, compute
      tr(A B) = \sum_{i,j} A(i, j) B(i, j).
     @param [in] Adata        Data pointer of matrix A
     @param [in] a_num_rows   Number of rows of matrix A
     @param [in] a_num_cols   Number of columns of matrix A
     @param [in] a_stride     Row stride of matrix A; may have any value.
     @param [in] a_col_stride Column stride of A, would be 1 for
                              a normal matrix; must be positive.
     @param [in] b_data        Data pointer of matrix B; may be
                              the same as Adata.
     @param [in] b_trans       True if B is transposed.  Note: the
                              expression would have the same value
                              if the transpose was applied to A
                              instead.
     @param [in] b_stride     Row stride of matrix B; may have any
                              value.
     @param [in] b_col_stride Column stride of matrix B; must be
                              positive, will normally be 1.
 */
template <typename Real>
Real cblasext_trace_mat_mat(
    const Real *a_data,
    KaldiBlasInt a_num_rows, KaldiBlasInt a_num_cols,
    KaldiBlasInt a_stride, KaldiBlasInt a_col_stride,
    const Real *b_data, CBLAS_TRANSPOSE b_trans,
    KaldiBlasInt b_stride, KaldiBlasInt b_col_stride);





}
// namespace kaldi

#endif

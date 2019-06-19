// cblasext/cblas-extensions.cc

// Copyright 2019       Johns Hopkins University (author: Daniel Povey)

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

#include "cblasext/cblas-wrappers.h"
#include "cblasext/cblas-extensions.h"

namespace kaldi {

template<typename Real>
void cblasext_Xgemv_sparsevec(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                     KaldiBlasInt num_cols, Real alpha, const Real *Mdata,
                     KaldiBlasInt stride, const Real *xdata,
                     KaldiBlasInt incX, Real beta, Real *ydata,
                     KaldiBlasInt incY) {
  if (trans == CblasNoTrans) {
    if (beta != 1.0) cblas_Xscal(num_rows, beta, ydata, incY);
    for (KaldiBlasInt i = 0; i < num_cols; i++) {
      Real x_i = xdata[i * incX];
      if (x_i == 0.0) continue;
      // Add to ydata, the i'th column of M, times alpha * x_i
      cblas_Xaxpy(num_rows, x_i * alpha, Mdata + i, stride, ydata, incY);
    }
  } else {
    if (beta != 1.0) cblas_Xscal(num_cols, beta, ydata, incY);
    for (KaldiBlasInt i = 0; i < num_rows; i++) {
      Real x_i = xdata[i * incX];
      if (x_i == 0.0) continue;
      // Add to ydata, the i'th row of M, times alpha * x_i
      cblas_Xaxpy(num_cols, x_i * alpha,
                  Mdata + (i * stride), 1, ydata, incY);
    }
  }
}


template
void cblasext_Xgemv_sparsevec(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                              KaldiBlasInt num_cols, float alpha, const float *Mdata,
                              KaldiBlasInt stride, const float *xdata,
                              KaldiBlasInt incX, float beta, float *ydata,
                              KaldiBlasInt incY);
template
void cblasext_Xgemv_sparsevec(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                              KaldiBlasInt num_cols, double alpha, const double *Mdata,
                              KaldiBlasInt stride, const double *xdata,
                              KaldiBlasInt incX, double beta, double *ydata,
                              KaldiBlasInt incY);


template <typename Real>
void cblasext_mul_elements_vec(
    const KaldiBlasInt dim,
    const Real *a,
    Real *b) { // does b *= a, elementwise.
  Real c1, c2, c3, c4;
  KaldiBlasInt i;
  for (i = 0; i + 4 <= dim; i += 4) {
    c1 = a[i] * b[i];
    c2 = a[i+1] * b[i+1];
    c3 = a[i+2] * b[i+2];
    c4 = a[i+3] * b[i+3];
    b[i] = c1;
    b[i+1] = c2;
    b[i+2] = c3;
    b[i+3] = c4;
  }
  for (; i < dim; i++)
    b[i] *= a[i];
}

template void cblasext_mul_elements_vec(const KaldiBlasInt dim,
                                        const float *a, float *b);
template void cblasext_mul_elements_vec(const KaldiBlasInt dim,
                                        const double *a, double *b);


template <typename Real>
void cblasext_mul_elements_mat(
    const Real *Adata,
    KaldiBlasInt a_num_rows,
    KaldiBlasInt a_num_cols,
    KaldiBlasInt a_stride,
    Real *Bdata,
    KaldiBlasInt b_stride) {
  if (a_num_cols == a_stride && a_num_cols == b_stride) {
    cblasext_mul_elements_vec(a_num_rows * a_num_cols, Adata, Bdata);
  } else {
    for (KaldiBlasInt i = 0; i < a_num_rows; i++) {
      cblasext_mul_elements_vec(a_num_cols, Adata, Bdata);
      Adata += a_stride;
      Bdata += b_stride;
    }
  }
}


template void cblasext_mul_elements_mat(
    const float *Adata, KaldiBlasInt a_num_rows,
    KaldiBlasInt a_num_cols, KaldiBlasInt a_stride,
    float *Bdata, KaldiBlasInt b_stride);
template void cblasext_mul_elements_mat(
    const double *Adata, KaldiBlasInt a_num_rows,
    KaldiBlasInt a_num_cols, KaldiBlasInt a_stride,
    double *Bdata, KaldiBlasInt b_stride);


template <typename Real>
Real cblasext_trace_mat_mat(
    const Real *a_data,
    KaldiBlasInt a_num_rows, KaldiBlasInt a_num_cols,
    KaldiBlasInt a_stride, KaldiBlasInt a_col_stride,
    const Real *b_data, CBLAS_TRANSPOSE b_trans,
    KaldiBlasInt b_stride, KaldiBlasInt b_col_stride) {
  Real ans = 0.0;
  if (b_trans == CblasNoTrans) {
    for (KaldiBlasInt i = 0; i < a_num_rows;
         i++, a_data += a_stride, b_data += b_col_stride) {
      ans += cblas_Xdot(a_num_cols, a_data, a_col_stride, b_data, b_stride);
    }
    return ans;
  } else {
    for (KaldiBlasInt i = 0; i < a_num_rows;
         i++, a_data += a_stride, b_data += b_stride) {
      ans += cblas_Xdot(a_num_cols, a_data, a_col_stride,
                        b_data, b_col_stride);
    }
    return ans;
  }
}

template float cblasext_trace_mat_mat(
    const float *a_data,
    KaldiBlasInt a_num_rows, KaldiBlasInt a_num_cols,
    KaldiBlasInt a_stride, KaldiBlasInt a_col_stride,
    const float *b_data, CBLAS_TRANSPOSE b_trans,
    KaldiBlasInt b_stride, KaldiBlasInt b_col_stride);
template double cblasext_trace_mat_mat(
    const double *a_data,
    KaldiBlasInt a_num_rows, KaldiBlasInt a_num_cols,
    KaldiBlasInt a_stride, KaldiBlasInt a_col_stride,
    const double *b_data, CBLAS_TRANSPOSE b_trans,
    KaldiBlasInt b_stride, KaldiBlasInt b_col_stride);



} // namespace kaldi

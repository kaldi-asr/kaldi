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


} // namespace kaldi

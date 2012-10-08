// matrix/cblas-wrappers.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#include <limits>

#include "matrix/sp-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-functions.h"

// Do not include this file directly.  It is to be included
// by .cc files in this directory.

namespace kaldi {

inline void cblas_Xscal(const int N, float *X, const int incX, float *Y,
                        const int incY, const float c, const float s) {
  cblas_srot(N, X, incX, Y, incY, c, s);
}

inline void cblas_Xrot(const int N, float *X, const int incX, float *Y,
                       const int incY, const float c, const float s) {
  cblas_srot(N, X, incX, Y, incY, c, s);
}
inline void cblas_Xrot(const int N, double *X, const int incX, double *Y,
                       const int incY, const double c, const double s) {
  cblas_drot(N, X, incX, Y, incY, c, s);
}
inline float cblas_Xdot(const int N, float *X, const int incX, float *Y,
                        const int incY) {
  return cblas_sdot(N, X, incX, Y, incY);
}
inline double cblas_Xdot(const int N, double *X, const int incX, double *Y,
                        const int incY) {
  return cblas_ddot(N, X, incX, Y, incY);
}

inline void cblas_Xaxpy(const int N, const float alpha, const float *X,
                        const int incX, float *Y, const int incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_Xaxpy(const int N, const double alpha, const double *X,
                        const int incX, double *Y, const int incY) {
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

// x = alpha * M * y + beta * x
inline void cblas_Xspmv(MatrixIndexT dim, float alpha, const float *Mdata,
                        const float *ydata, MatrixIndexT ystride,
                        float beta, float *xdata, MatrixIndexT xstride) {
  cblas_sspmv(CblasRowMajor, CblasLower, dim, alpha, Mdata,
              ydata, ystride, beta, xdata, xstride);
}
inline void cblas_Xspmv(MatrixIndexT dim, double alpha, const double *Mdata,
                        const double *ydata, MatrixIndexT ystride,
                        double beta, double *xdata, MatrixIndexT xstride) {
  cblas_dspmv(CblasRowMajor, CblasLower, dim, alpha, Mdata,
              ydata, ystride, beta, xdata, xstride);
}

// Implements  A += alpha * (x y'  + y x'); A is symmetric matrix.
inline void cblas_Xspr2(MatrixIndexT dim, float alpha, const float *Xdata,
                        MatrixIndexT incX, const float *Ydata, MatrixIndexT incY,
                          float *Adata) {
  cblas_sspr2(CblasRowMajor, CblasLower, dim, alpha, Xdata,
              incX, Ydata, incY, Adata);
}
inline void cblas_Xspr2(MatrixIndexT dim, double alpha, const double *Xdata,
                        MatrixIndexT incX, const double *Ydata, MatrixIndexT incY,
                        double *Adata) {
  cblas_dspr2(CblasRowMajor, CblasLower, dim, alpha, Xdata,
              incX, Ydata, incY, Adata);
}

// Implements  A += alpha * (x x'); A is symmetric matrix.
inline void cblas_Xspr(MatrixIndexT dim, float alpha, const float *Xdata,
                       MatrixIndexT incX, float *Adata) {
  cblas_sspr(CblasRowMajor, CblasLower, dim, alpha, Xdata, incX, Adata);
}
inline void cblas_Xspr(MatrixIndexT dim, double alpha, const double *Xdata,
                       MatrixIndexT incX, double *Adata) {
  cblas_dspr(CblasRowMajor, CblasLower, dim, alpha, Xdata, incX, Adata);
}

// sgemv,dgemv: y = alpha M x + beta y.
inline void cblas_Xgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, float alpha, const float *Mdata,
                        MatrixIndexT stride, const float *xdata,
                        MatrixIndexT incX, float beta, float *ydata, MatrixIndexT incY) {
  cblas_sgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride, xdata, incX, beta, ydata, incY);
}
inline void cblas_Xgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, double alpha, const double *Mdata,
                        MatrixIndexT stride, const double *xdata,
                        MatrixIndexT incX, float beta, double *ydata, MatrixIndexT incY) {
  cblas_dgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride, xdata, incX, beta, ydata, incY);
}

// ger: M += alpha x y^T.
inline void cblas_Xger(MatrixIndexT num_rows, MatrixIndexT num_cols, float alpha,
                       const float *xdata, MatrixIndexT incX, const float *ydata,
                       MatrixIndexT incY, float *Mdata, MatrixIndexT stride) {
  cblas_sger(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
             Mdata, stride);
}
inline void cblas_Xger(MatrixIndexT num_rows, MatrixIndexT num_cols, double alpha,
                       const double *xdata, MatrixIndexT incX, const double *ydata,
                       MatrixIndexT incY, double *Mdata, MatrixIndexT stride) {
  cblas_dger(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
             Mdata, stride);
}



}
// namespace kaldi

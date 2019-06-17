// matrix/cblas-wrappers.h

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
#ifndef KALDI_MATRIX_CBLAS_WRAPPERS_H_
#define KALDI_MATRIX_CBLAS_WRAPPERS_H_ 1


#include "cblasext/kaldi-blas.h"

// In directories other than this directory, this file is intended to mostly be
// included from .cc files, not from headers, since it includes cblas headers
// (via kaldi-blas.h) and those can be quite polluting.

// This file contains templated wrappers for CBLAS functions, which enable C++
// code calling these functions to be templated.
namespace kaldi {


inline void cblas_Xcopy(const KaldiBlasInt N, const float *X, const KaldiBlasInt incX, float *Y,
                        const KaldiBlasInt incY) {
  cblas_scopy(N, X, incX, Y, incY);
}

inline void cblas_Xcopy(const KaldiBlasInt N, const double *X, const KaldiBlasInt incX, double *Y,
                        const KaldiBlasInt incY) {
  cblas_dcopy(N, X, incX, Y, incY);
}

inline float cblas_Xasum(const KaldiBlasInt N, const float *X, const KaldiBlasInt incX) {
  return cblas_sasum(N, X, incX);
}

inline double cblas_Xasum(const KaldiBlasInt N, const double *X, const KaldiBlasInt incX) {
  return cblas_dasum(N, X, incX);
}

inline void cblas_Xrot(const KaldiBlasInt N, float *X, const KaldiBlasInt incX, float *Y,
                       const KaldiBlasInt incY, const float c, const float s) {
  cblas_srot(N, X, incX, Y, incY, c, s);
}
inline void cblas_Xrot(const KaldiBlasInt N, double *X, const KaldiBlasInt incX, double *Y,
                       const KaldiBlasInt incY, const double c, const double s) {
  cblas_drot(N, X, incX, Y, incY, c, s);
}
inline float cblas_Xdot(const KaldiBlasInt N, const float *const X,
                        const KaldiBlasInt incX, const float *const Y,
                        const KaldiBlasInt incY) {
  return cblas_sdot(N, X, incX, Y, incY);
}
inline double cblas_Xdot(const KaldiBlasInt N, const double *const X,
                        const KaldiBlasInt incX, const double *const Y,
                        const KaldiBlasInt incY) {
  return cblas_ddot(N, X, incX, Y, incY);
}
inline void cblas_Xaxpy(const KaldiBlasInt N, const float alpha, const float *X,
                        const KaldiBlasInt incX, float *Y, const KaldiBlasInt incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_Xaxpy(const KaldiBlasInt N, const double alpha, const double *X,
                        const KaldiBlasInt incX, double *Y, const KaldiBlasInt incY) {
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_Xscal(const KaldiBlasInt N, const float alpha, float *data,
                        const KaldiBlasInt inc) {
  cblas_sscal(N, alpha, data, inc);
}
inline void cblas_Xscal(const KaldiBlasInt N, const double alpha, double *data,
                        const KaldiBlasInt inc) {
  cblas_dscal(N, alpha, data, inc);
}
inline void cblas_Xtpmv(CBLAS_TRANSPOSE trans, const float *Mdata,
                        const KaldiBlasInt num_rows, float *y, const KaldiBlasInt y_inc) {
  cblas_stpmv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}
inline void cblas_Xtpmv(CBLAS_TRANSPOSE trans, const double *Mdata,
                        const KaldiBlasInt num_rows, double *y, const KaldiBlasInt y_inc) {
  cblas_dtpmv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}


inline void cblas_Xtpsv(CBLAS_TRANSPOSE trans, const float *Mdata,
                        const KaldiBlasInt num_rows, float *y, const KaldiBlasInt y_inc) {
  cblas_stpsv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}
inline void cblas_Xtpsv(CBLAS_TRANSPOSE trans, const double *Mdata,
                        const KaldiBlasInt num_rows, double *y, const KaldiBlasInt y_inc) {
  cblas_dtpsv(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              CblasNonUnit, num_rows, Mdata, y, y_inc);
}

// x = alpha * M * y + beta * x
inline void cblas_Xspmv(KaldiBlasInt dim, float alpha, const float *Mdata,
                        const float *ydata, KaldiBlasInt ystride,
                        float beta, float *xdata, KaldiBlasInt xstride) {
  cblas_sspmv(CblasRowMajor, CblasLower, dim, alpha, Mdata,
              ydata, ystride, beta, xdata, xstride);
}
inline void cblas_Xspmv(KaldiBlasInt dim, double alpha, const double *Mdata,
                        const double *ydata, KaldiBlasInt ystride,
                        double beta, double *xdata, KaldiBlasInt xstride) {
  cblas_dspmv(CblasRowMajor, CblasLower, dim, alpha, Mdata,
              ydata, ystride, beta, xdata, xstride);
}

// Implements  A += alpha * (x y'  + y x'); A is symmetric matrix.
inline void cblas_Xspr2(KaldiBlasInt dim, float alpha, const float *Xdata,
                        KaldiBlasInt incX, const float *Ydata, KaldiBlasInt incY,
                          float *Adata) {
  cblas_sspr2(CblasRowMajor, CblasLower, dim, alpha, Xdata,
              incX, Ydata, incY, Adata);
}
inline void cblas_Xspr2(KaldiBlasInt dim, double alpha, const double *Xdata,
                        KaldiBlasInt incX, const double *Ydata, KaldiBlasInt incY,
                        double *Adata) {
  cblas_dspr2(CblasRowMajor, CblasLower, dim, alpha, Xdata,
              incX, Ydata, incY, Adata);
}

// Implements  A += alpha * (x x'); A is symmetric matrix.
inline void cblas_Xspr(KaldiBlasInt dim, float alpha, const float *Xdata,
                       KaldiBlasInt incX, float *Adata) {
  cblas_sspr(CblasRowMajor, CblasLower, dim, alpha, Xdata, incX, Adata);
}
inline void cblas_Xspr(KaldiBlasInt dim, double alpha, const double *Xdata,
                       KaldiBlasInt incX, double *Adata) {
  cblas_dspr(CblasRowMajor, CblasLower, dim, alpha, Xdata, incX, Adata);
}

// sgemv,dgemv: y = alpha M x + beta y.
inline void cblas_Xgemv(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                        KaldiBlasInt num_cols, float alpha, const float *Mdata,
                        KaldiBlasInt stride, const float *xdata,
                        KaldiBlasInt incX, float beta, float *ydata, KaldiBlasInt incY) {
  cblas_sgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride, xdata, incX, beta, ydata, incY);
}
inline void cblas_Xgemv(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                        KaldiBlasInt num_cols, double alpha, const double *Mdata,
                        KaldiBlasInt stride, const double *xdata,
                        KaldiBlasInt incX, double beta, double *ydata, KaldiBlasInt incY) {
  cblas_dgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride, xdata, incX, beta, ydata, incY);
}

// sgbmv, dgmmv: y = alpha M x +  + beta * y.
inline void cblas_Xgbmv(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                        KaldiBlasInt num_cols, KaldiBlasInt num_below,
                        KaldiBlasInt num_above, float alpha, const float *Mdata,
                        KaldiBlasInt stride, const float *xdata,
                        KaldiBlasInt incX, float beta, float *ydata, KaldiBlasInt incY) {
  cblas_sgbmv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, num_below, num_above, alpha, Mdata, stride, xdata,
              incX, beta, ydata, incY);
}
inline void cblas_Xgbmv(CBLAS_TRANSPOSE trans, KaldiBlasInt num_rows,
                        KaldiBlasInt num_cols, KaldiBlasInt num_below,
                        KaldiBlasInt num_above, double alpha, const double *Mdata,
                        KaldiBlasInt stride, const double *xdata,
                        KaldiBlasInt incX, double beta, double *ydata, KaldiBlasInt incY) {
  cblas_dgbmv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, num_below, num_above, alpha, Mdata, stride, xdata,
              incX, beta, ydata, incY);
}

inline void cblas_Xgemm(const float alpha,
                        CBLAS_TRANSPOSE transA,
                        const float *Adata,
                        KaldiBlasInt a_num_rows, KaldiBlasInt a_num_cols, KaldiBlasInt a_stride,
                        CBLAS_TRANSPOSE transB,
                        const float *Bdata, KaldiBlasInt b_stride,
                        const float beta,
                        float *Mdata,
                        KaldiBlasInt num_rows, KaldiBlasInt num_cols,KaldiBlasInt stride) {
  cblas_sgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA),
              static_cast<CBLAS_TRANSPOSE>(transB),
              num_rows, num_cols, transA == CblasNoTrans ? a_num_cols : a_num_rows,
              alpha, Adata, a_stride, Bdata, b_stride,
              beta, Mdata, stride);
}
inline void cblas_Xgemm(const double alpha,
                        CBLAS_TRANSPOSE transA,
                        const double *Adata,
                        KaldiBlasInt a_num_rows, KaldiBlasInt a_num_cols, KaldiBlasInt a_stride,
                        CBLAS_TRANSPOSE transB,
                        const double *Bdata, KaldiBlasInt b_stride,
                        const double beta,
                        double *Mdata,
                        KaldiBlasInt num_rows, KaldiBlasInt num_cols,KaldiBlasInt stride) {
  cblas_dgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA),
              static_cast<CBLAS_TRANSPOSE>(transB),
              num_rows, num_cols, transA == CblasNoTrans ? a_num_cols : a_num_rows,
              alpha, Adata, a_stride, Bdata, b_stride,
              beta, Mdata, stride);
}


inline void cblas_Xsymm(const float alpha,
                        KaldiBlasInt sz,
                        const float *Adata,KaldiBlasInt a_stride,
                        const float *Bdata,KaldiBlasInt b_stride,
                        const float beta,
                        float *Mdata, KaldiBlasInt stride) {
  cblas_ssymm(CblasRowMajor, CblasLeft, CblasLower, sz, sz, alpha, Adata,
              a_stride, Bdata, b_stride, beta, Mdata, stride);
}
inline void cblas_Xsymm(const double alpha,
                        KaldiBlasInt sz,
                        const double *Adata,KaldiBlasInt a_stride,
                        const double *Bdata,KaldiBlasInt b_stride,
                        const double beta,
                        double *Mdata, KaldiBlasInt stride) {
  cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, sz, sz, alpha, Adata,
              a_stride, Bdata, b_stride, beta, Mdata, stride);
}
// ger: M += alpha x y^T.
inline void cblas_Xger(KaldiBlasInt num_rows, KaldiBlasInt num_cols, float alpha,
                       const float *xdata, KaldiBlasInt incX, const float *ydata,
                       KaldiBlasInt incY, float *Mdata, KaldiBlasInt stride) {
  cblas_sger(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
             Mdata, stride);
}
inline void cblas_Xger(KaldiBlasInt num_rows, KaldiBlasInt num_cols, double alpha,
                       const double *xdata, KaldiBlasInt incX, const double *ydata,
                       KaldiBlasInt incY, double *Mdata, KaldiBlasInt stride) {
  cblas_dger(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
             Mdata, stride);
}

// syrk: symmetric rank-k update.
// if trans==CblasNoTrans, then C = alpha A A^T + beta C
// else C = alpha A^T A + beta C.
// note: dim_c is dim(C), other_dim_a is the "other" dimension of A, i.e.
// num-cols(A) if CblasNoTrans, or num-rows(A) if CblasTrans.
// We only need the row-major and lower-triangular option of this, and this
// is hard-coded.
inline void cblas_Xsyrk (
    const CBLAS_TRANSPOSE trans, const KaldiBlasInt dim_c,
    const KaldiBlasInt other_dim_a, const float alpha, const float *A,
    const KaldiBlasInt a_stride, const float beta, float *C,
    const KaldiBlasInt c_stride) {
  cblas_ssyrk(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              dim_c, other_dim_a, alpha, A, a_stride, beta, C, c_stride);
}

inline void cblas_Xsyrk(
    const CBLAS_TRANSPOSE trans, const KaldiBlasInt dim_c,
    const KaldiBlasInt other_dim_a, const double alpha, const double *A,
    const KaldiBlasInt a_stride, const double beta, double *C,
    const KaldiBlasInt c_stride) {
  cblas_dsyrk(CblasRowMajor, CblasLower, static_cast<CBLAS_TRANSPOSE>(trans),
              dim_c, other_dim_a, alpha, A, a_stride, beta, C, c_stride);
}

/// matrix-vector multiply using a banded matrix; we always call this
/// with b = 1 meaning we're multiplying by a diagonal matrix.  This is used for
/// elementwise multiplication.  We miss some of the arguments out of this
/// wrapper.
inline void cblas_Xsbmv1(
    const KaldiBlasInt dim,
    const double *A,
    const double alpha,
    const double *x,
    const double beta,
    double *y) {
  cblas_dsbmv(CblasRowMajor, CblasLower, dim, 0, alpha, A,
              1, x, 1, beta, y, 1);
}

inline void cblas_Xsbmv1(
    const KaldiBlasInt dim,
    const float *A,
    const float alpha,
    const float *x,
    const float beta,
    float *y) {
  cblas_ssbmv(CblasRowMajor, CblasLower, dim, 0, alpha, A,
              1, x, 1, beta, y, 1);
}


// add clapack here
#if !defined(HAVE_ATLAS)
inline void clapack_Xtptri(KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *result) {
  stptri_(const_cast<char *>("U"), const_cast<char *>("N"), num_rows, Mdata, result);
}
inline void clapack_Xtptri(KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *result) {
  dtptri_(const_cast<char *>("U"), const_cast<char *>("N"), num_rows, Mdata, result);
}
//
inline void clapack_Xgetrf2(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols,
                            float *Mdata, KaldiBlasInt *stride, KaldiBlasInt *pivot,
                            KaldiBlasInt *result) {
  sgetrf_(num_rows, num_cols, Mdata, stride, pivot, result);
}
inline void clapack_Xgetrf2(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols,
                            double *Mdata, KaldiBlasInt *stride, KaldiBlasInt *pivot,
                            KaldiBlasInt *result) {
  dgetrf_(num_rows, num_cols, Mdata, stride, pivot, result);
}

//
inline void clapack_Xgetri2(KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *stride,
                           KaldiBlasInt *pivot, float *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  sgetri_(num_rows, Mdata, stride, pivot, p_work, l_work, result);
}
inline void clapack_Xgetri2(KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *stride,
                           KaldiBlasInt *pivot, double *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  dgetri_(num_rows, Mdata, stride, pivot, p_work, l_work, result);
}
//
inline void clapack_Xgesvd(char *v, char *u, KaldiBlasInt *num_cols,
                           KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *stride,
                           float *sv, float *Vdata, KaldiBlasInt *vstride,
                           float *Udata, KaldiBlasInt *ustride, float *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  sgesvd_(v, u,
          num_cols, num_rows, Mdata, stride,
          sv, Vdata, vstride, Udata, ustride,
          p_work, l_work, result);
}
inline void clapack_Xgesvd(char *v, char *u, KaldiBlasInt *num_cols,
                           KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *stride,
                           double *sv, double *Vdata, KaldiBlasInt *vstride,
                           double *Udata, KaldiBlasInt *ustride, double *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  dgesvd_(v, u,
          num_cols, num_rows, Mdata, stride,
          sv, Vdata, vstride, Udata, ustride,
          p_work, l_work, result);
}
//
void inline clapack_Xsptri(KaldiBlasInt *num_rows, float *Mdata,
                           KaldiBlasInt *ipiv, float *work, KaldiBlasInt *result) {
  ssptri_(const_cast<char *>("U"), num_rows, Mdata, ipiv, work, result);
}
void inline clapack_Xsptri(KaldiBlasInt *num_rows, double *Mdata,
                           KaldiBlasInt *ipiv, double *work, KaldiBlasInt *result) {
  dsptri_(const_cast<char *>("U"), num_rows, Mdata, ipiv, work, result);
}
//
void inline clapack_Xsptrf(KaldiBlasInt *num_rows, float *Mdata,
                           KaldiBlasInt *ipiv, KaldiBlasInt *result) {
  ssptrf_(const_cast<char *>("U"), num_rows, Mdata, ipiv, result);
}
void inline clapack_Xsptrf(KaldiBlasInt *num_rows, double *Mdata,
                           KaldiBlasInt *ipiv, KaldiBlasInt *result) {
  dsptrf_(const_cast<char *>("U"), num_rows, Mdata, ipiv, result);
}
#else
inline void clapack_Xgetrf(KaldiBlasInt num_rows, KaldiBlasInt num_cols,
                           float *Mdata, KaldiBlasInt stride,
                           KaldiBlasInt *pivot, KaldiBlasInt *result) {
  *result = clapack_sgetrf(CblasColMajor, num_rows, num_cols,
                              Mdata, stride, pivot);
}

inline void clapack_Xgetrf(KaldiBlasInt num_rows, KaldiBlasInt num_cols,
                           double *Mdata, KaldiBlasInt stride,
                           KaldiBlasInt *pivot, KaldiBlasInt *result) {
  *result = clapack_dgetrf(CblasColMajor, num_rows, num_cols,
                              Mdata, stride, pivot);
}
//
inline KaldiBlasInt clapack_Xtrtri(KaldiBlasInt num_rows, float *Mdata, KaldiBlasInt stride) {
  return  clapack_strtri(CblasColMajor, CblasUpper, CblasNonUnit, num_rows,
                              Mdata, stride);
}

inline KaldiBlasInt clapack_Xtrtri(KaldiBlasInt num_rows, double *Mdata, KaldiBlasInt stride) {
  return  clapack_dtrtri(CblasColMajor, CblasUpper, CblasNonUnit, num_rows,
                              Mdata, stride);
}
//
inline void clapack_Xgetri(KaldiBlasInt num_rows, float *Mdata, KaldiBlasInt stride,
                      KaldiBlasInt *pivot, KaldiBlasInt *result) {
  *result = clapack_sgetri(CblasColMajor, num_rows, Mdata, stride, pivot);
}
inline void clapack_Xgetri(KaldiBlasInt num_rows, double *Mdata, KaldiBlasInt stride,
                      KaldiBlasInt *pivot, KaldiBlasInt *result) {
  *result = clapack_dgetri(CblasColMajor, num_rows, Mdata, stride, pivot);
}
#endif

}
// namespace kaldi

#endif

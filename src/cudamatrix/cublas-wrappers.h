// cudamatrix/cublas-wrappers.h

// Copyright 2013  Johns Hopkins University (author: Daniel Povey);

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
#ifndef KALDI_MATRIX_CUBLAS_WRAPPERS_H_
#define KALDI_MATRIX_CUBLAS_WRAPPERS_H_ 1

// Do not include this file directly.  It is to be included
// by .cc files in this directory.

namespace kaldi {
#if HAVE_CUDA == 1

inline void cublas_gemm(char transa, char transb, int m, int n,int k, float alpha, const float *A, int lda,const float *B, int ldb, float beta, float *C, int ldc) {
  cublasSgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void cublas_gemm(char transa, char transb, int m, int n,int k, double alpha, const double *A, int lda,const double *B, int ldb, double beta, double *C, int ldc) {
  cublasDgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void cublas_trsm(int m, int n, float alpha, const float* A, int lda, float* B, int ldb) {
  cublasStrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
inline void cublas_trsm(int m, int n, double alpha, const double* A, int lda, double* B, int ldb) {
  cublasDtrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
inline void cublas_syrk(char uplo, char trans, int n, int k,
                        float alpha, const float *A, int lda,
                        float beta, float *C, int ldc) {
  cublasSsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
inline void cublas_syrk(char uplo, char trans, int n, int k,
                        double alpha, const double *A, int lda,
                        double beta, double *C, int ldc) {
  cublasDsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
inline float cublas_dot(int n, const float *x, int incx, const float *y, int incy) {
  return cublasSdot(n, x, incx, y, incy);
}
inline double cublas_dot(int n, const double *x, int incx, const double *y, int incy) {
  return cublasDdot(n, x, incx, y, incy);
}
inline float cublas_asum(int n, const float* x, int incx) {
  return cublasSasum(n, x, incx);
}
inline double cublas_asum(int n, const double* x, int incx) {
  return cublasDasum(n, x, incx);
}
inline float cublas_nrm2(int n, const float* x, int incx) {
  return cublasSnrm2(n, x, incx);
}
inline double cublas_nrm2(int n, const double* x, int incx) {
  return cublasDnrm2(n, x, incx);
}
inline void cublas_copy(int n, const float* x, int incx,
                        float* y, int incy) {
  cublasScopy(n,x,incx,y,incy);
}
inline void cublas_copy(int n, const double* x, int incx,
                          double* y, int incy) {
  cublasDcopy(n,x,incx,y,incy);
}
inline void cublas_scal(int n, float alpha, float* mat, int incx) {
  cublasSscal(n, alpha, mat, incx);
}
inline void cublas_scal(int n, double alpha, double* mat, int incx) {
  cublasDscal(n, alpha, mat, incx);
}

inline void cublas_axpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
  cublasSaxpy(n, alpha, x, incx, y, incy);
}
inline void cublas_axpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
  cublasDaxpy(n, alpha, x, incx, y, incy);
}
inline void cublas_gemv(char trans, int m, int n, float alpha,
                        const float* A, int lda, const float* x,
                        int incx, float beta, float* y, int incy) {
  cublasSgemv(trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}
inline void cublas_gemv(char trans, int m, int n, double alpha,
                        const double* A, int lda, const double* x,
                        int incx, double beta, double* y, int incy) {
  cublasDgemv(trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
}

inline void cublas_spmv(char uplo, int n, float alpha, const float *AP, const float *x,
                        int incx, float beta, float *y, int incy) {
  cublasSspmv(uplo, n, alpha, AP, x, incx, beta, y, incy);
}
inline void cublas_spmv(char uplo, int n, double alpha, const double *AP, const double *x,
                        int incx, double beta, double *y, int incy) {
  cublasDspmv(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

// Use caution with these, the 'transpose' argument is the opposite of what it
// should really be, due to CUDA storing things in column major order.  We also
// had to switch 'l' to 'u'; we view our packed matrices as lower-triangular,
// row-by-row, but CUDA views the same layout as upper-triangular,
// column-by-column.
inline void cublas_tpmv(char trans, int n,
                        const float* Ap, float* x, int incx) {
  return cublasStpmv('u', trans, 'n', n, Ap, x, incx);
}
inline void cublas_tpmv(char trans, int n, const double* Ap,
                        double* x,int incx) {
  return cublasDtpmv('u', trans, 'n', n, Ap, x, incx);
}

inline void cublas_spr(char uplo, int n, float alpha, const float *x,
                      int incx, float *AP) {
  cublasSspr(uplo, n, alpha, x, incx, AP);
}
inline void cublas_spr(char uplo, int n, double alpha, const double *x,
                      int incx, double *AP) {
  cublasDspr(uplo, n, alpha, x, incx, AP);
}

#endif
}
// namespace kaldi

#endif

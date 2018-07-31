// cudamatrix/cublas-wrappers.h

// Copyright 2013  Johns Hopkins University (author: Daniel Povey);
//           2017  Shiyin Kang

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
#ifndef KALDI_CUDAMATRIX_CUBLAS_WRAPPERS_H_
#define KALDI_CUDAMATRIX_CUBLAS_WRAPPERS_H_ 1

// Do not include this file directly.  It is to be included
// by .cc files in this directory.

namespace kaldi {
#if HAVE_CUDA == 1

inline cublasStatus_t cublas_gemm(
    cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,int k, float alpha,
    const float *A, int lda, const float *B, int ldb, float beta,
    float *C, int ldc) {
  return cublasSgemm_v2(handle,transa,transb,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc);
}
inline cublasStatus_t cublas_gemm(
    cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,int k, double alpha,
    const double *A, int lda, const double *B, int ldb, double beta,
    double *C, int ldc) {
  return cublasDgemm_v2(handle,transa,transb,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc);
}
inline cublasStatus_t cublas_ger(
    cublasHandle_t handle, int m, int n, float alpha,
    const float *x, int incx, const float *y, int incy, float *A, int lda ) {
  return cublasSger_v2(handle,m,n,&alpha,x,incx,y,incy,A,lda);
}
inline cublasStatus_t cublas_ger(cublasHandle_t handle, int m, int n, double alpha,
        const double *x, int incx, const double *y, int incy, double *A, int lda ) {
  return cublasDger_v2(handle,m,n,&alpha,x,incx,y,incy,A,lda);
}
inline cublasStatus_t cublas_gemmBatched(
    cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, float alpha,
    const float *A[], int lda, const float *B[], int ldb, float beta,
    float *C[], int ldc, int batchCount) {
  return cublasSgemmBatched(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, batchCount);
}
inline cublasStatus_t cublas_gemmBatched(
    cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k, double alpha,
    const double *A[], int lda, const double *B[], int ldb, double beta,
    double *C[], int ldc, int batchCount) {
  return cublasDgemmBatched(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, batchCount);
}
inline cublasStatus_t cublas_trsm(cublasHandle_t handle, int m, int n,
                                  float alpha, const float* A, int lda,
                                  float* B, int ldb) {
  return cublasStrsm_v2(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,m,n,&alpha,A,lda,B,ldb);
}
inline cublasStatus_t cublas_trsm(cublasHandle_t handle, int m, int n,
                                  double alpha, const double* A, int lda,
                                  double* B, int ldb) {
  return cublasDtrsm_v2(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,m,n,&alpha,A,lda,B,ldb);
}
inline cublasStatus_t cublas_syrk(
    cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k, float alpha,
    const float *A, int lda, float beta, float *C, int ldc) {
  return cublasSsyrk_v2(handle,uplo,trans,n,k,&alpha,A,lda,&beta,C,ldc);
}
inline cublasStatus_t cublas_syrk(
    cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k, double alpha,
    const double *A, int lda, double beta, double *C, int ldc) {
  return cublasDsyrk_v2(handle,uplo,trans,n,k,&alpha,A,lda,&beta,C,ldc);
}
inline cublasStatus_t cublas_dot(cublasHandle_t handle, int n, const float *x,
                                 int incx, const float *y, int incy,
                                 float *result) {
  return cublasSdot_v2(handle, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublas_dot(cublasHandle_t handle, int n, const double *x,
                                 int incx, const double *y, int incy,
                                 double *result) {
  return cublasDdot_v2(handle, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublas_asum(cublasHandle_t handle, int n, const float* x,
                                  int incx, float *result) {
  return cublasSasum_v2(handle, n, x, incx, result);
}
inline cublasStatus_t cublas_asum(cublasHandle_t handle, int n, const double* x,
                                  int incx, double *result) {
  return cublasDasum_v2(handle, n, x, incx, result);
}
inline cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n, const float* x,
                                  int incx, float *result) {
  return cublasSnrm2_v2(handle, n, x, incx, result);
}
inline cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n, const double* x,
                                  int incx, double *result) {
  return cublasDnrm2_v2(handle, n, x, incx, result);
}
inline cudaError_t cublas_copy(cublasHandle_t handle, int n, const float* x,
    int incx, double* y, int incy) {
  int dimBlock(CU1DBLOCK);
  int dimGrid(n_blocks(n, CU1DBLOCK));
  cublas_copy_kaldi_fd(dimGrid, dimBlock, n, x, incx, y, incy);
  return cudaGetLastError();
}
inline cudaError_t cublas_copy(cublasHandle_t handle, int n, const double* x,
    int incx, float* y, int incy) {
  int dimBlock(CU1DBLOCK);
  int dimGrid(n_blocks(n, CU1DBLOCK));
  cublas_copy_kaldi_df(dimGrid, dimBlock, n, x, incx, y, incy);
  return cudaGetLastError();
}
inline cublasStatus_t cublas_copy(cublasHandle_t handle, int n, const float* x,
                                  int incx, float* y, int incy) {
  return cublasScopy_v2(handle,n,x,incx,y,incy);
}
inline cublasStatus_t cublas_copy(cublasHandle_t handle, int n, const double* x,
                                  int incx, double* y, int incy) {
  return cublasDcopy_v2(handle,n,x,incx,y,incy);
}
inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n, float alpha,
                                  float* mat, int incx) {
  return cublasSscal_v2(handle, n, &alpha, mat, incx);
}
inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n, double alpha,
                                  double* mat, int incx) {
  return cublasDscal_v2(handle, n, &alpha, mat, incx);
}

inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, float alpha,
                                  const float* x, int incx, float* y, int incy) {
  return cublasSaxpy_v2(handle, n, &alpha, x, incx, y, incy);
}
inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n, double alpha,
                                  const double* x, int incx, double* y, int incy) {
  return cublasDaxpy_v2(handle, n, &alpha, x, incx, y, incy);
}
inline cublasStatus_t cublas_gemv(
    cublasHandle_t handle, cublasOperation_t trans,
    int m, int n, float alpha, const float* A, int lda, const float* x,
    int incx, float beta, float* y, int incy) {
  return cublasSgemv_v2(handle,trans,m,n,&alpha,A,lda,x,incx,&beta,y,incy);
}
inline cublasStatus_t cublas_gemv(
    cublasHandle_t handle, cublasOperation_t trans,
    int m, int n, double alpha, const double* A, int lda, const double* x,
    int incx, double beta, double* y, int incy) {
  return cublasDgemv_v2(handle,trans,m,n,&alpha,A,lda,x,incx,&beta,y,incy);
}

inline cublasStatus_t cublas_spmv(
    cublasHandle_t handle, cublasFillMode_t uplo,
    int n, float alpha, const float *AP, const float *x, int incx,
    float beta, float *y, int incy) {
  return cublasSspmv_v2(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
}
inline cublasStatus_t cublas_spmv(
    cublasHandle_t handle, cublasFillMode_t uplo,
    int n, double alpha, const double *AP, const double *x, int incx,
    double beta, double *y, int incy) {
  return cublasDspmv_v2(handle, uplo, n, &alpha, AP, x, incx, &beta, y, incy);
}

// Use caution with these, the 'transpose' argument is the opposite of what it
// should really be, due to CUDA storing things in column major order.  We also
// had to switch 'l' to 'u'; we view our packed matrices as lower-triangular,
// row-by-row, but CUDA views the same layout as upper-triangular,
// column-by-column.
inline cublasStatus_t cublas_tpmv(cublasHandle_t handle, cublasOperation_t trans,
                                  int n, const float* Ap, float* x, int incx) {
  return cublasStpmv_v2(handle, CUBLAS_FILL_MODE_UPPER, trans, CUBLAS_DIAG_NON_UNIT, n, Ap, x, incx);
}
inline cublasStatus_t cublas_tpmv(cublasHandle_t handle, cublasOperation_t trans,
                                  int n, const double* Ap, double* x,int incx) {
  return cublasDtpmv_v2(handle, CUBLAS_FILL_MODE_UPPER, trans, CUBLAS_DIAG_NON_UNIT, n, Ap, x, incx);
}

inline cublasStatus_t cublas_spr(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int n, float alpha, const float *x, int incx,
                                 float *AP) {
  return cublasSspr_v2(handle, uplo, n, &alpha, x, incx, AP);
}
inline cublasStatus_t cublas_spr(cublasHandle_t handle, cublasFillMode_t uplo,
                                 int n, double alpha, const double *x, int incx,
                                 double *AP) {
  return cublasDspr_v2(handle, uplo, n, &alpha, x, incx, AP);
}

//
// cuSPARSE wrappers
//

inline cusparseStatus_t cusparse_csr2csc(cusparseHandle_t handle, int m, int n,
                                         int nnz, const float *csrVal,
                                         const int *csrRowPtr,
                                         const int *csrColInd, float *cscVal,
                                         int *cscRowInd, int *cscColPtr,
                                         cusparseAction_t copyValues,
                                         cusparseIndexBase_t idxBase) {
  return cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
}
inline cusparseStatus_t cusparse_csr2csc(cusparseHandle_t handle, int m, int n,
                                         int nnz, const double *csrVal,
                                         const int *csrRowPtr,
                                         const int *csrColInd, double *cscVal,
                                         int *cscRowInd, int *cscColPtr,
                                         cusparseAction_t copyValues,
                                         cusparseIndexBase_t idxBase) {
  return cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
}

inline cusparseStatus_t cusparse_csrmm(cusparseHandle_t handle,
                                       cusparseOperation_t transA, int m, int n,
                                       int k, int nnz, const float *alpha,
                                       const cusparseMatDescr_t descrA,
                                       const float *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA, const float *B,
                                       int ldb, const float *beta, float *C,
                                       int ldc) {
  return cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA,
                        csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}
inline cusparseStatus_t cusparse_csrmm(cusparseHandle_t handle,
                                       cusparseOperation_t transA, int m, int n,
                                       int k, int nnz, const double *alpha,
                                       const cusparseMatDescr_t descrA,
                                       const double *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA, const double *B,
                                       int ldb, const double *beta, double *C,
                                       int ldc) {
  return cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA,
                        csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

inline cusparseStatus_t cusparse_csrmm2(cusparseHandle_t handle,
                                        cusparseOperation_t transA,
                                        cusparseOperation_t transB, int m,
                                        int n, int k, int nnz,
                                        const float *alpha,
                                        const cusparseMatDescr_t descrA,
                                        const float *csrValA,
                                        const int *csrRowPtrA,
                                        const int *csrColIndA, const float *B,
                                        int ldb, const float *beta, float *C,
                                        int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}
inline cusparseStatus_t cusparse_csrmm2(cusparseHandle_t handle,
                                        cusparseOperation_t transA,
                                        cusparseOperation_t transB, int m,
                                        int n, int k, int nnz,
                                        const double *alpha,
                                        const cusparseMatDescr_t descrA,
                                        const double *csrValA,
                                        const int *csrRowPtrA,
                                        const int *csrColIndA, const double *B,
                                        int ldb, const double *beta, double *C,
                                        int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}


#endif
}
// namespace kaldi

#endif

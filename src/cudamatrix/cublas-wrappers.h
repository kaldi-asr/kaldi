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

#include "cudamatrix/cu-device.h"

namespace kaldi {
#if HAVE_CUDA == 1

inline cublasStatus_t cublas_gemm(
    cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,int k, float alpha,
    const float *A, int lda, const float *B, int ldb, float beta,
    float *C, int ldc) {
#if CUDA_VERSION >= 11000
  return cublasGemmEx(handle,transa,transb,m,n,k,&alpha,A,CUDA_R_32F,lda,B,CUDA_R_32F,ldb,&beta,
                      C,CUDA_R_32F,ldc,CuDevice::Instantiate().GetCublasComputeType(),
                      CuDevice::Instantiate().GetCublasGemmAlgo());
#else
  return cublasSgemm_v2(handle,transa,transb,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc);
#endif
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
#if CUDA_VERSION >= 11000
  return cublasGemmBatchedEx(handle, transa, transb, m, n, k, &alpha, (const void**)A, CUDA_R_32F,  lda,
                             (const void**)B, CUDA_R_32F, ldb, &beta, (void**)C, CUDA_R_32F, ldc, batchCount,
                             CuDevice::Instantiate().GetCublasComputeType(), CuDevice::Instantiate().GetCublasGemmAlgo());
#else
  return cublasSgemmBatched(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, batchCount);
#endif
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
#if CUDA_VERSION >= 10020
inline cusparseStatus_t cusparse_csr2csc(cusparseHandle_t handle, int m, int n,
                                         int nnz, const void *csrVal,
                                         const int *csrRowPtr,
                                         const int *csrColInd, void *cscVal,
                                         int *cscRowInd, int *cscColPtr,
					 cudaDataType valType,
                                         cusparseAction_t copyValues,
                                         cusparseIndexBase_t idxBase) {
  cusparseStatus_t status;
  size_t buffer_size;
  status = cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, 
			  CUSPARSE_CSR2CSC_ALG1, &buffer_size);
  if(status != CUSPARSE_STATUS_SUCCESS) return status;

  void *buffer = (buffer_size > 0) ? CuDevice::Instantiate().Malloc(buffer_size) : NULL; 
  status = cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, 
			  CUSPARSE_CSR2CSC_ALG1, buffer);
  if(buffer)
 	 CuDevice::Instantiate().Free(buffer); // allocator will take care of syncing if necessary 

  return status;
}

inline cusparseStatus_t cusparse_csrmm2(cusparseHandle_t handle,
                                       cusparseOperation_t transA, 
				       cusparseOperation_t transB, int m, int n,
                                       int k, int nnz, const void *alpha,
                                       const cusparseMatDescr_t descrA,
                                       const void *csrValA,
                                       const int *csrRowPtrA,
                                       const int *csrColIndA, const void *B,
                                       int ldb, const void *beta, void *C,
                                       int ldc, cudaDataType valType) {
  cusparseStatus_t status;
  cusparseSpMatDescr_t matA;
  cusparseIndexBase_t idxBase = cusparseGetMatIndexBase(descrA);
  KALDI_ASSERT(transA == CUSPARSE_OPERATION_NON_TRANSPOSE);
  // Casting away the const-ness. We won't write to those pointers, but that's
  // needed to create the matrix descriptor
  status =
      cusparseCreateCsr(&matA, m, k, nnz, const_cast<int *>(csrRowPtrA),
                        const_cast<int *>(csrColIndA),
                        const_cast<void *>(csrValA), CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I, idxBase, valType);
  if (status != CUSPARSE_STATUS_SUCCESS) return status;
  cusparseDnMatDescr_t matB;
  int nrowsB=k, ncolsB=n;
  if(transB == CUSPARSE_OPERATION_TRANSPOSE) std::swap(nrowsB, ncolsB);
  status = cusparseCreateDnMat(&matB, nrowsB, ncolsB, ldb, const_cast<void *>(B), valType,
                               CUSPARSE_ORDER_COL);
  if (status != CUSPARSE_STATUS_SUCCESS) return status;
  cusparseDnMatDescr_t matC;
  status =
      cusparseCreateDnMat(&matC, m, n, ldc, C, valType, CUSPARSE_ORDER_COL);
  if (status != CUSPARSE_STATUS_SUCCESS) return status;

  size_t buffer_size;
#if CUDA_VERSION >= 11000
  status = cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA, matB,
                                   beta, matC, valType, CUSPARSE_SPMM_CSR_ALG2,
                                   &buffer_size);
#else
  status = cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA, matB,
                                   beta, matC, valType, CUSPARSE_MM_ALG_DEFAULT,
                                   &buffer_size);
#endif
  if (status != CUSPARSE_STATUS_SUCCESS) return status;

  void *buffer = (buffer_size > 0) ? CuDevice::Instantiate().Malloc(buffer_size) : NULL;
#if CUDA_VERSION >= 11000
  status = cusparseSpMM(handle, transA, transB, alpha, matA, matB, beta, matC,
                        valType, CUSPARSE_SPMM_CSR_ALG2, buffer);
#else
  status = cusparseSpMM(handle, transA, transB, alpha, matA, matB, beta, matC,
                        valType, CUSPARSE_MM_ALG_DEFAULT, buffer);
#endif

  if (status != CUSPARSE_STATUS_SUCCESS) return status;
  if(buffer)
  	CuDevice::Instantiate().Free(buffer); 

  status = cusparseDestroySpMat(matA);
  if (status != CUSPARSE_STATUS_SUCCESS) return status;
  status = cusparseDestroyDnMat(matB);
  if (status != CUSPARSE_STATUS_SUCCESS) return status;
  status = cusparseDestroyDnMat(matC);

  return status;
}
#endif

inline cusparseStatus_t cusparse_csr2csc(cusparseHandle_t handle, int m, int n,
                                         int nnz, const float *csrVal,
                                         const int *csrRowPtr,
                                         const int *csrColInd, float *cscVal,
                                         int *cscRowInd, int *cscColPtr,
                                         cusparseAction_t copyValues,
                                         cusparseIndexBase_t idxBase) {
#if CUDA_VERSION >= 10020
  return cusparse_csr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, CUDA_R_32F, copyValues,
			  idxBase);
#else
  return cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
#endif
}

inline cusparseStatus_t cusparse_csr2csc(cusparseHandle_t handle, int m, int n,
                                         int nnz, const double *csrVal,
                                         const int *csrRowPtr,
                                         const int *csrColInd, double *cscVal,
                                         int *cscRowInd, int *cscColPtr,
                                         cusparseAction_t copyValues,
                                         cusparseIndexBase_t idxBase) {
#if CUDA_VERSION >= 10020
  return cusparse_csr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, CUDA_R_64F, copyValues,
                          idxBase);
#else
  return cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
#endif
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
#if CUDA_VERSION >= 10020
  return cusparse_csrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                        csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc,
                        CUDA_R_32F); // overloaded with valtype (CUDA_R_32F)
#else
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
#endif
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
#if CUDA_VERSION >= 10020
  return cusparse_csrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                        csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc,
                        CUDA_R_64F); // overloaded with valtype (CUDA_R_64F)
#else
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
#endif
}


#endif // HAVE_CUDA
}
// namespace kaldi

#endif

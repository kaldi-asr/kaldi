#ifndef CLAPACK_H

#define CLAPACK_H
#include "cblas.h"

#ifndef ATL_INT
   #define ATL_INT int
#endif
#ifndef ATL_CINT
   #define ATL_CINT const ATL_INT
#endif
#ifndef ATLAS_ORDER
   #define ATLAS_ORDER CBLAS_ORDER
#endif
#ifndef ATLAS_UPLO
   #define ATLAS_UPLO CBLAS_UPLO
#endif
#ifndef ATLAS_DIAG
   #define ATLAS_DIAG CBLAS_DIAG
#endif
int clapack_sgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  float *A, const int lda, int *ipiv,
                  float *B, const int ldb);
int clapack_sgetrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   float *A, const int lda, int *ipiv);
int clapack_sgetrs
   (const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const float *A, const int lda,
    const int *ipiv, float *B, const int ldb);
int clapack_sgetri(const enum CBLAS_ORDER Order, const int N, float *A,
                   const int lda, const int *ipiv);
int clapack_sposv(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                  const int N, const int NRHS, float *A, const int lda,
                  float *B, const int ldb);
int clapack_spotrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);
int clapack_spotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const float *A, const int lda,
                   float *B, const int ldb);
int clapack_spotri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);
int clapack_slauum(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);
int clapack_strtri(const enum ATLAS_ORDER Order,const enum ATLAS_UPLO Uplo,
                   const enum ATLAS_DIAG Diag, const int N, float *A,
                   const int lda);
int clapack_sgels(const enum CBLAS_ORDER Order,
                  const enum CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, float *A,
                  ATL_CINT lda, float *B, const int ldb);
int clapack_sgelqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);
int clapack_sgeqlf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);
int clapack_sgerqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);
int clapack_sgeqrf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);

int clapack_dgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb);
int clapack_dgetrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv);
int clapack_dgetrs
   (const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const double *A, const int lda,
    const int *ipiv, double *B, const int ldb);
int clapack_dgetri(const enum CBLAS_ORDER Order, const int N, double *A,
                   const int lda, const int *ipiv);
int clapack_dposv(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                  const int N, const int NRHS, double *A, const int lda,
                  double *B, const int ldb);
int clapack_dpotrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dpotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const double *A, const int lda,
                   double *B, const int ldb);
int clapack_dpotri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dlauum(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dtrtri(const enum ATLAS_ORDER Order,const enum ATLAS_UPLO Uplo,
                   const enum ATLAS_DIAG Diag, const int N, double *A,
                   const int lda);
int clapack_dgels(const enum CBLAS_ORDER Order,
                  const enum CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, double *A,
                  ATL_CINT lda, double *B, const int ldb);
int clapack_dgelqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);
int clapack_dgeqlf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);
int clapack_dgerqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);
int clapack_dgeqrf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);

int clapack_cgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  void *A, const int lda, int *ipiv,
                  void *B, const int ldb);
int clapack_cgetrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   void *A, const int lda, int *ipiv);
int clapack_cgetrs
   (const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const void *A, const int lda,
    const int *ipiv, void *B, const int ldb);
int clapack_cgetri(const enum CBLAS_ORDER Order, const int N, void *A,
                   const int lda, const int *ipiv);
int clapack_cposv(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                  const int N, const int NRHS, void *A, const int lda,
                  void *B, const int ldb);
int clapack_cpotrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_cpotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const void *A, const int lda,
                   void *B, const int ldb);
int clapack_cpotri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_clauum(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_ctrtri(const enum ATLAS_ORDER Order,const enum ATLAS_UPLO Uplo,
                   const enum ATLAS_DIAG Diag, const int N, void *A,
                   const int lda);
int clapack_cgels(const enum CBLAS_ORDER Order,
                  const enum CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, void *A,
                  ATL_CINT lda, void *B, const int ldb);
int clapack_cgelqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_cgeqlf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_cgerqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_cgeqrf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);

int clapack_zgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  void *A, const int lda, int *ipiv,
                  void *B, const int ldb);
int clapack_zgetrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   void *A, const int lda, int *ipiv);
int clapack_zgetrs
   (const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const void *A, const int lda,
    const int *ipiv, void *B, const int ldb);
int clapack_zgetri(const enum CBLAS_ORDER Order, const int N, void *A,
                   const int lda, const int *ipiv);
int clapack_zposv(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                  const int N, const int NRHS, void *A, const int lda,
                  void *B, const int ldb);
int clapack_zpotrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_zpotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const void *A, const int lda,
                   void *B, const int ldb);
int clapack_zpotri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_zlauum(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_ztrtri(const enum ATLAS_ORDER Order,const enum ATLAS_UPLO Uplo,
                   const enum ATLAS_DIAG Diag, const int N, void *A,
                   const int lda);
int clapack_zgels(const enum CBLAS_ORDER Order,
                  const enum CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, void *A,
                  ATL_CINT lda, void *B, const int ldb);
int clapack_zgelqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_zgeqlf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_zgerqf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_zgeqrf(const enum CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);

#endif

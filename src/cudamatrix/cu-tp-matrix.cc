#if HAVE_CUDA==1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-sp-matrix.h"

namespace kaldi {

template<typename Real>
void CuTpMatrix<Real>::Cholesky(const CuSpMatrix<Real> &orig) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    CuMatrix<Real> tmp(orig);
    tmp.Cholesky();
    this->CopyFromMat(tmp, kNoTrans);
  } else
#endif
  {
    // TODO!
  }
}

#if HAVE_CUDA
template<typename Real> inline void cublas_trsm(int m, int n, Real alpha, const Real*\
                                                A, int lda, Real* B, int ldb) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_trsm<float>(int m, int n, float alpha, const float* A, \
                                          int lda, float* B, int ldb) {
  cublasStrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
template<> inline void cublas_trsm<double>(int m, int n, double alpha, const double* \
                                           A, int lda, double* B, int ldb) {
  cublasDtrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
#endif

template<typename Real>
void CuTpMatrix<Real>::Invert() {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(this->NumRows(), CU2DBLOCK));
    CuMatrix<Real> tmp(this->NumRows(), this->NumRows());
    int dim = this->NumRows();
    Real alpha = 1.0;
    cuda_set_diag(dimGrid, dimBlock, tmp.Data(), alpha, tmp.Dim());
    //Matrix<Real> A(dim,dim);
    //tmp.CopyToMat(&A);
    CuMatrix<Real> tmp2(dim, dim);
    tmp2.CopyFromTp(*this);
    cublas_trsm(dim, dim, alpha, tmp2.Data(), tmp2.Dim().stride, 
      tmp.Data(), tmp.Dim().stride);
    this->CopyFromMat(tmp, kNoTrans);
  } else
#endif
  {
    Mat().Invert();
  }
}

template<typename Real>
void CuTpMatrix<Real>::CopyFromMat(CuMatrixBase<Real> &M,
                                   MatrixTransposeType Trans) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CU2DBLOCK), n_blocks(M.NumRows(), CU2DBLOCK));
    if (Trans == kNoTrans) {
      cuda_take_lower(dimGrid, dimBlock, M.Data(), this->data_, M.Dim(), this->NumRows());
      cudaThreadSynchronize();
    } else {
      cuda_take_upper(dimGrid, dimBlock, M.Data(), this->data_, M.Dim(), this->NumRows());
      cudaThreadSynchronize();
    }      
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}


template class CuTpMatrix<float>;
template class CuTpMatrix<double>;

} // namespace

#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"
#include "cu-math.h"
#include "cu-sp-matrix.h"
#include "cu-matrix.h"

namespace kaldi {

template<typename Real>
void CuSpMatrix<Real>::CopyFromMat(const CuMatrixBase<Real> &M,
                                   SpCopyType copy_type) {
  KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT D = this->NumRows();
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CUBLOCK), n_blocks(M.NumRows(), CUBLOCK));
    switch (copy_type) {
      case kTakeMeanAndCheck:
        KALDI_LOG << "kTakeMeanAndCheck!";
      case kTakeMean:
        {
          cuda_take_mean(dimGrid, dimBlock, M.Data(), this->data_, M.Dim(), D);
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      case kTakeLower:
        {
          cuda_take_lower(dimGrid, dimBlock, M.Data(), this->data_, M.Dim(), D);
          CU_SAFE_CALL(cudaGetLastError());
          cudaThreadSynchronize();
        }
        break;
      case kTakeUpper:
        {
          cuda_take_upper(dimGrid, dimBlock, M.Data(), this->data_, M.Dim(), D);
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      default:
        KALDI_ASSERT("Invalid argument to CuSpMatrix::CopyFromMat");
    }
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::Invert", tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), kTakeLower);
  }
}

template<class Real>
void CuSpMatrix<Real>::Invert() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuMatrix<Real> mat(this->num_rows_, this->num_rows_);
    mat.CopyFromSp(*this);
    mat.InvertPSD();
    this->CopyFromMat(mat);
  } else
#endif
  { // Use inversion of CPU-based SpMatrix.
    Mat().Invert();
  }
}

#if HAVE_CUDA == 1
inline void CublasSpr(char uplo, int n, float alpha, const float *x,
                      int incx, float *AP) {
  cublasSspr(uplo, n, alpha, x, incx, AP);
}
inline void CublasSpr(char uplo, int n, double alpha, const double *x,
                      int incx, double *AP) {
  cublasDspr(uplo, n, alpha, x, incx, AP);
}
#endif

template<class Real>
void CuSpMatrix<Real>::AddVec2(const Real alpha, const CuVectorBase<Real> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    size_t nr = this->num_rows_;
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(nr, CUBLOCK), n_blocks(nr, CUBLOCK));

    CublasSpr('U', this->num_rows_, alpha, v.Data(),
              1, this->Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddVec2", tim.Elapsed());
  } else
#endif
  {
    Mat().AddVec2(alpha, v.Vec());
  }
}

#if HAVE_CUDA == 1
inline void CublasSyrk(char uplo, char trans, int n, int k,
                       float alpha, const float *A, int lda,
                       float beta, float *C, int ldc) {
  cublasSsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
inline void CublasSyrk(char uplo, char trans, int n, int k,
                       double alpha, const double *A, int lda,
                       double beta, double *C, int ldc) {
  cublasDsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
#endif

template<class Real>
void CuSpMatrix<Real>::AddMat2(const Real alpha, const CuMatrixBase<Real> &M,
                               MatrixTransposeType transM, const Real beta) {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows())
               || (transM == kTrans && this->NumRows() == M.NumCols()));

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT this_dim = this->NumRows(),
        m_other_dim = (transM == kNoTrans ? M.NumCols() : M.NumRows());

    if (this_dim == 0) return;
    if (alpha == 0.0) {
      if (beta != 1.0) this->Scale(beta);
      return;
    }

    char trans = (transM == kTrans ? 'N' : 'T');

    CuMatrix<Real> tmp_mat(*this);
    CublasSyrk('U', trans, this_dim, m_other_dim, alpha, M.Data(),
               M.Stride(), beta, tmp_mat.Data(), tmp_mat.Stride());
    this->CopyFromMat(tmp_mat, kTakeLower);
    
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddMat2", tim.Elapsed());
  } else
#endif
  {
    Mat().AddMat2(alpha, M.Mat(), transM, beta);
  }
}

/**
 * C++ templatd wrapper of ANSI-C CUBLAS function GEMM (matrix multiply)
 */
#if HAVE_CUDA == 1
template<typename Real> inline Real cublas_dot(int n, const Real *x, int incx, const Real* y, int incy) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline float cublas_dot<float>(int n, const float *x, int incx, const float *y, int incy) {
  return cublasSdot(n, x, incx, y, incy);
}
template<> inline double cublas_dot<double>(int n, const double *x, int incx, const double *y, int incy) {
  return cublasDdot(n, x, incx, y, incy);
}
#endif

double TraceSpSp(const CuSpMatrix<double> &A, const CuSpMatrix<double> &B) {
  double result;
  KALDI_ASSERT(A.NumRows() == B.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid = 1;
    int dimBlock= A.NumRows();

    // copy the diagonal componenets

    size_t nr = static_cast<size_t>(A.NumRows()),
        num_elems = (nr * (nr+1)) / 2,
        num_bytes = nr * sizeof(double);
                     
    double* diag_A;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&diag_A), num_bytes));
    cuda_copy_diag(dimGrid, dimBlock, diag_A, A.Data(), A.NumRows());

    double* diag_B;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&diag_B), num_bytes));    
    cuda_copy_diag(dimGrid, dimBlock, diag_B, B.Data(), B.NumRows());

    double dot_diag = 0.0;
    dot_diag = cublas_dot(A.NumRows(), diag_A, 1, diag_B, 1);
    double dot_all = 0.0;
    dot_all = cublas_dot(num_elems, A.Data(), 1, B.Data(), 1);
    
    result = 2 * dot_all - dot_diag;
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::TraceSpSp", tim.Elapsed());
  } else
#endif
  {
    result = TraceSpSp(A.Mat(), B.Mat());
  }
  return result;
}

float TraceSpSp(const CuSpMatrix<float> &A, const CuSpMatrix<float> &B) {
  float result;
  KALDI_ASSERT(A.NumRows() == B.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid = 1;
    int dimBlock= A.NumRows();

    // copy the diagonal componenets

    size_t nr = static_cast<size_t>(A.NumRows()),
        num_elems = (nr * (nr+1)) / 2,
        num_bytes = nr * sizeof(float);
                     
    float* diag_A;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&diag_A), num_bytes));
    cuda_copy_diag(dimGrid, dimBlock, diag_A, A.Data(), A.NumRows());

    float* diag_B;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&diag_B), num_bytes));    
    cuda_copy_diag(dimGrid, dimBlock, diag_B, B.Data(), B.NumRows());

    float dot_diag = 0.0;
    dot_diag = cublas_dot(A.NumRows(), diag_A, 1, diag_B, 1);
    float dot_all = 0.0;
    dot_all = cublas_dot(num_elems, A.Data(), 1, B.Data(), 1);
    
    result = 2 * dot_all - dot_diag;
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::TraceSpSp", tim.Elapsed());
  } else
#endif
  {
    result = TraceSpSp(A.Mat(), B.Mat());
  }
  return result;
}

double TraceSpSp(const CuSpMatrix<double> &A, const CuSpMatrix<float> &B) {
  double result;
  KALDI_ASSERT(A.NumRows() == B.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid = 1;
    int dimBlock = A.NumRows();

    double* device_result;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(double)));
    CU_SAFE_CALL(cudaMemset(device_result,0, sizeof(double)));
    cuda_trace_sp_sp_df(dimGrid, dimBlock, A.Data(), B.Data(), device_result, A.NumRows());
    CU_SAFE_CALL(cudaGetLastError());
    CU_SAFE_CALL(cudaMemcpy(&result, device_result, sizeof(double), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::TraceSpSp", tim.Elapsed());
  } else
#endif
  {
    result = TraceSpSp(A.Mat(), B.Mat());
  }
  return result;
}

float TraceSpSp(const CuSpMatrix<float> &A, const CuSpMatrix<double> &B) {
  float result;
  KALDI_ASSERT(A.NumRows() == B.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid = 1;
    int dimBlock = A.NumRows();

    float* device_result;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(float)));
    CU_SAFE_CALL(cudaMemset(device_result,0, sizeof(float)));
    cuda_trace_sp_sp_fd(dimGrid, dimBlock, A.Data(), B.Data(), device_result, A.NumRows());
    CU_SAFE_CALL(cudaGetLastError());
    CU_SAFE_CALL(cudaMemcpy(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::TraceSpSp", tim.Elapsed());
  } else
#endif
  {
    result = TraceSpSp(A.Mat(), B.Mat());
  }
  return result;
}

template class CuSpMatrix<float>;
template class CuSpMatrix<double>;

} // namespace

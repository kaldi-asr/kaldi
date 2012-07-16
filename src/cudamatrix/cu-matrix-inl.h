// cudamatrix/cu-matrix-inl.h

// Copyright 2009-2012  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CUMATRIX_INL_H_
#define KALDI_CUDAMATRIX_CUMATRIX_INL_H_

#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
  #include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"

namespace kaldi {


template<typename Real>
const Real* CuMatrix<Real>::Data() const {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_;
  } else 
  #endif
  {
    return mat_.Data();
  }
}



template<typename Real>
Real* CuMatrix<Real>::Data() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_;
  } else 
  #endif
  {
    return mat_.Data();
  }
}



template<typename Real>
const Real* CuMatrix<Real>::RowData(MatrixIndexT r) const { 
  assert(r < NumRows()); 
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_+r*stride_; 
  } else
  #endif
  {
    return mat_.RowData(r);
  }
}



template<typename Real>
Real* CuMatrix<Real>::RowData(MatrixIndexT r) {
  assert(r < NumRows()); 
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_+r*stride_; 
  } else
  #endif
  {
    return mat_.RowData(r);
  }
}



template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols) {
  if (num_rows_ == rows && num_cols_ == cols) {
    // SetZero();
    return *this;
  }

  Destroy();

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    MatrixIndexT row_bytes = cols * sizeof(Real);
    size_t pitch;
    cuSafeCall(cudaMallocPitch((void**)&data_, &pitch, row_bytes, rows));
    num_rows_ = rows; num_cols_ = cols; 
    stride_ = pitch/sizeof(Real);
    SetZero();
  } else
  #endif
  {
    mat_.Resize(rows, cols);
    num_rows_=rows;
    num_cols_=cols;
    stride_=mat_.Stride();
  }
  
  return *this;
}



template<typename Real>
void CuMatrix<Real>::Destroy() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    if (NULL != data_) {
      cuSafeCall(cudaFree(data_));
      data_ = NULL;
    }
  } else
  #endif
  {
    mat_.Destroy();
  }
  num_rows_ = num_cols_ = stride_ = 0;
}



template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::CopyFromMat(const CuMatrix<Real> &src) {
  Resize(src.NumRows(), src.NumCols());
 
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch, width, src.NumRows(), cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatD2D",tim.Elapsed());
  } else
  #endif
  {
    mat_.CopyFromMat(src.mat_);
  }

  return *this;
}



template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::CopyFromMat(const Matrix<Real> &src) {
  Resize(src.NumRows(), src.NumCols());

  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch, width, src.NumRows(), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatH2D",tim.Elapsed());
  } else
  #endif
  {
    mat_.CopyFromMat(src);
  }

  return *this;
}



template<typename Real>
void CuMatrix<Real>::CopyToMat(Matrix<Real> *dst) const {
  if (dst->NumRows() != NumRows()  ||  dst->NumCols() != NumCols()) {
    dst->Resize(NumRows(), NumCols());
  }

  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 

    Timer tim;
   
    MatrixIndexT src_pitch = stride_*sizeof(Real);
    MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
    MatrixIndexT width = NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(dst->Data(), dst_pitch, Data(), src_pitch, width, NumRows(), cudaMemcpyDeviceToHost));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
  } else
  #endif
  {
    dst->CopyFromMat(mat_);
  }
}



template<typename Real>
void CuMatrix<Real>::CopyRowsFromMat(int32 r, const CuMatrix<Real> &src, int32 src_ro, int32 dst_ro) {
  assert(r+src_ro <= src.NumRows());
  assert(r+dst_ro <= NumRows());
  assert(NumCols() == src.NumCols());
   
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);

    const Real *p_src = src.Data() + src_ro*src.Stride();  
    Real *p_dst = data_ + dst_ro*stride_;

    cuSafeCall(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, r, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyRowsD2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(Data()+dst_ro*stride_, src.Data()+src_ro*src.Stride(), r*stride_*sizeof(Real));
  }
}



template<typename Real>
void CuMatrix<Real>::Read(std::istream &is, bool binary) {
  Matrix<BaseFloat> tmp;
  tmp.Read(is, binary);
  CopyFromMat(tmp);    
}



template<typename Real>
void CuMatrix<Real>::Write(std::ostream &os, bool binary) const {
  Matrix<BaseFloat> tmp;
  CopyToMat(&tmp);
  tmp.Write(os, binary); 
}



template<typename Real>
void CuMatrix<Real>::SetZero() {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, num_rows_*stride_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero",tim.Elapsed());
  } else
  #endif
  {
    mat_.SetZero();
  }
}



/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrix<Real> &mat) {
  Matrix<Real> tmp;
  mat.CopyToMat(&tmp);
  out << tmp;
  return out;
}



/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real> 
void CuMatrix<Real>::Set(Real value) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_set_const(dimGrid, dimBlock, data_, value, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.Set(value);
  }
}



template<typename Real> 
void CuMatrix<Real>::ApplyLog() { 
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_apply_log(dimGrid, dimBlock, data_, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.ApplyLog();
  }
}



template<typename Real>
void CuMatrix<Real>::MulElements(const CuMatrix<Real>& A) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(num_cols_ == A.NumCols());
    assert(num_rows_ == A.NumRows());
    assert(stride_ == A.Stride());
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_elements(dimGrid, dimBlock, data_, A.Data(), Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.MulElements(A.mat_);
  }
}



template<typename Real>
void CuMatrix<Real>::MulColsVec(const CuVector<Real> &scale) {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    assert(scale.Dim() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_cols_vec(dimGrid, dimBlock, data_, scale.Data(), Dim());
    cuSafeCall(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.MulColsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrix<Real>::MulRowsVec(const CuVector<Real> &scale) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(scale.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_rows_vec(dimGrid, dimBlock, data_, scale.Data(), Dim());
    cuSafeCall(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    mat_.MulRowsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrix<Real>::DivRowsVec(const CuVector<Real> &div) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(div.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_div_rows_vec(dimGrid, dimBlock, data_, div.Data(), Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    Vector<Real> tmp(div.Vec());
    tmp.InvertElements();
    mat_.MulRowsVec(tmp);
  }
}



template<typename Real>
void CuMatrix<Real>::AddMat(Real alpha, const CuMatrix<Real>& A, Real beta) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    assert(A.NumRows() == NumRows());
    assert(A.NumCols() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_mat(dimGrid, dimBlock, alpha, A.Data(), beta, data_, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.Scale(beta);
    mat_.AddMat(alpha, A.mat_);
  }
}



template<typename Real>
void CuMatrix<Real>::AddVecToCols(Real alpha, const CuVector<Real> &col, Real beta) { 
  
  if (col.Dim() != NumRows()) {
    KALDI_ERR << "Non matching dimensions: Rows:" << NumRows() << " VectorDim:" << col.Dim();
  }

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_cols(dimGrid, dimBlock, alpha, col.Data(), beta, data_, Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.Scale(beta);
    mat_.AddVecToCols(alpha,col.Vec());
  }
}



template<typename Real>
void CuMatrix<Real>::AddVecToRows(Real alpha, const CuVector<Real> &row, Real beta) { 
  
  if (row.Dim() != NumCols()) {
    KALDI_ERR << "Non matching dimensions: Cols:" << NumCols() << " VectorDim:" << row.Dim();
  }

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_rows(dimGrid, dimBlock, alpha, row.Data(), beta, data_, Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.Scale(beta);
    mat_.AddVecToRows(alpha,row.Vec());
  }
}



/**
 * C++ templated wrapper of ANSI-C CUBLAS function GEMM (matrix multiply)
 */
#if HAVE_CUDA==1
template<typename Real> inline void cublas_gemm(char transa, char transb, int m, int n,int k, Real alpha, const Real *A, int lda,const Real *B, int ldb, Real beta, Real *C, int ldc) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_gemm<float>(char transa, char transb, int m, int n,int k, float alpha, const float *A, int lda,const float *B, int ldb, float beta, float *C, int ldc) {
  cublasSgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
template<> inline void cublas_gemm<double>(char transa, char transb, int m, int n,int k, double alpha, const double *A, int lda,const double *B, int ldb, double beta, double *C, int ldc) {
  cublasDgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
#endif



/*
 * Method wrapping the CUBLAS function GEMM
 */
template<typename Real>
void CuMatrix<Real>::AddMatMat(
               Real alpha, const CuMatrix<Real>& A, MatrixTransposeType transA,
               const CuMatrix<Real>& B, MatrixTransposeType transB, Real beta) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
    // keep trans..., just swap A&B matrices: A->B B->A
    MatrixIndexT m = ((transB==kTrans)? B.NumRows() : B.NumCols()); 
    MatrixIndexT n = ((transA==kTrans)? A.NumCols() : A.NumRows());
    MatrixIndexT k = ((transB==kTrans)? B.NumCols() : B.NumRows());
    MatrixIndexT k1 = ((transA==kTrans)? A.NumRows() : A.NumCols());

    assert(m == NumCols());
    assert(n == NumRows());
    assert(k == k1);

    Timer tim;

    cublas_gemm((transB==kTrans?'T':'N'), (transA==kTrans?'T':'N'), m, n, k, 
                alpha, B.Data(), B.Stride(), A.Data(), A.Stride(), 
                beta, data_, Stride());

    cuSafeCall(cublasGetError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    mat_.AddMatMat(alpha, A.mat_, transA, B.mat_, transB, beta);
  }
}



} // namespace kaldi

#endif

  

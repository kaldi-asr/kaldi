// cudamatrix/cu-matrix.cc

// Copyright 2009-2012  Karel Vesely
//                      Lucas Ondel
//                      Johns Hopkins University (author: Daniel Povey)

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


#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-randkernels.h"
#include "cudamatrix/cu-rand-inl.h"
#include "cudamatrix/cu-choleskykernels.h"
#include "cudamatrix/cu-stlvector.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"

namespace kaldi {

template<typename Real>
void CuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols,
                            MatrixResizeType resize_type) {
  // This code does not currently support the other resize_type options.
  KALDI_ASSERT(resize_type == kSetZero || resize_type == kUndefined);
  if (rows * cols == 0) KALDI_ASSERT(rows == 0 && cols == 0);
  if (this->num_rows_ == rows && this->num_cols_ == cols) {
    if (resize_type == kSetZero) this->SetZero();
    return;
  }

  if (this->num_rows_ != 0)
    this->Destroy();
  if (rows == 0) return;  
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    MatrixIndexT row_bytes = cols * sizeof(Real);
    size_t pitch;
    CU_SAFE_CALL(cudaMallocPitch(reinterpret_cast<void**>(&this->data_), &pitch,
                                 row_bytes, rows));
    this->num_rows_ = rows;
    this->num_cols_ = cols; 
    this->stride_ = pitch / sizeof(Real);
    if (resize_type == kSetZero) this->SetZero();
  } else
#endif
  { // Let the initializer of Matrix<Real> handle the allocation,
    // and then just do Swap which will switch the pointers.
    // This wastes a few instructions but is simple to code.
    Matrix<Real> mat(rows, cols, resize_type);
    this->Swap(&mat);
  }
}

template<typename Real>
void CuMatrix<Real>::Destroy() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    if (this->data_ != NULL) {
      CU_SAFE_CALL(cudaFree(this->data_));
    }
  } else
#endif
  {
    if (this->data_ != NULL) KALDI_MEMALIGN_FREE(this->data_);
  }
  this->data_ = NULL;
  this->num_rows_ = 0;
  this->num_cols_ = 0;
  this->stride_ = 0;
}

template<typename Real>
void CuMatrix<Real>::Swap(CuMatrix<Real> *mat) {
  std::swap(mat->data_, this->data_);
  std::swap(mat->num_cols_, this->num_cols_);
  std::swap(mat->num_rows_, this->num_rows_);
  std::swap(mat->stride_, this->stride_);
}


template<typename Real>
void CuMatrix<Real>::Swap(Matrix<Real> *mat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) {
      if (mat->num_rows_ != 0) {
        // *this is empty, but mat is nonempty.
        this->Resize(mat->num_rows_, mat->num_cols_, kUndefined);
        this->CopyFromMat(*mat);
        mat->Resize(0, 0);
      }
      // else both are empty.
    } else { // *this is nonempty.
      if (mat->num_rows_ != 0) {
        // Both *this and *mat are nonempty.  Recurse to simpler cases.
        // this could be done more efficiently in the case where
        // the size does not change.
        Matrix<Real> temp;
        this->Swap(&temp); // now temp is full, *this is empty.
        mat->Swap(&temp); // now mat has data from *this, temp has
        // data from mat.
        this->Swap(&temp); // copy data in mat to *this, which is now empty.
      } else { // *this is full but *mat is empty.
        mat->Resize(this->num_rows_, this->num_cols_, kUndefined);
        this->CopyToMat(mat);
        this->Destroy();
      }
    }
  } else
#endif
  {
    std::swap(mat->data_, this->data_);
    std::swap(mat->num_cols_, this->num_cols_);
    std::swap(mat->num_rows_, this->num_rows_);
    std::swap(mat->stride_, this->stride_);
  }
}

template <typename Real>
template <typename OtherReal>
 CuMatrix<Real>::CuMatrix(const CuTpMatrix<OtherReal> & M,
                          MatrixTransposeType trans): CuMatrixBase<Real>() {
  if (trans == kNoTrans) {
    Resize(M.NumRows(), M.NumCols(), kUndefined);
    this->CopyFromTp(M);
  } else {
    Resize(M.NumCols(), M.NumRows(), kUndefined);
    this->CopyFromTp(M, kTrans);
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const CuMatrixBase<Real> &src,
                                     MatrixTransposeType trans) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);
      Timer tim;
      
      MatrixIndexT dst_pitch = stride_ * sizeof(Real);
      MatrixIndexT src_pitch = src.Stride() * sizeof(Real);
      MatrixIndexT width = src.NumCols() * sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.data_, src_pitch,
                                width, src.num_rows_, cudaMemcpyDeviceToDevice));
      
      CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatD2D",tim.Elapsed());
    } else {
      this->CopyFromMat<Real>(src, trans);
      // call double-templated version which we'll make sure works for the
      // transposed case.
    }
  } else
#endif
  {
    Mat().CopyFromMat(src.Mat(), trans);
  }
}

template<>
template<>
void CuMatrixBase<double>::CopyFromMat(const CuMatrixBase<float> &M,
                                       MatrixTransposeType Trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CU2DBLOCK), n_blocks(M.NumRows(), CU2DBLOCK));
    if (Trans == kNoTrans) {
      KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
      cuda_copy_from_mat_df(dimGrid, dimBlock, data_, M.data_,
                            Dim(), M.Dim());
    } else {
      KALDI_ASSERT(num_rows_ == M.NumCols() && num_cols_ == M.NumRows ());
      cuda_copy_from_mat_df_trans(dimGrid, dimBlock, data_, M.data_, Dim(), M.Dim());
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}

template<>
template<>
void CuMatrixBase<double>::CopyFromMat(const CuMatrixBase<double> &M,
                                       MatrixTransposeType Trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CU2DBLOCK), n_blocks(M.NumRows(), CU2DBLOCK));
    if (Trans == kNoTrans) {
      KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
      cuda_copy_from_mat_dd(dimGrid, dimBlock, data_, M.data_,
                            Dim(), M.Dim());
    } else {
      KALDI_ASSERT(num_rows_ == M.NumCols() && num_cols_ == M.NumRows ());
      cuda_copy_from_mat_dd_trans(dimGrid, dimBlock, data_, M.data_, Dim(), M.Dim());
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}


template<>
template<>
void CuMatrixBase<float>::CopyFromMat(const CuMatrixBase<double> &M,
                                      MatrixTransposeType Trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CU2DBLOCK), n_blocks(M.NumRows(), CU2DBLOCK));
    if (Trans == kNoTrans) {
      KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());

      cuda_copy_from_mat_fd(dimGrid, dimBlock, data_, M.data_,
                            Dim(), M.Dim());
    } else {

      KALDI_ASSERT(num_rows_ == M.NumCols() && num_cols_ == M.NumRows ());
      cuda_copy_from_mat_fd_trans(dimGrid, dimBlock, data_, M.Data(),
                                    Dim(), M.Dim());
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}

template<>
template<>
void CuMatrixBase<float>::CopyFromMat(const CuMatrixBase<float> &M,
                                      MatrixTransposeType Trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(M.NumCols(), CU2DBLOCK), n_blocks(M.NumRows(), CU2DBLOCK));
    if (Trans == kNoTrans) {
      KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());

      cuda_copy_from_mat_ff(dimGrid, dimBlock, data_, M.data_,
                            Dim(), M.Dim());
    } else {

      KALDI_ASSERT(num_rows_ == M.NumCols() && num_cols_ == M.NumRows ());
      cuda_copy_from_mat_ff_trans(dimGrid, dimBlock, data_, M.Data(),
                                  Dim(), M.Dim());
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}



template<typename Real>
template<typename OtherReal>
void CuMatrixBase<Real>::CopyFromTp(const CuTpMatrix<OtherReal> &M,
                                    MatrixTransposeType Trans) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid = 1;
    int dimBlock = num_rows_;
    SetZero();
    if (Trans == kNoTrans) {
      cuda_copy_from_tp(dimGrid, dimBlock, data_, M.Data(), Dim());
    } else {
      cuda_copy_from_tp_trans(dimGrid, dimBlock, data_, M.Data(), Dim());      
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
  } else
#endif
  {
    Mat().CopyFromTp(M.Mat(), Trans);
  }
}
// instantiate the template above.
template void CuMatrixBase<float>::CopyFromTp(const CuTpMatrix<float> &M,
                                              MatrixTransposeType Trans);
template void CuMatrixBase<float>::CopyFromTp(const CuTpMatrix<double> &M,
                                              MatrixTransposeType Trans);
template void CuMatrixBase<double>::CopyFromTp(const CuTpMatrix<float> &M,
                                              MatrixTransposeType Trans);
template void CuMatrixBase<double>::CopyFromTp(const CuTpMatrix<double> &M,
                                              MatrixTransposeType Trans);

/*
// template instantiations.
template
void CuMatrixBase<float>::CopyFromMat(const CuMatrixBase<double> & M,
                                      MatrixTransposeType Trans);
template
void CuMatrixBase<double>::CopyFromMat(const CuMatrixBase<float> & M,
                                       MatrixTransposeType Trans);

template
void CuMatrixBase<float>::CopyFromMat(const CuMatrixBase<float> & M,
                                      MatrixTransposeType Trans);
template
void CuMatrixBase<double>::CopyFromMat(const CuMatrixBase<double> & M,
MatrixTransposeType Trans);
*/

template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const MatrixBase<Real> &src,
                                     MatrixTransposeType trans) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);      
      Timer tim;

      MatrixIndexT dst_pitch = stride_*sizeof(Real);
      MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
      MatrixIndexT width = src.NumCols()*sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch,
                                width, src.NumRows(), cudaMemcpyHostToDevice));
      
      CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatH2D",tim.Elapsed());
    } else {
      CuMatrix<Real> trans_mat(src); // Do the transpose on the GPU board.
      this->CopyFromMat(trans_mat, kTrans);
    }
  } else
#endif
  {
    Mat().CopyFromMat(src, trans);
  }
}

template<typename Real>
template<typename OtherReal>
void CuMatrixBase<Real>::CopyFromMat(const MatrixBase<OtherReal> &src,
                                     MatrixTransposeType trans) {
  CuMatrix<OtherReal> temp(src);
  this->CopyFromMat(temp, trans);
}


template<typename Real>
void CuMatrixBase<Real>::CopyFromSp(const CuSpMatrix<Real> &M) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(NumRows(),CU2DBLOCK));
    cuda_copy_from_sp(dimGrid, dimBlock, M.Data(), data_, num_rows_, Dim());
    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromSp",tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromSp(M.Mat());
  }
}

template<typename Real>
CuMatrix<Real>::CuMatrix(const CuMatrix<Real> &other, MatrixTransposeType trans) {
  if (trans == kNoTrans)
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
  else
    this->Resize(other.NumCols(), other.NumRows(), kUndefined);
  this->CopyFromMat(other, trans);
}

template<typename Real>
CuMatrix<Real>::CuMatrix(const CuMatrixBase<Real> &other, MatrixTransposeType trans) {
  if (trans == kNoTrans)
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
  else
    this->Resize(other.NumCols(), other.NumRows(), kUndefined);
  this->CopyFromMat(other, trans);
}


template<typename Real>
template<typename OtherReal>
CuMatrix<Real>::CuMatrix(const MatrixBase<OtherReal> &other, MatrixTransposeType trans) {
  if (trans == kNoTrans)
    this->Resize(other.NumRows(), other.NumCols(), kUndefined);
  else
    this->Resize(other.NumCols(), other.NumRows(), kUndefined);
  this->CopyFromMat(other, trans);
}
// Instantiate the template above.
template
CuMatrix<float>::CuMatrix(const MatrixBase<float> &other, MatrixTransposeType trans);
template
CuMatrix<double>::CuMatrix(const MatrixBase<float> &other, MatrixTransposeType trans);
template
CuMatrix<float>::CuMatrix(const MatrixBase<double> &other, MatrixTransposeType trans);
template
CuMatrix<double>::CuMatrix(const MatrixBase<double> &other, MatrixTransposeType trans);


template<typename Real>
template<typename OtherReal>
void CuMatrixBase<Real>::CopyToMat(MatrixBase<OtherReal> *dst,
                                   MatrixTransposeType trans) const {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kTrans || sizeof(OtherReal) != sizeof(Real)) {
      CuMatrix<OtherReal> this_trans(*this, trans);
      this_trans.CopyToMat(dst, kNoTrans);
    } else {
      KALDI_ASSERT(dst->NumRows() == NumRows() && dst->NumCols() == NumCols());
      Timer tim;
   
      MatrixIndexT src_pitch = stride_*sizeof(Real);
      MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
      MatrixIndexT width = NumCols()*sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(dst->Data(), dst_pitch, this->data_, src_pitch,
                                width, this->num_rows_, cudaMemcpyDeviceToHost));

      CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
    }
  } else
  #endif
  {
    dst->CopyFromMat(Mat(), trans);
  }
}

// instantiate the template above.
template
void CuMatrixBase<float>::CopyToMat(MatrixBase<float> *dst,
                                    MatrixTransposeType trans) const;
template
void CuMatrixBase<double>::CopyToMat(MatrixBase<float> *dst,
                                     MatrixTransposeType trans) const;
template
void CuMatrixBase<float>::CopyToMat(MatrixBase<double> *dst,
                                    MatrixTransposeType trans) const;
template
void CuMatrixBase<double>::CopyToMat(MatrixBase<double> *dst,
                                     MatrixTransposeType trans) const;





template<typename Real>
void CuMatrix<Real>::Read(std::istream &is, bool binary) {
  Matrix<Real> temp;
  temp.Read(is, binary);
  Destroy();
  Swap(&temp);
}

template<typename Real>
void CuMatrix<Real>::Write(std::ostream &os, bool binary) const {
  Matrix<Real> temp(this->num_rows_, this->num_cols_, kUndefined);
  this->CopyToMat(&temp);
  temp.Write(os, binary); 
}

template<typename Real>
void CuMatrixBase<Real>::SetZero() {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    CU_SAFE_CALL(cudaMemset(data_, 0, num_rows_*stride_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero", tim.Elapsed());
  } else
  #endif
  {
    Mat().SetZero();
  }
}




/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real> 
void CuMatrixBase<Real>::Set(Real value) {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_set_const(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Set(value);
  }
}

// set zero the upper diagonal
// no cpu implementation yet. Check with Dan.
template<typename Real>
void CuMatrixBase<Real>::SetZeroUpperDiag() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_set_zero_above_diag(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real> 
void CuMatrixBase<Real>::Add(Real value) { 
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_add(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Add(value);
  }
}


template<typename Real> 
void CuMatrixBase<Real>::Scale(Real value) { 
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_scale(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(value);
  }
}



template<typename Real> 
void CuMatrixBase<Real>::ApplyLog() { 
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_log(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().ApplyLog();
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulElements(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());
    
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_mul_elements(dimGrid, dimBlock, data_, A.data_, Dim(), A.Stride(), this->Stride());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().MulElements(A.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::Max(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());
    
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_max(dimGrid, dimBlock, data_, A.data_, Dim(), A.Stride(), this->Stride());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Max(A.Mat());
  }
}


template<typename Real>
void CuMatrixBase<Real>::MulColsVec(const CuVectorBase<Real> &scale) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumCols());

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_mul_cols_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().MulColsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulRowsVec(const CuVectorBase<Real> &scale) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumRows());

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_mul_rows_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    Mat().MulRowsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::DivRowsVec(const CuVectorBase<Real> &div) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(div.Dim() == NumRows());

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_div_rows_vec(dimGrid, dimBlock, data_, div.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
#endif
  {
    Vector<Real> temp(div.Vec()); // will copy.
    temp.InvertElements();
    Mat().MulRowsVec(temp);
  }
}



template<typename Real>
void CuMatrixBase<Real>::AddMat(Real alpha, const CuMatrixBase<Real>& A, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(A.NumRows() == NumRows());
    KALDI_ASSERT(A.NumCols() == NumCols());

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_add_mat(dimGrid, dimBlock, alpha, A.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(beta);
    Mat().AddMat(alpha, A.Mat());
  }
}



template<typename Real>
void CuMatrixBase<Real>::AddVecToCols(Real alpha,
                                      const CuVectorBase<Real> &col,
                                      Real beta) { 
  if (col.Dim() != NumRows()) {
    KALDI_ERR << "Non matching dimensions: Rows:" << NumRows() << " VectorDim:" << col.Dim();
  }

  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_add_vec_to_cols(dimGrid, dimBlock, alpha, col.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToCols(alpha, col.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::AddVecToRows(Real alpha,
                                      const CuVectorBase<Real> &row,
                                      Real beta) { 
  if (row.Dim() != NumCols()) {
    KALDI_ERR << "Non matching dimensions: Cols:" << NumCols() << " VectorDim:" << row.Dim();
  }
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_add_vec_to_rows(dimGrid, dimBlock, alpha, row.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToRows(alpha, row.Vec());
  }
}



/**
 * C++ templated wrapper of ANSI-C CUBLAS function GEMM (matrix multiply)
 */
#if HAVE_CUDA == 1
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
void CuMatrixBase<Real>::AddMatMat(
    Real alpha, const CuMatrixBase<Real>& A, MatrixTransposeType transA,
    const CuMatrixBase<Real>& B, MatrixTransposeType transB, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
    // keep trans..., just swap A&B matrices: A->B B->A
    MatrixIndexT m = ((transB==kTrans)? B.NumRows() : B.NumCols()); 
    MatrixIndexT n = ((transA==kTrans)? A.NumCols() : A.NumRows());
    MatrixIndexT k = ((transB==kTrans)? B.NumCols() : B.NumRows());
    MatrixIndexT k1 = ((transA==kTrans)? A.NumRows() : A.NumCols());

    KALDI_ASSERT(m == NumCols());
    KALDI_ASSERT(n == NumRows());
    KALDI_ASSERT(k == k1);

    Timer tim;

    cublas_gemm((transB==kTrans?'T':'N'), (transA==kTrans?'T':'N'), m, n, k, 
                alpha, B.data_, B.Stride(), A.data_, A.Stride(), 
                beta, data_, Stride());

    CU_SAFE_CALL(cublasGetError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddMatMat(alpha, A.Mat(), transA, B.Mat(), transB, beta);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddDiagVecMat(
    const Real alpha, CuVectorBase<Real> &v,
    const CuMatrixBase<Real> &M, MatrixTransposeType transM,
    Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (transM == kNoTrans) {
      KALDI_ASSERT(SameDim(*this, M));
    } else {
      KALDI_ASSERT(M.NumRows() == NumCols() && M.NumCols() == NumRows());
    }
    KALDI_ASSERT(v.Dim() == this->NumRows());

    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    // Caution, this dimGrid is not the same way around as much of the other
    // code: going forward, I want to use the (rows, cols) order.
    dim3 dimGrid(n_blocks(num_rows_, CU2DBLOCK), n_blocks(num_cols_, CU2DBLOCK));

    MatrixIndexT M_row_stride = M.Stride(), M_col_stride = 1;
    if (transM == kTrans) std::swap(M_row_stride, M_col_stride);

    cuda_add_diag_vec_mat(dimGrid, dimBlock, alpha, data_, Dim(),
                          v.Data(), M.Data(), M_row_stride, M_col_stride, beta);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddDiagVecMat(alpha, v.Vec(), M.Mat(), transM, beta);
  }
}  


template<typename Real>
void CuMatrixBase<Real>::Sigmoid(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CU2DBLOCK), n_blocks(src.NumRows(), CU2DBLOCK));

    cuda_sigmoid(dimGrid, dimBlock, this->data_, src.data_, src.Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Sigmoid(src.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::SoftHinge(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CU2DBLOCK), n_blocks(src.NumRows(), CU2DBLOCK));

    cuda_soft_hinge(dimGrid, dimBlock, this->data_, src.data_, src.Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().SoftHinge(src.Mat());
  }
}


template<typename Real> // Y->this, X->src
void CuMatrixBase<Real>::ApplySoftMaxPerRow(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
/*
#if 1
    // enable 'tree-reduce' functions, 
    //find maximum in each row (tree reduction)
    CuStlVector<int32> max_id;
    src.FindRowMaxId(&max_id); 
    //in each row subtract maximum, apply exp (grid kernel)
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(src.num_cols_, CU2DBLOCK), n_blocks(src.num_rows_, CU2DBLOCK));
    cuda_softmax_part(dimGrid, dimBlock, src.data_, max_id.Data(), this->data_, src.Dim()); 
    //sum the rows to get normalizers (tree reduction) 
    CuVector<Real> sum(src.num_rows_);
    sum.AddColSumMat(1.0, *this, 0.0);
    //divide by normalizers to get posteriors (grid kernel)
    this->DivRowsVec(sum);
#else
    // disable 'tree-reduce' functions, 
    // slower, but can be used for debugging
    size_t dimBlock = CU2DBLOCK;
    size_t dimGrid  = n_blocks(src.num_rows_, CU2DBLOCK);

    cuda_softmax(dimGrid, dimBlock, data_, src.data_, src.Dim());
    CU_SAFE_CALL(cudaGetLastError());
#endif
*/

    size_t dimBlock = src.num_cols_ > CU1DBLOCK ? CU1DBLOCK : src.num_cols_;
    size_t dimGrid = src.num_rows_;
    cuda_softmax_reduce(dimGrid, dimBlock, data_, src.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    MatrixBase<Real> &mat(this->Mat());
    mat.CopyFromMat(src.Mat());
    for(MatrixIndexT r = 0; r < mat.NumRows(); r++) {
      mat.Row(r).ApplySoftMax();
    }
  }
}

// DiffSigmoid(Ein, Y, Eout) -> Eout.DiffSigmoid(Y, Ein).
template<typename Real> // Eout -> *this, Ein -> diff, Y -> value
void CuMatrixBase<Real>::DiffSigmoid(const CuMatrixBase<Real> &value,
                                     const CuMatrixBase<Real> &diff) {
  KALDI_ASSERT(SameDimAndStride(*this, value) && SameDimAndStride(*this, diff));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK), n_blocks(num_rows_, CU2DBLOCK));

    cuda_diff_sigmoid(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().DiffSigmoid(value.Mat(), diff.Mat());
  }
}

  
template<typename Real>
void CuMatrixBase<Real>::Tanh(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CU2DBLOCK), n_blocks(src.NumRows(), CU2DBLOCK));

    cuda_tanh(dimGrid, dimBlock, this->data_, src.data_, src.Dim());
    CU_SAFE_CALL(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Tanh(src.Mat());
  }
}



template<typename Real> // Ein -> diff, Y -> value
void CuMatrixBase<Real>::DiffTanh(const CuMatrixBase<Real> &value,
                                  const CuMatrixBase<Real> &diff) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK), n_blocks(num_rows_, CU2DBLOCK));

    cuda_diff_tanh(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().DiffTanh(value.Mat(), diff.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::FindRowMaxId(CuStlVector<int32> *id) const {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
     
    // initialize the vectors
    CuVector<Real> max(num_rows_);
    max.Set(-1e21);
    id->Resize(num_rows_);
    id->Set(-1);

    MatrixDim d=Dim();// only stride will be used!
   
    // process per 256 column blocks 
    for(int32 block=0; (block+1)*256 <= num_cols_; block++) {
      dim3 dimBlock(256, 1);
      dim3 dimGrid(1, num_rows_);
      int32 offset=block*256;

      cuda_find_row_max_id(dimGrid, dimBlock, data_ + offset,
                           max.data_, id->Data(), offset, d);
    }
    
    // process the remainder
    int32 div = num_cols_ / 256;
    int32 mod = num_cols_ % 256;
    if (mod != 0) {
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, num_rows_);
      int32 offset=div*256;
      
      cuda_find_row_max_id(dimGrid, dimBlock, data_ + offset,
                           max.data_, id->Data(), offset, d);
    }
    // now we have the indices!
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    // allocate index buffer
    id->Resize(num_rows_);
    id->Set(-1);
    // find maxima
    MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
    for(MatrixIndexT r = 0; r < num_rows; r++) {
      Real max = -1e21;
      int32 max_id = -1;
      const Real *row_data = Mat().RowData(r);
      for(MatrixIndexT c = 0; c < num_cols; c++) {
        if (max < row_data[c]) {
          max = row_data[c];
          max_id = c;
        }
      }
      id->Vec()[r] = max_id;
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::DiffXent(const CuStlVector<int32> &tgt,
                                  CuVector<Real> *log_post_tgt) {
  
  KALDI_ASSERT(tgt.Dim() == num_rows_);
  log_post_tgt->Resize(tgt.Dim());

#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(1, CU2DBLOCK*8);
    dim3 dimGrid(1, n_blocks(tgt.Dim(), CU2DBLOCK*8));
    cuda_diff_xent(dimGrid, dimBlock, tgt.Data(), data_,
                   log_post_tgt->data_, Dim());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    MatrixIndexT num_rows = num_rows_;
    for(int32 r = 0; r < num_rows; r++) {
      int32 col_tgt = tgt.Vec()[r];
      Real &value = Mat()(r, col_tgt);
      log_post_tgt->Vec()(r) = log(value);
      value -= 1.0;
    }
  }
}

// Cholesky method may be only called for symmetric matrices.
template<typename Real>
void CuMatrixBase<Real>::Cholesky() {
  KALDI_ASSERT(this->NumRows() == this->NumCols());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int TILE_SIZE = 16;
    int n_blocks = (num_rows_ + TILE_SIZE - 1) / TILE_SIZE;

    dim3 threads(TILE_SIZE,TILE_SIZE);
    dim3 logrid;
     
    for (int i = n_blocks; i > 2; i--) {
      cuda_factorize_diagonal_block(data_, n_blocks-i, Dim());
      cudaThreadSynchronize();

      cuda_strip_update(data_, n_blocks-i, i, Dim());
      cudaThreadSynchronize();
      
      cuda_diag_update(data_, n_blocks-i, i, Dim());
      cudaThreadSynchronize();
      
      cuda_lo_update(data_, n_blocks-i, n_blocks, i, Dim());
      cudaThreadSynchronize();      
    }
    
    if (n_blocks > 1) {
      cuda_factorize_diagonal_block(data_, n_blocks-2, Dim());
      cudaThreadSynchronize();
      
      cuda_strip_update(data_, n_blocks-2, 2, Dim());
      cudaThreadSynchronize();
      
      cuda_diag_update(data_, n_blocks-2, 2, Dim());
      cudaThreadSynchronize();
      
    }

    
    cuda_factorize_diagonal_block(data_, n_blocks-1, Dim());
    cudaThreadSynchronize();

    // set the upper diagonal equal to zero
    this->SetZeroUpperDiag();
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    
  } else
#endif
  {
    SpMatrix<Real> sp(this->NumRows(), kUndefined);
    sp.CopyFromMat(this->Mat(), kTakeLower);
    TpMatrix<Real> tp(this->NumRows());
    tp.Cholesky(sp);
    this->Mat().CopyFromTp(tp);
  }
}

#if HAVE_CUDA
template<typename Real> inline void cublas_trsm(int m, int n, Real alpha, const Real* A, int lda, Real* B, int ldb) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_trsm<float>(int m, int n, float alpha, const float* A, int lda, float* B, int ldb) {
  cublasStrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
template<> inline void cublas_trsm<double>(int m, int n, double alpha, const double* A, int lda, double* B, int ldb) {
  cublasDtrsm('l','u','n','n',m,n,alpha,A,lda,B,ldb);
}
#endif

template<typename Real>
void CuMatrixBase<Real>::InvertPSD() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
 
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(NumRows(),CU2DBLOCK));
    CuMatrix<Real> temp(num_rows_,num_rows_);
    int dim = num_rows_;
    Real value = 1.0;
    cuda_set_diag(dimGrid, dimBlock, temp.Data(), value, temp.Dim());
    Matrix<Real> A(dim,dim);
    temp.CopyToMat(&A);
    this->Cholesky();
    //CuSpMatrix<Real> L(*this, kTakeLower);
    Real alpha = 1.0;
    cublas_trsm(num_rows_,num_rows_,alpha,data_,stride_,temp.Data(),temp.Dim().stride);
    
    //CuSpMatrix<Real> L(temp, kTakeLower);
    //CuMatrix<Real> L1(dim,dim);
    //L1.CopyFromSp(L);
    //L1.SetZeroUpperDiag();
    Matrix<Real> L_test(dim,dim);
    temp.CopyToMat(&L_test);
    this->AddMatMat(1, temp, kTrans, temp, kNoTrans, 0);
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    this->Mat().Invert(); // This is inefficient as we don't make
    // use of the fact that we're symmetric, but anyway if we're not
    // using CUDA this function typically shouldn't be called, because its
    // only envisaged usage is to be call from the CUDA version of
    // CuSpMatrix::Invert().
  }
}


template<class Real>
bool CuMatrixBase<Real>::ApproxEqual(const CuMatrixBase<Real> &other,
                                     float tol) const {
  CuMatrix<Real> diff(*this);
  diff.AddMat(-1.0, other);
  return (diff.FrobeniusNorm() <= tol);
}

template<class Real>
Real TraceMatMat(const CuMatrixBase<Real> &A,
                 const CuMatrixBase<Real> &B,
                 MatrixTransposeType trans) {
  Real result = 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    Real* device_result;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(Real)));
    CU_SAFE_CALL(cudaMemset(device_result, 0, sizeof(Real)));
    if (trans == kNoTrans) {
      KALDI_ASSERT(A.NumRows() == B.NumCols() && A.NumCols() == B.NumRows());
      cuda_trace_mat_mat(A.Data(), B.RowData(0), A.Dim(), B.Stride(), device_result);
    } else {
      KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
      cuda_trace_mat_mat_trans(A.Data(), B.RowData(0), A.Dim(), B.Stride(), device_result);
    }
    CU_SAFE_CALL(cudaGetLastError());
    CU_SAFE_CALL(cudaMemcpy(&result, device_result, sizeof(Real), cudaMemcpyDeviceToHost));
    CU_SAFE_CALL(cudaFree(device_result));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    result = TraceMatMat(A.Mat(), B.Mat(), trans);
  }
  return result;
}

template
float TraceMatMat(const CuMatrixBase<float> &A,
                  const CuMatrixBase<float> &B,
                  MatrixTransposeType trans);
template
double TraceMatMat(const CuMatrixBase<double> &A,
                   const CuMatrixBase<double> &B,
                   MatrixTransposeType trans);


template<typename Real>
void CuMatrixBase<Real>::CopyRowsFromVec(const CuVectorBase<Real> &v) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    if (v.Dim() == num_rows_*num_cols_) {
      if (stride_ == num_cols_) {
        const Real* v_data = v.Data();
        cudaMemcpy(data_, v_data, sizeof(Real)*num_rows_*num_cols_, cudaMemcpyDeviceToDevice);
      } else {
        const Real *v_data = v.Data();
        for (MatrixIndexT r = 0; r < num_rows_; r++) {
          Real *row_data = RowData(r);
          cudaMemcpy(row_data, v_data, sizeof(Real)*num_cols_, cudaMemcpyDeviceToDevice);
          v_data += num_cols_;
        }
      }
    } else if (v.Dim() == num_cols_) {
      const Real *v_data = v.Data();
      for (MatrixIndexT r = 0; r < num_rows_; r++)
        cudaMemcpy(RowData(r), v_data, sizeof(Real)*num_cols_, cudaMemcpyDeviceToDevice);
    } else {
      KALDI_ERR << "Wrong sized arguments";
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyRowsFromVec(v.Vec());
  }
}

template<typename Real>
void CuMatrixBase<Real>::CopyRowsFromVec(const VectorBase<Real> &v) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    if (v.Dim() == num_rows_*num_cols_) {
      if (stride_ == num_cols_) {
        const Real* v_data = v.Data();
        cudaMemcpy(data_, v_data, sizeof(Real)*num_rows_*num_cols_, cudaMemcpyHostToDevice);
      } else {
        const Real *v_data = v.Data();
        for (MatrixIndexT r = 0; r < num_rows_; r++) {
          Real *row_data = RowData(r);
          cudaMemcpy(row_data, v_data, sizeof(Real)*num_cols_, cudaMemcpyHostToDevice);
          v_data += num_cols_;
        }
      }
    } else if (v.Dim() == num_cols_) {
      const Real *v_data = v.Data();
      for (MatrixIndexT r = 0; r < num_rows_; r++)
        cudaMemcpy(RowData(r), v_data, sizeof(Real)*num_cols_, cudaMemcpyHostToDevice);
    } else {
      KALDI_ERR << "Wrong sized arguments";
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyRowsFromVec(v);
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyColFromVec(const CuVectorBase<Real> &v,
                                        const MatrixIndexT col) {
  KALDI_ASSERT(v.Dim() == num_rows_ &&
               static_cast<UnsignedMatrixIndexT>(col) <
               static_cast<UnsignedMatrixIndexT>(num_cols_));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(NumRows(), CU2DBLOCK));
    cuda_copy_col_from_vec(dimGrid, dimBlock, data_, v.Data(), col, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().CopyColFromVec(v.Vec(), col);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyPow(Real power) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumRows(), CU2DBLOCK),
                 n_blocks(NumCols(), CU2DBLOCK));
    
    cuda_apply_pow(dimGrid, dimBlock, data_, power, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyPow(power);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyHeaviside() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumRows(), CU2DBLOCK),
                 n_blocks(NumCols(), CU2DBLOCK));
    
    cuda_apply_heaviside(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyHeaviside();
  }
}


template<typename Real>
void CuMatrixBase<Real>::ApplyExp() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_exp(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyExp();
  }
}


template<typename Real>
void CuMatrixBase<Real>::ApplyFloor(Real floor_val) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_floor(dimGrid, dimBlock, data_, floor_val, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyFloor(floor_val);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyCeiling(Real ceiling_val) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));

    cuda_apply_ceiling(dimGrid, dimBlock, data_, ceiling_val, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().ApplyCeiling(ceiling_val);
  }
}


template<typename Real>
void VectorBase<Real>::CopyRowsFromMat(const CuMatrixBase<Real> &mat) {
  KALDI_ASSERT(dim_ == mat.NumCols() * mat.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    if (mat.Stride() == mat.NumCols()) {
      cudaMemcpy(data_, mat.Data(), sizeof(Real)*dim_, cudaMemcpyDeviceToHost);
    } else {
      Real* vec_data = data_;
      for (MatrixIndexT r = 0; r < mat.NumRows(); r++) {
        cudaMemcpy(vec_data, mat.RowData(r), sizeof(Real) * mat.NumCols(),
                   cudaMemcpyDeviceToHost);
        vec_data += mat.NumCols();
      }
    }
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyRowsFromMat", tim.Elapsed());
  } else
#endif
  {
    CopyRowsFromMat(mat.Mat());
  }
}

// Instantiate the template above.
template
void VectorBase<float>::CopyRowsFromMat(const CuMatrixBase<float> &mat);
template
void VectorBase<double>::CopyRowsFromMat(const CuMatrixBase<double> &mat);

template<typename Real>
void CuMatrixBase<Real>::PermuteColumns(const CuMatrixBase<Real> &src,
                                        const std::vector<int32> &reorder,
                                        bool forward) {
  KALDI_ASSERT(SameDimAndStride(*this, src));  
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<int32>(reorder.size()) == NumCols());
    KALDI_ASSERT(SameDim(*this, src));
    CuStlVector<int32> cuda_reorder;
    if (forward) {
      cuda_reorder.CopyFromVec(reorder);
    } else {
      int32 num_cols = NumCols();
      std::vector<int32> reorder_backward(num_cols);
      for (int32 i = 0; i < num_cols; i++) {
        KALDI_ASSERT(reorder[i] >= 0 && reorder[i] < num_cols);
        reorder_backward[reorder[i]] = i;
      }
      cuda_reorder.CopyFromVec(reorder_backward);
    }
    
    Timer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK), n_blocks(NumRows(), CU2DBLOCK));
    cuda_permute_columns(dimGrid, dimBlock, data_, src.Data(), cuda_reorder.Data(), Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().PermuteColumns(src.Mat(), reorder, forward);
  }
}


template<typename Real>
Real CuMatrixBase<Real>::Sum() const {
  CuVector<Real> row_sum(NumCols());
  row_sum.AddRowSumMat(1.0, *this, 0.0);
  return row_sum.Sum();
}

template<typename Real>
Real CuMatrixBase<Real>::Trace(bool check_square) const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    if (check_square) KALDI_ASSERT(this->num_rows_ == this->num_cols_);
    MatrixIndexT dim = std::min(this->num_rows_, this->num_cols_);
    CuVector<Real> tmp(1, kUndefined); // for result.
    int dimBlock(CU1DBLOCK);
    int dimGrid = 1;// only 1 block here. we have loops in each thread  //(n_blocks(dim_, CU1DBLOCK));
    cuda_vec_sum(dimGrid, dimBlock, data_, tmp.Data(), dim, Stride() + 1);
    CuDevice::Instantiate().AccuProfile("CuVectorBase::Sum", tim.Elapsed());    
    return tmp(0);
  } else 
#endif
  {
    return Mat().Trace(check_square);
  }
}




template<typename Real>
void CuMatrixBase<Real>::SetRandn() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuRand<Real> tmp;
    tmp.RandGaussian(this);
  } else 
#endif
  {
    Mat().SetRandn();
  }
}

template<class Real>
void Matrix<Real>::Swap(CuMatrix<Real> *mat) { mat->Swap(this); }
// instantiate the template above.
template void Matrix<float>::Swap(CuMatrix<float> *mat);
template void Matrix<double>::Swap(CuMatrix<double> *mat);

/// Copy constructor from another type.
template<typename Real>
template<typename OtherReal>
CuMatrix<Real>::CuMatrix(const CuMatrixBase<OtherReal> & M,
                         MatrixTransposeType trans) : CuMatrixBase<Real>() {

  if (trans == kNoTrans) {
    Resize(M.NumRows(), M.NumCols());
    this->CopyFromMat(M);
  } else {
    Resize(M.NumCols(), M.NumRows());
    this->CopyFromMat(M, kTrans);
  }

}

// Instantiate this constructor for float->double and double->float.
template
CuMatrix<float>::CuMatrix(const CuMatrixBase<double> & M,
                          MatrixTransposeType trans);
template
CuMatrix<double>::CuMatrix(const CuMatrixBase<float> & M,
                           MatrixTransposeType trans);

/*
template<typename Real>
CuMatrix<Real>::DeriveLastLayerComponent(int32 i, int32 label,
                                         Real weight, Real this_prob) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    cuda_derive_last_layer_component(i, label, weight, this_prob);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
  {

  }
}
*/

/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrixBase<Real> &mat) {
  Matrix<Real> temp(mat.NumRows(), mat.NumCols());
  mat.CopyToMat(&temp);
  out << temp;
  return out;
}

template<typename Real>
void CuMatrix<Real>::Transpose() {
  CuMatrix<Real> tmp(*this, kTrans);
  *this = tmp;
}

// instantiate the template
template
std::ostream &operator << (std::ostream &out, const CuMatrixBase<float> &mat);
template 
std::ostream &operator << (std::ostream &out, const CuMatrixBase<double> &mat);


// Instantiate classes CuMatrix and CuMatrixBase for float and double.
template class CuMatrix<float>;
template class CuMatrix<double>;
template class CuMatrixBase<float>;
template class CuMatrixBase<double>;

} // namespace kaldi

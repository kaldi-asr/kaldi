// cudamatrix/cu-matrix.cc

// Copyright 2009-2012  Karel Vesely, Lucas Ondel
//                2013  Ehsan Variani
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  Hainan Xu
//                2013  Xiaohui Zhang
//           2013-2015  Guoguo Chen
//           2016-2017  Shiyin Kang
//                2017  Hossein Hadian

// See ../../COPYING for clarification regarding multiple authors
//
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
#include <cublas_v2.h>
#endif

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-block-matrix.h"
#include "cudamatrix/cu-sparse-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {

template<typename Real>
void CuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols,
                            MatrixResizeType resize_type,
                            MatrixStrideType stride_type) {
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
    CuTimer tim;
    MatrixIndexT row_bytes = cols * sizeof(Real);
    size_t pitch;
    if (stride_type == kDefaultStride) {
      this->data_ = static_cast<Real*>(CuDevice::Instantiate().MallocPitch(
          row_bytes, rows, &pitch));
      this->num_rows_ = rows;
      this->num_cols_ = cols;
      this->stride_ = pitch / sizeof(Real);
    } else {  // kStrideEqualNumCols
      size_t bytes = rows * cols * sizeof(Real);
      this->data_ = static_cast<Real*>(CuDevice::Instantiate().Malloc(bytes));
      this->num_rows_ = rows;
      this->num_cols_ = cols;
      this->stride_ = cols;
    }
    if (resize_type == kSetZero) this->SetZero();
    CuDevice::Instantiate().AccuProfile("CuMatrix::Resize", tim);
  } else
#endif
  { // Let the initializer of Matrix<Real> handle the allocation,
    // and then just do Swap which will switch the pointers.
    // This wastes a few instructions but is simple to code.
    Matrix<Real> mat(rows, cols, resize_type, stride_type);
    this->Swap(&mat);
  }
}

template<typename Real>
void CuMatrix<Real>::Destroy() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->data_ != NULL) {
      CuTimer tim;
      CuDevice::Instantiate().Free(this->data_);
      CuDevice::Instantiate().AccuProfile(__func__, tim);
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
void CuMatrixBase<Real>::CopyFromBlock(const CuBlockMatrix<Real> &B,
                                       MatrixTransposeType trans) {
  this->SetZero();
  if (trans == kNoTrans) {
    KALDI_ASSERT(NumRows() == B.NumRows() && NumCols() == B.NumCols());
    int32 row_offset = 0, col_offset = 0;
    for (int32 b = 0; b < B.NumBlocks(); b++) {
      const CuMatrixBase<Real> &block = B.Block(b);
      int32 num_rows = block.NumRows(), num_cols = block.NumCols();
      CuSubMatrix<Real> this_block(*this, row_offset, num_rows,
                                   col_offset, num_cols);
      this_block.CopyFromMat(block);
      row_offset += num_rows;
      col_offset += num_cols;
    }
    KALDI_ASSERT(row_offset == NumRows() && col_offset == NumCols());
  } else {
    KALDI_ASSERT(NumRows() == B.NumCols() && NumCols() == B.NumRows());
    int32 row_offset = 0, col_offset = 0;
    for (int32 b = 0; b < B.NumBlocks(); b++) {
      const CuMatrixBase<Real> &block = B.Block(b);
      int32 num_rows = block.NumCols(), num_cols = block.NumRows();
      CuSubMatrix<Real> this_block(*this, row_offset, num_rows,
                                   col_offset, num_cols);
      this_block.CopyFromMat(block, kTrans);
      row_offset += num_rows;
      col_offset += num_cols;
    }
    KALDI_ASSERT(row_offset == NumRows() && col_offset == NumCols());
  }
}


template <typename Real>
 CuMatrix<Real>::CuMatrix(const CuBlockMatrix<Real> &B,
                          MatrixTransposeType trans): CuMatrixBase<Real>() {
  if (trans == kNoTrans) {
    Resize(B.NumRows(), B.NumCols(), kUndefined);
    this->CopyFromBlock(B);
  } else {
    Resize(B.NumCols(), B.NumRows(), kUndefined);
    this->CopyFromBlock(B, kTrans);
  }
}

template<class Real>
template<class OtherReal>
void CuMatrixBase<Real>::CopyFromMat(const CuMatrixBase<OtherReal> &M,
                                     MatrixTransposeType trans) {
  if (sizeof(Real) == sizeof(OtherReal) &&
      static_cast<const void*>(M.Data()) ==
      static_cast<const void*>(this->Data())) {
    if (M.Data() == NULL)
      return;
    // CopyFromMat called on same data.  Nothing to do (except sanity checks)
    KALDI_ASSERT(trans == kNoTrans && M.NumRows() == NumRows() &&
                 M.NumCols() == NumCols() && M.Stride() == Stride());
    return;
  }
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      KALDI_ASSERT(M.NumRows() == num_rows_ && M.NumCols() == num_cols_);
    } else {
      KALDI_ASSERT(M.NumCols() == num_rows_ && M.NumRows() == num_cols_);
    }
    if (M.num_rows_ == 0) return; // Nothing to do.
    CuTimer tim;
    if (sizeof(Real) == sizeof(OtherReal) && trans == kNoTrans ) {
      MatrixIndexT dst_pitch = stride_ * sizeof(Real);
      MatrixIndexT src_pitch = M.Stride() * sizeof(Real);
      MatrixIndexT width = M.NumCols() * sizeof(Real);
      CU_SAFE_CALL(
        cudaMemcpy2DAsync(data_, dst_pitch, M.data_, src_pitch,
                          width, M.num_rows_, cudaMemcpyDeviceToDevice,
                          cudaStreamPerThread));
    } else {
      if (trans == kNoTrans) {
        dim3 dimGrid, dimBlock;
        GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                              &dimGrid, &dimBlock);
        cuda_copy_from_mat(dimGrid, dimBlock, data_, M.data_, Dim(), M.Dim());
      } else {
        // 2D thread block with warps (blockDim.x) along the row-dim of input M.
        // Each (8x32) thread block will transpose (32x32) data
        const int32 warpSize = 32;
        dim3 dimBlock(warpSize, CU1DBLOCK / warpSize);
        dim3 dimGrid(n_blocks(M.NumCols(), warpSize),
            n_blocks(M.NumRows(), warpSize));
        cuda_copy_from_mat_trans(dimGrid, dimBlock, data_, M.data_, Dim(),
            M.Dim());
      }
      CU_SAFE_CALL(cudaGetLastError());
    }
    CuDevice::Instantiate().AccuProfile("CuMatrixBase::CopyFromMat(from other CuMatrixBase)", tim);
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), trans);
  }
}

// Instantiate the template above.
template
void CuMatrixBase<float>::CopyFromMat<float>(const CuMatrixBase<float> &M,
                                             MatrixTransposeType trans);
template
void CuMatrixBase<float>::CopyFromMat<double>(const CuMatrixBase<double> &M,
                                              MatrixTransposeType trans);
template
void CuMatrixBase<double>::CopyFromMat<float>(const CuMatrixBase<float> &M,
                                              MatrixTransposeType trans);
template
void CuMatrixBase<double>::CopyFromMat<double>(const CuMatrixBase<double> &M,
                                               MatrixTransposeType trans);


template<typename Real>
template<typename OtherReal>
void CuMatrixBase<Real>::CopyFromTp(const CuTpMatrix<OtherReal> &M,
                                    MatrixTransposeType trans) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
  if (num_rows_ == 0)
    return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_rows_, CU2DBLOCK),
                 n_blocks(num_rows_, CU2DBLOCK));
    if (trans == kNoTrans) {
      cuda_copy_from_tp(dimGrid, dimBlock, data_, M.Data(), Dim());
    } else {
      cuda_copy_from_tp_trans(dimGrid, dimBlock, data_, M.Data(), Dim());
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyFromTp(M.Mat(), trans);
  }
}
// instantiate the template above.
template void CuMatrixBase<float>::CopyFromTp(const CuTpMatrix<float> &M,
                                              MatrixTransposeType trans);
template void CuMatrixBase<float>::CopyFromTp(const CuTpMatrix<double> &M,
                                              MatrixTransposeType trans);
template void CuMatrixBase<double>::CopyFromTp(const CuTpMatrix<float> &M,
                                              MatrixTransposeType trans);
template void CuMatrixBase<double>::CopyFromTp(const CuTpMatrix<double> &M,
                                              MatrixTransposeType trans);

template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const MatrixBase<Real> &src,
                                     MatrixTransposeType trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);
      CuTimer tim;

      MatrixIndexT dst_pitch = stride_*sizeof(Real);
      MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
      MatrixIndexT width = src.NumCols()*sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch,
                                width, src.NumRows(), cudaMemcpyHostToDevice));

      CuDevice::Instantiate().AccuProfile("CuMatrixBase::CopyFromMat(from CPU)", tim);
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

// instantiate the template above.
template
void CuMatrixBase<float>::CopyFromMat(const MatrixBase<double> &src,
                                      MatrixTransposeType trans);
template
void CuMatrixBase<double>::CopyFromMat(const MatrixBase<float> &src,
                                       MatrixTransposeType trans);


template<typename Real>
void CuMatrixBase<Real>::CopyFromSp(const CuSpMatrix<Real> &M) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
  if (num_rows_ == 0)
    return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumRows(), CU2DBLOCK),
                 n_blocks(NumRows(), CU2DBLOCK));
    cuda_copy_from_sp(dimGrid, dimBlock, M.Data(), data_, Dim());
    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromSp", tim);
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
      if (num_rows_ == 0) return;
      CuTimer tim;

      MatrixIndexT src_pitch = stride_*sizeof(Real);
      MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
      MatrixIndexT width = NumCols()*sizeof(Real);
      CU_SAFE_CALL(cudaMemcpy2D(dst->Data(), dst_pitch, this->data_, src_pitch,
                                width, this->num_rows_, cudaMemcpyDeviceToHost));

      CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H", tim);
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
void CuMatrixBase<Real>::Write(std::ostream &os, bool binary) const {
  Matrix<Real> temp(this->num_rows_, this->num_cols_, kUndefined);
  this->CopyToMat(&temp);
  temp.Write(os, binary);
}

template<typename Real>
void CuMatrixBase<Real>::SetZero() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(cudaMemset2D(data_, stride_ * sizeof(Real), 0,
                              num_cols_ * sizeof(Real), num_rows_ ));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero", tim);
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
    if (num_rows_ == 0) return;
    CuTimer tim;

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_set_const(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().Set(value);
  }
}

// set zero the elements above the diagonal.
template<typename Real>
void CuMatrixBase<Real>::SetZeroAboveDiag() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    CuTimer tim;

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_set_zero_above_diag(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    MatrixBase<Real> &mat = Mat();
    int32 num_rows = mat.NumRows(), num_cols = mat.NumCols();
    for (int32 r = 0; r + 1 < num_rows; r++) {
      SubVector<Real> vec(mat, r),
          vec_part(vec, r + 1, num_cols - (r + 1));
      vec_part.SetZero();
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::Add(Real value) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    CuTimer tim;

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_add(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().Add(value);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddToDiag(Real value) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    CuTimer tim;
    // We'll create a fake matrix with "num_diag" rows, one
    // columnn, and a stride of "this_stride".  The y-value of
    // the grid/blocks corresponds to the row, in this kernel.
    MatrixIndexT num_diag = std::min(num_rows_, num_cols_),
        this_stride = stride_ + 1;
    dim3 dimBlock(1, CU1DBLOCK);
    dim3 dimGrid(1, n_blocks(num_diag, CU1DBLOCK));
    ::MatrixDim d = { num_diag, 1, this_stride };
    cuda_add(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().AddToDiag(value);
  }
}

template<typename Real>
bool CuMatrixBase<Real>::IsUnit(Real tol) const {
  // want to return:
  //FrobeniusNorm(*this - I) <= tol * NumRows(), i.e.:
  //sqrt (trace((*this - I)(*this-I)) <= tol * NumRows()
  //    trace((*this - I)(*this - I)) <= tol * NumRows()
  // trace(*this * *this) + trace(I) - 2 * trace(*this) <= tol * NumRows()
  // trace(*this * *this) + dim - 2*this.Trace() <= tol * NumRows()
  KALDI_ASSERT(this->NumRows() == this->NumCols());
  return (TraceMatMat(*this, *this, kTrans) + this->NumRows() - 2.0 * this->Trace() <=
          tol * this->NumRows());
}



template<typename Real>
void CuMatrixBase<Real>::Scale(Real value) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    CuTimer tim;

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_scale(dimGrid, dimBlock, data_, value, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    if (num_rows_ == 0) return;
    CuTimer tim;

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_apply_log(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    CuTimer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_mul_elements(dimGrid, dimBlock, data_, A.data_, Dim(), A.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().MulElements(A.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::DivElements(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_div_elements(dimGrid, dimBlock, data_, A.data_, Dim(), A.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().DivElements(A.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::Max(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_max(dimGrid, dimBlock, data_, A.data_, Dim(), A.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().Max(A.Mat());
  }
}


template<typename Real>
void CuMatrixBase<Real>::Min(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_min(dimGrid, dimBlock, data_, A.data_, Dim(), A.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().Min(A.Mat());
  }
}


template<typename Real>
void CuMatrixBase<Real>::MulColsVec(const CuVectorBase<Real> &scale) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    KALDI_ASSERT(scale.Dim() == NumCols());


    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_mul_cols_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    CuTimer tim;

    KALDI_ASSERT(scale.Dim() == NumRows());

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_mul_rows_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().MulRowsVec(scale.Vec());
  }
}

template<typename Real>
void CuMatrixBase<Real>::MulRowsGroupMat(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(src.NumCols() > 0);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    int group_size = this->NumCols() / src.NumCols();

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_mul_rows_group_mat(dimGrid, dimBlock, this->data_, src.data_,
                            this->Dim(), src.Stride(), group_size);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().MulRowsGroupMat(src.Mat());
  }
}


template<typename Real>
void CuMatrixBase<Real>::DiffGroupPnorm(const CuMatrixBase<Real> &in_value,
                                        const CuMatrixBase<Real> &out_value,
                                        const CuMatrixBase<Real> &out_deriv,
                                        Real power) {
  KALDI_ASSERT(out_value.NumCols() > 0);
  KALDI_ASSERT(out_value.NumCols() == out_deriv.NumCols());
  int group_size = this->NumCols() / out_value.NumCols();
  KALDI_ASSERT(this->NumCols() == out_value.NumCols() * group_size);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    const int kWarpSize = 32;
    dim3 dimBlock(kWarpSize, CU1DBLOCK / kWarpSize);
    dim3 dimGrid(n_blocks(NumCols(), dimBlock.x),
                 n_blocks(NumRows(), dimBlock.y));
    if (dimGrid.x * dimGrid.y > 1024) {
      dimGrid.y = std::max(1024 / dimGrid.x, unsigned(1));
    }
    cuda_diff_group_pnorm(dimGrid, dimBlock, this->data_, in_value.Data(),
                          out_value.Data(), out_deriv.Data(), Dim(),
                          in_value.Stride(), out_value.Stride(),
                          out_deriv.Stride(), group_size, power);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().GroupPnormDeriv(in_value.Mat(), out_value.Mat(), power);
    MulRowsGroupMat(out_deriv);
  }
}

template<typename Real>
void CuMatrixBase<Real>::GroupMaxDeriv(const CuMatrixBase<Real> &src1,
                                       const CuMatrixBase<Real> &src2) {
  KALDI_ASSERT(src2.NumCols() > 0);
  int group_size = this->NumCols() / src2.NumCols();
  KALDI_ASSERT(this->NumCols() == src2.NumCols() * group_size);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK),
                 n_blocks(NumRows(), CU2DBLOCK));
    cuda_calc_group_max_deriv(dimGrid, dimBlock, this->data_, src1.Data(),
                              src2.Data(), Dim(), src1.Stride(), src2.Stride(),
                              group_size);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().GroupMaxDeriv(src1.Mat(), src2.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::DivRowsVec(const CuVectorBase<Real> &div) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    KALDI_ASSERT(div.Dim() == NumRows());

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    // For large matrix we do more work per thread by limiting the
    // the grid size to reduce the block launching overhead.
    if (dimGrid.x * dimGrid.y > 1024) {
      dimGrid.x = 1024 / dimGrid.y;
      if (dimGrid.x == 0) {
        dimGrid.x = 1;
      }
    }
    cuda_div_rows_vec(dimGrid, dimBlock, data_, div.data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Vector<Real> temp(div.Vec()); // will copy.
    temp.InvertElements();
    Mat().MulRowsVec(temp);
  }
}


template<typename Real>
void CuMatrixBase<Real>::InvertElements() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_invert_elements(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().InvertElements();
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddMat(Real alpha, const CuMatrixBase<Real>& A,
                                MatrixTransposeType transA) {

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (transA == kNoTrans) {
      KALDI_ASSERT(A.NumRows() == num_rows_ && A.NumCols() == num_cols_);
    } else {
      KALDI_ASSERT(A.NumCols() == num_rows_ && A.NumRows() == num_cols_);
    }
    if (num_rows_ == 0) return;
    CuTimer tim;
    // This block dimension seems to work better than the
    // one from GetBlockSizesForSimpleMatrixOperation().
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK),
                 n_blocks(NumRows(), CU2DBLOCK));
    cuda_add_mat(dimGrid, dimBlock, alpha, A.data_,
                 data_, Dim(), A.Stride(),
                 (transA == kTrans ? 1 : 0));
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddMat(alpha, A.Mat(), transA);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddSmat(Real alpha, const CuSparseMatrix<Real> &A,
                                 MatrixTransposeType trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      KALDI_ASSERT(NumRows() == A.NumRows());
      KALDI_ASSERT(NumCols() == A.NumCols());
    } else {
      KALDI_ASSERT(NumRows() == A.NumCols());
      KALDI_ASSERT(NumCols() == A.NumRows());
    }

    CuTimer tim;

    // We use warpSize threads per row to access only the nonzero elements.
    // Every CU1DBLOCK/warpSize rows share one thread block.
    // 1D grid to cover all rows of A.
    const int warpSize = 32;
    dim3 dimBlock(warpSize, CU1DBLOCK / warpSize);
    dim3 dimGrid(n_blocks(A.NumRows(), dimBlock.y));

    if (trans == kNoTrans) {
      cuda_add_smat(dimGrid, dimBlock, Data(), Dim(), alpha, A.CsrRowPtr(),
                    A.CsrColIdx(), A.CsrVal());
    } else {
      cuda_add_smat_trans(dimGrid, dimBlock, Data(), Dim(), alpha,
      A.CsrRowPtr(), A.CsrColIdx(), A.CsrVal());
    }

    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddSmat(alpha, A.Smat(), trans);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddSmatMat(Real alpha, const CuSparseMatrix<Real> &A,
                                    MatrixTransposeType transA,
                                    const CuMatrixBase<Real> &B, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (transA == kNoTrans) {
      KALDI_ASSERT(NumRows() == A.NumRows());
      KALDI_ASSERT(NumCols() == B.NumCols());
      KALDI_ASSERT(A.NumCols() == B.NumRows());
    } else {
      KALDI_ASSERT(NumRows() == A.NumCols());
      KALDI_ASSERT(NumCols() == B.NumCols());
      KALDI_ASSERT(A.NumRows() == B.NumRows());
    }

    CuTimer tim;

    // We have op(A) and BT in col-major (B in row-major).
    // We first compute C in col-major (CT in row-major)
    // with C = op(A) * BT^T by cusparse_csrmm2,
    // then transpose CT to get C in row-major
    CuMatrix<Real> CT(*this, kTrans);

    cusparseMatDescr_t descr;
    CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&descr));
    if (transA == kTrans) {
      // Note: only op(A)=A is supported if op(B)=B^T according to cusparse doc
      // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmm2
      CuSparseMatrix<Real> AT(A, kTrans);
      CU_SAFE_CALL(
          cusparse_csrmm2(GetCusparseHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_TRANSPOSE, AT.NumRows(),
                          CT.NumRows(), AT.NumCols(), AT.NumElements(), &alpha,
                          descr, AT.CsrVal(), AT.CsrRowPtr(), AT.CsrColIdx(),
                          B.Data(), B.Stride(), &beta, CT.Data(), CT.Stride()));
    } else {
      CU_SAFE_CALL(
          cusparse_csrmm2(GetCusparseHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_TRANSPOSE, A.NumRows(),
                          CT.NumRows(), A.NumCols(), A.NumElements(), &alpha,
                          descr, A.CsrVal(), A.CsrRowPtr(), A.CsrColIdx(),
                          B.Data(), B.Stride(), &beta, CT.Data(), CT.Stride()));
    }
    CUSPARSE_SAFE_CALL(cusparseDestroyMatDescr(descr));

    this->CopyFromMat(CT, kTrans);

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddSmatMat(alpha, A.Smat(), transA, B.Mat(), beta);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddMatSmat(Real alpha, const CuMatrixBase<Real> &A,
                                    const CuSparseMatrix<Real> &B,
                                    MatrixTransposeType transB, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (transB == kNoTrans) {
      KALDI_ASSERT(NumRows() == A.NumRows());
      KALDI_ASSERT(NumCols() == B.NumCols());
      KALDI_ASSERT(A.NumCols() == B.NumRows());
    } else {
      KALDI_ASSERT(NumRows() == A.NumRows());
      KALDI_ASSERT(NumCols() == B.NumRows());
      KALDI_ASSERT(A.NumCols() == B.NumCols());
    }

    CuTimer tim;

    cusparseMatDescr_t descr;
    CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&descr));
    CU_SAFE_CALL(
        cusparse_csrmm(
            GetCusparseHandle(),
            transB == kNoTrans ?
                CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
            B.NumRows(), NumRows(), B.NumCols(), B.NumElements(), &alpha, descr,
            B.CsrVal(), B.CsrRowPtr(), B.CsrColIdx(), A.Data(), A.Stride(),
            &beta, Data(), Stride()));
    CUSPARSE_SAFE_CALL(cusparseDestroyMatDescr(descr));

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddMatSmat(alpha, A.Mat(), B.Smat(), transB, beta);
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddMatBlocks(Real alpha, const CuMatrixBase<Real> &A,
                                      MatrixTransposeType transA) {
  if (num_rows_ == 0 || num_cols_ == 0) return;

  if (A.NumRows() >= (transA == kNoTrans ? num_rows_ : num_cols_) &&
      A.NumCols() >= (transA == kNoTrans ? num_cols_ : num_rows_)) {
    // This is the "summing", not broadcasting, version of AddMatBlocks.
    // It supports both regular and transposed operation.
    int32 num_row_blocks, num_col_blocks;
    if (transA == kNoTrans) {
      KALDI_ASSERT(A.NumRows() % num_rows_ == 0 && A.NumCols() % num_cols_ == 0);
      num_row_blocks = A.Mat().NumRows() / num_rows_;
      num_col_blocks = A.Mat().NumCols() / num_cols_;
    } else {
      KALDI_ASSERT(A.NumRows() % num_cols_ == 0 && A.NumCols() % num_rows_ == 0);
      num_row_blocks = A.Mat().NumRows() / num_cols_;
      num_col_blocks = A.Mat().NumCols() / num_rows_;
    }
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().Enabled()) {
      CuTimer tim;
      dim3 dimGrid, dimBlock;
      GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                            &dimGrid, &dimBlock);
      cuda_add_mat_blocks(dimGrid, dimBlock, alpha, A.data_, num_row_blocks,
                          num_col_blocks, data_, Dim(), A.Stride(),
                          (transA == kTrans ? 1 : 0));
      CU_SAFE_CALL(cudaGetLastError());

      CuDevice::Instantiate().AccuProfile(__func__, tim);
    } else
#endif
    {
      int32 nr, nc;
      if (transA == kNoTrans) {
        nr = num_rows_;
        nc = num_cols_;
      } else {
        nr = num_cols_;
        nc = num_rows_;
      }
      for (int32 i = 0; i < num_row_blocks; i++) {
        for (int32 j = 0; j < num_col_blocks; j++) {
          Mat().AddMat(alpha, SubMatrix<Real>(A.Mat(), i * nr, nr, j * nc, nc),
                       transA);
        }
      }
    }
  } else {
    // This is the "broadcasting" version of AddMatBlocks, where
    // *this is larger than src.
    if (transA != kNoTrans)
      KALDI_ERR << "Transposed operation not supported currently.";
    if (!(num_rows_ % A.NumRows() == 0 && num_cols_ % A.NumCols() == 0))
      KALDI_ERR << "Invalid sizes of arguments";
#if HAVE_CUDA == 1
    if (CuDevice::Instantiate().Enabled()) {
      CuTimer tim;
      dim3 dimGrid, dimBlock;
      GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                            &dimGrid, &dimBlock);
      cuda_add_mat_repeated(dimGrid, dimBlock, alpha,
                            A.data_, A.Dim(), data_, Dim());
      CU_SAFE_CALL(cudaGetLastError());
      CuDevice::Instantiate().AccuProfile(__func__, tim);
    } else
#endif
    {
      const MatrixBase<Real> &src_mat = A.Mat(),
          &this_mat = this->Mat();
      for (int32 row_offset = 0; row_offset < NumRows();
           row_offset += src_mat.NumRows()) {
        for (int32 col_offset = 0; col_offset < NumCols();
             col_offset += src_mat.NumCols()) {
          SubMatrix<Real> this_part(this_mat,
                                    row_offset, src_mat.NumRows(),
                                    col_offset, src_mat.NumCols());
          this_part.AddMat(alpha, src_mat);
        }
      }
    }
  }
}

/// dst = a * b / c (by element; when c = 0, dst = a)
/// dst can be an alias of a, b or c safely and get expected result.
template<typename Real>
void CuMatrixBase<Real>::SetMatMatDivMat(const CuMatrixBase<Real> &A,
                    const CuMatrixBase<Real> &B, const CuMatrixBase<Real> &C) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    KALDI_ASSERT(num_rows_ == A.num_rows_ && num_cols_ == A.num_cols_);
    KALDI_ASSERT(num_rows_ == B.num_rows_ && num_cols_ == B.num_cols_);
    KALDI_ASSERT(num_rows_ == C.num_rows_ && num_cols_ == C.num_cols_);
    if (num_rows_ == 0) return;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_set_mat_mat_div_mat(dimGrid, dimBlock, A.data_, B.data_, C.data_,
                             data_, Dim(), A.Stride(), B.Stride(), C.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().SetMatMatDivMat(A.Mat(), B.Mat(), C.Mat());
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
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_vec_to_cols(dimGrid, dimBlock, alpha, col.data_, beta,
                         data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_vec_to_rows(dimGrid, dimBlock, alpha, row.data_, beta, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    if (beta != 1.0) Mat().Scale(beta);
    Mat().AddVecToRows(alpha, row.Vec());
  }
}



/*
 * Method wrapping the CUBLAS function GEMM
 */
template<typename Real>
void CuMatrixBase<Real>::AddMatMat(
    Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
    const CuMatrixBase<Real> &B, MatrixTransposeType transB, Real beta) {


    // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
    // keep trans..., just swap A&B matrices: A->B B->A
    MatrixIndexT m = ((transB==kTrans)? B.NumRows() : B.NumCols());
    MatrixIndexT n = ((transA==kTrans)? A.NumCols() : A.NumRows());
    MatrixIndexT k = ((transB==kTrans)? B.NumCols() : B.NumRows());
    MatrixIndexT k1 = ((transA==kTrans)? A.NumRows() : A.NumCols());

    KALDI_ASSERT(m == NumCols());
    KALDI_ASSERT(n == NumRows());
    KALDI_ASSERT(k == k1);

    if (m == 0) return;


#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CUBLAS_SAFE_CALL(cublas_gemm(GetCublasHandle(),
                             (transB==kTrans? CUBLAS_OP_T:CUBLAS_OP_N),
                             (transA==kTrans? CUBLAS_OP_T:CUBLAS_OP_N),
                             m, n, k, alpha, B.data_, B.Stride(),
                             A.data_, A.Stride(), beta, data_, Stride()));

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddMatMat(alpha, A.Mat(), transA, B.Mat(), transB, beta);
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddVecVec(
    Real alpha, const CuVectorBase<Real> &x, const CuVectorBase<Real> &y) {

    MatrixIndexT m = y.Dim();
    MatrixIndexT n = x.Dim();
    KALDI_ASSERT(m == NumCols());
    KALDI_ASSERT(n == NumRows());

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CUBLAS_SAFE_CALL(cublas_ger(GetCublasHandle(), m, n, alpha,
                     y.Data(), 1, x.Data(), 1, data_, Stride()));

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddVecVec(alpha, x.Vec(), y.Vec());
  }
}


template<typename Real>
void CuMatrixBase<Real>::SymAddMat2(
    Real alpha, const CuMatrixBase<Real> &A, MatrixTransposeType transA,
    Real beta) {
  KALDI_ASSERT(num_rows_ == num_cols_ &&
               ((transA == kNoTrans && A.num_rows_ == num_rows_) ||
                (transA == kTrans && A.num_cols_ == num_cols_)));
  if (num_rows_ == 0) return;
  KALDI_ASSERT(A.data_ != data_);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    cublasOperation_t trans = (transA == kTrans ? CUBLAS_OP_N : CUBLAS_OP_T);
    MatrixIndexT A_other_dim = (transA == kNoTrans ? A.num_cols_ : A.num_rows_);
    CUBLAS_SAFE_CALL(cublas_syrk(GetCublasHandle(), CUBLAS_FILL_MODE_UPPER,
                                 trans, num_rows_, A_other_dim,
                                 alpha, A.Data(), A.Stride(),
                                 beta, this->data_, this->stride_));

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().SymAddMat2(alpha, A.Mat(), transA, beta);
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddDiagVecMat(
    const Real alpha, const CuVectorBase<Real> &v,
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

    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK),
                 n_blocks(num_rows_, CU2DBLOCK));
    MatrixIndexT M_row_stride = M.Stride(), M_col_stride = 1;
    if (transM == kTrans)
      std::swap(M_row_stride, M_col_stride);
    cuda_add_diag_vec_mat(dimGrid, dimBlock, alpha, data_, Dim(),
                          v.Data(), M.Data(), M_row_stride, M_col_stride, beta);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddDiagVecMat(alpha, v.Vec(), M.Mat(), transM, beta);
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddMatDiagVec(
    const Real alpha,
    const CuMatrixBase<Real> &M, MatrixTransposeType transM,
    CuVectorBase<Real> &v,
    Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (transM == kNoTrans) {
      KALDI_ASSERT(SameDim(*this, M));
    } else {
      KALDI_ASSERT(M.NumRows() == NumCols() && M.NumCols() == NumRows());
    }
    KALDI_ASSERT(v.Dim() == this->NumCols());

    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    MatrixIndexT M_row_stride = M.Stride(), M_col_stride = 1;
    if (transM == kTrans) std::swap(M_row_stride, M_col_stride);
    cuda_add_mat_diag_vec(dimGrid, dimBlock, alpha, data_, Dim(),
                          M.Data(), M_row_stride, M_col_stride, v.Data(),  beta);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddMatDiagVec(alpha, M.Mat(), transM, v.Vec(), beta);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddMatMatElements(Real alpha,
    const CuMatrixBase<Real> &A, const CuMatrixBase<Real> &B, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(SameDim(*this, A) && SameDim(A, B));
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_mat_mat_elements(dimGrid, dimBlock, this->data_, A.Data(),
                              B.Data(), Dim(), A.Stride(), B.Stride(), alpha, beta);
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddMatMatElements(alpha, A.Mat(), B.Mat(), beta);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ParametricRelu(
     const CuMatrixBase<Real> &src,
     const CuVectorBase<Real> &alpha,
     const CuVectorBase<Real> &beta) {
  KALDI_ASSERT(src.NumRows() == this->NumRows());
  KALDI_ASSERT(src.NumCols() == this->NumCols());
  KALDI_ASSERT(alpha.Dim() == this->NumCols());
  KALDI_ASSERT(beta.Dim() == this->NumCols());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CU2DBLOCK), n_blocks(src.NumRows(), CU2DBLOCK));

    cuda_parametric_relu(dimGrid, dimBlock, this->data_, src.data_, this->Dim(),
                         src.Stride(), alpha.data_, beta.data_);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    // Do it on CPU,
    for (MatrixIndexT r = 0; r < NumRows(); r++) {
      for (MatrixIndexT c = 0; c < NumCols(); c++) {
        Real src_elem = src.Mat()(r,c);
        this->Mat()(r,c) = src_elem * (src_elem >= 0.0 ? alpha.Vec()(c) : beta.Vec()(c));
      }
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::DiffParametricRelu(
     const CuMatrixBase<Real> &value,
     const CuMatrixBase<Real> &diff,
     const CuVectorBase<Real> &alpha,
     const CuVectorBase<Real> &beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK), n_blocks(num_rows_, CU2DBLOCK));

    cuda_diff_parametric_relu(dimGrid, dimBlock, data_, diff.data_, value.data_,
                              Dim(), diff.Stride(), value.Stride(),
                              alpha.data_, beta.data_);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    // Do it on CPU,
    for (MatrixIndexT r = 0; r < NumRows(); r++) {
      for (MatrixIndexT c = 0; c < NumCols(); c++) {
        Real value_elem = value.Mat()(r,c);
        this->Mat()(r,c) = diff.Mat()(r,c) *
          (value_elem >= 0.0 ? alpha.Vec()(c) : beta.Vec()(c));
      }
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::Sigmoid(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_sigmoid(dimGrid, dimBlock, this->data_, src.data_, this->Dim(),
                 src.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().Sigmoid(src.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::SoftHinge(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_soft_hinge(dimGrid, dimBlock, this->data_, src.data_, this->Dim(),
                    src.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().SoftHinge(src.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::GroupPnorm(const CuMatrixBase<Real> &src, Real power) {
  int group_size = src.NumCols() / this->NumCols();
  KALDI_ASSERT(src.NumCols() == this->NumCols() * group_size &&
               this->NumRows() == src.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    if (power == Real(0) || power == Real(1) || power == Real(2)
        || power == std::numeric_limits<Real>::infinity()) {
      // One thread block per row.
      // Use 2D block for small group size to simplify the calculation
      // Each group is reduced by threads_per_group threads.
      // threads_per_group should be a power of 2 for fast tree reduction.
      int threads_per_group = CU1DBLOCK;
      while (threads_per_group * 3 / 2 >= group_size) {
        threads_per_group >>= 1;
      }
      if (group_size == 1) {
        threads_per_group = 1;
      }
      dim3 dimBlock(threads_per_group, CU1DBLOCK / threads_per_group);
      dim3 dimGrid(NumRows());
      cuda_group_spec_pnorm(dimGrid, dimBlock, this->data_, src.data_,
                            this->Dim(), src.Stride(), group_size, power);
    } else {
      dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
      dim3 dimGrid(n_blocks(NumCols(), CU2DBLOCK),
                   n_blocks(NumRows(), CU2DBLOCK));
      cuda_group_pnorm(dimGrid, dimBlock, this->data_, src.data_, this->Dim(),
                       src.Stride(), group_size, power);
    }
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().GroupPnorm(src.Mat(), power);
  }
}

template<typename Real>
void CuMatrixBase<Real>::GroupMax(const CuMatrixBase<Real> &src) {
  int group_size = src.NumCols() / this->NumCols();
  KALDI_ASSERT(src.NumCols() == this->NumCols() * group_size &&
               this->NumRows() == src.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    // One thread block per row.
    // Use 2D block for small group size to simplify the calculation.
    // Each group is reduced by threads_per_group threads.
    // threads_per_group should be a power of 2 for fast tree reduction.
    //        group size: 1 2 3 4 5 6 7 .. 12 13 .. 24 25 .. 48 ...
    // threads_per_group: 1 1 1 2 2 2 4 ..  4  8 ..  8 16 .. 16 ...
    int threads_per_group = CU1DBLOCK;
    while (threads_per_group * 3 / 2 >= group_size) {
      threads_per_group >>= 1;
    }
    if (group_size == 1) {
      threads_per_group = 1;
    }
    dim3 dimBlock(threads_per_group, CU1DBLOCK / threads_per_group);
    dim3 dimGrid(NumRows());
    cuda_group_max(dimGrid, dimBlock, this->data_, src.data_, this->Dim(),
        src.Stride(), group_size);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().GroupMax(src.Mat());
  }
}

/*
Think of sv_labels as a Matrix, denoting the "correct" label of each frame to
each phone-state; it's very likely to contain a LOT of zeros

tot_weight = the sum of ALL element in matrix sv_labels
tot_objf = the sum of the product of (each element in matrix sv_labels) and (the
           log of its counterpart in matrix output)

an element in "this" matrix = (the element in matrix sv_labels) divided by (the element in matrix output)
*/
template<typename Real>
void CuMatrix<Real>::CompObjfAndDeriv(const std::vector<MatrixElement<Real> >& sv_labels,
                                      const CuMatrix<Real> &output,
                                      Real *tot_objf, Real* tot_weight) {
  { // check the input.
    typedef typename std::vector<MatrixElement<Real> >::const_iterator Iter;
    MatrixIndexT num_rows = this->num_rows_, num_cols = this->num_cols_;
    for (Iter iter = sv_labels.begin(); iter != sv_labels.end(); ++iter) {
      KALDI_ASSERT(iter->row < num_rows && iter->row >= 0 &&
                   iter->column < num_cols && iter->column >= 0);
    }
  }

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (sv_labels.empty()) {
      KALDI_WARN << "Empty supervision labels";
      *tot_objf = 0.0;
      *tot_weight = 0.0;
      return;
    }
    void *addr = CuDevice::Instantiate().Malloc(sv_labels.size() * sizeof(MatrixElement<Real>));
    CU_SAFE_CALL(cudaMemcpy(addr, sv_labels.data(), sv_labels.size() * sizeof(MatrixElement<Real>), cudaMemcpyHostToDevice));
    CuTimer tim;
    CuVector<Real> tmp(2, kUndefined);
    int dimBlock(CU1DBLOCK);
    int dimGrid = 1; // only 1 block here. we have loops in each thread.
    cuda_comp_obj_deriv(dimGrid, dimBlock, (MatrixElement<Real>*)addr,
                        sv_labels.size(), output.Data(), output.Dim(),
                        this->Data(), this->Dim(), tmp.Data());
    Vector<Real> tmp_cpu(tmp);
    *tot_objf = tmp_cpu(0);
    *tot_weight = tmp_cpu(1);
    CuDevice::Instantiate().Free(addr);
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    *tot_objf = 0.0;
    *tot_weight = 0.0;
    for(int32 i = 0; i<sv_labels.size(); i++) {
      int32 m = sv_labels[i].row, label = sv_labels[i].column;
      Real weight = sv_labels[i].weight;
      //KALDI_ASSERT(label >= 0 && label < nnet_.OutputDim());
      Real this_prob = output(m, label);
      KALDI_ASSERT(this_prob >= 0.99e-20); // we floored to 1.0e-20 in SoftmaxLayer.
      *tot_objf += weight * Log(this_prob);
      *tot_weight += weight;
      (*this)(m, label) += weight / this_prob;
    }
  }
}

template<typename Real> // Y->this, X->src
void CuMatrixBase<Real>::ApplySoftMaxPerRow(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    size_t dimBlock = CU1DBLOCK;
    size_t dimGrid = src.num_rows_;
    cuda_softmax_reduce(dimGrid, dimBlock, data_, src.data_, Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
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

template<typename Real> // Y->this, X->src
void CuMatrixBase<Real>::ApplyLogSoftMaxPerRow(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    size_t dimBlock = CU1DBLOCK;
    size_t dimGrid = src.num_rows_;
    cuda_log_softmax_reduce(dimGrid, dimBlock,
                            data_, src.data_, Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    MatrixBase<Real> &mat(this->Mat());
    mat.CopyFromMat(src.Mat());
    for(MatrixIndexT r = 0; r < mat.NumRows(); r++) {
      mat.Row(r).ApplyLogSoftMax();
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::DiffSigmoid(const CuMatrixBase<Real> &value,
                                     const CuMatrixBase<Real> &diff) {
  KALDI_ASSERT(SameDim(*this, value) && SameDim(*this, diff));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_diff_sigmoid(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim(), diff.Stride(), value.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().DiffSigmoid(value.Mat(), diff.Mat());
  }
}


template<typename Real>
void CuMatrixBase<Real>::Tanh(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);

    cuda_tanh(dimGrid, dimBlock, this->data_, src.data_, this->Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().Tanh(src.Mat());
  }
}



template<typename Real>
void CuMatrixBase<Real>::DiffTanh(const CuMatrixBase<Real> &value,
                                  const CuMatrixBase<Real> &diff) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_diff_tanh(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim(), diff.Stride(), value.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().DiffTanh(value.Mat(), diff.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::FindRowMaxId(CuArray<int32> *id) const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    id->Resize(num_rows_);
    MatrixDim d = Dim();

    // CUDA thread layout: one thread block per matrix-row.
    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(num_rows_);
    cuda_find_row_max_id(dimGrid, dimBlock, data_, NULL, id->Data(), d);
    CU_SAFE_CALL(cudaGetLastError());

    // now we have the indices!
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    // allocate index buffer
    id->Resize(num_rows_);
    id->Set(-1);
    // find maxima
    MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
    for (MatrixIndexT r = 0; r < num_rows; r++) {
      Real max = -1e21;
      int32 max_id = -1;
      const Real *row_data = Mat().RowData(r);
      for (MatrixIndexT c = 0; c < num_cols; c++) {
        if (max < row_data[c]) {
          max = row_data[c];
          max_id = c;
        }
      }
      id->Data()[r] = max_id;
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::DiffSoftmaxPerRow(const CuMatrixBase<Real> &value,
                                           const CuMatrixBase<Real> &diff) {

  KALDI_ASSERT(SameDim(value, diff) && SameDim(value, *this) &&
               this != &value);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    // CUDA thread layout: one thread block per matrix-row.
    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(num_rows_);
    cuda_diff_softmax(dimGrid, dimBlock, data_, this->Dim(), value.Data(),
        value.Stride(), diff.Data(), diff.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    const CuMatrixBase<Real> &P(value), &E(diff);
    CuMatrixBase<Real> &D(*this);

    CuVector<Real> pe_vec(D.NumRows()); // For each row i, the dot product (p_t . e_t).
    pe_vec.AddDiagMatMat(1.0, P, kNoTrans, E, kTrans, 0.0);

    D.CopyFromMat(E);
    D.MulElements(P);
    // At this point, D = P .* E (in matlab notation)
    D.AddDiagVecMat(-1.0, pe_vec, P, kNoTrans, 1.0); // does D -= diag(pe_vec) * P.
  }
}

template<typename Real>
void CuMatrixBase<Real>::DiffLogSoftmaxPerRow(
    const CuMatrixBase<Real> &out_value, const CuMatrixBase<Real> &out_deriv) {

  KALDI_ASSERT(SameDim(out_value, out_deriv) && SameDim(out_value, *this) &&
               this != &out_value);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    // CUDA thread layout: one thread block per matrix-row.
    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(num_rows_);
    cuda_diff_log_softmax(dimGrid, dimBlock, this->Dim(), out_value.Data(),
                          out_value.Stride(), out_deriv.Data(),
                          out_deriv.Stride(), data_);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    if (this == &out_deriv) {
      // the code below doesn't work for in-place, so make a copy and recurse.
      CuMatrix<Real> temp(NumRows(), NumCols(), kUndefined);
      temp.DiffLogSoftmaxPerRow(out_value, out_deriv);
      CopyFromMat(temp);
      return;
    }
    /*
     Let the output be y, then
     y_i = x_i - log(sum_i exp(x_i))
     where x_i is the input to the component. The Jacobian matrix of this
     function is
     J = I - 1 exp(y^T)
     where 1 is a vector of ones. Let the derivative vector at the output be e,
     and at the input be d, then we have
     d = e - exp(y) Sum(e)
     d_i = e_i - exp(y_i) Sum(e)
     */
    const CuMatrixBase<Real> &Y(out_value), &E(out_deriv);
    CuMatrixBase<Real> &D(*this);

    D.CopyFromMat(Y);
    D.ApplyExp();                           // exp(y)
    CuVector<Real> E_sum(D.NumRows()); // Initializes to zero
    E_sum.AddColSumMat(1.0, E);             // Sum(e)
    D.MulRowsVec(E_sum);                    // exp(y) Sum(e)
    D.Scale(-1.0);                          // - exp(y) Sum(e)
    D.AddMat(1.0, E, kNoTrans);             // e - exp(y_i) Sum(e)
  }
}

template<typename Real>
void CuMatrixBase<Real>::DiffXent(const CuArrayBase<int32> &tgt,
                                  CuVector<Real> *log_post_tgt) {

  KALDI_ASSERT(tgt.Dim() == num_rows_);
  log_post_tgt->Resize(tgt.Dim());

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(1, CU2DBLOCK*8);
    dim3 dimGrid(1, n_blocks(tgt.Dim(), CU2DBLOCK*8));
    cuda_diff_xent(dimGrid, dimBlock, tgt.Data(), data_,
                   log_post_tgt->data_, Dim());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    MatrixIndexT num_rows = num_rows_;
    for(int32 r = 0; r < num_rows; r++) {
      int32 col_tgt = tgt.Data()[r];
      Real &value = Mat()(r, col_tgt);
      log_post_tgt->Vec()(r) = Log(value);
      value -= 1.0;
    }
  }
}


template<typename Real>
void CuMatrixBase<Real>::Cholesky(CuMatrixBase<Real> *inv_cholesky) {
  KALDI_ASSERT(this->NumRows() == this->NumCols());
  const int32 block_size = 64;  // We can tune this.
#if HAVE_CUDA == 1
  bool have_gpu = CuDevice::Instantiate().Enabled();
#else
  bool have_gpu = false;
#endif
  if (this->NumRows() == 0) {
    return;
  }
  if (inv_cholesky == NULL && this->NumRows() >= block_size * 2 && have_gpu) {
    // Even if the user did not request the inverse Cholesky, for large enough
    // matrices (on GPUs) it's going to be more efficient to compute it anyway
    // as the recursion depends on it.
    CuMatrix<Real> inv(this->NumRows(), this->NumCols());
    Cholesky(&inv);
    return;
  }
  if (this->NumRows() <= block_size || inv_cholesky == NULL || !have_gpu) {
    // Don't recurse: compute the Cholesky (and inverse Cholesky, if requested)
    // directly, on the CPu.
    int32 dim = this->NumRows();
    CuSpMatrix<Real> this_sp(dim, kUndefined);
    this_sp.CopyFromMat(*this, kTakeLower);
    SpMatrix<Real> this_sp_cpu(this_sp);
    TpMatrix<Real> C_cpu(dim);
    C_cpu.Cholesky(this_sp_cpu);
    CuTpMatrix<Real> C(C_cpu);
    this->CopyFromTp(C);
    if (inv_cholesky != NULL) {
      C_cpu.Invert();  // Get inverse Cholesky on CPU.
      C.CopyFromTp(C_cpu);
      inv_cholesky->CopyFromTp(C); // Copy inverse Cholesky from CPU.
    }
    return;
  }
  // At this point, if none of the other cases apply, we recurse.

  // The selection of dim1 is a heuristic.  We could also just take half.
  int32 tot_dim = this->NumRows();
  int32 dim1;
  // Break it up into a whole number of blocks, for better memory alignment.
  // The line below, setting dim1 can be decided on a heuristic basis: from
  // the point of view of correctness, it can really be any value
  // 0 < dim1 < tot_dim.
  dim1 = block_size * std::max<int32>(1, tot_dim / (2 * block_size));

  int32 dim2 = tot_dim - dim1;
  CuSubMatrix<Real> this_11(*this, 0, dim1, 0, dim1),
      this_12(*this, 0, dim1, dim1, dim2),
      this_21(*this, dim1, dim2, 0, dim1),
      this_22(*this, dim1, dim2, dim1, dim2);
  CuSubMatrix<Real> inv_11(*inv_cholesky, 0, dim1, 0, dim1),
      inv_12(*inv_cholesky, 0, dim1, dim1, dim2),
      inv_21(*inv_cholesky, dim1, dim2, 0, dim1),
      inv_22(*inv_cholesky, dim1, dim2, dim1, dim2);
  /*
    Here is the math on block-wise Cholesky.  We'll use a Matlab-like notation for blocks of a matrix,
    e.g. [ A B; C D ], and also for transposes, e.g. A' is the transpose of A.
    Let A be the input matrix; we want to compute both its Cholesky L and its inverse Cholesky, which
    we'll call M.
    OK. let  L = [ L11 0; L21 L22 ] be the Cholesky factor of A.
    We have A = L L' = [ L11 0; L21 L22 ] * [ L11' L21'; 0 L22' ].  Multiplying it out,
    if A = [ A11 A12; A21 A22 ]; then
    A11 = L11 L11',  A21 = L21 L11', A22 = L21 L21' + L22 L22', and A12 = A21'.

    We also want an expression for the inverse of L (we call this M).
    If M = [ M11 0; M21 M22 ], then it's not hard to see that
    M11 = inv(L11), M22 = inv(L22).
    We can work out M21 as follows.  We know that [ L11 0; L21 L22 ] [ M11 0; M21 M22 ] = [ I 0; 0 I ].
    Considering the zero on the bottom of the rhs, we have: L21 M11 + L22 M21 = 0, which gives us:
    M21 = - L22^{-1} L21 M11 = - M22 L21 M11.

    Next, we want expressions for L21 and L22.  From the equation A21 = L21 L11', we have:
    L21 = A21 inv(L11') = A21 M11'
    We can compute L22 and M22 recursively by doing Cholesky (and computing the inverse Cholesky)
    on the quantity T = (A22 - L21 L21').   [we give it the name T just for easy reference.]

    Computationally, we do this as follows:
    (1) Recurse to get L11 and M11.
    (2) Compute L21 = A21 M11'
    (3) Compute T = A22 - L21 L21'
    (4) Recurse on T to get L22 and M22.
    (5) Compute M21 = -M22 L21 M11.
    Next, we have to consider the in-place nature of the computation, since L overwrites A
    [M has its own storage, in "inv_cholesky"].
    We address this here:
    (1) is in-place [L11 replaces A11, M11 has its own storage].
    (2) L21 gets written where M21 belongs.
    (3) T replaces A22.
    (4) is in-place [L22 replaces T where A22 was, M22 has its own storage]
    (5):(a)  we first compute the transpose of (L21 M11) is done in the upper part of A/L,
    where A12 or L12 would be.  Define a temporary expression
    U = (L21 M11)' = M11' L21'; this goes where A12 or L12 would be.
    (b) copy L21 to where it should be, in *this.
    (c) Compute M21 = -M22 U', in the correct place for M21.
    (d) zero L12 and M12.  */

  // (1) compute L11 and M11.
  this_11.Cholesky(&inv_11);
  // (2) compute L21 = A21 M11'.  For now it's in the "wrong place", where M21 should be.
  inv_21.AddMatMat(1.0, this_21, kNoTrans, inv_11, kTrans, 0.0);
  // (3) compute T = A22 - L21 L21'.  Note: only the lower triangle of T will be valid, but
  //      that's OK because Cholesky will ignore the upper part.
  this_22.SymAddMat2(-1.0, inv_21, kNoTrans, 1.0);
  // (4) Recurse to compute L22 and M22.
  this_22.Cholesky(&inv_22);
  // (5)(a) compute U = M11' L21'.  We use the storage of this_12 for this.  Note that L21 is
  //        currently where M21 should be.
  this_12.AddMatMat(1.0, inv_11, kTrans, inv_21, kTrans, 0.0);
  // (5)(b) copy L21 to where it should be.
  this_21.CopyFromMat(inv_21);
  // (5)(c) compute M21 = -M22 U'.
  inv_21.AddMatMat(-1.0, inv_22, kNoTrans, this_12, kTrans, 0.0);
  // (5)(d) zero L12 and M12.
  this_12.SetZero();
  inv_12.SetZero();
}



template<typename Real>
void CuMatrixBase<Real>::SymInvertPosDef() {
  KALDI_ASSERT(num_rows_ == num_cols_);
  if (num_rows_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CuMatrix<Real> inv_cholesky(num_rows_, num_rows_);
    this->Cholesky(&inv_cholesky);
    // note: SymAddMat2 only updates lower part of *this.
    this->SymAddMat2(1.0, inv_cholesky, kTrans, 0.0);
    this->CopyLowerToUpper();
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    SpMatrix<Real> temp_sp(this->Mat(), kTakeLower);
    TpMatrix<Real> C(temp_sp.NumRows(), kUndefined);
    C.Cholesky(temp_sp);
    C.Invert();
    temp_sp.AddTp2(1.0, C, kTrans, 0.0);
    this->Mat().CopyFromSp(temp_sp);
    // was previously just: CuSpMatrix::Invert().
  }
}

template<typename Real>
bool CuMatrixBase<Real>::ApproxEqual(const CuMatrixBase<Real> &other,
                                     float tol) const {
  CuMatrix<Real> diff(*this);
  diff.AddMat(-1.0, other);
  return (diff.FrobeniusNorm() <= tol * (*this).FrobeniusNorm());
}

template<typename Real>
Real TraceMatMat(const CuMatrixBase<Real> &A,
                 const CuMatrixBase<Real> &B,
                 MatrixTransposeType trans) {
  if (A.num_rows_ == 0) {
    KALDI_ASSERT(B.num_rows_ == 0);
    return 0.0;
  }
  Real result = 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      KALDI_ASSERT(A.NumRows() == B.NumCols() && A.NumCols() == B.NumRows());
    } else {
      KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
    }
    CuTimer tim;
    // 2D blocks: each (8x32) block sums up (32x32) elements.
    // 2D grid: try to cover all the matrix A unless it is too big.
    // Kernel will reduce to ~256 elements with good performance,
    // if the matrix is not in a very bad shape.
    // (wider or taller than 32x8192)
    // CPU will then reduce to 1 element.
    const int kWarpSize = 32;
    dim3 dimBlock(kWarpSize, CU1DBLOCK / kWarpSize);
    dim3 dimGrid(n_blocks(A.NumCols(), kWarpSize),
        n_blocks(A.NumRows(), kWarpSize));
    if (dimGrid.x * dimGrid.y > 256) {
      dimGrid.y = 256 / dimGrid.x;
      if (dimGrid.y == 0) {
        dimGrid.y = 1;
      }
    }
    CuVector<Real> result_vec(dimGrid.x * dimGrid.y, kUndefined);
    if (trans == kNoTrans) {
      cuda_trace_mat_mat(dimGrid, dimBlock, A.Data(), B.Data(), A.Dim(),
          B.Stride(), result_vec.Data());
    } else {
      cuda_trace_mat_mat_trans(dimGrid, dimBlock, A.Data(), B.Data(), A.Dim(),
          B.Stride(), result_vec.Data());
    }
    CU_SAFE_CALL(cudaGetLastError());
    Vector<Real> result_cpu(result_vec); // copying from CUDA faster than summing in CUDA.
    result = result_cpu.Sum();
    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
void AddMatMatBatched(const Real alpha, std::vector<CuSubMatrix<Real>* > &C,
                      const std::vector<CuSubMatrix<Real>* > &A,
                      MatrixTransposeType transA,
                      const std::vector<CuSubMatrix<Real>* > &B,
                      MatrixTransposeType transB, const Real beta) {
  KALDI_ASSERT(A.size() == B.size() && B.size() == C.size());
  int32 size = A.size();

  if (size == 0) return;

  // all elements must have the same num-rows, num-cols and stride
  for (int32 i = 0; i + 1 < size; i++) {
    KALDI_ASSERT(A[i]->NumRows() == A[i+1]->NumRows());
    KALDI_ASSERT(A[i]->NumCols() == A[i+1]->NumCols());
    KALDI_ASSERT(A[i]->Stride() == A[i+1]->Stride());
    KALDI_ASSERT(B[i]->NumRows() == B[i+1]->NumRows());
    KALDI_ASSERT(B[i]->NumCols() == B[i+1]->NumCols());
    KALDI_ASSERT(B[i]->Stride() == B[i+1]->Stride());
    KALDI_ASSERT(C[i]->NumRows() == C[i+1]->NumRows());
    KALDI_ASSERT(C[i]->NumCols() == C[i+1]->NumCols());
    KALDI_ASSERT(C[i]->Stride() == C[i+1]->Stride());
  }
  // CUBLAS is col-major, cudamatrix is row-major, how to do the mapping?
  // keep trans..., just swap A&B matrices: A->B B->A
  MatrixIndexT m = ((transB==kTrans)? B[0]->NumRows() : B[0]->NumCols());
  MatrixIndexT n = ((transA==kTrans)? A[0]->NumCols() : A[0]->NumRows());
  MatrixIndexT k = ((transB==kTrans)? B[0]->NumCols() : B[0]->NumRows());
  MatrixIndexT k1 = ((transA==kTrans)? A[0]->NumRows() : A[0]->NumCols());

  KALDI_ASSERT(m == C[0]->NumCols());
  KALDI_ASSERT(n == C[0]->NumRows());
  KALDI_ASSERT(k == k1);

  if (m == 0) return;

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    Real **device_abc_array =
        static_cast<Real**>(CuDevice::Instantiate().Malloc(3 * size * sizeof(Real*)));
    const Real **device_a_array = const_cast<const Real**>(device_abc_array);
    const Real **device_b_array = const_cast<const Real**>(device_abc_array) + size;
    Real **device_c_array = device_abc_array + 2 * size;
    const Real **host_abc_array = new const Real*[3*size];
    const Real **host_a_array = host_abc_array;
    const Real **host_b_array = host_abc_array + size;
    const Real **host_c_array = host_abc_array + 2 * size;

    for (int32 i = 0; i < size; i++) {
      host_a_array[i] = A[i]->data_;
      host_b_array[i] = B[i]->data_;
      host_c_array[i] = C[i]->data_;
    }

    CU_SAFE_CALL(cudaMemcpy(device_abc_array, host_abc_array, 3*size*sizeof(Real*), cudaMemcpyHostToDevice));

    CUBLAS_SAFE_CALL(cublas_gemmBatched(GetCublasHandle(),
                                        (transB==kTrans? CUBLAS_OP_T:CUBLAS_OP_N),
                                        (transA==kTrans? CUBLAS_OP_T:CUBLAS_OP_N),
                                        m, n, k, alpha, device_b_array,
                                        B[0]->Stride(), device_a_array,
                                        A[0]->Stride(), beta, device_c_array,
                                        C[0]->Stride(), size));

    CuDevice::Instantiate().Free(device_abc_array);
    delete[] host_abc_array;

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    for (int32 i = 0; i < size; i++) {
      C[i]->Mat().AddMatMat(alpha, A[i]->Mat(), transA, B[i]->Mat(), transB, beta);
    }
  }
}

template
void AddMatMatBatched(const float alpha, std::vector<CuSubMatrix<float>* > &C,
                      const std::vector<CuSubMatrix<float>* > &A,
                      MatrixTransposeType transA,
                      const std::vector<CuSubMatrix<float>* > &B,
                      MatrixTransposeType transB, const float beta);

template
void AddMatMatBatched(const double alpha, std::vector<CuSubMatrix<double>* > &C,
                      const std::vector<CuSubMatrix<double>* > &A,
                      MatrixTransposeType transA,
                      const std::vector<CuSubMatrix<double>* > &B,
                      MatrixTransposeType transB, const double beta);

template<typename Real>
void CuMatrixBase<Real>::CopyRowsFromVec(const CuVectorBase<Real> &v) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    if (v.Dim() == num_rows_*num_cols_) {
      if (stride_ == num_cols_) {
        const Real* v_data = v.Data();
        CU_SAFE_CALL(
          cudaMemcpyAsync(data_, v_data, sizeof(Real)*num_rows_*num_cols_,
                          cudaMemcpyDeviceToDevice, cudaStreamPerThread));
      } else {
        CU_SAFE_CALL(
          cudaMemcpy2DAsync(data_, stride_ * sizeof(Real), v.Data(),
                            num_cols_*sizeof(Real), num_cols_*sizeof(Real),
                            num_rows_, cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
      }
    } else if (v.Dim() == num_cols_) {
      dim3 dimGrid, dimBlock;
      GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                            &dimGrid, &dimBlock);
      cuda_copy_rows_from_vec(dimGrid, dimBlock, data_, this->Dim(), v.Data());
      CU_SAFE_CALL(cudaGetLastError());
    } else {
      KALDI_ERR << "Wrong sized arguments";
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    CuTimer tim;
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
      dim3 dimGrid, dimBlock;
      GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                            &dimGrid, &dimBlock);
      cuda_copy_rows_from_vec(dimGrid, dimBlock, this->data_, this->Dim(), v.Data());
      CU_SAFE_CALL(cudaGetLastError());
    } else {
      KALDI_ERR << "Wrong sized arguments";
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyRowsFromVec(v);
  }
}

template<typename Real>
void CuMatrixBase<Real>::CopyColsFromVec(const CuVectorBase<Real> &rv) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    if (rv.Dim() == num_rows_ * num_cols_) {
      // treat rv as a matrix of the size (num_cols x num_rows_)
      // and use transposed copy to fill *this
      // see CuMatrixBase<Real>::CopyFromMat() for more detail of the impl
      MatrixDim rv_dim = { num_cols_, num_rows_, num_rows_ };
      const int32 warpSize = 32;
      dim3 dimBlock(warpSize, CU1DBLOCK / warpSize);
      dim3 dimGrid(n_blocks(rv_dim.cols, warpSize),
                   n_blocks(rv_dim.rows, warpSize));
      cuda_copy_from_mat_trans(dimGrid, dimBlock, data_, rv.Data(), Dim(),
                               rv_dim);
      CU_SAFE_CALL(cudaGetLastError());
    } else if (rv.Dim() == num_rows_) {
      // use 2D block (8x32) and large enough grid to cover matrix *this
      // dimBlock.x need to be at least warpSize for coalesced memory access.
      const int32 warpSize = 32;
      dim3 dimBlock(warpSize, CU1DBLOCK / warpSize);
      dim3 dimGrid(n_blocks(num_cols_, dimBlock.x),
                   n_blocks(num_rows_, dimBlock.y));
      cuda_copy_cols_from_vec(dimGrid, dimBlock, Data(), Dim(), rv.Data());
      CU_SAFE_CALL(cudaGetLastError());
    } else {
      KALDI_ERR<< "Wrong sized arguments";
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyColsFromVec(rv.Vec());
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
    CuTimer tim;
    cublas_copy(GetCublasHandle(),
                v.Dim(), v.Data(), 1,
                this->data_ + col, this->stride_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_apply_pow(dimGrid, dimBlock, data_, power, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().ApplyPow(power);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyPowAbs(Real power, bool include_sign) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_apply_pow_abs(dimGrid, dimBlock, data_, power, include_sign, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().ApplyPowAbs(power, include_sign);
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyHeaviside() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_apply_heaviside(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().ApplyHeaviside();
  }
}

template<typename Real>
void CuMatrixBase<Real>::Heaviside(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_heaviside(dimGrid, dimBlock, this->data_, src.data_, this->Dim(),
                   src.Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
  #endif
  {
    Mat().Heaviside(src.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyExp() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_apply_exp(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().ApplyExp();
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyExpLimited(Real lower_limit, Real upper_limit) {
  KALDI_ASSERT(upper_limit > lower_limit);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_apply_exp_limited(dimGrid, dimBlock, data_, Dim(), lower_limit, upper_limit);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    int32 num_rows = num_rows_, num_cols = num_cols_;
    for (int32 r = 0; r < num_rows; r++) {
      Real *row_data = this->RowData(r);
      for (int32 c = 0; c < num_cols; c++) {
        Real x = row_data[c];
        if (!(x >= lower_limit))
          x = lower_limit;
        if (x > upper_limit)
          x = upper_limit;
        row_data[c] = Exp(x);
      }
    }
  }
}


template<typename Real>
void CuMatrixBase<Real>::ApplyExpSpecial() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    const int warpSize = 32;
    dim3 dimBlock(CU1DBLOCK / warpSize, warpSize);
    dim3 dimGrid(n_blocks(NumRows(), dimBlock.x),
                 n_blocks(NumCols(), dimBlock.y));

    cuda_apply_exp_special(dimGrid, dimBlock, Data(), Dim(), Data(), Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().ApplyExpSpecial();
  }
}

template<typename Real>
void CuMatrixBase<Real>::ApplyFloor(Real floor_val) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_apply_floor(dimGrid, dimBlock, data_, floor_val, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_apply_ceiling(dimGrid, dimBlock, data_, ceiling_val, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
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
    CuTimer tim;
    if (mat.Stride() == mat.NumCols()) {
      cudaMemcpy(data_, mat.Data(), sizeof(Real)*dim_, cudaMemcpyDeviceToHost);
    } else {
      // we could definitely do better than the following.
      Real* vec_data = data_;
      for (MatrixIndexT r = 0; r < mat.NumRows(); r++) {
        cudaMemcpy(vec_data, mat.RowData(r), sizeof(Real) * mat.NumCols(),
                   cudaMemcpyDeviceToHost);
        vec_data += mat.NumCols();
      }
    }
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyRowsFromMat", tim);
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
void CuMatrixBase<Real>::CopyCols(const CuMatrixBase<Real> &src,
                                  const CuArrayBase<MatrixIndexT> &indices) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(indices.Dim() == NumCols());
    KALDI_ASSERT(NumRows() == src.NumRows());
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_copy_cols(dimGrid, dimBlock, data_, src.Data(), indices.Data(), Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyCols(src.Mat(), indices.Data());
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyRows(const CuMatrixBase<Real> &src,
                                  const CuArrayBase<MatrixIndexT> &indices) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<MatrixIndexT>(indices.Dim()) == NumRows());
    KALDI_ASSERT(NumCols() == src.NumCols());

    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_copy_rows(dimGrid, dimBlock, data_, src.Data(), indices.Data(),
                   Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyRows(src.Mat(), indices.Data());
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddCols(const CuMatrixBase<Real> &src,
                                 const CuArrayBase<MatrixIndexT> &indices) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(indices.Dim() == NumCols());
    KALDI_ASSERT(NumRows() == src.NumRows());
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_cols(dimGrid, dimBlock, data_, src.Data(), indices.Data(),
                  Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddCols(src.Mat(), indices.Data());
  }
}

template<typename Real>
void CuMatrixBase<Real>::CopyRows(const CuArrayBase<const Real*> &src) {
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<MatrixIndexT>(src.Dim()) == NumRows());
    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK),
                 n_blocks(num_rows_, CU2DBLOCK));
    cuda_copy_rows(dimGrid, dimBlock, data_, src.Data(), Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyRows(src.Data());
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyToRows(const CuArrayBase<Real*> &dst) const {
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<MatrixIndexT>(dst.Dim()) == NumRows());

    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CU2DBLOCK),
                 n_blocks(num_rows_, CU2DBLOCK));
    cuda_copy_to_rows(dimGrid, dimBlock, dst.Data(), data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyToRows(dst.Data());
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddRows(Real alpha,
                                 const CuMatrixBase<Real> &src,
                                 const CuArrayBase<MatrixIndexT> &indexes) {
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<MatrixIndexT>(indexes.Dim()) == NumRows());
    KALDI_ASSERT(src.NumCols() == NumCols());
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_rows(dimGrid, dimBlock, alpha,
                  data_, src.Data(), indexes.Data(), Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddRows(alpha, src.Mat(), indexes.Data());
  }
}

template<typename Real>
void CuMatrixBase<Real>::MulRows(const CuMatrixBase<Real> &src,
                                 const CuArrayBase<MatrixIndexT> &indexes) {
  if (NumRows() == 0) return;
  KALDI_ASSERT(static_cast<MatrixIndexT>(indexes.Dim()) == NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(src.NumCols() == NumCols());
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_mul_rows(dimGrid, dimBlock,
                  data_, src.Data(), indexes.Data(), Dim(), src.Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    MatrixBase<Real> &this_mat(Mat());
    const MatrixBase<Real> &src_mat(src.Mat());
    int32 num_rows = NumRows();
    const MatrixIndexT *index_ptr = indexes.Data();
    for (int32 r = 0; r < num_rows; r++) {
      int32 src_r = index_ptr[r];
      if (src_r < 0)
        continue;
      SubVector<Real> this_row(this_mat, r),
          src_row(src_mat, src_r);
      this_row.MulElements(src_row);
    }
  }
}



template<typename Real>
void CuMatrixBase<Real>::AddRows(Real alpha, const CuArrayBase<const Real*> &src) {
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<MatrixIndexT>(src.Dim()) == NumRows());
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_rows(dimGrid, dimBlock, alpha, data_, src.Data(), Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddRows(alpha, src.Data());
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddToRows(Real alpha,
                                   const CuArrayBase<Real*> &dst) const {
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<MatrixIndexT>(dst.Dim()) == NumRows());
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_to_rows(dimGrid, dimBlock, alpha, dst.Data(), data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddToRows(alpha, dst.Data());
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddToRows(Real alpha,
                                   const CuArrayBase<MatrixIndexT> &indexes,
                                   CuMatrixBase<Real> *dst) const {
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(static_cast<MatrixIndexT>(indexes.Dim()) == NumRows());
    KALDI_ASSERT(dst->NumCols() == NumCols());
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_to_rows(dimGrid, dimBlock, alpha, dst->Data(), data_, indexes.Data(), Dim(), dst->Stride());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().AddToRows(alpha, indexes.Data(), &(dst->Mat()));
  }
}


template<typename Real>
void CuMatrixBase<Real>::SumColumnRanges(const CuMatrixBase<Real> &src,
                                         const CuArrayBase<Int32Pair> &indices) {
  KALDI_ASSERT(static_cast<MatrixIndexT>(indices.Dim()) == NumCols());
  KALDI_ASSERT(NumRows() == src.NumRows());
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_sum_column_ranges(dimGrid, dimBlock, data_, Dim(), src.Data(),
                           src.Dim(), indices.Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    int32 num_rows = this->num_rows_, num_cols = this->num_cols_,
       this_stride = this->stride_, src_stride = src.stride_;
    Real *data = this->data_;
    const Real *src_data = src.data_;
    const Int32Pair *indices_data = indices.Data();
    for (int32 row = 0; row < num_rows; row++) {
      for (int32 col = 0; col < num_cols; col++) {
        int32 start_col = indices_data[col].first,
                end_col = indices_data[col].second;
        Real sum = 0.0;
        for (int32 src_col = start_col; src_col < end_col; src_col++)
          sum += src_data[row * src_stride + src_col];
        data[row * this_stride + col] = sum;
      }
    }
  }
}


template<typename Real>
void CuMatrixBase<Real>::AddRowRanges(const CuMatrixBase<Real> &src,
                                      const CuArrayBase<Int32Pair> &indexes) {
  KALDI_ASSERT(static_cast<MatrixIndexT>(indexes.Dim()) == NumRows());
  KALDI_ASSERT(src.NumCols() == NumCols());
  if (NumRows() == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_add_row_ranges(dimGrid, dimBlock,
                        data_, Dim(), src.Data(), src.Dim(), indexes.Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  { // Implement here for the CPU..
    int32 num_rows = this->num_rows_, num_cols = this->num_cols_,
          this_stride = this->stride_, src_stride = src.stride_;
    Real *data = this->data_;
    const Real *src_data = src.data_;
    const Int32Pair *indexes_data = indexes.Data();
    for (int32 row = 0; row < num_rows; row++) {
      int32 start_row = indexes_data[row].first,
          end_row = indexes_data[row].second;
      for (int32 col = 0; col < num_cols; col++) {
        Real sum = 0.0;
        for (int32 src_row = start_row; src_row < end_row; src_row++)
          sum += src_data[src_row * src_stride + col];
        data[row * this_stride + col] += sum;
      }
    }
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyLowerToUpper() {
  KALDI_ASSERT(num_cols_ == num_rows_);
  if (num_rows_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    int32 dim = num_rows_;
    dim3 dimGrid(n_blocks(dim, CU2DBLOCK),
                 n_blocks(dim, CU2DBLOCK));
    cuda_copy_low_upp(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyLowerToUpper();
  }
}

template<typename Real>
void CuMatrixBase<Real>::CopyUpperToLower() {
  KALDI_ASSERT(num_cols_ == num_rows_);
  if (num_rows_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    int32 dim = this->num_rows_;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(dim, CU2DBLOCK),
                 n_blocks(dim, CU2DBLOCK));
    cuda_copy_upp_low(dimGrid, dimBlock, data_, Dim());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyUpperToLower();
  }
}


template<typename Real>
Real CuMatrixBase<Real>::Sum() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(num_rows_ > 0 && num_cols_ > 0);
    CuTimer tim;

    CuVector<Real> col_sum(num_rows_, kUndefined);
    cuda_sum_mat_cols(num_rows_, CU1DBLOCK, col_sum.Data(), data_, Dim());
    Real ans = col_sum.Sum();

    CuDevice::Instantiate().AccuProfile(__func__, tim);
    return ans;
  } else
#endif
  {
    return Mat().Sum();
  }
}


template<typename Real>
Real CuMatrixBase<Real>::Max() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(num_rows_ > 0 && num_cols_ > 0);
    CuTimer tim;

    CuVector<Real> col_max(num_rows_, kUndefined);
    cuda_max_mat_cols(num_rows_, CU1DBLOCK, col_max.Data(), data_, Dim());
    Real ans = col_max.Max();

    CuDevice::Instantiate().AccuProfile(__func__, tim);
    return ans;
  } else
#endif
  {
    return Mat().Max();
  }
}


template<typename Real>
Real CuMatrixBase<Real>::Min() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(num_rows_ > 0 && num_cols_ > 0);
    CuTimer tim;

    CuVector<Real> col_min(num_rows_, kUndefined);
    cuda_min_mat_cols(num_rows_, CU1DBLOCK, col_min.Data(), data_, Dim());
    Real ans = col_min.Min();

    CuDevice::Instantiate().AccuProfile(__func__, tim);
    return ans;
  } else
#endif
  {
    return Mat().Min();
  }
}


template<typename Real>
Real CuMatrixBase<Real>::Trace(bool check_square) const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    if (check_square) KALDI_ASSERT(this->num_rows_ == this->num_cols_);
    MatrixIndexT dim = std::min(this->num_rows_, this->num_cols_);
    CuVector<Real> tmp(1, kUndefined); // for result.
    int dimBlock(CU1DBLOCK);
    int dimGrid = 1;// only 1 block here. we have loops in each thread  //(n_blocks(dim_, CU1DBLOCK));
    cuda_vec_sum(dimGrid, dimBlock, data_, tmp.Data(), dim, Stride() + 1);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::Sum", tim);
    return tmp(0);
  } else
#endif
  {
    return Mat().Trace(check_square);
  }
}

template <typename Real>
void CuMatrixBase<Real>::CopyFromGeneralMat(const GeneralMatrix &src,
                                            MatrixTransposeType trans) {
  switch (src.Type()) {
    case kFullMatrix: {
      const Matrix<BaseFloat> &src_full_mat = src.GetFullMatrix();
      this->CopyFromMat(src_full_mat, trans);
      return;
    }
    case kCompressedMatrix: {
      Matrix<BaseFloat> mat;
      src.GetMatrix(&mat);
      this->CopyFromMat(mat, trans);
      return;
    }
    case kSparseMatrix: {
      const SparseMatrix<BaseFloat> &smat = src.GetSparseMatrix();
#if HAVE_CUDA == 1
      if (CuDevice::Instantiate().Enabled()) {
        // only take this branch if we're actually using CUDA, or it would
        // entail a wasteful copy of the sparse matrix.
        CuSparseMatrix<BaseFloat> cu_smat(smat);
        cu_smat.CopyToMat(this, trans);
        return;
      }
#endif
      smat.CopyToMat(&(Mat()), trans);
      return;
    }
    default:
      KALDI_ERR << "Invalid GeneralMatrix type.";
  }
}



template<typename Real>
void CuMatrixBase<Real>::SetRandn() {
  if (num_rows_ == 0) return;
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

template<typename Real>
void CuMatrixBase<Real>::SetRandUniform() {
  if (num_rows_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuRand<Real> tmp;
    tmp.RandUniform(this);
  } else
#endif
  {
    Mat().SetRandUniform();
  }
}

template<typename Real>
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


template<typename Real>
void CuMatrix<Real>::Transpose() {
  if (this->num_rows_ == 0)
    return;
  // Copy and swap for all cases.
  // No need for a separate kernel of squared matrix in-place transpose.
  // It has the same possible peak performance as copy_transpose,
  // if allocate/deallocate overhead can be ignored.
  CuMatrix<Real> tmp(*this, kTrans);
  this->Swap(&tmp);
}


// Version of AddMatMat where 2nd argument is of type CuBlockMatrix.
// Caution:
template<typename Real>
void CuMatrixBase<Real>::AddMatBlock(
    Real alpha,
    const CuMatrixBase<Real> &A, MatrixTransposeType transA,
    const CuBlockMatrix<Real> &B, MatrixTransposeType transB,
    Real beta) {
  // Check dimensions
  int32 A_num_rows = A.NumRows(), A_num_cols = A.NumCols(),
      A_row_stride = A.Stride(), A_col_stride = 1,
      B_num_rows = B.NumRows(), B_num_cols = B.NumCols();
  if (transA == kTrans) {
    std::swap(A_num_rows, A_num_cols);
    std::swap(A_row_stride, A_col_stride);
  }
  if (transB == kTrans) {
    std::swap(B_num_rows, B_num_cols);
  }
  // At this point the {A,B}_{rows,cols} variables are
  // after any transposition.
  KALDI_ASSERT(NumRows() == A_num_rows && NumCols() == B_num_cols);
  KALDI_ASSERT(A_num_cols == B_num_rows);
  int32 B_num_blocks = B.NumBlocks();

  if (num_rows_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    MatrixDim this_dim = Dim();

    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    // (x,y) indices will be (row of *this, block of B)
    dim3 dimGrid(n_blocks(num_rows_, CU2DBLOCK),
                 n_blocks(B_num_blocks, CU2DBLOCK));

    // caution: the use of x as the row-index is not good, but
    // this code is not much used, so I'm not updating it.a
    cuda_add_mat_blockmat(dimGrid, dimBlock, data_, this_dim, A.Data(),
                          A_num_rows, A_num_cols, A_row_stride, A_col_stride,
                          B.CuData(), B_num_blocks, alpha, beta,
                          (transB == kTrans ? 1 : 0));

    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    // "row_offset" and "col_offset" are offsets into B (or into B^T, if
    // transB == kTrans).
    int32 row_offset = 0, col_offset = 0;
    for (int32 b = 0; b < B_num_blocks; b++) {
      const CuSubMatrix<Real> this_block = B.Block(b);
      int32 this_num_rows = this_block.NumRows(),
          this_num_cols = this_block.NumCols();
      if (transB == kTrans) std::swap(this_num_rows, this_num_cols);
      CuSubMatrix<Real> this_part(*this, 0, num_rows_,
                                  col_offset, this_num_cols);
      CuSubMatrix<Real> A_part = (transA == kNoTrans ?
                                  CuSubMatrix<Real>(A, 0, num_rows_,
                                                    row_offset, this_num_rows) :
                                  CuSubMatrix<Real>(A, row_offset, this_num_rows,
                                                    0, num_rows_));
      this_part.AddMatMat(alpha, A_part, transA, this_block, transB, beta);
      row_offset += this_num_rows;
      col_offset += this_num_cols;
    }
    // Note: the values being compared below are all after applying any
    // transposition to B.
    KALDI_ASSERT(row_offset == B_num_rows && col_offset == B_num_cols);
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddElements(Real alpha,
                                     const std::vector<MatrixElement<Real> >& input) {
  // Checks the dimension.
  MatrixIndexT num_rows = this->num_rows_, num_cols = this->num_cols_;
  for (int32 i = 0; i < input.size(); ++i) {
    KALDI_ASSERT(input[i].row < num_rows && input[i].row >= 0 &&
                 input[i].column < num_cols && input[i].column >= 0);
  }
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    void *addr = CuDevice::Instantiate().Malloc(input.size() * sizeof(MatrixElement<Real>));
    CU_SAFE_CALL(cudaMemcpy(addr, input.data(),
                        input.size() * sizeof(MatrixElement<Real>),
                            cudaMemcpyHostToDevice));

    CuTimer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(input.size(), CU1DBLOCK));

    cuda_matrix_add_elements(dimGrid, dimBlock, this->data_, this->Dim(),
                             alpha, (MatrixElement<Real>*)addr, input.size());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().Free(addr);
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    for (int32 i = 0; i < input.size(); i++) {
      (*this)(input[i].row, input[i].column) += alpha * input[i].weight;
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddElements(Real alpha, const CuArrayBase<Int32Pair> &indexes,
                                     const Real *input) {
  if (indexes.Dim() == 0) return;
  KALDI_ASSERT(input != NULL);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CuVector<Real> tmp_vec(indexes.Dim(), kUndefined);
    CU_SAFE_CALL(cudaMemcpy(tmp_vec.Data(), input, indexes.Dim() * sizeof(Real),
                            cudaMemcpyHostToDevice));

    int dimBlock(CU1DBLOCK);
    int dimGrid = n_blocks(indexes.Dim(), CU1DBLOCK);
    cuda_matrix_add_indexed_values(dimGrid, dimBlock, this->Dim(), alpha,
                                   indexes.Data(), tmp_vec.Data(), indexes.Dim(), this->data_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    MatrixIndexT num_rows = this->num_rows_, num_cols = this->num_cols_;
    const Int32Pair *index = indexes.Data();
    for (int32 i = 0; i < indexes.Dim(); i++) {
      KALDI_ASSERT(index[i].first < num_rows && index[i].first >= 0 &&
                   index[i].second < num_cols && index[i].second >= 0);
      (*this)(index[i].first, index[i].second) += alpha * input[i];
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::AddToElements(Real alpha, const CuArrayBase<int32> &elements) {
  KALDI_ASSERT(elements.Dim() == NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(n_blocks(NumRows(), CU1DBLOCK));

    cuda_matrix_add_to_elements(dimGrid, dimBlock, alpha, data_, Dim(), elements.Data());
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    MatrixBase<Real> &this_mat = this->Mat();
    const int32* row_to_col = elements.Data();
    for (int32 r = 0; r < this_mat.NumRows(); r++) {
      KALDI_ASSERT(row_to_col[r] >= -1);
      if (row_to_col[r] >= 0)
        this_mat(r, row_to_col[r]) += alpha;
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::Lookup(const std::vector<Int32Pair> &indices,
                                Real *output) const {
  // Checks the dimension.
  MatrixIndexT num_rows = this->num_rows_, num_cols = this->num_cols_;
  for (int32 i = 0; i < indices.size(); ++i) {
    KALDI_ASSERT(indices[i].first < num_rows && indices[i].first >= 0 &&
                 indices[i].second < num_cols && indices[i].second >= 0);
  }
  if (indices.size() == 0) return;
  KALDI_ASSERT(output != NULL);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuArray<Int32Pair> cuda_indices(indices);
    Lookup(cuda_indices, output);
  } else
#endif
  {
    for (int32 i = 0; i < indices.size(); i++) {
      output[i] = (*this)(indices[i].first, indices[i].second);
    }
  }
}

template<typename Real>
void CuMatrixBase<Real>::Lookup(const CuArrayBase<Int32Pair> &indices,
                                Real *output) const {
  int32 num_elements = indices.Dim();
  if (num_elements == 0) return;
  KALDI_ASSERT(output != NULL);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuArray<Real> cuda_output(num_elements);
    CuTimer tim;
    dim3 dimBlock(CU1DBLOCK, 1);
    dim3 dimGrid(n_blocks(num_elements, CU1DBLOCK), 1);

    cuda_matrix_lookup(dimGrid, dimBlock, this->data_, this->Dim(),
                       indices.Data(), num_elements, cuda_output.Data());
    CU_SAFE_CALL(cudaGetLastError());

    cuda_output.CopyToHost(output);
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    MatrixIndexT num_rows = this->num_rows_, num_cols = this->num_cols_;
    const Int32Pair *index = indices.Data();
    for (int32 i = 0; i < num_elements; i++) {
      KALDI_ASSERT(index[i].first < num_rows && index[i].first >= 0 &&
                   index[i].second < num_cols && index[i].second >= 0);
      output[i] = (*this)(index[i].first, index[i].second);
    }
  }
}


template<typename Real>
void CuMatrixBase<Real>::EqualElementMask(const CuMatrixBase<Real> &mat, CuMatrix<Real> *mask) const {
  // Check the inputs:
  KALDI_ASSERT(mat.NumRows() == NumRows() && mat.NumCols() == NumCols());
  KALDI_ASSERT(mask != NULL);
  // Resizes the output matrix:
  mask->Resize(NumRows(), NumCols(), kSetZero);

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    dim3 dimGrid, dimBlock;
    GetBlockSizesForSimpleMatrixOperation(NumRows(), NumCols(),
                                          &dimGrid, &dimBlock);
    cuda_equal_element_mask(dimGrid, dimBlock, this->data_, mat.Data(),
                            mask->Data(), this->Dim(), mat.Stride(),
                            mask->Stride());
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    for (int32 r = 0; r < NumRows(); r++) {
      for (int32 c = 0; c < NumCols(); c++) {
        (*mask)(r,c) = ((*this)(r,c) ==  mat(r,c) ? 1.0 : 0.0);
      }
    }
  }
}


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

// cudamatrix/cu-block-matrix.cc

// Copyright 2013      Johns Hopkins University (author: Daniel Povey)

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
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cudamatrix/cu-block-matrix.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {

template<class Real>
CuBlockMatrix<Real>::CuBlockMatrix() {
#if HAVE_CUDA == 1
  cu_data_ = NULL;
#endif
}

template<class Real>
CuBlockMatrix<Real>::CuBlockMatrix(const std::vector<CuMatrix<Real> >&data) {
#if HAVE_CUDA == 1
  cu_data_ = NULL;
#endif
  block_data_.resize(data.size());
  MatrixIndexT row_offset = 0, col_offset = 0, max_num_rows = 0;
  for (size_t b = 0; b < data.size(); b++) {
    MatrixIndexT num_rows = data[b].NumRows(), num_cols = data[b].NumCols();
    KALDI_ASSERT(num_rows > 0 && num_cols > 0);
    BlockMatrixData block_data;
    block_data.num_rows = num_rows;
    block_data.num_cols = num_cols;
    block_data.row_offset = row_offset;
    block_data.col_offset = col_offset;
    row_offset += num_rows;
    col_offset += num_cols;
    max_num_rows = std::max(max_num_rows, num_rows);
    block_data_[b] = block_data;
  }
  num_rows_ = row_offset;
  data_.Resize(max_num_rows, col_offset);
  for (int32 b = 0; b < NumBlocks(); b++)
    Block(b).CopyFromMat(data[b]);
  SetCudaData();
}


template<class Real>
const CuSubMatrix<Real> CuBlockMatrix<Real>::Block(int32 b) const {
  KALDI_ASSERT(static_cast<size_t>(b) < block_data_.size());
  const BlockMatrixData &block_data = block_data_[b];
  return CuSubMatrix<Real>(data_, 0, block_data.num_rows,
                           block_data.col_offset, block_data.num_cols);
}

template<class Real>
CuSubMatrix<Real> CuBlockMatrix<Real>::Block(int32 b) {
  KALDI_ASSERT(static_cast<size_t>(b) < block_data_.size());
  BlockMatrixData &block_data = block_data_[b];
  return CuSubMatrix<Real>(data_, 0, block_data.num_rows,
                           block_data.col_offset, block_data.num_cols);
}


template<class Real>
CuBlockMatrix<Real>::CuBlockMatrix(const CuBlockMatrix<Real> &other):
    data_(other.data_), block_data_(other.block_data_), num_rows_(other.num_rows_) {
#if HAVE_CUDA == 1
  cu_data_ = NULL;
#endif
  SetCudaData();
}

template<class Real>
CuBlockMatrix<Real> &CuBlockMatrix<Real>::operator =(const CuBlockMatrix<Real> &other) {
  FreeCudaData();
  data_ = other.data_;
  block_data_ = other.block_data_;
  num_rows_ = other.num_rows_;
  SetCudaData();
  return *this;
}

template<class Real>
void CuBlockMatrix<Real>::FreeCudaData() {
#if HAVE_CUDA == 1
  if (cu_data_ != NULL) {
    if (CuDevice::Instantiate().Enabled()) {
      CuDevice::Instantiate().Free(cu_data_);
      cu_data_ = NULL;
    } else {
      KALDI_ERR << "CuBlockMatrix: you have CUDA data pointer but "
                << "no GPU is enabled: likely code error.";
    }
  }
#endif
}


template<class Real>
void CuBlockMatrix<Real>::SetCudaData() {
#if HAVE_CUDA == 1
  KALDI_ASSERT(cu_data_ == NULL);
  if (block_data_.size() == 0) return; // Nothing to do.
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    std::vector<CuBlockMatrixData> tmp_cu_data(NumBlocks());
    int32 row_offset = 0, col_offset = 0;
    for (size_t b = 0; b < NumBlocks(); b++) {
      CuSubMatrix<Real> this_mat = Block(b);
      CuBlockMatrixData &this_cu_data = tmp_cu_data[b];
      this_cu_data.row_offset = row_offset;
      this_cu_data.col_offset = col_offset;
      this_cu_data.matrix_dim = this_mat.Dim();
      this_cu_data.matrix_data = static_cast<void*>(this_mat.Data());
      row_offset += this_mat.NumRows();
      col_offset += this_mat.NumCols();
    }
    size_t size = NumBlocks() * sizeof(CuBlockMatrixData);
    cu_data_ = static_cast<CuBlockMatrixData*>(
        CuDevice::Instantiate().Malloc(size));
    CU_SAFE_CALL(cudaMemcpy(cu_data_, &(tmp_cu_data[0]), size, cudaMemcpyHostToDevice));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
  }
#endif
}

template<class Real>
void CuBlockMatrix<Real>::Swap(CuBlockMatrix<Real> *other) {
  data_.Swap(&other->data_);
  block_data_.swap(other->block_data_);
  std::swap(num_rows_, other->num_rows_);
#if HAVE_CUDA == 1
  std::swap(cu_data_, other->cu_data_);
#endif
}

template<class Real>
void CuBlockMatrix<Real>::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<CuBlockMatrix>");
  int32 num_blocks = NumBlocks();
  WriteBasicType(os, binary, num_blocks);
  for (int32 b = 0; b < num_blocks; b++)
    this->Block(b).Write(os, binary);
  WriteToken(os, binary, "</CuBlockMatrix>");  
}


template<class Real>
void CuBlockMatrix<Real>::Read(std::istream &is, bool binary) {
  Destroy();
  int i = Peek(is, binary);
  std::vector<CuMatrix<Real> > data;
  if (i != static_cast<int>('<')) {
    // back-compatibility code so we can read the older format of
    // MixtureProbComponent.  This code should be deleted eventually.
    int32 size;
    ReadBasicType(is, binary, &size);
    KALDI_ASSERT(size >= 0);
    data.resize(size);
    for (int32 i = 0; i < size; i++)
      data[i].Read(is, binary);
  } else {
    ExpectToken(is, binary, "<CuBlockMatrix>");
    int32 size;
    ReadBasicType(is, binary, &size);
    KALDI_ASSERT(size >= 0);
    data.resize(size);
    for (int32 i = 0; i < size; i++)
      data[i].Read(is, binary);
    ExpectToken(is, binary, "</CuBlockMatrix>");    
  }

  CuBlockMatrix<Real> block_mat(data); // initializer from std::vector<CuMatrix<Real> > does
  // the main job of initialization.
  this->Swap(&block_mat);
}

template<class Real>
void CuBlockMatrix<Real>::Destroy() {
  data_.Resize(0, 0);
  block_data_.clear();
  num_rows_ = 0;  
  FreeCudaData();
}

// Does *this = alpha A B + beta * *this, discarding elements outside
// the block structure of the *this matrix. 
template<class Real>
void CuBlockMatrix<Real>::AddMatMat(
    BaseFloat alpha,
    const CuMatrix<Real> &A, MatrixTransposeType transA,
    const CuMatrix<Real> &B, MatrixTransposeType transB,
    BaseFloat beta) {
  MatrixIndexT A_num_rows = A.NumRows(), A_num_cols = A.NumCols(),
      A_row_stride = A.Stride(), A_col_stride = 1,
      B_num_rows = B.NumRows(), B_num_cols = B.NumCols(),
      B_row_stride = B.Stride(), B_col_stride = 1;
  if (transA == kTrans) {
    std::swap(A_num_rows, A_num_cols);
    std::swap(A_row_stride, A_col_stride);
  }
  if (transB == kTrans) {
    std::swap(B_num_rows, B_num_cols);
    std::swap(B_row_stride, B_col_stride);
  }
  KALDI_ASSERT(A_num_rows == NumRows() && B_num_cols == NumCols()
               && A_num_cols == B_num_rows);
  if (NumBlocks() == 0) return; // empty matrix.
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    // (x,y,z) dimensions are (block-id, row-of-block, col-of-block)
    // First some logic to choose block dims...
    // we assume (which we can, safely) that CU1DBLOCK is <= the max threads per block.
    int32 x_blocksize = std::min(CU1DBLOCK, NumBlocks()); // x dim corresponds to block-idx.
    int32 max_block_rows = MaxBlockRows(), max_block_cols = MaxBlockCols();
    int32 y_blocksize = max_block_rows;
    while (y_blocksize * x_blocksize > CU1DBLOCK || y_blocksize > CU2DBLOCK)
      y_blocksize--;
    int32 z_blocksize = max_block_cols;
    while (z_blocksize * x_blocksize * y_blocksize > CU1DBLOCK || z_blocksize > CU2DBLOCK)
      z_blocksize--;
    
    dim3 dimBlock(x_blocksize, y_blocksize, z_blocksize);
    dim3 dimGrid(n_blocks(NumBlocks(), x_blocksize),
                 n_blocks(max_block_rows, y_blocksize),
                 n_blocks(max_block_cols, z_blocksize));
    cuda_block_add_mat_mat(dimGrid, dimBlock, cu_data_, NumBlocks(),
                           A.Data(), A_num_cols, A_row_stride, A_col_stride,
                           B.Data(), B_row_stride, B_col_stride, alpha, beta);
    CU_SAFE_CALL(cudaGetLastError());    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());    
  } else
#endif
  {
    int32 row_offset = 0, col_offset = 0;    
    for (MatrixIndexT b = 0; b < NumBlocks(); b++) {
      CuSubMatrix<Real> this_block = Block(b);
      MatrixIndexT this_num_rows = this_block.NumRows(),
          this_num_cols = this_block.NumCols();
      CuSubMatrix<Real> A_part = (transA == kNoTrans ?
                                  A.Range(row_offset, this_num_rows,
                                          0, A.NumCols()) :
                                  A.Range(0, A.NumRows(),
                                          row_offset, this_num_rows)),
          B_part = (transB == kNoTrans ?
                    B.Range(0, B.NumRows(),
                            col_offset, this_num_cols) :
                    B.Range(col_offset, this_num_cols,
                            0, B.NumCols()));
      this_block.AddMatMat(alpha, A_part, transA, B_part, transB, beta);
      row_offset += this_num_rows;
      col_offset += this_num_cols;
    }
    KALDI_ASSERT(row_offset == NumRows() && col_offset == NumCols());
  }
}

template<class Real>
MatrixIndexT CuBlockMatrix<Real>::MaxBlockCols() const {
  MatrixIndexT max_cols = 0;
  for (size_t i = 0; i < block_data_.size(); i++)
    max_cols = std::max(max_cols, block_data_[i].num_cols);
  return max_cols;
}

template<class Real>
MatrixIndexT CuBlockMatrix<Real>::MaxBlockRows() const {
  return data_.NumRows();
}

template<class Real>
void CuBlockMatrix<Real>::CopyFromMat(const CuMatrix<Real> &M) {
  KALDI_ASSERT(NumRows() == M.NumRows() && NumCols() == M.NumCols());
  MatrixIndexT row_offset = 0, col_offset = 0;
  for (MatrixIndexT b = 0; b < NumBlocks(); b++) {
    CuSubMatrix<Real> this_block = Block(b);
    MatrixIndexT this_num_rows = this_block.NumRows(),
        this_num_cols = this_block.NumCols();
    const CuSubMatrix<Real> src(M, row_offset, this_num_rows,
                                col_offset, this_num_cols);
    this_block.CopyFromMat(src);
    row_offset += this_num_rows;
    col_offset += this_num_cols;
  }
  KALDI_ASSERT(row_offset == NumRows() && col_offset == NumCols());
}

/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuBlockMatrix<Real> &mat) {
  bool binary = false;
  mat.Write(out, binary);
  return out;
}
// instantiate the template
template
std::ostream &operator << (std::ostream &out, const CuBlockMatrix<float> &mat);
template 
std::ostream &operator << (std::ostream &out, const CuBlockMatrix<double> &mat);

// Instantiate the class for float and double.
template class CuBlockMatrix<float>;
template class CuBlockMatrix<double>;

} // namespace kaldi

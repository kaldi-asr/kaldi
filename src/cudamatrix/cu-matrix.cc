// cudamatrix/cu-matrix.cc

// Copyright 2009-2012  Karel Vesely
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


#if HAVE_CUDA==1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"
#include "cu-stlvector.h"
#include "cu-math.h"

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
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    MatrixIndexT row_bytes = cols * sizeof(Real);
    size_t pitch;
    cuSafeCall(cudaMallocPitch(reinterpret_cast<void**>(&this->data_), &pitch,
                               row_bytes, rows));
    this->num_rows_ = rows;
    this->num_cols_ = cols; 
    this->stride_ = pitch/sizeof(Real);
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
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    if (this->data_ != NULL) {
      cuSafeCall(cudaFree(this->data_));
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
void CuMatrix<Real>::Swap(Matrix<Real> *mat) {
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) {
      if (mat->num_rows_ != 0) {
        // *this is empty, but mat is nonempty.
        Resize(mat->num_rows_, mat->num_cols_, kUndefined);
        CopyFromMat(*mat);
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
        this->Swap(mat); // copy data in mat to *this, which is now empty.
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


template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_ * sizeof(Real);
    MatrixIndexT src_pitch = src.Stride() * sizeof(Real);
    MatrixIndexT width = src.NumCols() * sizeof(Real);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.data_, src_pitch,
                            width, src.num_rows_, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatD2D",tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(src.Mat());
  }
}



template<typename Real>
void CuMatrixBase<Real>::CopyFromMat(const MatrixBase<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_ && src.NumCols() == num_cols_);
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch,
                            width, src.NumRows(), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatH2D",tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromMat(src);
  }
}


template<typename Real>
void CuMatrixBase<Real>::CopyToMat(MatrixBase<Real> *dst) const {
  KALDI_ASSERT(dst->NumRows() == NumRows() && dst->NumCols() == NumCols());
  
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 

    Timer tim;
   
    MatrixIndexT src_pitch = stride_*sizeof(Real);
    MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
    MatrixIndexT width = NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(dst->data_, dst_pitch, this->data_, src_pitch,
                            width, this->num_rows_, cudaMemcpyDeviceToHost));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
  } else
  #endif
  {
    dst->CopyFromMat(Mat());
  }
}


/*
template<typename Real>
void CuMatrixBase<Real>::CopyRowsFromMat(int32 r, const CuMatrixBase<Real> &src, int32 src_ro, int32 dst_ro) {
  KALDI_ASSERT(r+src_ro <= src.NumRows());
  KALDI_ASSERT(r+dst_ro <= NumRows());
  KALDI_ASSERT(NumCols() == src.NumCols());
   
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
} */



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
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, num_rows_*stride_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero", tim.Elapsed());
  } else
  #endif
  {
    Mat().SetZero();
  }
}



/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrix<Real> &mat) {
  Matrix<Real> temp;
  mat.CopyToMat(&temp);
  out << temp;
  return out;
}



/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real> 
void CuMatrixBase<Real>::Set(Real value) {
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
    Mat().Set(value);
  }
}



template<typename Real> 
void CuMatrixBase<Real>::Add(Real value) { 
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add(dimGrid, dimBlock, data_, value, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Add(value);
  }
}


template<typename Real> 
void CuMatrixBase<Real>::Scale(Real value) { 
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_scale(dimGrid, dimBlock, data_, value, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(value);
  }
}



template<typename Real> 
void CuMatrixBase<Real>::ApplyLog() { 
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
    Mat().ApplyLog();
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulElements(const CuMatrixBase<Real>& A) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(num_cols_ == A.NumCols());
    KALDI_ASSERT(num_rows_ == A.NumRows());
    KALDI_ASSERT(stride_ == A.Stride());
    
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_elements(dimGrid, dimBlock, data_, A.data_, Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().MulElements(A.Mat());
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulColsVec(const CuVectorBase<Real> &scale) {
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_cols_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    cuSafeCall(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().MulColsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::MulRowsVec(const CuVectorBase<Real> &scale) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(scale.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_mul_rows_vec(dimGrid, dimBlock, data_, scale.data_, Dim());
    cuSafeCall(cudaGetLastError());


    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else 
  #endif
  {
    Mat().MulRowsVec(scale.Vec());
  }
}



template<typename Real>
void CuMatrixBase<Real>::DivRowsVec(const CuVectorBase<Real> &div) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(div.Dim() == NumRows());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_div_rows_vec(dimGrid, dimBlock, data_, div.data_, Dim());
    cuSafeCall(cudaGetLastError());

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
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    KALDI_ASSERT(A.NumRows() == NumRows());
    KALDI_ASSERT(A.NumCols() == NumCols());

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_mat(dimGrid, dimBlock, alpha, A.data_, beta, data_, Dim());
    cuSafeCall(cudaGetLastError());

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

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_cols(dimGrid, dimBlock, alpha, col.data_, beta, data_, Dim());
    cuSafeCall(cudaGetLastError());
    
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
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
   
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(NumCols(), CUBLOCK), n_blocks(NumRows(), CUBLOCK));

    cuda_add_vec_to_rows(dimGrid, dimBlock, alpha, row.data_, beta, data_, Dim());
    cuSafeCall(cudaGetLastError());
    
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
void CuMatrixBase<Real>::AddMatMat(
    Real alpha, const CuMatrixBase<Real>& A, MatrixTransposeType transA,
    const CuMatrixBase<Real>& B, MatrixTransposeType transB, Real beta) {
#if HAVE_CUDA==1
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

    cuSafeCall(cublasGetError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().AddMatMat(alpha, A.Mat(), transA, B.Mat(), transB, beta);
  }
}


template<typename Real>
void CuMatrixBase<Real>::Sigmoid(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CUBLOCK), n_blocks(src.NumRows(), CUBLOCK));

    cuda_sigmoid(dimGrid, dimBlock, this->data_, src.data_, src.Dim());
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Mat().Sigmoid(src.Mat());
  }
}


template<typename Real> // Y->this, X->src
void CuMatrixBase<Real>::Softmax(const CuMatrixBase<Real> &src) {
  KALDI_ASSERT(SameDimAndStride(*this, src));
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

#if 1
    // enable 'tree-reduce' functions, 
    //find maximum in each row (tree reduction)
    CuStlVector<int32> max_id;
    src.FindRowMaxId(&max_id); 
    //in each row subtract maximum, apply exp (grid kernel)
    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(src.num_cols_, CUBLOCK), n_blocks(src.num_rows_, CUBLOCK));
    cuda_softmax_part(dimGrid, dimBlock, src.data_, max_id.Data(), this->data_, src.Dim()); 
    //sum the rows to get normalizers (tree reduction) 
    CuVector<Real> sum(src.num_rows_);
    sum.AddColSumMat(1.0, *this, 0.0);
    //divide by normalizers to get posteriors (grid kernel)
    this->DivRowsVec(sum);
#else
    // disable 'tree-reduce' functions, 
    // slower, but can be used for debugging
    size_t dimBlock = CUBLOCK;
    size_t dimGrid  = n_blocks(src.num_rows_, CUBLOCK);

    cuda_softmax(dimGrid, dimBlock, data_, src.data_, src.Dim());
    cuSafeCall(cudaGetLastError());
#endif

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
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CUBLOCK), n_blocks(num_rows_, CUBLOCK));

    cuda_diff_sigmoid(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim());
    cuSafeCall(cudaGetLastError());

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
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(src.NumCols(), CUBLOCK), n_blocks(src.NumRows(), CUBLOCK));

    cuda_tanh(dimGrid, dimBlock, this->data_, src.data_, src.Dim());
    cuSafeCall(cudaGetLastError());
    
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
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK, CUBLOCK);
    dim3 dimGrid(n_blocks(num_cols_, CUBLOCK), n_blocks(num_rows_, CUBLOCK));

    cuda_diff_tanh(dimGrid, dimBlock, data_, diff.data_, value.data_, Dim());
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Mat().DiffTanh(value.Mat(), diff.Mat());
  }
}

template<typename Real>
void CuMatrixBase<Real>::FindRowMaxId(CuStlVector<int32> *id) const {
#if HAVE_CUDA==1 
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

#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(1, CUBLOCK*8);
    dim3 dimGrid(1, n_blocks(tgt.Dim(), CUBLOCK*8));
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

// Instantiate classes CuMatrix and CuMatrixBase for float and double.
template class CuMatrix<float>;
template class CuMatrix<double>;
template class CuMatrixBase<float>;
template class CuMatrixBase<double>;


} // namespace kaldi

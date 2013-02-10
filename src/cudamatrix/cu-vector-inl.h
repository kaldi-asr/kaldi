// cudamatrix/cu-vector-inl.h

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



#ifndef KALDI_CUDAMATRIX_CUVECTOR_INL_H_
#define KALDI_CUDAMATRIX_CUVECTOR_INL_H_

#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
#endif

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"

namespace kaldi {


template<typename Real>
CuVector<Real>::CuVector(const CuVectorBase<Real> &v) {
  Resize(v.dim_);
  CopyFromVec(v);
}

template<typename Real>
CuVector<Real>::CuVector(const VectorBase<Real> &v) {
  Resize(v.dim_);
  CopyFromVec(v);
}

template<typename Real>
void CuVector<Real>::Resize(MatrixIndexT dim, MatrixResizeType t) {
  KALDI_ASSERT(t == kSetZero || t == kUndefined); // Others not implemented
  // yet.
  if (this->dim_ == dim) {
    this->SetZero();
    return;
  }
  if (this->dim_ != 0)
    Destroy();
  if (dim == 0) return;
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    cuSafeCall(cudaMalloc(reinterpret_cast<void**>(&this->data_), dim * sizeof(Real)));
    this->dim_ = dim;
    if (t == kSetZero) this->SetZero();
  } else
#endif
  {
    Vector<Real> vec(dim);
    this->Swap(&vec); 
  }
}

template<typename Real>
void CuVector<Real>::Swap(Vector<Real> *vec) {
#if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) {
    if (this->dim_ == 0) {
      if (vec->dim_ != 0) {
        // *this is empty, but vec is nonempty.
        Resize(vec->dim_, kUndefined);
        CopyFromVec(*vec);
        vec->Resize(0);
      }
      // else both are empty.
    } else { // *this is nonempty.
      if (vec->dim_ != 0) {
        // Both *this and *vec are nonempty.  Recurse to simpler cases.
        // this could be done more efficiently in the case where
        // the size does not change.
        Vector<Real> temp;
        this->Swap(&temp); // now temp is full, *this is empty.
        vec->Swap(&temp); // now vec has data from *this, temp has
        // data from vec.
        Swap(vec); // copy data in vec to *this, which is now empty.
      } else { // *this is full but *vec is empty.
        vec->Resize(this->dim_, kUndefined);
        this->CopyToVec(vec);
        Destroy();
      }
    }
  } else
#endif
  {
    std::swap(vec->data_, this->data_);
    std::swap(vec->dim_, this->dim_);
  }
}

template<typename Real>
void CuVector<Real>::Destroy() {
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
  this->dim_ = 0;
}



template<typename Real>
void CuVectorBase<Real>::CopyFromVec(const CuVectorBase<Real> &src) {
  KALDI_ASSERT(src.Dim() == dim_);
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(data_, src.data_, src.dim_ * sizeof(Real), cudaMemcpyDeviceToDevice));
    CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecD2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(static_cast<void*>(data_), static_cast<void*>(src.data_),
           dim_ * sizeof(Real));
  }
}



template<typename Real>
void CuVectorBase<Real>::CopyFromVec(const VectorBase<Real> &src) {
  KALDI_ASSERT(src.Dim() == dim_);
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    cuSafeCall(cudaMemcpy(data_, src.Data(), src.Dim()*sizeof(Real), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecH2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(static_cast<void*>(data_), static_cast<const void*>(src.Data()),
           dim_ * sizeof(Real));
  }
}



template<typename Real>
void CuVectorBase<Real>::CopyToVec(VectorBase<Real> *dst) const {
  KALDI_ASSERT(dst->Dim() == dim_);
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(dst->Data(), this->data_,
                          dim_*sizeof(Real), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuVector::CopyToVecD2H",tim.Elapsed());
  } else
  #endif
  {
    dst->CopyFromVec(Vec());
  }
}



template<typename Real>
void CuVector<Real>::Read(std::istream &is, bool binary) {
  Vector<BaseFloat> temp;
  temp.Read(is, binary);
  Destroy();
  Swap(&temp);
}



template<typename Real>
void CuVector<Real>::Write(std::ostream &os, bool binary) const {
  Vector<BaseFloat> temp(this->dim_);
  this->CopyToVec(&temp);
  temp.Write(os, binary); 
}



template<typename Real>
void CuVectorBase<Real>::SetZero() {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    KALDI_ASSERT(dim_>0);
    KALDI_ASSERT(data_!=NULL);
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, dim_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuVector::SetZero",tim.Elapsed());
  } else
#endif
  {
    Vec().SetZero();
  }
}



/**
 * Print the vector to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVectorBase<Real> &vec) {
  Vector<Real> temp;
  vec.CopyToVec(&temp);
  out << temp;
  return out;
}




/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real>
void CuVectorBase<Real>::Set(Real value) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    
    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };
    
    cuda_set_const(dimGrid, dimBlock, data_, value, d);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().Set(value);
  }
}



template<typename Real>
void CuVectorBase<Real>::Add(Real value) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_add(dimGrid, dimBlock, data_, value, d);
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Vec().Add(value);
  }
}



template<typename Real>
void CuVectorBase<Real>::Scale(Real value) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_scale(dimGrid, dimBlock, data_, value, d);
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Vec().Scale(value);
  }
}


template<class Real>
void CuVectorBase<Real>::AddVec(Real alpha, const CuVectorBase<Real> &vec,
                                Real beta) {
  KALDI_ASSERT(vec.Dim() == Dim());
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_add_mat(dimGrid, dimBlock, alpha, vec.data_, beta, data_, d);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    if (beta != 1.0) Vec().Scale(beta);
    Vec().AddVec(alpha, vec.Vec());
  }
}



template<typename Real>
void CuVectorBase<Real>::AddRowSumMat(Real alpha, const CuMatrixBase<Real> &mat,
                                      Real beta) {
  KALDI_ASSERT(mat.NumCols() == Dim());
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
   
    CuVector<Real> temp(Dim()); // create a buffer
    temp.SetZero();
    
    MatrixDim d = mat.Dim(); // only stride will be used!
  
    // process per 256 row blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumRows(); block++) {
      // 1st dim ... rows, 2nd dim ... cols
      dim3 dimBlock(256, 1); 
      dim3 dimGrid(1, mat.NumCols());
      int32 offset = block*256*d.stride;

      cuda_add_row_sum_mat(dimGrid, dimBlock, mat.data_ + offset, temp.data_, d);
    }
    
    // process the remainder
    int32 div = mat.NumRows() / 256;
    int32 mod = mat.NumRows() % 256;
    if (mod != 0) {
      // 1st dim ... rows, 2nd dim ... cols
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumCols());
      int32 offset = div*256*d.stride;
      
      cuda_add_row_sum_mat(dimGrid, dimBlock, mat.data_ + offset, temp.data_, d);
    }
    // now we have the sum!
    
    // add buffer rmp to this vector using alpha and beta
    this->AddVec(alpha,temp,beta);

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().AddRowSumMat(alpha, mat.Mat(), beta);
  }
}



template<typename Real>
void CuVectorBase<Real>::AddColSumMat(Real alpha,
                                      const CuMatrixBase<Real> &mat,
                                      Real beta) {
  KALDI_ASSERT(mat.NumRows() == Dim());
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    CuVector<Real> temp(Dim()); // create a buffer
    
    MatrixDim d = mat.Dim(); // only stride will be used!
  
    // process per 256 column blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumCols(); block++) {
      // 1st dim ... cols, 2nd dim ... rows
      dim3 dimBlock(256, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset = block*256;

      cuda_add_col_sum_mat(dimGrid, dimBlock, mat.data_ + offset, temp.data_, d);
    }
    
    // process the remainder
    int32 div = mat.NumCols() / 256;
    int32 mod = mat.NumCols() % 256;
    if (mod != 0) {
      // 1st dim ... cols, 2nd dim ... rows
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset=div*256;
      
      cuda_add_col_sum_mat(dimGrid, dimBlock, mat.data_ +offset, temp.data_, d);
    }
    // now we have the sum!
    
    // add buffer rmp to this vector using alpha and beta
    this->AddVec(alpha, temp, beta);
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Vec().AddColSumMat(alpha, mat.Mat(), beta);
  }
}


 
template<typename Real> 
void CuVectorBase<Real>::InvertElements() {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    
    dim3 dimBlock(CUBLOCK*8, 1);
    dim3 dimGrid(n_blocks(dim_, CUBLOCK*8));
    MatrixDim d = {1, dim_, dim_};

    cuda_invert_elements(dimGrid, dimBlock, data_, d);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().InvertElements();
  }
}

 
} // namespace kaldi

#endif



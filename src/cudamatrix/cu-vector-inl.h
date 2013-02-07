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
const Real* CuVector<Real>::Data() const {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return vec_.Data();
  }
}

template<typename Real>
Real* CuVector<Real>::Data() { 
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return vec_.Data();
  }
}

template<typename Real>
void CuVector<Real>::Resize(MatrixIndexT dim) {
  if (dim_ == dim) {
    SetZero();
    return;
  }
  Destroy();

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    cuSafeCall(cudaMalloc((void**)&data_, dim*sizeof(Real)));
  } else
  #endif
  {
    vec_.Resize(dim);
  }

  dim_ = dim;
  SetZero();
}



template<typename Real>
void CuVector<Real>::Destroy() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    if (NULL != data_) {
      cuSafeCall(cudaFree(data_));
      data_ = NULL;
    }
  } else
  #endif
  {
    vec_.Resize(0);
  }

  dim_ = 0;
}



template<typename Real>
void CuVector<Real>::CopyFromVec(const CuVector<Real> &src) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(data_, src.Data(), src.Dim()*sizeof(Real), cudaMemcpyDeviceToDevice));
    CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecD2D",tim.Elapsed());
  } else
  #endif
  {
    vec_.CopyFromVec(src.vec_);
  }
}



template<typename Real>
void CuVector<Real>::CopyFromVec(const Vector<Real> &src) {
  KALDI_ASSERT(src.Dim() == dim_);
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    cuSafeCall(cudaMemcpy(data_, src.Data(), src.Dim()*sizeof(Real), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecH2D",tim.Elapsed());
  } else
  #endif
  {
    vec_.CopyFromVec(src);
  }
}



template<typename Real>
void CuVector<Real>::CopyToVec(Vector<Real> *dst) const {
  KALDI_ASSERT(dst->Dim() == dim_);


  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(dst->Data(), Data(), dim_*sizeof(Real), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuVector::CopyToVecD2H",tim.Elapsed());
  } else
  #endif
  {
    dst->CopyFromVec(vec_);
  }
}



template<typename Real>
void CuVector<Real>::Read(std::istream &is, bool binary) {
  Vector<BaseFloat> tmp;
  tmp.Read(is, binary);
  Resize(tmp.Dim());
  CopyFromVec(tmp);    
}



template<typename Real>
void CuVector<Real>::Write(std::ostream &os, bool binary) const {
  Vector<BaseFloat> tmp(Dim());
  CopyToVec(&tmp);
  tmp.Write(os, binary); 
}



template<typename Real>
void CuVector<Real>::SetZero() {
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
    vec_.SetZero();
  }
}



/**
 * Print the vector to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVector<Real> &vec) {
  Vector<Real> tmp;
  vec.CopyToVec(&tmp);
  out << tmp;
  return out;
}




/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real>
void CuVector<Real>::Set(Real value) {
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
    vec_.Set(value);
  }
}



template<typename Real>
void CuVector<Real>::Add(Real value) {
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
    vec_.Add(value);
  }
}



template<typename Real>
void CuVector<Real>::Scale(Real value) {
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
    vec_.Scale(value);
  }
}



template<typename Real>
void CuVector<Real>::AddVec(Real alpha, const CuVector<Real> &vec, Real beta) {
  assert(vec.Dim() == Dim());
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_add_mat(dimGrid, dimBlock, alpha, vec.Data(), beta, data_, d);
    cuSafeCall(cudaGetLastError());
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    if (beta != 1.0) { vec_.Scale(beta); }
    vec_.AddVec(alpha, vec.Vec());
  }
}



template<typename Real>
void CuVector<Real>::AddRowSumMat(Real alpha, const CuMatrix<Real> &mat, Real beta) {
  assert(mat.NumCols() == Dim());
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
   
    CuVector<Real> tmp(Dim()); // create a buffer
    tmp.SetZero();
    
    MatrixDim d = mat.Dim(); // only stride will be used!
  
    // process per 256 row blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumRows(); block++) {
      // 1st dim ... rows, 2nd dim ... cols
      dim3 dimBlock(256, 1); 
      dim3 dimGrid(1, mat.NumCols());
      int32 offset = block*256*d.stride;

      cuda_add_row_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    
    // process the remainder
    int32 div = mat.NumRows() / 256;
    int32 mod = mat.NumRows() % 256;
    if (mod != 0) {
      // 1st dim ... rows, 2nd dim ... cols
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumCols());
      int32 offset = div*256*d.stride;
      
      cuda_add_row_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    // now we have the sum!
    
    // add buffer rmp to this vector using alpha and beta
    this->AddVec(alpha,tmp,beta);

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
    vec_.AddRowSumMat(alpha, mat.Mat(), beta);
}



template<typename Real>
void CuVector<Real>::AddColSumMat(Real alpha, const CuMatrix<Real> &mat, Real beta) {
  assert(mat.NumRows() == Dim());
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    CuVector<Real> tmp(Dim()); // create a buffer
    
    MatrixDim d = mat.Dim(); // only stride will be used!
  
    // process per 256 column blocks 
    for(int32 block=0; (block+1)*256 <= mat.NumCols(); block++) {
      // 1st dim ... cols, 2nd dim ... rows
      dim3 dimBlock(256, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset = block*256;

      cuda_add_col_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    
    // process the remainder
    int32 div = mat.NumCols() / 256;
    int32 mod = mat.NumCols() % 256;
    if (mod != 0) {
      // 1st dim ... cols, 2nd dim ... rows
      dim3 dimBlock(mod, 1);
      dim3 dimGrid(1, mat.NumRows());
      int32 offset=div*256;
      
      cuda_add_col_sum_mat(dimGrid, dimBlock, mat.Data()+offset, tmp.Data(), d);
    }
    // now we have the sum!
    
    // add buffer rmp to this vector using alpha and beta
    this->AddVec(alpha,tmp,beta);
    
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
    vec_.AddColSumMat(alpha, mat.Mat(), beta);
}


 
template<typename Real> 
void CuVector<Real>::InvertElements() {
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
    vec_.InvertElements();
  }
}

 
} // namespace kaldi

#endif



// cudamatrix/cu-stlvector-inl.h

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



#ifndef KALDI_CUDAMATRIX_CUSTLVECTOR_INL_H_
#define KALDI_CUDAMATRIX_CUSTLVECTOR_INL_H_

#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
  #include "cudamatrix/cu-common.h"
  #include "cudamatrix/cu-device.h"
  #include "cudamatrix/cu-kernels.h"
#endif

#include "util/timer.h"

namespace kaldi {


template<typename IntType>
const IntType* CuStlVector<IntType>::Data() const {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return &vec_.front();
  }
}



template<typename IntType>
IntType* CuStlVector<IntType>::Data() { 
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return &vec_.front();
  }
}



template<typename IntType>
CuStlVector<IntType>& CuStlVector<IntType>::Resize(size_t dim) {
  if (dim_ == dim) {
    // SetZero();
    return *this;
  }

  Destroy();

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    cuSafeCall(cudaMalloc((void**)&data_, dim*sizeof(IntType)));
  } else
  #endif
  {
    vec_.resize(dim);
  }

  dim_ = dim;
  SetZero();

  return *this;
}



template<typename IntType>
void CuStlVector<IntType>::Destroy() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    if (NULL != data_) {
      cuSafeCall(cudaFree(data_));
      data_ = NULL;
    }
  } else
  #endif
  {
    vec_.resize(0);
  }

  dim_ = 0;
}



template<typename IntType>
CuStlVector<IntType>& CuStlVector<IntType>::CopyFromVec(const std::vector<IntType> &src) {
  Resize(src.size());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    cuSafeCall(cudaMemcpy(data_, &src.front(), src.size()*sizeof(IntType), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuStlVector::CopyFromVecH2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(&vec_.front(), &src.front(), src.size()*sizeof(IntType));
  }
  return *this;
}



template<typename IntType>
void CuStlVector<IntType>::CopyToVec(std::vector<IntType> *dst) const {
  if (dst->size() != dim_) {
    dst->resize(dim_);
  }

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(&dst->front(), Data(), dim_*sizeof(IntType), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuStlVector::CopyToVecD2H",tim.Elapsed());
  } else
  #endif
  {
    memcpy(&dst->front(), &vec_.front(), dim_*sizeof(IntType));
  }
}



template<typename IntType>
void CuStlVector<IntType>::SetZero() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, dim_*sizeof(IntType)));
    CuDevice::Instantiate().AccuProfile("CuStlVector::SetZero",tim.Elapsed());
  } else
  #endif
  {
    vec_.assign(dim_, 0);
  }
}



/**
 * Print the vector to stream
 */
template<typename IntType>
std::ostream &operator << (std::ostream &out, const CuStlVector<IntType> &vec) {
  std::vector<IntType> tmp;
  vec.CopyToVec(&tmp);
  out << "[";
  for(int32 i=0; i<tmp.size(); i++) {
    out << " " << tmp[i];
  }
  out << " ]\n";
  return out;
}



/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<> 
inline void CuStlVector<int32>::Set(int32 value) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaI32_set_const(dimGrid, dimBlock, data_, value, d);
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    vec_.assign(vec_.size(), value);
  }
}


} // namespace kaldi

#endif



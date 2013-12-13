// cudamatrix/cu-array-inl.h

// Copyright 2009-2012  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CUDAMATRIX_CU_ARRAY_INL_H_
#define KALDI_CUDAMATRIX_CU_ARRAY_INL_H_

#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#endif

#include "util/timer.h"

namespace kaldi {


template<typename T>
void CuArray<T>::Resize(MatrixIndexT dim, MatrixResizeType resize_type) {
  KALDI_ASSERT((resize_type == kSetZero || resize_type == kUndefined) && dim >= 0);
  if (dim_ == dim) {
    if (resize_type == kSetZero)
      SetZero();
    return;
  }

  Destroy();

  if (dim == 0) return;
  
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    this->data_ = static_cast<T*>(CuDevice::Instantiate().Malloc(dim * sizeof(T)));
    this->dim_ = dim;
    if (resize_type == kSetZero) this->SetZero();
    CuDevice::Instantiate().AccuProfile("CuArray::Resize", tim.Elapsed());    
  } else
#endif
  {
    data_ = static_cast<T*>(malloc(dim * sizeof(T)));
    // We allocate with malloc because we don't want constructors being called.
    // We basically ignore memory alignment issues here-- we assume the malloc
    // implementation is forgiving enough that it will automatically align on
    // sensible boundaries.
    if (data_ == 0)
      KALDI_ERR << "Memory allocation failed when initializing CuVector "
                << "with dimension " << dim << " object size in bytes: "
                << sizeof(T);
  }

  dim_ = dim;
  if (resize_type == kSetZero)
    SetZero();
}

template<typename T>
void CuArray<T>::Destroy() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    if (data_ != NULL) {
      CuDevice::Instantiate().Free(this->data_);
    }
  } else
#endif
  {
    if (data_ != NULL)
      free(data_);
  }
  dim_ = 0;
  data_ = NULL;
}


template<typename T>
void CuArray<T>::CopyFromVec(const std::vector<T> &src) {
  Resize(src.size(), kUndefined);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    CU_SAFE_CALL(cudaMemcpy(data_, &src.front(), src.size()*sizeof(T), cudaMemcpyHostToDevice));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    memcpy(data_, &src.front(), src.size()*sizeof(T));
  }
}



template<typename T>
void CuArray<T>::CopyToVec(std::vector<T> *dst) const {
  if (static_cast<MatrixIndexT>(dst->size()) != dim_) {
    dst->resize(dim_);
  }
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    CU_SAFE_CALL(cudaMemcpy(&dst->front(), Data(), dim_*sizeof(T), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuArray::CopyToVecD2H", tim.Elapsed());
  } else
#endif
  {
    memcpy(&dst->front(), data_, dim_*sizeof(T));
  }
}


template<typename T>
void CuArray<T>::SetZero() {
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    CU_SAFE_CALL(cudaMemset(data_, 0, dim_ * sizeof(T)));
    CuDevice::Instantiate().AccuProfile("CuArray::SetZero", tim.Elapsed());
  } else
#endif
  {
    memset(static_cast<void*>(data_), 0, dim_ * sizeof(T));
  }
}



/**
 * Print the vector to stream
 */
template<typename T>
std::ostream &operator << (std::ostream &out, const CuArray<T> &vec) {
  std::vector<T> tmp;
  vec.CopyToVec(&tmp);
  out << "[";
  for(int32 i=0; i<tmp.size(); i++) {
    out << " " << tmp[i];
  }
  out << " ]\n";
  return out;
}


template<class T> 
inline void CuArray<T>::Set(const T &value) {
  // This is not implemented yet, we'll do so if it's needed.
  KALDI_ERR << "CuArray<T>::Set not implemented yet for this type.";
}

template<> 
inline void CuArray<int32>::Set(const int32 &value) {
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CU2DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU2DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaI32_set_const(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    for (int32 i = 0; i < dim_; i++)
      data_[i] = value;
  }
}

template<typename T>
void CuArray<T>::CopyFromArray(const CuArray<T> &src) {
  this->Resize(src.Dim(), kUndefined);
  if (dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CU_SAFE_CALL(cudaMemcpy(this->data_, src.data_, dim_ * sizeof(T),
                            cudaMemcpyDeviceToDevice));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    memcpy(this->data_, src.data_, dim_ * sizeof(T));
  }
}


} // namespace kaldi

#endif

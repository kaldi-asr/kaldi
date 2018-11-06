// cudamatrix/cu-array-inl.h

// Copyright 2009-2016  Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2017  Shiyin Kang


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

#include <algorithm>

#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#endif

#include "base/timer.h"

namespace kaldi {


template<typename T>
void CuArray<T>::Resize(MatrixIndexT dim, MatrixResizeType resize_type) {
  KALDI_ASSERT((resize_type == kSetZero || resize_type == kUndefined) && dim >= 0);
  if (this->dim_ == dim) {
    if (resize_type == kSetZero)
      this->SetZero();
    return;
  }

  Destroy();

  if (dim == 0) return;

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    this->data_ = static_cast<T*>(CuDevice::Instantiate().Malloc(dim * sizeof(T)));
    this->dim_ = dim;
    if (resize_type == kSetZero) this->SetZero();
    CuDevice::Instantiate().AccuProfile("CuArray::Resize", tim);
  } else
#endif
  {
    this->data_ = static_cast<T*>(malloc(dim * sizeof(T)));
    // We allocate with malloc because we don't want constructors being called.
    // We basically ignore memory alignment issues here-- we assume the malloc
    // implementation is forgiving enough that it will automatically align on
    // sensible boundaries.
    if (this->data_ == 0)
      KALDI_ERR << "Memory allocation failed when initializing CuVector "
                << "with dimension " << dim << " object size in bytes: "
                << sizeof(T);
  }

  this->dim_ = dim;
  if (resize_type == kSetZero)
    this->SetZero();
}

template<typename T>
void CuArray<T>::Destroy() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->data_ != NULL) {
      CuDevice::Instantiate().Free(this->data_);
    }
  } else
#endif
  {
    if (this->data_ != NULL)
      free(this->data_);
  }
  this->dim_ = 0;
  this->data_ = NULL;
}


template<typename T>
void CuArrayBase<T>::CopyFromVec(const std::vector<T> &src) {
  KALDI_ASSERT(dim_ == src.size());
  if (src.empty())
    return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(
        cudaMemcpy(data_, &src.front(), src.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    memcpy(data_, &src.front(), src.size() * sizeof(T));
  }
}

template<typename T>
void CuArray<T>::CopyFromVec(const std::vector<T> &src) {
  Resize(src.size(), kUndefined);
  if (src.empty()) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(cudaMemcpy(this->data_, &src.front(), src.size()*sizeof(T), cudaMemcpyHostToDevice));
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    memcpy(this->data_, &src.front(), src.size()*sizeof(T));
  }
}


template<typename T>
void CuArray<T>::CopyFromArray(const CuArrayBase<T> &src) {
  this->Resize(src.Dim(), kUndefined);
  if (this->dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(cudaMemcpyAsync(this->data_, src.data_, this->dim_ * sizeof(T),
                                 cudaMemcpyDeviceToDevice,
                                 cudaStreamPerThread));
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    memcpy(this->data_, src.data_, this->dim_ * sizeof(T));
  }
}

template<typename T>
void CuArrayBase<T>::CopyFromArray(const CuArrayBase<T> &src) {
  KALDI_ASSERT(src.Dim() == Dim());
  if (dim_ == 0)
    return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(
      cudaMemcpyAsync(this->data_, src.data_, dim_ * sizeof(T),
                      cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    memcpy(this->data_, src.data_, dim_ * sizeof(T));
  }
}


template<typename T>
void CuArrayBase<T>::CopyToVec(std::vector<T> *dst) const {
  if (static_cast<MatrixIndexT>(dst->size()) != this->dim_) {
    dst->resize(this->dim_);
  }
  if (this->dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(cudaMemcpy(&dst->front(), Data(), this->dim_ * sizeof(T), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuArray::CopyToVecD2H", tim);
  } else
#endif
  {
    memcpy(&dst->front(), this->data_, this->dim_ * sizeof(T));
  }
}


template<typename T>
void CuArrayBase<T>::CopyToHost(T *dst) const {
  if (this->dim_ == 0) return;
  KALDI_ASSERT(dst != NULL);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(cudaMemcpy(dst, Data(), this->dim_ * sizeof(T), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuArray::CopyToVecD2H", tim);
  } else
#endif
  {
    memcpy(dst, this->data_, this->dim_ * sizeof(T));
  }
}


template<typename T>
void CuArrayBase<T>::SetZero() {
  if (this->dim_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    CU_SAFE_CALL(cudaMemset(this->data_, 0, this->dim_ * sizeof(T)));
    CuDevice::Instantiate().AccuProfile("CuArray::SetZero", tim);
  } else
#endif
  {
    memset(static_cast<void*>(this->data_), 0, this->dim_ * sizeof(T));
  }
}


template<class T>
void CuArrayBase<T>::Set(const T &value) {
  // This is not implemented yet, we'll do so if it's needed.
  KALDI_ERR << "CuArray<T>::Set not implemented yet for this type.";
}
// int32 specialization implemented in 'cudamatrix/cu-array.cc',
template<>
void CuArrayBase<int32>::Set(const int32 &value);


template<class T>
void CuArrayBase<T>::Sequence(const T base) {
  // This is not implemented yet, we'll do so if it's needed.
  KALDI_ERR << "CuArray<T>::Sequence not implemented yet for this type.";
}
// int32 specialization implemented in 'cudamatrix/cu-array.cc',
template<>
void CuArrayBase<int32>::Sequence(const int32 base);


template<class T>
void CuArrayBase<T>::Add(const T &value) {
  // This is not implemented yet, we'll do so if it's needed.
  KALDI_ERR << "CuArray<T>::Add not implemented yet for this type.";
}
// int32 specialization implemented in 'cudamatrix/cu-array.cc',
template<>
void CuArrayBase<int32>::Add(const int32 &value);


template<class T>
inline T CuArrayBase<T>::Min() const {
  KALDI_ASSERT(this->Dim() > 0);
#if HAVE_CUDA == 1
  CuTimer tim;
#endif
  std::vector<T> tmp(Dim());
  CopyToVec(&tmp);
  T ans = *std::min_element(tmp.begin(), tmp.end());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  }
#endif
  return ans;
}


template<class T>
inline T CuArrayBase<T>::Max() const {
  KALDI_ASSERT(this->Dim() > 0);
#if HAVE_CUDA == 1
  CuTimer tim;
#endif
  std::vector<T> tmp(Dim());
  CopyToVec(&tmp);
  T ans = *std::max_element(tmp.begin(), tmp.end());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  }
#endif
  return ans;
}


template<typename T>
void CuArray<T>::Read(std::istream& in, bool binary) {
  std::vector<T> tmp;
  ReadIntegerVector(in, binary, &tmp);
  (*this) = tmp;
}

template<typename T>
void CuArray<T>::Write(std::ostream& out, bool binary) const {
  std::vector<T> tmp(this->Dim());
  this->CopyToVec(&tmp);
  WriteIntegerVector(out, binary, tmp);
}


template<typename T>
CuSubArray<T>::CuSubArray(const CuArrayBase<T> &src,
                          MatrixIndexT offset,
                          MatrixIndexT dim) {
  KALDI_ASSERT(offset >= 0 && dim >= 0 &&
               offset + dim <= src.Dim());
  this->data_ = src.data_ + offset;
  this->dim_ = dim;
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

template <typename T>
void CuArray<T>::Swap(CuArray<T> *other) {
  std::swap(this->dim_, other->dim_);
  std::swap(this->data_, other->data_);
}


} // namespace kaldi

#endif

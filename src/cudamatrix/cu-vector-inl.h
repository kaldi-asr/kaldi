
#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
#endif

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {

template<typename _ElemT>
const _ElemT* CuVector<_ElemT>::Data() const {
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return vec_.Data();
  }
}


template<typename _ElemT>
_ElemT* CuVector<_ElemT>::Data() { 
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return vec_.Data();
  }
}


template<typename _ElemT>
CuVector<_ElemT>& CuVector<_ElemT>::Resize(size_t dim) {
  if(dim_ == dim) {
    //SetZero();
    return *this;
  }

  Destroy();

  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    cuSafeCall(cudaMalloc((void**)&data_, dim*sizeof(_ElemT)));
  } else
  #endif
  {
    vec_.Resize(dim);
  }

  dim_ = dim;
  SetZero();

  return *this;
}


template<typename _ElemT>
void CuVector<_ElemT>::Destroy() {
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    if(NULL != data_) {
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


template<typename _ElemT>
CuVector<_ElemT>& CuVector<_ElemT>::CopyFromVec(const CuVector<_ElemT>& src) {
  Resize(src.Dim());
  
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(data_, src.Data(), src.Dim()*sizeof(_ElemT), cudaMemcpyDeviceToDevice));
    CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecD2D",tim.Elapsed());
  } else
  #endif
  {
    vec_.CopyFromVec(src.vec_);
  }

  return *this;
}


template<typename _ElemT>
CuVector<_ElemT>& CuVector<_ElemT>::CopyFromVec(const Vector<_ElemT>& src) {
  Resize(src.Dim());

  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    cuSafeCall(cudaMemcpy(data_, src.Data(), src.Dim()*sizeof(_ElemT), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecH2D",tim.Elapsed());
  } else
  #endif
  {
    vec_.CopyFromVec(src);
  }
  return *this;
}


template<typename _ElemT>
void CuVector<_ElemT>::CopyToVec(Vector<_ElemT>& dst) const {
  if(dst.Dim() != dim_) {
    dst.Resize(dim_);
  }

  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(dst.Data(), Data(), dim_*sizeof(_ElemT), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuVector::CopyToVecD2H",tim.Elapsed());
  } else
  #endif
  {
    dst.CopyFromVec(vec_);
  }
}


template<typename _ElemT>
CuVector<_ElemT>& CuVector<_ElemT>::CopyFromVec(const std::vector<_ElemT>& src) {
  Resize(src.size());

  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    cuSafeCall(cudaMemcpy(data_, &src.front(), src.size()*sizeof(_ElemT), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecH2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(vec_.Data(),&src.front(),src.size()*sizeof(_ElemT));
  }
  return *this;
}


template<typename _ElemT>
void CuVector<_ElemT>::CopyToVec(std::vector<_ElemT>& dst) const {
  if(dst.Dim() != dim_) {
    dst.resize(dim_);
  }

  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(&dst.front(), Data(), dim_*sizeof(_ElemT), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuVector::CopyToVecD2H",tim.Elapsed());
  } else
  #endif
  {
    memcpy(&dst.front(), vec_.Data(), dim_*sizeof(_ElemT));
  }
}


template<typename _ElemT>
void CuVector<_ElemT>::Read(std::istream& is, bool binary) {
  Vector<BaseFloat> tmp;
  tmp.Read(is,binary);
  CopyFromVec(tmp);    
}


template<typename _ElemT>
void CuVector<_ElemT>::Write(std::ostream& os, bool binary) const {
  Vector<BaseFloat> tmp;
  CopyToVec(tmp);
  tmp.Write(os,binary); 
}


template<typename _ElemT>
void CuVector<_ElemT>::SetZero() {
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, dim_*sizeof(_ElemT)));
    CuDevice::Instantiate().AccuProfile("CuVector::SetZero",tim.Elapsed());
  } else
  #endif
  {
    vec_.SetZero();
  }
}



/// Prints the vector to stream
template<typename _ElemT>
std::ostream& operator << (std::ostream& out, const CuVector<_ElemT>& vec) {
  Vector<_ElemT> tmp;
  vec.CopyToVec(tmp);
  out << tmp;
  return out;
}


 
/*
 * declare the float specialized methods
 */
template<> void CuVector<float>::Set(float value);
template<> void CuVector<float>::AddVec(float alpha, const CuVector<float>& vec, float beta);
template<> void CuVector<float>::AddColSum(float alpha, const CuMatrix<float>& mat, float beta);

 
} //namespace kaldi




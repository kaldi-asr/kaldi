
#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
  #include "cudamatrix/cu-common.h"
  #include "cudamatrix/cu-device.h"
#endif

#include "util/timer.h"

namespace kaldi {


template<typename _ElemT>
const _ElemT* CuStlVector<_ElemT>::Data() const {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return &vec_.front();
  }
}


template<typename _ElemT>
_ElemT* CuStlVector<_ElemT>::Data() { 
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_; 
  } else
  #endif
  {
    return &vec_.front();
  }
}


template<typename _ElemT>
CuStlVector<_ElemT>& CuStlVector<_ElemT>::Resize(size_t dim) {
  if (dim_ == dim) {
    // SetZero();
    return *this;
  }

  Destroy();

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    cuSafeCall(cudaMalloc((void**)&data_, dim*sizeof(_ElemT)));
  } else
  #endif
  {
    vec_.resize(dim);
  }

  dim_ = dim;
  SetZero();

  return *this;
}


template<typename _ElemT>
void CuStlVector<_ElemT>::Destroy() {
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



template<typename _ElemT>
CuStlVector<_ElemT>& CuStlVector<_ElemT>::CopyFromVec(const std::vector<_ElemT>& src) {
  Resize(src.size());

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    cuSafeCall(cudaMemcpy(data_, &src.front(), src.size()*sizeof(_ElemT), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuStlVector::CopyFromVecH2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(&vec_.front(), &src.front(), src.size()*sizeof(_ElemT));
  }
  return *this;
}


template<typename _ElemT>
void CuStlVector<_ElemT>::CopyToVec(std::vector<_ElemT>* dst) const {
  if (dst->size() != dim_) {
    dst->resize(dim_);
  }

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemcpy(&dst->front(), Data(), dim_*sizeof(_ElemT), cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuStlVector::CopyToVecD2H",tim.Elapsed());
  } else
  #endif
  {
    memcpy(&dst->front(), &vec_.front(), dim_*sizeof(_ElemT));
  }
}


template<typename _ElemT>
void CuStlVector<_ElemT>::SetZero() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, dim_*sizeof(_ElemT)));
    CuDevice::Instantiate().AccuProfile("CuStlVector::SetZero",tim.Elapsed());
  } else
  #endif
  {
    vec_.assign(dim_, 0);
  }
}


/// Prints the vector to stream
template<typename _ElemT>
std::ostream& operator << (std::ostream& out, const CuStlVector<_ElemT>& vec) {
  std::vector<_ElemT> tmp;
  vec.CopyToVec(&tmp);
  out << "[";
  for(int32 i=0; i<tmp.size(); i++) {
    out << " " << tmp[i];
  }
  out << " ]\n";
  return out;
}



/*
 * declare the specialized methods
 */
template<> void CuStlVector<int32>::Set(int32 value);


} // namespace kaldi

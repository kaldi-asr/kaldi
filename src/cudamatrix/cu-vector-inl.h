
#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
#endif

#include "util/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"

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
CuVector<Real>& CuVector<Real>::Resize(size_t dim) {
  if (dim_ == dim) {
    // SetZero();
    return *this;
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

  return *this;
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
CuVector<Real>& CuVector<Real>::CopyFromVec(const CuVector<Real> &src) {
  Resize(src.Dim());
  
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

  return *this;
}


template<typename Real>
CuVector<Real>& CuVector<Real>::CopyFromVec(const Vector<Real> &src) {
  Resize(src.Dim());

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
  return *this;
}


template<typename Real>
void CuVector<Real>::CopyToVec(Vector<Real> *dst) const {
  if (dst->Dim() != dim_) {
    dst->Resize(dim_);
  }

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
  CopyFromVec(tmp);    
}


template<typename Real>
void CuVector<Real>::Write(std::ostream &os, bool binary) const {
  Vector<BaseFloat> tmp;
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



/// Prints the vector to stream
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVector<Real> &vec) {
  Vector<Real> tmp;
  vec.CopyToVec(&tmp);
  out << tmp;
  return out;
}


 
/*
 * declare the float specialized methods
 */
template<> void CuVector<float>::Set(float value);
template<> void CuVector<float>::AddVec(float alpha, const CuVector<float> &vec, float beta);
template<> void CuVector<float>::AddColSum(float alpha, const CuMatrix<float> &mat, float beta);
template<> void CuVector<float>::AddRowSum(float alpha, const CuMatrix<float> &mat, float beta);
template<> void CuVector<float>::InvertElements();
 
} // namespace kaldi




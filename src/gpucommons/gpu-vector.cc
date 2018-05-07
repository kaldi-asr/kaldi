#include "gpucommons/gpu-vector.h"

namespace kaldi{

template<typename Real>
_GPUVector<Real>::_GPUVector() {}

template<typename Real>
_GPUVector<Real>::_GPUVector(Vector<Real> &M) : dim_(M.Dim()){
  const size_t m_dim = M.SizeInBytes() / sizeof(Real);
  Real* m_data = M.Data();
  thrust::copy(m_data, m_data + m_dim, data_.begin());
  data = data_.data().get();
}

template<typename Real>
_GPUVector<Real>::_GPUVector(const Vector<Real> &M) : dim_(M.Dim()){
  const size_t m_dim = M.SizeInBytes() / sizeof(Real);
  Real* m_data = M.Data();
  thrust::copy(m_data, m_data + m_dim, data_.begin());
  data = data_.data().get();
}

template<typename Real>
int32 _GPUVector<Real>::Dim() const {
  return dim_;
}

template<typename Real>
int32 _GPUVector<Real>::Index(int32 idx) const {
  return idx;
}

}

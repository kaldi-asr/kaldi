#include "gpucommons/gpu-vector.h"

namespace kaldi{

template<typename Real>
GPUVector<Real>::GPUVector() {}

template<typename Real>
GPUVector<Real>::GPUVector(Vector<Real> &M) : dim_(M.Dim()){
  const size_t m_dim = M.SizeInBytes() / sizeof(Real);
  Real* m_data = M.Data();
  thrust::copy(m_data, m_data + m_dim, data_.begin());
  data = data_.data().get();
}

template<typename Real>
GPUVector<Real>::GPUVector(const Vector<Real> &M) : dim_(M.Dim()){
  const size_t m_dim = M.SizeInBytes() / sizeof(Real);
  Real* m_data = M.Data();
  thrust::copy(m_data, m_data + m_dim, data_.begin());
  data = data_.data().get();
}

__host__ __device__
template<typename Real>
int32 GPUVector<Real>::Dim() const {
  return dim_;
}

__host__ __device__
template<typename Real>
int32 GPUVector<Real>::Index(int32 idx) const {
  return idx;
}

}

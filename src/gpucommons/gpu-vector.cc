#include "gpucommons/gpu-vector.h"

namespace kaldi{

template<typename Real>
struct _GPUVector{
  thrust::device_vector<Real> data_;
  int32 dim_;
  Real* data;

  int32 Dim() const { return dim_; }

  int32 Index(int32 idx) const { return idx; }

  _GPUVector(Vector<Real> &M) : dim_(M.Dim())
  {
    const size_t m_dim = M.SizeInBytes() / sizeof(Real);
    Real* m_data = M.Data();
    thrust::copy(m_data, m_data + m_dim, data_.begin());
    data = data_.data().get();
  }

  _GPUVector(const Vector<Real> &M) : dim_(M.Dim())
  {
    const size_t m_dim = M.SizeInBytes() / sizeof(Real);
    Real* m_data = M.Data();
    thrust::copy(m_data, m_data + m_dim, data_.begin());
    data = data_.data().get();
  }
};

}

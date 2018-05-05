#ifndef GPU_VECTOR_HPP
#define GPU_VECTOR_HPP

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace kaldi{


template<typename Real>
struct GPUVector{
  thrust::device_vector<Real> data_;
  int32 dim_;

  int32 Dim() const { return dim_; }

  int32 Index(int32 idx) const { return idx; }

  GPUVector(Vector<Real> &M) : dim_(M.Dim())
  {
    const size_t m_dim = M.SizeInBytes() / sizeof(Real);
    Real* m_data = M.Data();
    thrust::copy(m_data, m_data + m_dim, data_.begin());
  }

  GPUVector(const Vector<Real> &M) : dim_(M.Dim())
  {
    const size_t m_dim = M.SizeInBytes() / sizeof(Real);
    Real* m_data = M.Data();
    thrust::copy(m_data, m_data + m_dim, data_.begin());
  }
};

}

#endif
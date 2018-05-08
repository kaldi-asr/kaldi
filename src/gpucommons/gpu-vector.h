#ifndef GPU_VECTOR_H
#define GPU_VECTOR_H

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "matrix/kaldi-vector.h"
#include "base/kaldi-common.h"

namespace kaldi{

template<typename Real>
struct GPUVector{
  thrust::device_vector<Real> data_;
  int32 dim_;
  Real* data;

  __host__ __device__
  int32 Dim() const;

  __host__ __device__
  int32 Index(int32 idx) const;

  GPUVector();
  GPUVector(Vector<Real> &M);
  GPUVector(const Vector<Real> &M);
};

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

#endif

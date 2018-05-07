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
}

#endif

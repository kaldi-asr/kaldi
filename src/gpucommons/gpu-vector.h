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
struct _GPUVector{
  thrust::device_vector<Real> data_;
  int32 dim_;
  Real* data;

  int32 Dim() const { return dim_; }

  int32 Index(int32 idx) const { return idx; }

  _GPUVector(Vector<Real> &M);
  _GPUVector(const Vector<Real> &M);
};

template<typename Real>
using GPUVector = _GPUVector<Real>;

}

#endif

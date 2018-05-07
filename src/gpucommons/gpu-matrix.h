#ifndef KALDI_GPUCOMMONS_GPU_MATRIX_H
#define KALDI_GPUCOMMONS_GPU_MATRIX_H

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "matrix/kaldi-matrix.h"
#include "base/kaldi-common.h"

namespace kaldi{

template<typename Real>
struct GPUMatrix{
  thrust::device_vector<Real> data_;
  int32 numrows_, numcols_, stride_;
  Real* data;

  __host__ __device__
  int32 NumRows() const;

  __host__ __device__
  int32 NumCols() const;

  __host__ __device__
  int32 Stride() const;

  __host__ __device__
  int32 Index(int32 r, int32 c) const;

  GPUMatrix();
  GPUMatrix(Matrix<Real> &M);
  GPUMatrix(const Matrix<Real> &M);
};

}

#endif

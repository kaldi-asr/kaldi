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
struct _GPUMatrix{
  thrust::device_vector<Real> data_;
  int32 numrows_, numcols_, stride_;
  Real* data;

  int32 NumRows() const;
  int32 NumCols() const;
  int32 Stride() const;
  int32 Index(int32 r, int32 c) const;

  _GPUMatrix();
  _GPUMatrix(Matrix<Real> &M);
  _GPUMatrix(const Matrix<Real> &M);
};

template<typename Real>
using GPUMatrix = _GPUMatrix<Real>;

}

#endif

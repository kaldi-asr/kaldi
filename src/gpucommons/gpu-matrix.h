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

template<typename Real>
GPUMatrix<Real>::GPUMatrix() {}

template<typename Real>
GPUMatrix<Real>::GPUMatrix(Matrix<Real> &M) :
  numcols_(M.NumCols()),
  numrows_(M.NumRows()),
  stride_(M.Stride())
{
  const size_t m_dim = M.SizeInBytes() / sizeof(Real);
  Real* m_data = M.Data();
  data_.resize(m_dim);
  thrust::copy(m_data, m_data + m_dim, data_.begin());
  data = data_.data().get();
}

template<typename Real>
GPUMatrix<Real>::GPUMatrix(const Matrix<Real> &M) :
  numcols_(M.NumCols()),
  numrows_(M.NumRows()),
  stride_(M.Stride())
{
  const size_t m_dim = M.SizeInBytes() / sizeof(Real);
  const Real* m_data = M.Data();
  data_.resize(m_dim);
  thrust::copy(m_data, m_data + m_dim, data_.begin());
  data = data_.data().get();
}

template<typename Real>
int32 GPUMatrix<Real>::NumRows() const { return numrows_; }

template<typename Real>
int32 GPUMatrix<Real>::NumCols() const { return numcols_; }

template<typename Real>
int32 GPUMatrix<Real>::Stride() const { return stride_; }

template<typename Real>
int32 GPUMatrix<Real>::Index(int32 r, int32 c) const { return r * stride_ + c; }


}

#endif

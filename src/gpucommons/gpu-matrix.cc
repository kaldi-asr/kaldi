#include "gpucommons/gpu-matrix.h"

namespace kaldi{

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
  Real* m_data = M.Data();

  thrust::copy(m_data, m_data + m_dim, data_.begin());
  data = data_.data().get();
}

__host__ __device__
template<typename Real>
int32 GPUMatrix<Real>::NumRows() const { return numrows_; }

__host__ __device__
template<typename Real>
int32 GPUMatrix<Real>::NumCols() const { return numcols_; }

__host__ __device__
template<typename Real>
int32 GPUMatrix<Real>::Stride() const { return stride_; }

__host__ __device__
template<typename Real>
int32 GPUMatrix<Real>::Index(int32 r, int32 c) const { return r * stride_ + c; }

}

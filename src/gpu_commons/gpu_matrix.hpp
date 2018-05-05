#ifndef GPU_MATRIX_HPP
#define GPU_MATRIX_HPP

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace kaldi{


template<typename Real>
struct GPUMatrix{
  thrust::device_vector<Real> data_;
  int32 numrows_, numcols_, stride_;
  Real* data;

  int32 NumRows() const { return numrows_; }
  int32 NumCols() const { return numcols_; }
  int32 Stride() const { return stride_; }

  int32 Index(int32 r, int32 c) const{
    return r * stride_ + c;
  }

  GPUMatrix(Matrix<Real> &M) : 
    numcols_(M.NumCols()),  
    numrows_(M.NumRows()), 
    stride_(M.Stride()) 
  {
    const size_t m_dim = M.SizeInBytes() / sizeof(Real);
    Real* m_data = M.Data();

    thrust::copy(m_data, m_data + m_dim, data_.begin());
    data = data_.data().get();
  }

  GPUMatrix(const Matrix<Real> &M)
    numcols_(M.NumCols()),  
    numrows_(M.NumRows()), 
    stride_(M.Stride()) 
  {
    const size_t m_dim = M.SizeInBytes() / sizeof(Real);
    Real* m_data = M.Data();

    thrust::copy(m_data, m_data + m_dim, data_.begin());
    data = data_.data().get();
  }
};

}

#endif
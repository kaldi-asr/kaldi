
#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
  #include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"

namespace kaldi {


template<typename Real>
const Real* CuMatrix<Real>::Data() const {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_;
  } else 
  #endif
  {
    return mat_.Data();
  }
}


template<typename Real>
Real* CuMatrix<Real>::Data() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_;
  } else 
  #endif
  {
    return mat_.Data();
  }
}


template<typename Real>
const Real* CuMatrix<Real>::RowData(MatrixIndexT r) const { 
  assert(r < NumRows()); 
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_+r*stride_; 
  } else
  #endif
  {
    return mat_.RowData(r);
  }
}


template<typename Real>
Real* CuMatrix<Real>::RowData(MatrixIndexT r) {
  assert(r < NumRows()); 
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    return data_+r*stride_; 
  } else
  #endif
  {
    return mat_.RowData(r);
  }
}


template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::Resize(MatrixIndexT rows, MatrixIndexT cols) {
  if (num_rows_ == rows && num_cols_ == cols) {
    // SetZero();
    return *this;
  }

  Destroy();

  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    MatrixIndexT row_bytes = cols * sizeof(Real);
    size_t pitch;
    cuSafeCall(cudaMallocPitch((void**)&data_, &pitch, row_bytes, rows));
    num_rows_ = rows; num_cols_ = cols; 
    stride_ = pitch/sizeof(Real);
    SetZero();
  } else
  #endif
  {
    mat_.Resize(rows, cols);
    num_rows_=rows;
    num_cols_=cols;
    stride_=mat_.Stride();
  }
  
  return *this;
}


template<typename Real>
void CuMatrix<Real>::Destroy() {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    if (NULL != data_) {
      cuSafeCall(cudaFree(data_));
      data_ = NULL;
    }
  } else
  #endif
  {
    mat_.Destroy();
  }
  num_rows_ = num_cols_ = stride_ = 0;
}


template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::CopyFromMat(const CuMatrix<Real> &src) {
  Resize(src.NumRows(), src.NumCols());
 
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch, width, src.NumRows(), cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatD2D",tim.Elapsed());
  } else
  #endif
  {
    mat_.CopyFromMat(src.mat_);
  }

  return *this;
}


template<typename Real>
CuMatrix<Real>& CuMatrix<Real>::CopyFromMat(const Matrix<Real> &src) {
  Resize(src.NumRows(), src.NumCols());

  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch, width, src.NumRows(), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatH2D",tim.Elapsed());
  } else
  #endif
  {
    mat_.CopyFromMat(src);
  }

  return *this;
}


template<typename Real>
void CuMatrix<Real>::CopyToMat(Matrix<Real> *dst) const {
  if (dst->NumRows() != NumRows()  ||  dst->NumCols() != NumCols()) {
    dst->Resize(NumRows(), NumCols());
  }

  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 

    Timer tim;
   
    MatrixIndexT src_pitch = stride_*sizeof(Real);
    MatrixIndexT dst_pitch = dst->Stride()*sizeof(Real);
    MatrixIndexT width = NumCols()*sizeof(Real);
    cuSafeCall(cudaMemcpy2D(dst->Data(), dst_pitch, Data(), src_pitch, width, NumRows(), cudaMemcpyDeviceToHost));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
  } else
  #endif
  {
    dst->CopyFromMat(mat_);
  }
}



template<typename Real>
void CuMatrix<Real>::CopyRowsFromMat(int32 r, const CuMatrix<Real> &src, int32 src_ro, int32 dst_ro) {
  
  assert(r+src_ro <= src.NumRows());
  assert(r+dst_ro <= NumRows());
  assert(NumCols() == src.NumCols());

   
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);

    const Real *p_src = src.Data() + src_ro*src.Stride();  
    Real *p_dst = data_ + dst_ro*stride_;

    cuSafeCall(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, r, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyRowsD2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(Data()+dst_ro*stride_, src.Data()+src_ro*src.Stride(), r*stride_*sizeof(Real));
  }
   
}


template<typename Real>
void CuMatrix<Real>::Read(std::istream &is, bool binary) {
  Matrix<BaseFloat> tmp;
  tmp.Read(is, binary);
  CopyFromMat(tmp);    
}


template<typename Real>
void CuMatrix<Real>::Write(std::ostream &os, bool binary) const {
  Matrix<BaseFloat> tmp;
  CopyToMat(&tmp);
  tmp.Write(os, binary); 
}



template<typename Real>
void CuMatrix<Real>::SetZero() {
  #if HAVE_CUDA==1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, num_rows_*stride_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero",tim.Elapsed());
  } else
  #endif
  {
    mat_.SetZero();
  }
}


template<typename Real>
std::ostream &operator << (std::ostream &out, const CuMatrix<Real> &mat) {
  Matrix<Real> tmp;
  mat.CopyToMat(&tmp);
  out << tmp;
  return out;
}



/*
 * declare the float specialized methods
 */
template<> void CuMatrix<float>::Set(float value);
template<> void CuMatrix<float>::ApplyLog();

template<> void CuMatrix<float>::MulElements(const CuMatrix<float>& A);
template<> void CuMatrix<float>::MulColsVec(const CuVector<float> &scale);
template<> void CuMatrix<float>::MulRowsVec(const CuVector<float> &scale);
template<> void CuMatrix<float>::DivRowsVec(const CuVector<float> &div); 

template<> void CuMatrix<float>::AddMat(float alpha, const CuMatrix<float>& A, float beta);

template<> void CuMatrix<float>::AddScaledRow(float alpha, const CuVector<float> &row, float beta);

template<> void CuMatrix<float>::AddMatMat(float alpha, const CuMatrix<float>& A, MatrixTransposeType transA, const CuMatrix<float>& B, MatrixTransposeType transB, float beta);



} // namespace kaldi
  

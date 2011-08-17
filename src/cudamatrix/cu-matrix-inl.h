
#if HAVE_CUDA==1
  #include <cuda_runtime_api.h>
  #include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"

namespace kaldi {


template<typename _ElemT>
const _ElemT* CuMatrix<_ElemT>::Data() const {
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    return data_;
  } else 
  #endif
  {
    return mat_.Data();
  }
}


template<typename _ElemT>
_ElemT* CuMatrix<_ElemT>::Data() {
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    return data_;
  } else 
  #endif
  {
    return mat_.Data();
  }
}


template<typename _ElemT>
const _ElemT* CuMatrix<_ElemT>::RowData(MatrixIndexT r) const { 
  assert(r < NumRows()); 
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    return data_+r*stride_; 
  } else
  #endif
  {
    return mat_.RowData(r);
  }
}


template<typename _ElemT>
_ElemT* CuMatrix<_ElemT>::RowData(MatrixIndexT r) {
  assert(r < NumRows()); 
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    return data_+r*stride_; 
  } else
  #endif
  {
    return mat_.RowData(r);
  }
}


template<typename _ElemT>
CuMatrix<_ElemT>& CuMatrix<_ElemT>::Resize(MatrixIndexT rows, MatrixIndexT cols) {
  if(num_rows_ == rows && num_cols_ == cols) {
    //SetZero();
    return *this;
  }

  Destroy();

  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    MatrixIndexT row_bytes = cols * sizeof(_ElemT);
    size_t pitch;
    cuSafeCall(cudaMallocPitch((void**)&data_, &pitch, row_bytes, rows));
    num_rows_ = rows; num_cols_ = cols; 
    stride_ = pitch/sizeof(_ElemT);
    SetZero();
  } else
  #endif
  {
    mat_.Resize(rows,cols);
    num_rows_=rows;
    num_cols_=cols;
    stride_=mat_.Stride();
  }
  
  return *this;
}


template<typename _ElemT>
void CuMatrix<_ElemT>::Destroy() {
  #if HAVE_CUDA==1
  if(CuDevice::Instantiate().Enabled()) { 
    if(NULL != data_) {
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


template<typename _ElemT>
CuMatrix<_ElemT>& CuMatrix<_ElemT>::CopyFromMat(const CuMatrix<_ElemT>& src) {
  Resize(src.NumRows(),src.NumCols());
 
  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(_ElemT);
    MatrixIndexT src_pitch = src.Stride()*sizeof(_ElemT);
    MatrixIndexT width = src.NumCols()*sizeof(_ElemT);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch, width, src.NumRows(), cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatD2D",tim.Elapsed());
  } else
  #endif
  {
    mat_.CopyFromMat(src.mat_);
  }

  return *this;
}


template<typename _ElemT>
CuMatrix<_ElemT>& CuMatrix<_ElemT>::CopyFromMat(const Matrix<_ElemT>& src) {
  Resize(src.NumRows(),src.NumCols());

  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(_ElemT);
    MatrixIndexT src_pitch = src.Stride()*sizeof(_ElemT);
    MatrixIndexT width = src.NumCols()*sizeof(_ElemT);
    cuSafeCall(cudaMemcpy2D(data_, dst_pitch, src.Data(), src_pitch, width, src.NumRows(), cudaMemcpyHostToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyFromMatH2D",tim.Elapsed());
  } else
  #endif
  {
    mat_.CopyFromMat(src);
  }

  return *this;
}


template<typename _ElemT>
Matrix<_ElemT>& CuMatrix<_ElemT>::CopyToMat(Matrix<_ElemT>& dst) const {
  if(dst.NumRows() != NumRows()  ||  dst.NumCols() != NumCols()) {
    dst.Resize(NumRows(),NumCols());
  }

  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 

    Timer tim;
   
    MatrixIndexT src_pitch = stride_*sizeof(_ElemT);
    MatrixIndexT dst_pitch = dst.Stride()*sizeof(_ElemT);
    MatrixIndexT width = NumCols()*sizeof(_ElemT);
    cuSafeCall(cudaMemcpy2D(dst.Data(), dst_pitch, Data(), src_pitch, width, NumRows(), cudaMemcpyDeviceToHost));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyToMatD2H",tim.Elapsed());
  } else
  #endif
  {
    dst.CopyFromMat(mat_);
  }

  return dst;
}


template<typename _ElemT>
void CuMatrix<_ElemT>::CopyNumRows(MatrixIndexT rowCnt, MatrixIndexT srcOri, const CuMatrix<_ElemT>& src, MatrixIndexT dstOri) {
  assert(rowCnt+srcOri <= src.NumRows());
  assert(rowCnt+dstOri <= NumRows());
  assert(NumCols() == src.NumCols());

   
  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(_ElemT);
    MatrixIndexT src_pitch = src.Stride()*sizeof(_ElemT);
    MatrixIndexT width = src.NumCols()*sizeof(_ElemT);

    const _ElemT* p_src = src.Data() + srcOri*src.Stride();  
    _ElemT* p_dst = data_ + dstOri*stride_;

    cuSafeCall(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, rowCnt, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyRowsD2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(Data()+dstOri*stride_,src.Data()+srcOri*src.Stride(),rowCnt*stride_*sizeof(_ElemT));
  }
   
}


template<typename _ElemT>
void CuMatrix<_ElemT>::Read(std::istream& is, bool binary) {
  Matrix<BaseFloat> tmp;
  tmp.Read(is,binary);
  CopyFromMat(tmp);    
}


template<typename _ElemT>
void CuMatrix<_ElemT>::Write(std::ostream& os, bool binary) const {
  Matrix<BaseFloat> tmp;
  CopyToMat(tmp);
  tmp.Write(os,binary); 
}



template<typename _ElemT>
void CuMatrix<_ElemT>::SetZero() {
  #if HAVE_CUDA==1 
  if(CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    cuSafeCall(cudaMemset(data_, 0, num_rows_*stride_*sizeof(_ElemT)));
    CuDevice::Instantiate().AccuProfile("CuMatrix::SetZero",tim.Elapsed());
  } else
  #endif
  {
    mat_.SetZero();
  }
}


template<typename _ElemT>
std::ostream& operator << (std::ostream& out, const CuMatrix<_ElemT>& mat) {
  Matrix<_ElemT> tmp;
  mat.CopyToMat(tmp);
  out << tmp;
  return out;
}



/*
 * declare the float specialized methods
 */
template<> void CuMatrix<float>::Set(float value);
template<> void CuMatrix<float>::ApplyLog();

template<> void CuMatrix<float>::MulElements(const CuMatrix<float>& A);
template<> void CuMatrix<float>::MulColsVec(const CuVector<float>& scale);
template<> void CuMatrix<float>::MulRowsVec(const CuVector<float>& scale);

template<> void CuMatrix<float>::AddMat(float alpha, const CuMatrix<float>& A, float beta);

template<> void CuMatrix<float>::AddScaledRow(float alpha, const CuVector<float>& row, float beta);

template<> void CuMatrix<float>::AddMatMat(float alpha, const CuMatrix<float>& A, MatrixTransposeType transA, const CuMatrix<float>& B, MatrixTransposeType transB, float beta);



} // namespace kaldi
  

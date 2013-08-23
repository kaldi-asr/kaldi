// cudamatrix/cu-packed-matrix.cc

// Copyright 2009-2013  Johns Hopkins University (author: Daniel Povey)
//                      Karel Vesely

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include "util/timer.h"
#include "cu-common.h"
#include "cu-vector.h"
#include "cu-device.h"
#include "cu-kernels.h"
#include "cu-math.h"
#include "cu-packed-matrix.h"

namespace kaldi {

template<typename Real>
void CuPackedMatrix<Real>::Resize(MatrixIndexT rows,
                                  MatrixResizeType resize_type) {
  // This code does not currently support the other resize_type options.
  KALDI_ASSERT(resize_type == kSetZero || resize_type == kUndefined);

  if (this->num_rows_ == rows) {
    if (resize_type == kSetZero) this->SetZero();
    return;
  }

  if (this->num_rows_ != 0)
    this->Destroy();
  if (rows == 0) return;  
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    this->num_rows_ = rows;
    size_t nr = static_cast<size_t>(num_rows_),
        num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&this->data_), num_bytes));

    if (resize_type == kSetZero) this->SetZero();
  } else
#endif
  { // Let the initializer of SpMatrix<Real> handle the allocation,
    // and then just do Swap which will switch the pointers.
    // This wastes a few instructions but is simple to code.
    SpMatrix<Real> mat(rows, resize_type);
    this->Swap(&mat);
  }
}


template<typename Real>
void CuPackedMatrix<Real>::Destroy() {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) { 
    if (this->data_ != NULL) {
      CU_SAFE_CALL(cudaFree(this->data_));
    }
  } else
  #endif
  {
    if (this->data_ != NULL) KALDI_MEMALIGN_FREE(this->data_);
  }
  this->data_ = NULL;
  this->num_rows_ = 0;
}

template<typename Real>
void CuPackedMatrix<Real>::Swap(PackedMatrix<Real> *mat) {
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) {
      if (mat->num_rows_ != 0) {
        // *this is empty, but mat is nonempty.
        Resize(mat->num_rows_, kUndefined);
        CopyFromPacked(*mat);
        mat->Resize(0);
      }
      // else both are empty.
    } else { // *this is nonempty.
      if (mat->num_rows_ != 0) {
        // Both *this and *mat are nonempty.  Recurse to simpler cases.
        // this could be done more efficiently in the case where
        // the size does not change.
        PackedMatrix<Real> temp;
        this->Swap(&temp); // now temp is full, *this is empty.
        mat->Swap(&temp); // now mat has data from *this, temp has
        // data from mat.
        this->Swap(mat); // copy data in mat to *this, which is now empty.
      } else { // *this is full but *mat is empty.
        mat->Resize(this->num_rows_, kUndefined);
        this->CopyToPacked(mat);
        this->Destroy();
      }
    }
  } else
#endif
  {
    std::swap(mat->data_, this->data_);
    std::swap(mat->num_rows_, this->num_rows_);
  }
}

template<typename Real>
void CuPackedMatrix<Real>::CopyFromPacked(const CuPackedMatrix<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_);
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    size_t nr = static_cast<size_t>(num_rows_),
        num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);

    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&this->data_), num_bytes));    
    CU_SAFE_CALL(cudaMemcpy(data_, src.data_, num_bytes,
                            cudaMemcpyDeviceToDevice));
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::CopyFromPacked1",tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromPacked(src.Mat());
    //memcpy(data_, src.Data(), SizeInBytes());
  }
}

template<typename Real>
void CuPackedMatrix<Real>::CopyFromPacked(const PackedMatrix<Real> &src) {
  KALDI_ASSERT(src.NumRows() == num_rows_);
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    size_t nr = static_cast<size_t>(num_rows_),
        num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&this->data_), num_bytes));    
    CU_SAFE_CALL(cudaMemcpy(data_, src.data_, src.SizeInBytes(),
                            cudaMemcpyHostToDevice));
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::CopyFromPacked2",tim.Elapsed());
  } else
#endif
  {
    Mat().CopyFromPacked(src);
    //memcpy(data_, src.Data(), SizeInBytes());
  }
}

template<typename Real>
void CuPackedMatrix<Real>::CopyToPacked(PackedMatrix<Real> *dst) const {
  KALDI_ASSERT(dst->NumRows() == NumRows() && dst->NumCols() == NumCols());
  
#if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 

    Timer tim;
    size_t nr = static_cast<size_t>(num_rows_),
      num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);
    
    CU_SAFE_CALL(cudaMemcpy(dst->data_, data_, num_bytes,
                            cudaMemcpyDeviceToHost));
    CuDevice::Instantiate().AccuProfile("CuPackedMatrixMatrix::CopyToPackedD2H",tim.Elapsed());
  } else
#endif
  {
    //memcpy(data_, dst->Data(), SizeInBytes());
    dst->CopyFromPacked(Mat());
  }
}

/*
template<typename Real>
void CuPackedMatrix<Real>::CopyRowsFromPacked(int32 r, const CuPackedMatrix<Real> &src, int32 src_ro, int32 dst_ro) {
  KALDI_ASSERT(r+src_ro <= src.NumRows());
  KALDI_ASSERT(r+dst_ro <= NumRows());
  KALDI_ASSERT(NumCols() == src.NumCols());
   
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    MatrixIndexT dst_pitch = stride_*sizeof(Real);
    MatrixIndexT src_pitch = src.Stride()*sizeof(Real);
    MatrixIndexT width = src.NumCols()*sizeof(Real);

    const Real *p_src = src.Data() + src_ro*src.Stride();  
    Real *p_dst = data_ + dst_ro*stride_;

    CU_SAFE_CALL(cudaMemcpy2D(p_dst, dst_pitch, p_src, src_pitch, width, r, cudaMemcpyDeviceToDevice));

    CuDevice::Instantiate().AccuProfile("CuMatrix::CopyRowsD2D",tim.Elapsed());
  } else
  #endif
  {
    memcpy(Data()+dst_ro*stride_, src.Data()+src_ro*src.Stride(), r*stride_*sizeof(Real));
  }
} */



template<typename Real>
void CuPackedMatrix<Real>::Read(std::istream &is, bool binary) {
  PackedMatrix<Real> temp;
  temp.Read(is, binary);
  Destroy();
  Swap(&temp);
}

template<typename Real>
void CuPackedMatrix<Real>::Write(std::ostream &os, bool binary) const {
  PackedMatrix<Real> temp(this->num_rows_, kUndefined);
  this->CopyToPacked(&temp);
  temp.Write(os, binary); 
}

template<typename Real>
void CuPackedMatrix<Real>::SetZero() {
  #if HAVE_CUDA == 1 
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;
    size_t nr = static_cast<size_t>(num_rows_),
      num_bytes = ((nr * (nr+1)) / 2) * sizeof(Real);

    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&this->data_), num_bytes));
    CU_SAFE_CALL(cudaMemset(reinterpret_cast<void*>(this->data_), 0, num_bytes));
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::SetZero", tim.Elapsed());
  } else
  #endif
  {
    Mat().SetZero();
  }
}

/**
 * C++ templatd wrapper of ANSI-C CUBLAS function GEMM (matrix multiply)
 */
#if HAVE_CUDA == 1
template<typename Real> inline Real cublas_dot(int n, const Real *x, int incx, const Real *y, int incy) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline float cublas_dot<float>(int n, const float *x, int incx, const float *y, int incy) {
  return cublasSdot(n, x, incx, y, incy);
}
template<> inline double cublas_dot<double>(int n, const double *x, int incx, const double *y, int incy) {
  return cublasDdot(n, x, incx, y, incy);
}
#endif


template<class Real>
Real CuPackedMatrix<Real>::Trace() const {
  Real result = 0.0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    //int dimBlock(CU2DBLOCK);
    //int dimGrid(n_blocks(NumRows(), CU2DBLOCK));
    int dimBlock(NumRows());
    int dimGrid(1);
    //this is the cublas implementation
    
    size_t nr = static_cast<size_t>(num_rows_),
        num_bytes = nr * sizeof(Real);

    Real *device_ones = 0;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_ones), num_bytes));
    cuda_one(dimGrid,dimBlock,device_ones,num_rows_);
    
    Real* device_array;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_array),num_bytes));
    cuda_copy_diag(dimGrid,dimBlock,device_array,data_,num_rows_);
    result = cublas_dot(num_rows_, device_array, 1, device_ones, 1); 
    
    
    // implementaion using sum_reduce
    /*
    Real* device_result;
    CU_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(Real)));
    CU_SAFE_CALL(cudaMemset(device_result,0, sizeof(Real)));
    cuda_trace(dimGrid, dimBlock, data_, device_result, num_rows_);
    CU_SAFE_CALL(cudaGetLastError());
    CU_SAFE_CALL(cudaMemcpy(&result, device_result, sizeof(Real), cudaMemcpyDeviceToHost));
    */ 
   
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    result = Mat().Trace();
  }
  return result;
}

template<typename Real>
void CuPackedMatrix<Real>::SetDiag(Real alpha) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(NumRows(),CU2DBLOCK));
    cuda_set_diag_packed(dimGrid,dimBlock,data_,alpha,num_rows_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::SetDiag", tim.Elapsed());
  } else
#endif
  {
    Mat().SetDiag(alpha);
  }
}

#if HAVE_CUDA == 1
template<typename Real> inline void cublas_scal(int n, Real alpha, Real* mat, int incx) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_scal<float>(int n, float alpha, float* mat, int incx) {
  cublasSscal(n, alpha, mat, incx);
}
template<> inline void cublas_scal<double>(int n, double alpha, double* mat, int incx) {
  cublasDscal(n, alpha, mat, incx);
}
#endif

template<typename Real>
void CuPackedMatrix<Real>::Scale(Real alpha) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    size_t nr = static_cast<size_t>(num_rows_),
        num_elements = ((nr * (nr+1)) / 2);
    cublas_scal(num_elements, alpha, data_, 1);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::Scale", tim.Elapsed());
  } else
#endif
  {
    Mat().Scale(alpha);
  }
}

template<typename Real>
void CuPackedMatrix<Real>::ScaleDiag(Real alpha) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(NumRows(),CU2DBLOCK));
    cuda_scale_diag(dimGrid,dimBlock,data_,alpha,num_rows_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::ScaleDiag", tim.Elapsed());
  } else
#endif
  {
    Mat().ScaleDiag(alpha);
  }
}

#if HAVE_CUDA == 1
template<typename Real> inline void cublas_axpy(int n, Real alpha, const Real* x, int incx, Real* y, int incy) {
  KALDI_ERR << __func__ << " Not implemented!";
}
template<> inline void cublas_axpy<float>(int n, float alpha, const float* x, int incx, float* y, int incy) {
  cublasSaxpy(n, alpha, x, incx, y, incy);
}
template<> inline void cublas_axpy<double>(int n, double alpha, const double* x, int incx, double* y, int incy) {
  cublasDaxpy(n, alpha, x, incx, y, incy);
}
#endif

template<typename Real>
void CuPackedMatrix<Real>::AddPacked(const Real alpha, const CuPackedMatrix<Real> &M) {
  KALDI_ASSERT(num_rows_ == M.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    size_t nr = num_rows_,
        sz = (nr * (nr + 1)) / 2;
    cublas_axpy(sz, alpha, M.Data(), 1, data_, 1);
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::AddPacked", tim.Elapsed());
  } else
#endif
  {
    Mat().AddPacked(alpha, M.Mat());
  }
}

template<typename Real>
void CuPackedMatrix<Real>::AddToDiag(Real r) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid(1);
    int dimBlock(NumRows());
    cuda_add_diag_packed(dimGrid,dimBlock,data_,r,num_rows_);
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::AddToDiag", tim.Elapsed());
  } else
#endif
  {
    // TODO
    Mat().AddToDiag(r);
  }
}

template<typename Real>
void CuPackedMatrix<Real>::SetUnit() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimGrid(1);
    int dimBlock(NumRows());
    Real alpha = 1;
    cuda_set_diag_packed(dimGrid,dimBlock,data_,alpha,num_rows_);
    CuDevice::Instantiate().AccuProfile("CuPackedMatrix::SetUnit", tim.Elapsed());
  } else 
#endif
  { 
    Mat().SetUnit(); 
  }
}

/**
 * Print the matrix to stream
 */
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuPackedMatrix<Real> &mat) {
  PackedMatrix<Real> temp(mat.NumRows());
  mat.CopyToPacked(&temp);
  out << temp;
  return out;
}

// instantiate the template
template
std::ostream &operator << (std::ostream &out, const CuPackedMatrix<float> &mat);
template
std::ostream &operator << (std::ostream &out, const CuPackedMatrix<double> &mat);


// Instantiate class CuPackedMatrix for float and double.
template class CuPackedMatrix<float>;
template class CuPackedMatrix<double>;


} // namespace kaldi

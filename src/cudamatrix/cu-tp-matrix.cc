// cudamatrix/cu-tp-matrix.cc

// Copyright 2009-2013  Karel Vesely
//           2014-2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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

#if HAVE_CUDA==1
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {

template<typename Real>
CuTpMatrix<Real>::CuTpMatrix(const CuMatrixBase<Real> &orig, MatrixTransposeType trans):
    CuPackedMatrix<Real>(orig.NumRows(), kUndefined) {
  KALDI_ASSERT(orig.NumRows() == orig.NumCols());
  this->CopyFromMat(orig, trans);
}


template<typename Real>
void CuTpMatrix<Real>::Cholesky(const CuSpMatrix<Real> &orig) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    CuMatrix<Real> tmp(orig);
    tmp.Cholesky();
    this->CopyFromMat(tmp, kNoTrans);
  } else
#endif
  {
    this->Mat().Cholesky(orig.Mat());
  }
}


template<typename Real>
void CuTpMatrix<Real>::Invert() {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) return;
    CuTimer tim;
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(this->NumRows(), CU2DBLOCK));
    CuMatrix<Real> tmp(this->NumRows(), this->NumRows());
    int dim = this->NumRows();
    Real alpha = 1.0;
    cuda_set_diag(dimGrid, dimBlock, tmp.Data(), alpha, tmp.Dim());
    CU_SAFE_CALL(cudaGetLastError());        
    CuMatrix<Real> tmp2(dim, dim);
    tmp2.CopyFromTp(*this);
    CU_SAFE_CALL(cublas_trsm(GetCublasHandle(), dim, dim, alpha, tmp2.Data(), tmp2.Dim().stride, 
      tmp.Data(), tmp.Dim().stride));
    this->CopyFromMat(tmp, kNoTrans);
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().Invert();
  }
}

template<typename Real>
void CuTpMatrix<Real>::CopyFromMat(const CuMatrixBase<Real> &M,
                                   MatrixTransposeType Trans) {
#if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) {
    MatrixIndexT num_rows = this->num_rows_;
    KALDI_ASSERT(num_rows == M.NumRows() && this->num_rows_ == M.NumCols());
    if (num_rows == 0) return;
    CuTimer tim;
    dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
    dim3 dimGrid(n_blocks(num_rows, CU2DBLOCK), n_blocks(num_rows, CU2DBLOCK));
    if (Trans == kNoTrans) {
      cuda_take_lower(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
    } else {
      cuda_take_upper(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
    }
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), Trans);
  }
}

template<class Real>
TpMatrix<Real>::TpMatrix(const CuTpMatrix<Real> &cu) {
  this->Resize(cu.NumRows());
  this->CopyFromMat(cu);
}
template TpMatrix<float>::TpMatrix(const CuTpMatrix<float> &cu);
template TpMatrix<double>::TpMatrix(const CuTpMatrix<double> &cu);

template<class Real>
void TpMatrix<Real>::CopyFromMat(const CuTpMatrix<Real> &other) {
  other.CopyToPacked(this);
}
// instantiate the template above.
template void TpMatrix<float>::CopyFromMat(const CuTpMatrix<float> &other);
template void TpMatrix<double>::CopyFromMat(const CuTpMatrix<double> &other);

template <class Real>
CuTpMatrix<Real>& CuTpMatrix<Real>::operator = (const CuTpMatrix<Real> &in) {
  this->Resize(in.NumRows(), kUndefined);
  this->CopyFromPacked(in);
  return *this;
}


template class CuTpMatrix<float>;
template class CuTpMatrix<double>;

} // namespace

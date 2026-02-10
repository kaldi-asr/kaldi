// cudamatrix/cu-sp-matrix.cc

// Copyright      2013  Karel Vesely
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

#if HAVE_CUDA == 1
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {

template<typename Real>
void CuSpMatrix<Real>::CopyFromMat(const CuMatrixBase<Real> &M,
                                   SpCopyType copy_type) {
  KALDI_ASSERT(this->num_rows_ == M.NumRows() &&
               this->num_rows_ == M.NumCols());
  if (this->num_rows_ == 0)
    return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    MatrixIndexT D = this->NumRows();
    if (D == 0)
      return;
    switch (copy_type) {
      case kTakeMeanAndCheck:
        KALDI_ERR << "kTakeMeanAndCheck not supported!";
      // The grid/block dimensions have been very roughly tuned for the
      // individual cases.
      case kTakeMean:
        {
          dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
          dim3 dimGrid(n_blocks(D, CU2DBLOCK), n_blocks(D, CU2DBLOCK));
          cuda_take_mean(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      case kTakeLower:
        {
          int32 block_size = std::min(CU1DBLOCK, this->num_rows_);
          dim3 dimBlock(1, block_size);
          dim3 dimGrid(D, n_blocks(D, block_size));
          cuda_take_lower(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      case kTakeUpper:
        {
          dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
          dim3 dimGrid(n_blocks(D, CU2DBLOCK), n_blocks(D, CU2DBLOCK));
          cuda_take_upper(dimGrid, dimBlock, M.Data(), this->data_, M.Dim());
          CU_SAFE_CALL(cudaGetLastError());
        }
        break;
      default:
        KALDI_ASSERT("Invalid argument to CuSpMatrix::CopyFromMat");
    }
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::CopyFromMat(from CuMatrixBase)", tim);
  } else
#endif
  {
    Mat().CopyFromMat(M.Mat(), copy_type);
  }
}

template<typename Real>
void CuSpMatrix<Real>::Invert() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuMatrix<Real> mat(this->num_rows_, this->num_rows_);
    mat.CopyFromSp(*this);
    mat.SymInvertPosDef();
    this->CopyFromMat(mat);
  } else
#endif
  { // Use inversion of CPU-based SpMatrix.
    Mat().Invert();
  }
}

template<typename Real>
void CuSpMatrix<Real>::AddVec2(const Real alpha, const CuVectorBase<Real> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) return;
    CuTimer tim;
    CUBLAS_SAFE_CALL(cublas_spr(GetCublasHandle(), CUBLAS_FILL_MODE_UPPER,
                                this->num_rows_, alpha, v.Data(),
                                1, this->Data()));
    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddVec2", tim);
  } else
#endif
  {
    Mat().AddVec2(alpha, v.Vec());
  }
}

template<typename Real>
void CuSpMatrix<Real>::AddMat2(const Real alpha, const CuMatrixBase<Real> &M,
                               MatrixTransposeType transM, const Real beta) {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows())
               || (transM == kTrans && this->NumRows() == M.NumCols()));

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->num_rows_ == 0) return;
    CuTimer tim;
    MatrixIndexT this_dim = this->NumRows(),
        m_other_dim = (transM == kNoTrans ? M.NumCols() : M.NumRows());

    if (this_dim == 0) return;
    if (alpha == 0.0) {
      if (beta != 1.0) this->Scale(beta);
      return;
    }

    cublasOperation_t trans = (transM == kTrans ? CUBLAS_OP_N : CUBLAS_OP_T);

    CuMatrix<Real> tmp_mat(*this);
    cublas_syrk(GetCublasHandle(), CUBLAS_FILL_MODE_UPPER, trans, this_dim, m_other_dim, alpha, M.Data(),
                M.Stride(), beta, tmp_mat.Data(), tmp_mat.Stride());
    this->CopyFromMat(tmp_mat, kTakeLower);

    CuDevice::Instantiate().AccuProfile("CuSpMatrix::AddMat2", tim);
  } else
#endif
  {
    Mat().AddMat2(alpha, M.Mat(), transM, beta);
  }
}

/**
 * C++ templatd wrapper of ANSI-C CUBLAS function GEMM (matrix multiply)
 */

template<typename Real, typename OtherReal>
Real TraceSpSp(const CuSpMatrix<Real> &A, const CuSpMatrix<OtherReal> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (A.NumRows() == 0) return 0.0;
    MatrixIndexT nr = A.NumRows(), size = nr * (nr+1) / 2;
    CuVector<Real> Adiag(nr, kUndefined);
    CuVector<OtherReal> Bdiag(nr, kUndefined);
    Adiag.CopyDiagFromPacked(A);
    Bdiag.CopyDiagFromPacked(B);
    CuSubVector<Real> Aall(A.Data(), size);
    CuSubVector<OtherReal> Ball(B.Data(), size);
    // Below, we subtrace VecVec(Adiag, Bdiag) to remove double-counting
    // on the diagonal.
    return 2.0 * VecVec(Aall, Ball) - VecVec(Adiag, Bdiag);
  } else
#endif
  {
    return TraceSpSp(A.Mat(), B.Mat());
  }
}
template
float TraceSpSp(const CuSpMatrix<float> &A, const CuSpMatrix<float> &B);
template
float TraceSpSp(const CuSpMatrix<float> &A, const CuSpMatrix<double> &B);
template
double TraceSpSp(const CuSpMatrix<double> &A, const CuSpMatrix<float> &B);
template
double TraceSpSp(const CuSpMatrix<double> &A, const CuSpMatrix<double> &B);


template<typename Real>
bool CuSpMatrix<Real>::ApproxEqual(const CuSpMatrix<Real> &B, Real tol) const {
  KALDI_ASSERT(this->NumRows() == B.NumRows());
  CuSpMatrix<Real> diff(*this);
  diff.AddSp(-1.0, B);
  Real a = this->FrobeniusNorm(), b = B.FrobeniusNorm(),
      d = diff.FrobeniusNorm();
  return (d <= tol * std::max(a, b));
}

template<typename Real>
bool CuSpMatrix<Real>::IsUnit(Real tol) const {
  // want to return:
  //FrobeniusNorm(*this - I) <= tol * NumRows(), i.e.:
  //sqrt (trace((*this - I)(*this-I)) <= tol * NumRows()
  //    trace((*this - I)(*this - I)) <= tol * NumRows()
  // trace(*this * *this) + trace(I) - 2 * trace(*this) <= tol * NumRows()
  // trace(*this * *this) + dim - 2*this.Trace() <= tol * NumRows()

  // Note: we could do this more efficiently still, by slightly changing the
  // definition of IsUnit and getting rid of the extra stuff inside TraceSpSp
  // that corrects for the diagonal being counted twice.
  return (TraceSpSp(*this, *this) + this->NumRows() - 2.0 * this->Trace() <=
          tol * this->NumRows());
}

template <class Real>
CuSpMatrix<Real>& CuSpMatrix<Real>::operator = (const CuSpMatrix<Real> &in) {
  this->Resize(in.NumRows(), kUndefined);
  this->CopyFromPacked(in);
  return *this;
}

template class CuSpMatrix<float>;
template class CuSpMatrix<double>;



} // namespace

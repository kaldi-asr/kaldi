// cudamatrix/cu-vector.cc

// Copyright 2012-2013  Karel Vesely
//           2012-2014  Johns Hopkins University (author: Daniel Povey)

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
#include "cudamatrix/cu-randkernels.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-rand.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-sparse-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {


template<typename Real>
Real VecVec(const CuVectorBase<Real> &a,
            const CuVectorBase<Real> &b) {
  //MatrixIndexT a_dim = a.Dim();
  KALDI_ASSERT(a.Dim() == b.Dim());
  Real result = 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CU_SAFE_CALL(cublas_dot(GetCublasHandle(), a.Dim(), a.Data(), 1, b.Data(),
			    1, &result));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
} else
#endif
  {
    result = VecVec(a.Vec(), b.Vec());
  }
  return result;
}
// instantiate the template above
template float VecVec(const CuVectorBase<float> &a, const CuVectorBase<float> &b);
template double VecVec(const CuVectorBase<double> &a, const CuVectorBase<double> &b);

// The version of VecVec that can do type conversion.  For now we give this a
// stupid implementation that converts one of the vectors.  If it ever becomes
// an efficiency bottleneck, we can revisit this.
template<typename Real, typename OtherReal>
Real VecVec(const CuVectorBase<Real> &A, const CuVectorBase<OtherReal> &B) {
  CuVector<Real> B2(B);
  return VecVec(A, B2); // This will call the single-parameter template.
}
// instantiate the template above
template float VecVec(const CuVectorBase<float> &A, const CuVectorBase<double> &B);
template double VecVec(const CuVectorBase<double> &A, const CuVectorBase<float> &B);

template<typename Real>
void CuVectorBase<Real>::CopyColFromMat(const CuMatrixBase<Real> &mat, MatrixIndexT col) {
  KALDI_ASSERT(col < mat.NumCols());
  KALDI_ASSERT(dim_ == mat.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    cublas_copy(GetCublasHandle(),
                this->dim_, mat.Data() + col, mat.Stride(), this->data_, 1);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyColFromMat", tim.Elapsed());
  } else
#endif
  {
    Vec().CopyColFromMat(mat.Mat(),col);
  }
}

template<>
template<>
void CuVectorBase<double>::CopyColFromMat(const CuMatrixBase<float> &mat, MatrixIndexT col) {
  KALDI_ASSERT(col < mat.NumCols());
  KALDI_ASSERT(dim_ == mat.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_,CU1DBLOCK));

    cuda_copy_col_from_mat_df(dimGrid, dimBlock, data_, col, mat.Data(), mat.Dim(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyColFromMat", tim.Elapsed());
  } else
#endif
  {
    Vec().CopyColFromMat(mat.Mat(), col);
  }
}


template<>
template<>
void CuVectorBase<float>::CopyColFromMat(const CuMatrixBase<double> &mat, MatrixIndexT col) {
  KALDI_ASSERT(col < mat.NumCols());
  KALDI_ASSERT(dim_ == mat.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_,CU1DBLOCK));

    cuda_copy_col_from_mat_fd(dimGrid, dimBlock, data_, col, mat.Data(), mat.Dim(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyColFromMat", tim.Elapsed());
  } else
#endif
  {
    Vec().CopyColFromMat(mat.Mat(), col);
  }
}

template<typename Real>
void CuVectorBase<Real>::CopyRowsFromMat(const CuMatrixBase<Real> &mat) {
  KALDI_ASSERT(dim_ == mat.NumCols() * mat.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    if (mat.Stride() == mat.NumCols() && mat.NumRows() != 0) {
      CU_SAFE_CALL(cudaMemcpy(data_, mat.Data(), sizeof(Real)*dim_,
                              cudaMemcpyDeviceToDevice));
    } else {
      Real* vec_data = data_;
      for (MatrixIndexT r = 0; r < mat.NumRows(); r++) {
        CU_SAFE_CALL(cudaMemcpy(vec_data, mat.RowData(r),
                                sizeof(Real) * mat.NumCols(),
                                cudaMemcpyDeviceToDevice));
        vec_data += mat.NumCols();
      }
    }
    CuDevice::Instantiate().AccuProfile("CuVectorBase::CopyRowsFromMat", tim.Elapsed());
  } else
#endif
  {
    Vec().CopyRowsFromMat(mat.Mat());
  }
}

template<typename Real>
Real CuVectorBase<Real>::Norm(Real p) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    Real ans;
    KALDI_ASSERT(p == 1.0 || p == 2.0);
    if (dim_ == 0) return 0.0;
    if (p == 1.0) {
      cublas_asum(GetCublasHandle(), dim_, data_, 1, &ans);
    } else {
      cublas_nrm2(GetCublasHandle(), dim_, data_, 1, &ans);
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    if (ans != ans) {
      KALDI_ERR << "NaN in norm " << *this;
    }
    return ans;
  } else
#endif
  {
    return Vec().Norm(p);
  }
}

template<typename Real>
void CuVectorBase<Real>::CopyRowsFromMat(const MatrixBase<Real> &mat) {
  KALDI_ASSERT(dim_ == mat.NumCols() * mat.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    if (mat.Stride() == mat.NumCols()) {
      CU_SAFE_CALL(cudaMemcpy(data_, mat.Data(), sizeof(Real)*dim_,
                              cudaMemcpyHostToDevice));
    } else {
      Real* vec_data = data_;
      for (MatrixIndexT r = 0; r < mat.NumRows(); r++) {
        CU_SAFE_CALL(cudaMemcpy(vec_data, mat.RowData(r),
                                sizeof(Real) * mat.NumCols(),
                                cudaMemcpyHostToDevice));
        vec_data += mat.NumCols();
      }
    }
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().CopyRowsFromMat(mat);
  }
}

template<typename Real>
void MatrixBase<Real>::CopyRowsFromVec(const CuVectorBase<Real> &v) {
  KALDI_ASSERT(v.Dim() == NumCols() * NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0) return;
    Timer tim;
    if (Stride() == NumCols()) {
      CU_SAFE_CALL(cudaMemcpy(data_, v.Data(),
                              sizeof(Real)*v.Dim(),
                              cudaMemcpyDeviceToHost));
    } else {
      const Real* vec_data = v.Data();
      for (MatrixIndexT r = 0; r < NumRows(); r++) {
        CU_SAFE_CALL(cudaMemcpy(RowData(r), vec_data,
                                sizeof(Real) * NumCols(),
                                cudaMemcpyDeviceToHost));
        vec_data += NumCols();
      }
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    CopyRowsFromVec(v.Vec());
  }
}

// instantiate the template above.
template void MatrixBase<float>::CopyRowsFromVec(const CuVectorBase<float> &v);
template void MatrixBase<double>::CopyRowsFromVec(const CuVectorBase<double> &v);

template<typename Real>
void CuVectorBase<Real>::SetRandn() {
  if (dim_ == 0) return;
  CuRand<Real> tmp;
  tmp.RandGaussian(this);
}


template<typename Real>
Real CuVectorBase<Real>::Sum() const {
  if (dim_ == 0)
    return 0.0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int max_threads = 2048;
    // This is the smallest block of consecutive vector elements, which
    // its sum will save at the partial vector.
    int block_size = (dim_ + max_threads - 1) / max_threads;
    if (block_size > 3) {
      int dimBlock(CU1DBLOCK);
      int dimGrid(n_blocks(max_threads, CU1DBLOCK));
      CuVector<Real> g(dimGrid);
      cuda_pvec_sum(dimGrid, dimBlock, data_, g.Data(), dim_, block_size);
      CU_SAFE_CALL(cudaGetLastError());
      Vector<Real> tmp(dimGrid);
      g.CopyToVec(&tmp);
      CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      return tmp.Sum();
    } else {
      CuVector<Real> tmp(1, kUndefined);
      int dimBlock(CU1DBLOCK);
      int dimGrid = 1; // only 1 block here. we have loops in each thread.
      cuda_vec_sum(dimGrid, dimBlock, data_, tmp.Data(), dim_, 1);
      CU_SAFE_CALL(cudaGetLastError());
      CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
      return tmp(0);
    }
  } else
#endif
  {
    return Vec().Sum();
  }
}

template<typename Real>
void CuVectorBase<Real>::ApplySoftMax() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    size_t dimBlock = dim_ > CU1DBLOCK ? CU1DBLOCK : dim_; // for cuda_softmax_reduce function, dimBlock value is fixed min(CU1DBLOCK, dim) , represent CU1DBLOCK threads reduce a row at the same time.
    size_t dimGrid = 1;       // dimGrid value represent the number of rows
    ::MatrixDim dim = { 1, this->dim_, this->dim_};
    cuda_softmax_reduce(dimGrid, dimBlock, data_, data_, dim, this->dim_);//actually dim is not stride...
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().ApplySoftMax();
  }
}

template<typename Real>
MatrixIndexT CuVectorBase<Real>::ApplyFloor(Real floor_val) {
  MatrixIndexT num_floored = 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return 0;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_,CU1DBLOCK));

    CuVector<float> count_vec(dim_, kUndefined);

    cuda_vec_apply_floor(dimGrid, dimBlock, data_, floor_val, count_vec.Data(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    num_floored = count_vec.Sum();
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyFloor", tim.Elapsed());
  } else
#endif
  {
    num_floored = Vec().ApplyFloor(floor_val);
  }
  return num_floored;

}

template<typename Real>
MatrixIndexT CuVectorBase<Real>::ApplyCeiling(Real ceiling_val) {
  MatrixIndexT num_ceiled = 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return 0;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_,CU1DBLOCK));

    CuVector<float> count_vec(dim_, kUndefined);

    cuda_vec_apply_ceiling(dimGrid, dimBlock, data_, ceiling_val, count_vec.Data(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    num_ceiled = count_vec.Sum();
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyFloor", tim.Elapsed());
  } else
#endif
  {
    num_ceiled = Vec().ApplyCeiling(ceiling_val);
  }
  return num_ceiled;
}

template<typename Real>
void CuVectorBase<Real>::ApplyPow(Real power) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    // for this particular kernel, x is #rows, y is #cols.  so
    // fake matrix with 1 row, Dim() cols.
    dim3 dimBlock(CU1DBLOCK, 1);
    dim3 dimGrid(n_blocks(Dim(), CU1DBLOCK), 1);
    ::MatrixDim fake_matrix_dim = { 1, Dim(), 1 };
    // num_cols is Dim(), num_rows is 1, stride is 1 (it's a don't-care).
    cuda_apply_pow(dimGrid, dimBlock, data_, power, fake_matrix_dim);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyFloor", tim.Elapsed());
  } else
#endif
  {
    Vec().ApplyPow(power);
  }
}


template<typename Real>
void CuVectorBase<Real>::ApplyExp() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_,CU1DBLOCK));

    cuda_vec_apply_exp(dimGrid, dimBlock, data_, dim_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyExp", tim.Elapsed());
  } else
#endif
  {
    Vec().ApplyExp();
  }
}


template<typename Real>
void CuVectorBase<Real>::ApplyLog() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_,CU1DBLOCK));

    CuVector<Real> flag(1);
    cuda_vec_apply_log(dimGrid, dimBlock, data_, flag.Data(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    if (flag(0) > 0)
      KALDI_ERR << "Trying to take log of a negative number.";
    CuDevice::Instantiate().AccuProfile("CuVectorBase::ApplyLog", tim.Elapsed());
  } else
#endif
  {
    Vec().ApplyLog();
  }
}


template<typename Real>
void CuVectorBase<Real>::AddMatVec(const Real alpha,
                                   const CuMatrixBase<Real> &M,
                                   MatrixTransposeType trans,
                                   const CuVectorBase<Real> &v,
                                   const Real beta) {
  KALDI_ASSERT((trans == kNoTrans && M.NumCols() == v.dim_ && M.NumRows() == dim_) ||
               (trans == kTrans && M.NumRows() == v.dim_ && M.NumCols() == dim_));
  KALDI_ASSERT(&v != this);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;

    // Everything is backwards in CuBlas.  We need to reverse rows, columns,
    // transpose-ness.
    CU_SAFE_CALL(cublas_gemv(GetCublasHandle(),
			    (trans==kTrans? CUBLAS_OP_N:CUBLAS_OP_T),
			    M.NumCols(), M.NumRows(), alpha, M.Data(),
			    M.Stride(), v.Data(), 1, beta, data_, 1));

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().AddMatVec(alpha,M.Mat(),trans,v.Vec(),beta);
  }
}

template<typename Real>
void CuVectorBase<Real>::AddSpVec(const Real alpha,
                                  const CuSpMatrix<Real> &M,
                                  const CuVectorBase<Real> &v,
                                  const Real beta) {
  KALDI_ASSERT(M.NumCols() == v.dim_ && M.NumRows() == dim_);
  KALDI_ASSERT(&v != this);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;

    // Note: in our opinion the CuSpMatrix represents a lower-triangular matrix, but
    // in CUBLAS, for some stupid reason, everything is reversed.
    CU_SAFE_CALL(cublas_spmv(GetCublasHandle(), CUBLAS_FILL_MODE_UPPER, Dim(),
			    alpha, M.Data(), v.Data(), 1, beta, data_, 1));

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().AddSpVec(alpha,M.Mat(),v.Vec(),beta);
  }
}

template<typename Real>
void CuVectorBase<Real>::AddVecVec(Real alpha, const CuVectorBase<Real> &v,
                                   const CuVectorBase<Real> &r, Real beta) {
  KALDI_ASSERT((dim_ == v.dim_ && dim_ == r.dim_));
  KALDI_ASSERT(this != &v && this != &r);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_,CU1DBLOCK));

    cuda_add_vec_vec(dimGrid, dimBlock, alpha, data_, v.Data(), r.Data(), beta, dim_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::AddVecVec", tim.Elapsed());
  } else
#endif
  {
    Vec().AddVecVec(alpha, v.Vec(), r.Vec(), beta);
  }
}


template<typename Real>
bool CuVectorBase<Real>::ApproxEqual(const CuVectorBase<Real> &other, float tol) const {
  if (dim_ != other.dim_) KALDI_ERR << "ApproxEqual: size mismatch "
                                    << dim_ << " vs. " << other.dim_;
  KALDI_ASSERT(tol >= 0.0);
  CuVector<Real> tmp(*this);
  tmp.AddVec(-1.0, other);
  BaseFloat tmp_norm = sqrt(VecVec(tmp, tmp)), this_norm = sqrt(VecVec(*this, *this));
  return tmp_norm <= static_cast<Real>(tol) * this_norm;
}


template<typename Real>
void CuVectorBase<Real>::AddDiagMat2(Real alpha, const CuMatrixBase<Real> &M,
                                     MatrixTransposeType trans, Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    MatrixTransposeType other_trans = (trans == kTrans ? kNoTrans : kTrans);
    this->AddDiagMatMat(alpha, M, trans,
                        M, other_trans, beta);
  } else
#endif
  {
    Vec().AddDiagMat2(alpha, M.Mat(), trans, beta);
  }
}

template<typename Real>
void CuVectorBase<Real>::AddDiagMatMat(
    Real alpha,
    const CuMatrixBase<Real> &M, MatrixTransposeType transM,
    const CuMatrixBase<Real> &N, MatrixTransposeType transN,
    Real beta) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT dim = this->dim_,
        M_col_dim = (transM == kTrans ? M.NumRows() : M.NumCols()),
        N_row_dim = (transN == kTrans ? N.NumCols() : N.NumRows());
    KALDI_ASSERT(M_col_dim == N_row_dim); // this is the dimension we sum over
    MatrixIndexT M_row_stride = M.Stride(), M_col_stride = 1;
    if (transM == kTrans) std::swap(M_row_stride, M_col_stride);
    MatrixIndexT N_row_stride = N.Stride(), N_col_stride = 1;
    if (transN == kTrans) std::swap(N_row_stride, N_col_stride);
    if (dim_ == 0) return;

    // This kernel can take a variable grid dimension, it makes use
    // of the extra threads by partitioning each vector-vector dot
    // product into multiple pieces.
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim,CU1DBLOCK));
    int threads_per_element = 1;
    // dimGridLimit may be any power of two between 1 and 256 inclusive; it was
    // determined empirically based on speed tests.
    int dimGridLimit = (transM == kNoTrans && transN == kTrans ? 2048 :
                        (transM == kTrans && transN == kNoTrans ? 16 : 32));


    while (M_col_dim > 10 * threads_per_element &&
           dimGrid < dimGridLimit && threads_per_element < 256) {
      threads_per_element *= 2;
      dimGrid = n_blocks(dim * threads_per_element, CU1DBLOCK);
    }

    cuda_add_diag_mat_mat(dimGrid, dimBlock, alpha, data_, dim,
                          M.Data(), M_col_dim, M_row_stride, M_col_stride,
                          N.Data(), N_row_stride, N_col_stride,
                          threads_per_element, beta);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().AddDiagMatMat(alpha, M.Mat(), transM, N.Mat(), transN, beta);
  }
}

template<typename Real>
void CuVectorBase<Real>::AddTpVec(const Real alpha, const CuTpMatrix<Real> &M,
                                  const MatrixTransposeType trans,
                                  const CuVectorBase<Real> &v,
                                  const Real beta) {
  KALDI_ASSERT(dim_ == v.dim_ && dim_ == M.NumRows());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    if (beta == 0.0) {
      if (&v != this) CopyFromVec(v);
      MulTp(M, trans);
      if (alpha != 1.0) Scale(alpha);
    } else {
      CuVector<Real> tmp(v);
      tmp.MulTp(M, trans);
      if (beta != 1.0) Scale(beta);  // *this <-- beta * *this
      AddVec(alpha, tmp, 1.0);          // *this += alpha * M * v
    }
  } else
#endif
  {
    Vec().AddTpVec(alpha, M.Mat(), trans, v.Vec(), beta);
  }
}


template<typename Real>
void CuVectorBase<Real>::MulTp(const CuTpMatrix<Real> &M, const MatrixTransposeType trans) {
  KALDI_ASSERT(M.NumRows() == dim_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    cublas_tpmv(GetCublasHandle(), (trans==kTrans? CUBLAS_OP_N:CUBLAS_OP_T),
		M.NumRows(), M.Data(), data_, 1);
    CuDevice::Instantiate().AccuProfile("CuVectorBase::MulTp", tim.Elapsed());
  } else
#endif
  {
    Vec().MulTp(M.Mat(), trans);
  }
}

template<typename Real>
Real CuVectorBase<Real>::Min() const {
  Real result = 0.0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) {  // min of an empty set is infinity.
      return std::numeric_limits<Real>::infinity();
    }
    Timer tim;
    CuVector<Real> ans(1);
    cuda_vec_min(data_, ans.Data(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    result = ans(0);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    result = (this->Vec()).Min();
  }
  return result;
}

template<typename Real>
Real CuVectorBase<Real>::Max() const {
  Real result = 0.0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) {  // max of an empty set is -infinity.
      return -std::numeric_limits<Real>::infinity();
    }
    Timer tim;
    CuVector<Real> ans(1);
    cuda_vec_max(data_, ans.Data(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    result = ans(0);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    result = (this->Vec()).Max();
  }
  return result;
}

template<typename Real>
void CuVectorBase<Real>::ReplaceValue(Real orig, Real changed) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_, CU1DBLOCK));
    cuda_replace_value(dimGrid, dimBlock, data_, dim_, orig, changed);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().ReplaceValue(orig, changed);
  }
}

template<typename Real>
void CuVectorBase<Real>::MulElements(const CuVectorBase<Real> &v) {
  KALDI_ASSERT(dim_ == v.dim_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_, CU1DBLOCK));
    cuda_vec_mul_elements(dimGrid, dimBlock, data_, v.Data(), dim_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile("CuVectorBase::MulElements", tim.Elapsed());
  } else
#endif
  {
    Vec().MulElements(v.Vec());
  }
}

template<>
template<>
void CuVectorBase<double>::CopyFromVec(const CuVectorBase<float> &src) {
  KALDI_ASSERT(src.Dim() == dim_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU2DBLOCK);
    int dimGrid(n_blocks(dim_, CU2DBLOCK));
    cuda_copy_from_vec_df(dimGrid, dimBlock, data_, src.data_, dim_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().CopyFromVec(src.Vec());
  }
}

template<>
template<>
void CuVectorBase<float>::CopyFromVec(const CuVectorBase<double> &src) {
  KALDI_ASSERT(src.Dim() == dim_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(dim_, CU1DBLOCK));
    cuda_copy_from_vec_fd(dimGrid, dimBlock, data_, src.data_, dim_);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().CopyFromVec(src.Vec());
  }
}


template<typename Real>
template<typename OtherReal>
void CuVectorBase<Real>::CopyFromVec(const VectorBase<OtherReal> &src) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (sizeof(Real) != sizeof(OtherReal)) {
      CuVector<OtherReal> temp(dim_, kUndefined);
      temp.CopyFromVec(src);
      this->CopyFromVec(temp);
    } else {
      KALDI_ASSERT(src.Dim() == dim_);
      if (dim_ == 0) return;
      Timer tim;
      CU_SAFE_CALL(cudaMemcpy(data_, src.Data(), src.Dim()*sizeof(Real), cudaMemcpyHostToDevice));
      CuDevice::Instantiate().AccuProfile("CuVector::CopyFromVecH2D",tim.Elapsed());
    }
  } else
  #endif
  {
    Vec().CopyFromVec(src);
  }
}
// Instantiate the template above.
template
void CuVectorBase<float>::CopyFromVec(const VectorBase<float> &src);
template
void CuVectorBase<double>::CopyFromVec(const VectorBase<float> &src);
template
void CuVectorBase<float>::CopyFromVec(const VectorBase<double> &src);
template
void CuVectorBase<double>::CopyFromVec(const VectorBase<double> &src);

template<typename Real>
template<typename OtherReal>
void CuVectorBase<Real>::CopyToVec(VectorBase<OtherReal> *dst) const {
  KALDI_ASSERT(dim_ == dst->Dim());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (sizeof(Real) != sizeof(OtherReal)) {
      CuVector<OtherReal> temp(*this);
      temp.CopyToVec(dst);
    } else {
      if (dim_ == 0) return;
      Timer tim;
      CU_SAFE_CALL(cudaMemcpy(dst->Data(), this->data_,
                              sizeof(Real) * dim_, cudaMemcpyDeviceToHost));
      CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    }
  } else
#endif
  {
    dst->CopyFromVec(this->Vec());
  }
}


template<typename Real>
void CuVector<Real>::Read(std::istream &is, bool binary) {
  Vector<Real> temp;
  temp.Read(is, binary);
  Destroy();
  Swap(&temp);
}



template<typename Real>
void CuVector<Real>::Write(std::ostream &os, bool binary) const {
  Vector<BaseFloat> temp(this->dim_, kUndefined);
  this->CopyToVec(&temp);
  temp.Write(os, binary);
}


template<typename Real>
CuVector<Real>::CuVector(const CuVectorBase<Real> &v) {
  this->Resize(v.Dim());
  this->CopyFromVec(v);
}

template<typename Real>
CuVector<Real>::CuVector(const VectorBase<Real> &v) {
  this->Resize(v.dim_);
  this->CopyFromVec(v);
}

template<typename Real>
void CuVector<Real>::Resize(MatrixIndexT dim, MatrixResizeType t) {
  KALDI_ASSERT(t == kSetZero || t == kUndefined); // Others not implemented
  // yet.
  if (this->dim_ == dim) {
    this->SetZero();
    return;
  }
  if (this->dim_ != 0)
    this->Destroy();
  if (dim == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    this->data_ = static_cast<Real*>(CuDevice::Instantiate().Malloc(dim * sizeof(Real)));
    this->dim_ = dim;
    if (t == kSetZero) this->SetZero();
    CuDevice::Instantiate().AccuProfile("CuVector::Resize", tim.Elapsed());
  } else
#endif
  {
    Vector<Real> vec(dim);
    this->Swap(&vec);
  }
}

template<typename Real>
void CuVector<Real>::Swap(Vector<Real> *vec) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->dim_ == 0) {
      if (vec->dim_ != 0) {
        // *this is empty, but vec is nonempty.
        Resize(vec->dim_, kUndefined);
        this->CopyFromVec(*vec);
        vec->Resize(0);
      }
      // else both are empty.
    } else { // *this is nonempty.
      if (vec->dim_ != 0) {
        // Both *this and *vec are nonempty.  Recurse to simpler cases.
        // this could be done more efficiently in the case where
        // the size does not change.
        Vector<Real> temp;
        this->Swap(&temp); // now temp is full, *this is empty.
        vec->Swap(&temp); // now vec has data from *this, temp has
        // data from vec.
        Swap(vec); // copy data in vec to *this, which is now empty.
      } else { // *this is full but *vec is empty.
        vec->Resize(this->dim_, kUndefined);
        this->CopyToVec(vec);
        this->Destroy();
      }
    }
  } else
#endif
  {
    std::swap(vec->data_, this->data_);
    std::swap(vec->dim_, this->dim_);
  }
}

template<typename Real>
void CuVector<Real>::Destroy() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (this->data_ != NULL)
      CuDevice::Instantiate().Free(this->data_);
  } else
#endif
  {
    if (this->data_ != NULL) KALDI_MEMALIGN_FREE(this->data_);
  }
  this->data_ = NULL;
  this->dim_ = 0;
}


template<typename Real>
void CuVectorBase<Real>::CopyFromVec(const CuVectorBase<Real> &src) {
  KALDI_ASSERT(src.Dim() == dim_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (dim_ == 0) return;
    Timer tim;
    CU_SAFE_CALL(cudaMemcpy(data_, src.data_, src.dim_ * sizeof(Real), cudaMemcpyDeviceToDevice));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    memcpy(static_cast<void*>(data_), static_cast<void*>(src.data_),
           dim_ * sizeof(Real));
  }
}


template<typename Real>
void CuVectorBase<Real>::SetZero() {
  if (dim_==0 || data_==NULL) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(dim_>=0);
    KALDI_ASSERT(data_!=NULL);
    Timer tim;
    CU_SAFE_CALL(cudaMemset(data_, 0, dim_*sizeof(Real)));
    CuDevice::Instantiate().AccuProfile("CuVector::SetZero",tim.Elapsed());
  } else
#endif
  {
    Vec().SetZero();
  }
}



/// Print the vector to stream
template<typename Real>
std::ostream &operator << (std::ostream &out, const CuVectorBase<Real> &vec) {
  Vector<Real> temp(vec.Dim());
  vec.CopyToVec(&temp);
  out << temp;
  return out;
}
// Instantiate the above.
template
std::ostream &operator << (std::ostream &out, const CuVectorBase<float> &vec);
template
std::ostream &operator << (std::ostream &out, const CuVectorBase<double> &vec);

/*
 * Methods wrapping the ANSI-C CUDA kernels
 */
template<typename Real>
void CuVectorBase<Real>::Set(Real value) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU1DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_set_const(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().Set(value);
  }
}



template<typename Real>
void CuVectorBase<Real>::Add(Real value) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU1DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cuda_add(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().Add(value);
  }
}

template<typename Real>
void CuVectorBase<Real>::CopyDiagFromPacked(const CuPackedMatrix<Real> &M) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(dim_ == M.NumRows());
    if (dim_ == 0) return;
    Timer tim;
    int dimBlock(CU1DBLOCK);
    int dimGrid(n_blocks(Dim(), CU1DBLOCK));
    cuda_vec_copy_diag_from_packed(dimGrid, dimBlock, data_, M.Data(), dim_);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().CopyDiagFromPacked(M.Mat());
  }
}


template<typename Real>
void CuVectorBase<Real>::CopyDiagFromMat(const CuMatrix<Real> &M) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(dim_ == std::min(M.NumRows(), M.NumCols()));
    Timer tim;
    CU_SAFE_CALL(cublas_copy(GetCublasHandle(), dim_, M.Data(), M.Stride() + 1,
			    data_, 1));

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().CopyDiagFromMat(M.Mat());
  }
}


template<typename Real>
void CuVectorBase<Real>::Scale(Real value) {
  #if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (Dim() == 0 ) return;

    Timer tim;
    dim3 dimBlock(CU1DBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CU1DBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };
    cuda_scale(dimGrid, dimBlock, data_, value, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    Vec().Scale(value);
  }
}

template<typename Real>
void CuVectorBase<Real>::AddVec(Real alpha, const CuVectorBase<Real> &vec,
                                Real beta) {
  KALDI_ASSERT(vec.Dim() == Dim());

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    int32 dim = this->dim_;
    Real *data = this->data_;
    const Real *vec_data = vec.data_;
    if (beta != 1.0) CU_SAFE_CALL(cuda_scal(GetCublasHandle(), dim, beta, data, 1));
    if (alpha != 0.0) CU_SAFE_CALL(cuda_axpy(GetCublasHandle(), dim, alpha, vec_data, 1, data, 1));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    if (beta != 1.0) Vec().Scale(beta);
    Vec().AddVec(alpha, vec.Vec());
  }
}


template<typename Real>
template<typename OtherReal>
void CuVectorBase<Real>::AddVec(Real alpha, const CuVectorBase<OtherReal> &vec,
                                Real beta) {
  // We could implement this directly, without using a temporary-- this can
  // be done later, when we have time.
  CuVector<Real> temp(vec);
  this->AddVec(alpha, temp, beta);
}
// instantiate the template above.
template
void CuVectorBase<float>::AddVec(float alpha, const CuVectorBase<double> &vec,
                                 float beta);
template
void CuVectorBase<double>::AddVec(double alpha, const CuVectorBase<float> &vec,
                                  double beta);

template<typename Real>
void CuVectorBase<Real>::AddRowSumMat(Real alpha, const CuMatrixBase<Real> &mat,
                                      Real beta) {
  KALDI_ASSERT(mat.NumCols() == Dim());
  if (Dim() == 0)
    return;
  CuVector<Real> ones(mat.NumRows());
  ones.Set(1.0);
  this->AddMatVec(alpha, mat, kTrans, ones, beta);

}


template<typename Real>
void CuVectorBase<Real>::AddColSumMat(Real alpha,
                                      const CuMatrixBase<Real> &mat,
                                      Real beta) {
  KALDI_ASSERT(mat.NumRows() == Dim());

  CuVector<Real> ones(mat.NumCols());
  ones.Set(1.0);
  this->AddMatVec(alpha, mat, kNoTrans, ones, beta);
}



template<typename Real>
void CuVectorBase<Real>::InvertElements() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;

    dim3 dimBlock(CU1DBLOCK, 1);
    dim3 dimGrid(n_blocks(dim_, CU1DBLOCK));
    MatrixDim d = {1, dim_, dim_};

    cuda_invert_elements(dimGrid, dimBlock, data_, d);
    CU_SAFE_CALL(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    Vec().InvertElements();
  }
}



template
void CuVectorBase<float>::CopyToVec(VectorBase<float> *dst) const;
template
void CuVectorBase<double>::CopyToVec(VectorBase<float> *dst) const;
template
void CuVectorBase<float>::CopyToVec(VectorBase<double> *dst) const;
template
void CuVectorBase<double>::CopyToVec(VectorBase<double> *dst) const;

template class CuVectorBase<float>;
template class CuVectorBase<double>;

template class CuVector<float>;
template class CuVector<double>;

} // namespace

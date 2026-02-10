// cudamatrix/cu-sparse-matrix.cc

// Copyright      2015  Guoguo Chen
//                2015  Johns Hopkins University (author: Daniel Povey)
//                2017  Shiyin Kang


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

#include <utility>
#include <vector>

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sparse-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {

template <typename Real>
MatrixIndexT CuSparseMatrix<Real>::NumRows() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    return num_rows_;
  } else
#endif
  {
    return Smat().NumRows();
  }
}

template <typename Real>
MatrixIndexT CuSparseMatrix<Real>::NumCols() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    return num_cols_;
  } else
#endif
  {
    return Smat().NumCols();
  }
}

template <typename Real>
MatrixIndexT CuSparseMatrix<Real>::NumElements() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    return nnz_;
  } else
#endif
  {
    return Smat().NumElements();
  }
}

template <typename Real>
Real CuSparseMatrix<Real>::Sum() const {
  if (NumElements() == 0)
    return 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuSubVector<Real> sum_vec(CsrVal(), NumElements());
    return sum_vec.Sum();
  } else
#endif
  {
    return Smat().Sum();
  }
}

template <typename Real>
Real CuSparseMatrix<Real>::FrobeniusNorm() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuSubVector<Real> element_vec(CsrVal(), NumElements());
    return element_vec.Norm(2);
  } else
#endif
  {
    return Smat().FrobeniusNorm();
  }
}

template<typename Real>
void CuSparseMatrix<Real>::SelectRows(const CuArray<int32> &row_indexes,
                                      const CuSparseMatrix<Real> &smat_other) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    // Calculate nnz and row_ptr before copying selected col_idx and val.
    // We do this on CPU for now. We will move this part to GPU is mem copy
    // becomes a bottle-neck here.
    std::vector<int32> row_indexes_cpu(row_indexes.Dim());
    row_indexes.CopyToVec(&row_indexes_cpu);
    CuSubArray<int> other_row_ptr(smat_other.CsrRowPtr(),
                                  smat_other.NumRows() + 1);
    std::vector<int> other_row_ptr_cpu(smat_other.NumRows() + 1);
    other_row_ptr.CopyToVec(&other_row_ptr_cpu);
    int nnz = 0;
    std::vector<int> row_ptr_cpu(row_indexes_cpu.size() + 1);
    for (int i = 0; i < row_indexes_cpu.size(); ++i) {
      row_ptr_cpu[i] = nnz;
      nnz += other_row_ptr_cpu[row_indexes_cpu[i] + 1]
          - other_row_ptr_cpu[row_indexes_cpu[i]];
    }
    row_ptr_cpu[row_indexes_cpu.size()] = nnz;

    Resize(row_indexes.Dim(), smat_other.NumCols(), nnz, kUndefined);
    CuSubArray<int> row_ptr(CsrRowPtr(), NumRows() + 1);
    row_ptr.CopyFromVec(row_ptr_cpu);

    // We use warpSize threads per row to access only the nnz elements.
    // Every CU1DBLOCK/warpSize rows share one thread block.
    // 1D grid to cover all selected rows.
    const int warpSize = 32;
    dim3 dimBlock(warpSize, CU1DBLOCK / warpSize);
    dim3 dimGrid(n_blocks(row_indexes.Dim(), dimBlock.y));

    cuda_select_rows(dimGrid, dimBlock, CsrRowPtr(), CsrColIdx(), CsrVal(),
                     row_indexes.Data(), row_indexes.Dim(),
                     smat_other.CsrRowPtr(), smat_other.CsrColIdx(),
                     smat_other.CsrVal());

    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    std::vector<int32> row_indexes_cpu(row_indexes.Dim());
    row_indexes.CopyToVec(&row_indexes_cpu);
    Smat().SelectRows(row_indexes_cpu, smat_other.Smat());
  }
}

template<typename Real>
CuSparseMatrix<Real>::CuSparseMatrix(const CuArray<int32> &indexes, int32 dim,
                                     MatrixTransposeType trans) :
    num_rows_(0), num_cols_(0), nnz_(0), csr_row_ptr_col_idx_(NULL), csr_val_(
    NULL) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Resize(indexes.Dim(), dim, indexes.Dim(), kUndefined);
    if (NumElements() == 0) {
      return;
    }
    CuSubArray<int> row_ptr(CsrRowPtr(), NumRows() + 1);
    row_ptr.Sequence(0);
    CuSubArray<int> col_idx(CsrColIdx(), NumElements());
    col_idx.CopyFromArray(indexes);
    CuSubVector<Real> val(CsrVal(), NumElements());
    val.Set(1);

    if (trans == kTrans) {
      CuSparseMatrix<Real> tmp(*this, kTrans);
      this->Swap(&tmp);
    }
  } else
#endif
  {
    std::vector<int32> idx(indexes.Dim());
    indexes.CopyToVec(&idx);
    SparseMatrix<Real> tmp(idx, dim, trans);
    Smat().Swap(&tmp);
  }
}

template<typename Real>
CuSparseMatrix<Real>::CuSparseMatrix(const CuArray<int32> &indexes,
                                     const CuVectorBase<Real> &weights,
                                     int32 dim, MatrixTransposeType trans) :
    num_rows_(0), num_cols_(0), nnz_(0), csr_row_ptr_col_idx_(NULL), csr_val_(
    NULL) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Resize(indexes.Dim(), dim, indexes.Dim(), kUndefined);
    if (NumElements() == 0) {
      return;
    }
    CuSubArray<int> row_ptr(CsrRowPtr(), NumRows() + 1);
    row_ptr.Sequence(0);
    CuSubArray<int> col_idx(CsrColIdx(), NumElements());
    col_idx.CopyFromArray(indexes);
    CuSubVector<Real> val(CsrVal(), NumElements());
    val.CopyFromVec(weights);

    if (trans == kTrans) {
      CuSparseMatrix<Real> tmp(*this, kTrans);
      this->Swap(&tmp);
    }
  } else
#endif
  {
    std::vector<int32> idx(indexes.Dim());
    indexes.CopyToVec(&idx);
    SparseMatrix<Real> tmp(idx, weights.Vec(), dim, trans);
    Smat().Swap(&tmp);
  }
}

template <typename Real>
CuSparseMatrix<Real>& CuSparseMatrix<Real>::operator = (
    const SparseMatrix<Real> &smat) {
  this->CopyFromSmat(smat);
  return *this;
}

template <typename Real>
CuSparseMatrix<Real>& CuSparseMatrix<Real>::operator = (
    const CuSparseMatrix<Real> &smat) {
  this->CopyFromSmat(smat, kNoTrans);
  return *this;
}

template<typename Real>
void CuSparseMatrix<Real>::Resize(const MatrixIndexT num_rows,
                                  const MatrixIndexT num_cols,
                                  const MatrixIndexT nnz,
                                  MatrixResizeType resize_type) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    KALDI_ASSERT(resize_type == kSetZero || resize_type == kUndefined);

    if (num_rows == NumRows() && num_cols == NumCols()
        && nnz == NumElements()) {
      if (resize_type == kSetZero) {
        CuSubVector<Real> val(CsrVal(), NumElements());
        val.Set(0);
      }
      return;
    }

    Destroy();

    CuTimer tim;

    if (num_rows * num_cols == 0) {
      KALDI_ASSERT(num_rows == 0);
      KALDI_ASSERT(num_cols == 0);
      KALDI_ASSERT(nnz == 0);
      num_rows_ = 0;
      num_cols_ = 0;
      nnz_ = 0;
      csr_row_ptr_col_idx_ = static_cast<int*>(CuDevice::Instantiate().Malloc(
          1 * sizeof(int)));
      csr_val_ = NULL;
    } else {
      KALDI_ASSERT(num_rows > 0);
      KALDI_ASSERT(num_cols > 0);
      KALDI_ASSERT(nnz >= 0 && nnz <= num_rows * static_cast<int64>(num_cols));

      num_rows_ = num_rows;
      num_cols_ = num_cols;
      nnz_ = nnz;
      csr_row_ptr_col_idx_ = static_cast<int*>(CuDevice::Instantiate().Malloc(
          (num_rows + 1 + nnz) * sizeof(int)));
      csr_val_ = static_cast<Real*>(CuDevice::Instantiate().Malloc(
          nnz * sizeof(Real)));
      CuSubArray<int> row_ptr(CsrRowPtr(), NumRows() + 1);
      row_ptr.Set(nnz);
      if (resize_type == kSetZero) {
        CuSubVector<Real> val(CsrVal(), NumElements());
        val.Set(0);
      }
    }

    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Smat().Resize(num_rows, num_cols, resize_type);
  }
}

template<typename Real>
void CuSparseMatrix<Real>::Destroy() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;
    if (csr_row_ptr_col_idx_) {
      CuDevice::Instantiate().Free(csr_row_ptr_col_idx_);
    }
    if (csr_val_) {
      CuDevice::Instantiate().Free(csr_val_);
    }
    num_rows_ = 0;
    num_cols_ = 0;
    nnz_ = 0;
    csr_row_ptr_col_idx_ = NULL;
    csr_val_ = NULL;
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Smat().Resize(0, 0);
  }
}

template<typename Real>
template<typename OtherReal>
void CuSparseMatrix<Real>::CopyFromSmat(const SparseMatrix<OtherReal> &smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Resize(smat.NumRows(), smat.NumCols(), smat.NumElements(), kUndefined);
    if (NumElements() == 0) {
      return;
    }
    std::vector<int> row_ptr(NumRows() + 1);
    std::vector<int> col_idx(NumElements());
    Vector<Real> val(NumElements(), kUndefined);

    int n = 0;
    for (int32 i = 0; i < smat.NumRows(); ++i) {
      row_ptr[i] = n;
      for (int32 j = 0; j < (smat.Data() + i)->NumElements(); ++j, ++n) {
        col_idx[n] = ((smat.Data() + i)->Data() + j)->first;
        val(n) = static_cast<Real>(((smat.Data() + i)->Data() + j)->second);
      }
    }
    row_ptr[NumRows()] = n;
    KALDI_ASSERT(n == NumElements());

    CuSubArray<int> cu_row_ptr(CsrRowPtr(), NumRows() + 1);
    cu_row_ptr.CopyFromVec(row_ptr);
    CuSubArray<int> cu_col_idx(CsrColIdx(), NumElements());
    cu_col_idx.CopyFromVec(col_idx);
    CuSubVector<Real> cu_val(CsrVal(), NumElements());
    cu_val.CopyFromVec(val);
  } else
#endif
  {
    this->Smat().CopyFromSmat(smat);
  }
}
template
void CuSparseMatrix<float>::CopyFromSmat(const SparseMatrix<float> &smat);
template
void CuSparseMatrix<float>::CopyFromSmat(const SparseMatrix<double> &smat);
template
void CuSparseMatrix<double>::CopyFromSmat(const SparseMatrix<float> &smat);
template
void CuSparseMatrix<double>::CopyFromSmat(const SparseMatrix<double> &smat);

template<typename Real>
void CuSparseMatrix<Real>::CopyFromSmat(const CuSparseMatrix<Real>& smat,
                                        MatrixTransposeType trans) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kNoTrans) {
      Resize(smat.NumRows(), smat.NumCols(), smat.NumElements(), kUndefined);

      CuSubVector<Real> val_to(CsrVal(), NumElements());
      CuSubVector<Real> val_from(smat.CsrVal(), smat.NumElements());
      val_to.CopyFromVec(val_from);

      CuSubArray<int> idx_to(csr_row_ptr_col_idx_,
                             NumRows() + 1 + NumElements());
      CuSubArray<int> idx_from(smat.csr_row_ptr_col_idx_,
                               smat.NumRows() + 1 + smat.NumElements());
      idx_to.CopyFromArray(idx_from);

    } else {
      Resize(smat.NumCols(), smat.NumRows(), smat.NumElements(), kUndefined);
      CuTimer tim;

      CUSPARSE_SAFE_CALL(
          cusparse_csr2csc(GetCusparseHandle(), smat.NumRows(), smat.NumCols(),
                           smat.NumElements(), smat.CsrVal(), smat.CsrRowPtr(),
                           smat.CsrColIdx(), CsrVal(), CsrColIdx(), CsrRowPtr(),
                           CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));

      CuDevice::Instantiate().AccuProfile(__func__, tim);
    }
  } else
#endif
  {
    Smat().CopyFromSmat(smat.Smat(), trans);
  }
}

template<typename Real>
template<typename OtherReal>
void CuSparseMatrix<Real>::CopyToSmat(SparseMatrix<OtherReal> *smat) const {
  KALDI_ASSERT(smat != NULL);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (NumRows() == 0) {
      smat->Resize(0, 0);
      return;
    }
    CuSubArray<int> idx(csr_row_ptr_col_idx_, NumRows() + 1 + NumElements());
    std::vector<int> idx_cpu;
    idx.CopyToVec(&idx_cpu);

    CuSubVector<Real> val(CsrVal(), NumElements());
    Vector<OtherReal> val_cpu(NumElements(), kUndefined);
    val.CopyToVec(&val_cpu);

    std::vector<std::vector<std::pair<MatrixIndexT, OtherReal> > > pairs(
        NumRows());
    int n = 0;
    for (int i = 0; i < NumRows(); ++i) {
      for (; n < idx_cpu[i + 1]; ++n) {
        const MatrixIndexT j = idx_cpu[NumRows() + 1 + n];
        pairs[i].push_back( { j, val_cpu(n) });
      }
    }
    KALDI_ASSERT(n == NumElements());
    SparseMatrix<OtherReal> tmp(num_cols_, pairs);
    smat->Swap(&tmp);
  } else
#endif
  {
    smat->CopyFromSmat(this->Smat());
  }
}
template
void CuSparseMatrix<float>::CopyToSmat(SparseMatrix<float> *smat) const;
template
void CuSparseMatrix<float>::CopyToSmat(SparseMatrix<double> *smat) const;
template
void CuSparseMatrix<double>::CopyToSmat(SparseMatrix<float> *smat) const;
template
void CuSparseMatrix<double>::CopyToSmat(SparseMatrix<double> *smat) const;

template<typename Real>
void CuSparseMatrix<Real>::CopyElementsToVec(CuVectorBase<Real> *vec) const {
  KALDI_ASSERT(vec != NULL);
  KALDI_ASSERT(this->NumElements() == vec->Dim());
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuSubVector<Real> val(CsrVal(), NumElements());
    vec->CopyFromVec(val);
  } else
#endif
  {
    Smat().CopyElementsToVec(&(vec->Vec()));
  }
}

template <typename Real>
void CuSparseMatrix<Real>::Swap(SparseMatrix<Real> *smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuSparseMatrix<Real> tmp(*smat);
    Swap(&tmp);
    tmp.CopyToSmat(smat);
  } else
#endif
  {
    Smat().Swap(smat);
  }
}

template<typename Real>
void CuSparseMatrix<Real>::Swap(CuSparseMatrix<Real> *smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    std::swap(num_rows_, smat->num_rows_);
    std::swap(num_cols_, smat->num_cols_);
    std::swap(nnz_, smat->nnz_);
    std::swap(csr_row_ptr_col_idx_, smat->csr_row_ptr_col_idx_);
    std::swap(csr_val_, smat->csr_val_);
  } else
#endif
  {
    Smat().Swap(&(smat->Smat()));
  }
}

template<typename Real>
void CuSparseMatrix<Real>::SetRandn(BaseFloat zero_prob) {
  if (num_rows_ == 0)
    return;
  // Use the CPU function for the moment, not efficient...
  SparseMatrix<Real> tmp(num_rows_, num_cols_);
  tmp.SetRandn(zero_prob);
  Swap(&tmp);
}

template<typename Real>
void CuSparseMatrix<Real>::Write(std::ostream &os, bool binary) const {
  SparseMatrix<Real> tmp;
  CopyToSmat(&tmp);
  tmp.Write(os, binary);
}

template<typename Real>
void CuSparseMatrix<Real>::Read(std::istream &is, bool binary) {
  SparseMatrix<Real> tmp;
  tmp.Read(is, binary);
  this->Swap(&tmp);
}

template class CuSparseMatrix<float>;
template class CuSparseMatrix<double>;

template <typename Real>
Real TraceMatSmat(const CuMatrixBase<Real> &A,
                  const CuSparseMatrix<Real> &B,
                  MatrixTransposeType trans) {
  if (A.NumCols() == 0) {
    KALDI_ASSERT(B.NumCols() == 0);
    return 0.0;
  }
  if (B.NumElements() == 0) {
    return 0.0;
  }
  Real result = 0;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (trans == kTrans) {
      KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
    } else {
      KALDI_ASSERT(A.NumCols() == B.NumRows() && A.NumRows() == B.NumCols());
    }

    // The Sum() method in CuVector handles a bunch of logic, we use that to
    // comptue the trace.
    CuVector<Real> sum_vec(B.NumElements());
    CuTimer tim;

    // We use warpSize threads per row to access only the nnz elements.
    // Every CU1DBLOCK/warpSize rows share one thread block.
    // 1D grid to cover all rows of B.
    const int warpSize = 32;
    dim3 dimBlock(warpSize, CU1DBLOCK / warpSize);
    dim3 dimGrid(n_blocks(B.NumRows(), dimBlock.y));

    if (trans == kNoTrans) {
      cuda_trace_mat_smat(dimGrid, dimBlock, A.Data(), A.Dim(), B.CsrRowPtr(),
                          B.CsrColIdx(), B.CsrVal(), sum_vec.Data());
    } else {
      cuda_trace_mat_smat_trans(dimGrid, dimBlock, A.Data(), A.Dim(),
                                B.CsrRowPtr(), B.CsrColIdx(), B.CsrVal(),
                                sum_vec.Data());
    }
    result = sum_vec.Sum();
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    result = TraceMatSmat(A.Mat(), B.Smat(), trans);
  }
  return result;
}

template
float TraceMatSmat(const CuMatrixBase<float> &A,
                   const CuSparseMatrix<float> &B,
                   MatrixTransposeType trans);
template
double TraceMatSmat(const CuMatrixBase<double> &A,
                    const CuSparseMatrix<double> &B,
                    MatrixTransposeType trans);

void GeneralMatrix::CopyToMat(CuMatrixBase<BaseFloat> *cu_mat,
                              MatrixTransposeType trans) const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    switch (Type()) {
      case kFullMatrix: {
        cu_mat->CopyFromMat(mat_);
        break;
      }
      case kSparseMatrix: {
        CuSparseMatrix<BaseFloat> smat(smat_);
        smat.CopyToMat(cu_mat, trans);
        break;
      }
      case kCompressedMatrix: {
        Matrix<BaseFloat> mat(cmat_);
        if (trans == kNoTrans) {
          cu_mat->CopyFromMat(mat);
          break;
        } else {
          CuMatrix<BaseFloat> temp_cu;
          temp_cu.Swap(&mat);
          cu_mat->CopyFromMat(temp_cu, kTrans);
          break;
        }
      }
      default:
        KALDI_ERR << "Invalid GeneralMatrix type.";
    }
    return;
  } else
#endif
  {
    CopyToMat(&(cu_mat->Mat()), trans);
  }
}


template <typename Real>
template <typename OtherReal>
void CuSparseMatrix<Real>::CopyToMat(CuMatrixBase<OtherReal> *M,
                                     MatrixTransposeType trans) const {
  if (trans == kNoTrans) {
    KALDI_ASSERT(M->NumRows() == NumRows() && M->NumCols() == NumCols());
  } else {
    KALDI_ASSERT(M->NumRows() == NumCols() && M->NumCols() == NumRows());
  }
  M->SetZero();
  if (NumElements() == 0) {
    return;
  }

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuTimer tim;

    // We use warpSize threads per row to access only the nnz elements.
    // Every CU1DBLOCK/warpSize rows share one thread block.
    // 1D grid to cover all rows.
    const int warpSize = 32;
    dim3 dimBlock(warpSize, CU1DBLOCK / warpSize);
    dim3 dimGrid(n_blocks(NumRows(), dimBlock.y));

    if (trans == kNoTrans) {
      cuda_copy_from_smat(dimGrid, dimBlock, M->Data(), M->Dim(), CsrRowPtr(),
                          CsrColIdx(), CsrVal());
    } else {
      cuda_copy_from_smat_trans(dimGrid, dimBlock, M->Data(), M->Dim(),
                                CsrRowPtr(), CsrColIdx(), CsrVal());
    }
    CU_SAFE_CALL(cudaGetLastError());
    CuDevice::Instantiate().AccuProfile(__func__, tim);
  } else
#endif
  {
    Smat().CopyToMat(&(M->Mat()), trans);
  }
}

// Instantiate the template above.
template
void CuSparseMatrix<float>::CopyToMat(CuMatrixBase<float> *M,
                                      MatrixTransposeType trans) const;

template
void CuSparseMatrix<float>::CopyToMat(CuMatrixBase<double> *M,
                                      MatrixTransposeType trans) const;

template
void CuSparseMatrix<double>::CopyToMat(CuMatrixBase<float> *M,
                                       MatrixTransposeType trans) const;

template
void CuSparseMatrix<double>::CopyToMat(CuMatrixBase<double> *M,
                                       MatrixTransposeType trans) const;


void GeneralMatrix::AddToMat(BaseFloat alpha,
                             CuMatrixBase<BaseFloat> *cu_mat,
                             MatrixTransposeType trans) const {
  switch (Type()) {
    case kFullMatrix: {
#if HAVE_CUDA == 1
      if (CuDevice::Instantiate().Enabled()) {
        CuMatrix<BaseFloat> cu_copy(mat_);
        cu_mat->AddMat(alpha, cu_copy);
        break;
      }
#endif
      cu_mat->Mat().AddMat(alpha, mat_);
      break;
    }
    case kSparseMatrix: {
#if HAVE_CUDA == 1
      if (CuDevice::Instantiate().Enabled()) {
        CuSparseMatrix<BaseFloat> cu_smat(smat_);
        cu_mat->AddSmat(alpha, cu_smat, trans);
        break;
      }
#endif
      cu_mat->Mat().AddSmat(alpha, smat_, trans);
      break;
    }
    case kCompressedMatrix: {
      Matrix<BaseFloat> mat(cmat_);
#if HAVE_CUDA == 1
      if (CuDevice::Instantiate().Enabled()) {
        CuMatrix<BaseFloat> cu_mat_copy(mat);
        cu_mat->AddMat(alpha, cu_mat_copy, trans);
        break;
      }
#endif
      cu_mat->Mat().AddMat(alpha, mat, trans);
      break;
    }
    default:
      KALDI_ERR << "Invalid GeneralMatrix type.";
  }
}



}  // namespace kaldi

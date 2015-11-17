// cudamatrix/cu-sparse-matrix.cc

// Copyright      2015  Guoguo Chen
//                2015  Johns Hopkins University (author: Daniel Povey)

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
#include "cudamatrix/cu-randkernels.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sparse-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {

template <typename Real>
const MatrixIndexT* CuRowSparseMatrix<Real>::NumElementsPerRow() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (elements_per_row_.Dim() == 0)
      return NULL;
    else
      return elements_per_row_.Data();
  } else
#endif
  {
    return NULL;
  }
}

template <typename Real>
RowElement<Real>* CuRowSparseMatrix<Real>::Data() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (data_.Dim() == 0)
      return NULL;
    else
      return data_.Data();
  } else
#endif
  {
    return NULL;
  }
}

template <typename Real>
const RowElement<Real>* CuRowSparseMatrix<Real>::Data() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (data_.Dim() == 0)
      return NULL;
    else
      return data_.Data();
  } else
#endif
  {
    return NULL;
  }
}

template <typename Real>
CuRowSparseMatrix<Real>& CuRowSparseMatrix<Real>::operator = (
    const SparseMatrix<Real> &smat) {
  this->CopyFromSmat(smat);
  return *this;
}

template <typename Real>
CuRowSparseMatrix<Real>& CuRowSparseMatrix<Real>::operator = (
    const CuRowSparseMatrix<Real> &smat) {
    num_rows_ = smat.num_rows_;
    num_cols_ = smat.num_cols_;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    data_ = smat.data_;
    stride_ = smat.stride_;
    elements_per_row_ = smat.elements_per_row_;
  } else
#endif
  {
    this->Mat() = smat.Mat();
  }
  return *this;
}

template <typename Real>
MatrixIndexT CuRowSparseMatrix<Real>::NumRows() const{
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    return num_rows_;
  } else
#endif
  {
    return this->Mat().NumRows();
  }
}
template <typename Real>
MatrixIndexT CuRowSparseMatrix<Real>::NumCols() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    return num_cols_;
  } else
#endif
  {
    return this->Mat().NumCols();
  }
}
template <typename Real>
MatrixIndexT CuRowSparseMatrix<Real>::NumElements() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    return num_rows_ * elements_per_row_;
  } else
#endif
  {
    return this->Mat().NumElements();
  }
}

template <typename Real>
void CuRowSparseMatrix<Real>::Swap(SparseMatrix<Real> *smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuRowSparseMatrix<Real> tmp(*smat);
    Swap(&tmp);
    tmp.CopyToSmat(smat);
  } else
#endif
  {
    num_rows_ = smat->NumRows();
    num_cols_ = smat->NumCols();
    Mat().Swap(smat);
  }
}

template <typename Real>
void CuRowSparseMatrix<Real>::Swap(CuRowSparseMatrix<Real> *smat) {
  std::swap(num_rows_, smat->num_rows_);
  std::swap(num_cols_, smat->num_cols_);
  std::swap(stride_, smat->stride_);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuArray<RowElement<Real> > tmp_data(data_);
    data_ = smat->data_;
    smat->data_ = tmp_data;
    CuArray<MatrixIndexT> tmp_elements_per_row(elements_per_row_);
    elements_per_row_ = smat->elements_per_row_;
    smat->elements_per_row_ = tmp_elements_per_row;
  } else
#endif
  {
    Mat().Swap(&(smat->Mat()));
  }
}

template <typename Real>
void CuRowSparseMatrix<Real>::SetRandn(BaseFloat zero_prob) {
  if (num_rows_ == 0) return;
  // Use the CPU function for the moment, not efficient...
  SparseMatrix<Real> tmp(num_rows_, num_cols_);
  tmp.SetRandn(zero_prob);
  Swap(&tmp);
}

template <typename Real>
void CuRowSparseMatrix<Real>::Write(std::ostream &os, bool binary) const {
  SparseMatrix<Real> tmp;
  CopyToSmat(&tmp);
  tmp.Write(os, binary);
}

template <typename Real>
void CuRowSparseMatrix<Real>::Read(std::istream &is, bool binary) {
  SparseMatrix<Real> tmp;
  tmp.Read(is, binary);
  this->Swap(&tmp);
}

template <typename Real>
CuRowSparseMatrix<Real>::CuRowSparseMatrix(const CuRowSparseMatrix<Real> &other) {
  num_rows_ = other.num_rows_;
  num_cols_ = other.num_cols_;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    data_ = other.data_;
    stride_ = smat.stride_;
    elements_per_row_ = other.elements_per_row_;
  } else
#endif
  {
    this->Mat() = other.Mat();
  }
}

template <typename Real>
template <typename OtherReal>
void CuRowSparseMatrix<Real>::CopyToMat(MatrixBase<OtherReal> *other) const {
  SparseMatrix<Real> tmp;
  this->CopyToSmat(&tmp);
  tmp.CopyToMat(other);
}

template
void CuRowSparseMatrix<float>::CopyToMat(MatrixBase<float> *other) const;
template
void CuRowSparseMatrix<float>::CopyToMat(MatrixBase<double> *other) const;
template
void CuRowSparseMatrix<double>::CopyToMat(MatrixBase<float> *other) const;
template
void CuRowSparseMatrix<double>::CopyToMat(MatrixBase<double> *other) const;

template <typename Real>
template <typename OtherReal>
void CuRowSparseMatrix<Real>::CopyFromMat(const MatrixBase<OtherReal> &other) {
  SparseMatrix<Real> tmp;
  tmp.CopyFromMat(other);
  this->CopyFromSmat(tmp);
}

template
void CuRowSparseMatrix<float>::CopyFromMat(const MatrixBase<float> &other);
template
void CuRowSparseMatrix<float>::CopyFromMat(const MatrixBase<double> &other);
template
void CuRowSparseMatrix<double>::CopyFromMat(const MatrixBase<float> &other);
template
void CuRowSparseMatrix<double>::CopyFromMat(const MatrixBase<double> &other);


template <typename Real>
template <typename OtherReal>
void CuRowSparseMatrix<Real>::CopyFromSmat(const SparseMatrix<OtherReal> &smat) {
  num_rows_ = smat.NumRows();
  num_cols_ = smat.NumCols();
  std::vector<MatrixIndexT> cpu_elements_per_row;
  cpu_elements_per_row.resize(num_rows_);
  elements_per_row_.Resize(0);
  MatrixIndexT max_num_elements = 0;
  for (int32 i = 0; i < num_rows_; ++i) {
    MatrixIndexT num_elements = (smat.Data() + i)->NumElements();
    cpu_elements_per_row[i] = num_elements;
    if (num_elements > max_num_elements)
      max_num_elements = num_elements;
  }
  elements_per_row_.CopyFromVec(cpu_elements_per_row);
  stride_ = max_num_elements;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0 || num_cols_ == 0 || stride_ == 0) {
      data_.Resize(0);
      return;
    }
    RowElement<Real> pad;
    pad.column = -1;
    pad.weight = 0.0;
    std::vector<RowElement<Real> > cpu_elements(num_rows_ * stride_, pad);
    for (int32 i = 0; i < num_rows_; ++i) {
      for (int32 j = 0; j < (smat.Data() + i)->NumElements(); ++j) {
        RowElement<Real>* cpu_element = &(cpu_elements[i * stride_ + j]);
        cpu_element->column = ((smat.Data() + i)->Data() + j)->first;
        cpu_element->weight = ((smat.Data() + i)->Data() + j)->second;
      }
    }
    data_.CopyFromVec(cpu_elements);
  } else
#endif
  {
    this->Mat().CopyFromSmat(smat);
  }
}
template
void CuRowSparseMatrix<float>::CopyFromSmat(const SparseMatrix<float> &smat);
template
void CuRowSparseMatrix<float>::CopyFromSmat(const SparseMatrix<double> &smat);
template
void CuRowSparseMatrix<double>::CopyFromSmat(const SparseMatrix<float> &smat);
template
void CuRowSparseMatrix<double>::CopyFromSmat(const SparseMatrix<double> &smat);

template <typename Real>
template <typename OtherReal>
void CuRowSparseMatrix<Real>::CopyToSmat(SparseMatrix<OtherReal> *smat) const {
  KALDI_ASSERT(smat != NULL);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    std::vector<RowElement<Real> > cpu_data;
    data_.CopyToVec(&cpu_data);
    std::vector<std::vector<std::pair<MatrixIndexT, Real> > > pairs(num_rows_);
    for (int32 i = 0; i < cpu_data.size(); ++i) {
      pairs[i].push_back(std::make_pair(cpu_data[i].column, cpu_data[i].weight));
    }
    SparseMatrix<Real> tmp(num_cols_, pairs);
    smat->CopyFromSmat(tmp);
  } else
#endif
  {
    smat->CopyFromSmat(this->Mat());
  }
}
template
void CuRowSparseMatrix<float>::CopyToSmat(SparseMatrix<float> *smat) const;
template
void CuRowSparseMatrix<float>::CopyToSmat(SparseMatrix<double> *smat) const;
template
void CuRowSparseMatrix<double>::CopyToSmat(SparseMatrix<float> *smat) const;
template
void CuRowSparseMatrix<double>::CopyToSmat(SparseMatrix<double> *smat) const;

template <typename Real>
MatrixIndexT CuSparseMatrix<Real>::NumElements() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    return elements_.Dim();
  } else
#endif
  {
    return Mat().NumElements();
  }
}

template <typename Real>
Real CuSparseMatrix<Real>::Sum() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<Real> sum_vec(*this);
    return sum_vec.Sum();
  } else
#endif
  {
    return Mat().Sum();
  }
}

template <typename Real>
Real CuSparseMatrix<Real>::FrobeniusNorm() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<Real> element_vec(*this);
    return element_vec.Norm(2);
  } else
#endif
  {
    return Mat().FrobeniusNorm();
  }
}

template <typename Real>
MatrixElement<Real>* CuSparseMatrix<Real>::Data() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (elements_.Dim() == 0)
      return NULL;
    else
      return elements_.Data();
  } else
#endif
  {
    return NULL;
  }
}

template <typename Real>
const MatrixElement<Real>* CuSparseMatrix<Real>::Data() const {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (elements_.Dim() == 0)
      return NULL;
    else
      return elements_.Data();
  } else
#endif
  {
    return NULL;
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
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    elements_ = smat.elements_;
    num_rows_ = smat.num_rows_;
    num_cols_ = smat.num_cols_;
  } else
#endif
  {
    this->Mat() = smat.Mat();
  }
  return *this;
}

template <typename Real>
template <typename OtherReal>
void CuSparseMatrix<Real>::CopyFromSmat(const SparseMatrix<OtherReal> &smat) {
  num_rows_ = smat.NumRows();
  num_cols_ = smat.NumCols();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (num_rows_ == 0 || num_cols_ == 0) {
      elements_.Resize(0);
      return;
    }
    // We first prepare <elements_> on CPU, we then move it to GPU by calling
    // CopyFromVec. This piece of code should be changed if we change the data
    // structure later.
    std::vector<MatrixElement<Real> > cpu_elements;
    for (int32 i = 0; i < smat.NumRows(); ++i) {
      for (int32 j = 0; j < (smat.Data() + i)->NumElements(); ++j) {
        MatrixElement<Real> cpu_element;
        cpu_element.row = i;
        cpu_element.column = ((smat.Data() + i)->Data() + j)->first;
        cpu_element.weight = ((smat.Data() + i)->Data() + j)->second;
        cpu_elements.push_back(cpu_element);
      }
    }
    elements_.CopyFromVec(cpu_elements);
  } else
#endif
  {
    this->Mat().CopyFromSmat(smat);
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

template <typename Real>
template <typename OtherReal>
void CuSparseMatrix<Real>::CopyToSmat(SparseMatrix<OtherReal> *smat) const {
  KALDI_ASSERT(smat != NULL);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    std::vector<MatrixElement<Real> > cpu_elements;
    elements_.CopyToVec(&cpu_elements);
    std::vector<std::vector<std::pair<MatrixIndexT, Real> > > pairs(num_rows_);
    for (int32 i = 0; i < cpu_elements.size(); ++i) {
      pairs[cpu_elements[i].row].push_back(
          std::make_pair(cpu_elements[i].column, cpu_elements[i].weight));
    }
    SparseMatrix<Real> tmp(num_cols_, pairs);
    smat->CopyFromSmat(tmp);
  } else
#endif
  {
    smat->CopyFromSmat(this->Mat());
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
    num_rows_ = smat->NumRows();
    num_cols_ = smat->NumCols();
    Mat().Swap(smat);
  }
}

template <typename Real>
void CuSparseMatrix<Real>::Swap(CuSparseMatrix<Real> *smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuArray<MatrixElement<Real> > tmp_elements(elements_);
    elements_ = smat->elements_;
    smat->elements_ = tmp_elements;
    MatrixIndexT tmp_dim = num_rows_;
    num_rows_ = smat->num_rows_;
    smat->num_rows_ = tmp_dim;
    tmp_dim = num_cols_;
    num_cols_ = smat->num_cols_;
    smat->num_cols_ = tmp_dim;
  } else
#endif
  {
    Real dim = num_rows_;
    num_rows_ = smat->num_rows_;
    smat->num_rows_ = dim;
    dim = num_cols_;
    num_cols_ = smat->num_cols_;
    smat->num_cols_ = dim;
    Mat().Swap(&(smat->Mat()));
  }
}

template <typename Real>
void CuSparseMatrix<Real>::SetRandn(BaseFloat zero_prob) {
  if (num_rows_ == 0) return;
  // Use the CPU function for the moment, not efficient...
  SparseMatrix<Real> tmp(num_rows_, num_cols_);
  tmp.SetRandn(zero_prob);
  Swap(&tmp);
}

template <typename Real>
void CuSparseMatrix<Real>::Write(std::ostream &os, bool binary) const {
  SparseMatrix<Real> tmp;
  CopyToSmat(&tmp);
  tmp.Write(os, binary);
}

template <typename Real>
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
    Timer tim;
    dim3 dimBlock(CU1DBLOCK, 1);
    dim3 dimGrid(n_blocks(B.NumElements(), CU1DBLOCK), 1);
    if (trans == kNoTrans) {
      cuda_trace_mat_smat(dimGrid, dimBlock, A.Data(), B.Data(),
                          A.Dim(), B.NumElements(), sum_vec.Data());
    } else {
      cuda_trace_mat_smat_trans(dimGrid, dimBlock, A.Data(), B.Data(),
                                A.Dim(), B.NumElements(), sum_vec.Data());
    }
    result = sum_vec.Sum();
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    result = TraceMatSmat(A.Mat(), B.Mat(), trans);
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
    Timer tim;
    dim3 dimBlock(CU1DBLOCK, 1);
    dim3 dimGrid(n_blocks(this->NumElements(), CU1DBLOCK), 1);
    if (trans == kNoTrans) {
      cuda_copy_from_smat(dimGrid, dimBlock, M->Data(),
                          this->Data(), M->Dim(), this->NumElements());
    } else {
      cuda_copy_from_smat_trans(dimGrid, dimBlock, M->Data(),
                                this->Data(), M->Dim(), this->NumElements());
    }
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    reinterpret_cast<const SparseMatrix<Real>*>(this)->CopyToMat(&(M->Mat()),
                                                                 trans);
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
        // TODO: we could make this more efficient by
        // implementing an AddSmat function in class CuMatrixBase.
        CuSparseMatrix<BaseFloat> sparse_cu_mat(smat_);
        CuMatrix<BaseFloat> cu_temp(
            trans == kNoTrans ? sparse_cu_mat.NumRows() :
                                sparse_cu_mat.NumCols(),
            trans == kNoTrans ? sparse_cu_mat.NumCols() :
                                sparse_cu_mat.NumRows(),
            kUndefined);
        sparse_cu_mat.CopyToMat(&cu_temp, trans);
        cu_mat->AddMat(alpha, cu_temp, kNoTrans);
        break;
      }
#endif
      smat_.AddToMat(alpha, &(cu_mat->Mat()), trans);
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

template class CuRowSparseMatrix<float>;
template class CuRowSparseMatrix<double>;


}  // namespace kaldi

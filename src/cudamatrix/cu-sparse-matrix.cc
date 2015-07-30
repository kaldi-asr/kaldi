// cudamatrix/cu-sparse-matrix.cc

// Copyright      2015  Guoguo Chen

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
#include <cublas.h>
#endif

#include "base/timer.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-kernels.h"
#include "cudamatrix/cu-randkernels.h"
#include "cudamatrix/cu-array.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-sparse-matrix.h"
#include "cudamatrix/cublas-wrappers.h"

namespace kaldi {

template <typename Real>
CuSparseMatrix<Real>::CuSparseMatrix(const SparseMatrix<Real> &smat) {
  num_rows_ = smat.NumRows();
  num_cols_ = smat.NumCols();
  if (num_rows_ == 0 || num_cols_ == 0) return;
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    std::vector<MatrixElement<Real> > cpu_elements;
    for (int32 i = 0; i < smat.NumRows(); ++i) {
      for (int32 j = 0; j < (smat.Data() + i)->NumElements(); ++j) {
        MatrixElement<Real> e;
        e.row = i;
        e.column = (smat.Data() + i)->Data()->first;
        e.weight = (smat.Data() + i)->Data()->second;
        cpu_elements.push_back(e);
      }
    }
    elements_.CopyFromVec(cpu_elements);
  } else
#endif
  {
    this->Mat() = smat;
  }
}

template <typename Real>
MatrixElement<Real>* CuSparseMatrix<Real>::Data() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    if (elements_.Dim() == 0) return NULL;
    else return elements_.Data();
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
    if (elements_.Dim() == 0) return NULL;
    else return elements_.Data();
  } else
#endif
  {
    return NULL;
  }
}

template <typename Real>
CuSparseMatrix<Real>& CuSparseMatrix<Real>::operator = (
    const SparseMatrix<Real> &smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuSparseMatrix<Real> tmp(smat);
    elements_ = tmp.elements_;
  } else
#endif
  {
    this->Mat() = smat;
  }
  return *this;
}

template <typename Real>
CuSparseMatrix<Real>& CuSparseMatrix<Real>::operator = (
    const CuSparseMatrix<Real> &smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    elements_ = smat.elements_;
  } else
#endif
  {
    this->Mat() = smat.Mat();
  }
  return *this;
}

template <typename Real>
void CuSparseMatrix<Real>::Swap(SparseMatrix<Real> *smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
  } else
#endif
  {
    Mat().Swap(smat);
  }
}

template <typename Real>
void CuSparseMatrix<Real>::Swap(CuSparseMatrix<Real> *smat) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    CuArray<MatrixElement<Real> > tmp(elements_);
    elements_ = smat->elements_;
    smat->elements_ = tmp;
  } else
#endif
  {
    Mat().Swap(&(smat->Mat()));
  }
}

template <typename Real>
void CuSparseMatrix<Real>::SetRandn(BaseFloat zero_prob) {
  if (num_rows_ == 0) return;
  // Not efficient at the moment...
  SparseMatrix<Real> tmp(num_rows_, num_cols_);
  tmp.SetRandn(zero_prob);
  Swap(&tmp);
}

template <typename Real>
void CuSparseMatrix<Real>::Write(std::ostream &os, bool binary) const {
  SparseMatrix<Real> tmp(this->Mat());
  tmp.Write(os, binary);
}

template <typename Real>
void CuSparseMatrix<Real>::Read(std::istream &is, bool binary) {
  SparseMatrix<Real> tmp;
  tmp.Read(is, binary);
  Swap(&tmp);
}

template class CuSparseMatrix<float>;
template class CuSparseMatrix<double>;

} // namespace kaldi

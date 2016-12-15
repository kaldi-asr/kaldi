// cudamatrix/cu-rand.cc

// Copyright 2016  Brno University of Technology (author Karel Vesely)

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

#include "cudamatrix/cu-rand.h"

namespace kaldi {

template<>
void CuRand<float>::RandUniform(CuMatrixBase<float> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    // Better use 'tmp' matrix, 'tgt' can be a window into a larger matrix,
    // so we should not use it to generate random numbers over whole stride.
    CuMatrix<float> tmp(tgt->NumRows(), tgt->NumCols(), kUndefined);
    // We need even number of `elements', or it crahes!
    // (possibly touching 1 element after array, into the padding of memory alignment),
    size_t tmp_elems_even = (1 + (tmp.NumRows()*tmp.Stride() - 1) / 2) * 2;
    CU_SAFE_CALL(curandGenerateUniform(gen_, tmp.Data(), tmp_elems_even));
    tgt->CopyFromMat(tmp);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->Mat().SetRandUniform();
  }
}

template<>
void CuRand<double>::RandUniform(CuMatrixBase<double> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    // Better use 'tmp' matrix, 'tgt' can be a window into a larger matrix,
    // so we should not use it to generate random numbers over whole stride.
    CuMatrix<double> tmp(tgt->NumRows(), tgt->NumCols(), kUndefined);
    // We need even number of `elements', or it crahes!
    // (possibly touching 1 element after array, into the padding of memory alignment),
    size_t tmp_elems_even = (1 + (tmp.NumRows()*tmp.Stride() - 1) / 2) * 2;
    CU_SAFE_CALL(curandGenerateUniformDouble(gen_, tmp.Data(), tmp_elems_even));
    tgt->CopyFromMat(tmp);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->Mat().SetRandUniform();
  }
}

template<>
void CuRand<float>::RandGaussian(CuMatrixBase<float> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    // Better use 'tmp' matrix, 'tgt' can be a window into a larger matrix,
    // so we should not use it to generate random numbers over whole stride.
    CuMatrix<float> tmp(tgt->NumRows(), tgt->NumCols(), kUndefined);
    // We need even number of `elements', or it crahes!
    // (possibly touching 1 element after array, into the padding of memory alignment),
    size_t tmp_elems_even = (1 + (tmp.NumRows()*tmp.Stride() - 1) / 2) * 2;
    CU_SAFE_CALL(curandGenerateNormal(gen_, tmp.Data(), tmp_elems_even, 0.0, 1.0));
    tgt->CopyFromMat(tmp);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->Mat().SetRandn();
  }
}

template<>
void CuRand<double>::RandGaussian(CuMatrixBase<double> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    // Better use 'tmp' matrix, 'tgt' can be a window into a larger matrix,
    // so we should not use it to generate random numbers over whole stride.
    CuMatrix<double> tmp(tgt->NumRows(), tgt->NumCols(), kUndefined);
    // We need even number of `elements', or it crahes!
    // (possibly touching 1 element after array, into the padding of memory alignment),
    size_t tmp_elems_even = (1 + (tmp.NumRows()*tmp.Stride() - 1) / 2) * 2;
    CU_SAFE_CALL(curandGenerateNormalDouble(gen_, tmp.Data(), tmp_elems_even, 0.0, 1.0));
    tgt->CopyFromMat(tmp);
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->Mat().SetRandn();
  }
}

template<>
void CuRand<float>::RandGaussian(CuVectorBase<float> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT dim_even = (1 + (tgt->Dim() - 1) / 2) * 2;
    CU_SAFE_CALL(curandGenerateNormal(gen_, tgt->Data(), dim_even, 0.0, 1.0));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->Vec().SetRandn();
  }
}

template<>
void CuRand<double>::RandGaussian(CuVectorBase<double> *tgt) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    MatrixIndexT dim_even = (1 + (tgt->Dim() - 1) / 2) * 2;
    CU_SAFE_CALL(curandGenerateNormalDouble(gen_, tgt->Data(), dim_even, 0.0, 1.0));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
  {
    tgt->Vec().SetRandn();
  }
}

/// convert probabilities binary values,
template<typename Real>
void CuRand<Real>::BinarizeProbs(const CuMatrix<Real> &probs, CuMatrix<Real> *states) {
  CuMatrix<Real> tmp(probs.NumRows(), probs.NumCols());
  this->RandUniform(&tmp);  // [0..1]
  tmp.Scale(-1.0);  // [-1..0]
  tmp.AddMat(1.0, probs);  // [-1..+1]
  states->Heaviside(tmp);  // negative
}

/// add gaussian noise to each element
template<typename Real>
void CuRand<Real>::AddGaussNoise(CuMatrix<Real> *tgt, Real gscale) {
  CuMatrix<Real> tmp(tgt->NumRows(), tgt->NumCols());
  this->RandGaussian(&tmp);
  tgt->AddMat(gscale, tmp);
}

// explicit instantiation,
template class CuRand<float>;
template class CuRand<double>;

}  // namespace,


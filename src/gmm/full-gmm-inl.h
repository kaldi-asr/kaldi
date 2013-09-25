// gmm/full-gmm-inl.h

// Copyright 2009-2011  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation

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

#ifndef KALDI_GMM_FULL_GMM_INL_H_
#define KALDI_GMM_FULL_GMM_INL_H_

#include <vector>

#include "util/stl-utils.h"

namespace kaldi {

template<class Real>
void FullGmm::SetWeights(const Vector<Real> &w) {
  KALDI_ASSERT(weights_.Dim() == w.Dim());
  weights_.CopyFromVec(w);
  valid_gconsts_ = false;
}

template<class Real>
void FullGmm::SetMeans(const Matrix<Real> &m) {
  KALDI_ASSERT(means_invcovars_.NumRows() == m.NumRows()
    && means_invcovars_.NumCols() == m.NumCols());
  size_t num_comp = NumGauss();
  Matrix<BaseFloat> m_bf(m);
  for (size_t i = 0; i < num_comp; i++) {
    means_invcovars_.Row(i).AddSpVec(1.0, inv_covars_[i], m_bf.Row(i), 0.0);
  }
  valid_gconsts_ = false;
}

template<class Real>
void FullGmm::SetInvCovarsAndMeans(
    const std::vector<SpMatrix<Real> > &invcovars, const Matrix<Real> &means) {
  KALDI_ASSERT(means_invcovars_.NumRows() == means.NumRows()
    && means_invcovars_.NumCols() == means.NumCols()
    && inv_covars_.size() == invcovars.size());

  size_t num_comp = NumGauss();
  for (size_t i = 0; i < num_comp; i++) {
    inv_covars_[i].CopyFromSp(invcovars[i]);
    Vector<Real> mean_times_inv(Dim());
    mean_times_inv.AddSpVec(1.0, invcovars[i], means.Row(i), 0.0);
    means_invcovars_.Row(i).CopyFromVec(mean_times_inv);
  }
  valid_gconsts_ = false;
}

template<class Real>
void FullGmm::SetInvCovarsAndMeansInvCovars(
    const std::vector<SpMatrix<Real> > &invcovars,
    const Matrix<Real> &means_invcovars) {
  KALDI_ASSERT(means_invcovars_.NumRows() == means_invcovars.NumRows()
               && means_invcovars_.NumCols() == means_invcovars.NumCols()
               && inv_covars_.size() == invcovars.size());

  size_t num_comp = NumGauss();
  for (size_t i = 0; i < num_comp; i++) {
    inv_covars_[i].CopyFromSp(invcovars[i]);
  }
  means_invcovars_.CopyFromMat(means_invcovars);
  valid_gconsts_ = false;
}

template<class Real>
void FullGmm::SetInvCovars(const std::vector<SpMatrix<Real> > &v) {
  KALDI_ASSERT(inv_covars_.size() == v.size());
  size_t num_comp = NumGauss();

  Vector<Real> orig_mean_times_invvar(Dim());
  Vector<Real> orig_mean(Dim());
  Vector<Real> new_mean_times_invvar(Dim());
  SpMatrix<Real> covar(Dim());

  for (size_t i = 0; i < num_comp; i++) {
    orig_mean_times_invvar.CopyFromVec(means_invcovars_.Row(i));
    covar.CopyFromSp(inv_covars_[i]);
    covar.InvertDouble();
    orig_mean.AddSpVec(1.0, covar, orig_mean_times_invvar, 0.0);
    new_mean_times_invvar.AddSpVec(1.0, v[i], orig_mean, 0.0);
    // v[i] is already inverted covar
    means_invcovars_.Row(i).CopyFromVec(new_mean_times_invvar);
    inv_covars_[i].CopyFromSp(v[i]);
  }
  valid_gconsts_ = false;
}

template<class Real>
void FullGmm::GetCovars(std::vector<SpMatrix<Real> > *v) const {
  KALDI_ASSERT(v != NULL);
  v->resize(inv_covars_.size());
  size_t dim = Dim();
  for (size_t i = 0; i < inv_covars_.size(); i++) {
    (*v)[i].Resize(dim);
    (*v)[i].CopyFromSp(inv_covars_[i]);
    (*v)[i].InvertDouble();
  }
}

template<class Real>
void FullGmm::GetMeans(Matrix<Real> *M) const {
  KALDI_ASSERT(M != NULL);
  M->Resize(NumGauss(), Dim());
  SpMatrix<Real> covar(Dim());
  Vector<Real> mean_times_invcovar(Dim());
  for (int32 i = 0; i < NumGauss(); i++) {
    covar.CopyFromSp(inv_covars_[i]);
    covar.InvertDouble();
    mean_times_invcovar.CopyFromVec(means_invcovars_.Row(i));
    (M->Row(i)).AddSpVec(1.0, covar, mean_times_invcovar, 0.0);
  }
}

template<class Real>
void FullGmm::GetCovarsAndMeans(std::vector< SpMatrix<Real> > *covars,
                                Matrix<Real> *means) const {
  KALDI_ASSERT(covars != NULL && means != NULL);
  size_t dim = Dim();
  size_t num_gauss = NumGauss();
  covars->resize(num_gauss);
  means->Resize(num_gauss, dim);
  Vector<Real> mean_times_invcovar(Dim());
  for (size_t i = 0; i < num_gauss; i++) {
    (*covars)[i].Resize(dim);
    (*covars)[i].CopyFromSp(inv_covars_[i]);
    (*covars)[i].InvertDouble();
    mean_times_invcovar.CopyFromVec(means_invcovars_.Row(i));
    (means->Row(i)).AddSpVec(1.0, (*covars)[i], mean_times_invcovar, 0.0);
  }
}


template<class Real>
void FullGmm::GetComponentMean(int32 gauss,
                               VectorBase<Real> *out) const {
  KALDI_ASSERT(gauss < NumGauss() && out != NULL);
  KALDI_ASSERT(out->Dim() == Dim());
  out->SetZero();
  SpMatrix<Real> covar(Dim());
  Vector<Real> mean_times_invcovar(Dim());
  covar.CopyFromSp(inv_covars_[gauss]);
  covar.InvertDouble();
  mean_times_invcovar.CopyFromVec(means_invcovars_.Row(gauss));
  out->AddSpVec(1.0, covar, mean_times_invcovar, 0.0);
}


}  // End namespace kaldi

#endif  // KALDI_GMM_FULL_GMM_INL_H_

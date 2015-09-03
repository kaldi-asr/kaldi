// gmm/diag-gmm-inl.h

// Copyright 2009-2011  Jan Silovsky

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

#ifndef KALDI_GMM_DIAG_GMM_INL_H_
#define KALDI_GMM_DIAG_GMM_INL_H_

#include "util/stl-utils.h"

namespace kaldi {

template<class Real>
void DiagGmm::SetWeights(const VectorBase<Real> &w) {
  KALDI_ASSERT(weights_.Dim() == w.Dim());
  weights_.CopyFromVec(w);
  valid_gconsts_ = false;
}

inline void DiagGmm::SetComponentWeight(int32 g, BaseFloat w) {
  KALDI_ASSERT(w > 0.0);
  KALDI_ASSERT(g < NumGauss());
  weights_(g) = w;
  valid_gconsts_ = false;
}


template<class Real>
void DiagGmm::SetMeans(const MatrixBase<Real> &m) {
  KALDI_ASSERT(means_invvars_.NumRows() == m.NumRows()
    && means_invvars_.NumCols() == m.NumCols());
  means_invvars_.CopyFromMat(m);
  means_invvars_.MulElements(inv_vars_);
  valid_gconsts_ = false;
}

template<class Real>
void DiagGmm::SetComponentMean(int32 g, const VectorBase<Real> &in) {
  KALDI_ASSERT(g < NumGauss() && Dim() == in.Dim());
  Vector<Real> tmp(Dim());
  tmp.CopyRowFromMat(inv_vars_, g);
  tmp.MulElements(in);
  means_invvars_.CopyRowFromVec(tmp, g);
  valid_gconsts_ = false;
}


template<class Real>
void DiagGmm::SetInvVarsAndMeans(const MatrixBase<Real> &invvars,
                                 const MatrixBase<Real> &means) {
  KALDI_ASSERT(means_invvars_.NumRows() == means.NumRows()
    && means_invvars_.NumCols() == means.NumCols()
    && inv_vars_.NumRows() == invvars.NumRows()
    && inv_vars_.NumCols() == invvars.NumCols());

  inv_vars_.CopyFromMat(invvars);
  Matrix<Real> new_means_invvars(means);
  new_means_invvars.MulElements(invvars);
  means_invvars_.CopyFromMat(new_means_invvars);
  valid_gconsts_ = false;
}

template<class Real>
void DiagGmm::SetInvVars(const MatrixBase<Real> &v) {
  KALDI_ASSERT(inv_vars_.NumRows() == v.NumRows()
    && inv_vars_.NumCols() == v.NumCols());

  int32 num_comp = NumGauss(), dim = Dim();
  Matrix<Real> means(num_comp, dim);
  Matrix<Real> vars(num_comp, dim);

  vars.CopyFromMat(inv_vars_);
  vars.InvertElements();  // This inversion happens in double if Real == double
  means.CopyFromMat(means_invvars_);
  means.MulElements(vars);  // These are real means now
  means.MulElements(v);  // v is inverted (in double if Real == double)
  means_invvars_.CopyFromMat(means);  // Means times new inverse variance
  inv_vars_.CopyFromMat(v);
  valid_gconsts_ = false;
}

template<class Real>
void DiagGmm::SetComponentInvVar(int32 g, const VectorBase<Real> &v) {
  KALDI_ASSERT(g < NumGauss() && v.Dim() == Dim());

  int32 dim = Dim();
  Vector<Real> mean(dim), var(dim);

  var.CopyFromVec(inv_vars_.Row(g));
  var.InvertElements();  // This inversion happens in double if Real == double
  mean.CopyFromVec(means_invvars_.Row(g));
  mean.MulElements(var);  // This is a real mean now.
  mean.MulElements(v);  // currently, v is inverted (in double if Real == double)
  means_invvars_.Row(g).CopyFromVec(mean);  // Mean times new inverse variance
  inv_vars_.Row(g).CopyFromVec(v);
  valid_gconsts_ = false;
}


template<class Real>
void DiagGmm::GetVars(Matrix<Real> *v) const {
  KALDI_ASSERT(v != NULL);
  v->Resize(NumGauss(), Dim());
  v->CopyFromMat(inv_vars_);
  v->InvertElements();
}

template<class Real>
void DiagGmm::GetMeans(Matrix<Real> *m) const {
  KALDI_ASSERT(m != NULL);
  m->Resize(NumGauss(), Dim());
  Matrix<Real> vars(NumGauss(), Dim());
  vars.CopyFromMat(inv_vars_);
  vars.InvertElements();
  m->CopyFromMat(means_invvars_);
  m->MulElements(vars);
}


template<class Real>
void DiagGmm::GetComponentMean(int32 gauss, VectorBase<Real> *out) const {
  KALDI_ASSERT(gauss < NumGauss());
  KALDI_ASSERT(static_cast<int32>(out->Dim()) == Dim());
  Vector<Real> tmp(Dim());
  tmp.CopyRowFromMat(inv_vars_, gauss);
  out->CopyRowFromMat(means_invvars_, gauss);
  out->DivElements(tmp);
}

template<class Real>
void DiagGmm::GetComponentVariance(int32 gauss, VectorBase<Real> *out) const {
  KALDI_ASSERT(gauss < NumGauss());
  KALDI_ASSERT(static_cast<int32>(out->Dim()) == Dim());
  out->CopyRowFromMat(inv_vars_, gauss);
  out->InvertElements();
}


}  // End namespace kaldi

#endif  // KALDI_GMM_DIAG_GMM_INL_H_

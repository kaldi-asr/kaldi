// gmm/diag-gmm-normal-inl.h

// Copyright 2009-2011  Yanmin Qian

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

#ifndef KALDI_GMM_DIAG_GMM_NORMAL_INL_H_
#define KALDI_GMM_DIAG_GMM_NORMAL_INL_H_

#include "util/stl-utils.h"

namespace kaldi {

inline void DiagGmmNormal::SetComponentWeight(int32 g, double w) {
  KALDI_ASSERT(w > 0.0);
  KALDI_ASSERT(g < NumGauss());
  weights_(g) = w;
}


template<class Real>
void DiagGmmNormal::SetComponentMean(int32 g, const VectorBase<Real>& in) {
  KALDI_ASSERT(g < NumGauss() && Dim() == in.Dim());
  means_.CopyRowFromVec(in, g);
}

template<class Real>
void DiagGmmNormal::SetComponentVar(int32 g, const VectorBase<Real>& v) {
  KALDI_ASSERT(g < NumGauss() && v.Dim() == Dim());
  vars_.Row(g).CopyFromVec(v);
}

template<class Real>
void DiagGmmNormal::GetComponentMean(int32 gauss, VectorBase<Real>* out) const {
  assert(gauss < NumGauss());
  assert(static_cast<int32>(out->Dim()) == Dim());
  out->CopyRowFromMat(means_, gauss);
}

template<class Real>
void DiagGmmNormal::GetComponentVariance(int32 gauss, VectorBase<Real>* out) const {
  assert(gauss < NumGauss());
  assert(static_cast<int32>(out->Dim()) == Dim());
  out->CopyRowFromMat(vars_, gauss);
}

}  // End namespace kaldi

#endif  // KALDI_GMM_DIAG_GMM_NORMAL_INL_H_

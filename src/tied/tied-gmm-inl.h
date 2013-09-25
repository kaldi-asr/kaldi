// tied/tied-gmm-inl.h

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#ifndef KALDI_TIED_TIED_GMM_INL_H_
#define KALDI_TIED_TIED_GMM_INL_H_

#include "util/stl-utils.h"

namespace kaldi {

template<class Real>
void TiedGmm::SetWeights(const VectorBase<Real> &w) {
  KALDI_ASSERT(weights_.Dim() == w.Dim());
  weights_.CopyFromVec(w);
}

inline void TiedGmm::SetComponentWeight(int32 g, BaseFloat w) {
  KALDI_ASSERT(w > 0.0);
  KALDI_ASSERT(g < NumGauss());
  weights_(g) = w;
}

}  // End namespace kaldi

#endif  // KALDI_TIED_TIED_GMM_INL_H_

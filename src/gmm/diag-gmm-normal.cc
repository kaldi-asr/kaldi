// gmm/diag-gmm-normal.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Yanmin Qian

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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "gmm/diag-gmm-normal.h"
#include "gmm/diag-gmm.h"

namespace kaldi {

void DiagGmmNormal::Resize(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);
  if (weights_.Dim() != nmix) weights_.Resize(nmix);
  if (vars_.NumRows() != nmix ||
      vars_.NumCols() != dim) {
    vars_.Resize(nmix, dim);
    vars_.Set(1.0);
    // must be initialized to unit for case of calling SetMeans while having
    // covars/invcovars that are not set yet (i.e. zero)
  }
  if (means_.NumRows() != nmix ||
      means_.NumCols() != dim)
    means_.Resize(nmix, dim);
}

void DiagGmmNormal::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  int32 num_comp = diaggmm.NumGauss(), dim = diaggmm.Dim();
  Resize(num_comp, dim);
  weights_.CopyFromVec(diaggmm.weights());
  vars_.CopyFromMat(diaggmm.inv_vars());
  vars_.InvertElements();
  means_.CopyFromMat(diaggmm.means_invvars());
  means_.MulElements(vars_);
}

void DiagGmmNormal::CopyToDiagGmm(DiagGmm *diaggmm) {
    assert((static_cast<int32>(diaggmm->Dim()) == means_.NumCols()) && (static_cast<int32>(diaggmm->NumGauss()) == weights_.Dim()));
    diaggmm->SetWeights(weights_);
    Matrix<double> means(weights_.Dim(), means_.NumCols()), invvars(weights_.Dim(), means_.NumCols());
    means = means_;
    invvars = vars_;
    invvars.InvertElements();
    diaggmm->SetInvVarsAndMeans(invvars, means);
}

}  // End namespace kaldi

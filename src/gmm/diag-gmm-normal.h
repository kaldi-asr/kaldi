// gmm/diag-gmm-normal.h

// Copyright 2009-2011  Saarland University  Korbinian Riedhammer  Yanmin Qian
//                      

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

#ifndef KALDI_GMM_DIAG_GMM_NORMAL_H_
#define KALDI_GMM_DIAG_GMM_NORMAL_H_ 1

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "matrix/matrix-lib.h"

namespace kaldi {


class DiagGmm;

/** \class DiagGmmNormal
 *  Definition for Gaussian Mixture Model with diagonal covariances in normal
 *  mode: where the parameters are stored as means and variances (instead of
 *  the exponential form that the DiagGmm class is stored as). This class will
 *  be used in the update (since the update formulas are for the standard
 *  parameterization) and then copied to the exponential form of the DiagGmm
 *  class. The DiagGmmNormal class will not be used anywhere else, and should
 *  not have any extra methods that are not needed.
 */
class DiagGmmNormal {
 public:
  /// Empty constructor.
  DiagGmmNormal() { }

  explicit DiagGmmNormal(const DiagGmm &gmm) {
    CopyFromDiagGmm(gmm);
  }

  /// Resizes arrays to this dim. Does not initialize data.
  void Resize(int32 nMix, int32 dim);

  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

  /// Copies to DiagGmm the requested parameters
  void CopyToDiagGmm(DiagGmm *diaggmm, GmmFlagsType flags = kGmmAll) const;

  int32 NumGauss() { return weights_.Dim(); }
  int32 Dim() { return means_.NumCols(); }

  Vector<double> weights_;  ///< weights (not log).
  Matrix<double> means_;    ///< Means
  Matrix<double> vars_;     ///< diagonal variance

  KALDI_DISALLOW_COPY_AND_ASSIGN(DiagGmmNormal);
};

}  // End namespace kaldi

#endif  // KALDI_GMM_DIAG_GMM_NORMAL_H_

// gmm/full-gmm-normal.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Yanmin Qian
//                      Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#ifndef KALDI_GMM_FULL_GMM_NORMAL_H_
#define KALDI_GMM_FULL_GMM_NORMAL_H_ 1

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

class FullGmm;

/** \class FullGmmNormal
 *  Definition for Gaussian Mixture Model with full covariances in normal
 *  mode: where the parameters are stored as means and variances (instead of
 *  the exponential form that the FullGmm class is stored as). This class will
 *  be used in the update (since the update formulas are for the standard
 *  parameterization) and then copied to the exponential form of the FullGmm
 *  class. The FullGmmNormal class will not be used anywhere else, and should
 *  not have any extra methods that are not needed.
 */
class FullGmmNormal {
 public:
  /// Empty constructor.
  FullGmmNormal() { }

  explicit FullGmmNormal(const FullGmm &gmm) {
    CopyFromFullGmm(gmm);
  }

  /// Resizes arrays to this dim. Does not initialize data.
  void Resize(int32 nMix, int32 dim);

  /// Copies from given FullGmm
  void CopyFromFullGmm(const FullGmm &fullgmm);

  /// Copies to FullGmm
  void CopyToFullGmm(FullGmm *fullgmm, GmmFlagsType flags = kGmmAll);

  /// Generates random features from the model.
  void Rand(MatrixBase<BaseFloat> *feats);

  Vector<double> weights_;              ///< weights (not log).
  Matrix<double> means_;                ///< Means
  std::vector<SpMatrix<double> > vars_;  ///< covariances

  KALDI_DISALLOW_COPY_AND_ASSIGN(FullGmmNormal);
};

}  // End namespace kaldi

#endif  // KALDI_GMM_FULL_GMM_NORMAL_H_

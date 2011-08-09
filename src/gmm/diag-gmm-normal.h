// gmm/diag-gmm-normal.h

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

#ifndef KALDI_GMM_DIAG_GMM_NORMAL_H_
#define KALDI_GMM_DIAG_GMM_NORMAL_H_ 1

#include<vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {


class DiagGmm;

/** \class DiagGmmNormal Definition for Gaussian Mixture Model with diagonal covariances in a normal mode: where the paremeters are stored as means and variances (instead of the exponential form that the DiagGmm class is stored as). This class will be used in the update (since the update formulas are for the standard parametrization) and then copied to the exponential form of the DiagGmm class. The DiagGmmNormal class will not be used anywhere else, and should not have any extra methods that are not needed.
 */
class DiagGmmNormal {
 public:
  /// Empty constructor.
  DiagGmmNormal() { }

  /// Resizes arrays to this dim. Does not initialize data.
  void Resize(int32 nMix, int32 dim);

  /// Returns the number of mixture components in the GMM
  int32 NumGauss() const { return weights_.Dim(); }
  /// Returns the dimensionality of the Gaussian mean vectors
  int32 Dim() const { return means_.NumCols(); }

  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);
  /// Copies to DiagGmm
  void CopyToDiagGmm(DiagGmm *diaggmm);

  void Write(std::ostream &rOut, bool binary) const;
  void Read(std::istream &rIn, bool binary);

  const Vector<double>& weights() const { return weights_; }
  const Matrix<double>& means() const { return means_; }
  const Matrix<double>& vars() const { return vars_; }

   /// Mutators for single component, supports float or double
  /// Set mean for a single component
  template<class Real>
  void SetComponentMean(int32 gauss, const VectorBase<Real>& in);
  /// Set var for single component
  template<class Real>
  void SetComponentVar(int32 gauss, const VectorBase<Real>& in);
  /// Set weight for single component.
  inline void SetComponentWeight(int32 gauss, double weight);

  /// Removes single component from model
  void RemoveComponent(int32 gauss, bool renorm_weights);

  /// Removes multiple components from model; "gauss" must not have dups.
  void RemoveComponents(const std::vector<int32> &gauss, bool renorm_weights);

   /// Accessor for single component mean
  template<class Real>
  void GetComponentMean(int32 gauss, VectorBase<Real>* out) const;

  /// Accessor for single component variance.
  template<class Real>
  void GetComponentVariance(int32 gauss, VectorBase<Real>* out) const;

 private:
  Vector<double> weights_;        ///< weights (not log).
  Matrix<double> means_;       ///< Means
  Matrix<double> vars_;  ///< diagonal variance

  KALDI_DISALLOW_COPY_AND_ASSIGN(DiagGmmNormal);
};

/// ostream operator that calls DiagGMM::Write()
std::ostream &
operator << (std::ostream & rOut, const kaldi::DiagGmmNormal &gmm);
/// istream operator that calls DiagGMM::Read()
std::istream &
operator >> (std::istream & rIn, kaldi::DiagGmmNormal &gmm);

}  // End namespace kaldi

#include "gmm/diag-gmm-normal-inl.h"  // templated functions.

#endif  // KALDI_GMM_DIAG_GMM_NORMAL_H_

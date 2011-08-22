// gmm/diag-gmm.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Georg Stemmer;  Jan Silovsky

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

#ifndef KALDI_GMM_DIAG_GMM_H_
#define KALDI_GMM_DIAG_GMM_H_ 1

#include<vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

class FullGmm;
class DiagGmmNormal;

/** \class DiagGmm Definition for Gaussian Mixture Model with diagonal covariances
 */
class DiagGmm {

 /// this makes it a little easier to modify the internals
 friend class DiagGmmNormal;

 public:
  /// Empty constructor.
  DiagGmm() : valid_gconsts_(false) { }

  /// Resizes arrays to this dim. Does not initialize data.
  void Resize(int32 nMix, int32 dim);

  /// Returns the number of mixture components in the GMM
  int32 NumGauss() const { return weights_.Dim(); }
  /// Returns the dimensionality of the Gaussian mean vectors
  int32 Dim() const { return means_invvars_.NumCols(); }

  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);
  /// Copies from given FullGmm
  void CopyFromFullGmm(const FullGmm &fullgmm);

  /// Returns the log-likelihood of a data point (vector) given the GMM
  BaseFloat LogLikelihood(const VectorBase<BaseFloat> &data) const;

  /// Outputs the per-component log-likelihoods
  void LogLikelihoods(const VectorBase<BaseFloat> &data,
                      Vector<BaseFloat> *loglikes) const;

  /// Outputs the per-component log-likelihoods of a subset
  /// of mixture components.  Note: indices.size() will
  /// equal loglikes->Dim() at output.  loglikes[i] will 
  /// correspond to the log-likelihood of the Gaussian
  /// indexed indices[i].
  void LogLikelihoodsPreselect(const VectorBase<BaseFloat> &data,
                               const std::vector<int32> &indices,
                               Vector<BaseFloat> *loglikes) const;

  
  /// Computes the posterior probabilities of all Gaussian components given
  /// a data point. Returns the log-likehood of the data given the GMM.
  BaseFloat ComponentPosteriors(const VectorBase<BaseFloat> &data,
                                Vector<BaseFloat> *posteriors) const;

  /// Computes the log-likelihood of a data point given a single Gaussian
  /// component. NOTE: Currently we make no guarantees about what happens if
  /// one of the variances is zero.
  BaseFloat ComponentLogLikelihood(const VectorBase<BaseFloat> &data,
                                   int32 comp_id) const;

  /// Sets the gconsts.  Returns the number that are "invalid" e.g. because of
  /// zero weights or variances.
  int32 ComputeGconsts();

  /// Generates a random data-point from this distribution.
  void Generate(VectorBase<BaseFloat> *output);

  void Split(int32 target_components, float perturb_factor);
  void Merge(int32 target_components);

  void Write(std::ostream &rOut, bool binary) const;
  void Read(std::istream &rIn, bool binary);

  /// Const accessors
  const Vector<BaseFloat>& gconsts() const {
    KALDI_ASSERT(valid_gconsts_);
    return gconsts_;
  }
  const Vector<BaseFloat>& weights() const { return weights_; }
  const Matrix<BaseFloat>& means_invvars() const { return means_invvars_; }
  const Matrix<BaseFloat>& inv_vars() const { return inv_vars_; }
  bool valid_gconsts() const { return valid_gconsts_; }

  /// Removes single component from model
  void RemoveComponent(int32 gauss, bool renorm_weights);

  /// Removes multiple components from model; "gauss" must not have dups.
  void RemoveComponents(const std::vector<int32> &gauss, bool renorm_weights);

  /// Mutators for both float or double
  template<class Real>
  void SetWeights(const VectorBase<Real>& w);    ///< Set mixure weights

  /// Use SetMeans to update only the Gaussian means (and not variances)
  template<class Real>
  void SetMeans(const MatrixBase<Real>& m);
  /// Use SetInvVarsAndMeans if updating both means and (inverse) variances
  template<class Real>
  void SetInvVarsAndMeans(const MatrixBase<Real>& invvars,
                          const MatrixBase<Real>& means);
  /// Set the (inverse) variances and recompute means_invvars_
  template<class Real>
  void SetInvVars(const MatrixBase<Real>& v);

  /// Accessor for covariances.
  template<class Real>
  void GetVars(Matrix<Real>* v) const;
  /// Accessor for means.
  template<class Real>
  void GetMeans(Matrix<Real> *m) const;

  /// Mutators for single component, supports float or double
  /// Set mean for a single component - internally multiplies with inv(var)
  template<class Real>
  void SetComponentMean(int32 gauss, const VectorBase<Real>& in);
  /// Set inv-var for single component (recommend to do this before
  /// setting the mean, if doing both, for numerical reasons).
  template<class Real>
  void SetComponentInvVar(int32 gauss, const VectorBase<Real>& in);
  /// Set weight for single component.
  inline void SetComponentWeight(int32 gauss, BaseFloat weight);
  
  /// Accessor for single component mean
  template<class Real>
  void GetComponentMean(int32 gauss, VectorBase<Real>* out) const;

  /// Accessor for single component variance.
  template<class Real>
  void GetComponentVariance(int32 gauss, VectorBase<Real>* out) const;

 private:
  /// Equals log(weight) - 0.5 * (log det(var) + mean*mean*inv(var))
  Vector<BaseFloat> gconsts_;
  bool valid_gconsts_;   ///< Recompute gconsts_ if false
  Vector<BaseFloat> weights_;        ///< weights (not log).
  Matrix<BaseFloat> inv_vars_;       ///< Inverted (diagonal) variances
  Matrix<BaseFloat> means_invvars_;  ///< Means times inverted variance

  // merged_components_logdet computes logdet for merged components
  // f1, f2 are first-order stats (normalized by zero-order stats)
  // s1, s2 are second-order stats (normalized by zero-order stats)
  BaseFloat merged_components_logdet(BaseFloat w1, BaseFloat w2,
                                     const VectorBase<BaseFloat> &f1,
                                     const VectorBase<BaseFloat> &f2,
                                     const VectorBase<BaseFloat> &s1,
                                     const VectorBase<BaseFloat> &s2) const;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DiagGmm);
};

/// ostream operator that calls DiagGMM::Write()
std::ostream &
operator << (std::ostream & rOut, const kaldi::DiagGmm &gmm);
/// istream operator that calls DiagGMM::Read()
std::istream &
operator >> (std::istream & rIn, kaldi::DiagGmm &gmm);

}  // End namespace kaldi

#include "gmm/diag-gmm-inl.h"  // templated functions.

#endif  // KALDI_GMM_DIAG_GMM_H_

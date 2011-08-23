// gmm/full-gmm.h

// Copyright 2009-2011  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation

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

#ifndef KALDI_GMM_FULL_GMM_H_
#define KALDI_GMM_FULL_GMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/model-common.h"
#include "matrix/matrix-lib.h"
#include "util/parse-options.h"

namespace kaldi {

class DiagGmm;
class FullGmmNormal;

/** \class Definition for Gaussian Mixture Model with full covariances
  */
class FullGmm {
 
 /// this makes it a little easier to modify the internals
 friend class FullGmmNormal;

 public:
  /// Empty constructor.
  FullGmm() : valid_gconsts_(false) {}

  /// Resizes arrays to this dim. Does not initialize data.
  void Resize(int32 nMix, int32 dim);

  /// Returns the number of mixture components in the GMM
  int32 NumGauss() const { return weights_.Dim(); }
  /// Returns the dimensionality of the Gaussian mean vectors
  int32 Dim() const { return means_invcovars_.NumCols(); }

  /// Copies from given FullGmm
  void CopyFromFullGmm(const FullGmm &fullgmm);
  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

  /// Returns the log-likelihood of a data point (vector) given the GMM
  BaseFloat LogLikelihood(const VectorBase<BaseFloat> &data) const;

  /// Outputs the per-component contributions to the
  /// log-likelihood
  void LogLikelihoods(const VectorBase<BaseFloat> &data,
                      Vector<BaseFloat>* loglikes) const;

  /// Computes the posterior probabilities of all Gaussian components given
  /// a data point. Returns the log-likehood of the data given the GMM.
  BaseFloat ComponentPosteriors(const VectorBase<BaseFloat> &data,
                                Vector<BaseFloat> *posterior) const;

  /// Computes the contribution log-likelihood of a data point from a single
  /// Gaussian component. NOTE: Currently we make no guarantees about what
  /// happens if one of the variances is zero.
  BaseFloat ComponentLogLikelihood(const VectorBase<BaseFloat> &data,
                                   int32 comp_id) const;

  /// Sets the gconsts.  Returns the number that are "invalid" e.g. because of
  /// zero weights or variances.
  int32 ComputeGconsts();

  void Split(int32 target_components, float perturb_factor);
  void Merge(int32 target_components);
  void Write(std::ostream &rOut, bool binary) const;
  void Read(std::istream &rIn, bool binary);

  void SmoothWithFullGmm(BaseFloat rho, const FullGmm *source, GmmFlagsType flags = kGmmAll);

  /// Const accessors
  const Vector<BaseFloat>& gconsts() const { return gconsts_; }
  const Vector<BaseFloat>& weights() const { return weights_; }
  const Matrix<BaseFloat>& means_invcovars() const { return means_invcovars_; }
  const std::vector<SpMatrix<BaseFloat> >& inv_covars() const {
    return inv_covars_; }

  /// Non-const accessors
  Matrix<BaseFloat>& means_invcovars() { return means_invcovars_; }
  std::vector<SpMatrix<BaseFloat> >& inv_covars() { return inv_covars_; }

  /// Mutators for both float or double
  template<class Real>
  void SetWeights(const Vector<Real>& w);    ///< Set mixure weights

  /// Use SetMeans to update only the Gaussian means (and not variances)
  template<class Real>
  void SetMeans(const Matrix<Real>& m);
  
  /// Use SetInvCovarsAndMeans if updating both means and (inverse) covariances
  template<class Real>
  void SetInvCovarsAndMeans(const std::vector<SpMatrix<Real> >& invcovars,
                            const Matrix<Real>& means);

  /// Use this if setting both, in the class's native format.
  template<class Real>
  void SetInvCovarsAndMeansInvCovars(const std::vector<SpMatrix<Real> >& invcovars,
                                     const Matrix<Real>& means_invcovars);

  /// Set the (inverse) covariances and recompute means_invcovars_
  template<class Real>
  void SetInvCovars(const std::vector<SpMatrix<Real> >& v);

  /// Accessor for covariances.
  template<class Real>
  void GetCovars(std::vector<SpMatrix<Real> >* v) const;
  /// Accessor for means.
  template<class Real>
  void GetMeans(Matrix<Real> *m) const;
  /// Accessor for covariances and means
  template<class Real>
  void GetCovarsAndMeans(std::vector< SpMatrix<Real> >* covars,
                         Matrix<Real> *means) const;

  /// Mutators for single component, supports float or double
  /// Removes single component from model
  void RemoveComponent(int32 gauss, bool renorm_weights);

  /// Removes multiple components from model; "gauss" must not have dups.
  void RemoveComponents(const std::vector<int32> &gauss, bool renorm_weights);

  /// Accessor for component mean
  template<class Real>
  void GetComponentMean(int32 gauss, VectorBase<Real> *out) const;

 private:
  /// Equals log(weight) - 0.5 * (log det(var) + mean'*inv(var)*mean)
  Vector<BaseFloat> gconsts_;
  bool valid_gconsts_;  ///< Recompute gconsts_ if false
  Vector<BaseFloat> weights_;  ///< weights (not log).
  std::vector<SpMatrix<BaseFloat> > inv_covars_;  ///< Inverse covariances
  Matrix<BaseFloat> means_invcovars_;  ///< Means times inverse covariances

  /// Resizes arrays to this dim. Does not initialize data.
  void ResizeInvCovars(int32 nMix, int32 dim);

  // merged_components_logdet computes logdet for merged components
  // f1, f2 are first-order stats (normalized by zero-order stats)
  // s1, s2 are second-order stats (normalized by zero-order stats)
  BaseFloat merged_components_logdet(BaseFloat w1, BaseFloat w2,
                                     const VectorBase<BaseFloat> &f1,
                                     const VectorBase<BaseFloat> &f2,
                                     const SpMatrix<BaseFloat> &s1,
                                     const SpMatrix<BaseFloat> &s2) const;


  KALDI_DISALLOW_COPY_AND_ASSIGN(FullGmm);
};

/// ostream operator that calls FullGmm::Write()
std::ostream &
operator << (std::ostream & rOut, const kaldi::FullGmm &gmm);
/// istream operator that calls FullGmm::Read()
std::istream &
operator >> (std::istream & rIn, kaldi::FullGmm &gmm);

}  // End namespace kaldi

#include "gmm/full-gmm-inl.h"  // templated functions.

#endif  // KALDI_GMM_FULL_GMM_H_

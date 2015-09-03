// gmm/full-gmm.h

// Copyright 2009-2011  Jan Silovsky;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Microsoft Corporation
//           2012       Arnab Ghoshal
//           2013       Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_GMM_FULL_GMM_H_
#define KALDI_GMM_FULL_GMM_H_

#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "gmm/model-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

class DiagGmm;
class FullGmmNormal;  // a simplified representation, see full-gmm-normal.h

/** \class Definition for Gaussian Mixture Model with full covariances
  */
class FullGmm {
  /// this makes it a little easier to modify the internals
  friend class FullGmmNormal;

 public:
  /// Empty constructor.
  FullGmm() : valid_gconsts_(false) {}

  explicit FullGmm(const FullGmm &gmm): valid_gconsts_(false) {
    CopyFromFullGmm(gmm);
  }

  FullGmm(int32 nMix, int32 dim): valid_gconsts_(false) { Resize(nMix, dim); }

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
                      Vector<BaseFloat> *loglikes) const;

  /// Outputs the per-component log-likelihoods of a subset of mixture
  /// components. Note: indices.size() will equal loglikes->Dim() at output.
  /// loglikes[i] will correspond to the log-likelihood of the Gaussian
  /// indexed indices[i].
  void LogLikelihoodsPreselect(const VectorBase<BaseFloat> &data,
                               const std::vector<int32> &indices,
                               Vector<BaseFloat> *loglikes) const;

  /// Get gaussian selection information for one frame.  Returns log-like for
  /// this frame.  Output is the best "num_gselect" indices, sorted from best to
  /// worst likelihood.  If "num_gselect" > NumGauss(), sets it to NumGauss().
  BaseFloat GaussianSelection(const VectorBase<BaseFloat> &data,
                              int32 num_gselect,
                              std::vector<int32> *output) const;

  /// Get gaussian selection information for one frame.  Returns log-like for
  /// this frame.  Output is the best "num_gselect" indices that were
  /// preselected, sorted from best to worst likelihood.  If "num_gselect" >
  /// NumGauss(), sets it to NumGauss().
  BaseFloat GaussianSelectionPreselect(const VectorBase<BaseFloat> &data,
                                       const std::vector<int32> &preselect,
                                       int32 num_gselect,
                                       std::vector<int32> *output) const;
  
  /// Computes the posterior probabilities of all Gaussian components given
  /// a data point. Returns the log-likehood of the data given the GMM.
  BaseFloat ComponentPosteriors(const VectorBase<BaseFloat> &data,
                                VectorBase<BaseFloat> *posterior) const;

  /// Computes the contribution log-likelihood of a data point from a single
  /// Gaussian component. NOTE: Currently we make no guarantees about what
  /// happens if one of the variances is zero.
  BaseFloat ComponentLogLikelihood(const VectorBase<BaseFloat> &data,
                                   int32 comp_id) const;

  /// Sets the gconsts.  Returns the number that are "invalid" e.g. because of
  /// zero weights or variances.
  int32 ComputeGconsts();

  /// Merge the components and remember the order in which the components were
  /// merged (flat list of pairs)
  void Split(int32 target_components, float perturb_factor,
             std::vector<int32> *history = NULL);

  /// Perturbs the component means with a random vector multiplied by the
  /// pertrub factor.
  void Perturb(float perturb_factor);

  /// Merge the components and remember the order in which the components were
  /// merged (flat list of pairs)
  void Merge(int32 target_components,
             std::vector<int32> *history = NULL);

  /// Merge the components and remember the order in which the components were
  /// merged (flat list of pairs); this version only considers merging
  /// pairs in "preselect_pairs" (or their descendants after merging).
  /// This is for efficiency, for large models.  Returns the delta likelihood.
  BaseFloat MergePreselect(int32 target_components,
                           const std::vector<std::pair<int32, int32> > &preselect_pairs);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  /// this = rho x source + (1-rho) x this
  void Interpolate(BaseFloat rho, const FullGmm &source,
                   GmmFlagsType flags = kGmmAll);

  /// Const accessors
  const Vector<BaseFloat> &gconsts() const { return gconsts_; }
  const Vector<BaseFloat> &weights() const { return weights_; }
  const Matrix<BaseFloat> &means_invcovars() const { return means_invcovars_; }
  const std::vector<SpMatrix<BaseFloat> > &inv_covars() const {
    return inv_covars_; }

  /// Non-const accessors
  Matrix<BaseFloat> &means_invcovars() { return means_invcovars_; }
  std::vector<SpMatrix<BaseFloat> > &inv_covars() { return inv_covars_; }

  /// Mutators for both float or double
  template<class Real>
  void SetWeights(const Vector<Real> &w);    ///< Set mixure weights

  /// Use SetMeans to update only the Gaussian means (and not variances)
  template<class Real>
  void SetMeans(const Matrix<Real> &m);

  /// Use SetInvCovarsAndMeans if updating both means and (inverse) covariances
  template<class Real>
  void SetInvCovarsAndMeans(const std::vector<SpMatrix<Real> > &invcovars,
                            const Matrix<Real> &means);

  /// Use this if setting both, in the class's native format.
  template<class Real>
  void SetInvCovarsAndMeansInvCovars(const std::vector<SpMatrix<Real> > &invcovars,
                                     const Matrix<Real> &means_invcovars);

  /// Set the (inverse) covariances and recompute means_invcovars_
  template<class Real>
  void SetInvCovars(const std::vector<SpMatrix<Real> > &v);

  /// Accessor for covariances.
  template<class Real>
  void GetCovars(std::vector<SpMatrix<Real> > *v) const;
  /// Accessor for means.
  template<class Real>
  void GetMeans(Matrix<Real> *m) const;
  /// Accessor for covariances and means
  template<class Real>
  void GetCovarsAndMeans(std::vector< SpMatrix<Real> > *covars,
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
  BaseFloat MergedComponentsLogdet(BaseFloat w1, BaseFloat w2,
                                     const VectorBase<BaseFloat> &f1,
                                     const VectorBase<BaseFloat> &f2,
                                     const SpMatrix<BaseFloat> &s1,
                                     const SpMatrix<BaseFloat> &s2) const;

  const FullGmm &operator=(const FullGmm &other);  // Disallow assignment.
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

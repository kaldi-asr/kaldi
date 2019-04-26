// gmm/diag-gmm.h

// Copyright 2009-2011  Microsoft Corporation;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Georg Stemmer;  Jan Silovsky
//           2012       Arnab Ghoshal
//           2013-2014  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_GMM_DIAG_GMM_H_
#define KALDI_GMM_DIAG_GMM_H_ 1

#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "gmm/model-common.h"
#include "matrix/matrix-lib.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"

namespace kaldi {

class FullGmm;
class DiagGmmNormal;

/// Definition for Gaussian Mixture Model with diagonal covariances
class DiagGmm {
  /// this makes it a little easier to modify the internals
  friend class DiagGmmNormal;

 public:
  /// Empty constructor.
  DiagGmm() : valid_gconsts_(false) { }

  explicit DiagGmm(const DiagGmm &gmm): valid_gconsts_(false) {
    CopyFromDiagGmm(gmm);
  }

  /// Initializer from GaussClusterable initializes the DiagGmm as
  /// a single Gaussian from tree stats.
  DiagGmm(const GaussClusterable &gc, BaseFloat var_floor);

  /// Copies from DiagGmmNormal; does not resize.
  void CopyFromNormal(const DiagGmmNormal &diag_gmm_normal);

  DiagGmm(int32 nMix, int32 dim): valid_gconsts_(false) { Resize(nMix, dim); }

  /// Constructor that allows us to merge GMMs with weights.  Weights must sum
  /// to one, or this GMM will not be properly normalized (we don't check this).
  /// Weights must be positive (we check this).
  explicit DiagGmm(const std::vector<std::pair<BaseFloat, const DiagGmm*> > &gmms);

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

  /// This version of the LogLikelihoods function operates on
  /// a sequence of frames simultaneously; the row index of both "data" and
  /// "loglikes" is the frame index.
  void LogLikelihoods(const MatrixBase<BaseFloat> &data,
                      Matrix<BaseFloat> *loglikes) const;


  /// Outputs the per-component log-likelihoods of a subset of mixture
  /// components.  Note: at output, loglikes->Dim() will equal indices.size().
  /// loglikes[i] will correspond to the log-likelihood of the Gaussian
  /// indexed indices[i], including the mixture weight.
  void LogLikelihoodsPreselect(const VectorBase<BaseFloat> &data,
                               const std::vector<int32> &indices,
                               Vector<BaseFloat> *loglikes) const;

  /// Get gaussian selection information for one frame.  Returns log-like
  /// this frame.  Output is the best "num_gselect" indices, sorted from best to
  /// worst likelihood.  If "num_gselect" > NumGauss(), sets it to NumGauss().
  BaseFloat GaussianSelection(const VectorBase<BaseFloat> &data,
                              int32 num_gselect,
                              std::vector<int32> *output) const;

  /// This version of the Gaussian selection function works for a sequence
  /// of frames rather than just a single frame.  Returns sum of the log-likes
  /// over all frames.
  BaseFloat GaussianSelection(const MatrixBase<BaseFloat> &data,
                              int32 num_gselect,
                              std::vector<std::vector<int32> > *output) const;

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

  /// Split the components and remember the order in which the components were
  /// split
  void Split(int32 target_components, float perturb_factor,
             std::vector<int32> *history = NULL);

  /// Perturbs the component means with a random vector multiplied by the
  /// pertrub factor.
  void Perturb(float perturb_factor);

  /// Merge the components and remember the order in which the components were
  /// merged (flat list of pairs)
  void Merge(int32 target_components, std::vector<int32> *history = NULL);

  // Merge the components to a specified target #components: this
  // version uses a different approach based on K-means.
  void MergeKmeans(int32 target_components,
                   ClusterKMeansOptions cfg = ClusterKMeansOptions());

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &in, bool binary);

  /// this = rho x source + (1-rho) x this
  void Interpolate(BaseFloat rho, const DiagGmm &source,
                   GmmFlagsType flags = kGmmAll);

  /// this = rho x source + (1-rho) x this
  void Interpolate(BaseFloat rho, const FullGmm &source,
                   GmmFlagsType flags = kGmmAll);

  /// Const accessors
  const Vector<BaseFloat> &gconsts() const {
    KALDI_ASSERT(valid_gconsts_);
    return gconsts_;
  }
  const Vector<BaseFloat> &weights() const { return weights_; }
  const Matrix<BaseFloat> &means_invvars() const { return means_invvars_; }
  const Matrix<BaseFloat> &inv_vars() const { return inv_vars_; }
  bool valid_gconsts() const { return valid_gconsts_; }

  /// Removes single component from model
  void RemoveComponent(int32 gauss, bool renorm_weights);

  /// Removes multiple components from model; "gauss" must not have dups.
  void RemoveComponents(const std::vector<int32> &gauss, bool renorm_weights);

  /// Mutators for both float or double
  template<class Real>
  void SetWeights(const VectorBase<Real> &w);    ///< Set mixure weights

  /// Use SetMeans to update only the Gaussian means (and not variances)
  template<class Real>
  void SetMeans(const MatrixBase<Real> &m);
  /// Use SetInvVarsAndMeans if updating both means and (inverse) variances
  template<class Real>
  void SetInvVarsAndMeans(const MatrixBase<Real> &invvars,
                          const MatrixBase<Real> &means);
  /// Set the (inverse) variances and recompute means_invvars_
  template<class Real>
  void SetInvVars(const MatrixBase<Real> &v);

  /// Accessor for covariances.
  template<class Real>
  void GetVars(Matrix<Real> *v) const;
  /// Accessor for means.
  template<class Real>
  void GetMeans(Matrix<Real> *m) const;

  /// Mutators for single component, supports float or double
  /// Set mean for a single component - internally multiplies with inv(var)
  template<class Real>
  void SetComponentMean(int32 gauss, const VectorBase<Real> &in);
  /// Set inv-var for single component (recommend to do this before
  /// setting the mean, if doing both, for numerical reasons).
  template<class Real>
  void SetComponentInvVar(int32 gauss, const VectorBase<Real> &in);
  /// Set weight for single component.
  inline void SetComponentWeight(int32 gauss, BaseFloat weight);

  /// Accessor for single component mean
  template<class Real>
  void GetComponentMean(int32 gauss, VectorBase<Real> *out) const;

  /// Accessor for single component variance.
  template<class Real>
  void GetComponentVariance(int32 gauss, VectorBase<Real> *out) const;

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

 private:
  const DiagGmm &operator=(const DiagGmm &other);  // Disallow assignment
};

/// ostream operator that calls DiagGMM::Write()
std::ostream &
operator << (std::ostream &os, const kaldi::DiagGmm &gmm);
/// istream operator that calls DiagGMM::Read()
std::istream &
operator >> (std::istream &is, kaldi::DiagGmm &gmm);

}  // End namespace kaldi

#include "gmm/diag-gmm-inl.h"  // templated functions.

#endif  // KALDI_GMM_DIAG_GMM_H_

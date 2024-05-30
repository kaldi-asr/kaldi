// gmm/mle-full-gmm.h

// Copyright 2009-2011  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation;
//                      Univ. Erlangen Nuremberg, Korbinian Riedhammer

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

#ifndef KALDI_GMM_MLE_FULL_GMM_H_
#define KALDI_GMM_MLE_FULL_GMM_H_

#include <vector>

#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/full-gmm-normal.h"
#include "gmm/mle-diag-gmm.h"  // for AugmentGmmFlags()

namespace kaldi {

/** \struct MleFullGmmOptions
 *  Configuration variables like variance floor, minimum occupancy, etc.
 *  needed in the estimation process.
 */
struct MleFullGmmOptions {
  /// Minimum weight below which a Gaussian is removed
  BaseFloat min_gaussian_weight;
  /// Minimum occupancy count below which a Gaussian is removed
  BaseFloat min_gaussian_occupancy;
  /// Floor on eigenvalues of covariance matrices
  BaseFloat variance_floor;
  /// Maximum condition number of covariance matrices (apply
  /// floor to eigenvalues if they pass this).
  BaseFloat max_condition;
  bool remove_low_count_gaussians;
  MleFullGmmOptions() {
    min_gaussian_weight    = 1.0e-05;
    min_gaussian_occupancy     = 100.0;
    variance_floor         = 0.001;
    max_condition          = 1.0e+04;
    remove_low_count_gaussians = true;
  }
  void Register(OptionsItf *opts) {
    std::string module = "MleFullGmmOptions: ";
    opts->Register("min-gaussian-weight", &min_gaussian_weight,
                 module+"Min Gaussian weight before we remove it.");
    opts->Register("min-gaussian-occupancy", &min_gaussian_occupancy,
                 module+"Minimum count before we remove a Gaussian.");
    opts->Register("variance-floor", &variance_floor,
                 module+"Minimum eigenvalue of covariance matrix.");
    opts->Register("max-condition", &max_condition,
                 module+"Maximum condition number of covariance matrix (use it to floor).");
    opts->Register("remove-low-count-gaussians", &remove_low_count_gaussians,
                 module+"If true, remove Gaussians that fall below the floors.");
  }
};

/** Class for computing the maximum-likelihood estimates of the parameters of
 *  a Gaussian mixture model.
 */
class AccumFullGmm {
 public:
  AccumFullGmm(): dim_(0), num_comp_(0), flags_(0) { }
  AccumFullGmm(int32 num_comp, int32 dim, GmmFlagsType flags):
      dim_(0), num_comp_(0), flags_(0) {
    Resize(num_comp, dim, flags);
  }
  explicit AccumFullGmm(const FullGmm &gmm, GmmFlagsType flags) {
    Resize(gmm, flags);
  }
  // provide copy constructor.
  explicit AccumFullGmm(const AccumFullGmm &other);

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Allocates memory for accumulators
  void Resize(int32 num_components, int32 dim, GmmFlagsType flags);
  /// Calls Resize with arguments based on gmm_ptr_
  void Resize(const FullGmm &gmm, GmmFlagsType flags);

  void ResizeVarAccumulator(int32 num_comp, int32 dim);
  /// Returns the number of mixture components
  int32 NumGauss() const { return num_comp_; }
  /// Returns the dimensionality of the feature vectors
  int32 Dim() const { return dim_; }

  void SetZero(GmmFlagsType flags);

  void Scale(BaseFloat f, GmmFlagsType flags);  // scale stats.

  /// Accumulate for a single component, given the posterior
  void AccumulateForComponent(const VectorBase<BaseFloat> &data,
                              int32 comp_index, BaseFloat weight);

  /// Accumulate for all components, given the posteriors.
  void AccumulateFromPosteriors(const VectorBase<BaseFloat> &data,
                                const VectorBase<BaseFloat> &gauss_posteriors);

  /// Accumulate for all components given a full-covariance GMM.
  /// Computes posteriors and returns log-likelihood
  BaseFloat AccumulateFromFull(const FullGmm &gmm,
                               const VectorBase<BaseFloat> &data,
                               BaseFloat frame_posterior);

  /// Accumulate for all components given a diagonal-covariance GMM.
  /// Computes posteriors and returns log-likelihood
  BaseFloat AccumulateFromDiag(const DiagGmm &gmm,
                               const VectorBase<BaseFloat> &data,
                               BaseFloat frame_posterior);

  /// Accessors
  GmmFlagsType Flags() const { return flags_; }
  const Vector<double> &occupancy() const { return occupancy_; }
  const Matrix<double> &mean_accumulator() const { return mean_accumulator_; }
  const std::vector<SpMatrix<double> > &covariance_accumulator() const { return covariance_accumulator_; }

 private:
  int32 dim_;
  int32 num_comp_;
  GmmFlagsType flags_;

  Vector<double> occupancy_;
  Matrix<double> mean_accumulator_;
  std::vector<SpMatrix<double> > covariance_accumulator_;
};

inline void AccumFullGmm::Resize(const FullGmm &gmm, GmmFlagsType flags) {
  Resize(gmm.NumGauss(), gmm.Dim(), flags);
}

/// for computing the maximum-likelihood estimates of the parameters of a
/// Gaussian mixture model.  Update using the FullGmm exponential form
void MleFullGmmUpdate(const MleFullGmmOptions &config,
            const AccumFullGmm &fullgmm_acc,
            GmmFlagsType flags,
            FullGmm *gmm,
            BaseFloat *obj_change_out,
            BaseFloat *count_out);

/// Calc using the DiagGMM exponential form
BaseFloat MlObjective(const FullGmm &gmm,
                      const AccumFullGmm &fullgmm_acc);

}  // End namespace kaldi

#endif  // KALDI_GMM_MLE_FULL_GMM_H_

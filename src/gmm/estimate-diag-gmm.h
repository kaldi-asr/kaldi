// gmm/estimate-diag-gmm.h

// Copyright 2009-2011  Saarland University;  Georg Stemmer;
//                      Microsoft Corporation;  Jan Silovsky

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


#ifndef KALDI_GMM_ESTIMATE_DIAG_GMM_H_
#define KALDI_GMM_ESTIMATE_DIAG_GMM_H_ 1

#include "gmm/diag-gmm.h"
#include "gmm/model-common.h"
#include "util/parse-options.h"

namespace kaldi {

/** \struct MleDiagGmmOptions
 *  Configuration variables like variance floor, minimum occupancy, etc.
 *  needed in the estimation process.
 */
struct MleDiagGmmOptions {
  /// Flags to control which parameters to update

  /// Variance floor for each dimension [empty if not supplied].
  Vector<BaseFloat> variance_floor_vector;
  /// Minimum weight below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true);
  BaseFloat min_gaussian_weight;
  /// Minimum count below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true).
  BaseFloat min_gaussian_occupancy;
  /// Minimum allowed variance in any dimension (if no variance floor)
  BaseFloat min_variance;
  bool remove_low_count_gaussians;
  MleDiagGmmOptions() {
    // don't set var floor vector by default.
    min_gaussian_weight     = 1.0e-05;
    min_gaussian_occupancy  = 10.0;
    min_variance            = 0.001;
    remove_low_count_gaussians = true;
  }
  void Register(ParseOptions *po) {
    std::string module = "MleDiagGmmOptions: ";
    po->Register("min-gaussian-weight", &min_gaussian_weight,
                 module+"Min Gaussian weight before we remove it.");
    po->Register("min-gaussian-occupancy", &min_gaussian_occupancy,
                 module+"Minimum occupancy to update a Gaussian.");
    po->Register("min-variance", &min_variance,
                 module+"Variance floor (absolute variance).");
    po->Register("remove-low-count-gaussians", &remove_low_count_gaussians,
                 module+"If true, remove Gaussians that fall below the floors.");
  }
};

/** Class for computing the maximum-likelihood estimates of the parameters of
 *  a Gaussian mixture model.
 */
class MlEstimateDiagGmm {
 public:
  MlEstimateDiagGmm(): dim_(0), num_comp_(0), flags_(0) { }
  explicit MlEstimateDiagGmm(const DiagGmm &gmm, GmmFlagsType flags) {
    ResizeAccumulators(gmm, flags);
  }
  // provide copy constructor.
  explicit MlEstimateDiagGmm(const MlEstimateDiagGmm &other);

  /// Allocates memory for accumulators
  void ResizeAccumulators(int32 num_comp, int32 dim, GmmFlagsType flags);
  /// Calls ResizeAccumulators with arguments based on gmm
  void ResizeAccumulators(const DiagGmm &gmm, GmmFlagsType flags);
  /// Returns the number of mixture components
  int32 NumGauss() const { return num_comp_; }
  /// Returns the dimensionality of the feature vectors
  int32 Dim() const { return dim_; }

  void ZeroAccumulators(GmmFlagsType flags);

  void ScaleAccumulators(BaseFloat f, GmmFlagsType flags);

  /// Accumulate for a single component, given the posterior
  void AccumulateForComponent(const VectorBase<BaseFloat>& data,
                              int32 comp_index, BaseFloat weight);

  /// Accumulate for all components, given the posteriors.
  void AccumulateFromPosteriors(const VectorBase<BaseFloat>& data,
                                const VectorBase<BaseFloat>& gauss_posteriors);

  /// Accumulate for all components given a diagonal-covariance GMM.
  /// Computes posteriors and returns log-likelihood
  BaseFloat AccumulateFromDiag(const DiagGmm &gmm,
                               const VectorBase<BaseFloat>& data,
                               BaseFloat frame_posterior);

  void Update(const MleDiagGmmOptions &config,
              GmmFlagsType flags,
              DiagGmm *gmm,
              BaseFloat *obj_change_out,
              BaseFloat *count_out) const;

  BaseFloat MlObjective(const DiagGmm& gmm) const;

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  // Accessors
  const GmmFlagsType Flags() const { return flags_; }
  Vector<double>& occupancy() { return occupancy_; }
  const Vector<double>& occupancy() const { return occupancy_; }

 private:
  static int32 FloorVariance(const MleDiagGmmOptions &config,
                             VectorBase<BaseFloat> *var);

  int32 dim_;
  int32 num_comp_;
  /// Flags corresponding to the accumulators that are stored.
  GmmFlagsType flags_;
  /// Accumulators
  Vector<double> occupancy_;
  Matrix<double> mean_accumulator_;
  Matrix<double> variance_accumulator_;

  /// Returns "augmented" version of flags: e.g. if just updating means, need
  /// weights too.
  static GmmFlagsType AugmentFlags(GmmFlagsType f);

  // Disallow assignment operator.
  MlEstimateDiagGmm &operator = (const MlEstimateDiagGmm &other);
};

inline void MlEstimateDiagGmm::ResizeAccumulators(const DiagGmm &gmm,
                                                  GmmFlagsType flags) {
  ResizeAccumulators(gmm.NumGauss(), gmm.Dim(), flags);
}

}  // End namespace kaldi


#endif  // KALDI_GMM_ESTIMATE_DIAG_GMM_H_

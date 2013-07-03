// tied/mle-tied-gmm.h

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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


#ifndef KALDI_TIED_MLE_TIED_GMM_H_
#define KALDI_TIED_MLE_TIED_GMM_H_ 1

#include <string>

#include "gmm/model-common.h"
#include "tied/tied-gmm.h"
#include "itf/options-itf.h"

namespace kaldi {

/** \struct MleTiedGmmOptions
 *  Configuration variables like minimum Gaussian weight
 */
struct MleTiedGmmOptions {
  /// minimum component weight
  BaseFloat min_gaussian_weight;

  /// minimum total occupancy for update
  BaseFloat min_gaussian_occupancy;

  /// smoothing weight to smooth the newly estimated parameters with the old
  BaseFloat interpolation_weight;

  bool interpolate_weights;
  bool interpolate_means;
  bool interpolate_variances;

  MleTiedGmmOptions() {
    min_gaussian_weight     = 0.01;  ///< floor = weight / numgauss
    min_gaussian_occupancy  = 3.0;

    interpolation_weight    = 0.5;   ///< interpolation with prior iteration

    interpolate_weights   = false;
    interpolate_means     = false;
    interpolate_variances = false;
  }

  void Register(OptionsItf *po) {
    std::string module = "MleTiedGmmOptions: ";
    po->Register("min-tied-gaussian-weight", &min_gaussian_weight,
                 module+"Minimum gaussian weight w.r.t. the number of "
                 "components (floor = weight / num_comp)");
    po->Register("min-tied-gaussian-occupancy", &min_gaussian_occupancy,
                 module+"Minimum occupancy to update a tied Gaussian.");
    po->Register("interpolation-weight", &interpolation_weight,
                 module+"Interpolate new estimate with prior estimate "
                 "new = weigth x prior-est + (1-weight) x new-est");
    po->Register("interpolate-weights", &interpolate_weights,
                 module+"Interpolate tied mixture weights with prior "
                 "iteration.");
    po->Register("interpolate-means", &interpolate_means,
                 module+"Interpolate codebook means with prior iteration.");
    po->Register("interpolate-variances", &interpolate_variances,
                 module+"Interpolate codebook variances with prior "
                 "iteration.");
  }

  bool interpolate() const {
    return interpolate_weights || interpolate_means || interpolate_variances;
  }
};

class AccumTiedGmm {
 public:
  AccumTiedGmm(): num_comp_(0), flags_(0) { }

  explicit AccumTiedGmm(const TiedGmm &tied, GmmFlagsType flags) {
    Resize(tied, flags);
  }

  // provide copy constructor.
  explicit AccumTiedGmm(const AccumTiedGmm &other);

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Allocates memory for accumulators
  void Resize(int32 num_comp, GmmFlagsType flags);
  /// Calls ResizeAccumulators with arguments based on gmm
  void Resize(const TiedGmm &tied, GmmFlagsType flags);

  /// Returns the number of mixture components
  int32 NumGauss() const { return num_comp_; }

  void SetZero(GmmFlagsType flags);
  void Scale(BaseFloat f, GmmFlagsType flags);

  /// Accumulate for a single component, given the posterior
  void AccumulateForComponent(int32 comp_index, BaseFloat weight);

  /// Accumulate for all components, given the posteriors.
  void AccumulateFromPosteriors(const VectorBase<BaseFloat> &gauss_posteriors);

  /// Propagate the sufficient statistics to the target accumulator
  void Propagate(AccumTiedGmm *target) const;

  /// Interpolate this accumulator with the source depending on the occupancies
  /// tau' <- tau / (tau + occupancy_.Sum())
  /// this <- tau' x source + (1-tau') x this
  /// i.e., if the occupancy is high, tau' vanishes, and the local stats are
  /// kept; if gamma is zero, the stats are completely replaced by the source.
  /// Note that this does not preserve the occupancies and thus may distort
  /// the estimate.
  void Interpolate1(BaseFloat tau, const AccumTiedGmm &source);

  /// Interpolate this accumulator with the source but preserve the
  /// occupancies (different from Interpolate1)
  void Interpolate2(BaseFloat tau, const AccumTiedGmm &source);

  // Accessors
  const GmmFlagsType Flags() const { return flags_; }
  const Vector<double> &occupancy() const { return occupancy_; }

 private:
  int32 num_comp_;
  GmmFlagsType flags_;
  Vector<double> occupancy_;
};

inline void AccumTiedGmm::Resize(const TiedGmm &tied, GmmFlagsType flags) {
  Resize(tied.NumGauss(), flags);
}

/// for computing the maximum-likelihood estimates of the parameters of
/// a Gaussian mixture model.
/// Update using the DiagGmm: exponential form
void MleTiedGmmUpdate(const MleTiedGmmOptions &config,
            const AccumTiedGmm &tiedgmm_acc,
            GmmFlagsType flags,
            TiedGmm *tied,
            BaseFloat *obj_change_out,
            BaseFloat *count_out);

/// Only considering the weights
BaseFloat MlObjective(const TiedGmm &tied, const AccumTiedGmm &tiedgmm_acc);

}  // End namespace kaldi


#endif  // KALDI_TIED_MLE_TIED_GMM_H_

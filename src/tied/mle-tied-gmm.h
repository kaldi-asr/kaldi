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

#include "gmm/model-common.h"
#include "tied/tied-gmm.h"
#include "util/parse-options.h"

namespace kaldi {

/** \struct MleTiedGmmOptions
 *  Configuration variables like minimum Gaussian weight
 */
struct MleTiedGmmOptions {
  /// minimum component weight
  BaseFloat min_gaussian_weight;
  
  /// minimum total occupancy for update
  BaseFloat min_gaussian_occupancy;
  
  /// interpolation weight to smooth the newly estimated parameters with the old ones
  BaseFloat interpolation_weight;

  bool interpolate_weights;
  bool interpolate_means;
  bool interpolate_variances;

  MleTiedGmmOptions() {
    /// gaussian weight should be rather small, due to large (> 512) code book sizes
    min_gaussian_weight     = 1.0e-8;
    min_gaussian_occupancy  = 3.0;

    // apis stuff
    interpolation_weight    = 0.5;

    interpolate_weights   = true;
    interpolate_means     = false;
    interpolate_variances = false;
  }
  
  void Register(ParseOptions *po) {
    std::string module = "MleTiedGmmOptions: ";
    po->Register("min-tied-gaussian-weight", &min_gaussian_weight,
                 module+"Min tied Gaussian weight to enforce.");
    po->Register("min-tied-gaussian-occupancy", &min_gaussian_occupancy,
                 module+"Minimum occupancy to update a tied Gaussian.");
    po->Register("interpolation-weight", &interpolation_weight,
                 module+"Interpolation weight to smooth the newly estimated parameters with the old ones. new <- wt*est + (1-wt*old)");
    po->Register("interpolate-weights", &interpolate_weights,
                 module+"Interpolate tied mixture weights?");
    po->Register("interpolate-means", &interpolate_means,
                 module+"Interpolate codebook means?");
    po->Register("interpolate-variances", &interpolate_variances,
                 module+"Interpolate codebook variances?");
  }

  bool interpolate() const { return interpolate_weights || interpolate_means || interpolate_variances; }
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
  void AccumulateFromPosteriors(const VectorBase<BaseFloat>& gauss_posteriors);

  /// Smooths the accumulated counts by adding 'tau' extra frames. An example
  /// use for this is I-smoothing for MMIE/MPE.
  void SmoothStats(BaseFloat tau);

  /// Smooths the accumulated counts using some other accumulator. Performs
  /// a weighted sum of the current accumulator with the given one. Both
  /// accumulators must have the same dimension and number of components.
  void SmoothWithAccum(BaseFloat tau, const AccumTiedGmm& src_acc);

  // Accessors
  const GmmFlagsType Flags() const { return flags_; }
  const Vector<double>& occupancy() const { return occupancy_; }

 private:
  int32 num_comp_;
  GmmFlagsType flags_;
  Vector<double> occupancy_;
  
  /// Returns "augmented" version of flags: e.g. if just updating means, need
  /// weights too.
  static GmmFlagsType AugmentFlags(GmmFlagsType f);
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

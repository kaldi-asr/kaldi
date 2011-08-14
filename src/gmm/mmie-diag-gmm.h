// gmm/mmie-diag-gmm.h

// Copyright 2009-2011  Arnab Ghoshal

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


#ifndef KALDI_GMM_MMIE_DIAG_GMM_H_
#define KALDI_GMM_MMIE_DIAG_GMM_H_ 1

#include <string>

#include "gmm/diag-gmm.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/model-common.h"
#include "util/parse-options.h"

namespace kaldi {

/** \struct MmieDiagGmmOptions
 *  Configuration variables like variance floor, minimum occupancy, etc.
 *  needed in the estimation process.
 */
struct MmieDiagGmmOptions : public MleDiagGmmOptions {
  BaseFloat i_smooth_tau;
  BaseFloat ebw_e;
  MmieDiagGmmOptions() : MleDiagGmmOptions() {
    i_smooth_tau = 100.0;
    ebw_e = 2.0;
  }
  void Register(ParseOptions *po) {
    std::string module = "MmieDiagGmmOptions: ";
    po->Register("min-gaussian-weight", &min_gaussian_weight,
                 module+"Min Gaussian weight before we remove it.");
    po->Register("min-gaussian-occupancy", &min_gaussian_occupancy,
                 module+"Minimum occupancy to update a Gaussian.");
    po->Register("min-variance", &min_variance,
                 module+"Variance floor (absolute variance).");
    po->Register("remove-low-count-gaussians", &remove_low_count_gaussians,
                 module+"If true, remove Gaussians that fall below the floors.");
    po->Register("i-smooth-tau", &i_smooth_tau,
                 module+"Coefficient for I-smoothing.");
    po->Register("ebw-e", &ebw_e, module+"Smoothing constant for EBW update.");
  }
};


/** Class for computing the maximum mutual information estimate of the
 *  parameters of a Gaussian mixture model.
 */
class MmieDiagGmm {
 public:
  MmieDiagGmm(): dim_(0), num_comp_(0), flags_(0) {}

  /// Computes the difference between the numerator and denominator accumulators
  /// and applies I-smoothing to the numerator accs, if needed.
  void SubtractAccumulators(const AccumDiagGmm& num_acc,
                            const AccumDiagGmm& den_acc,
                            const MmieDiagGmmOptions& opts);

  void Update(const MmieDiagGmmOptions &config,
              GmmFlagsType flags,
              DiagGmm *gmm,
              BaseFloat *obj_change_out,
              BaseFloat *count_out) const;

  BaseFloat MmiObjective(const DiagGmm& gmm) const;

 private:
  /// Accumulators
  // TODO(arnab): not decided yet whether to store the difference or keep the
  //              num and den accs for mean and var.
  Vector<double> num_occupancy_;
  Vector<double> den_occupancy_;
  Matrix<double> mean_accumulator_;
  Matrix<double> variance_accumulator_;

  BaseFloat ComputeD(const DiagGmm& old_gmm, int32 mix_index, BaseFloat ebw_e);

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(MmieDiagGmm);
};

inline void AccumDiagGmm::Resize(const DiagGmm &gmm, GmmFlagsType flags) {
  Resize(gmm.NumGauss(), gmm.Dim(), flags);
}

}  // End namespace kaldi


#endif  // KALDI_GMM_MMIE_DIAG_GMM_H_

// gmm/mmie-diag-gmm.h

// Copyright 2009-2011  Petr Motlicek, Arnab Ghoshal 

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
  BaseFloat min_count_weight_update;
  MmieDiagGmmOptions() : MleDiagGmmOptions() {
    i_smooth_tau = 100.0;
    ebw_e = 2.0;
    min_count_weight_update = 10.0;
  }
  void Register(ParseOptions *po) {
    std::string module = "MmieDiagGmmOptions: ";
    po->Register("min-gaussian-weight", &min_gaussian_weight,
                 module+"Min Gaussian weight before we remove it.");
    po->Register("min-count-weight-update", &min_count_weight_update,
                 module+"Minimum state-level numerator count required to do the weight update");
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
class MmieAccumDiagGmm {
 public:
  MmieAccumDiagGmm(): dim_(0), num_comp_(0), flags_(0) {}
  //MmieDiagGmm() {}
  explicit MmieAccumDiagGmm(const DiagGmm &gmm, GmmFlagsType flags) {
    Resize(gmm.NumGauss(), gmm.Dim(), flags);
  }

  // provide copy constructor.
  explicit MmieAccumDiagGmm(const MmieAccumDiagGmm &other);


  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Allocates memory for accumulators
  void Resize(int32 num_comp, int32 dim, GmmFlagsType flags);
/// Calls ResizeAccumulators with arguments based on gmm
  void Resize(const DiagGmm &gmm, GmmFlagsType flags);


  /// Returns the number of mixture components
  int32 NumGauss() const { return num_comp_; }
  /// Returns the dimensionality of the feature vectors
  int32 Dim() const { return dim_; }

  void SetZero(GmmFlagsType flags);
  void Scale(BaseFloat f, GmmFlagsType flags);


  /// Computes the difference between the numerator and denominator accumulators
  /// and applies I-smoothing to the numerator accs, if needed.
  void SubtractAccumulatorsISmoothing(const AccumDiagGmm& num_acc,
                                      const AccumDiagGmm& den_acc,
                                      const MmieDiagGmmOptions& opts);

  /// Does EBW update on one diagonal Gaussian; returns true if resulting
  /// variance was all positive.
  bool EBWUpdateGaussian(
      BaseFloat D,
      GmmFlagsType flags,
      const VectorBase<double> &orig_mean,
      const VectorBase<double> &orig_var,
      const VectorBase<double> &x_stats,
      const VectorBase<double> &x2_stats,
      double occ,
      VectorBase<double> *mean,
      VectorBase<double> *var,
      double *auxf_impr) const;
  
  /// MMIE update
  void Update(const MmieDiagGmmOptions &config,
              GmmFlagsType flags,
              DiagGmm *gmm,
              BaseFloat *auxf_change_out_gauss, // gets set to EBW auxf impr.
              BaseFloat *auxf_change_out_weights, // auxf impr in weights auxf.
              BaseFloat *count_out, // gets set to numerator count.
              int32 *num_floored_out) const;
  


  // Accessors
  const GmmFlagsType Flags() const { return flags_; }
  const Vector<double>& num_occupancy() const { return num_occupancy_; }
  const Vector<double>& den_occupancy() const { return den_occupancy_; }
  const Vector<double>& occupancy() const { return occupancy_; }  
  const Matrix<double>& mean_accumulator() const { return mean_accumulator_; }
  const Matrix<double>& variance_accumulator() const { return variance_accumulator_; }


  private:
  int32 dim_;
  int32 num_comp_;
  /// Flags corresponding to the accumulators that are stored.
  GmmFlagsType flags_;

  /// Accumulators.
  /// We store the difference of mean and var; we keep occupancy
  /// for num and den and their difference (with I-smoothing)
  
  Vector<double> num_occupancy_; // Numerator occupancy
  Vector<double> den_occupancy_; // Denominator occupancy
  Vector<double> occupancy_; // Num-Den occupancy *plus I-smoothing*
  Matrix<double> mean_accumulator_; // Sum of num-den+I-smooth stats.
  Matrix<double> variance_accumulator_;  // Sum of num-den+I-smooth stats.

  //  BaseFloat ComputeD(const DiagGmm& old_gmm, int32 mix_index, BaseFloat ebw_e);

  /// Cannot have copy constructor and assigment operator
  //KALDI_DISALLOW_COPY_AND_ASSIGN(MmieDiagGmm);
};


inline void MmieAccumDiagGmm::Resize(const DiagGmm &gmm, GmmFlagsType flags) {
  Resize(gmm.NumGauss(), gmm.Dim(), flags);
}


}  // End namespace kaldi


#endif  // KALDI_GMM_MMIE_DIAG_GMM_H_

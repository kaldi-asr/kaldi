// sgmm/estimate-am-sgmm-ebw.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_SGMM_ESTIMATE_AM_SGMM_EBW_H_
#define KALDI_SGMM_ESTIMATE_AM_SGMM_EBW_H_ 1

#include <string>
#include <vector>

#include "gmm/model-common.h"
#include "itf/options-itf.h"
#include "sgmm/estimate-am-sgmm.h"

namespace kaldi {

/**
   This header implements a form of Extended Baum-Welch training for SGMMs.
   If you are confused by this comment, see Dan Povey's thesis for an explanation of
   Extended Baum-Welch.
   A note on the EBW (Extended Baum-Welch) updates for the SGMMs... In general there is
   a parameter-specific value D that is similar to the D in EBW for GMMs.  The value of
   D is generally set to:
     E * (denominator-count for that parameter)   +   tau-value for that parameter
   where the tau-values are user-specified parameters that are specific to the type of
   the parameter (e.g. phonetic vector, subspace projection, etc.).  Things are a bit
   more complex for this update than for GMMs, because it's not just a question of picking
   a tau-value for smoothing: there is sometimes a scatter-matrix of some kind (e.g.
   an outer product of vectors, or something) that defines a quadratic objective function
   that we'll add as smoothing.  We have to pick where to get this scatter-matrix from.
   We feel that it's appropriate for the "E" part of the D to get its scatter-matrix from
   denominator stats, and the tau part of the D to get half its scatter-matrix from the
   both the numerator and denominator stats, assigned a weight proportional to how much
   stats there were.  When you see the auxiliary function written out, it's clear why this
   makes sense.

 */

struct EbwAmSgmmOptions {
  BaseFloat tau_v; ///<  Smoothing constant for updates of sub-state vectors v_{jm}
  BaseFloat lrate_v; ///< Learning rate used in updating v-- default 0.5
  BaseFloat tau_M; ///<  Smoothing constant for the M quantities (phone-subspace projections)
  BaseFloat lrate_M; ///< Learning rate used in updating M-- default 0.5
  BaseFloat tau_N; ///<  Smoothing constant for the N quantities (speaker-subspace projections)
  BaseFloat lrate_N; ///< Learning rate used in updating N-- default 0.5
  BaseFloat tau_c;  ///< Tau value for smoothing substate weights (c)
  BaseFloat tau_w;  ///< Tau value for smoothing update of weight projectsions (w)
  BaseFloat lrate_w; ///< Learning rate used in updating w-- default 0.5
  BaseFloat tau_Sigma; ///< Tau value for smoothing covariance-matrices Sigma.
  BaseFloat lrate_Sigma; ///< Learning rate used in updating Sigma-- default 0.5
  BaseFloat min_substate_weight; ///< Minimum allowed weight in a sub-state.

  BaseFloat cov_min_value; ///< E.g. 0.5-- the maximum any eigenvalue of a covariance
  /// is allowed to change.  [this is the minimum; the maximum is the inverse of this,
  /// i.e. 2.0 in this case.  For example, 0.9 would constrain the covariance quite tightly,
  /// 0.1 would be a loose setting.
  
  BaseFloat max_cond; ///< large value used in SolveQuadraticProblem.
  BaseFloat epsilon;  ///< very small value used in SolveQuadraticProblem; workaround
  /// for an issue in some implementations of SVD.
  
  EbwAmSgmmOptions() {
    tau_v = 50.0;
    lrate_v = 0.5;
    tau_M = 500.0;
    lrate_M = 0.5;
    tau_N = 500.0;
    lrate_N = 0.5;
    tau_c = 10.0;
    tau_w = 50.0;
    lrate_w = 1.0;
    tau_Sigma = 500.0;
    lrate_Sigma = 0.5;

    min_substate_weight = 1.0e-05;
    cov_min_value = 0.5;
    
    max_cond = 1.0e+05;
    epsilon = 1.0e-40;
  }

  void Register(OptionsItf *po) {
    std::string module = "EbwAmSgmmOptions: ";
    po->Register("tau-v", &tau_v, module+
                 "Smoothing constant for phone vector estimation.");
    po->Register("lrate-v", &lrate_v, module+
                 "Learning rate constant for phone vector estimation.");
    po->Register("tau-m", &tau_M, module+
                 "Smoothing constant for estimation of phonetic-subspace projections (M).");
    po->Register("lrate-m", &lrate_M, module+
                 "Learning rate constant for phonetic-subspace projections.");
    po->Register("tau-n", &tau_N, module+
                 "Smoothing constant for estimation of speaker-subspace projections (N).");
    po->Register("lrate-n", &lrate_N, module+
                 "Learning rate constant for speaker-subspace projections.");
    po->Register("tau-c", &tau_c, module+
                 "Smoothing constant for estimation of substate weights (c)");
    po->Register("tau-w", &tau_w, module+
                 "Smoothing constant for estimation of weight projections (w)");
    po->Register("lrate-w", &lrate_w, module+
                 "Learning rate constant for weight-projections");
    po->Register("tau-sigma", &tau_Sigma, module+
                 "Smoothing constant for estimation of within-class covariances (Sigma)");
    po->Register("lrate-sigma", &lrate_Sigma, module+
                 "Constant that controls speed of learning for variances (larger->slower)");
    po->Register("cov-min-value", &cov_min_value, module+
                 "Minimum value that an eigenvalue of the updated covariance matrix can take, "
                 "relative to its old value (maximum is inverse of this.)");
    po->Register("min-substate-weight", &min_substate_weight, module+
                 "Floor for weights of sub-states.");
    po->Register("max-cond", &max_cond, module+
                 "Value used in handling singular matrices during update.");
    po->Register("epsilon", &max_cond, module+
                 "Value used in handling singular matrices during update.");
  }
};


/** \class EbwAmSgmmUpdater
 *  Contains the functions needed to update the SGMM parameters.
 */
class EbwAmSgmmUpdater {
 public:
  explicit EbwAmSgmmUpdater(const EbwAmSgmmOptions &options):
      options_(options) {}
  
  void Update(const MleAmSgmmAccs &num_accs,
              const MleAmSgmmAccs &den_accs,
              AmSgmm *model,
              SgmmUpdateFlagsType flags,
              BaseFloat *auxf_change_out,
              BaseFloat *count_out);
    
 protected:
  // The following two classes relate to multi-core parallelization of some
  // phases of the update.
  friend class EbwUpdateWParallelClass;
  friend class EbwUpdatePhoneVectorsClass;
 private:
  EbwAmSgmmOptions options_;

  Vector<double> gamma_j_;  ///< State occupancies

  double UpdatePhoneVectors(const MleAmSgmmAccs &num_accs,
                            const MleAmSgmmAccs &den_accs,
                            AmSgmm *model,
                            const std::vector< SpMatrix<double> > &H) const;
  
  // Called from UpdatePhoneVectors; updates a subset of states
  // (relates to multi-threading).
  void UpdatePhoneVectorsInternal(const MleAmSgmmAccs &num_accs,
                                  const MleAmSgmmAccs &den_accs,
                                  AmSgmm *model,
                                  const std::vector<SpMatrix<double> > &H,
                                  double *auxf_impr,
                                  int32 num_threads,
                                  int32 thread_id) const;
  // Called from UpdatePhoneVectorsInternal
  static void ComputePhoneVecStats(const MleAmSgmmAccs &accs,
                                   const AmSgmm &model,
                                   const std::vector<SpMatrix<double> > &H,
                                   int32 j,
                                   int32 m,
                                   const Vector<double> &w_jm,
                                   double gamma_jm,
                                   Vector<double> *g_jm,
                                   SpMatrix<double> *H_jm);
                                    
  double UpdateM(const MleAmSgmmAccs &num_accs,
                 const MleAmSgmmAccs &den_accs,
                 const std::vector< SpMatrix<double> > &Q_num,
                 const std::vector< SpMatrix<double> > &Q_den,
                 AmSgmm *model) const;
  
  double UpdateN(const MleAmSgmmAccs &num_accs,
                 const MleAmSgmmAccs &den_accs,
                 AmSgmm *model) const;
  
  double UpdateVars(const MleAmSgmmAccs &num_accs,
                    const MleAmSgmmAccs &den_accs,
                    const std::vector< SpMatrix<double> > &S_means,
                    AmSgmm *model) const;

  /// Note: in the discriminative case we do just one iteration of
  /// updating the w quantities.
  double UpdateWParallel(const MleAmSgmmAccs &num_accs,
                         const MleAmSgmmAccs &den_accs,
                         AmSgmm *model);
  
  double UpdateSubstateWeights(const MleAmSgmmAccs &num_accs,
                               const MleAmSgmmAccs &den_accs,
                               AmSgmm *model);

  KALDI_DISALLOW_COPY_AND_ASSIGN(EbwAmSgmmUpdater);
  EbwAmSgmmUpdater() {}  // Prevent unconfigured updater.
};


}  // namespace kaldi


#endif  // KALDI_SGMM_ESTIMATE_AM_SGMM_EBW_H_

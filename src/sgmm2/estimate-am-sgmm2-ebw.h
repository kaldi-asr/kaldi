// sgmm2/estimate-am-sgmm2-ebw.h

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

#ifndef KALDI_SGMM2_ESTIMATE_AM_SGMM2_EBW_H_
#define KALDI_SGMM2_ESTIMATE_AM_SGMM2_EBW_H_ 1

#include <string>
#include <vector>

#include "gmm/model-common.h"
#include "itf/options-itf.h"
#include "sgmm2/estimate-am-sgmm2.h"

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

struct EbwAmSgmm2Options {
  BaseFloat tau_v; ///<  Smoothing constant for updates of sub-state vectors v_{jm}
  BaseFloat lrate_v; ///< Learning rate used in updating v-- default 0.5
  BaseFloat tau_M; ///<  Smoothing constant for the M quantities (phone-subspace projections)
  BaseFloat lrate_M; ///< Learning rate used in updating M-- default 0.5
  BaseFloat tau_N; ///<  Smoothing constant for the N quantities (speaker-subspace projections)
  BaseFloat lrate_N; ///< Learning rate used in updating N-- default 0.5
  BaseFloat tau_c;  ///< Tau value for smoothing substate weights (c)
  BaseFloat tau_w;  ///< Tau value for smoothing update of phonetic-subspace weight projectsions (w)
  BaseFloat lrate_w; ///< Learning rate used in updating w-- default 1.0
  BaseFloat tau_u;  ///< Tau value for smoothing update of speaker-subspace weight projectsions (u)
  BaseFloat lrate_u; ///< Learning rate used in updating u-- default 1.0
  BaseFloat max_impr_u; ///< Maximum improvement/frame allowed for u [0.25, carried over from ML update.]
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
  
  EbwAmSgmm2Options() {
    tau_v = 50.0;
    lrate_v = 0.5;
    tau_M = 500.0;
    lrate_M = 0.5;
    tau_N = 500.0;
    lrate_N = 0.5;
    tau_c = 10.0;
    tau_w = 50.0;
    lrate_w = 1.0;
    tau_u = 50.0;
    lrate_u = 1.0;
    max_impr_u = 0.25;
    tau_Sigma = 500.0;
    lrate_Sigma = 0.5;

    min_substate_weight = 1.0e-05;
    cov_min_value = 0.5;
    
    max_cond = 1.0e+05;
    epsilon = 1.0e-40;
  }

  void Register(OptionsItf *opts) {
    std::string module = "EbwAmSgmm2Options: ";
    opts->Register("tau-v", &tau_v, module+
                   "Smoothing constant for phone vector estimation.");
    opts->Register("lrate-v", &lrate_v, module+
                   "Learning rate constant for phone vector estimation.");
    opts->Register("tau-m", &tau_M, module+
                   "Smoothing constant for estimation of phonetic-subspace projections (M).");
    opts->Register("lrate-m", &lrate_M, module+
                   "Learning rate constant for phonetic-subspace projections.");
    opts->Register("tau-n", &tau_N, module+
                   "Smoothing constant for estimation of speaker-subspace projections (N).");
    opts->Register("lrate-n", &lrate_N, module+
                   "Learning rate constant for speaker-subspace projections.");
    opts->Register("tau-c", &tau_c, module+
                   "Smoothing constant for estimation of substate weights (c)");
    opts->Register("tau-w", &tau_w, module+
                   "Smoothing constant for estimation of phonetic-space weight projections (w)");
    opts->Register("lrate-w", &lrate_w, module+
                   "Learning rate constant for phonetic-space weight-projections (w)");
    opts->Register("tau-u", &tau_u, module+
                   "Smoothing constant for estimation of speaker-space weight projections (u)");
    opts->Register("lrate-u", &lrate_u, module+
                   "Learning rate constant for speaker-space weight-projections (u)");
    opts->Register("tau-sigma", &tau_Sigma, module+
                   "Smoothing constant for estimation of within-class covariances (Sigma)");
    opts->Register("lrate-sigma", &lrate_Sigma, module+
                   "Constant that controls speed of learning for variances (larger->slower)");
    opts->Register("cov-min-value", &cov_min_value, module+
                   "Minimum value that an eigenvalue of the updated covariance matrix can take, "
                   "relative to its old value (maximum is inverse of this.)");
    opts->Register("min-substate-weight", &min_substate_weight, module+
                   "Floor for weights of sub-states.");
    opts->Register("max-cond", &max_cond, module+
                   "Value used in handling singular matrices during update.");
    opts->Register("epsilon", &max_cond, module+
                   "Value used in handling singular matrices during update.");
  }
};


/** \class EbwAmSgmmUpdater
 *  Contains the functions needed to update the SGMM parameters.
 */
class EbwAmSgmm2Updater {
 public:
  explicit EbwAmSgmm2Updater(const EbwAmSgmm2Options &options):
      options_(options) {}
  
  void Update(const MleAmSgmm2Accs &num_accs,
              const MleAmSgmm2Accs &den_accs,
              AmSgmm2 *model,
              SgmmUpdateFlagsType flags,
              BaseFloat *auxf_change_out,
              BaseFloat *count_out);
    
 protected:
  // The following two classes relate to multi-core parallelization of some
  // phases of the update.
  friend class EbwUpdateWClass;
  friend class EbwUpdatePhoneVectorsClass;
 private:
  EbwAmSgmm2Options options_;

  Vector<double> gamma_j_;  ///< State occupancies

  double UpdatePhoneVectors(const MleAmSgmm2Accs &num_accs,
                            const MleAmSgmm2Accs &den_accs,
                            const std::vector< SpMatrix<double> > &H,
                            AmSgmm2 *model) const;
  
  // Called from UpdatePhoneVectors; updates a subset of states
  // (relates to multi-threading).
  void UpdatePhoneVectorsInternal(const MleAmSgmm2Accs &num_accs,
                                  const MleAmSgmm2Accs &den_accs,
                                  const std::vector<SpMatrix<double> > &H,
                                  AmSgmm2 *model,
                                  double *auxf_impr,
                                  int32 num_threads,
                                  int32 thread_id) const;
  // Called from UpdatePhoneVectorsInternal
  static void ComputePhoneVecStats(const MleAmSgmm2Accs &accs,
                                   const AmSgmm2 &model,
                                   const std::vector<SpMatrix<double> > &H,
                                   int32 j1,
                                   int32 m,
                                   const Vector<double> &w_jm,
                                   double gamma_jm,
                                   Vector<double> *g_jm,
                                   SpMatrix<double> *H_jm);
                                    
  double UpdateM(const MleAmSgmm2Accs &num_accs,
                 const MleAmSgmm2Accs &den_accs,
                 const std::vector< SpMatrix<double> > &Q_num,
                 const std::vector< SpMatrix<double> > &Q_den,
                 const Vector<double> &gamma_num,
                 const Vector<double> &gamma_den,
                 AmSgmm2 *model) const;
  
  double UpdateN(const MleAmSgmm2Accs &num_accs,
                 const MleAmSgmm2Accs &den_accs,
                 const Vector<double> &gamma_num,
                 const Vector<double> &gamma_den,
                 AmSgmm2 *model) const;
  
  double UpdateVars(const MleAmSgmm2Accs &num_accs,
                    const MleAmSgmm2Accs &den_accs,
                    const Vector<double> &gamma_num,
                    const Vector<double> &gamma_den,
                    const std::vector< SpMatrix<double> > &S_means,
                    AmSgmm2 *model) const;

  /// Note: in the discriminative case we do just one iteration of
  /// updating the w quantities.
  double UpdateW(const MleAmSgmm2Accs &num_accs,
                 const MleAmSgmm2Accs &den_accs,
                 const Vector<double> &gamma_num,
                 const Vector<double> &gamma_den,
                 AmSgmm2 *model);


  double UpdateU(const MleAmSgmm2Accs &num_accs,
                 const MleAmSgmm2Accs &den_accs,
                 const Vector<double> &gamma_num,
                 const Vector<double> &gamma_den,
                 AmSgmm2 *model);
  
  double UpdateSubstateWeights(const MleAmSgmm2Accs &num_accs,
                               const MleAmSgmm2Accs &den_accs,
                               AmSgmm2 *model);

  KALDI_DISALLOW_COPY_AND_ASSIGN(EbwAmSgmm2Updater);
  EbwAmSgmm2Updater() {}  // Prevent unconfigured updater.
};


}  // namespace kaldi


#endif  // KALDI_SGMM2_ESTIMATE_AM_SGMM2_EBW_H_

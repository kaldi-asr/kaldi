// gmm/ebw-diag-gmm.h

// Copyright 2009-2011  Arnab Ghoshal, Petr Motlicek

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


#ifndef KALDI_GMM_EBW_DIAG_GMM_H_
#define KALDI_GMM_EBW_DIAG_GMM_H_ 1

#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "gmm/model-common.h"
#include "itf/options-itf.h"

namespace kaldi {

// Options for Extended Baum-Welch Gaussian update.
struct EbwOptions {
  BaseFloat E;
  BaseFloat tau; // This is only useful for smoothing "to the model":
  // if you want to smooth to ML stats, you need to use gmm-ismooth-stats
  EbwOptions(): E(2.0), tau(0.0) { }
  void Register(OptionsItf *opts) {
    std::string module = "EbwOptions: ";
    opts->Register("E", &E, module+"Constant E for Extended Baum-Welch (EBW) update");
    opts->Register("tau", &tau, module+"Tau value for smoothing to the model "
                   "parameters only (for smoothing to ML stats, use gmm-ismooth-stats");
  }
};

struct EbwWeightOptions {
  BaseFloat min_num_count_weight_update; // minimum numerator count at state level, before we update.
  BaseFloat min_gaussian_weight;
  BaseFloat tau; // tau value for smoothing stats in weight update.  Should probably
  // be 10.0 or so, leaving it at 0 for back-compatibility.
  EbwWeightOptions(): min_num_count_weight_update(10.0),
                      min_gaussian_weight(1.0e-05),
                      tau(0.0) { }
  void Register(OptionsItf *opts) {
    std::string module = "EbwWeightOptions: ";
    opts->Register("min-num-count-weight-update", &min_num_count_weight_update,
                 module+"Minimum numerator count required at "
                 "state level before we update the weights (only active if tau == 0.0)");
    opts->Register("min-gaussian-weight", &min_gaussian_weight,
                 module+"Minimum Gaussian weight allowed in EBW update of weights");
    opts->Register("weight-tau", &tau,
                 module+"Tau value for smoothing Gaussian weight update.");
  }
};


// Update Gaussian parameters only (no weights)
// The pointer parameters auxf_change_out etc. are incremented, not set.
void UpdateEbwDiagGmm(const AccumDiagGmm &num_stats, // with I-smoothing, if used.
                      const AccumDiagGmm &den_stats,
                      GmmFlagsType flags,
                      const EbwOptions &opts,
                      DiagGmm *gmm,
                      BaseFloat *auxf_change_out,
                      BaseFloat *count_out,
                      int32 *num_floored_out);

void UpdateEbwAmDiagGmm(const AccumAmDiagGmm &num_stats, // with I-smoothing, if used.
                        const AccumAmDiagGmm &den_stats,
                        GmmFlagsType flags,
                        const EbwOptions &opts,
                        AmDiagGmm *am_gmm,
                        BaseFloat *auxf_change_out,
                        BaseFloat *count_out,
                        int32 *num_floored_out);

// Updates the weights using the EBW-like method described in Dan Povey's thesis
// (this method has no tunable parameters).
// The pointer parameters auxf_change_out etc. are incremented, not set.
void UpdateEbwWeightsDiagGmm(const AccumDiagGmm &num_stats, // should have no I-smoothing
                             const AccumDiagGmm &den_stats,
                             const EbwWeightOptions &opts,
                             DiagGmm *gmm,
                             BaseFloat *auxf_change_out,
                             BaseFloat *count_out);

void UpdateEbwWeightsAmDiagGmm(const AccumAmDiagGmm &num_stats, // should have no I-smoothing
                               const AccumAmDiagGmm &den_stats,
                               const EbwWeightOptions &opts,
                               AmDiagGmm *am_gmm,
                               BaseFloat *auxf_change_out,
                               BaseFloat *count_out);

/// I-Smooth the stats.  src_stats and dst_stats do not have to be different.
void IsmoothStatsDiagGmm(const AccumDiagGmm &src_stats,
                         double tau,
                         AccumDiagGmm *dst_stats);

/// Creates stats from the GMM.  Resizes them as needed.
void DiagGmmToStats(const DiagGmm &gmm,
                    GmmFlagsType flags,
                    double state_occ,
                    AccumDiagGmm *dst_stats);

/// Smooth "dst_stats" with "src_stats".  They don't have to be
/// different.
void IsmoothStatsAmDiagGmm(const AccumAmDiagGmm &src_stats,
                           double tau,
                           AccumAmDiagGmm *dst_stats);

/// This version of the I-smoothing function takes a model as input.
void IsmoothStatsAmDiagGmmFromModel(const AmDiagGmm &src_model,
                                    double tau,
                                    AccumAmDiagGmm *dst_stats);



}  // End namespace kaldi


#endif  // KALDI_GMM_EBW_DIAG_GMM_H_

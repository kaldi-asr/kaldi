// tied/mle-am-tied-diag-gmm.h

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


#ifndef KALDI_TIED_MLE_AM_TIED_FULL_GMM_H_
#define KALDI_TIED_MLE_AM_TIED_FULL_GMM_H_ 1

#include <vector>

#include "gmm/mle-full-gmm.h"
#include "tied/am-tied-full-gmm.h"
#include "tied/mle-tied-gmm.h"

namespace kaldi {

class AccumAmTiedFullGmm {
 public:
  AccumAmTiedFullGmm() {}
  ~AccumAmTiedFullGmm();

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Initializes accumulators for each diagonal and tied GMM
  void Init(const AmTiedFullGmm &model, GmmFlagsType flags);

  void SetZero(GmmFlagsType flags);

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// This does not work with multiple feature transforms.
  BaseFloat Accumulate(const AmTiedFullGmm &model,
                       const TiedGmmPerFrameVars &per_frame_vars,
                       int32 pdf_index,
                       BaseFloat frame_posterior);

  /// Accumulate stats for single GMM in the model; returns log likelihood
  /// This will evaluate the associated codebook; use Accumulate for
  /// pre-computed codebook scores
  BaseFloat AccumulateForGmm(const AmTiedFullGmm &model,
                             const VectorBase<BaseFloat> &data,
                             int32 pdf_index,
                             BaseFloat frame_posterior);

  /// Accumulate for a certain codebook (pdf_index) and tied gmm
  /// (tied_pdf_index) given the data and posteriors
  void AccumulateFromPosteriors(const VectorBase<BaseFloat> &data,
                                int32 pdf_index,
                                int32 tied_pdf_index,
                                const VectorBase<BaseFloat> &posteriors);

  int32 NumAccs() {
    return gmm_accumulators_.size() + tied_gmm_accumulators_.size();
  }

  int32 NumAccs() const {
    return gmm_accumulators_.size() + tied_gmm_accumulators_.size();
  }

  int32 NumFullAccs() { return gmm_accumulators_.size(); }
  int32 NumFullAccs() const { return gmm_accumulators_.size(); }

  int32 NumTiedAccs() { return tied_gmm_accumulators_.size(); }
  int32 NumTiedAccs() const { return tied_gmm_accumulators_.size(); }

  AccumFullGmm& GetFullAcc(int32 index) const;
  AccumTiedGmm& GetTiedAcc(int32 pdf_index) const;

 private:
  /// MLE accumulators and update methods for the GMMs
  std::vector<AccumFullGmm*> gmm_accumulators_;
  std::vector<AccumTiedGmm*> tied_gmm_accumulators_;

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(AccumAmTiedFullGmm);
};

/// for computing the maximum-likelihood estimates of the parameters of
/// an acoustic model that uses (tied) diagonal Gaussian mixture models
void MleAmTiedFullGmmUpdate(const MleFullGmmOptions &config_diag,
                            const MleTiedGmmOptions &config_tied,
                            const AccumAmTiedFullGmm &amtieddiaggmm_acc,
                            GmmFlagsType flags,
                            AmTiedFullGmm *model,
                            BaseFloat *obj_change_out,
                            BaseFloat *count_out);

}  // End namespace kaldi

#endif  // KALDI_TIED_MLE_AM_TIED_FULL_GMM_H_

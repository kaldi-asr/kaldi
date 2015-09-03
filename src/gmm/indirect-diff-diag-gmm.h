// gmm/indirect-diff-diag-gmm.h

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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


#ifndef KALDI_GMM_INDIRECT_DIFF_DIAG_GMM_H_
#define KALDI_GMM_INDIRECT_DIFF_DIAG_GMM_H_ 1

#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/mle-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"
#include "gmm/model-common.h"

namespace kaldi {

// This gets the derivative of the (MMI or MPE) objective function w.r.t. the
// statistics for ML update, assuming we're doing an ML update-- as described in
// the original fMPE paper.  This is used in fMPE/fMMI, for the "indirect
// differential".  This derivative is represented as class AccumDiagGmm, as
// derivatives w.r.t. the x and x^2 stats directly (not w.r.t. the mean and
// variance).
//
// If the parameter "rescaling" is true, this function will assume that instead
// of the ML update, you will do a "rescaling" update as in the function
// DoRescalingUpdate().
//
// CAUTION: for fMPE (as opposed to fMMI), to get the right answer, you would have
// to pre-scale the num and den accs by the acoustic scale (e.g. 0.1).
void GetStatsDerivative(const AmDiagGmm &gmm,
                        const AccumAmDiagGmm &num_accs, // for MMI, would equal ml accs.
                        const AccumAmDiagGmm &den_accs,
                        const AccumAmDiagGmm &ml_accs,
                        BaseFloat min_variance,
                        BaseFloat min_gaussian_occupancy,
                        AccumAmDiagGmm *out_accs);


// This function "DoRescalingUpdate" updates the GMMs in a special way-- it
// first works out how the Gaussians differ from the old stats (in terms of an
// offset on the mean, a scale on the variance, and a factor on the weights),
// and it updates the model so that it will differ in the same way from the new
// stats.
//
// The idea here is that the original model may have been discriminatively
// trained, but we may have changed the features or the domain or something
// of that nature, and we want to update the model but preserve the discriminative
// training (viewed as an offset).
void DoRescalingUpdate(const AccumAmDiagGmm &old_ml_accs,
                       const AccumAmDiagGmm &new_ml_accs,
                       BaseFloat min_variance,
                       BaseFloat min_gaussian_occupancy,
                       AmDiagGmm *gmm);


} // end namespace kaldi


#endif  // KALDI_GMM_INDIRECT_DIFF_DIAG_GMM_H_

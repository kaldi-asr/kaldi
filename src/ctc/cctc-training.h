// ctc/cctc-training.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CTC_CCTC_TRAINING_H_
#define KALDI_CTC_CCTC_TRAINING_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "ctc/language-model.h"

namespace kaldi {
namespace ctc {

// CTC means Connectionist Temporal Classification, see the paper by Graves et
// al.  CCTC means context-dependent CTC, it's an extension of the original model,
// in which the next phone is dependent on the phone history (actually, a truncation
// thereof) in addition to the acoustic history.


struct CctcTrainingOptions {
  BaseFloat normalizing_weight;

  CctcTrainingOptions(): normalizing_weight(0.0001) { }

  void Register(OptionsItf *opts) {
    opts->Register("normalizing-weight", &normalizing_weight, "Weight on a "
                   "term in the objective function that's a squared "
                   "log of the denominator in the CCTC likelihood; it "
                   "exists to keep the network outputs in a reasonable "
                   "range so we can exp() them without overflow.");
  }
  
};

// This class is not responsible for the entire process of CCTC model training;
// it is only responsible for the forward-backward from the neural net output,
// and the derivative computation.
class CctcTraining {

  struct ForwardData {

    CuMatrix<BaseFloat> exp_nnet_output;
    CuMatrix<BaseFloat> normalizers;

    // The log-alpha value (forward score) for each state in the lattice.
    Vector<double> alpha;
    BaseFloat tot_like;
  };
  

  CctcTraining(const CctcTrainingOptions &opts,
               const CctcTransitionModel &trans_model):
      opts_(opts), trans_model_(trans_model) { }

  /**
     This function does the forward computation, up to
     computing the objective.
   */
  void Forward(const CtcSupervision &supervision,
               const CuMatrixBase<BaseFloat> &nnet_output,
               ForwardData *forward_data);

  void Backward(const CtcSupervision &supervision,
                const CuMatrixBase<BaseFloat> &nnet_output,
                const ForwardData &forward_data,
                CuMatrixBase<BaseFloat> *nnet_deriv);

  
 private:
  CctcTrainingOptions &opts_;
  const CctcTransitionModel &trans_model_;

  // CUDA copy of trans_model_.Weights().
  CuMatrix<BaseFloat> weights_;
};



}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_TRAINING_H_


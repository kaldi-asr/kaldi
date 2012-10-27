// nnet-cpu/decodable-am-nnet1.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_DECODABLE_AM_NNET1_H_
#define KALDI_NNET_CPU_DECODABLE_AM_NNET1_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet-cpu/am-nnet.h"
#include "nnet-cpu/nnet-compute.h"

namespace kaldi {

/// DecodableAmNnet1 is a decodable object that decodes
/// with a neural net acoustic model of type AmNnet.

class DecodableAmNnet: public DecodableInterface {
 public:
  DecodableAmNnet(const TransitionModel &trans_model,
                  const AmNnet &am_nnet,
                  const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> &spk_info,
                  bool pad_input = true, // if !pad_input, the NumIndices()
                  // will be < feats.NumRows().
                  BaseFloat prob_scale = 1.0):
      trans_model_(trans_model) {
    // Note: we could make this more memory-efficient by doing the
    // computation in smaller chunks than the whole utterance, and not
    // storing the whole thing.  We'll leave this for later.
    log_probs_.Resize(feats.NumRows(), trans_model.NumPdfs());
    // the following function is declared in nnet-compute.h
    NnetComputation(am_nnet.GetNnet(), feats, spk_info, pad_input, &log_probs_);
    Vector<BaseFloat> priors(am_nnet.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model.NumPdfs() &&
                 "Priors in neural network not set up.");
    priors.ApplyLog();
    log_probs_.AddVecToRows(-1.0, priors);
    // subtract log-prior (divide by prior)...
    log_probs_.Scale(prob_scale);
  }
  
  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id) {
    return log_probs_(frame,
                     trans_model_.TransitionIdToPdf(transition_id));
  }

  int32 NumFrames() { return log_probs_.NumRows(); }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }
  
  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }
 protected:
  const TransitionModel &trans_model_;
  Matrix<BaseFloat> log_probs_; // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnet);
};

}  // namespace kaldi

#endif  // KALDI_NNET_CPU_DECODABLE_AM_NNET1_H_

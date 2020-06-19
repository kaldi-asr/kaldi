// nnet2/decodable-am-nnet.h

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

#ifndef KALDI_NNET2_DECODABLE_AM_NNET_H_
#define KALDI_NNET2_DECODABLE_AM_NNET_H_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute.h"

namespace kaldi {
namespace nnet2 {

/// DecodableAmNnet is a decodable object that decodes
/// with a neural net acoustic model of type AmNnet.

class DecodableAmNnet: public DecodableInterface {
 public:
  DecodableAmNnet(const TransitionModel &trans_model,
                  const AmNnet &am_nnet,
                  const CuMatrixBase<BaseFloat> &feats,
                  bool pad_input = true, // if !pad_input, the NumIndices()
                                         // will be < feats.NumRows().
                  BaseFloat prob_scale = 1.0):
      trans_model_(trans_model) {
    // Note: we could make this more memory-efficient by doing the
    // computation in smaller chunks than the whole utterance, and not
    // storing the whole thing.  We'll leave this for later.
    int32 num_rows = feats.NumRows() -
        (pad_input ? 0 : am_nnet.GetNnet().LeftContext() +
                         am_nnet.GetNnet().RightContext());
    if (num_rows <= 0) {
      KALDI_WARN << "Input with " << feats.NumRows()  << " rows will produce "
                 << "empty output.";
      return;
    }
    CuMatrix<BaseFloat> log_probs(num_rows, trans_model.NumPdfs());
    // the following function is declared in nnet-compute.h
    NnetComputation(am_nnet.GetNnet(), feats, pad_input, &log_probs);
    log_probs.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
    log_probs.ApplyLog();
    CuVector<BaseFloat> priors(am_nnet.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model.NumPdfs() &&
                 "Priors in neural network not set up.");
    priors.ApplyLog();
    // subtract log-prior (divide by prior)
    log_probs.AddVecToRows(-1.0, priors);
    // apply probability scale.
    log_probs.Scale(prob_scale);
    // Transfer the log-probs to the CPU for faster access by the
    // decoding process.
    log_probs_.Swap(&log_probs);
  }

  // Note, frames are numbered from zero.  But transition_id is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id) {
    return log_probs_(frame,
                      trans_model_.TransitionIdToPdfFast(transition_id));
  }

  virtual int32 NumFramesReady() const { return log_probs_.NumRows(); }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 protected:
  const TransitionModel &trans_model_;
  Matrix<BaseFloat> log_probs_; // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnet);
};

/// This version of DecodableAmNnet is intended for a version of the decoder
/// that processes different utterances with multiple threads.  It needs to do
/// the computation in a different place than the initializer, since the
/// initializer gets called in the main thread of the program.

class DecodableAmNnetParallel: public DecodableInterface {
 public:
  DecodableAmNnetParallel(
      const TransitionModel &trans_model,
      const AmNnet &am_nnet,
      const CuMatrix<BaseFloat> *feats,
      bool pad_input = true,
      BaseFloat prob_scale = 1.0):
      trans_model_(trans_model), am_nnet_(am_nnet), feats_(feats),
      pad_input_(pad_input), prob_scale_(prob_scale) {
    KALDI_ASSERT(feats_ != NULL);
  }

  void Compute() {
    log_probs_.Resize(feats_->NumRows(), trans_model_.NumPdfs());
    // the following function is declared in nnet-compute.h
    NnetComputation(am_nnet_.GetNnet(), *feats_,
                    pad_input_, &log_probs_);
    log_probs_.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
    log_probs_.ApplyLog();
    CuVector<BaseFloat> priors(am_nnet_.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model_.NumPdfs() &&
                 "Priors in neural network not set up.");
    priors.ApplyLog();
    // subtract log-prior (divide by prior)
    log_probs_.AddVecToRows(-1.0, priors);
    // apply probability scale.
    log_probs_.Scale(prob_scale_);
    delete feats_;
    feats_ = NULL;
  }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id) {
    if (feats_) Compute(); // this function sets feats_ to NULL.
    return log_probs_(frame,
                      trans_model_.TransitionIdToPdfFast(transition_id));
  }

  int32 NumFramesReady() const {
    if (feats_) {
      if (pad_input_) return feats_->NumRows();
      else {
        int32 ans = feats_->NumRows() - am_nnet_.GetNnet().LeftContext() -
            am_nnet_.GetNnet().RightContext();
        if (ans < 0) ans = 0;
        return ans;
      }
    } else {
      return log_probs_.NumRows();
    }
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }
  ~DecodableAmNnetParallel() {
    delete feats_;
  }
 protected:
  const TransitionModel &trans_model_;
  const AmNnet &am_nnet_;
  CuMatrix<BaseFloat> log_probs_; // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.
  const CuMatrix<BaseFloat> *feats_;
  bool pad_input_;
  BaseFloat prob_scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetParallel);
};





} // namespace nnet2
} // namespace kaldi

#endif  // KALDI_NNET2_DECODABLE_AM_NNET_H_

// nnet2/decodable-am-nnet-multi.h

// Copyright 2015 Hainan Xu

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

#ifndef KALDI_NNET2_DECODABLE_AM_NNET_H_MULTI_
#define KALDI_NNET2_DECODABLE_AM_NNET_H_MULTI_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute.h"

namespace kaldi {
namespace nnet2 {

class DecodableAmNnetMulti: public DecodableInterface {
 public:
  DecodableAmNnetMulti(const TransitionModel &trans_model,
                  const vector<AmNnet> &am_nnets,
                  const unordered_map<int32, vector<int32> > &mapping,
                  const CuMatrixBase<BaseFloat> &feats,
                  const CuVectorBase<BaseFloat> &spk_info,
                  bool pad_input = true, // if !pad_input, the NumIndices()
                  // will be < feats.NumRows().
                  BaseFloat exp_weight = 0.1,
                  BaseFloat prob_scale = 1.0):
      trans_model_(trans_model), mapping_(mapping), exp_weight_(exp_weight) {
    log_probs_vec_.resize(am_nnets.size());

    vector<int32> num_pdf_vec = mapping_[-1];

    for (size_t i = 0; i < am_nnets.size(); i++) {
      // Note: we could make this more memory-efficient by doing the
      // computation in smaller chunks than the whole utterance, and not
      // storing the whole thing.  We'll leave this for later.
      CuMatrix<BaseFloat> log_probs(feats.NumRows(), num_pdf_vec[i]);
      // the following function is declared in nnet-compute.h
      NnetComputation(am_nnets[i].GetNnet(), feats, /*spk_info,*/
                             pad_input, &log_probs);
      log_probs.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
      log_probs.ApplyLog();
      CuVector<BaseFloat> priors(am_nnets[i].Priors());
      KALDI_ASSERT(priors.Dim() == num_pdf_vec[i] &&
		   "Priors in neural network not set up.");
      priors.ApplyLog();
      // subtract log-prior (divide by prior)
      log_probs.AddVecToRows(-1.0, priors);
      // apply probability scale.
      log_probs.Scale(prob_scale);
      // Transfer the log-probs to the CPU for faster access by the
      // decoding process.
      log_probs_vec_[i].Swap(&log_probs);
    }
  }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id) {
    int32 state = trans_model_.TransitionIdToPdf(transition_id);
    size_t num_trees = log_probs_vec_.size();
    vector<int32> states = mapping_[state];
    vector<BaseFloat> log_sums(num_trees);
    BaseFloat ans = 0.0;

    for (size_t i = 0; i < num_trees; i++) {
      log_sums[i] = log_probs_vec_[i](frame, states[i]);
    }

    double weighted_sum = 0;
    double weight = 0;

    for (size_t i = 0; i < num_trees; i++) {
      weighted_sum += log_sums[i] * exp(log_sums[i] * exp_weight_);
      weight += exp(log_sums[i] * exp_weight_);
    }
    ans = weighted_sum / weight;

    ans = 0;
    for (size_t i = 0; i < num_trees; i++) {
      ans += log_sums[i] / num_trees;
    }

    return ans;
  }

  int32 NumFrames() const { return log_probs_vec_[0].NumRows(); }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }
  
  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

 protected:
  const TransitionModel &trans_model_;
  vector<Matrix<BaseFloat> > log_probs_vec_; 
  // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.

  unordered_map<int32, vector<int32> > mapping_;
  BaseFloat exp_weight_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetMulti);
};

} // namespace nnet2
} // namespace kaldi

#endif  // KALDI_NNET2_DECODABLE_AM_NNET_H_

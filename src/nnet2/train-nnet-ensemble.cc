// nnet2/train-nnet-ensemble.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)
//           2014   Xiaohui Zhang

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

#include "nnet2/train-nnet-ensemble.h"
#include <numeric> // for std::accumulate

namespace kaldi {
namespace nnet2 {

static inline Int32Pair MakePair(int32 first, int32 second) {
  Int32Pair ans;
  ans.first = first;
  ans.second = second;
  return ans;
}

NnetEnsembleTrainer::NnetEnsembleTrainer(
    const NnetEnsembleTrainerConfig &config,
    std::vector<Nnet*> nnet_ensemble):
    config_(config), nnet_ensemble_(nnet_ensemble) {
  beta_ = config_.beta;
  num_phases_ = 0;
  bool first_time = true;
  BeginNewPhase(first_time);
}

void NnetEnsembleTrainer::TrainOnExample(const NnetExample &value) {
  buffer_.push_back(value);
  if (static_cast<int32>(buffer_.size()) == config_.minibatch_size)
    TrainOneMinibatch();
}

void NnetEnsembleTrainer::TrainOneMinibatch() {
  KALDI_ASSERT(!buffer_.empty());
  
  int32 num_states = nnet_ensemble_[0]->GetComponent(nnet_ensemble_[0]->NumComponents() - 1).OutputDim();
  // average of posteriors matrix, storing averaged outputs of net ensemble.
  CuMatrix<BaseFloat> post_avg(buffer_.size(), num_states);
  updater_ensemble_.reserve(nnet_ensemble_.size());
  std::vector<CuMatrix<BaseFloat> > post_mat;
  post_mat.resize(nnet_ensemble_.size());
  for (int32 i = 0; i < nnet_ensemble_.size(); i++) {
    updater_ensemble_.push_back(new NnetUpdater(*(nnet_ensemble_[i]), nnet_ensemble_[i]));
    updater_ensemble_[i]->FormatInput(buffer_);
    updater_ensemble_[i]->Propagate();
    // posterior matrix, storing output of one net.
    updater_ensemble_[i]->GetOutput(&post_mat[i]);
    CuVector<BaseFloat> row_sum(post_mat[i].NumRows());
    post_avg.AddMat(1.0, post_mat[i]);
  }

  // calculate the interpolated posterios as new supervision labels, and also 
  // collect the indices of the original supervision labels for later use (calc. objf.).
  std::vector<MatrixElement<BaseFloat> > sv_labels;
  std::vector<Int32Pair > sv_labels_ind;
  sv_labels.reserve(buffer_.size()); // We must have at least this many labels.
  sv_labels_ind.reserve(buffer_.size()); // We must have at least this many labels.
  for (int32 m = 0; m < buffer_.size(); m++) {
    KALDI_ASSERT(buffer_[m].labels.size() == 1 &&
                 "Currently this code only supports single-frame egs.");
    const std::vector<std::pair<int32,BaseFloat> > &labels = buffer_[m].labels[0];
    for (size_t i = 0; i < labels.size(); i++) {
      MatrixElement<BaseFloat> 
          tmp = {m, labels[i].first, labels[i].second};
      sv_labels.push_back(tmp);
      sv_labels_ind.push_back(MakePair(m, labels[i].first));
    }
  }
  post_avg.Scale(1.0 / nnet_ensemble_.size());
  post_avg.Scale(beta_);
  post_avg.AddElements(1.0, sv_labels);

  // calculate the deriv, do backprop, and calculate the objf.
  for (int32 i = 0; i < nnet_ensemble_.size(); i++) {  
    CuMatrix<BaseFloat> tmp_deriv(post_mat[i]);
    post_mat[i].ApplyLog();
    std::vector<BaseFloat> log_post_correct;
    log_post_correct.resize(sv_labels_ind.size());
    post_mat[i].Lookup(sv_labels_ind, &(log_post_correct[0]));
    BaseFloat log_prob_this_net = std::accumulate(log_post_correct.begin(),
                                                  log_post_correct.end(),
                                                  static_cast<BaseFloat>(0));
    avg_logprob_this_phase_ += log_prob_this_net;
    tmp_deriv.InvertElements();
    tmp_deriv.MulElements(post_avg);
    updater_ensemble_[i]->Backprop(&tmp_deriv);
  }
  count_this_phase_ += buffer_.size();
  buffer_.clear();
  minibatches_seen_this_phase_++;
  if (minibatches_seen_this_phase_ == config_.minibatches_per_phase) {
    avg_logprob_this_phase_ /= static_cast<BaseFloat>(nnet_ensemble_.size());
    bool first_time = false;
    BeginNewPhase(first_time);
  }
}

void NnetEnsembleTrainer::BeginNewPhase(bool first_time) {
  if (!first_time)
    KALDI_LOG << "Averaged cross-entropy between the supervision labels and the output is "
              << (avg_logprob_this_phase_/count_this_phase_) << " over "
              << count_this_phase_ << " frames, during this phase";
  avg_logprob_this_phase_ = 0.0;
  count_this_phase_ = 0.0;
  minibatches_seen_this_phase_ = 0;
  num_phases_++;
}


NnetEnsembleTrainer::~NnetEnsembleTrainer() {
  if (!buffer_.empty()) {
    KALDI_LOG << "Doing partial minibatch of size "
              << buffer_.size();
    TrainOneMinibatch();
    if (minibatches_seen_this_phase_ != 0) {
      bool first_time = false;
      BeginNewPhase(first_time);
    }
  }
}


} // namespace nnet2
} // namespace kaldi

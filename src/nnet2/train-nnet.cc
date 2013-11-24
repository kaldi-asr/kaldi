// nnet2/train-nnet.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/train-nnet.h"

namespace kaldi {
namespace nnet2 {


NnetSimpleTrainer::NnetSimpleTrainer(
    const NnetSimpleTrainerConfig &config,
    Nnet *nnet):
    config_(config), nnet_(nnet) {
  num_phases_ = 0;
  bool first_time = true;
  BeginNewPhase(first_time);
}

void NnetSimpleTrainer::TrainOnExample(const NnetExample &value) {
  buffer_.push_back(value);
  if (static_cast<int32>(buffer_.size()) == config_.minibatch_size)
    TrainOneMinibatch();
}

void NnetSimpleTrainer::TrainOneMinibatch() {
  KALDI_ASSERT(!buffer_.empty());
  // The following function is declared in nnet-update.h.
  logprob_this_phase_ += DoBackprop(*nnet_,
                                    buffer_,
                                    nnet_);
  count_this_phase_ += buffer_.size();
  buffer_.clear();
  minibatches_seen_this_phase_++;
  if (minibatches_seen_this_phase_ == config_.minibatches_per_phase) {
    bool first_time = false;
    BeginNewPhase(first_time);
  }
}

void NnetSimpleTrainer::BeginNewPhase(bool first_time) {
  if (!first_time)
    KALDI_LOG << "Training objective function (this phase) is "
              << (logprob_this_phase_/count_this_phase_) << " over "
              << count_this_phase_ << " frames.";
  logprob_this_phase_ = 0.0;
  count_this_phase_ = 0.0;
  minibatches_seen_this_phase_ = 0;
  num_phases_++;
}


NnetSimpleTrainer::~NnetSimpleTrainer() {
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

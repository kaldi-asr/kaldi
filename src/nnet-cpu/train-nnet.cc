// nnet-cpu/train-nnet.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/train-nnet.h"

namespace kaldi {
namespace nnet2 {

NnetAdaptiveTrainer::NnetAdaptiveTrainer(
    const NnetAdaptiveTrainerConfig &config,
    const std::vector<NnetTrainingExample> &validation_set,
    Nnet *nnet):
    config_(config), validation_set_(validation_set), nnet_(nnet) {
  num_phases_ = 0;
  bool first_time = true;
  BeginNewPhase(first_time);
}

void NnetAdaptiveTrainer::BeginNewPhase(bool first_time) {
  int32 num_components = nnet_->NumComponents();
  { // First take care of training objf.
    if (!first_time)
      KALDI_LOG << "Training objective function (this phase) is "
                << (logprob_this_phase_/count_this_phase_) << " over "
                << count_this_phase_ << " frames.";
    logprob_this_phase_ = 0.0;
    count_this_phase_ = 0.0;
  }
  if (!first_time) { // normal case-- end old phase + begin new one.
    // Declare dot products (each element is a layer's dot product)
    Vector<BaseFloat> old_model_old_gradient(num_components),
        new_model_old_gradient(num_components),
        old_model_new_gradient(num_components),
        new_model_new_gradient(num_components);
    nnet_snapshot_.ComponentDotProducts(validation_gradient_,
                                        &old_model_old_gradient);
    nnet_->ComponentDotProducts(validation_gradient_,
                                &new_model_old_gradient);
    BaseFloat old_objf = validation_objf_;
    int32 batch_size = 1024;
    BaseFloat new_objf = ComputeNnetGradient(*nnet_,
                                             validation_set_,
                                             batch_size,
                                             &validation_gradient_);
    validation_objf_ = new_objf;
    nnet_snapshot_.ComponentDotProducts(validation_gradient_,
                                        &old_model_new_gradient);
    nnet_->ComponentDotProducts(validation_gradient_,
                                &new_model_new_gradient);
    // old_gradient_delta is the old gradient * delta-parameters.
    Vector<BaseFloat> old_gradient_delta(new_model_old_gradient);
    old_gradient_delta.AddVec(-1.0, old_model_old_gradient);
    // new_gradient_delta is the new gradient * delta-parameters.
    Vector<BaseFloat> new_gradient_delta(new_model_new_gradient);
    new_gradient_delta.AddVec(-1.0, old_model_new_gradient);

    // If we use a quadratic model for the objective function, we
    // can show that the progress (change in objective function) this
    // iter is the average of old_gradient_delta and new_gradient_delta.
    Vector<BaseFloat> progress_this_iter(num_components);
    progress_this_iter.AddVec(0.5, old_gradient_delta);
    progress_this_iter.AddVec(0.5, new_gradient_delta);

    // For ease of viewing, scale down the progress stats by
    // the total weight of the validation-set frames.
    progress_this_iter.Scale(1.0 / validation_set_.size());
    
    if (new_objf < old_objf && !config_.always_accept) {
      KALDI_LOG << "Objf degraded from " << old_objf << " to " << new_objf
                << ", reverting parameters to old values.";
      validation_objf_ = old_objf; // revert.
      *nnet_ = nnet_snapshot_;
    } else {
      // Accumulate a record of progress (broken down by layer).
      progress_stats_.AddVec(1.0, progress_this_iter);
    }
    KALDI_VLOG(2) << "Progress this iteration (based on gradients) is "
                  << progress_this_iter << ", total "
                  << progress_this_iter.Sum();
    KALDI_VLOG(2) << "Actual progress this iteration is " << old_objf
                  << " to " << new_objf << ": change is " << (new_objf-old_objf);
    KALDI_VLOG(2) << "Overall progress so far (based on gradients) is "
                  << progress_stats_ << ", total "
                  << progress_stats_.Sum();
    KALDI_VLOG(2) << "Actual progress so far " << initial_validation_objf_
                  << " to " << new_objf << ": change is "
                  << (new_objf-initial_validation_objf_);

    // Update the learning rates.  This is done after possibly reverting
    // the model, so the changed learning rates don't get thrown away.
    nnet_->AdjustLearningRates(old_model_old_gradient,
                               new_model_old_gradient,
                               old_model_new_gradient,
                               new_model_new_gradient,
                               config_.measure_gradient_at,
                               config_.learning_rate_ratio,
                               config_.max_learning_rate);
    KALDI_VLOG(3) << "Current model info: " << nnet_->Info();
  } else { // first time.
    validation_gradient_ = *nnet_;
    int32 batch_size = 1024;
    validation_objf_ = ComputeNnetGradient(*nnet_,
                                           validation_set_,
                                           batch_size,
                                           &validation_gradient_);
    initial_validation_objf_ = validation_objf_;
    progress_stats_.Resize(num_components);
  }
  
  nnet_snapshot_ = *nnet_;
  minibatches_seen_this_phase_ = 0;
  num_phases_++;  
}

void NnetAdaptiveTrainer::TrainOneMinibatch() {
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

NnetAdaptiveTrainer::~NnetAdaptiveTrainer() {
  if (!buffer_.empty()) {
    KALDI_LOG << "Doing partial minibatch of size "
              << buffer_.size();
    TrainOneMinibatch();
  }
  KALDI_LOG << "Trained for " << num_phases_ << " phases of training.";
  KALDI_LOG << "The following numbers include only whole phases of training\n";
  KALDI_LOG << "Predicted progress (based on gradients) broken down by "
            << "component is " << progress_stats_;
  KALDI_LOG << "Total of the above over all layers is " << progress_stats_.Sum();
  KALDI_LOG << "Actual progress " << initial_validation_objf_
            << " to " << validation_objf_ << ": change is "
            << (validation_objf_-initial_validation_objf_);  
}

void NnetAdaptiveTrainer::TrainOnExample(const NnetTrainingExample &value) {
  buffer_.push_back(value);
  if (static_cast<int32>(buffer_.size()) == config_.minibatch_size)
    TrainOneMinibatch();
}

NnetSimpleTrainer::NnetSimpleTrainer(
    const NnetSimpleTrainerConfig &config,
    Nnet *nnet):
    config_(config), nnet_(nnet) {
  num_phases_ = 0;
  bool first_time = true;
  BeginNewPhase(first_time);
}

void NnetSimpleTrainer::TrainOnExample(const NnetTrainingExample &value) {
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
    bool first_time = false;
    BeginNewPhase(first_time);
  }
}


} // namespace nnet2
} // namespace kaldi

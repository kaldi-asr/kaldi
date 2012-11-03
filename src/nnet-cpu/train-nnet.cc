// nnet/train-nnet.cc

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

void NnetValidationSet::AddUtterance(
    const MatrixBase<BaseFloat> &features,
    const VectorBase<BaseFloat> &spk_info, // may be empty
    const std::vector<int32> &pdf_ids,
    BaseFloat utterance_weight) {
  KALDI_ASSERT(pdf_ids.size() == static_cast<size_t>(features.NumRows()));
  KALDI_ASSERT(utterance_weight > 0.0);
  utterances_.push_back(new Utterance(features, spk_info,
                                      pdf_ids, utterance_weight));
  if (utterances_.size() != 0) { // Check they have consistent dimensions.
    KALDI_ASSERT(features.NumCols() == utterances_[0]->features.NumCols());
    KALDI_ASSERT(spk_info.Dim() == utterances_[0]->spk_info.Dim());
  }
}

NnetValidationSet::~NnetValidationSet() {
  for (size_t i = 0; i < utterances_.size(); i++)
    delete utterances_[i];
}


BaseFloat NnetValidationSet::ComputeGradient(const Nnet &nnet,
                                             Nnet *nnet_gradient) const {
  KALDI_ASSERT(!utterances_.empty());
  bool treat_as_gradient = true, pad_input = true;
  BaseFloat tot_objf = 0.0, tot_weight = 0.0;
  nnet_gradient->SetZero(treat_as_gradient);  
  for (size_t i = 0; i < utterances_.size(); i++) {
    const Utterance &utt = *(utterances_[i]);
    tot_objf += NnetGradientComputation(nnet,
                                        utt.features, utt.spk_info,
                                        pad_input, utt.weight,
                                        utt.pdf_ids, nnet_gradient);
    tot_weight += utt.weight * utt.features.NumRows();
  }
  KALDI_VLOG(2) << "Validation set objective function " << (tot_objf / tot_weight)
                << " over " << tot_weight << " frames.";
  return tot_objf / tot_weight;
}

BaseFloat NnetValidationSet::TotalWeight() const {
  double ans = 0.0;
  for (size_t i = 0; i < utterances_.size(); i++)
    ans += utterances_[i]->features.NumRows() *
        utterances_[i]->weight;
  return ans;
}

NnetAdaptiveTrainer::NnetAdaptiveTrainer(
    const NnetAdaptiveTrainerConfig &config,
    const std::vector<NnetTrainingExample> &validation_set,
    Nnet *nnet):
    config_(config), validation_set_(validation_set), nnet_(nnet) {
  num_phases_ = 0;
  validation_tot_weight_ = TotalNnetTrainingWeight(validation_set);
  bool first_time = true;
  BeginNewPhase(first_time);
}

BaseFloat NnetAdaptiveTrainer::ComputeValidationSetGradient(
    Nnet *gradient) const {
  bool treat_as_gradient = true;
  gradient->SetZero(treat_as_gradient);
  int32 batch_size = 2000;
  std::vector<NnetTrainingExample> batch;
  batch.reserve(batch_size);
  BaseFloat tot_objf = 0.0;
  for (int32 start_pos = 0;
       start_pos < static_cast<int32>(validation_set_.size());
       start_pos += batch_size) {
    batch.clear();
    for (int32 i = start_pos;
         i < std::min(start_pos + batch_size,
                      static_cast<int32>(validation_set_.size()));
         i++) {
      batch.push_back(validation_set_[i]);
    }
    tot_objf += DoBackprop(*nnet_,
                           batch,
                           gradient);
  }
  return tot_objf / validation_tot_weight_;
}

void NnetAdaptiveTrainer::BeginNewPhase(bool first_time) {
  int32 num_components = nnet_->NumComponents();  
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
    BaseFloat new_objf = ComputeValidationSetGradient(&validation_gradient_);
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
    progress_this_iter.Scale(1.0 / validation_tot_weight_);
    
    if (new_objf < old_objf) {
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
    nnet_->AdjustLearningRatesAndL2Penalties(old_model_old_gradient,
                                             new_model_old_gradient,
                                             old_model_new_gradient,
                                             new_model_new_gradient,
                                             config_.measure_gradient_at,
                                             config_.learning_rate_ratio,
                                             config_.max_learning_rate,
                                             config_.min_l2_penalty,
                                             config_.max_l2_penalty);
    KALDI_VLOG(3) << "Current model info: " << nnet_->Info();
  } else { // first time.
    validation_gradient_ = *nnet_;
    validation_objf_ = ComputeValidationSetGradient(&validation_gradient_);
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
  DoBackprop(*nnet_,
             buffer_,
             nnet_);
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
  
} // namespace

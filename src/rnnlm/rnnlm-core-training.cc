// rnnlm/rnnlm-core-training.cc

// Copyright 2017  Daniel Povey

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

#include <numeric>
#include "rnnlm/rnnlm-core-training.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace rnnlm {


ObjectiveTracker::ObjectiveTracker(int32 reporting_interval):
    reporting_interval_(reporting_interval),
    num_egs_this_interval_(0),
    tot_weight_this_interval_(0.0),
    num_objf_this_interval_(0.0),
    den_objf_this_interval_(0.0),
    exact_den_objf_this_interval_(0.0),
    num_egs_(0),
    tot_weight_(0.0),
    num_objf_(0.0),
    den_objf_(0.0),
    exact_den_objf_(0.0) {
  KALDI_ASSERT(reporting_interval > 0);
}


void ObjectiveTracker::AddStats(
    BaseFloat weight, BaseFloat num_objf,
    BaseFloat den_objf,
    BaseFloat exact_den_objf) {
  num_egs_this_interval_++;
  tot_weight_this_interval_ += weight;
  num_objf_this_interval_ += num_objf;
  den_objf_this_interval_ += den_objf;
  exact_den_objf_this_interval_ += exact_den_objf;
  if (num_egs_this_interval_ >= reporting_interval_) {
    PrintStatsThisInterval();
    CommitIntervalStats();
  }
}

ObjectiveTracker::~ObjectiveTracker() {
  if (num_egs_this_interval_ != 0) {
    PrintStatsThisInterval();
    CommitIntervalStats();
  }
  PrintStatsOverall();
}

void ObjectiveTracker::CommitIntervalStats() {
  num_egs_ += num_egs_this_interval_;
  num_egs_this_interval_ = 0;
  tot_weight_ += tot_weight_this_interval_;
  tot_weight_this_interval_ = 0.0;
  num_objf_ += num_objf_this_interval_;
  num_objf_this_interval_ = 0.0;
  den_objf_ += den_objf_this_interval_;
  den_objf_this_interval_ = 0.0;
  exact_den_objf_ += exact_den_objf_this_interval_;
  exact_den_objf_this_interval_ = 0.0;
}

void ObjectiveTracker::PrintStatsThisInterval() const {
  int32 interval_start = num_egs_,
      interval_end = num_egs_ + num_egs_this_interval_ - 1;
  double weight = tot_weight_this_interval_,
      num_objf = num_objf_this_interval_ / weight,
      den_objf = den_objf_this_interval_ / weight,
      tot_objf = num_objf + den_objf,
      exact_den_objf = exact_den_objf_this_interval_ / weight,
      exact_tot_objf = num_objf + exact_den_objf;
  std::ostringstream os;
  os.precision(4);
  os << "Objf for minibatches " << interval_start << " to "
     << interval_end << " is (" << num_objf << " + "
     << den_objf << ") = " << tot_objf << " over "
     << weight << " words (weighted)";

  os << "; exact = (" << num_objf << " + " << exact_den_objf
     << ") = " << exact_tot_objf ;

  KALDI_LOG << os.str();
}

void ObjectiveTracker::PrintStatsOverall() const {
  double weight = tot_weight_,
      num_objf = num_objf_ / weight,
      den_objf = den_objf_ / weight,
      tot_objf = num_objf + den_objf,
      exact_den_objf = exact_den_objf_ / weight,
      exact_tot_objf = num_objf + exact_den_objf;
  std::ostringstream os;
  os.precision(4);
  os << "Overall objf is (" << num_objf << " + " << den_objf
     << ") = " << tot_objf << " over " << weight << " words (weighted) in "
     << num_egs_ << " minibatches";
  os << "; exact = (" << num_objf << " + " << exact_den_objf
     << ") = " << exact_tot_objf ;

  KALDI_LOG << os.str();
}


RnnlmCoreTrainer::RnnlmCoreTrainer(const RnnlmCoreTrainerOptions &config,
                                   const RnnlmObjectiveOptions &objective_config,
                                   nnet3::Nnet *nnet):
    config_(config),
    objective_config_(objective_config),
    nnet_(nnet),
    compiler_(*nnet),  // for now we don't make available other options
    num_minibatches_processed_(0),
    objf_info_(10) {
  ZeroComponentStats(nnet);
  KALDI_ASSERT(config.momentum >= 0.0 &&
               config.max_param_change >= 0.0);
  delta_nnet_ = nnet_->Copy();
  ScaleNnet(0.0, delta_nnet_);
  const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
  num_max_change_per_component_applied_.resize(num_updatable, 0);
  num_max_change_global_applied_ = 0;
}


void RnnlmCoreTrainer::Train(
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    CuMatrixBase<BaseFloat> *word_embedding_deriv) {
  using namespace nnet3;

  bool need_model_derivative = true;
  bool need_input_derivative = (word_embedding_deriv != NULL);
  bool store_component_stats = true;

  ComputationRequest request;
  GetRnnlmComputationRequest(minibatch, need_model_derivative,
                             need_input_derivative,
                             store_component_stats,
                             &request);

  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);

  NnetComputeOptions compute_opts;

  NnetComputer computer(compute_opts, *computation,
                        *nnet_, delta_nnet_);

  ProvideInput(minibatch, derived, word_embedding, &computer);
  computer.Run();  // This is the forward pass.

  bool is_backstitch_step1 = true;
  ProcessOutput(is_backstitch_step1, minibatch, derived, word_embedding,
                &computer, word_embedding_deriv);

  computer.Run();  // This is the backward pass.

  if (word_embedding_deriv != NULL) {
    CuMatrix<BaseFloat> input_deriv;
    computer.GetOutputDestructive("input", &input_deriv);
    word_embedding_deriv->AddSmatMat(1.0, derived.input_words_smat, kNoTrans,
                                     input_deriv, 1.0);
  }
  // If relevant, add in the part of the gradient that comes from L2
  // regularization.
  ApplyL2Regularization(*nnet_,
                        minibatch.num_chunks * config_.l2_regularize_factor,
                        delta_nnet_);

  bool success = UpdateNnetWithMaxChange(*delta_nnet_, config_.max_param_change,
      1.0, 1.0 - config_.momentum, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);
  if (success) ScaleNnet(config_.momentum, delta_nnet_);
  else ScaleNnet(0.0, delta_nnet_);

  num_minibatches_processed_++;
}

void RnnlmCoreTrainer::TrainBackstitch(
    bool is_backstitch_step1,
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    CuMatrixBase<BaseFloat> *word_embedding_deriv) {
  using namespace nnet3;

  // backstitch training is incompatible with momentum > 0
  KALDI_ASSERT(config_.momentum == 0.0);

  bool need_model_derivative = true;
  bool need_input_derivative = (word_embedding_deriv != NULL);
  bool store_component_stats = true;

  ComputationRequest request;
  GetRnnlmComputationRequest(minibatch, need_model_derivative,
                             need_input_derivative,
                             store_component_stats,
                             &request);

  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);

  NnetComputeOptions compute_opts;

  if (is_backstitch_step1) {
    FreezeNaturalGradient(true, delta_nnet_);
  }
  ResetGenerators(nnet_);
  NnetComputer computer(compute_opts, *computation,
                        *nnet_, delta_nnet_);

  ProvideInput(minibatch, derived, word_embedding, &computer);
  computer.Run();  // This is the forward pass.

  ProcessOutput(is_backstitch_step1, minibatch, derived, word_embedding,
                &computer, word_embedding_deriv);

  computer.Run();  // This is the backward pass.

  if (word_embedding_deriv != NULL) {
    CuMatrix<BaseFloat> input_deriv;
    computer.GetOutputDestructive("input", &input_deriv);
    word_embedding_deriv->AddSmatMat(1.0, derived.input_words_smat, kNoTrans,
                                     input_deriv, 1.0);
  }

  BaseFloat max_change_scale, scale_adding;
  if (is_backstitch_step1) {
    // max-change is scaled by backstitch_training_scale;
    // delta_nnet is scaled by -backstitch_training_scale when added to nnet;
    max_change_scale = config_.backstitch_training_scale;
    scale_adding = -config_.backstitch_training_scale;
  } else {
    // max-change is scaled by 1 + backstitch_training_scale;
    // delta_nnet is scaled by 1 + backstitch_training_scale when added to nnet;
    max_change_scale = 1.0 + config_.backstitch_training_scale;
    scale_adding = 1.0 + config_.backstitch_training_scale;
    num_minibatches_processed_++;
    // If relevant, add in the part of the gradient that comes from L2
    // regularization.
    ApplyL2Regularization(*nnet_,
                          1.0 / scale_adding *
                          minibatch.num_chunks * config_.l2_regularize_factor,
                          delta_nnet_);
  }

  UpdateNnetWithMaxChange(*delta_nnet_, config_.max_param_change,
      max_change_scale, scale_adding, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  ScaleNnet(0.0, delta_nnet_);

  if (is_backstitch_step1) {
    FreezeNaturalGradient(false, delta_nnet_);
  }
}

void RnnlmCoreTrainer::ProvideInput(
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    nnet3::NnetComputer *computer) {
  int32 embedding_dim = word_embedding.NumCols();
  CuMatrix<BaseFloat> input_embeddings(derived.cu_input_words.Dim(),
                                       embedding_dim,
                                       kUndefined);
  input_embeddings.CopyRows(word_embedding,
                            derived.cu_input_words);
  computer->AcceptInput("input", &input_embeddings);
}


void RnnlmCoreTrainer::PrintMaxChangeStats() const {
  using namespace nnet3;
  KALDI_ASSERT(delta_nnet_ != NULL);
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->NumComponents(); c++) {
    Component *comp = delta_nnet_->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      if (num_max_change_per_component_applied_[i] > 0)
        KALDI_LOG << "For " << delta_nnet_->GetComponentName(c)
                  << ", per-component max-change was enforced "
                  << ((100.0 * num_max_change_per_component_applied_[i]) /
                      num_minibatches_processed_)
                  << "% of the time.";
      i++;
    }
  }
  if (num_max_change_global_applied_ > 0)
    KALDI_LOG << "The global max-change was enforced "
              << (100.0 * num_max_change_global_applied_) /
                 (num_minibatches_processed_ *
                 (config_.backstitch_training_scale == 0.0 ? 1.0 :
                 1.0 + 1.0 / config_.backstitch_training_interval))
              << "% of the time.";
}

void RnnlmCoreTrainer::ProcessOutput(
    bool is_backstitch_step1,
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    nnet3::NnetComputer *computer,
    CuMatrixBase<BaseFloat> *word_embedding_deriv) {
  // 'output' is the output of the neural network.  The row-index
  // combines the time (with higher stride) and the member 'n'
  // of the minibatch (with stride 1); the number of columns is
  // the word-embedding dimension.
  CuMatrix<BaseFloat> output;
  CuMatrix<BaseFloat> output_deriv;
  computer->GetOutputDestructive("output", &output);
  output_deriv.Resize(output.NumRows(), output.NumCols());

  BaseFloat weight, objf_num, objf_den, objf_den_exact;
  ProcessRnnlmOutput(objective_config_,
                     minibatch, derived, word_embedding,
                     output, word_embedding_deriv, &output_deriv,
                     &weight, &objf_num, &objf_den,
                     &objf_den_exact);

  if (is_backstitch_step1)
    objf_info_.AddStats(weight, objf_num, objf_den, objf_den_exact);
  computer->AcceptInput("output", &output_deriv);
}

void RnnlmCoreTrainer::ConsolidateMemory() {
  kaldi::nnet3::ConsolidateMemory(nnet_);
  kaldi::nnet3::ConsolidateMemory(delta_nnet_);
}

RnnlmCoreTrainer::~RnnlmCoreTrainer() {
  PrintMaxChangeStats();
  // Note: the objective-function stats are printed out in the destructor of the
  // ObjectiveTracker object.
}





}  // namespace rnnlm
}  // namespace kaldi

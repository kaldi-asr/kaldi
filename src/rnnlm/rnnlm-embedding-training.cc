// rnnlm/rnnlm-embedding-training.cc

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
#include "rnnlm/rnnlm-embedding-training.h"
#include "nnet3/natural-gradient-online.h"

namespace kaldi {
namespace rnnlm {

void RnnlmEmbeddingTrainerOptions::Check() const {
  KALDI_ASSERT(print_interval > 0 &&
               momentum >= 0.0 && momentum < 1.0 &&
               learning_rate > 0.0 &&
               natural_gradient_alpha > 0.0 &&
               natural_gradient_rank > 0 &&
               natural_gradient_update_period >= 1 &&
               natural_gradient_num_minibatches_history > 1.0);
}


RnnlmEmbeddingTrainer::RnnlmEmbeddingTrainer(
    const RnnlmEmbeddingTrainerOptions &config,
    CuMatrix<BaseFloat> *embedding_mat):
    config_(config),
    embedding_mat_(embedding_mat),
    num_minibatches_(0),
    max_change_count_(0) {
  KALDI_ASSERT(embedding_mat->NumRows() > 0);
  initial_embedding_mat_.Resize(embedding_mat->NumRows(),
                                embedding_mat->NumCols(),
                                kUndefined);
  embedding_mat->CopyToMat(&initial_embedding_mat_);
  if (config_.momentum > 0.0)
    embedding_mat_momentum_.Resize(embedding_mat->NumRows(),
                                   embedding_mat->NumCols());
  SetNaturalGradientOptions();
}

void RnnlmEmbeddingTrainer::SetNaturalGradientOptions() {
  config_.Check();
  if (!config_.use_natural_gradient)
    return;
  preconditioner_.SetAlpha(config_.natural_gradient_alpha);
  preconditioner_.SetRank(config_.natural_gradient_rank);
  preconditioner_.SetUpdatePeriod(config_.natural_gradient_update_period);
  preconditioner_.SetNumMinibatchesHistory(
      config_.natural_gradient_num_minibatches_history);
}


void RnnlmEmbeddingTrainer::Train(
    CuMatrixBase<BaseFloat> *embedding_deriv) {

  // If relevant, do the following:
  // "embedding_deriv += - 2 * l2_regularize * embedding_mat_"
  // This is an approximate to the regular l2 regularization (add l2 regularization
  // to the objective function).
  if (config_.l2_regularize > 0.0) {
    BaseFloat l2_term = -2 * config_.l2_regularize;
    if (l2_term != 0.0) {
      embedding_deriv->AddMat(l2_term, *embedding_mat_);
    }
  }

  BaseFloat scale = 1.0;
  if (config_.use_natural_gradient) {
    preconditioner_.PreconditionDirections(embedding_deriv, &scale);
  }
  scale *= config_.learning_rate;
  num_minibatches_++;
  if (config_.max_param_change > 0.0) {
    BaseFloat delta = scale * embedding_deriv->FrobeniusNorm();
    // 'delta' is the 2-norm of the change in parameters.
    if (delta > config_.max_param_change) {
      BaseFloat max_change_scale = config_.max_param_change / delta;
      KALDI_LOG << "Applying max-change with scale " << max_change_scale
                << " since param-change=" << delta << " > "
                << " --embedding.max-param-change="
                << config_.max_param_change;
      max_change_count_++;
      scale *= max_change_scale;
    }
  }

  if (config_.momentum > 0.0) {
    // Multiply the factor (1 - momentum) into the learning rate, which cancels
    // out the scale (1 / (1 - momentum)) which otherwise appears in the
    // effective learning rate due to the geometric sum of (1 + momentum +
    // momentum^2, ...).
    scale *= (1.0 - config_.momentum);
    embedding_mat_momentum_.AddMat(scale, *embedding_deriv);
    embedding_mat_->AddMat(1.0, embedding_mat_momentum_);
    embedding_mat_momentum_.Scale(config_.momentum);
  } else {
    embedding_mat_->AddMat(scale, *embedding_deriv);
  }
}

void RnnlmEmbeddingTrainer::TrainBackstitch(
    bool is_backstitch_step1,
    CuMatrixBase<BaseFloat> *embedding_deriv) {

  // backstitch training is incompatible with momentum > 0
  KALDI_ASSERT(config_.momentum == 0.0);

  // If relevant, do the following:
  // "embedding_deriv += - 2 * l2_regularize * embedding_mat_"
  // This is an approximate to the regular l2 regularization (add l2 regularization
  // to the objective function).
  if (config_.l2_regularize > 0.0 && !is_backstitch_step1) {
    BaseFloat l2_term = -2 * config_.l2_regularize;
    if (l2_term != 0.0) {
      embedding_deriv->AddMat(1.0 / (1.0 + config_.backstitch_training_scale) *
          l2_term, *embedding_mat_);
    }
  }

  BaseFloat scale = 1.0;
  if (config_.use_natural_gradient) {
    if (is_backstitch_step1) preconditioner_.Freeze(true);
    preconditioner_.PreconditionDirections(embedding_deriv, &scale);
  }
  scale *= config_.learning_rate;
  num_minibatches_++;
  if (config_.max_param_change > 0.0) {
    BaseFloat delta = scale * embedding_deriv->FrobeniusNorm();
    // 'delta' is the 2-norm of the change in parameters.
    if (delta > config_.max_param_change) {
      BaseFloat max_change_scale = config_.max_param_change / delta;
      KALDI_LOG << "Applying max-change with scale " << max_change_scale
                << " since param-change=" << delta << " > "
                << " --embedding.max-param-change="
                << config_.max_param_change;
      max_change_count_++;
      scale *= max_change_scale;
    }
  }
  if (is_backstitch_step1) {
    scale *= -config_.backstitch_training_scale;
    if (config_.use_natural_gradient) preconditioner_.Freeze(false);
  } else {
    scale *= 1.0 + config_.backstitch_training_scale;
    num_minibatches_++;
  }
  embedding_mat_->AddMat(scale, *embedding_deriv);
}

void RnnlmEmbeddingTrainer::Train(
    const CuArrayBase<int32> &active_words,
    CuMatrixBase<BaseFloat> *embedding_deriv) {

  KALDI_ASSERT(active_words.Dim() == embedding_deriv->NumRows());

  // If relevant, do the following:
  // "embedding_deriv += - 2 * l2_regularize * embedding_mat_"
  // This is an approximate to the regular l2 regularization (add l2 regularization
  // to the objective function).
  if (config_.l2_regularize > 0.0) {
    BaseFloat l2_term = -2 * config_.l2_regularize;
    if (l2_term != 0.0) {
      embedding_deriv->AddRows(l2_term, *embedding_mat_, active_words);
    }
  }
  BaseFloat scale = 1.0;
  if (config_.use_natural_gradient) {
    preconditioner_.PreconditionDirections(embedding_deriv, &scale);
  }
  scale *= config_.learning_rate;
  num_minibatches_++;
  if (config_.max_param_change > 0.0) {
    BaseFloat delta = scale * embedding_deriv->FrobeniusNorm();
    // 'delta' is the 2-norm of the change in parameters.
    if (delta > config_.max_param_change) {
      BaseFloat max_change_scale = config_.max_param_change / delta;
      KALDI_LOG << "Applying max-change with scale " << max_change_scale
                << " since param-change=" << delta << " > "
                << " --embedding.max-param-change="
                << config_.max_param_change;
      max_change_count_++;
      scale *= max_change_scale;
    }
  }

  if (config_.momentum > 0.0) {
    // Multiply the factor (1 - momentum) into the learning rate, which cancels
    // out the scale (1 / (1 - momentum)) which otherwise appears in the
    // effective learning rate due to the geometric sum of (1 + momentum +
    // momentum^2, ...).
    scale *= (1.0 - config_.momentum);
    embedding_deriv->AddToRows(scale, active_words, &embedding_mat_momentum_);
    embedding_mat_->AddMat(1.0, embedding_mat_momentum_);
    embedding_mat_momentum_.Scale(config_.momentum);
  } else {
    embedding_deriv->AddToRows(scale, active_words, embedding_mat_);
  }
}

void RnnlmEmbeddingTrainer::TrainBackstitch(
    bool is_backstitch_step1,
    const CuArrayBase<int32> &active_words,
    CuMatrixBase<BaseFloat> *embedding_deriv) {

  // backstitch training is incompatible with momentum > 0
  KALDI_ASSERT(config_.momentum == 0.0);

  KALDI_ASSERT(active_words.Dim() == embedding_deriv->NumRows());

  // If relevant, do the following:
  // "embedding_deriv += - 2 * l2_regularize * embedding_mat_"
  // This is an approximate to the regular l2 regularization (add l2 regularization
  // to the objective function).
  if (config_.l2_regularize > 0.0 && !is_backstitch_step1) {
    BaseFloat l2_term = -2 * config_.l2_regularize;
    if (l2_term != 0.0) {
      embedding_deriv->AddRows(l2_term / (1.0 + config_.backstitch_training_scale),
                               *embedding_mat_, active_words);
    }
  }
  BaseFloat scale = 1.0;
  if (config_.use_natural_gradient) {
    if (is_backstitch_step1) preconditioner_.Freeze(true);
    preconditioner_.PreconditionDirections(embedding_deriv, &scale);
  }
  scale *= config_.learning_rate;
  if (config_.max_param_change > 0.0) {
    BaseFloat delta = scale * embedding_deriv->FrobeniusNorm();
    // 'delta' is the 2-norm of the change in parameters.
    if (delta > config_.max_param_change) {
      BaseFloat max_change_scale = config_.max_param_change / delta;
      KALDI_LOG << "Applying max-change with scale " << max_change_scale
                << " since param-change=" << delta << " > "
                << " --embedding.max-param-change="
                << config_.max_param_change;
      max_change_count_++;
      scale *= max_change_scale;
    }
  }
  if (is_backstitch_step1) {
    scale *= -config_.backstitch_training_scale;
    if (config_.use_natural_gradient) preconditioner_.Freeze(false);
  } else {
    scale *= 1.0 + config_.backstitch_training_scale;
    num_minibatches_++;
  }
  embedding_deriv->AddToRows(scale, active_words, embedding_mat_);
}

RnnlmEmbeddingTrainer::~RnnlmEmbeddingTrainer() {
  PrintStats();
}

void RnnlmEmbeddingTrainer::PrintStats() {
  KALDI_LOG << "Processed a total of " << num_minibatches_ << " minibatches."
            << "max-change was enforced "
            << (100.0 * max_change_count_) /
               (num_minibatches_ *
               (config_.backstitch_training_scale == 0.0 ? 1.0 :
               1.0 + 1.0 / config_.backstitch_training_interval))
            << " % of the time.";

  Matrix<BaseFloat> delta_embedding_mat(*embedding_mat_);
  delta_embedding_mat.AddMat(-1.0, initial_embedding_mat_);

  BaseFloat param_change_2norm = delta_embedding_mat.FrobeniusNorm(),
      baseline_params_2norm = initial_embedding_mat_.FrobeniusNorm(),
      final_params_2norm = embedding_mat_->FrobeniusNorm(),
      relative_param_change = param_change_2norm / baseline_params_2norm;

  KALDI_LOG << "Norm of embedding-matrix differences is " << param_change_2norm
            << " (initial norm of matrix was " << baseline_params_2norm
            << "; now it is " << final_params_2norm << ")";
  KALDI_LOG << "Relative change in embedding matrix is "
            << relative_param_change;
}

}  // namespace rnnlm
}  // namespace kaldi

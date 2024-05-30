// rnnlm/rnnlm-training.cc

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

#include "rnnlm/rnnlm-training.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace rnnlm {



RnnlmTrainer::RnnlmTrainer(bool train_embedding,
                           const RnnlmCoreTrainerOptions &core_config,
                           const RnnlmEmbeddingTrainerOptions &embedding_config,
                           const RnnlmObjectiveOptions &objective_config,
                           const CuSparseMatrix<BaseFloat> *word_feature_mat,
                           CuMatrix<BaseFloat> *embedding_mat,
                           nnet3::Nnet *rnnlm):
    train_embedding_(train_embedding),
    core_config_(core_config),
    objective_config_(objective_config),
    rnnlm_(rnnlm),
    core_trainer_(NULL),
    embedding_mat_(embedding_mat),
    embedding_trainer_(NULL),
    word_feature_mat_(word_feature_mat),
    num_minibatches_processed_(0),
    srand_seed_(RandInt(0, 100000)) {


  int32 rnnlm_input_dim = rnnlm_->InputDim("input"),
      rnnlm_output_dim = rnnlm_->OutputDim("output"),
      embedding_dim = embedding_mat->NumCols();
  if (rnnlm_input_dim != embedding_dim ||
      rnnlm_output_dim != embedding_dim)
    KALDI_ERR << "Expected RNNLM to have input-dim and output-dim "
              << "equal to embedding dimension " << embedding_dim
              << " but got " << rnnlm_input_dim << " and "
              << rnnlm_output_dim;
  core_trainer_ = new RnnlmCoreTrainer(core_config_, objective_config_, rnnlm_);

  if (train_embedding) {
    embedding_trainer_ = new RnnlmEmbeddingTrainer(embedding_config,
                                                   embedding_mat_);
  } else {
    embedding_trainer_ = NULL;
  }

  if (word_feature_mat_ != NULL) {
    int32 feature_dim = word_feature_mat_->NumCols();
    if (feature_dim != embedding_mat_->NumRows()) {
      KALDI_ERR << "Word-feature mat (e.g. from --read-sparse-word-features) "
          "has num-cols/feature-dim=" << word_feature_mat_->NumCols()
              << " but embedding matrix has num-rows/feature-dim="
                << embedding_mat_->NumRows() << " (mismatch).";
    }
  }
}


void RnnlmTrainer::Train(RnnlmExample *minibatch) {
  // check the minibatch for sanity.
  if (minibatch->vocab_size != VocabSize())
      KALDI_ERR << "Vocabulary size mismatch: expected "
                << VocabSize() << ", got "
                << minibatch->vocab_size;

  current_minibatch_.Swap(minibatch);
  num_minibatches_processed_++;
  RnnlmExampleDerived derived;
  CuArray<int32> active_words_cuda;
  CuSparseMatrix<BaseFloat> active_word_features;
  CuSparseMatrix<BaseFloat> active_word_features_trans;

  if (!current_minibatch_.sampled_words.empty()) {
    std::vector<int32> active_words;
    RenumberRnnlmExample(&current_minibatch_, &active_words);
    active_words_cuda.CopyFromVec(active_words);

    if (word_feature_mat_ != NULL) {
      active_word_features.SelectRows(active_words_cuda,
                                      *word_feature_mat_);
      active_word_features_trans.CopyFromSmat(active_word_features,
                                              kTrans);
    }
  }
  GetRnnlmExampleDerived(current_minibatch_, train_embedding_,
                         &derived);

  derived_.Swap(&derived);
  active_words_.Swap(&active_words_cuda);
  active_word_features_.Swap(&active_word_features);
  active_word_features_trans_.Swap(&active_word_features_trans);

  TrainInternal();

  if (num_minibatches_processed_ == 1)
    core_trainer_->ConsolidateMemory();
}


void RnnlmTrainer::GetWordEmbedding(CuMatrix<BaseFloat> *word_embedding_storage,
                                    CuMatrix<BaseFloat> **word_embedding) {
  RnnlmExample &minibatch = current_minibatch_;
  bool sampling = !minibatch.sampled_words.empty();

  if (word_feature_mat_ == NULL) {
    // There is no sparse word-feature matrix.
    if (!sampling) {
      KALDI_ASSERT(active_words_.Dim() == 0);
      // There is no sparse word-feature matrix, so the embedding matrix is just
      // embedding_mat_ (the embedding matrix for all words).
      *word_embedding = embedding_mat_;
      KALDI_ASSERT(minibatch.vocab_size == embedding_mat_->NumRows());
    } else {
      // There is sampling-- we're using a subset of the words so the user wants
      // an embedding matrix for just those rows.
      KALDI_ASSERT(active_words_.Dim() != 0);
      word_embedding_storage->Resize(active_words_.Dim(),
                                     embedding_mat_->NumCols(),
                                     kUndefined);
      word_embedding_storage->CopyRows(*embedding_mat_, active_words_);
      *word_embedding = word_embedding_storage;
    }
  } else {
    // There is a sparse word-feature matrix, so we need to multiply it by the
    // feature-embedding matrix in order to get the word-embedding matrix.
    const CuSparseMatrix<BaseFloat> &word_feature_mat =
        sampling ? active_word_features_ : *word_feature_mat_;
    word_embedding_storage->Resize(word_feature_mat.NumRows(),
                                   embedding_mat_->NumCols());
    word_embedding_storage->AddSmatMat(1.0, word_feature_mat, kNoTrans,
                                       *embedding_mat_, 0.0);
    *word_embedding = word_embedding_storage;
  }
}



void RnnlmTrainer::TrainWordEmbedding(
    CuMatrixBase<BaseFloat> *word_embedding_deriv) {
  RnnlmExample &minibatch = current_minibatch_;
  bool sampling = !minibatch.sampled_words.empty();

  if (word_feature_mat_ == NULL) {
    // There is no sparse word-feature matrix.
    if (!sampling) {
      embedding_trainer_->Train(word_embedding_deriv);
    } else {
      embedding_trainer_->Train(active_words_,
                                word_embedding_deriv);
    }
  } else {
    // There is a sparse word-feature matrix, so we need to multiply by it
    // to get the derivative w.r.t. the feature-embedding matrix.

    if (!sampling && word_feature_mat_transpose_.NumRows() == 0)
      word_feature_mat_transpose_.CopyFromSmat(*word_feature_mat_, kTrans);

    CuMatrix<BaseFloat> feature_embedding_deriv(embedding_mat_->NumRows(),
                                                embedding_mat_->NumCols());
    const CuSparseMatrix<BaseFloat> &word_features_trans =
        (sampling ? active_word_features_trans_ : word_feature_mat_transpose_);

    feature_embedding_deriv.AddSmatMat(1.0, word_features_trans, kNoTrans,
                                       *word_embedding_deriv, 0.0);

    // TODO: eventually remove these lines.
    KALDI_VLOG(3) << "word-features-trans sum is " << word_features_trans.Sum()
                  << ", word-embedding-deriv-sum is " << word_embedding_deriv->Sum()
                  << ", feature-embedding-deriv-sum is " << feature_embedding_deriv.Sum();

    embedding_trainer_->Train(&feature_embedding_deriv);
  }
}

void RnnlmTrainer::TrainBackstitchWordEmbedding(
    bool is_backstitch_step1,
    CuMatrixBase<BaseFloat> *word_embedding_deriv) {
  RnnlmExample &minibatch = current_minibatch_;
  bool sampling = !minibatch.sampled_words.empty();

  if (word_feature_mat_ == NULL) {
    // There is no sparse word-feature matrix.
    if (!sampling) {
      embedding_trainer_->TrainBackstitch(is_backstitch_step1,
                                          word_embedding_deriv);
    } else {
      embedding_trainer_->TrainBackstitch(is_backstitch_step1, active_words_,
                                          word_embedding_deriv);
    }
  } else {
    // There is a sparse word-feature matrix, so we need to multiply by it
    // to get the derivative w.r.t. the feature-embedding matrix.

    if (!sampling && word_feature_mat_transpose_.NumRows() == 0)
      word_feature_mat_transpose_.CopyFromSmat(*word_feature_mat_, kTrans);

    CuMatrix<BaseFloat> feature_embedding_deriv(embedding_mat_->NumRows(),
                                                embedding_mat_->NumCols());
    const CuSparseMatrix<BaseFloat> &word_features_trans =
        (sampling ? active_word_features_trans_ : word_feature_mat_transpose_);

    feature_embedding_deriv.AddSmatMat(1.0, word_features_trans, kNoTrans,
                                       *word_embedding_deriv, 0.0);

    // TODO: eventually remove these lines.
    KALDI_VLOG(3) << "word-features-trans sum is " << word_features_trans.Sum()
                  << ", word-embedding-deriv-sum is " << word_embedding_deriv->Sum()
                  << ", feature-embedding-deriv-sum is " << feature_embedding_deriv.Sum();

    embedding_trainer_->TrainBackstitch(is_backstitch_step1,
                                        &feature_embedding_deriv);
  }
}


void RnnlmTrainer::TrainInternal() {
  CuMatrix<BaseFloat> word_embedding_storage;
  CuMatrix<BaseFloat> *word_embedding;
  GetWordEmbedding(&word_embedding_storage, &word_embedding);

  CuMatrix<BaseFloat> word_embedding_deriv;
  if (train_embedding_)
    word_embedding_deriv.Resize(word_embedding->NumRows(),
                                word_embedding->NumCols());

  if (core_config_.backstitch_training_scale > 0.0 &&
      num_minibatches_processed_ % core_config_.backstitch_training_interval ==
      srand_seed_ % core_config_.backstitch_training_interval) {
    bool is_backstitch_step1 = true;
    srand(srand_seed_ + num_minibatches_processed_);
    core_trainer_->TrainBackstitch(is_backstitch_step1, current_minibatch_,
        derived_, *word_embedding,
        (train_embedding_ ? &word_embedding_deriv : NULL));
    if (train_embedding_)
      TrainBackstitchWordEmbedding(is_backstitch_step1, &word_embedding_deriv);

    is_backstitch_step1 = false;
    srand(srand_seed_ + num_minibatches_processed_);
    core_trainer_->TrainBackstitch(is_backstitch_step1, current_minibatch_,
        derived_, *word_embedding,
        (train_embedding_ ? &word_embedding_deriv : NULL));
    if (train_embedding_)
      TrainBackstitchWordEmbedding(is_backstitch_step1, &word_embedding_deriv);
  } else {
    core_trainer_->Train(current_minibatch_, derived_, *word_embedding,
                         (train_embedding_ ? &word_embedding_deriv : NULL));
    if (train_embedding_)
      TrainWordEmbedding(&word_embedding_deriv);
  }
}

int32 RnnlmTrainer::VocabSize() {
  if (word_feature_mat_ != NULL) return word_feature_mat_->NumRows();
  else return embedding_mat_->NumRows();
}

RnnlmTrainer::~RnnlmTrainer() {
  // Note: the following delete statements may cause some diagnostics to be
  // issued, from the destructors of those classes.
  if (core_trainer_)
    delete core_trainer_;
  if (embedding_trainer_)
    delete embedding_trainer_;

  KALDI_LOG << "Trained on " << num_minibatches_processed_
            << " minibatches.\n";
}



}  // namespace rnnlm
}  // namespace kaldi

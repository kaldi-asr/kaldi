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
#include "cudamatrix/cu-rand.h"

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
    embedding_config_(embedding_config),
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



RnnlmTrainerAdapt::RnnlmTrainerAdapt(bool train_embedding,
                           const RnnlmCoreTrainerOptions &core_config,
                           const RnnlmEmbeddingTrainerOptions &embedding_config,
                           const RnnlmObjectiveOptions &objective_config,
                           CuMatrix<BaseFloat> *embedding_mat_large,
                           CuMatrix<BaseFloat> *embedding_mat_med,
                           CuMatrix<BaseFloat> *embedding_mat_small,
                           nnet3::Nnet *rnnlm,
                           int32 cutofflarge,
                           int32 cutoffmed):
    train_embedding_(train_embedding),
    core_config_(core_config),
    embedding_config_(embedding_config),
    objective_config_(objective_config),
    rnnlm_(rnnlm),
    core_trainer_(NULL),
    embedding_mat_large_(embedding_mat_large),
    embedding_mat_med_(embedding_mat_med),
    embedding_mat_small_(embedding_mat_small),
    embedding_trainer_large_(NULL),
    embedding_trainer_med_(NULL),
    embedding_trainer_small_(NULL),
    num_minibatches_processed_(0),
    cutoff_large(cutofflarge),
    cutoff_med(cutoffmed),
    srand_seed_(RandInt(0, 100000)) {

  core_trainer_ = new RnnlmCoreTrainerAdapt(core_config_, objective_config_, rnnlm_);

  KALDI_ASSERT (train_embedding);
  embedding_trainer_large_ = new RnnlmEmbeddingTrainer(embedding_config,
                                                 embedding_mat_large_);
  embedding_trainer_med_ = new RnnlmEmbeddingTrainer(embedding_config,
                                                 embedding_mat_med_);
  embedding_trainer_small_ = new RnnlmEmbeddingTrainer(embedding_config,
                                                 embedding_mat_small_);

}


void RnnlmTrainerAdapt::Train(RnnlmExample *minibatch) {
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

  std::vector<int32> active_words_cpu;
  if (!current_minibatch_.sampled_words.empty()) {
    RenumberRnnlmExample(&current_minibatch_, &active_words_cpu);
    active_words_cuda.CopyFromVec(active_words_cpu);

  }
  SetMaskVectors(&active_words_cpu, &current_minibatch_, cutoff_large, cutoff_med);

  GetRnnlmExampleDerivedAdapt(current_minibatch_, train_embedding_,
                              &derived);

  int32 sz = active_words_cpu.size();
  std::vector<int32> active_words_large(sz);
  std::vector<int32> active_words_med(sz);
  std::vector<int32> active_words_small(sz);
  std::vector<int32> active_words_large_filt;
  std::vector<int32> active_words_med_filt;
  std::vector<int32> active_words_small_filt;
  std::vector<int32> large_embed_active;
  std::vector<int32> med_embed_active;
  std::vector<int32> small_embed_active;
  for(int32 i = 0; i < sz; i++) {
    int32 idx = active_words_cpu[i];
    if (idx < cutoff_large) {
      large_embed_active.push_back(i);
      active_words_large_filt.push_back(idx);
      active_words_large[i] = idx;
      active_words_med[i] = -1;
      active_words_small[i] = -1;
    } else if (idx < (cutoff_med + cutoff_large)) {
      med_embed_active.push_back(i);
      active_words_med_filt.push_back(idx - cutoff_large);
      active_words_large[i] = -1;
      active_words_med[i] = idx - cutoff_large;
      active_words_small[i] = -1;
    } else {
      small_embed_active.push_back(i);
      active_words_small_filt.push_back(idx - cutoff_large - cutoff_med);
      active_words_large[i] = -1;
      active_words_med[i] = -1;
      active_words_small[i] = idx - cutoff_large - cutoff_med;
    }
  }
  active_words_large_ = active_words_large;
  active_words_med_ = active_words_med;
  active_words_small_ = active_words_small;
  large_embed_active_ = large_embed_active;
  med_embed_active_ = med_embed_active;
  small_embed_active_ = small_embed_active;
  active_words_large_filt_ = active_words_large_filt;
  active_words_med_filt_ = active_words_med_filt;
  active_words_small_filt_ = active_words_small_filt;

  derived_.Swap(&derived);
  active_words_.Swap(&active_words_cuda);
  active_word_features_.Swap(&active_word_features);
  active_word_features_trans_.Swap(&active_word_features_trans);

  TrainInternal();

  if (num_minibatches_processed_ == 1)
    core_trainer_->ConsolidateMemory();
}


void RnnlmTrainerAdapt::GetWordEmbedding(CuMatrix<BaseFloat> *word_embedding_storage_large,
                                    CuMatrix<BaseFloat> *word_embedding_storage_med,
                                    CuMatrix<BaseFloat> *word_embedding_storage_small,
                                    CuMatrix<BaseFloat> **word_embedding_large,
                                    CuMatrix<BaseFloat> **word_embedding_med,
                                    CuMatrix<BaseFloat> **word_embedding_small) {
  RnnlmExample &minibatch = current_minibatch_;
  bool sampling = !minibatch.sampled_words.empty();

  KALDI_ASSERT(word_feature_mat_ == NULL);
   // There is no sparse word-feature matrix.
  KALDI_ASSERT (sampling);

  word_embedding_storage_large->Resize(active_words_.Dim(),
                                 embedding_mat_large_->NumCols(),
                                   kUndefined);
  word_embedding_storage_large->CopyRows(*embedding_mat_large_, active_words_large_);

  word_embedding_storage_med->Resize(active_words_.Dim(),
                                 embedding_mat_med_->NumCols(),
                                   kUndefined);
  word_embedding_storage_med->CopyRows(*embedding_mat_med_, active_words_med_);

  word_embedding_storage_small->Resize(active_words_.Dim(),
                                 embedding_mat_small_->NumCols(),
                                   kUndefined);
  word_embedding_storage_small->CopyRows(*embedding_mat_small_, active_words_small_);

  *word_embedding_large = word_embedding_storage_large;
  *word_embedding_med = word_embedding_storage_med;
  *word_embedding_small = word_embedding_storage_small;
}



void RnnlmTrainerAdapt::TrainWordEmbedding(
    CuMatrixBase<BaseFloat> *word_embedding_deriv_large,
    CuMatrixBase<BaseFloat> *word_embedding_deriv_med,
    CuMatrixBase<BaseFloat> *word_embedding_deriv_small) {
  RnnlmExample &minibatch = current_minibatch_;
  bool sampling = !minibatch.sampled_words.empty();

  KALDI_ASSERT (word_feature_mat_ == NULL);
  KALDI_ASSERT (sampling);
  int32 sz = large_embed_active_.Dim();

  CuMatrix<BaseFloat> word_emb_deriv_large_filtered(sz, word_embedding_deriv_large->NumCols());
  word_emb_deriv_large_filtered.CopyRows(*word_embedding_deriv_large, large_embed_active_);

  embedding_trainer_large_->Train(active_words_large_filt_,
                            &word_emb_deriv_large_filtered);


  sz = med_embed_active_.Dim();

  CuMatrix<BaseFloat> word_emb_deriv_med_filtered(sz, word_embedding_deriv_med->NumCols());
  word_emb_deriv_med_filtered.CopyRows(*word_embedding_deriv_med, med_embed_active_);

  embedding_trainer_med_->Train(active_words_med_filt_,
                            &word_emb_deriv_med_filtered);

  sz = small_embed_active_.Dim();
  CuMatrix<BaseFloat> word_emb_deriv_small_filtered(sz, word_embedding_deriv_small->NumCols());
    word_emb_deriv_small_filtered.CopyRows(*word_embedding_deriv_small, small_embed_active_);

  embedding_trainer_small_->Train(active_words_small_filt_,
                            &word_emb_deriv_small_filtered);

}


void RnnlmTrainerAdapt::TrainInternal() {
  CuMatrix<BaseFloat> word_embedding_storage_large;
  CuMatrix<BaseFloat> *word_embedding_large;
  CuMatrix<BaseFloat> word_embedding_storage_med;
  CuMatrix<BaseFloat> *word_embedding_med;
  CuMatrix<BaseFloat> word_embedding_storage_small;
  CuMatrix<BaseFloat> *word_embedding_small;
  GetWordEmbedding(&word_embedding_storage_large, &word_embedding_storage_med, &word_embedding_storage_small, 
    &word_embedding_large, &word_embedding_med, &word_embedding_small);

  CuMatrix<BaseFloat> word_embedding_deriv_large;
  CuMatrix<BaseFloat> word_embedding_deriv_med;
  CuMatrix<BaseFloat> word_embedding_deriv_small;

  word_embedding_deriv_large.Resize(word_embedding_large->NumRows(),
                                word_embedding_large->NumCols());
  word_embedding_deriv_med.Resize(word_embedding_med->NumRows(),
                                word_embedding_med->NumCols());                                
  word_embedding_deriv_small.Resize(word_embedding_small->NumRows(),
                                word_embedding_small->NumCols());


  core_trainer_->Train(current_minibatch_, derived_, *word_embedding_large, *word_embedding_med, *word_embedding_small,
                       &word_embedding_deriv_large, &word_embedding_deriv_med, &word_embedding_deriv_small);
  KALDI_ASSERT (train_embedding_);
  TrainWordEmbedding(&word_embedding_deriv_large, &word_embedding_deriv_med, &word_embedding_deriv_small);
}

int32 RnnlmTrainerAdapt::VocabSize() {
  return embedding_mat_large_->NumRows() + embedding_mat_med_->NumRows() + embedding_mat_small_->NumRows();
}

RnnlmTrainerAdapt::~RnnlmTrainerAdapt() {
  // Note: the following delete statements may cause some diagnostics to be
  // issued, from the destructors of those classes.
  if (core_trainer_)
    delete core_trainer_;
  if (embedding_trainer_large_)
    delete embedding_trainer_large_;
    if (embedding_trainer_med_)
    delete embedding_trainer_med_;
  if (embedding_trainer_small_)
    delete embedding_trainer_small_;

  KALDI_LOG << "Trained on " << num_minibatches_processed_
            << " minibatches.\n";
}


}  // namespace rnnlm
}  // namespace kaldi

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


bool RnnlmTrainerOptions::HasRequiredArgs() {
  bool ans = true;
  if (rnnlm_rxfilename == "") {
    KALDI_WARN << "--read-rnnlm option is required";
    ans = false;
  }
  if (rnnlm_wxfilename == "") {
    KALDI_WARN << "--write-rnnlm option is (currently) required";
    ans = false;
  }
  if (embedding_rxfilename == "") {
    KALDI_WARN << "--read-embedding option is required.";
    ans = false;
  }
  return ans;
}
RnnlmTrainer::RnnlmTrainer(const RnnlmTrainerOptions &config):
    config_(config),
    num_minibatches_processed_(0),
    end_of_input_(false),
    previous_minibatch_empty_(1),
    current_minibatch_empty_(1) {
  ReadKaldiObject(config_.rnnlm_rxfilename, &rnnlm_);
  if (!IsSimpleNnet(rnnlm_))
    KALDI_ERR << "Input RNNLM in " << config_.rnnlm_rxfilename
             << " is not the type of neural net we were looking for; "
                "failed IsSimpleNnet().";
  ReadKaldiObject(config_.embedding_rxfilename, &embedding_mat_);
  int32 rnnlm_input_dim = rnnlm_.InputDim("input"),
      rnnlm_output_dim = rnnlm_.OutputDim("output"),
      embedding_dim = embedding_mat_.NumCols();
  if (rnnlm_input_dim != embedding_dim ||
      rnnlm_output_dim != embedding_dim)
    KALDI_ERR << "Expected RNNLM to have input-dim and output-dim "
              << "equal to embedding dimension " << embedding_dim
              << " but got " << rnnlm_input_dim << " and "
              << rnnlm_output_dim;
  core_trainer_ = new RnnlmCoreTrainer(config_.core_config, &rnnlm_);

  if (config_.embedding_wxfilename != "") {
    embedding_trainer_ = new RnnlmEmbeddingTrainer(config_.embedding_config,
                                                   &embedding_mat_);
  } else {
    embedding_trainer_ = NULL;
  }

  if (config_.word_features_rxfilename != "") {
    // binary mode is not supported here; it's a text format.
    Input input(config_.word_features_rxfilename);
    int32 feature_dim = embedding_mat_.NumRows();
    SparseMatrix<BaseFloat> word_feature_mat;
    ReadSparseWordFeatures(input.Stream(), feature_dim,
                           &word_feature_mat);
    word_feature_mat_.Swap(&word_feature_mat);  // copy to GPU
  }

  // Start a thread that calls run_background_thread(this).
  // That thread will be responsible for computing derived variables of
  // the minibatch, since that can be done independently of the main
  // training process.
  background_thread_ = std::thread(run_background_thread, this);

}


void RnnlmTrainer::Train(RnnlmExample *minibatch) {
  // check the minibatch for sanity.
  if (minibatch->vocab_size != VocabSize())
      KALDI_ERR << "Vocabulary size mismatch: expected "
                << VocabSize() << ", got "
                << minibatch->vocab_size;

  // hand over 'minibatch' to the background thread to have its derived variable
  // computed, via the class variable 'current_minibatch_'.
  current_minibatch_empty_.Wait();
  current_minibatch_.Swap(minibatch);
  current_minibatch_full_.Signal();
  num_minibatches_processed_++;
  if (num_minibatches_processed_ == 1) {
    return;  // The first time this function is called, return immediately.
  }
  previous_minibatch_full_.Wait();
  TrainInternal();
  previous_minibatch_empty_.Signal();
}


void RnnlmTrainer::GetWordEmbedding(CuMatrix<BaseFloat> *word_embedding_storage,
                                    CuMatrix<BaseFloat> **word_embedding) {
  RnnlmExample &minibatch = previous_minibatch_;
  bool sampling = !minibatch.sampled_words.empty();

  if (word_feature_mat_.NumRows() == 0) {
    // There is no sparse word-feature matrix.
    if (!sampling) {
      KALDI_ASSERT(active_words_.Dim() == 0);
      // There is no sparse word-feature matrix, so the embedding matrix is just
      // embedding_mat_ (the embedding matrix for all words).
      *word_embedding = &embedding_mat_;
      KALDI_ASSERT(minibatch.vocab_size == embedding_mat_.NumRows());
    } else {
      // There is sampling-- we're using a subset of the words so the user wants
      // an embedding matrix for just those rows.
      KALDI_ASSERT(active_words_.Dim() != 0);
      word_embedding_storage->Resize(active_words_.Dim(),
                                     embedding_mat_.NumCols(),
                                     kUndefined);
      word_embedding_storage->CopyRows(embedding_mat_, active_words_);
      *word_embedding = word_embedding_storage;
    }
  } else {
    // There is a sparse word-feature matrix, so we need to multiply it by the
    // feature-embedding matrix in order to get the word-embedding matrix.
    const CuSparseMatrix<BaseFloat> &word_feature_mat =
        sampling ? active_word_features_ : word_feature_mat_;
    word_embedding_storage->Resize(word_feature_mat.NumRows(),
                                   embedding_mat_.NumCols());
    word_embedding_storage->AddSmatMat(1.0, word_feature_mat, kNoTrans,
                                       embedding_mat_, 0.0);
    *word_embedding = word_embedding_storage;
  }
}



void RnnlmTrainer::TrainWordEmbedding(
    CuMatrixBase<BaseFloat> *word_embedding_deriv) {
  RnnlmExample &minibatch = previous_minibatch_;
  bool sampling = !minibatch.sampled_words.empty();

  if (word_feature_mat_.NumRows() == 0) {
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
    CuMatrix<BaseFloat> feature_embedding_deriv(embedding_mat_.NumRows(),
                                                embedding_mat_.NumCols());
    const CuSparseMatrix<BaseFloat> &word_features_trans =
        (sampling ? active_word_features_trans_ : word_feature_mat_transpose_);
    feature_embedding_deriv.AddMatSmat(1.0, *word_embedding_deriv,
                                       word_features_trans, kTrans,
                                       0.0);
    embedding_trainer_->Train(&feature_embedding_deriv);
  }
}




void RnnlmTrainer::TrainInternal() {
  CuMatrix<BaseFloat> word_embedding_storage;
  CuMatrix<BaseFloat> *word_embedding;
  GetWordEmbedding(&word_embedding_storage, &word_embedding);
  // 'training_embedding' is true if we are training the embedding matrix
  bool training_embedding =  (config_.embedding_wxfilename != "");


  CuMatrix<BaseFloat> word_embedding_deriv;
  if (training_embedding)
    word_embedding_deriv.Resize(word_embedding->NumRows(),
                                word_embedding->NumCols());

  core_trainer_->Train(previous_minibatch_, derived_, *word_embedding,
                       (training_embedding ? &word_embedding_deriv : NULL));
  if (training_embedding)
    TrainWordEmbedding(&word_embedding_deriv);
}

int32 RnnlmTrainer::VocabSize() {
  if (word_feature_mat_.NumRows() > 0) return word_feature_mat_.NumRows();
  else return embedding_mat_.NumRows();
}

void RnnlmTrainer::RunBackgroundThread() {
  while (true) {
    current_minibatch_full_.Wait();
    if (end_of_input_)
      return;
    RnnlmExampleDerived derived;
    CuArray<int32> active_words_cuda;
    CuSparseMatrix<BaseFloat> active_word_features;
    CuSparseMatrix<BaseFloat> active_word_features_trans;

    if (!current_minibatch_.sampled_words.empty()) {
      std::vector<int32> active_words;
      RenumberRnnlmExample(&current_minibatch_, &active_words);
      active_words_cuda.CopyFromVec(active_words);

      if (word_feature_mat_.NumRows() > 0) {
        active_word_features.SelectRows(active_words_cuda,
                                        word_feature_mat_);
        active_word_features_trans.CopyFromSmat(active_word_features,
                                                kTrans);
      }
    }
    bool need_embedding_deriv = (config_.embedding_wxfilename != "");
    GetRnnlmExampleDerived(current_minibatch_, need_embedding_deriv,
                           &derived);

    // Wait until the main thread is not currently processing
    // previous_minibatch_; once we get this semaphore we are free to write to
    // it and other related variables such as 'derived_'.
    previous_minibatch_empty_.Wait();
    previous_minibatch_.Swap(&current_minibatch_);
    derived_.Swap(&derived);
    active_words_.Swap(&active_words_cuda);
    active_word_features_.Swap(&active_word_features);
    active_word_features_trans_.Swap(&active_word_features_trans);

    // The following statement signals that 'previous_minibatch_'
    // and related variables have been written to by this thread.
    previous_minibatch_full_.Signal();
    // The following statement signals that 'current_minibatch_'
    // has been consumed by this thread and is no longer needed.
    current_minibatch_empty_.Signal();
  }
}

RnnlmTrainer::~RnnlmTrainer() {
  // Note: the following delete statements may cause some diagnostics to be
  // issued, from the destructors of those classes.
  if (core_trainer_)
    delete core_trainer_;
  if (embedding_trainer_)
    delete embedding_trainer_;
  // Now write out the things we need to write out.
  if (config_.rnnlm_wxfilename != "") {
    WriteKaldiObject(rnnlm_, config_.rnnlm_wxfilename, config_.binary);
    KALDI_LOG << "Wrote RNNLM to "
              << PrintableWxfilename(config_.rnnlm_wxfilename);
  }
  if (config_.embedding_wxfilename != "") {
    WriteKaldiObject(embedding_mat_, config_.embedding_wxfilename,
                     config_.binary);
    KALDI_LOG << "Wrote embedding matrix to "
              << PrintableWxfilename(config_.embedding_wxfilename);
  }
}



}  // namespace rnnlm
}  // namespace kaldi



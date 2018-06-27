// rnnlm/rnnlm-training.h

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_RNNLM_RNNLM_TRAINING_H_
#define KALDI_RNNLM_RNNLM_TRAINING_H_

#include "rnnlm/rnnlm-core-training.h"
#include "rnnlm/rnnlm-embedding-training.h"
#include "rnnlm/rnnlm-utils.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "util/kaldi-semaphore.h"


namespace kaldi {
namespace rnnlm {

/*
  The class RnnlmTrainer is for training an RNNLM (one individual training job, not
  the top-level logic about learning rate schedules, parameter averaging, and the
  like); it contains the most of the logic that the command-line program rnnlm-train
  implements.
*/

class RnnlmTrainer {
 public:
  /**
     Constructor
      @param [in] train_embedding  True if the user wants us to
                             train the embedding matrix
      @param [in] core_config  Options for training the core
                              RNNLM
      @param [in] embedding_config  Options for training the
                              embedding matrix (only relevant
                              if train_embedding is true).
      @param [in] objective_config  Options relating to the objective
                              function used for training.
      @param [in] word_feature_mat Either NULL, or a pointer to a sparse
                             word-feature matrix of dimension vocab-size by
                             feature-dim, where vocab-size is the
                             highest-numbered word plus one.
      @param [in,out] embedding_mat  Pointer to the embedding
                             matrix; this is trained if train_embedding is true,
                             and in either case this class retains the pointer
                             to 'embedding_mat' during its livetime.
                             If word_feature_mat is NULL, this
                             is the word-embedding matrix of dimension
                             vocab-size by embedding-dim; otherwise it is the
                             feature-embedding matrix of dimension feature-dim by
                             by embedding-dim, and we have to multiply it by
                             word_feature_mat to get the word embedding matrix.
      @param [in,out] rnnlm  The RNNLM to be trained.  The class will retain
                             this pointer and modify the neural net in-place.
  */
  RnnlmTrainer(bool train_embedding,
               const RnnlmCoreTrainerOptions &core_config,
               const RnnlmEmbeddingTrainerOptions &embedding_config,
               const RnnlmObjectiveOptions &objective_config,
               const CuSparseMatrix<BaseFloat> *word_feature_mat,
               CuMatrix<BaseFloat> *embedding_mat,
               nnet3::Nnet *rnnlm);



  // Train on one example.  The example is provided as a pointer because we
  // acquire it destructively, via Swap().
  void Train(RnnlmExample *minibatch);


  // The destructor writes out any files that we need to write out.
  ~RnnlmTrainer();

  int32 NumMinibatchesProcessed() { return num_minibatches_processed_; }

 private:

  int32 VocabSize();

  /// This function contains the actual training code, it's called from Train();
  /// it trains on minibatch_previous_.
  void TrainInternal();

  /// This function works out the word-embedding matrix for the minibatch we're
  /// training on (previous_minibatch_).  The word-embedding matrix for this
  /// minibatch is a matrix of dimension current_minibatch_.vocab_size by
  /// embedding_mat_.NumRows().  This function sets '*word_embedding' to be a
  /// pointer to the embedding matrix, which will either be '&embedding_mat_'
  /// (in the case where there is no sampling and no sparse feature
  /// representation), or 'word_embedding_storage' otherwise.  In the latter
  /// case, 'word_embedding_storage' will be resized and written to
  /// appropriately.
  void GetWordEmbedding(CuMatrix<BaseFloat> *word_embedding_storage,
                        CuMatrix<BaseFloat> **word_embedding);


  /// This function trains the word-embedding matrix for the minibatch we're
  /// training on (in previous_minibatch_).  'embedding_deriv' is the derivative
  /// w.r.t. the word-embedding for this minibatch (of dimension
  /// previus_minibatch_.vocab_size by embedding_mat_.NumCols()).
  /// You can think of it as the backprop for the function 'GetWordEmbedding()'.
  ///   @param [in] word_embedding_deriv   The derivative w.r.t. the embeddings of
  ///                       just the words used in this minibatch
  ///                       (i.e. the minibatch-level word-embedding matrix,
  ///                       possibly using a subset of words).  This is an input
  ///                       but this function consumes it destructively.
  void TrainWordEmbedding(CuMatrixBase<BaseFloat> *word_embedding_deriv);

  /// The backstitch version of the above function.
  void TrainBackstitchWordEmbedding(
      bool is_backstitch_step1,
      CuMatrixBase<BaseFloat> *word_embedding_deriv);

  bool train_embedding_;  // true if we are training the embedding.
  const RnnlmCoreTrainerOptions &core_config_;
  const RnnlmEmbeddingTrainerOptions &embedding_config_;
  const RnnlmObjectiveOptions &objective_config_;

  // The neural net we are training (not owned here)
  nnet3::Nnet *rnnlm_;

  // Pointer to the object that trains 'rnnlm_' (owned here).
  RnnlmCoreTrainer *core_trainer_;

  // The (word or feature) embedding matrix; it's the word embedding matrix if
  // word_feature_mat_.NumRows() == 0, else it's the feature embedding matrix.
  // The dimension is (num-words or num-features) by embedding-dim.
  // It's owned outside this class.
  CuMatrix<BaseFloat> *embedding_mat_;


  // Pointer to the object that trains 'embedding_mat_', or NULL if we are not
  // training it.  Owned here.
  RnnlmEmbeddingTrainer *embedding_trainer_;

  // If the --read-sparse-word-features options is provided, then
  // word_feature_mat_ will contain the matrix of sparse word features, of
  // dimension num-words by num-features.  In this case, the word embedding
  // matrix is the product of this matrix times 'embedding_mat_'.
  // It's owned outside this class.
  const CuSparseMatrix<BaseFloat> *word_feature_mat_;

  // This is the transpose of word_feature_mat_, which is needed only if we
  // train on egs without sampling.  This is only computed once, if and when
  // it's needed.
  CuSparseMatrix<BaseFloat> word_feature_mat_transpose_;

  int32 num_minibatches_processed_;

  RnnlmExample current_minibatch_;

  // The variables derived_ and active_words_ corresponds to group as current_minibatch_.
  RnnlmExampleDerived derived_;
  // Only if we are doing subsampling (depends on the eg), active_words_
  // contains the list of active words for the minibatch 'current_minibatch_';
  // it is a CUDA version of the 'active_words' output by
  // RenumberRnnlmExample().  Otherwise it is empty.
  CuArray<int32> active_words_;
  // Only if we are doing subsampling AND we have sparse word features
  // (i.e. word_feature_mat_ is nonempty), active_word_features_ contains
  // just the rows of word_feature_mat_ which correspond to active_words_.
  // This is a derived quantity computed by the background thread.
  CuSparseMatrix<BaseFloat> active_word_features_;
  // Only if we are doing subsampling AND we have sparse word features,
  // active_word_features_trans_ is the transpose of active_word_features_;
  // This is a derived quantity computed by the background thread.
  CuSparseMatrix<BaseFloat> active_word_features_trans_;

  // This value is used in backstitch training when we need to ensure
  // consistent dropout masks.  It's set to a value derived from rand()
  // when the class is initialized.
  int32 srand_seed_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(RnnlmTrainer);
};


} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_TRAINING_H_

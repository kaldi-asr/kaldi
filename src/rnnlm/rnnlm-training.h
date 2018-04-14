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

#include <thread>
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
  // acquire it destructively, via Swap().  Note: this function doesn't
  // actually train on this eg; what it does is to train on the previous
  // example, and provide this eg to the background thread that computes the
  // derived parameters of the eg.
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

  /// This is the function-call that's run as the background thread which
  /// computes the derived parameters for each minibatch.
  void RunBackgroundThread();

  /// This function is invoked by the newly created background thread.
  static void run_background_thread(RnnlmTrainer *trainer) {
    trainer->RunBackgroundThread();
  }


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


  // num_minibatches_processed_ starts at zero is incremented each time we
  // provide an example to the background thread for computing the derived
  // parameters.
  int32 num_minibatches_processed_;

  // 'current_minibatch' is where the Train() function puts the minibatch that
  // is provided to Train(), so that the background thread can work on it.
  RnnlmExample current_minibatch_;
  // View 'end_of_input_' as part of a unit with current_minibatch_, for threading/access
  // purposes.  It is set by the foreground thread from the destructor, while
  // incrementing the current_minibatch_ready_ semaphore; and when the background
  // thread decrements the semaphore and notices that end_of_input_ is true, it will
  // exit.
  bool end_of_input_;


  // previous_minibatch_ is the previous minibatch that was provided to Train(),
  // but the minibatch that we're currently trainig on.
  RnnlmExample previous_minibatch_;
  // The variables derived_ and active_words_ [and more that I'll add, TODO] are in the same
  // group as previous_minibatch_ from the point of view
  // of threading and access control.
   RnnlmExampleDerived derived_;
  // Only if we are doing subsampling (depends on the eg), active_words_
  // contains the list of active words for the minibatch 'previous_minibatch_';
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


  // The 'previous_minibatch_full_' semaphore is incremented by the background
  // thread once it has written to 'previous_minibatch_' and
  // 'derived_previous_', to let the Train() function know that they are ready
  // to be trained on.  The Train() function waits on this semaphore.
  Semaphore previous_minibatch_full_;

  // The 'previous_minibatch_empty_' semaphore is incremented by the foreground
  // thread when it has done processing previous_minibatch_ and
  // derived_ and active_words_ (and hence, it is safe for the background thread to write
  // to these variables).  The background thread waits on this semaphore once it
  // has finished computing the derived variables; and when it successfully
  // decrements it, it will write to those variables (quickly, via Swap()).
  Semaphore previous_minibatch_empty_;


  // The 'current_minibatch_ready_' semaphore is incremented by the foreground
  // thread from Train(), when it has written the just-provided minibatch to
  // 'current_minibatch_' (it's also incremented by the destructor, together
  // with setting end_of_input_.  The background thread waits on this semaphore
  // before either processing previous_minibatch (if !end_of_input_), or exiting
  // (if end_of_input_).
  Semaphore current_minibatch_full_;

  // The 'current_minibatch_empty_' semaphore is incremented by the background
  // thread when it has done processing current_minibatch_,
  // so, it is safe for the foreground thread to write
  // to this variable).  The foreground thread waits on this semaphore before
  // writing to 'current_minibatch_' (in practice it should get the semaphore
  // immediately since we expect that the foreground thread will have more to
  // do than the background thread).
  Semaphore current_minibatch_empty_;

  std::thread background_thread_;  // Background thread for computing 'derived'
                                   // parameters of a minibatch.

  // This value is used in backstitch training when we need to ensure
  // consistent dropout masks.  It's set to a value derived from rand()
  // when the class is initialized.
  int32 srand_seed_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(RnnlmTrainer);
};


} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_TRAINING_H_

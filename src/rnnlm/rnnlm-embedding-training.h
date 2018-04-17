// rnnlm/rnnlm-embedding-training.h

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

#ifndef KALDI_RNNLM_RNNLM_EMBEDDING_TRAINING_H_
#define KALDI_RNNLM_RNNLM_EMBEDDING_TRAINING_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "rnnlm/rnnlm-example.h"
#include "nnet3/natural-gradient-online.h"
#include "rnnlm/rnnlm-example-utils.h"

namespace kaldi {
namespace rnnlm {


// These are options relating to the core RNNLM training,
// i.e. training the actual neural net that is for the RNNLM
// (when the word embeddings are given).  This is analogous
// to NnetTrainerOptions in ../nnet-training.h, except that
// with the RNNLM the training code a few things are different,
// so we're using a totally separate training class.
// We'll add options as we need them.

struct RnnlmEmbeddingTrainerOptions {
  int32 print_interval;
  BaseFloat momentum;
  BaseFloat max_param_change;
  BaseFloat l2_regularize;
  BaseFloat learning_rate;  // Note: don't set the learning rate to 0.0 if you
                            // don't want to train this; instead, you can turn
                            // off training of the embedding matrix by
                            // controlling the command line options to the
                            // training program (e.g. not providing a place to
                            // write the embedding matrix).
  BaseFloat backstitch_training_scale;
  int32 backstitch_training_interval;

  // Natural-gradient related options
  bool use_natural_gradient;
  BaseFloat natural_gradient_alpha;
  int32 natural_gradient_rank;
  int32 natural_gradient_update_period;
  int32 natural_gradient_num_minibatches_history;

  RnnlmEmbeddingTrainerOptions():
      print_interval(100),
      momentum(0.0),
      max_param_change(1.0),
      l2_regularize(0.0),
      learning_rate(0.01),
      backstitch_training_scale(0.0),
      backstitch_training_interval(1),
      use_natural_gradient(true),
      natural_gradient_alpha(4.0),
      natural_gradient_rank(80),
      natural_gradient_update_period(4),
      natural_gradient_num_minibatches_history(10) { }

  void Register(OptionsItf *opts) {
    opts->Register("momentum", &momentum, "Momentum constant to apply during "
                   "training of embedding (e.g. 0.5 or 0.9).  Note: we "
                   "automatically multiply the learning rate by (1-momenum) "
                   "so that the 'effective' learning rate is the same as "
                   "before (because momentum would normally increase the "
                   "effective learning rate by 1/(1-momentum))");
    opts->Register("max-param-change", &max_param_change, "The maximum change in "
                   "parameters allowed per minibatch, measured in Euclidean norm, "
                   "for the embedding matrix (the matrix of num-features by "
                   "embedding-dim -- or num-words by embedding-dim, if we're not "
                   "using a feature-based representation.");
    opts->Register("l2-regularize", &l2_regularize, "L2 regularize value that "
                   "affects the strength of l2 regularization on embedding "
                   "parameters.");
    opts->Register("learning-rate", &learning_rate, "The learning rate used in "
                   "training the word-embedding matrix.");
    opts->Register("backstitch-training-scale", &backstitch_training_scale,
                   "backstitch training factor. "
                   "if 0 then in the normal training mode. It is referred to as "
                   "'\\alpha' in our publications.");
    opts->Register("backstitch-training-interval",
                   &backstitch_training_interval,
                   "do backstitch training with the specified interval of "
                   "minibatches. It is referred to as 'n' in our publications.");
    opts->Register("use-natural-gradient", &use_natural_gradient,
                   "True if you want to use natural gradient to update the "
                   "embedding matrix");
    opts->Register("natural-gradient-alpha", &natural_gradient_alpha,
                   "Smoothing constant alpha to use for natural gradient when "
                   "updating the embedding matrix");
    opts->Register("natural-gradient-rank", &natural_gradient_rank,
                   "Rank of the Fisher matrix in natural gradient as applied to "
                   "learning the embedding matrix (this is in the embedding "
                   "space, so the rank should probably be less than the "
                   "embedding dimension");
    opts->Register("natural-gradient-update-period",
                   &natural_gradient_update_period,
                   "Determines how often the Fisher matrix is updated for natural "
                   "gradient as applied to the embedding matrix");
    opts->Register("natural-gradient-num-minibatches-history",
                   &natural_gradient_num_minibatches_history,
                   "Determines how quickly the Fisher estimate for the natural gradient "
                   "is updated, when training the word embedding.");
  }
  void Check() const;
};


/** This class is responsible for training the word embedding matrix or
    feature embedding matrix.
 */
class RnnlmEmbeddingTrainer {
 public:
  /** Constructor.
       @param [in] config  Structure that holds configuration options;
                          this class will keep a reference to it.
       @param [in] embedding_mat   The embedding matrix to be trained,
                          of dimension (num-words or num-features) by
                          embedding-dim (depending whether we are using a
                          feature representation of words, or not).  This class
                          keeps the pointer and will modify that variable.
   */
  RnnlmEmbeddingTrainer(const RnnlmEmbeddingTrainerOptions &config,
                        CuMatrix<BaseFloat> *embedding_mat);

  /* Train on one minibatch-- this version is used either when there is no
     subsampling, or when there is subsampling but we are using a feature
     representation so the subsampling is handled outside of this code.

      @param [in] embedding_deriv  The derivative w.r.t. the (word or feature)
                     embedding matrix; it's provided as a non-const pointer for
                     convenience so that we can modify it in-place if needed
                     for the natural gradient update.
  */
  void Train(CuMatrixBase<BaseFloat> *embedding_deriv);

  // The backstitch version of the above function. Depending
  // on whether is_backstitch_step1 is true, It could be either the first
  // (backward) step, or the second (forward) step of backstitch.
  void TrainBackstitch(bool is_backstitch_step1,
                       CuMatrixBase<BaseFloat> *embedding_deriv);


  /* Train on one minibatch-- this version is for when there is subsampling, and
     the user is providing the derivative w.r.t. just the word-indexes that were
     used in this minibatch.  'active_words' is a sorted, unique list of the
     word-indexes that were used in this minibatch, and 'word_embedding_deriv'
     is the derivative w.r.t. the embedding of that list of words.

      @param [in] active_words  A sorted, unique list of the word indexes
                      used, with Dim() equal to word_embedding_deriv->NumRows();
                      contains indexes 0 <= i < embedding_deriv_->NumRows().

      @param [in] word_embedding_deriv   The derivative w.r.t. the
                      word embedding matrix; it's provided as a non-const
                      pointer for convenience so that we can modify
                      it in-place if needed for the natural gradient
                      update.
  */
  void Train(const CuArrayBase<int32> &active_words,
             CuMatrixBase<BaseFloat> *word_embedding_deriv);

  // The backstitch version of the above function.
  void TrainBackstitch(bool is_backstitch_step1,
                       const CuArrayBase<int32> &active_words,
                       CuMatrixBase<BaseFloat> *word_embedding_deriv);

  ~RnnlmEmbeddingTrainer();


 private:

  // Sets options in the object 'preconditioner_', based on the config
  // (but not SetNumSamplesHistory(), we do that in the Train() functions because
  /// we don't have the right information at this point).
  void SetNaturalGradientOptions();

  // Called from the destructor, this prints some stats about how often the
  // max-change constraint was applied, how much data we trained on, and how
  // much the parameters changed during the lifetime of this object.
  // TODO: implement this.
  void PrintStats();


  const RnnlmEmbeddingTrainerOptions &config_;

  // Object that takes care of the natural-gradient update (this is in the
  // dimension of space equal to the embedding dim, which is the num-cols
  // of embedding_mat_.
  nnet3::OnlineNaturalGradient preconditioner_;

  // The matrix we are updating
  CuMatrix<BaseFloat> *embedding_mat_;


  // If momentum is to be used, this is sized to the same size as
  // *embedding_mat*, and used for the decaying sum of deltas.
  CuMatrix<BaseFloat> embedding_mat_momentum_;

  // This is a copy of the 'embedding_mat' that we were initialized with,
  // which we keep around for purposes of printing stats at the end about how
  // much the matrix changed; we keep it in CPU memory in case GPU memory is a
  // limiting factor.
  Matrix<BaseFloat> initial_embedding_mat_;

  // A count of the number of times we have updated the matrix.
  int32 num_minibatches_;

  // A count of the number of times the max-change constraint was applied.
  int32 max_change_count_;
};





} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_EMBEDDING_TRAINING_H_

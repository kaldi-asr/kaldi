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

#ifndef KALDI_RNNLM_RNNLM_CORE_TRAINING_H_
#define KALDI_RNNLM_RNNLM_CORE_TRAINING_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "rnnlm/rnnlm-example.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
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
  BaseFloat learning_rate;

  // Natural-gradient related options
  bool use_natural_gradient;
  BaseFloat natural_gradient_alpha;
  int32 natural_gradient_rank;
  int32 natural_gradient_update_period;

  RnnlmEmbeddingTrainerOptions():
      print_interval(100),
      momentum(0.0),
      learning_rate(0.01),
      max_param_change(1.0),
      use_natural_gradient(true),
      natural_gradient_alpha(4.0),
      natural_gradient_rank(80),
      natural_gradient_update_period(4) { }

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
    opts->Register("learning-rate", &learning_rate, "The learning rate used in "
                   "training the word-embedding matrix.");
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
  }
};


/** This class is responsible for training the word embedding matrix; it's to be
    used when you have only a (dense) word embedding matrix and no sparse
    feature representation of words.
 */
class RnnlmWordEmbeddingTrainer {
 public:
  /** Constructor: this version is to be used when we are training the
      word embedding directly without a sparse feature representation.

       @param [in] config  Structure that holds configuration options
       @param [in,out] embedding_mat   The embedding matrix to be trained,
                          of dimension num-words by embedding-dim.

       neural network that is to be trained.
                              Will be modified each time you call Train().
   */

  RnnlmEmbeddingTrainer(RnnlmCoreTrainerOptions &config,
                        CuMatrix<BaseFloat> *embedding_mat);

  /* Train on one minibatch.
       @param [in] minibatch  The RNNLM minibatch to train on, containing
                            a number of parallel word sequences.  It will not
                            necessarily contain words with the 'original'
                            numbering, it will in most circumstances contain
                            just the ones we used; see RenumberRnnlmMinibatch().
       @param [in] word_embedding  The matrix giving the embedding of words;
                            the row index is the word-id and the number of columns
                            is the word-embedding dimension.  The numbering
                            of the words does not have to be the 'real'
                            numbering of words, it can consist of words renumbered
                            by RenumberRnnlmMinibatch(); it just has to be
                            consistent with the word-ids present in 'minibatch'.
       @param [out] word_embedding_deriv  If supplied, the derivative of the
                            objective function w.r.t. the word embedding will be
                            *added* to this location; it must have the same
                            dimension as 'word_embedding'.
   */
  void Train(const RnnlmExample &minibatch,
             const CuMatrixBase<BaseFloat> &word_embedding,
             CuMatrixBase<BaseFloat> *word_embedding_deriv = NULL);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;

  ~RnnlmCoreTrainer();
 private:

  void ProvideInput(const RnnlmExample &minibatch,
                    const RnnlmExampleDerived &derived,
                    const CuMatrixBase<BaseFloat> &word_embedding,
                    nnet3::NnetComputer *computer);

  void ProcessOutput(const RnnlmExample &minibatch,
                     const RnnlmExampleDerived &derived,
                     const CuMatrixBase<BaseFloat> &word_embedding,
                     nnet3::NnetComputer *computer,
                     CuMatrixBase<BaseFloat> *word_embedding_deriv = NULL);

  // Applies per-component max-change and global max-change to all updatable
  // components in *delta_nnet_, and use *delta_nnet_ to update parameters
  // in *nnet_.
  void UpdateParamsWithMaxChange();

  const RnnlmCoreTrainerOptions config_;
  nnet3::Nnet *nnet_;
  nnet3::Nnet *delta_nnet_;  // Only used if momentum != 0.0 or max-param-change !=
                             // 0.0.  nnet representing accumulated parameter-change
                             // (we'd call this gradient_nnet_, but due to
                             // natural-gradient update, it's better to consider it as
                             // a delta-parameter nnet.
  nnet3::CachingOptimizingCompiler compiler_;

  int32 num_minibatches_processed_;

  // stats for max-change.
  std::vector<int32> num_max_change_per_component_applied_;
  int32 num_max_change_global_applied_;

  ObjectiveTracker objf_info_;
};






} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_CORE_TRAINING_H_

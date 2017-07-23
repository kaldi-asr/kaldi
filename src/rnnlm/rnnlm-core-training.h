// rnnlm/rnnlm-core-training.h

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

#ifndef KALDI_RNNLM_RNNLM_CORE_TRAIN_H_
#define KALDI_RNNLM_RNNLM_CORE_TRAIN_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "rnnlm/rnnlm-example.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace rnnlm {


// These are options relating to the core RNNLM training,
// i.e. training the actual neural net that is for the RNNLM
// (when the word embeddings are given).  This is analogous
// to NnetTrainerOptions in ../nnet-training.h, except that
// with the RNNLM the training code a few things are different,
// so we're using a totally separate training class.
// We'll add options as we need them.
class RnnlmCoreTrainerOptions {
  int32 print_interval;
  BaseFloat momentum;
  BaseFloat max_param_change;

  RnnlmCoreTrainerOptions():
      print_interval(100),
      momentum(0.0),
      max_param_change(2.0) { }

  void Register(OptionsItf *opts) {
    opts->Register("momentum", &momentum, "Momentum constant to apply during "
                   "training (help stabilize update).  e.g. 0.9.  Note: we "
                   "automatically multiply the learning rate by (1-momenum) "
                   "so that the 'effective' learning rate is the same as "
                   "before (because momentum would normally increase the "
                   "effective learning rate by 1/(1-momentum))");
    opts->Register("max-param-change", &max_param_change, "The maximum change in "
                   "parameters allowed per minibatch, measured in Euclidean norm "
                   "over the entire model (change will be clipped to this value)");
  }
};


class ObjectiveTracker {
  ObjectiveTracker(int32 reporting_interval);


  void AddStats(BaseFloat weight, BaseFloat num_objf,
                BaseFloat den_objf,
                BaseFloat exact_den_objf = 0.0);

  ~ObjectiveTracker();  // Prints stats for the final interval, and the overall
                        // stats.


 private:
  // prints the stats for the current interval.
  void PrintStatsThisInterval() const;
  // zeroes the stats for the current interval and adds them to
  // the global stats.
  void CommitIntervalStats();
  // prints the overall stats.
  void PrintStatsOverall() const;

  int32 reporting_interval_;
  int32 num_egs_this_interval_;
  double tot_weight_this_interval_;  // sum of weights of outputs-- the
                                     // objective is to be divided by this.
  double num_objf_this_interval_;  // numerator term in objective
  double den_objf_this_interval_;  // denominator term in objective, to be
                                  // added to numerator term.
  // exact_den_objf_this_interval_ is the exact version of the denominator term,
  // of the form log(sum(...)) instead of sum(...).  This is included for
  // debugging and diagnostic purposes, and it will be zero if we're using
  // sampling (which will be most of the time).
  int32 exact_den_objf_this_interval_;

  // the following versions of the variables are overall, not just for
  // this interval: once we finish an interval (e.g. 100 examples), we
  // add the '...this_interval_' versions of the variables to the
  // variables below.
  int32 num_egs_;
  double tot_weight_;
  double num_objf_;
  double den_objf_;
  double exact_den_objf_;
};


/** This class does the core part of the training of the RNNLM; the
    word embeddings are supplied to this class for each minibatch and
    while this class can compute objective function derivatives w.r.t.
    these embeddings, it is not responsible for updating them.
 */
class RnnlmCoreTrainer {
 public:
  RnnlmCoreTrainer(RnnlmCoreTrainerOptions &config,
                   nnet3::Nnet *nnet);

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
             CuMatrix<BaseFloat> *word_embedding_deriv = NULL);

  // Prints out the final stats, and return true if there was a nonzero count.
  bool PrintTotalStats() const;

  // Prints out the max-change stats (if nonzero): the percentage of time that
  // per-component max-change and global max-change were enforced.
  void PrintMaxChangeStats() const;

  ~RnnlmCoreTrainer();
 private:

  void ProvideInput(const RnnlmExample &minibatch,
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

  // This code supports multiple output layers, even though in the
  // normal case there will be just one output layer named "output".
  // So we store the objective functions per output layer.
  int32 num_minibatches_processed_;

  // stats for max-change.
  std::vector<int32> num_max_change_per_component_applied_;
  int32 num_max_change_global_applied_;

  ObjectiveFunctionTracker objf_info_;
};






} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_CORE_H_

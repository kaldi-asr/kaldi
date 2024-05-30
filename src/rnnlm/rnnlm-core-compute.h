// rnnlm/rnnlm-core-compute.h

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

#ifndef KALDI_RNNLM_RNNLM_CORE_COMPUTE_H_
#define KALDI_RNNLM_RNNLM_CORE_COMPUTE_H_

#include "rnnlm/rnnlm-example-utils.h"
#include "rnnlm/rnnlm-core-training.h"


namespace kaldi {
namespace rnnlm {


/** This class has a similar interface to RnnlmCoreTrainer, but it doesn't
    actually train the RNNLM; it's for computing likelihoods and (optionally)
    derivatives w.r.t. the embedding, in situations where you are not training
    the core part of the RNNLM.  It reads egs-- it's not for rescoring lattices
    and similar purposes.
 */
class RnnlmCoreComputer {
 public:
  /** Constructor.
       @param [in] nnet   The neural network that is to be used to evaluate
                          likelihoods (and possibly derivatives).
   */
  RnnlmCoreComputer(const nnet3::Nnet &nnet);

  /* Compute the objective on one minibatch (and possibly also derivatives
     w.r.t. the embedding).
       @param [in] minibatch  The RNNLM minibatch to evalute, containing
                            a number of parallel word sequences.  It will not
                            necessarily contain words with the 'original'
                            numbering, it will in most circumstances contain
                            just the ones we used; see RenumberRnnlmMinibatch().
       @param [in] derived   Derived parameters of the minibatch, computed
                            by previously calling GetRnnlmExampleDerived()
                            with suitable arguments.
       @param [in] word_embedding  The matrix giving the embedding of words, of
                            dimension minibatch.vocab_size by the embedding dimension.
                            The numbering of the words does not have to be the 'real'
                            numbering of words, it can consist of words renumbered
                            by RenumberRnnlmMinibatch(); it just has to be
                            consistent with the word-ids present in 'minibatch'.
       @para [out] weight  If non-NULL, the total weight of the words in the
                           minibatch will be written to here (this is just the sum
                           of minibatch.output_weights).
       @param [out] word_embedding_deriv  If supplied, the derivative of the
                            objective function w.r.t. the word embedding will be
                            *added* to this location; it must have the same
                            dimension as 'word_embedding'.
       @return objf      The total objective function for this minibatch; divide
                         this by '*weight' to normalize it (i.e. get the average
                         log-prob per word).
   */
  BaseFloat Compute(const RnnlmExample &minibatch,
                    const RnnlmExampleDerived &derived,
                    const CuMatrixBase<BaseFloat> &word_embedding,
                    BaseFloat *weight = NULL,
                    CuMatrixBase<BaseFloat> *word_embedding_deriv = NULL);

 private:

  void ProvideInput(const RnnlmExample &minibatch,
                    const RnnlmExampleDerived &derived,
                    const CuMatrixBase<BaseFloat> &word_embedding,
                    nnet3::NnetComputer *computer);

  /** Process the output of the neural net and compute the objective function;
      store stats from the objective function in objf_info_.
   @param [in] minibatch  The minibatch for which we're proessing the output.
   @param [in] derived  Derived quantities from the minibatch.
   @param [in] word_embedding  The word embedding, with the same numbering as
                      used in the minibatch (may be subsampled at this point).
   @param [out] word_embedding_deriv  If non-NULL, the part of the derivative
                      w.r.t. the word-embedding that arises from the output
                      computation will be *added* to here.
   @param [out] weight  If non-NULL, this function will output to this location
               the total weight of the output words, which can be used as
               the normalizer for the (returned) objective function.
   @return  Returns the total objective function (of the form:
            \sum_i weight(i) * ( num_term(i) + den_term(i) ), see rnnlm-example-utils.h
            for more information about this.
  */
  BaseFloat ProcessOutput(const RnnlmExample &minibatch,
                          const RnnlmExampleDerived &derived,
                          const CuMatrixBase<BaseFloat> &word_embedding,
                          nnet3::NnetComputer *computer,
                          CuMatrixBase<BaseFloat> *word_embedding_deriv,
                          BaseFloat *weight);

  const nnet3::Nnet &nnet_;

  nnet3::CachingOptimizingCompiler compiler_;

  int32 num_minibatches_processed_;

  ObjectiveTracker objf_info_;
};






} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_CORE_COMPUTE_H_

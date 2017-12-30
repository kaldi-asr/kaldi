// rnnlm/rnnlm-example-utils.h

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

#ifndef KALDI_RNNLM_RNNLM_EXAMPLE_UTILS_H_
#define KALDI_RNNLM_RNNLM_EXAMPLE_UTILS_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "nnet3/nnet-computation.h"
#include "rnnlm/sampling-lm.h"
#include "rnnlm/sampler.h"
#include "rnnlm/rnnlm-example.h"


namespace kaldi {
namespace rnnlm {

/**
   @file rnnlm-example-utils.h

   @brief  Declares various functions that operate on examples
           for RNNLM training (specifically, class RnnlmExample).
*/




/**
   This function renumbers the word-ids referred to in a minibatch, creating a
   numbering that covers exactly the words referred to in this minibatch.   It
   is only to be called when sampling is used, i.e. when minibatch->samples
   is not empty.

      @param [in,out] minibatch  The minibatch to be modified.
                              At entry the words-indexes in fields
                             'input_words', and 'sampled_words' will be in their
                             canonical numbering.  At exit the numbers present
                             in those arrays will be indexes into the
                             'active_words' vector that this function outputs.
                             For instance, suppose minibatch->input_words[9] ==
                             1034 at entry; at exit we might have
                             minibatch->input_words[9] == 94, with
                             (*active_words)[94] == 1034.  This function
                             requires that minibatch->sampled_words must be
                             nonempty.  If minibatch->sampled_words is empty, it
                             means that sampling has not been done, so the
                             negative part of the objf will use all the words.
                             In this case the minibatch implicitly uses all
                             words, so there is no use in renumbering.  At exit,
                             'minibatch->vocab_size' will have been set to the
                             same value as active_words->size().  Note: it is not
                             necessary for this function to renumber 'output_words'
                             because in the sampling case they are indexes into
                             blocks of 'sampled_words' (see documentation for
                             RnnlmExample).
      @param [out] active_words  The list of active words, i.e. the words that
                              were present in the fields 'input_words',
                              and 'sampled_words' in 'minibatch' on entry.  At
                              exit, this list will be sorted and unique.
 */
void RenumberRnnlmExample(RnnlmExample *minibatch,
                          std::vector<int32> *active_words);


/**  This function takes a NnetExample (which should already have been
     frame-selected, if desired, and merged into a minibatch) and produces a
     ComputationRequest.  It assumes you don't want the derivatives w.r.t. the
     inputs; if you do, you can create/modify the ComputationRequest manually.
     Assumes that if need_model_derivative is true, you will be supplying
     derivatives w.r.t. all outputs.
*/
void GetRnnlmComputationRequest(const RnnlmExample &minibatch,
                                bool need_model_derivative,
                                bool need_input_derivative,
                                bool store_component_stats,
                                nnet3::ComputationRequest *computation_request);


// This struct contains various quantities/expressions that are derived from the
// quantities found in RnnlmExample, and which are needed when training on that
// example, particularly by the function ProcessRnnlmOutput().  The reason to
// make this a struct, instead of making these things local variables inside
// function ProcessRnnlmOutput(), is so that we can re-use things in case they
// are needed multiple times, and so that we can compute these derived
// quantities in a separate thread (although this separate-thread thing isn't
// implemented yet.
struct RnnlmExampleDerived {
  CuArray<int32> cu_input_words;  // CUDA copy of minibatch.input_words.

  CuArray<int32> cu_output_words;  // CUDA copy of minibatch.output_words,
                                   // only used in the sampling case.

  // cu_sampled_words is a CUDA copy of minibatch.sampled_words; it's only used
  // in the sampling case (in the no-sampling case, minibatch.sampled_words
  // would be empty anyway).
  CuArray<int32> cu_sampled_words;

  // output_words_smat is  only used in the no-sampling case;
  // it is a CuSparseMatrix constructed from 'vocab_size', 'output_words' and
  // 'output_weights' of the RnnlmExample, which will be a sparse matrix with
  // num-rows equal to the RnnlmExample's 'output_words.size()' and num-cols
  // equal to its 'vocab_size'.
  CuSparseMatrix<BaseFloat> output_words_smat;

  // input_words_smat is a SparseMatrix used in backpropagating the
  // derivatives w.r.t. the input words back to the word-embedding.
  // It is of dimension minibatch.vocab_size by minibatch.input_words.size(),
  // and is all zeros except that it contains ones in positions
  // (minibatch.input_words[i],i).
  CuSparseMatrix<BaseFloat> input_words_smat;


  // Shallow swap; calls Swap() on all elements.
  void Swap(RnnlmExampleDerived *other);
};

/**
   Set up the structure containing derived parameters used in training and
   objective function computation.
      @param [in] minibatch  The input minibatch for which we are computing
                             the derived parameters.
      @param [in] need_embedding_deriv   True if we are going to be
                             computing derivatives w.r.t. the word embedding
                             (e.g., needed in a typical training configuration);
                             if this is true, it will compute
                             'input_words_tranpose'.
      @param [out] derived   The output structure that we are computing.
*/
void GetRnnlmExampleDerived(const RnnlmExample &minibatch,
                            bool need_embedding_deriv,
                            RnnlmExampleDerived *derived);

/**
   Configuration class relating to the objective function used for RNNLM
   training, more specifically for use by the function ProcessRnnlmOutputs().
 */
struct RnnlmObjectiveOptions {
  BaseFloat den_term_limit;
  uint32 max_logprob_elements;

  RnnlmObjectiveOptions(): den_term_limit(-10.0),
                           max_logprob_elements(1000000000) { }

  void Register(OptionsItf *po) {
    po->Register("den-term-limit", &den_term_limit,
                 "Modification to the with-sampling objective, that prevents "
                 "instability early in training, but in the end makes no difference. "
                 "We scale down the denominator part of the objective when the "
                 "average denominator part of the objective, for this minibatch, "
                 "is more negative than this value.  Set this to 0.0 to use "
                 "unmodified objective function.");
    po->Register("max-logprob-elements", &max_logprob_elements,
                 "Maximum number of elements when we allocate a matrix of size "
                 "[minibatch-size, num-words] for computing logprobs of words. "
                 "If the size is exceeded, we will break the matrix along the "
                 "minibatch axis and compute them separately");
  }
};

/**
     This function processes the output of the RNNLM computation for a single
     minibatch; it outputs the objective-function contributions from the
     'numerator' and 'denominator' terms, and [if requested] the derivatives of
     the objective function w.r.t. the data inputs.

     In the explanation below, the index 'i' encompasses both the time 't'
     and the member 'n' within the minibatch.
      The 'objective function' referred to here is of the form:
        objf = \sum_i weight(i) * ( num_term(i) + den_term(i) )
      where num_term(i) is the log-prob of the 'correct' word, which equals
      the dot product of the neural-network output with the word embedding,
      which we can write as follows:
         num_term(i) = l(i, minibatch.output_words(i))
      where l(i, w) is the unnormalized log-prob of word w for position i,
      specifically:
         l(i, w) = VecVec(nnet_output.Row(i), word_embedding.Row(w)).

      Without importance sampling (if minibatch.sampled_words.empty()):
          den_term(i) = 1.0 - (\sum_w q(i,w))

      This is a lower bound on the 'natural' normalizer term which is of the
      form -log(\sum_w p(i,w)), and its linearity in the p's allows importance
      sampling).  Here,
           p(i, w) = Exp(l(i, w))
           q(i, w) = Exp(l(i, w)) if l(i, w < 0) else  1 + l(i, w)
     [the reason we use q(i, w) instead of p(i, w) is that it gives a
      closer bound to the natural normalizer term and helps avoid
      instability in early phases of training.]

      With importance sampling (if minibatch.sampled_words.size() > 0):
      'den_term' equals
         den_term(i) =  1.0 - (\sum_w q(w,i) * sample_inv_prob(w,i))
      where sample_inv_prob(w, i) is zero if word w was not sampled
      for this 't', and 1.0 / (the probability with which it was sampled)
      if it was sampled.


       @param [in] minibatch  The minibatch for which we are processing the
                         output.
       @param [in] minibatch-derived  This struct contains certain quantities
                         which are precomputed from 'minibatch'; it's to be
                         generated by calling GenerateRnnlmExampleDerived()
                         prior to calling this function.
       @param [in] word_embedding  The word-embedding, dimension is num-words
                         by embedding-dimension.  This does not have to
                         be 'real' word-indexes, it can be fake word-indexes
                         renumbered to include only the required words if
                         sampling is done; c.f. RenumberRnnlmExample().
       @param [in] nnet_output  The neural net output.  Num-rows is
                         minibatch.chunk_length * minibatch.num_chunks,
                         where the stride for the time 0 <= t < chunk_length
                         is larger, so there are a block of rows for t=0,
                         a block for t=1, and so on.  Num-columns is
                         embedding-dimension.
       @param [out] word_embedding_deriv  If non-NULL, the derivative of the
                         objective function w.r.t. 'word_embedding' is *added*
                         to this location.
       @param [out] nnet_output_dirv  If non-NULL, the derivative of the
                         objective function w.r.t. 'nnet_output' is *added*
                         to this location.
       @param [out] weight  Must be non-NULL.  The total weight over this
                         minibatch will be *written to* here (will equal
                         minibatch.output_weights.Sum()).
       @param [out] objf_num  If non-NULL, the total numerator part of
                         the objective function will be written here, i.e.
                         the sum over i of weight(i) * num_term(i); see above
                         for details.
       @param [out] objf_den  Must be non-NULL.  The total denominator part of
                         the objective function will be written here, i.e.
                         the sum over i of weight(i) * den_term(i); see above
                         for details.  You add this to 'objf_num' to get the
                         total objective function.
       @param [out] objf_den_exact  If non-NULL, and if we're not
                         doing sampling (i.e. if minibatch.sampled_words.empty()),
                         the 'exact' denominator part of the objective function
                         will be written here, i.e. the weighted sum of
                           exact_den_term(i) = -log(\sum_w p(i,w)).
                         If we are sampling, then there is no exact denominator
                         part, and this will be set to zero.  This is provided
                         for diagnostic purposes.  Derivatives will be computed
                         w.r.t. the objective consisting of
                         'objf_num + objf_den', i.e. ignoring the 'exact' one.
                         For greatest efficiency you should probably not provide
                         this pointer.
 */
void ProcessRnnlmOutput(
    const RnnlmObjectiveOptions &objective_opts,
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    const CuMatrixBase<BaseFloat> &nnet_output,
    CuMatrixBase<BaseFloat> *word_embedding_deriv,
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    BaseFloat *weight,
    BaseFloat *objf_num,
    BaseFloat *objf_den,
    BaseFloat *objf_den_exact);




} // namespace rnnlm
} // namespace kaldi

#endif // KALDI_RNNLM_RNNLM_EGS_H_

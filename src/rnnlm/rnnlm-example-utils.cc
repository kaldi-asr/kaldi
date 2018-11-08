// rnnlm/rnnlm-example-utils.cc

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

#include <numeric>
#include "rnnlm/rnnlm-example-utils.h"

namespace kaldi {
namespace rnnlm {

void GetRnnlmComputationRequest(
    const RnnlmExample &minibatch,
    bool need_model_derivative,
    bool need_input_derivative,
    bool store_component_stats,
    nnet3::ComputationRequest *request) {

  request->inputs.clear();
  request->inputs.resize(1);
  request->outputs.clear();
  request->outputs.resize(1);
  request->need_model_derivative = need_model_derivative;
  request->store_component_stats = store_component_stats;

  nnet3::IoSpecification &input_spec = request->inputs[0],
      &output_spec = request->outputs[0];
  input_spec.name = "input";
  output_spec.name = "output";

  int32 num_chunks = minibatch.num_chunks,
      chunk_length = minibatch.chunk_length;
  input_spec.indexes.resize(num_chunks * chunk_length);
  KALDI_ASSERT(num_chunks > 0 && chunk_length > 0);
  nnet3::Index *cur_index = &(input_spec.indexes[0]);
  for (int32 t = 0; t < chunk_length; t++) {
    for (int32 n = 0; n < num_chunks; n++) {
      cur_index->t = t;
      cur_index->n = n;
      cur_index++;
    }
  }
  output_spec.indexes = input_spec.indexes;
  output_spec.has_deriv = (need_model_derivative ||
                           need_input_derivative);
  input_spec.has_deriv = need_input_derivative;
}


void RnnlmExampleDerived::Swap(RnnlmExampleDerived *other) {
  cu_input_words.Swap(&other->cu_input_words);
  cu_output_words.Swap(&other->cu_output_words);
  cu_sampled_words.Swap(&other->cu_sampled_words);
  output_words_smat.Swap(&other->output_words_smat);
  input_words_smat.Swap(&other->input_words_smat);
}

// This is called from ProcessRnnlmOutput() when we are doing importance
// sampling.
static void ProcessRnnlmOutputSampling(
    const RnnlmObjectiveOptions &objective_config,
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    const CuMatrixBase<BaseFloat> &nnet_output,
    CuMatrixBase<BaseFloat> *word_embedding_deriv,
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    BaseFloat *weight,
    BaseFloat *objf_num,
    BaseFloat *objf_den,
    BaseFloat *objf_den_exact) {
  KALDI_ASSERT(weight != NULL && objf_den != NULL);  // Others are optional.

  // In the case where minibatch.sample_group_size == 1, meaning for each 't' value we
  // sample separately, num_sample_groups would equal the chunk_length and
  // rows_per_sample would equal minibatch.num_chunks.  When trying to
  // understand this code, initially assume sample_group_size == 1.
  // For sample_group_size > 1, it means that for we use the same group
  // of sampled words for a number of time steps.
  int32 num_sample_groups = minibatch.chunk_length / minibatch.sample_group_size,
      rows_per_group = minibatch.num_chunks * minibatch.sample_group_size,
      samples_per_group = minibatch.num_samples,
      embedding_dim = word_embedding.NumCols();
  KALDI_ASSERT(nnet_output.NumRows() == num_sample_groups * rows_per_group);

  CuMatrix<BaseFloat> word_logprobs(rows_per_group,
                                    samples_per_group);
  // 'sampled_word_embedding' will contain those rows of 'word_embedding' that
  // pertain to the words sampled in this group.
  CuMatrix<BaseFloat> sampled_word_embedding(samples_per_group,
                                             embedding_dim,
                                             kUndefined);

  CuVector<BaseFloat> output_word_logprobs(rows_per_group * num_sample_groups,
                                           kUndefined);

  if (weight) *weight = minibatch.output_weights.Sum();
  if (objf_den) *objf_den = 0.0;
  // note: we don't update objf_den_exact in the loop, it's not applicable in
  // the sampling case.
  if (objf_den_exact) *objf_den_exact = 0.0;

  // caution: the 't' here is only the real 't' value if sample_group_size == 1.
  for (int32 t = 0; t < num_sample_groups; t++) {

    CuSubArray<int32> sampled_words_part(derived.cu_sampled_words,
                                         t * samples_per_group,
                                         samples_per_group),
        output_words_part(derived.cu_output_words,
                          t * rows_per_group,
                          rows_per_group);
    CuSubVector<BaseFloat> output_weights_part(minibatch.output_weights,
                                               t * rows_per_group,
                                               rows_per_group),
        sample_inv_probs_part(minibatch.sample_inv_probs,
                              t * samples_per_group,
                              samples_per_group);

    sampled_word_embedding.CopyRows(word_embedding,
                                    sampled_words_part);

    CuSubMatrix<BaseFloat> nnet_output_part(nnet_output,
                                            rows_per_group * t, rows_per_group,
                                            0, nnet_output.NumCols());
    word_logprobs.AddMatMat(1.0, nnet_output_part, kNoTrans,
                            sampled_word_embedding, kTrans, 0.0);


    // OK: now we have the log-probs for this group of words we sampled.
    // Get the logprobs of the correct words.
    if (objf_num != NULL) {
      CuSubVector<BaseFloat> this_output_word_logprobs(
          output_word_logprobs,
          t * rows_per_group, rows_per_group);
      this_output_word_logprobs.CopyElements(word_logprobs, kNoTrans,
                                             output_words_part);
    }


    // In preparation for computing the denominator objf contribution, change 'word_logprobs'
    // to contain q = (l < 0 ? exp(l) : l + 1.0), instead of the unnormalized
    // logprob l.  For most purposes you can think of this as behaving like
    // exp(l); it just has better behavior at the beginning of training when there
    // is a danger of divergence.  We can prove that the objective is a closer
    // bound on the true (log-probability-based) objective than if we used a
    // simple exp.  note: all this bound stuff is really geared towards the
    // sampling case, there is really no point in it if we're not doing sampling
    // and some of these code paths will only be used in test code.
    word_logprobs.ApplyExpSpecial();

    // The denominator part of this objective is something like:
    // - \sum_i \sum_w output_weight(i) * q(i, w) * sample_inv_prob(w).
    *objf_den +=  -VecMatVec(output_weights_part, word_logprobs,
                             sample_inv_probs_part);


    // The derivative of the function q(l) = (l < 0 ? exp(l) : l + 1.0)
    // equals (l < 0 ? exp(l) : 1.0), which we can compute by
    // applying a ceiling to q at 1.0.
    word_logprobs.ApplyCeiling(1.0);

    // the inverses of the sampling probabilities appear as a factor
    // in the deriviative of the objf w.r.t. the words' logprobs
    // (which is what we're computing now).
    word_logprobs.MulColsVec(sample_inv_probs_part);

    if (objective_config.den_term_limit != 0.0) {
      // If it's nonzero then check that it's negative, and not too close to zero,
      // which would likely cause failure to converge.  The default is 10.0.
      KALDI_ASSERT(objective_config.den_term_limit < -0.5);
      BaseFloat limit = objective_config.den_term_limit;
      if (weight != NULL && objf_den != NULL && *weight > 0 &&
          (*objf_den / *weight) < limit) {
        // note: both things being divided below are negative, and
        // 'scale' will be between zero and one.
        BaseFloat scale = limit / (*objf_den / *weight);
        // We scale the denominator part of the objective down by the inverse of
        // the factor by which the denominator part of the objective exceeds the
        // limit.  This point in the code should only be reached on the first few
        // iterations of training, or if there is some kind of instability,
        // because the (*objf_den / *weight) will usually be close to zero,
        // e.g. -0.01, while 'limit' is expected to be larger, like -10.0.
        word_logprobs.Scale(scale);
      }
    }


    // This adds -1.0 to the elements of 'word_logprobs' corresponding
    // to the output words.  This array 'word_logprobs' is going to
    // represent the negative of the derivative of the objf w.r.t.
    // the original 'word_logprobs' array.  The negative sign just
    // helps us avoid one operation.
    word_logprobs.AddToElements(-1.0, output_words_part);

    // The factor from 'output_weights' applies to both the denominator and
    // numerator terms in the derivative, so we waited to multiply by it until
    // after we'd included the numerator-related term.
    word_logprobs.MulRowsVec(output_weights_part);

    // OK, at this point, word_logprobs contains the negative of the derivative
    // of the objf w.r.t. the original 'word_logprobs' array.
    // The two if-blocks below are doing the backprop for the
    // statement:
    // word_logprobs.AddMat(1.0, nnet_output_part, kNoTrans,
    //                      sampled_word_embedding, kTrans, 0.0);
    if (nnet_output_deriv) {
      CuSubMatrix<BaseFloat> nnet_output_deriv_part(
          *nnet_output_deriv, rows_per_group * t, rows_per_group,
          0, nnet_output.NumCols());
      nnet_output_deriv_part.AddMatMat(-1.0, word_logprobs, kNoTrans,
                                       sampled_word_embedding, kNoTrans, 1.0);
    }
    if (word_embedding_deriv) {
      // We'll temporarily use 'sampled_word_embedding' to contain
      // the derivative of the objf w.r.t 'sampled_word_embedding', and
      // propagate it from there back to 'word_embedding_deriv'.
      sampled_word_embedding.AddMatMat(-1.0, word_logprobs, kTrans,
                                       nnet_output_part, kNoTrans, 0.0);
      sampled_word_embedding.AddToRows(1.0, sampled_words_part,
                                       word_embedding_deriv);

    }
  }
  if (objf_num)
    *objf_num = VecVec(output_word_logprobs, minibatch.output_weights);
  if (objf_den)
    *objf_den += minibatch.output_weights.Sum();
}

// This is called from ProcessRnnlmOutput() when we are not doing importance
// sampling.
static void ProcessRnnlmOutputNoSampling(
    const RnnlmObjectiveOptions &objective_config,
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    const CuMatrixBase<BaseFloat> &nnet_output,
    CuMatrixBase<BaseFloat> *word_embedding_deriv,
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    BaseFloat *weight,
    BaseFloat *objf_num,
    BaseFloat *objf_den,
    BaseFloat *objf_den_exact) {
  KALDI_ASSERT(weight != NULL && objf_den != NULL);  // Others are optional.

  int32 embedding_dim = word_embedding.NumCols();
  int32 num_words = word_embedding.NumRows();



  // 'word_logprobs' contains the unnormalized logprobs of the words.
  CuMatrix<BaseFloat> word_logprobs(nnet_output.NumRows(),
                                    num_words);
  word_logprobs.AddMatMat(1.0, nnet_output, kNoTrans,
                          word_embedding, kTrans, 0.0);

  *weight = minibatch.output_weights.Sum();

  if (objf_num) {
    *objf_num = TraceMatSmat(word_logprobs,
                             derived.output_words_smat, kTrans);
  }
  if (objf_den_exact) {
    // This code is not as optimized as it could be, but we don't take this
    // branch in training so efficiency doesn't matter as much.
    // The 'exact' denominator is the log of the sum of the
    // words' probs.

    // the -1 is to remove epsilon.
    CuMatrix<BaseFloat> word_probs(nnet_output.NumRows(),
                                   num_words - 1, kUndefined);
    word_probs.CopyFromMat(word_logprobs.ColRange(1, num_words - 1));
    word_probs.ApplyExpLimited(-80.0, 80.0);
    CuVector<BaseFloat> row_sums(nnet_output.NumRows());
    row_sums.AddColSumMat(1.0, word_probs, 0.0);
    row_sums.ApplyLog();
    BaseFloat ans = -VecVec(row_sums, minibatch.output_weights);
    *objf_den_exact =  ans;
    if (fabs(ans) > 1.0 * nnet_output.NumRows()) {
      KALDI_WARN << "Big den objf "  << ans;
    }
  }

  // In preparation for computing the denominator objf, change 'word_logprobs'
  // to contain q = (l < 0 ? exp(l) : l + 1.0), instead of the unnormalized
  // logprob l.  For most purposes you can think of this as behaving like
  // exp(l); it just has better behavior at the beginning of training when there
  // is a danger of divergence.  We can prove that the objective is a closer
  // bound on the true (log-probability-based) objective than if we used a
  // simple exp.  note: all this bound stuff is really geared towards the
  // sampling case, there is really no point in it if we're not doing sampling
  // and some of these code paths will only be used in test code.
  word_logprobs.ApplyExpSpecial();

  { // This block computes *objf_den.

    // we call this variable 'q_noeps' because in the math described in
    // rnnlm-example-utils.h it is described as q(i,w), and because we're
    // skipping over the epsilon symbol (which we don't want to include in the
    // sum because it can never be output) by removing the first row.
    CuSubMatrix<BaseFloat> q_noeps(word_logprobs,
                                   0, word_logprobs.NumRows(),
                                   1, num_words - 1);
    // den_term(i) = 1.0 - (\sum_w q(i,w))
    CuVector<BaseFloat> den_term(word_logprobs.NumRows(), kUndefined);
    den_term.Set(1.0);
    den_term.AddColSumMat(-1.0, q_noeps, 1.0);
    // note: objf = \sum_i weight(i) * ( num_term(i) + den_term(i) ),
    // this is the term \sum_i weight(i) * den_term(i).
    *objf_den = VecVec(den_term, minibatch.output_weights);
  }

  // The rest of this function computes the derivative w.r.t.
  // word_embedding_deriv and/or nnet_output_deriv, for which we
  if (!(word_embedding_deriv || nnet_output_deriv))
    return;

  // To avoid one CUDA operation, we're going to make 'word_logprobs'
  // the *negative* of the derivative of the objf w.r.t.
  // the original 'word_logprobs'  Note: don't worry about the
  // fact that we're including the epsilon word at this point;
  // later on we'll ignore the value of the first column.

  // The derivative of the function q(l) = (l < 0 ? exp(l) : l + 1.0)
  // equals (l < 0 ? exp(l) : 1.0), which we can compute by
  // applying a ceiling to q at 1.0.
  word_logprobs.ApplyCeiling(1.0);


  // Include the factor 'minibatch.output_weights'.
  word_logprobs.MulRowsVec(minibatch.output_weights);



  if (objective_config.den_term_limit != 0.0) {
    // If it's nonzero then check that it's negative, and not too close to zero,
    // which would likely cause failure to converge.  The default is 10.0.
    KALDI_ASSERT(objective_config.den_term_limit < -0.5);
    BaseFloat limit = objective_config.den_term_limit;
    if (weight != NULL && objf_den != NULL && *weight > 0 &&
        (*objf_den / *weight) < limit) {
      // note: both things being divided below are negative, and
      // 'scale' will be between zero and one.
      BaseFloat scale = limit / (*objf_den / *weight);
      // We scale the denominator part of the objective down by the inverse of
      // the factor by which the denominator part of the objective exceeds the
      // limit.  This point in the code should only be reached on the first few
      // iterations of training, or if there is some kind of instability,
      // because the (*objf_den / *weight) will usually be close to zero,
      // e.g. -0.01, while 'limit' is expected to be larger, like -10.0.
      word_logprobs.Scale(scale);
    }
  }

  // After the following statement, 'word_logprobs' will contains the negative
  // of the derivative of the objective function w.r.t. l(i, x), except that the
  // first column (for epsilon) should be ignored.

  word_logprobs.AddSmat(-1.0, derived.output_words_smat);

  // l_deriv_noeps is the negative of the derivative of the objective function
  // w.r.t. the unnormalized log-likelihoods 'l' (with the 0th row, for epsilon,
  // not included).
  CuSubMatrix<BaseFloat> l_deriv_noeps(word_logprobs,
                                       0, word_logprobs.NumRows(),
                                       1, num_words - 1);

  // The following statements are doing the backprop w.r.t. the statement:
  //  word_logprobs.AddMatMat(1.0, nnet_output, kNoTrans,
  //                          word_embedding, kTrans, 0.0);
  // (note: 'word_logprobs' in the code corresponds to 'l' in the formulas).
  // The -1.0's are because we have at this point the negative of the
  // derivative w.r.t the unnormalized log-likelihoods.
  if (word_embedding_deriv) {
    CuSubMatrix<BaseFloat> word_embedding_deriv_noeps(
        *word_embedding_deriv, 1, num_words - 1, 0, embedding_dim);
    word_embedding_deriv_noeps.AddMatMat(-1.0, l_deriv_noeps, kTrans,
                                         nnet_output, kNoTrans, 1.0);
  }
  if (nnet_output_deriv) {
    CuSubMatrix<BaseFloat> word_embedding_noeps(
        word_embedding, 1, num_words - 1, 0, embedding_dim);
    nnet_output_deriv->AddMatMat(-1.0, l_deriv_noeps, kNoTrans,
                                 word_embedding_noeps, kNoTrans, 1.0);
  }
}

// This is called from ProcessRnnlmOutput() when we are not doing importance
// sampling, when [minibatch-size, num-word] matrix it too large to allocate on
// GPU memory
static void ProcessRnnlmOutputNoSamplingBatched(
    const RnnlmObjectiveOptions &objective_config,
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    const CuMatrixBase<BaseFloat> &nnet_output,
    CuMatrixBase<BaseFloat> *word_embedding_deriv,
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    BaseFloat *weight,
    BaseFloat *objf_num,
    BaseFloat *objf_den,
    BaseFloat *objf_den_exact) {
  KALDI_ASSERT(weight != NULL && objf_den != NULL);  // Others are optional.

  int32 embedding_dim = word_embedding.NumCols();
  int32 num_words = word_embedding.NumRows();

  int32 batch_size = objective_config.max_logprob_elements / num_words;
  if (batch_size == 0) {
    batch_size = 1;
  }
  if (batch_size > nnet_output.NumRows()) {
    batch_size = nnet_output.NumRows();
  }

  int32 this_start = 0;
  *weight = minibatch.output_weights.Sum();

  if (objf_num) {
    *objf_num = 0.0;
  }
  if (objf_den_exact) {
    *objf_den_exact = 0.0;
  }
  *objf_den = 0.0;

  while (this_start < nnet_output.NumRows()) {
    int32 this_size = std::min(batch_size, nnet_output.NumRows() - this_start);
    // 'word_logprobs' contains the unnormalized logprobs of the words.
    CuMatrix<BaseFloat> word_logprobs(this_size, num_words);
    word_logprobs.AddMatMat(1.0, nnet_output.RowRange(this_start, this_size),
                            kNoTrans, word_embedding, kTrans, 0.0);

    // This is going to be equivalent to taking
    // derived.output_words_smat.RowRange(this_start, this_size)
    // though CuSparseMatrix does not have that interface
    CuSparseMatrix<BaseFloat> this_output_words_smat;
    if (objf_num) {
      std::vector<int32> output_words;
      std::vector<BaseFloat> output_weights;
      for (int32 i = this_start; i < this_start + this_size; i++) {
        output_words.push_back(minibatch.output_words[i]);
      }
      CuArray<int32> cu_output_words(output_words);

      CuSparseMatrix<BaseFloat> smat(cu_output_words,
        CuSubVector<BaseFloat>(minibatch.output_weights, this_start, this_size),
        num_words);
      this_output_words_smat.Swap(&smat);
      *objf_num += TraceMatSmat(word_logprobs,
                                this_output_words_smat, kTrans);
    }

    CuSubVector<BaseFloat> this_output_weights(minibatch.output_weights,
                                              this_start, this_size);
    if (objf_den_exact) {
      // This code is not as optimized as it could be, but we don't take this
      // branch in training so efficiency doesn't matter as much.
      // The 'exact' denominator is the log of the sum of the
      // words' probs.

      // the -1 is to remove epsilon.
      CuMatrix<BaseFloat> word_probs(this_size, num_words - 1, kUndefined);
      word_probs.CopyFromMat(word_logprobs.ColRange(1, num_words - 1));
      word_probs.ApplyExp();
      CuVector<BaseFloat> row_sums(this_size);
      row_sums.AddColSumMat(1.0, word_probs, 0.0);
      row_sums.ApplyLog();
      *objf_den_exact -= VecVec(row_sums, this_output_weights);
    }

    // In preparation for computing the denominator objf, change 'word_logprobs'
    // to contain q = (l < 0 ? exp(l) : l + 1.0), instead of the unnormalized
    // logprob l.  For most purposes you can think of this as behaving like
    // exp(l); it just has better behavior at the beginning of training when there
    // is a danger of divergence.  We can prove that the objective is a closer
    // bound on the true (log-probability-based) objective than if we used a
    // simple exp.  note: all this bound stuff is really geared towards the
    // sampling case, there is really no point in it if we're not doing sampling
    // and some of these code paths will only be used in test code.
    word_logprobs.ApplyExpSpecial();

    { // This block computes *objf_den.

      // we call this variable 'q_noeps' because in the math described in
      // rnnlm-example-utils.h it is described as q(i,w), and because we're
      // skipping over the epsilon symbol (which we don't want to include in the
      // sum because it can never be output) by removing the first row.
      CuSubMatrix<BaseFloat> q_noeps(word_logprobs,
                                     0, word_logprobs.NumRows(),
                                     1, num_words - 1);
      // den_term(i) = 1.0 - (\sum_w q(i,w))
      CuVector<BaseFloat> den_term(word_logprobs.NumRows(), kUndefined);
      den_term.Set(1.0);
      den_term.AddColSumMat(-1.0, q_noeps, 1.0);
      // note: objf = \sum_i weight(i) * ( num_term(i) + den_term(i) ),
      // this is the term \sum_i weight(i) * den_term(i).
      *objf_den += VecVec(den_term, this_output_weights);
    }

    // The rest of this function computes the derivative w.r.t.
    // word_embedding_deriv and/or nnet_output_deriv, for which we
    if (!(word_embedding_deriv || nnet_output_deriv))
      continue;

    // To avoid one CUDA operation, we're going to make 'word_logprobs'
    // the *negative* of the derivative of the objf w.r.t.
    // the original 'word_logprobs'  Note: don't worry about the
    // fact that we're including the epsilon word at this point;
    // later on we'll ignore the value of the first column.

    // The derivative of the function q(l) = (l < 0 ? exp(l) : l + 1.0)
    // equals (l < 0 ? exp(l) : 1.0), which we can compute by
    // applying a ceiling to q at 1.0.
    word_logprobs.ApplyCeiling(1.0);

    // Include the factor 'minibatch.output_weights'.
    word_logprobs.MulRowsVec(this_output_weights);

    if (objective_config.den_term_limit != 0.0) {
      // If it's nonzero then check that it's negative, and not too close to zero,
      // which would likely cause failure to converge.  The default is 10.0.
      KALDI_ASSERT(objective_config.den_term_limit < -0.5);
      BaseFloat limit = objective_config.den_term_limit;
      if (weight != NULL && objf_den != NULL && *weight > 0 &&
          (*objf_den / *weight) < limit) {
        // note: both things being divided below are negative, and
        // 'scale' will be between zero and one.
        BaseFloat scale = limit / (*objf_den / *weight);
        // We scale the denominator part of the objective down by the inverse of
        // the factor by which the denominator part of the objective exceeds the
        // limit.  This point in the code should only be reached on the first few
        // iterations of training, or if there is some kind of instability,
        // because the (*objf_den / *weight) will usually be close to zero,
        // e.g. -0.01, while 'limit' is expected to be larger, like -10.0.
        word_logprobs.Scale(scale);
      }
    }

    // After the following statement, 'word_logprobs' will contains the negative
    // of the derivative of the objective function w.r.t. l(i, x), except that the
    // first column (for epsilon) should be ignored.

    word_logprobs.AddSmat(-1.0, this_output_words_smat);

    // l_deriv_noeps is the negative of the derivative of the objective function
    // w.r.t. the unnormalized log-likelihoods 'l' (with the 0th row, for epsilon,
    // not included).
    CuSubMatrix<BaseFloat> l_deriv_noeps(word_logprobs,
                                         0, this_size,
                                         1, num_words - 1);

    // The following statements are doing the backprop w.r.t. the statement:
    //  word_logprobs.AddMatMat(1.0, nnet_output, kNoTrans,
    //                          word_embedding, kTrans, 0.0);
    // (note: 'word_logprobs' in the code corresponds to 'l' in the formulas).
    // The -1.0's are because we have at this point the negative of the
    // derivative w.r.t the unnormalized log-likelihoods.
    if (word_embedding_deriv) {
      CuSubMatrix<BaseFloat> word_embedding_deriv_noeps(
          *word_embedding_deriv, 1, num_words - 1, 0, embedding_dim);
      word_embedding_deriv_noeps.AddMatMat(-1.0, l_deriv_noeps, kTrans,
                    nnet_output.RowRange(this_start, this_size), kNoTrans, 1.0);
    }
    if (nnet_output_deriv) {
      CuSubMatrix<BaseFloat> word_embedding_noeps(
          word_embedding, 1, num_words - 1, 0, embedding_dim);
      nnet_output_deriv->RowRange(this_start, this_size).
                         AddMatMat(-1.0, l_deriv_noeps, kNoTrans,
                                   word_embedding_noeps, kNoTrans, 1.0);
    }
    this_start += this_size;
  }
}

void ProcessRnnlmOutput(
    const RnnlmObjectiveOptions &objective_config,
    const RnnlmExample &minibatch,
    const RnnlmExampleDerived &derived,
    const CuMatrixBase<BaseFloat> &word_embedding,
    const CuMatrixBase<BaseFloat> &nnet_output,
    CuMatrixBase<BaseFloat> *word_embedding_deriv,
    CuMatrixBase<BaseFloat> *nnet_output_deriv,
    BaseFloat *weight,
    BaseFloat *objf_num,
    BaseFloat *objf_den,
    BaseFloat *objf_den_exact) {
  int32 num_chunks = minibatch.num_chunks,
      chunk_length = minibatch.chunk_length;
  KALDI_ASSERT(nnet_output.NumRows() == num_chunks * chunk_length &&
               nnet_output.NumCols() == word_embedding.NumCols() &&
               minibatch.vocab_size == word_embedding.NumRows());

  bool using_sampling = !(minibatch.sampled_words.empty());
  if (using_sampling) {
    ProcessRnnlmOutputSampling(objective_config,
                               minibatch, derived, word_embedding,
                               nnet_output, word_embedding_deriv,
                               nnet_output_deriv, weight, objf_num,
                               objf_den, objf_den_exact);
  } else {
    int64 size = int64(word_embedding.NumRows()) * nnet_output.NumRows();
    if (size < objective_config.max_logprob_elements) {
      ProcessRnnlmOutputNoSampling(objective_config,
                                   minibatch, derived, word_embedding,
                                   nnet_output, word_embedding_deriv,
                                   nnet_output_deriv, weight, objf_num,
                                   objf_den, objf_den_exact);

    } else {
      ProcessRnnlmOutputNoSamplingBatched(objective_config,
                                          minibatch, derived, word_embedding,
                                          nnet_output, word_embedding_deriv,
                                          nnet_output_deriv, weight, objf_num,
                                          objf_den, objf_den_exact);
    }
  }
}

void GetRnnlmExampleDerived(const RnnlmExample &minibatch,
                            bool need_embedding_deriv,
                            RnnlmExampleDerived *derived) {
  derived->cu_input_words = minibatch.input_words;

  bool using_sampling = !(minibatch.sampled_words.empty());
  if (using_sampling) {
    derived->cu_output_words = minibatch.output_words;
    derived->cu_sampled_words = minibatch.sampled_words;
  } else {
    CuArray<int32> cu_output_words(minibatch.output_words);
    CuSparseMatrix<BaseFloat> output_words_smat(cu_output_words,
                                                minibatch.output_weights,
                                                minibatch.vocab_size);
    derived->output_words_smat.Swap(&output_words_smat);
  }

  if (need_embedding_deriv) {
    CuSparseMatrix<BaseFloat> input_words_smat(derived->cu_input_words,
                                               minibatch.vocab_size, kTrans);
    derived->input_words_smat.Swap(&input_words_smat);
  }
}

void RenumberRnnlmExample(RnnlmExample *minibatch,
                          std::vector<int32> *active_words) {
  KALDI_ASSERT(!minibatch->sampled_words.empty());
  unordered_set<int32> active_words_set;
  active_words_set.insert(minibatch->input_words.begin(),
                          minibatch->input_words.end());
  active_words_set.insert(minibatch->sampled_words.begin(),
                          minibatch->sampled_words.end());

  active_words->clear();
  active_words->insert(active_words->end(),
                       active_words_set.begin(),
                       active_words_set.end());
  std::sort(active_words->begin(),
            active_words->end());
  unordered_map<int32, int32> active_words_map;
  size_t n = active_words->size();
  for (size_t i = 0; i < n; i++)
    active_words_map[(*active_words)[i]] = i;

  // Now remap 'input_words' and 'sampled_words'

  std::vector<int32>::iterator iter = minibatch->input_words.begin(),
      end = minibatch->input_words.end();
  for (; iter != end; ++iter)
    *iter = active_words_map[*iter];
  iter = minibatch->sampled_words.begin();
  end = minibatch->sampled_words.end();
  for (; iter != end; ++iter)
    *iter = active_words_map[*iter];
  minibatch->vocab_size = static_cast<int32>(n);
}

}  // namespace rnnlm
}  // namespace kaldi

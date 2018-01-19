// src/rnnlm/rnnlm-compute-state.h

// Copyright 2017 Johns Hopkins University (author: Daniel Povey)
//           2017 Yiming Wang
//           2017 Hainan Xu

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

#ifndef KALDI_RNNLM_COMPUTE_STATE_H_
#define KALDI_RNNLM_COMPUTE_STATE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"
#include "rnnlm/rnnlm-core-compute.h"

namespace kaldi {
namespace rnnlm {

struct RnnlmComputeStateComputationOptions {
  bool debug_computation;
  bool normalize_probs;
  // We need this when we initialize the RnnlmComputeState and pass the BOS history.
  int32 bos_index;
  // We need this to compute the Final() cost of a state.
  int32 eos_index;
  // This is not needed for computation; included only for ease of scripting.
  int32 brk_index;
  nnet3::NnetOptimizeOptions optimize_config;
  nnet3::NnetComputeOptions compute_config;
  RnnlmComputeStateComputationOptions():
      debug_computation(false),
      normalize_probs(false),
      bos_index(-1),
      eos_index(-1),
      brk_index(-1)
      { }

  void Register(OptionsItf *opts) {
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");
    opts->Register("normalize-probs", &normalize_probs, "If true, word "
       "probabilities will be correctly normalized (otherwise the sum-to-one "
       "normalization is approximate)");
    opts->Register("bos-symbol", &bos_index, "Index in wordlist representing "
                   "the begin-of-sentence symbol");
    opts->Register("eos-symbol", &eos_index, "Index in wordlist representing "
                   "the end-of-sentence symbol");
    opts->Register("brk-symbol", &brk_index, "Index in wordlist representing "
                   "the break symbol. It is not needed in the computation "
                   "and we are including it for ease of scripting");

    // Register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);

    // Register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};

/*
  This class const references to the word-embedding, nnet3 part of rnnlm and
the RnnlmComputeStateComputationOptions. It handles the computation of the nnet3
object
*/
class RnnlmComputeStateInfo  {
 public:
  RnnlmComputeStateInfo(
      const RnnlmComputeStateComputationOptions &opts,
      const kaldi::nnet3::Nnet &rnnlm,
      const CuMatrix<BaseFloat> &word_embedding_mat);

  const RnnlmComputeStateComputationOptions &opts;
  const kaldi::nnet3::Nnet &rnnlm;
  const CuMatrix<BaseFloat> &word_embedding_mat;

  // The compiled, 'looped' computation.
  nnet3::NnetComputation computation;
};

/*
  This class handles the neural net computation; it's mostly accessed
  via other wrapper classes. 
 
  Each time this class takes a new word and advance the NNET computation by
  one step, and works out log-prob of words to be used in lattice rescoring. */

class RnnlmComputeState {
 public:
  /// We compile the computation and generate the state after the BOS history.
  RnnlmComputeState(const RnnlmComputeStateInfo &info, int32 bos_index);

  RnnlmComputeState(const RnnlmComputeState &other);

  /// Generate another state by passing the next-word.
  /// The pointer is owned by the caller.
  RnnlmComputeState* GetSuccessorState(int32 next_word) const;

  /// Return the log-prob that the model predicts for the provided word-index,
  /// given the previous history determined by the sequence of calls to AddWord()
  /// (implicitly starting with the BOS symbol).
  BaseFloat LogProbOfWord(int32 word_index) const;

  // This function computes logprobs of all words and set it to output Matrix
  // Note: (*output)(0, 0) corresponds to <eps> symbol and it should NEVER be
  // used in any computation by the caller. To avoid causing unexpected issues,
  // we here set it to a very small number
  void GetLogProbOfWords(CuMatrixBase<BaseFloat>* output) const;
  /// Advance the state of the RNNLM by appending this word to the word sequence.
  void AddWord(int32 word_index);
 private:
  /// This function does the computation for the next chunk.
  void AdvanceChunk();

  const RnnlmComputeStateInfo &info_;
  nnet3::NnetComputer computer_;
  int32 previous_word_;

  // This is the log of the sum of the exp'ed values in the output.
  // Only used if config_.normalize_probs is set to be true.
  BaseFloat normalization_factor_;

  // This points to the matrix returned by GetOutput() on the Nnet object.
  // This pointer is not owned by this class.
  const CuMatrixBase<BaseFloat> *predicted_word_embedding_;
};


} // namespace rnnlm
} // namespace kaldi

#endif  // KALDI_RNNLM_COMPUTE_STATE_H_

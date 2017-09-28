// rnnlm/kaldi-rnnlm-simple-looped.h

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

#ifndef KALDI_RNNLM_COMPUTE_STATEH_
#define KALDI_RNNLM_COMPUTE_STATEH_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"
#include "rnnlm/rnnlm-core-compute.h"

namespace kaldi {
namespace nnet3 {

struct RnnlmComputeStateComputationOptions {
  bool debug_computation;
  bool normalize_probs;
  string bos_symbol;
  string eos_symbol;
  string oos_symbol;
  string brk_symbol;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  RnnlmComputeStateComputationOptions():
      debug_computation(false),
      normalize_probs(false),
      bos_symbol("<s>"),
      eos_symbol("</s>"),
      oos_symbol("<oos>"),
      brk_symbol("<brk>") { }

  void Register(OptionsItf *opts) {
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");
    opts->Register("normalize-probs", &normalize_probs, "If true, word "
       "probabilities will be correctly normalized (otherwise the sum-to-one "
       "normalization is approximate)");
    opts->Register("bos-symbol", &bos_symbol, "symbol in wordlist representing "
                   "the begin-of-sentence symbol, usually <s>");
    opts->Register("eos-symbol", &bos_symbol, "symbol in wordlist representing "
                   "the end-of-sentence symbol, usually </s>");
    opts->Register("oos-symbol", &oos_symbol, "symbol in wordlist representing "
                   "the out-of-vocabulary symbol, usually <oos>");
    opts->Register("eos-symbol", &brk_symbol, "symbol in wordlist representing "
                   "the break symbol, usually <brk>");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);

    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};

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
  NnetComputation computation;
};

/*
  This class handles the neural net computation; it's mostly accessed
  via other wrapper classes.

  It accept just input word as features */
class RnnlmComputeState {
 public:
  /// we compile the computation and generate the state after the BOS history
  RnnlmComputeState(const RnnlmComputeStateInfo &info, int32 bos_index);
  /// copy constructor
  RnnlmComputeState(const RnnlmComputeState &other);

  /// generate another state by passing the next-word
  /// pointer owned by the caller
  RnnlmComputeState* GetSuccessorState(int32 next_word) const;

  /// Return the log-prob that the model predicts for the provided word-index,
  /// given the previous history determined by the sequence of calls to AddWord()
  /// (implicitly starting with the BOS symbol).
  BaseFloat LogProbOfWord(int32 word_index) const;
 private:
  /// Advance the state of the RNNLM by appending this word to the word sequence.
  void AddWord(int32 word_index);
  /// This function does the computation for the next chunk.
  void AdvanceChunk();

  const RnnlmComputeStateInfo &info_;
  NnetComputer computer_;
  int32 previous_word_;

  // this is the log of the sum of the exp'ed values in the output
  BaseFloat normalization_factor_;

  // this points to the matrix returned by GetOutput() on the Nnet object
  // pointer not owned here
  const CuMatrixBase<BaseFloat> *predicted_word_embedding_;
};


} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_RNNLM_COMPUTE_STATE_H

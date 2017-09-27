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

#ifndef KALDI_RNNLM_SIMPLE_LOOPED_H_
#define KALDI_RNNLM_SIMPLE_LOOPED_H_

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

// See also nnet-am-decodable-simple.h, which is a decodable object that's based
// on breaking up the input into fixed chunks.  The decodable object defined here is based on
// 'looped' computations, which naturally handle infinite left-context (but are
// only ideal for systems that have only recurrence in the forward direction,
// i.e. not BLSTMs... because there isn't a natural way to enforce extra right
// context for each chunk.)


// Note: the 'simple' in the name means it applies to networks for which
// IsSimpleNnet(nnet) would return true.  'looped' means we use looped
// computations, with a kGotoLabel statement at the end of it.
struct RnnlmComputeStateComputationOptions {
  bool debug_computation;
  bool force_normalize;
  string bos_symbol;
  string eos_symbol;
  string oos_symbol;
  string brk_symbol;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  RnnlmComputeStateComputationOptions():
      debug_computation(false),
      force_normalize(false),
      bos_symbol("<s>"),
      eos_symbol("</s>"),
      oos_symbol("<oos>"),
      brk_symbol("<brk>") { }

  void Check() const {
  }

  void Register(OptionsItf *opts) {
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");
    opts->Register("force-normalize", &force_normalize, "If true, force "
                   " normalize the word posteriors");
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

  void Init(const RnnlmComputeStateComputationOptions &opts,
            const kaldi::nnet3::Nnet &rnnlm,
            const CuMatrix<BaseFloat> &word_embedding_mat);

  const RnnlmComputeStateComputationOptions &opts;

  const kaldi::nnet3::Nnet &rnnlm;
  const CuMatrix<BaseFloat> &word_embedding_mat;

  // The output dimension of the nnet neural network (not the final output).
  int32 nnet_output_dim;

  // The compiled, 'looped' computation.
  NnetComputation computation;
};

/*
  This class handles the neural net computation; it's mostly accessed
  via other wrapper classes.

  It accept just input features */
class RnnlmComputeState {
 public:
  /**
     This constructor takes features as input.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] info   This helper class contains all the static pre-computed information
                        this class needs, and contains a pointer to the neural net.
     @param [in] feats  The input feature word
  */
  RnnlmComputeState(const RnnlmComputeStateInfo &info);
  RnnlmComputeState(const RnnlmComputeState &other);

  // Updates feats_ with the new incoming word specified in word_indexes
  // We usually do this one at a time
  void TakeFeatures(int32 word_index);
  CuVector<BaseFloat>* GetOutput();
  BaseFloat LogProbOfWord(int32 word_index,
                          const CuVectorBase<BaseFloat> &hidden) const;

 private:
  // This function does the computation for the next chunk.
  void AdvanceChunk();
  const RnnlmComputeStateInfo &info_;
  NnetComputer computer_;
  int32 feats_;

  // The current nnet's output that we got from the last time we
  // ran the computation.
  Matrix<BaseFloat> current_nnet_output_;

  // The time-offset of the current log-posteriors, equals
  // -1 when initialized, or takes a new word, or 0 once AdvanceChunk() was called
  int32 current_log_post_offset_;
};


} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_RNNLM_SIMPLE_LOOPED_H_

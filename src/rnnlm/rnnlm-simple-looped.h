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
//#include "itf/decodable-itf.h"
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
struct RnnlmSimpleLoopedComputationOptions {
  int32 frames_per_chunk;
  bool debug_computation;
  bool force_normalize;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  RnnlmSimpleLoopedComputationOptions():
      frames_per_chunk(1),
      debug_computation(false),
      force_normalize(false) { }

  void Check() const {
    KALDI_ASSERT(frames_per_chunk > 0);
  }

  void Register(OptionsItf *opts) {
    opts->Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately evaluated "
                   "by the neural net.");
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");
    opts->Register("force-normalize", &force_normalize, "If true, force "
                   " normalize the word posteriors");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);

    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};

class RnnlmSimpleLoopedInfo  {
 public:
  RnnlmSimpleLoopedInfo(
      const RnnlmSimpleLoopedComputationOptions &opts,
      const kaldi::nnet3::Nnet &rnnlm,
      const CuMatrix<BaseFloat> &word_embedding_mat);

  void Init(const RnnlmSimpleLoopedComputationOptions &opts,
            const kaldi::nnet3::Nnet &rnnlm,
            const CuMatrix<BaseFloat> &word_embedding_mat);

  const RnnlmSimpleLoopedComputationOptions &opts;

  const kaldi::nnet3::Nnet &rnnlm;
  const CuMatrix<BaseFloat> &word_embedding_mat;

  // frames_left_context equals the model left context plus the value of the
  // --extra-left-context-initial option.
  int32 frames_left_context;
  // frames_right_context is the same as the right-context of the model.
  int32 frames_right_context;
  // The frames_per_chunk equals the number of input frames we need for each
  // chunk (except for the first chunk).
  int32 frames_per_chunk;

  // The output dimension of the nnet neural network (not the final output).
  int32 nnet_output_dim;

  // The 3 computation requests that are used to create the looped
  // computation are stored in the class, as we need them to work out
  // exactly shich iVectors are needed.
  ComputationRequest request1, request2, request3;

  // The compiled, 'looped' computation.
  NnetComputation computation;
};

/*
  This class handles the neural net computation; it's mostly accessed
  via other wrapper classes.

  It accept just input features */
class RnnlmSimpleLooped {
 public:
  /**
     This constructor takes features as input.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] info   This helper class contains all the static pre-computed information
                        this class needs, and contains a pointer to the neural net.
     @param [in] feats  The input feature matrix.
  */
  RnnlmSimpleLooped(const RnnlmSimpleLoopedInfo &info);

  RnnlmSimpleLooped(const RnnlmSimpleLooped &other);

  inline int32 NnetOutputDim() const { return info_.nnet_output_dim; }

  // Gets the nnet's output for a particular frame, with 0 <= frame < NumFrames().
  // 'output' must be correctly sized (with dimension NnetOutputDim()).  Note:
  // you're expected to call this, and GetOutput(), in an order of increasing
  // frames.  If you deviate from this, one of these calls may crash.
//  void GetNnetOutputForFrame(int32 frame, VectorBase<BaseFloat> *output);

  // Updates feats_ with the new incoming word specified in word_indexes
  // We usually do this one at a time
  void TakeFeatures(const std::vector<int32> &word_indexes);

  // Gets the output for a particular frame and word_index, with
  // 0 <= frame < NumFrames().
//  BaseFloat GetOutput(int32 frame, int32 word_index);
  // create a CuVector in heap, pointer owned by the caller
  CuVector<BaseFloat>* GetOutput(int32 frame);

  BaseFloat LogProbOfWord(int32 word_index,
                          const CuVectorBase<BaseFloat> &hidden) const;

 private:
  // This function does the computation for the next chunk.
  void AdvanceChunk();

  const RnnlmSimpleLoopedInfo &info_;

  NnetComputer computer_;

  SparseMatrix<BaseFloat> feats_;

  // The current nnet's output that we got from the last time we
  // ran the computation.
  Matrix<BaseFloat> current_nnet_output_;

  // The time-offset of the current log-posteriors, equals
  // -1 when initialized, or 0 once AdvanceChunk() was called
  int32 current_log_post_offset_;
};


} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_RNNLM_SIMPLE_LOOPED_H_

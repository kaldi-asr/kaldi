// nnet3/nnet-simple-computer.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Vimal Manohar

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

#ifndef KALDI_NNET3_NNET_SIMPLE_COMPUTER_H
#define KALDI_NNET3_NNET_SIMPLE_COMPUTER_H

#include "base/kaldi-common.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"

namespace kaldi {
namespace nnet3 {

struct NnetSimpleComputerOptions {
  int32 extra_left_context;
  int32 frames_per_chunk;
  bool debug_computation;
  
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;

  NnetSimpleComputerOptions():
      extra_left_context(0),
      frames_per_chunk(50),
      debug_computation(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("extra-left-context", &extra_left_context,
                   "Number of frames of additional left-context to add on top "
                   "of the neural net's inherent left context "
                   "(may be useful in recurrent setups");
    opts->Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately "
                   "evaluated by the neural net.");
    opts->Register("debug-computation", &debug_computation, "If true, turn on "
                   "debug for the actual computation (very verbose!)");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);

    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);
  }
};

class NnetSimpleComputer {
 public:
  
  /// Constructor that just takes the features, and the left and right
  /// contexts as input, but can also optionally
  /// take batch-mode or online iVectors.  Note: it stores references to all
  /// arguments to the constructor, so don't delete them till this goes out of
  /// scope.
  /// This is mainly when the raw neural network is used without the 
  /// acoustic model.
  NnetSimpleComputer(const NnetSimpleComputerOptions &opts,
                     const Nnet &nnet,
                     const MatrixBase<BaseFloat> &feats,
                     int32 left_context,
                     int32 right_context,
                     const VectorBase<BaseFloat> *ivector = NULL,
                     const MatrixBase<BaseFloat> *online_ivectors = NULL,
                     int32 online_ivector_period = 1);

  /// Constructor that just takes the features as input, but can also optionally
  /// take batch-mode or online iVectors.  Note: it stores references to all
  /// arguments to the constructor, so don't delete them till this goes out of
  /// scope.
  NnetSimpleComputer(const NnetSimpleComputerOptions &opts,
                     const Nnet &nnet,
                     const MatrixBase<BaseFloat> &feats,
                     const VectorBase<BaseFloat> *ivector = NULL,
                     const MatrixBase<BaseFloat> *online_ivectors = NULL,
                     int32 online_ivector_period = 1);

  /// Constructor that also accepts iVectors estimated online;
  /// online_ivector_period is the time spacing between rows of the matrix.
  NnetSimpleComputer(const NnetSimpleComputerOptions &opts, 
                     const Nnet &nnet,
                     const MatrixBase<BaseFloat> &feats,
                     const MatrixBase<BaseFloat> &online_ivectors,
                     int32 online_ivector_period);

  /// Constructor that accepts iVectors estimated in batch mode
  NnetSimpleComputer(const NnetSimpleComputerOptions &opts,
                     const Nnet &nnet,
                     const MatrixBase<BaseFloat> &feats,
                     const VectorBase<BaseFloat> &ivector);

  /// This function does the forward pass through the neural network
  /// and returns the result to the output matrix
  void GetOutput(Matrix<BaseFloat> *output);

 protected:
  // This call is made to ensure that we have the log-probs for this frame
  // cached in current_log_post_.
  void EnsureFrameIsComputed(int32 frame);

  // This function does the actual nnet computation; it is called from
  // EnsureFrameIsComputed. It puts its output in current_log_post_.
  // This is just a wrapper to the DoNnetComputationInternal function.
  // This is virtual because its implementation in DecodableAmNnetSimple 
  // class must also add in acoustic scale and priors.
  virtual void DoNnetComputation(int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    const VectorBase<BaseFloat> &ivector,
    int32 output_t_start,
    int32 num_output_frames);
  
  // This function does the internal computations in the 
  // actual nnet computation; it is called from
  // DoNnetComputation. Any padding at file start/end is done by
  // the caller of this function (so the input should exceed the output
  // by a suitable amount of context).  
  // It returns the output as a CuMatrix
  void DoNnetComputationInternal(int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    const VectorBase<BaseFloat> &ivector,
    int32 output_t_start,
    int32 num_output_frames,
    CuMatrix<BaseFloat> *cu_output);

  // Gets the iVector that will be used for this chunk of frames, if
  // we are using iVectors (else does nothing).
  void GetCurrentIvector(int32 output_t_start, int32 num_output_frames,
    Vector<BaseFloat> *ivector);

  void PossiblyWarnForFramesPerChunk() const;

  // returns dimension of the provided iVectors if supplied, or 0 otherwise.
  int32 GetIvectorDim() const;

  int32 LeftContext() const { return left_context_; }
  int32 RightContext() const { return right_context_; }

  const NnetSimpleComputerOptions &opts_;

  const Nnet &nnet_;

  const MatrixBase<BaseFloat> &feats_;

  // ivector_ is the iVector if we're using iVectors that are estimated in batch
  // mode.
  const VectorBase<BaseFloat> *ivector_;

  // online_ivector_feats_ is the iVectors if we're using online-estimated ones.
  const MatrixBase<BaseFloat> *online_ivector_feats_;
  // online_ivector_period_ helps us interpret online_ivector_feats_; it's the
  // number of frames the rows of ivector_feats are separated by.
  int32 online_ivector_period_;

  CachingOptimizingCompiler compiler_;

  // The current log-posteriors that we got from the last time we
  // ran the computation. 
  // NOTE: This is just the output of the neural network, not necessarily the 
  // log-posteriors
  Matrix<BaseFloat> current_log_post_;
  // The time-offset of the current log-posteriors. 
  int32 current_log_post_offset_;
 
  // The left and right contexts that are needed for per-frame computation
  int32 left_context_;
  int32 right_context_;
};

} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_NNET_SIMPLE_COMPUTER_H

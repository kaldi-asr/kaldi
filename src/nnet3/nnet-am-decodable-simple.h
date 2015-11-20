// nnet3/nnet-am-decodable-simple.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_
#define KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"

namespace kaldi {
namespace nnet3 {


struct NnetSimpleComputationOptions {
  int32 extra_left_context;
  int32 frames_per_chunk;
  BaseFloat acoustic_scale;
  bool debug_computation;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;

  NnetComputationOptions():
      extra_left_context(0),
      frames_per_chunk(50),
      acoustic_scale(0.1),
      debug_computation(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("extra-left-context", &extra_left_context,
                   "Number of frames of additional left-context to add on top "
                   "of the neural net's inherent left context (may be useful in "
                   "recurrent setups");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic log-likelihoods");
    opts->Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately evaluated "
                   "by the neural net.");
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

/*
  This base-class for DecodableAmNnetSimple handles just the nnet computation;
  it can also be used if you just want the nnet output directly.

   It can accept just input features, or
   input features plus iVectors.
*/
class NnetDecodableBase {
 public:
  /**
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.
     @param [in] opts   The options class.  Warning: it includes an acoustic
                        weight, whose default is 0.1; you may sometimes want to
                        change this to 1.0.
     @param [in] nnet   The neural net that we're going to do the computation with
     @param [in] priors Vector of priors-- if supplied and nonempty, we subtract
                        the log of these priors from the nnet output.


  */
  NnetDecodableBase(const NnetSimpleComputationOptions &opts,
                    const Nnet &nnet,
                    const VectorBase<BaseFloat> &priors,
                    const MatrixBase<BaseFloat> &feats,
                    const VectorBase<BaseFloat> *ivector = NULL,
                    const MatrixBase<BaseFloat> *online_ivectors = NULL,
                    int32 online_ivector_period = 1);

  /// Constructor that also accepts iVectors estimated online;
  /// online_ivector_period is the time spacing between rows of the matrix.
  DecodableAmNnetSimple(const DecodableAmNnetSimpleOptions &opts,
                        const TransitionModel &trans_model,
                        const AmNnetSimple &am_nnet,
                        const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> &online_ivectors,
                        int32 online_ivector_period);

  /// Constructor that accepts iVectors estimated in batch mode
  DecodableAmNnetSimple(const DecodableAmNnetSimpleOptions &opts,
                        const TransitionModel &trans_model,
                        const AmNnetSimple &am_nnet,
                        const MatrixBase<BaseFloat> &feats,
                        const VectorBase<BaseFloat> &ivector);


  // Note, frames are numbered from zero.  But transition_id is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual int32 NumFramesReady() const { return feats_.NumRows(); }

  // Note: these indices are transition-ids; they are one-based!  This is for
  // compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 private:
  // This call is made to ensure that we have the log-probs for this frame
  // cached in current_log_post_.
  void EnsureFrameIsComputed(int32 frame);

  // This function does the actual nnet computation; it is called from
  // EnsureFrameIsComputed.  Any padding at file start/end is done by
  // the caller of this function (so the input should exceed the output
  // by a suitable amount of context).  It puts its output in current_log_post_.
  void DoNnetComputation(int32 input_t_start,
                         const MatrixBase<BaseFloat> &input_feats,
                         const VectorBase<BaseFloat> &ivector,
                         int32 output_t_start,
                         int32 num_output_frames);

  // Gets the iVector that will be used for this chunk of frames, if
  // we are using iVectors (else does nothing).
  void GetCurrentIvector(int32 output_t_start, int32 num_output_frames,
                         Vector<BaseFloat> *ivector);

  void PossiblyWarnForFramesPerChunk() const;

  // returns dimension of the provided iVectors if supplied, or 0 otherwise.
  int32 GetIvectorDim() const;

  const DecodableAmNnetSimpleOptions &opts_;
  const TransitionModel &trans_model_;
  const AmNnetSimple &am_nnet_;
  // the log priors (or the empty vector if the priors are not set in the model)
  CuVector<BaseFloat> log_priors_;
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
  Matrix<BaseFloat> current_log_post_;
  // The time-offset of the current log-posteriors.
  int32 current_log_post_offset_;


};

class DecodableAmNnetSimple: public NnetDecodableBase {

  /// This constructor takes features as input, and you can either supply
  /// a single iVector input, estimated in batch-mode ('ivector'), or
  /// 'online' iVectors ('online_ivectors' and 'online_ivector_period', or
  /// none at all.   Note: it stores references to all
  /// arguments to the constructor, so don't delete them till this goes out of
  /// scope.
  DecodableAmNnetSimple(const DecodableAmNnetSimpleOptions &opts,
                        const TransitionModel &trans_model,
                        const AmNnetSimple &am_nnet,
                        const MatrixBase<BaseFloat> &feats,
                        const VectorBase<BaseFloat> *ivector = NULL,
                        const MatrixBase<BaseFloat> *online_ivectors = NULL,
                        int32 online_ivector_period = 1);


 private:
  const TransitionModel &trans_model_;

};


} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_

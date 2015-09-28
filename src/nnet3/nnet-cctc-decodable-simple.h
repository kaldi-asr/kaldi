// nnet3/nnet-cctc-decodable-simple.h

// Copyright     2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_CCTC_DECODABLE_SIMPLE_H_
#define KALDI_NNET3_NNET_CCTC_DECODABLE_SIMPLE_H_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-simple.h"
#include "ctc/cctc-transition-model.h"

namespace kaldi {
namespace nnet3 {


// so far, these are the same as DecodableAmNnetSimpleOptions.
struct DecodableNnetCctcSimpleOptions {
  int32 frame_subsampling_factor;
  int32 extra_left_context;
  int32 frames_per_chunk;
  BaseFloat acoustic_scale;
  bool debug_computation;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;

  // for recurrent setups you may want to set frame_per_chunk to a larger number
  // like 200, and extra_left_context to a reasonably high value (e.g. 20 or
  // 40).
  DecodableNnetCctcSimpleOptions():
      frame_subsampling_factor(1),
      extra_left_context(0),
      frames_per_chunk(50),
      acoustic_scale(1.0),
      debug_computation(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor,
                   "Required if the frame-rate of the output in CTC is be less "
                   "than the frame-rate of the original alignment.");
    opts->Register("extra-left-context", &extra_left_context,
                   "Number of frames of additional left-context to add on top "
                   "of the neural net's inherent left context (may be useful in "
                   "recurrent setups");
    opts->Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately evaluated "
                   "by the neural net.  Measured before any subsampling (i.e. "
                   "counts input frames");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic log-likelihoods");
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

/* DecodableNnetCctcSimple is a decodable object that decodes with a neural net
   and CCTC.  It can accept just input features, or input features plus
   iVectors.
*/
class DecodableNnetCctcSimple: public DecodableInterface {
 public:
  
  /// Constructor that just takes the features as input, but can also optionally
  /// take batch-mode or online iVectors.  Note: it stores references to all
  /// arguments to the constructor, so don't delete them till this goes out of
  /// scope.
  DecodableNnetCctcSimple(const DecodableNnetCctcSimpleOptions &opts,
                          const ctc::CctcTransitionModel &trans_model,
                          const Nnet &nnet,
                          const MatrixBase<BaseFloat> &feats,
                          const VectorBase<BaseFloat> *ivector = NULL,
                          const MatrixBase<BaseFloat> *online_ivectors = NULL,
                          int32 online_ivector_period = 1);

  /// Constructor that also accepts iVectors estimated online;
  /// online_ivector_period is the time spacing between rows of the matrix.
  DecodableNnetCctcSimple(const DecodableNnetCctcSimpleOptions &opts,
                          const ctc::CctcTransitionModel &trans_model,
                          const Nnet &nnet,
                          const MatrixBase<BaseFloat> &feats,
                          const MatrixBase<BaseFloat> &online_ivectors,
                          int32 online_ivector_period);

  /// Constructor that accepts iVectors estimated in batch mode
  DecodableNnetCctcSimple(const DecodableNnetCctcSimpleOptions &opts,
                          const ctc::CctcTransitionModel &trans_model,
                          const Nnet &nnet,
                          const MatrixBase<BaseFloat> &feats,
                          const VectorBase<BaseFloat> &ivector);

  
  // Note, frames are numbered from zero.  But graph_label is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 graph_label);

  virtual int32 NumFramesReady() const {
    return feats_.NumRows() / opts_.frame_subsampling_factor;
  }

  // Note: these indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumGraphLabels(); }

  virtual bool IsLastFrame(int32 frame) const;
  
 private:
  // This call is made to ensure that we have the log-probs for this frame
  // cached in current_log_post_.
  void EnsureFrameIsComputed(int32 frame);

  // This function does the actual nnet computation; it is called from
  // EnsureFrameIsComputed.  Any padding at file start/end is done by the caller
  // of this function (so the input should exceed the output by a suitable
  // amount of context).  It puts its output in current_log_numerators_ (which come
  // directly from the neural net) and current_log_denominators_ (which
  // come from multiplying the exp of current_log_numerators_ by cu_weights_ and
  // then taking the log).
  // Note: output_t is a frame in the non-subsampled numbering;
  // num_subsampled_frames is in the subsampled numbering.
  void DoNnetComputation(int32 input_t_start,
                         const MatrixBase<BaseFloat> &input_feats,
                         const VectorBase<BaseFloat> &ivector,
                         int32 output_t_start,
                         int32 num_subsampled_frames);

  // Gets the iVector that will be used for this chunk of frames, if we are
  // using iVectors (else does nothing).  t is the (approximate) center of the
  // chunk of frames (in non-subsampled numbering).
  void GetCurrentIvector(int32 t,
                         Vector<BaseFloat> *ivector);

  void InitializeCommon();

  // returns dimension of the provided iVectors if supplied, or 0 otherwise.
  int32 GetIvectorDim() const;
  
  DecodableNnetCctcSimpleOptions opts_;
  const ctc::CctcTransitionModel &trans_model_;
  CuMatrix<BaseFloat> cu_weights_;  // derived from trans_model_.
  const Nnet &nnet_;
  // nnet_left_context_ and nnet_right_context_ are the minimal left-context and
  // right context of the network, derived from nnet_.
  int32 nnet_left_context_;
  int32 nnet_right_context_;
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
  Matrix<BaseFloat> current_log_numerators_;
  // these correspond to the same frames as current_log_numerators, and
  // are the denominator quantities for each history-state.
  Matrix<BaseFloat> current_log_denominators_;

  // The time-offset of current_log_numerators_ and current_log_denominators_.
  // This is measured in subsampled frames, i.e. in terms of the output frame
  // indexes after subsampling.
  int32 current_log_post_subsampled_offset_;
};

} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_NNET_CCTC_DECODABLE_SIMPLE_H_

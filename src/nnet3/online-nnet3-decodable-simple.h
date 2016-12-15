// nnet3/online-nnet3-decodable-simple.h

// Copyright  2014  Johns Hopkins Universithy (author: Daniel Povey)
//            2016  Api.ai (Author: Ilya Platonov)


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

#ifndef KALDI_NNET3_ONLINE_NNET3_DECODABLE_H_
#define KALDI_NNET3_ONLINE_NNET3_DECODABLE_H_

#include "itf/online-feature-itf.h"
#include "itf/decodable-itf.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace nnet3 {

// Note: see also nnet-compute-online.h, which provides a different
// (lower-level) interface and more efficient for progressive evaluation of an
// nnet throughout an utterance, with re-use of already-computed activations.

struct DecodableNnet3OnlineOptions {
  int32 frame_subsampling_factor;
  BaseFloat acoustic_scale;
  bool pad_input;
  int32 max_nnet_batch_size;
  NnetComputeOptions compute_config;
  NnetOptimizeOptions optimize_config;

  DecodableNnet3OnlineOptions():
      frame_subsampling_factor(1),
      acoustic_scale(0.1),
      pad_input(true),
      max_nnet_batch_size(256) { }

  void Register(OptionsItf *opts) {
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic likelihoods");
    opts->Register("pad-input", &pad_input,
                   "If true, pad acoustic features with required acoustic context "
                   "past edges of file.");
    opts->Register("max-nnet-batch-size", &max_nnet_batch_size,
                   "Maximum batch size we use in neural-network decodable object, "
                   "in cases where we are not constrained by currently available "
                   "frames (this will rarely make a difference)");

    opts->Register("frame-subsampling-factor", &frame_subsampling_factor,
                   "Required if the frame-rate of the output (e.g. in 'chain' "
                   "models) is less than the frame-rate of the original "
                   "alignment.");

    // register the optimization options with the prefix "optimization".
    ParseOptions optimization_opts("optimization", opts);
    optimize_config.Register(&optimization_opts);

    // register the compute options with the prefix "computation".
    ParseOptions compute_opts("computation", opts);
    compute_config.Register(&compute_opts);

  }
};


/**
   This Decodable object for class nnet3::AmNnetSimple takes feature input from class
   OnlineFeatureInterface, unlike, say, class DecodableAmNnet which takes
   feature input from a matrix.
*/

class DecodableNnet3SimpleOnline: public DecodableInterface {
 public:
  DecodableNnet3SimpleOnline(const AmNnetSimple &am_nnet,
                             const TransitionModel &trans_model,
                             const DecodableNnet3OnlineOptions &opts,
                             OnlineFeatureInterface *input_feats);


  /// Returns the scaled log likelihood
  virtual BaseFloat LogLikelihood(int32 frame, int32 index);

  virtual bool IsLastFrame(int32 frame) const;

  virtual int32 NumFramesReady() const;

  /// Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  int32 FrameSubsamplingFactor() const { return opts_.frame_subsampling_factor; }
 private:

  /// If the neural-network outputs for this frame are not cached, it computes
  /// them (and possibly for some succeeding frames)
  void ComputeForFrame(int32 frame);
  // corrects number of frames by frame_subsampling_factor;
  int32 NumSubsampledFrames(int32) const;

  void DoNnetComputation(
      int32 input_t_start,
      const MatrixBase<BaseFloat> &input_feats,
      const VectorBase<BaseFloat> &ivector,
      int32 output_t_start,
      int32 num_subsampled_frames);

  CachingOptimizingCompiler compiler_;

  OnlineFeatureInterface *features_;
  const AmNnetSimple &am_nnet_;
  const TransitionModel &trans_model_;
  DecodableNnet3OnlineOptions opts_;
  CuVector<BaseFloat> log_priors_;  // log-priors taken from the model.
  int32 feat_dim_;  // dimensionality of the input features.
  int32 left_context_;  // Left context of the network (cached here)
  int32 right_context_;  // Right context of the network (cached here)
  int32 num_pdfs_;  // Number of pdfs, equals output-dim of the network (cached
                    // here)

  int32 begin_frame_;  // First frame for which scaled_loglikes_ is valid
                       // (i.e. the first frame of the batch of frames for
                       // which we've computed the output).

  // scaled_loglikes_ contains the neural network pseudo-likelihoods: the log of
  // (prob divided by the prior), scaled by opts.acoustic_scale).  We may
  // compute this using the GPU, but we transfer it back to the system memory
  // when we store it here.  These scores are only kept for a subset of frames,
  // starting at begin_frame_, whose length depends how many frames were ready
  // at the time we called LogLikelihood(), and will never exceed
  // opts_.max_nnet_batch_size.
  Matrix<BaseFloat> scaled_loglikes_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnet3SimpleOnline);
};

} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_ONLINE_NNET3_DECODABLE_H_

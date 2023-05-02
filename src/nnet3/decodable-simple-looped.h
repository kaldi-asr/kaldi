// nnet3/decodable-simple-looped.h

// Copyright 2016 Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_DECODABLE_SIMPLE_LOOPED_H_
#define KALDI_NNET3_DECODABLE_SIMPLE_LOOPED_H_

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

// See also nnet-am-decodable-simple.h, which is a decodable object that's based
// on breaking up the input into fixed chunks.  The decodable object defined here is based on
// 'looped' computations, which naturally handle infinite left-context (but are
// only ideal for systems that have only recurrence in the forward direction,
// i.e. not BLSTMs... because there isn't a natural way to enforce extra right
// context for each chunk.)


// Note: the 'simple' in the name means it applies to networks for which
// IsSimpleNnet(nnet) would return true.  'looped' means we use looped
// computations, with a kGotoLabel statement at the end of it.
struct NnetSimpleLoopedComputationOptions {
  int32 extra_left_context_initial;
  int32 frame_subsampling_factor;
  int32 frames_per_chunk;
  BaseFloat acoustic_scale;
  bool debug_computation;
  NnetOptimizeOptions optimize_config;
  NnetComputeOptions compute_config;
  NnetSimpleLoopedComputationOptions():
      extra_left_context_initial(0),
      frame_subsampling_factor(1),
      frames_per_chunk(24),
      acoustic_scale(0.1),
      debug_computation(false) { }

  void Check() const {
    KALDI_ASSERT(extra_left_context_initial >= 0 &&
                 frame_subsampling_factor > 0 && frames_per_chunk > 0 &&
                 acoustic_scale > 0.0);
  }

  void Register(OptionsItf *opts) {
    opts->Register("extra-left-context-initial", &extra_left_context_initial,
                   "Extra left context to use at the first frame of an utterance (note: "
                   "this will just consist of repeats of the first frame, and should not "
                   "usually be necessary.");
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor,
                   "Required if the frame-rate of the output (e.g. in 'chain' "
                   "models) is less than the frame-rate of the original "
                   "alignment.");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic log-likelihoods");
    opts->Register("frames-per-chunk", &frames_per_chunk,
                   "Number of frames in each chunk that is separately evaluated "
                   "by the neural net.  Measured before any subsampling, if the "
                   "--frame-subsampling-factor options is used (i.e. counts "
                   "input frames.  This is only advisory (may be rounded up "
                   "if needed.");
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


/**
   When you instantiate class DecodableNnetSimpleLooped, you should give it
   a const reference to this class, that has been previously initialized.
 */
class DecodableNnetSimpleLoopedInfo  {
 public:
  // The constructor takes a non-const pointer to 'nnet' because it may have to
  // modify it to be able to take multiple iVectors.
  DecodableNnetSimpleLoopedInfo(const NnetSimpleLoopedComputationOptions &opts,
                                Nnet *nnet);

  // This constructor takes the priors from class AmNnetSimple (so it can divide by
  // them).
  DecodableNnetSimpleLoopedInfo(const NnetSimpleLoopedComputationOptions &opts,
                                AmNnetSimple *nnet);

  // this constructor is for use in testing.
  DecodableNnetSimpleLoopedInfo(const NnetSimpleLoopedComputationOptions &opts,
                                const Vector<BaseFloat> &priors,
                                Nnet *nnet);

  void Init(const NnetSimpleLoopedComputationOptions &opts,
            Nnet *nnet);

  const NnetSimpleLoopedComputationOptions &opts;

  const Nnet &nnet;

  // the log priors (or the empty vector if the priors are not set in the model)
  CuVector<BaseFloat> log_priors;


  // frames_left_context equals the model left context plus the value of the
  // --extra-left-context-initial option.
  int32 frames_left_context;
  // frames_right_context is the same as the right-context of the model.
  int32 frames_right_context;
  // The frames_per_chunk_ equals the number of input frames we need for each
  // chunk (except for the first chunk).  This divided by
  // opts_.frame_subsampling_factor gives the number of output frames.
  int32 frames_per_chunk;

  // The output dimension of the neural network.
  int32 output_dim;

  // True if the neural net accepts iVectors.  If so, the neural net will have been modified
  // to accept the iVectors
  bool has_ivectors;

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

  It can accept just input features, or input features plus iVectors.  */
class DecodableNnetSimpleLooped {
 public:
  /**
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.

     @param [in] info   This helper class contains all the static pre-computed information
                        this class needs, and contains a pointer to the neural net.
     @param [in] feats  The input feature matrix.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
  */
  DecodableNnetSimpleLooped(const DecodableNnetSimpleLoopedInfo &info,
                            const MatrixBase<BaseFloat> &feats,
                            const VectorBase<BaseFloat> *ivector = NULL,
                            const MatrixBase<BaseFloat> *online_ivectors = NULL,
                            int32 online_ivector_period = 1);


  // returns the number of frames of likelihoods.  The same as feats_.NumRows()
  // in the normal case (but may be less if opts_.frame_subsampling_factor !=
  // 1).
  inline int32 NumFrames() const { return num_subsampled_frames_; }

  inline int32 OutputDim() const { return info_.output_dim; }

  // Gets the output for a particular frame, with 0 <= frame < NumFrames().
  // 'output' must be correctly sized (with dimension OutputDim()).  Note:
  // you're expected to call this, and GetOutput(), in an order of increasing
  // frames.  If you deviate from this, one of these calls may crash.
  void GetOutputForFrame(int32 subsampled_frame,
                         VectorBase<BaseFloat> *output);

  // Gets the output for a particular frame and pdf_id, with
  // 0 <= subsampled_frame < NumFrames(),
  // and 0 <= pdf_id < OutputDim().
  inline BaseFloat GetOutput(int32 subsampled_frame, int32 pdf_id) {
    KALDI_ASSERT(subsampled_frame >= current_log_post_subsampled_offset_ &&
                 "Frames must be accessed in order.");
    while (subsampled_frame >= current_log_post_subsampled_offset_ +
                            current_log_post_.NumRows())
      AdvanceChunk();
    return current_log_post_(subsampled_frame -
                             current_log_post_subsampled_offset_,
                             pdf_id);
  }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnetSimpleLooped);

  // This function does the computation for the next chunk.
  void AdvanceChunk();

  void AdvanceChunkInternal(const MatrixBase<BaseFloat> &input_feats,
                            const VectorBase<BaseFloat> &ivector);

  // Gets the iVector for the specified frame., if we are
  // using iVectors (else does nothing).
  void GetCurrentIvector(int32 input_frame,
                         Vector<BaseFloat> *ivector);

  // returns dimension of the provided iVectors if supplied, or 0 otherwise.
  int32 GetIvectorDim() const;

  const DecodableNnetSimpleLoopedInfo &info_;

  NnetComputer computer_;

  const MatrixBase<BaseFloat> &feats_;
  // note: num_subsampled_frames_ will equal feats_.NumRows() in the normal case
  // when opts_.frame_subsampling_factor == 1.
  int32 num_subsampled_frames_;

  // ivector_ is the iVector if we're using iVectors that are estimated in batch
  // mode.
  const VectorBase<BaseFloat> *ivector_;

  // online_ivector_feats_ is the iVectors if we're using online-estimated ones.
  const MatrixBase<BaseFloat> *online_ivector_feats_;
  // online_ivector_period_ helps us interpret online_ivector_feats_; it's the
  // number of frames the rows of ivector_feats are separated by.
  int32 online_ivector_period_;

  // The current log-posteriors that we got from the last time we
  // ran the computation.
  Matrix<BaseFloat> current_log_post_;

  // The number of chunks we have computed so far.
  int32 num_chunks_computed_;

  // The time-offset of the current log-posteriors, equals
  // (num_chunks_computed_ - 1) *
  //    (info_.frames_per_chunk_ / info_.opts_.frame_subsampling_factor).
  int32 current_log_post_subsampled_offset_;
};

class DecodableAmNnetSimpleLooped: public DecodableInterface {
 public:
  /**
     This constructor takes features as input, and you can either supply a
     single iVector input, estimated in batch-mode ('ivector'), or 'online'
     iVectors ('online_ivectors' and 'online_ivector_period', or none at all.
     Note: it stores references to all arguments to the constructor, so don't
     delete them till this goes out of scope.


     @param [in] info   This helper class contains all the static pre-computed information
                        this class needs, and contains a pointer to the neural net.  If
                        you want prior subtraction to be done, you should have initialized
                        this with the constructor that takes class AmNnetSimple.
     @param [in] trans_model  The transition model to use.  This takes care of the
                        mapping from transition-id (which is an arg to
                        LogLikelihood()) to pdf-id (which is used internally).
     @param [in] feats   A pointer to the input feature matrix; must be non-NULL.
                         We
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] ivector If you are using iVectors estimated in batch mode,
                         a pointer to the iVector, else NULL.
     @param [in] online_ivectors
                        If you are using iVectors estimated 'online'
                        a pointer to the iVectors, else NULL.
     @param [in] online_ivector_period If you are using iVectors estimated 'online'
                        (i.e. if online_ivectors != NULL) gives the periodicity
                        (in frames) with which the iVectors are estimated.
  */
  DecodableAmNnetSimpleLooped(const DecodableNnetSimpleLoopedInfo &info,
                              const TransitionModel &trans_model,
                              const MatrixBase<BaseFloat> &feats,
                              const VectorBase<BaseFloat> *ivector = NULL,
                              const MatrixBase<BaseFloat> *online_ivectors = NULL,
                              int32 online_ivector_period = 1);


  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual inline int32 NumFramesReady() const {
    return decodable_nnet_.NumFrames();
  }

  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetSimpleLooped);
  DecodableNnetSimpleLooped decodable_nnet_;
  const TransitionModel &trans_model_;
};



} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_DECODABLE_SIMPLE_LOOPED_H_

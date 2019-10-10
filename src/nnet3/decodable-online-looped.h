// nnet3/decodable-online-looped.h

// Copyright  2014-2017  Johns Hopkins Universithy (author: Daniel Povey)
//                 2016  Api.ai (Author: Ilya Platonov)


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

#ifndef KALDI_NNET3_DECODABLE_ONLINE_LOOPED_H_
#define KALDI_NNET3_DECODABLE_ONLINE_LOOPED_H_

#include "itf/online-feature-itf.h"
#include "itf/decodable-itf.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/decodable-simple-looped.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace nnet3 {


// The Decodable objects that we define in this header do the neural net
// computation in a way that's compatible with online feature extraction.  It
// differs from the one declared in online-nnet3-decodable-simple.h because it
// uses the 'looped' network evaluation, which is more efficient because it
// re-uses hidden activations (and therefore doesn't have to pad chunks of data
// with extra left-context); it is applicable to TDNNs and to forwards-recurrent
// topologies like LSTMs, but not tobackwards-recurrent topologies such as
// BLSTMs.

// The options are passed in the same way as in decodable-simple-looped.h,
// we use the same options and info class.


// This object is used as a base class for DecodableNnetLoopedOnline
// and DecodableAmNnetLoopedOnline.
// It takes care of the neural net computation and computations related to how
// many frames are ready (etc.), but it does not override the LogLikelihood() or
// NumIndices() functions so it is not usable as an object of type
// DecodableInterface.
class DecodableNnetLoopedOnlineBase: public DecodableInterface {
 public:
  // Constructor.  'input_feature' is for the feature that will be given
  // as 'input' to the neural network; 'ivector_feature' is for the iVector
  // feature, or NULL if iVectors are not being used.
  DecodableNnetLoopedOnlineBase(const DecodableNnetSimpleLoopedInfo &info,
                                 OnlineFeatureInterface *input_features,
                                 OnlineFeatureInterface *ivector_features);

  // note: the LogLikelihood function is not overridden; the child
  // class needs to do this.
  //virtual BaseFloat LogLikelihood(int32 subsampled_frame, int32 index);

  // note: the frame argument is on the output of the network, i.e. after any
  // subsampling, so we call it 'subsampled_frame'.
  virtual bool IsLastFrame(int32 subsampled_frame) const;

  virtual int32 NumFramesReady() const;

  // Note: this function, present in the base-class, is overridden by the child class.
  // virtual int32 NumIndices() const;

  // this is not part of the standard Decodable interface but I think is needed for
  // something.
  int32 FrameSubsamplingFactor() const {
    return info_.opts.frame_subsampling_factor;
  }

  /// Sets the frame offset value. Frame offset is initialized to 0 when the
  /// decodable object is constructed and stays as 0 unless this method is
  /// called. This method is useful when we want to reset the decoder state,
  /// i.e. call decoder.InitDecoding(), but we want to keep using the same
  /// decodable object, e.g. in case of an endpoint. The frame offset affects
  /// the behavior of IsLastFrame(), NumFramesReady() and LogLikelihood()
  /// methods.
  void SetFrameOffset(int32 frame_offset);

  /// Returns the frame offset value.
  int32 GetFrameOffset() const { return frame_offset_; }

 protected:

  /// If the neural-network outputs for this frame are not cached, this function
  /// computes them (and possibly also some later frames).  Note:
  /// the frame-index is called 'subsampled_frame' because if frame-subsampling-factor
  /// is not 1, it's an index that is "after subsampling", i.e. it changes more
  /// slowly than the input-feature index.
  inline void EnsureFrameIsComputed(int32 subsampled_frame) {
    KALDI_ASSERT(subsampled_frame >= current_log_post_subsampled_offset_ &&
                 "Frames must be accessed in order.");
    while (subsampled_frame >= current_log_post_subsampled_offset_ +
           current_log_post_.NumRows())
      AdvanceChunk();
  }

  // The current log-posteriors that we got from the last time we
  // ran the computation.
  Matrix<BaseFloat> current_log_post_;

  // The number of chunks we have computed so far.
  int32 num_chunks_computed_;

  // The time-offset of the current log-posteriors, equals
  // (num_chunks_computed_ - 1) *
  //    (info_.frames_per_chunk_ / info_.opts_.frame_subsampling_factor).
  int32 current_log_post_subsampled_offset_;

  const DecodableNnetSimpleLoopedInfo &info_;

  // IsLastFrame(), NumFramesReady() and LogLikelihood() methods take into
  // account this offset value. We initialize frame_offset_ as 0 and it stays as
  // 0 unless SetFrameOffset() method is called.
  int32 frame_offset_;

 private:

  // This function does the computation for the next chunk.  It will change
  // current_log_post_ and current_log_post_subsampled_offset_, and
  // increment num_chunks_computed_.
  void AdvanceChunk();

  OnlineFeatureInterface *input_features_;
  OnlineFeatureInterface *ivector_features_;

  NnetComputer computer_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnetLoopedOnlineBase);
};

// This decodable object takes indexes of the form (pdf_id + 1),
// or whatever the output-dimension of the neural network represents,
// plus one.
// It fully implements DecodableInterface.
// Note: whether or not division by the prior takes place depends on
// whether you supplied class AmNnetSimple (or just Nnet), to the constructor
// of the DecodableNnetSimpleLoopedInfo that you initialized this
// with.
class DecodableNnetLoopedOnline: public DecodableNnetLoopedOnlineBase {
 public:
  DecodableNnetLoopedOnline(
      const DecodableNnetSimpleLoopedInfo &info,
      OnlineFeatureInterface *input_features,
      OnlineFeatureInterface *ivector_features):
      DecodableNnetLoopedOnlineBase(info, input_features, ivector_features) { }


  // returns the output-dim of the neural net.
  virtual int32 NumIndices() const { return info_.output_dim; }

  // 'subsampled_frame' is a frame, but if frame-subsampling-factor != 1, it's a
  // reduced-rate output frame (e.g. a 't' index divided by 3).  'index'
  // represents the pdf-id (or other output of the network) PLUS ONE.
  virtual BaseFloat LogLikelihood(int32 subsampled_frame, int32 index);

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnetLoopedOnline);

};


// This is for traditional decoding where the graph has transition-ids
// on the arcs, and you need the TransitionModel to map those to
// pdf-ids.
// Note: whether or not division by the prior takes place depends on
// whether you supplied class AmNnetSimple (or just Nnet), to the constructor
// of the DecodableNnetSimpleLoopedInfo that you initialized this
// with.
class DecodableAmNnetLoopedOnline: public DecodableNnetLoopedOnlineBase {
 public:
  DecodableAmNnetLoopedOnline(
      const TransitionModel &trans_model,
      const DecodableNnetSimpleLoopedInfo &info,
      OnlineFeatureInterface *input_features,
      OnlineFeatureInterface *ivector_features):
      DecodableNnetLoopedOnlineBase(info, input_features, ivector_features),
      trans_model_(trans_model) { }


  // returns the output-dim of the neural net.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  // 'subsampled_frame' is a frame, but if frame-subsampling-factor != 1, it's a
  // reduced-rate output frame (e.g. a 't' index divided by 3).
  virtual BaseFloat LogLikelihood(int32 subsampled_frame,
                                  int32 transition_id);

 private:
  const TransitionModel &trans_model_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnetLoopedOnline);

};




} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_DECODABLE_ONLINE_LOOPED_H_

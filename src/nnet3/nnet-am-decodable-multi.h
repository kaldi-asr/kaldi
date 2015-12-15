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

#ifndef KALDI_NNET3_NNET_AM_DECODABLE_MULTI_H_
#define KALDI_NNET3_NNET_AM_DECODABLE_MULTI_H_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/am-nnet-multi.h"
#include "nnet3/nnet-am-decodable-simple.h"

namespace kaldi {
namespace nnet3 {

/* DecodableAmNnetMulti is a decodable object that decodes with a neural net
   acoustic model of type AmNnetMulti.  It can accept just input features, or
   input features plus iVectors.
*/
class DecodableAmNnetMulti: public DecodableInterface {
 public:
  /// Constructor that just takes the features as input, but can also optionally
  /// take batch-mode or online iVectors.  Note: it stores references to all
  /// arguments to the constructor, so don't delete them till this goes out of
  /// scope.

  DecodableAmNnetMulti(int32 num_outputs,
                        const DecodableAmNnetSimpleOptions &opts,
                        const TransitionModel &trans_model,
                        const unordered_map<int32, std::vector<int32> > mapping,
                        const AmNnetMulti &am_nnet,
                        const MatrixBase<BaseFloat> &feats,
                        const VectorBase<BaseFloat> *ivector = NULL,
                        const MatrixBase<BaseFloat> *online_ivectors = NULL,
                        int32 online_ivector_period = 1,
                        BaseFloat exp_weight = 0.0);

  /// Constructor that also accepts iVectors estimated online;
  /// online_ivector_period is the time spacing between rows of the matrix.
  DecodableAmNnetMulti(int32 num_outputs,
                        const DecodableAmNnetSimpleOptions &opts,
                        const TransitionModel &trans_model,
                        const unordered_map<int32, std::vector<int32> > mapping,
                        const AmNnetMulti &am_nnet,
                        const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> &online_ivectors,
                        int32 online_ivector_period,
                        BaseFloat exp_weight = 0.0);

  /// Constructor that accepts iVectors estimated in batch mode
  DecodableAmNnetMulti(int32 num_outputs,
                        const DecodableAmNnetSimpleOptions &opts,
                        const TransitionModel &trans_model,
                        const unordered_map<int32, std::vector<int32> > mapping,
                        const AmNnetMulti &am_nnet,
                        const MatrixBase<BaseFloat> &feats,
                        const VectorBase<BaseFloat> &ivector,
                        BaseFloat exp_weight = 0.0);


  // Note, frames are numbered from zero.  But transition_id is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  virtual int32 NumFramesReady() const { return feats_.NumRows(); }

  // Note: these indices are one-based!  This is for compatibility with OpenFst.
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

  int32 num_outputs_;
  const DecodableAmNnetSimpleOptions &opts_;
  const TransitionModel &trans_model_;
  unordered_map<int32, vector<int32> > mapping_;

  const AmNnetMulti &am_nnet_;
  vector<CuVector<BaseFloat> > priors_vec_;
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
  std::vector<Matrix<BaseFloat> > current_log_post_vec_;
  // The time-offset of the current log-posteriors.
  int32 current_log_post_offset_;

  BaseFloat exp_weight_;
};

} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_

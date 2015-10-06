// nnet3/nnet-am-decodable-simple.h

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

#ifndef KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_
#define KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet3/nnet-simple-computer.h"

namespace kaldi {
namespace nnet3 {


struct DecodableAmNnetSimpleOptions : public NnetSimpleComputerOptions {
  BaseFloat acoustic_scale;
  NnetSimpleComputerOptions simple_computer_opts;

  DecodableAmNnetSimpleOptions():
      acoustic_scale(0.1) {}

  void Register(OptionsItf *opts) {
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic log-likelihoods");
  
    simple_computer_opts.Register(opts);

  }
};

/* DecodableAmNnetSimple is a decodable object that decodes with a neural net
   acoustic model of type AmNnetSimple.  It can accept just input features, or
   input features plus iVectors.
   It inherits from the NnetSimpleComputer class, which does the 
   neural network computation.
*/
class DecodableAmNnetSimple: public DecodableInterface, 
                             public NnetSimpleComputer {
 public:
  /// Constructor that just takes the features as input, but can also optionally
  /// take batch-mode or online iVectors.  Note: it stores references to all
  /// arguments to the constructor, so don't delete them till this goes out of
  /// scope.
  DecodableAmNnetSimple(const DecodableAmNnetSimpleOptions &opts,
                        const TransitionModel &trans_model,
                        const AmNnetSimple &am_nnet,
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

  // Note: these indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 private: 

  void DoNnetComputation(int32 input_t_start,
                         const MatrixBase<BaseFloat> &input_feats,
                         const VectorBase<BaseFloat> &ivector,
                         int32 output_t_start,
                         int32 num_output_frames);

  const DecodableAmNnetSimpleOptions &opts_;
  const TransitionModel &trans_model_;
  const AmNnetSimple &am_nnet_;
  CuVector<BaseFloat> priors_;

};

} // namespace nnet3
} // namespace kaldi

#endif  // KALDI_NNET3_NNET_AM_DECODABLE_SIMPLE_H_

// chain/chain-numerator.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_NUMERATOR_H_
#define KALDI_CHAIN_CHAIN_NUMERATOR_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "chain/chain-supervision.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace ctc {

// This class is responsible for the forward-backward of the 'supervision'
// (numerator) FST.
//
// note: the supervision.weight is ignored by this class, you have to apply
// it externally.
// Because the supervision FSTs are quite skinny, i.e. have very few paths for
// each frame, it's feasible to do this computation on the CPU, and that's what
// we do.  We transfer from/to the GPU only the things that we need.

class SupervisionForwardBackward {

 public:

  /// Initialize the objcect.  Note: we expect the 'nnet_output' to have the
  /// same number of rows as supervision.num_frames * supervision.num_sequences,
  /// and the same number of columns as the 'label-dim' of the supervision
  /// object (which will be the NumPdfs() of the transition model); but the
  /// ordering of the rows of 'nnet_output' is not the same as the ordering of
  /// frames in paths in the 'supervision' object (which has all frames of the
  /// 1st sequence first, then the 2nd sequence, and so on).  Instead, the
  /// frames in 'nnet_output' are ordered as: first the first frame of each
  /// sequence, then the second frame of each sequence, and so on.  This is more
  /// convenient both because the nnet3 code internally orders them that way,
  /// and because this makes it easier to order things in the way that class
  /// SingleHmmForwardBackward needs (we can just transpose, instead of doing a
  /// 3d tensor rearrangement).
  SupervisionForwardBackward(const Supervision &supervision,
                             const CuMatrixBase<BaseFloat> &nnet_output);

  // TODO: we could enable a Viterbi mode.

  // Does the forward computation.  Returns the total log-prob.
  BaseFloat Forward();

  // Does the backward computation and (efficiently) adds the direct part of the
  // derivative w.r.t. the neural network output to 'nnet_output_deriv'.
  void Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);


 private:

  const Supervision &supervision_;

  // the exp of the neural net output.
  const CuMatrixBase<BaseFloat> &nnet_output_;

  // 'fst_output_indexes' contains an entry for each arc in the supervision FST, in
  // the order you'd get them if you visit each arc of each state in order.
  // the contents of fst_output_indexes_ are indexes into nnet_output_indxes_.
  std::vector<int32> fst_output_indexes_;

  // numerator_indexes is a list of (row, column) indexes that we need to look
  // up in exp_nnet_output_ for the forward-backward computation.  The order is
  // not important, but indexes into this vector appear in .first members in
  // fst_output_indexes.
  CuArray<Int32Pair> nnet_output_indexes_;

  // the log-probs obtained from lookup in the nnet output, on the CPU.  This
  // vector has the same size as nnet_output_indexes_.  In the backward
  // computation, the storage is reused for derivatives.
  Vector<BaseFloat> nnet_logprobs_;

  // The log-alpha value (forward probability) for each state in the lattices.
  Vector<double> log_alpha_;

  // The total log-probability of the supervision, from the forward-backward
  // (you can interpret this as the posterior of this phone-sequence, after
  // adding in extra_log_prob_).
  double tot_log_prob_;

  // this is an extra term that gets added to tot_log_prob_prob_; it is the sum
  // over the individual sequences, of the negative log of the number of
  // history-states active on the first frame of that sequence (the idea being,
  // that we distribute the initial-probs evenly among those history-states on
  // those frames).
  double extra_log_prob_;

  // The log-beta value (backward probability) for each state in the lattice
  Vector<double> log_beta_;

  // This function, called from Forward(), creates fst_indexes_,
  // numerator_indexes_ and denominator_indexes_.
  // first_frame_alpha, if non-NULL, is where we write some info about
  // the first-frame's alpha probabilities for the sequences; see
  // the documentation for Forward() for more explanation.
  void ComputeLookupIndexes();

  // This function, called from Forward(), computes nnet_logprobs_ via batch
  // lookup operations in exp_nnet_output_.
  void LookUpLikelihoods();

  // This function, called from Forward(), does the actual forward-computation on
  // the FST, setting alpha_ and tot_log_prob_.
  void ComputeAlpha();

  // Computes the beta probabilities (called from Backward().)
  void ComputeBeta();

  // Computes derivatives (called from Backward()).
  // Returns true on success, false if a NaN or Inf was detected.
  void ComputeDerivatives(CuMatrixBase<BaseFloat> *nnet_output_deriv);


};

// This is a wrapping layer for both SupervisionForwardbackward and
// SingleHmmForwardBackward; it does the parts that both share, so that we can
// avoid duplication.
class ChainCommonComputation {
 public:
  ChainCommonComputation(const Hmm &hmm,
                        const CuMatrix<BaseFloat> &cu_weights,
                        const CctcSupervision &supervision,
                        int32 num_sequences,
                        const CuMatrixBase<BaseFloat> &nnet_output);


  // Does the forward part of the computation
  // the objf parts should be added together to get the real objf (including
  // the weighting factor in supervision.weight), and then
  // divided by the denominator (== num-frames * weight) for reporting purposes.
  // Note: negative_objf_part is the likelihood from the CctcNegativeComputation object
  // times -opts_.denominator_scale times supervision.weight.
  void Forward(BaseFloat *positive_objf_part, BaseFloat *negative_objf_part,
               BaseFloat *objf_denominator);

  // Does the backward part of the computation; outputs the derivative to 'nnet_output_deriv'.
  void Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);

  ~CctcCommonComputation();
 private:
  // This function, called from the constructor, checks various dimensions.
  void CheckDims() const;

  const CctcTrainingOptions &opts_;
  const CctcTransitionModel &trans_model_;
  const CctcHmm &hmm_;

  // cu_weights_ is derived from trans_model_.  Dimension is
  // trans_model_.NumHistoryStates() by trans_model_.NumOutputIndexes().
  const CuMatrix<BaseFloat> &cu_weights_;

  // vector, of dimension trans_model_.NumHistoryStates() by num_sequences_, of
  // alphas that we can use on the first frame in the negative computation
  // (taken from the positive computation).
  CuVector<BaseFloat> first_frame_alphas_;

  // The supervision object
  const CctcSupervision &supervision_;
  // The number of separate time-sequences that the supervision object covers,
  // which must all be of the same lengty.  This info has to be computed at the
  // nnet3 level of the code.
  int32 num_sequences_;
  // The neural net output
  const CuMatrixBase<BaseFloat> &nnet_output_;
  // the exponent of the neural net output.
  CuMatrix<BaseFloat> exp_nnet_output_;

  // the denominators, of dimension nnet_output_.NumRows() by
  // trans_model_.NumHistoryStates(), equal to exp_nnet_output_ * weights_'.
  // Equal to exp_nnet_output_ * cu_weights_'.
  CuMatrix<BaseFloat> denominators_;

  // used to store the derivative of the objf w.r.t. the log-denominators and w.r.t. the
  // denominators, at different times.
  CuMatrix<BaseFloat> denominators_deriv_;

  CctcPositiveComputation *positive_computation_;

  CctcNegativeComputation *negative_computation_;


};

}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_NUMERATOR_H_


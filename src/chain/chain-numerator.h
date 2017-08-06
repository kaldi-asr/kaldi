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
namespace chain {


// This class is responsible for the forward-backward of the 'supervision'
// (numerator) FST.
//
// note: the supervision.weight is ignored by this class, you have to apply
// it externally.
// Because the supervision FSTs are quite skinny, i.e. have very few paths for
// each frame, it's feasible to do this computation on the CPU, and that's what
// we do.  We transfer from/to the GPU only the things that we need.

class NumeratorComputation {

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
  NumeratorComputation(const Supervision &supervision,
                       const CuMatrixBase<BaseFloat> &nnet_output);

  // TODO: we could enable a Viterbi mode.

  // Does the forward computation.  Returns the total log-prob multiplied
  // by supervision_.weight.
  BaseFloat Forward();

  // Does the backward computation and (efficiently) adds the derivative of the
  // nnet output w.r.t. the (log-prob times supervision_.weight times
  // deriv_weight) to 'nnet_output_deriv'.
  void Backward(CuMatrixBase<BaseFloat> *nnet_output_deriv);

 private:

  const Supervision &supervision_;

  // state times of supervision_.fst.
  std::vector<int32> fst_state_times_;


  // the exp of the neural net output.
  const CuMatrixBase<BaseFloat> &nnet_output_;


  // 'fst_output_indexes' contains an entry for each arc in the supervision FST, in
  // the order you'd get them if you visit each arc of each state in order.
  // the contents of fst_output_indexes_ are indexes into nnet_output_indexes_
  // and nnet_logprobs_.
  std::vector<int32> fst_output_indexes_;

  // nnet_output_indexes is a list of (row, column) indexes that we need to look
  // up in nnet_output_ for the forward-backward computation.  The order is
  // arbitrary, but indexes into this vector appear in fst_output_indexes;
  // and it's important that each pair only appear once (in order for the
  // derivatives to be summed properly).
  CuArray<Int32Pair> nnet_output_indexes_;

  // the log-probs obtained from lookup in the nnet output, on the CPU.  This
  // vector has the same size as nnet_output_indexes_.  In the backward
  // computation, the storage is re-used for derivatives.
  Vector<BaseFloat> nnet_logprobs_;

  // derivatives w.r.t. the nnet logprobs.  These can be interpreted as
  // occupation probabilities.
  Vector<BaseFloat> nnet_logprob_derivs_;

  // The log-alpha value (forward probability) for each state in the lattices.
  Vector<double> log_alpha_;

  // The total pseudo-log-likelihood from the forward-backward.
  double tot_log_prob_;

  // The log-beta value (backward probability) for each state in the lattice
  Vector<double> log_beta_;

  // This function creates fst_output_indexes_ and nnet_output_indexes_.
  void ComputeLookupIndexes();

  // convert time-index in the FST to a row-index in the nnet-output (to account
  // for the fact that the sequences are interleaved in the nnet-output).
  inline int32 ComputeRowIndex(int32 t, int32 frames_per_sequence,
                               int32 num_sequences) {
    return t / frames_per_sequence +
        num_sequences * (t % frames_per_sequence);
  }

};




}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_NUMERATOR_H_


// ctc/cctc-tombstone.h

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


#ifndef KALDI_CTC_CCTC_TOMBSTONE_H_
#define KALDI_CTC_CCTC_TOMBSTONE_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "tree/context-dep.h"
#include "lat/kaldi-lattice.h"
#include "matrix/kaldi-matrix.h"
#include "ctc/language-model.h"
#include "ctc/cctc-transition-model.h"
#include "ctc/cctc-supervision.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace ctc {

// This function is used inside RearrangeNnetOutput and
// RearrangeNnetOutputReverse; it copies 3d tensors.
// note: we expect MatrixIndexT to equal int32.
// It does: for x=0..xdim-1, for y=0..ydim-1, for z=0..zdim-1,
//  dst[x*dst_xstride + y*dst_ystride+z*dst_zstride] =
//  src[x*src_xstride + y*src_ystride+z*src_zstride];
template <typename Real>
Tensor3dCopy(int32 xdim, int32 ydim, int32 zdim,
             int32 src_xstride, int32 src_ystride, int32 src_zstride,
             int32 dst_xstride, int32 dst_ystride, int32 dst_zstride,
             const Real *src, Real *dst);

// might also need this:
//template <typename Real>
//Tensor3dAdd(int32 xdim, int32 ydim, int32 zdim,
//            int32 src_xstride, int32 src_ystride, int32 src_zstride,
//            int32 dst_xstride, int32 dst_ystride, int32 dst_zstride,
//            const Real *src, Real *dst);


/**
 Rearranges neural net output from an input with
     num-rows = num-sequences * num-time-steps  [arranged first by sequence-index]
     num-cols = nnet-output-dim
 to an output 'nnet_output_rearranged' which has:
     num-rows = num-time-steps
     num-cols = nnet-output-dim * num-sequences [arranged first by nnet-output-index].

 The num-time-steps, num-sequences and nnet-output-dim are inferred from the
 dimensions of the matrices.  Note that this same function is used for the
 denominator indexes where we have num-history-states instead of
 nnet-output-dim, but the interface is the same.  */
void RearrangeNnetOutput(const CuMatrixBase<BaseFloat> &nnet_output
                         CuMatrixBase<BaseFloat> *nnet_output_rearranged);

/**
   This function does the opposite rearrangement to the one done in
   RearrangeNnetOutput.
*/
void RearrangeNnetOutputRevers(const CuMatrixBase<BaseFloat> &nnet_output_rearranged,
                               CuMatrixBase<BaseFloat> *nnet_output);


// This header relates to the 'tombstone' extension of CTC, and contains
// utilities for efficient forward-backward over the entire model
// (all word sequences), which becomes a negated term in the objective
// function (like the denominator lattice in MMI training).

// This class is supposed to be initialized just once and then used repeatedly,
// as it needs to do some startup work.  This class supports both CPU and GPU
// versions of the computation, and it uses the GPU is you have initialized the
// device (however, the CPU one will be very slow).

class CctcNegativeComputation {
  // note: num_sequences is the number of egs that have been combined into a
  // single eg (which must all be of the same size), which will be the number of
  // distinct values of 'n' in the output indexes.  All must have the same
  // number of frames, and we assume that we're sorted first on n and then on t,
  // since that's the way the positive computation requires them to be.
  CctcNegativeComputation(const CctcTransitionModel &trans_model,
                          const CuMatrix<BaseFloat> &cu_weights,
                          const CuMatrixBase<BaseFloat> &exp_nnet_output,
                          const CuMatrixBase<BaseFloat> &denominators,
                          int32 num_sequences);
  void Forward(


 public:
 // the numerator-probs rearranged
 CuMatrix<BaseFloat> numerators_rearranged_;
 CuMatrix<BaseFloat> denominators_rearranged_;



  const CctcTransitionModel &trans_model_;
  // Derived from trans_model_.  Dimension is
  // trans_model_.NumHistoryStates() by trans_model_.NumOutputIndexes().
  const CuMatrix<BaseFloat> &cu_weights_;

};


}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_TRAINING_H_


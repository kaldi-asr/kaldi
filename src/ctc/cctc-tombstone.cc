// ctc/cctc-training.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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


#include "ctc/cctc-training.h"

namespace kaldi {
namespace ctc {


void RearrangeNnetOutputForward(
    const CuMatrixBase<BaseFloat> &nnet_output,
    CuMatrixBase<BaseFloat> *nnet_output_rearranged) {
  int32 num_time_steps = nnet_output_rearranged->NumRows(),
      nnet_output_dim = nnet_output.NumCols();
  KALDI_ASSERT(nnet_output.NumRows() % num_time_steps == 0);
  int32 num_sequences = nnet_output.NumRows() / num_time_steps;
  KALDI_ASSERT(nnet_output_rearranged->NumCols() ==
               nnet_output_dim * num_sequences);
  int32 xdim = num_time_steps,
      ydim = nnet_output_dim,
      zdim = num_sequences,
      src_xstride = 1,
      src_ystride = nnet_output.Stride(),
      src_zstride = num_time_steps,
      dest_xstride = nnet_output_rearranged->Stride(),
      dest_ystride = num_sequences,
      dest_zstride = 1;
  Tensor3dCopy(xdim, ydim, zdim,
               src_xstride, src_ystride, src_zstride,
               dest_xstride, dest_ystride, dest_zstride,
               nnet_output.Data(), nnet_output_rearranged->Data());
}

void RearrangeNnetOutputBackward(
    const CuMatrixBase<BaseFloat> &nnet_output_rearranged,
    CuMatrixBase<BaseFloat> *nnet_output) {
  int32 num_time_steps = nnet_output_rearranged.NumRows(),
      nnet_output_dim = nnet_output->NumCols();
  KALDI_ASSERT(nnet_output->NumRows() % num_time_steps == 0);
  int32 num_sequences = nnet_output->NumRows() / num_time_steps;
  KALDI_ASSERT(nnet_output_rearranged.NumCols() ==
               nnet_output_dim * num_sequences);
  int32 xdim = num_time_steps,
      ydim = nnet_output_dim,
      zdim = num_sequences,
      src_xstride = nnet_output_rearranged->Stride(),
      src_ystride = num_sequences,
      src_zstride = 1,
      dest_xstride = 1,
      dest_ystride = nnet_output.Stride(),
      dest_zstride = num_time_steps;
  Tensor3dCopy(xdim, ydim, zdim,
               src_xstride, src_ystride, src_zstride,
               dest_xstride, dest_ystride, dest_zstride,
               nnet_output_rearranged.Data(), nnet_output->Data());
}



}  // namespace ctc
}  // namespace kaldi

// cudadecoder/decodable-cumatrix.cc
/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 * Authors:  Hugo Braun, Justin Luitjens, Ryan Leary
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if HAVE_CUDA == 1

#include "decodable-cumatrix.h"

namespace kaldi {
namespace cuda_decoder {

DecodableCuMatrixMapped::DecodableCuMatrixMapped(
    const TransitionModel &tm, const CuMatrixBase<BaseFloat> &likes,
    int32 frame_offset)
    : trans_model_(tm), likes_(&likes), frame_offset_(frame_offset) {
  if (likes.NumCols() != tm.NumPdfs())
    KALDI_ERR << "Mismatch, matrix has " << likes.NumCols()
              << " rows but transition-model has " << tm.NumPdfs()
              << " pdf-ids.";
}

int32 DecodableCuMatrixMapped::NumFramesReady() const {
  return frame_offset_ + likes_->NumRows();
}

bool DecodableCuMatrixMapped::IsLastFrame(int32 frame) const {
  KALDI_ASSERT(frame < NumFramesReady());
  return (frame == NumFramesReady() - 1);
}

// Indices are one-based!  This is for compatibility with OpenFst.
int32 DecodableCuMatrixMapped::NumIndices() const {
  return trans_model_.NumTransitionIds();
}

// returns cuda pointer to nnet3 output
BaseFloat *
DecodableCuMatrixMapped::GetLogLikelihoodsCudaPointer(int32 subsampled_frame) {
  BaseFloat *frame_nnet3_out =
      (BaseFloat *)likes_->Data() +
      (subsampled_frame - frame_offset_) * likes_->Stride();
  return frame_nnet3_out;
};

}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // HAVE_CUDA == 1

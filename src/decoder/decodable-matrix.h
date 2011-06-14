// decoder/decodable-matrix.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_DECODER_DECODABLE_MATRIX_H_
#define KALDI_DECODER_DECODABLE_MATRIX_H_

#include <vector>

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"

namespace kaldi {


class DecodableMatrixScaledMapped: public DecodableInterface {
 public:
  DecodableMatrixScaledMapped(const TransitionModel &tm,
                              const Matrix<BaseFloat> &likes,
                              BaseFloat scale): trans_model_(tm), likes_(likes),
                                                scale_(scale) {
    if (likes.NumCols() != tm.NumPdfs())
      KALDI_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                << likes.NumCols() << " rows but transition-model has "
                << tm.NumPdfs() << " pdf-ids.";
  }

  virtual int32 NumFrames() { return likes_.NumRows(); }

  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_ * likes_(frame, trans_model_.TransitionIdToPdf(tid));
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }

 private:
  const TransitionModel &trans_model_;  // for tid to pdf mapping
  const Matrix<BaseFloat> &likes_;
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaledMapped);
};


class DecodableMatrixScaled: public DecodableInterface {
 public:
  DecodableMatrixScaled(const Matrix<BaseFloat> &likes,
                        BaseFloat scale): likes_(likes),
                                          scale_(scale) { }
  
  virtual int32 NumFrames() { return likes_.NumRows(); }
  
  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }
  
  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_ * likes_(frame, tid);
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return likes_.NumCols(); }

 private:
  const Matrix<BaseFloat> &likes_;
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaled);
};


}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_MATRIX_H_

// simplehmm/decodable-simple-hmm.h

// Copyright 2016  Vimal Manohar

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

#ifndef KALDI_SIMPLEHMM_DECODABLE_SIMPLE_HMM_H_
#define KALDI_SIMPLEHMM_DECODABLE_SIMPLE_HMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "simplehmm/simple-hmm.h"
#include "itf/decodable-itf.h"

namespace kaldi {
namespace simple_hmm {

class DecodableMatrixSimpleHmm: public DecodableInterface {
 public:
  // This constructor creates an object that will not delete "likes"
  // when done.
  DecodableMatrixSimpleHmm(const SimpleHmm &model,
                           const Matrix<BaseFloat> &likes,
                           BaseFloat scale):
    model_(model), likes_(&likes), scale_(scale), delete_likes_(false)
  {
    if (likes.NumCols() != model.NumPdfs())
      KALDI_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                << likes.NumCols() << " rows but transition-model has "
                << model.NumPdfs() << " pdf-ids.";
  }

  // This constructor creates an object that will delete "likes"
  // when done.
  DecodableMatrixSimpleHmm(const SimpleHmm &model,
                           BaseFloat scale,
                           const Matrix<BaseFloat> *likes):
      model_(model), likes_(likes), scale_(scale), delete_likes_(true) {
    if (likes->NumCols() != model.NumPdfs())
      KALDI_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                << likes->NumCols() << " rows but transition-model has "
                << model.NumPdfs() << " pdf-ids.";
  }

  virtual int32 NumFramesReady() const { return likes_->NumRows(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_ * (*likes_)(frame, model_.TransitionIdToPdfClass(tid));
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return model_.NumTransitionIds(); }

  virtual ~DecodableMatrixSimpleHmm() {
    if (delete_likes_) delete likes_;
  }
 private:
  const SimpleHmm &model_;  // for tid to pdf mapping
  const Matrix<BaseFloat> *likes_;
  BaseFloat scale_;
  bool delete_likes_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixSimpleHmm);
};

}  // namespace simple_hmm
}  // namespace kaldi

#endif  // KALDI_SIMPLEHMM_DECODABLE_SIMPLE_HMM_H_

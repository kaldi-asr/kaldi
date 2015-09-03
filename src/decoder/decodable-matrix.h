// decoder/decodable-matrix.h

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_DECODABLE_MATRIX_H_
#define KALDI_DECODER_DECODABLE_MATRIX_H_

#include <vector>

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"

namespace kaldi {


class DecodableMatrixScaledMapped: public DecodableInterface {
 public:
  // This constructor creates an object that will not delete "likes"
  // when done.
  DecodableMatrixScaledMapped(const TransitionModel &tm,
                              const Matrix<BaseFloat> &likes,
                              BaseFloat scale): trans_model_(tm), likes_(&likes),
                                                scale_(scale), delete_likes_(false) {
    if (likes.NumCols() != tm.NumPdfs())
      KALDI_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                << likes.NumCols() << " rows but transition-model has "
                << tm.NumPdfs() << " pdf-ids.";
  }

  // This constructor creates an object that will delete "likes"
  // when done.
  DecodableMatrixScaledMapped(const TransitionModel &tm,
                              BaseFloat scale,
                              const Matrix<BaseFloat> *likes):
      trans_model_(tm), likes_(likes),
      scale_(scale), delete_likes_(true) {
    if (likes->NumCols() != tm.NumPdfs())
      KALDI_ERR << "DecodableMatrixScaledMapped: mismatch, matrix has "
                << likes->NumCols() << " rows but transition-model has "
                << tm.NumPdfs() << " pdf-ids.";
  }  

  virtual int32 NumFramesReady() const { return likes_->NumRows(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_ * (*likes_)(frame, trans_model_.TransitionIdToPdf(tid));
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual ~DecodableMatrixScaledMapped() {
    if (delete_likes_) delete likes_;
  }
 private:
  const TransitionModel &trans_model_;  // for tid to pdf mapping
  const Matrix<BaseFloat> *likes_;
  BaseFloat scale_;
  bool delete_likes_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaledMapped);
};

/**
   This decodable class returns log-likes stored in a matrix; it supports
   repeatedly writing to the matrix and setting a time-offset representing the
   frame-index of the first row of the matrix.  It's intended for use in
   multi-threaded decoding; mutex and semaphores are not included.  External
   code will call SetLoglikes() each time more log-likelihods are available.
   If you try to access a log-likelihood that's no longer available because
   the frame index is less than the current offset, it is of course an error.
*/
class DecodableMatrixMappedOffset: public DecodableInterface {
 public:
  DecodableMatrixMappedOffset(const TransitionModel &tm):
      trans_model_(tm), frame_offset_(0), input_is_finished_(false) { }  



  virtual int32 NumFramesReady() { return frame_offset_ + loglikes_.NumRows(); }

  // this is not part of the generic Decodable interface.
  int32 FirstAvailableFrame() { return frame_offset_; }
  
  // This function is destructive of the input "loglikes" because it may
  // under some circumstances do a shallow copy using Swap().  This function
  // appends loglikes to any existing likelihoods you've previously supplied.
  // frames_to_discard, if nonzero, will discard that number of previously
  // available frames, from the left, advancing FirstAvailableFrame() by
  // a number equal to frames_to_discard.  You should only set frames_to_discard
  // to nonzero if you know your decoder won't want to access the loglikes
  // for older frames.
  void AcceptLoglikes(Matrix<BaseFloat> *loglikes,
                      int32 frames_to_discard) {
    if (loglikes->NumRows() == 0) return;
    KALDI_ASSERT(loglikes->NumCols() == trans_model_.NumPdfs());
    KALDI_ASSERT(frames_to_discard <= loglikes_.NumRows() &&
                 frames_to_discard >= 0);
    if (frames_to_discard == loglikes_.NumRows()) {
      loglikes_.Swap(loglikes);
      loglikes->Resize(0, 0);
    } else {
      int32 old_rows_kept = loglikes_.NumRows() - frames_to_discard,
          new_num_rows = old_rows_kept + loglikes->NumRows();
      Matrix<BaseFloat> new_loglikes(new_num_rows, loglikes->NumCols());
      new_loglikes.RowRange(0, old_rows_kept).CopyFromMat(
          loglikes_.RowRange(frames_to_discard, old_rows_kept));
      new_loglikes.RowRange(old_rows_kept, loglikes->NumRows()).CopyFromMat(
          *loglikes);
      loglikes_.Swap(&new_loglikes);
    }
    frame_offset_ += frames_to_discard;
  }

  void InputIsFinished() { input_is_finished_ = true; }
  
  virtual int32 NumFramesReady() const {
    return loglikes_.NumRows() + frame_offset_;
  }
  
  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1 && input_is_finished_);
  }

  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    int32 index = frame - frame_offset_;
    KALDI_ASSERT(index >= 0 && index < loglikes_.NumRows());
    return loglikes_(index, trans_model_.TransitionIdToPdf(tid));
  }

                 
                 
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  // nothing special to do in destructor.
  virtual ~DecodableMatrixMappedOffset() { }
 private:
  const TransitionModel &trans_model_;  // for tid to pdf mapping
  Matrix<BaseFloat> loglikes_;
  int32 frame_offset_;
  bool input_is_finished_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixMappedOffset);
};


class DecodableMatrixScaled: public DecodableInterface {
 public:
  DecodableMatrixScaled(const Matrix<BaseFloat> &likes,
                        BaseFloat scale): likes_(likes),
                                          scale_(scale) { }
  
  virtual int32 NumFramesReady() const { return likes_.NumRows(); }
  
  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }
  
  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return scale_ * likes_(frame, tid);
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return likes_.NumCols(); }

 private:
  const Matrix<BaseFloat> &likes_;
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaled);
};


}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_MATRIX_H_

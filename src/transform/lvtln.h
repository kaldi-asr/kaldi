// transform/lvtln.h

// Copyright 2009-2011 Microsoft Corporation

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


#ifndef KALDI_TRANSFORM_LVTLN_H_
#define KALDI_TRANSFORM_LVTLN_H_

#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "transform/transform-common.h"
#include "transform/fmllr-diag-gmm.h"


namespace kaldi {

/*
  Class for applying linear approximations to VTLN transforms;
  see \ref transform_lvtln.
*/


class LinearVtln {
 public:
  LinearVtln() { } // This initializer will probably be used prior to calling
  // Read().

  LinearVtln(int32 dim, int32 num_classes, int32 default_class);
  // This initializer sets up the
  // model; the transforms will initially all be the same.

  // SetTransform is used when we initialize it as "normal" VTLN.
  // It's not necessary to ever call this function.  "transform" is "A",
  // the square part of the transform matrix.
  void SetTransform(int32 i, const MatrixBase<BaseFloat> &transform);

  void SetWarp(int32 i, BaseFloat warp);

  BaseFloat GetWarp(int32 i) const;

  // GetTransform gets the transform for class i.  The caller must
  // make sure the output matrix is sized Dim() by Dim().
  void GetTransform(int32 i, MatrixBase<BaseFloat> *transform) const;


  /// Compute the transform for the speaker.
  void ComputeTransform(const FmllrDiagGmmAccs &accs,
                        std::string norm_type,  // type of regular fMLLR computation: "none", "offset", "diag"
                        BaseFloat logdet_scale,  // scale on logdet (1.0 is "correct" but less may work better)
                        MatrixBase<BaseFloat> *Ws,  // output fMLLR transform, should be size dim x dim+1
                        int32 *class_idx,  // the transform that was chosen...
                        BaseFloat *logdet_out,
                        BaseFloat *objf_impr = NULL,  // versus no transform
                        BaseFloat *count = NULL);

  void Read(std::istream &is, bool binary);

  void Write(std::ostream &os, bool binary) const;

  int32 Dim() const { KALDI_ASSERT(!A_.empty()); return A_[0].NumRows(); }
  int32 NumClasses() const { return A_.size(); }
  // This computes the offset term for this class given these
  // stats.
  void GetOffset(const FmllrDiagGmmAccs &speaker_stats,
                 int32 class_idx,
                 VectorBase<BaseFloat> *offset) const;

  friend class LinearVtlnStats;
 protected:
  int32 default_class_;  // transform we return if we have no data.
  std::vector<Matrix<BaseFloat> > A_;  // Square parts of the FMLLR matrices.
  std::vector<BaseFloat> logdets_;
  std::vector<BaseFloat> warps_; // This variable can be used to store the
                                 // warp factors that each transform correspond to.
  

};



}  // namespace kaldi

#endif  // KALDI_TRANSFORM_LVTLN_H_

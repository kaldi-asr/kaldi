// transform/balanced-cmn.h

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_TRANSFORM_BALANCED_CMN_H_
#define KALDI_TRANSFORM_BALANCED_CMN_H_

#include "base/kaldi-common.h"
#include "util/parse-options.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

/// Configuration class for class BalancedCmn.

struct BalancedCmnConfig {
  /// silence_proportion defines the proportion of silence of the signal,
  /// for which we aim for the signal to be zero-mean, and also the
  /// maximum proportion of silence that we allow in the stats in the
  /// case where we have "enough" non-silence data.
  BaseFloat silence_proportion;

  /// count_cutoff is the amount of data beyond which we will exactly match the
  /// amount "silence_proportion" of silence.  We get "silence_proportion" by
  /// down-weighting either the speech or silence stats, whichever is needed to
  /// get that proportion.  If doing so would take the total count below
  /// count_cutoff, we we down-weight it just enough to get the total count to
  /// that point, and no further; and we then attempt to match the mean to what
  /// we expect the mean to be with those proportions of speech and silence.
  BaseFloat count_cutoff;

  BalancedCmnConfig():
      silence_proportion(0.15),
      count_cutoff(200.0)  { }
  
  void Register(ParseOptions *po) {
    po->Register("count-cutoff", &count_cutoff,
                 "A threshold, expressed as a number of frames, that affects "
                 "the cepstral mean computation; see balanced-cmn.{h,cc} "
                 "for details.");
  }
};
    

/// This class computes "fake" CMN stats that if used by
/// ApplyCmn, will give a CMN transform that takes into account
/// silence and non-silence probabilities for frames.
class BalancedCmn {
 public:
  /// The inputs "global_sil_stats" and "global_nonsil_stats"
  /// would be as computed by AccCmvnStats in cmvn.h, given
  /// respectively the silence probability as a weight, and the
  /// non-silence probability.
  BalancedCmn(const BalancedCmnConfig &config,
              const MatrixBase<double> &global_sil_stats,
              const MatrixBase<double> &global_nonsil_stats);
  
  void AccStats(const MatrixBase<BaseFloat> &feats,
                const VectorBase<BaseFloat> &nonsilence_weight);

  double TotCount();

  /// This gives the cmn offset-- actually the negative of the offset,
  /// represented as a single row matrix with the negative of the offset,
  /// followed by a "fake count" of 1.0.  This is a single row variant
  /// of the standard CMVN stats that we normally give to the program
  /// apply-cmvn.
  const Matrix<double> &GetStats();

  
 private:
  void ComputeTargets(const MatrixBase<double> &global_sil_stats,
                      const MatrixBase<double> &global_nonsil_stats);
  
  int32 Dim() const { return global_target_sil_.Dim(); }

  /// Returns the scaling factor we'll apply to silence frames and
  /// nonsilence frames respectively.  This is interpolated, so if
  /// s is the probability a frame was silence, we scale according
  /// to s * sil_scale + (1 - s) * nonsil_scale.
  void GetWeights(double *sil_scale, double *nonsil_scale);
      
  BaseFloat SilenceWeight() const;

  const BalancedCmnConfig &config_;
 
  /// after normalization, expected overall average of silence.
  Vector<double> global_target_sil_;
  /// after normalization, expected overall average of nonsilence.
  Vector<double> global_target_nonsil_; // 

  Matrix<double> speaker_stats_; // [3][dim]: weighted by p(nonsil)*p(nonsil),
                                 // p(sil)*p(sil), p(sil)*p(nonsil)
  Vector<double> speaker_counts_; // [3]
  
  /// This is used as the return value of GetStats().  It will be
  /// a 1 x (dim + 1) matrix.
  Matrix<double> final_stats_;
};


}  // namespace kaldi

#endif  // KALDI_TRANSFORM_BALANCED_CMN_H_

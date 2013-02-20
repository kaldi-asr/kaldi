// transform/balanced-cmvn.h

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


#ifndef KALDI_TRANSFORM_BALANCED_CMVN_H_
#define KALDI_TRANSFORM_BALANCED_CMVN_H_

#include "base/kaldi-common.h"
#include "util/parse-options.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

/// Configuration class for class BalancedCmvn.

struct BalancedCmvnConfig {
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

  BalancedCmvnConfig():
      silence_proportion(0.15),
      count_cutoff(200.0)  { }
  
  void Register(ParseOptions *po) {
    po->Register("count-cutoff", &count_cutoff,
                 "A threshold, expressed as a number of frames, that affects "
                 "the cepstral mean computation; see balanced-cmvn.{h,cc} "
                 "for details.");
    po->Register("silence-proportion", &silence_proportion,
                 "The proportion of silence stats that we aim for when "
                 "reweighting the CMVN stats.");
  }
};
    

/// This class computes "fake" CMVN stats that if used by
/// ApplyCmvn, will give a CMVN transform that takes into account
/// silence and non-silence probabilities for frames.
class BalancedCmvn {
 public:
  /// The inputs "global_sil_stats" and "global_nonsil_stats"
  /// would be as computed by AccCmvnStats in cmvn.h, given
  /// respectively the silence probability as a weight, and the
  /// non-silence probability.
  BalancedCmvn(const BalancedCmvnConfig &config,
              const MatrixBase<double> &global_sil_stats,
              const MatrixBase<double> &global_nonsil_stats);
  
  void AccStats(const MatrixBase<BaseFloat> &feats,
                const VectorBase<BaseFloat> &nonsilence_weight);

  double TotCount();

  /// This gives the cmvn offset-- actually the negative of the offset,
  /// represented as a single row matrix with the negative of the offset,
  /// followed by a "fake count" of 1.0.  This is a single row variant
  /// of the standard CMVN stats that we normally give to the program
  /// apply-cmvn.
  const Matrix<double> &GetStats();

  
 private:
  /// Compute the transform, in (shift, scale) format, that will
  /// modify stats with the given (silence, nonsilence) counts
  /// to look like stats with the given silence proportion.
  void ComputeTransform(BaseFloat sil_proportion,
                        Matrix<double> *transform);
  
  int32 Dim() const { return global_sil_stats_.NumCols() - 1; }

  /// Returns the scaling factor we'll apply to silence frames and
  /// nonsilence frames respectively.  This is interpolated, so if
  /// s is the probability a frame was silence, we scale according
  /// to s * sil_scale + (1 - s) * nonsil_scale.
  void GetWeights(double *sil_scale, double *nonsil_scale);
      
  BaseFloat SilenceWeight() const;

  const BalancedCmvnConfig &config_;
 
  /// after normalization, expected overall stats of silence [with unit count].
  Matrix<double> global_sil_stats_;
  /// after normalization, expected overall average of nonsilence [with unit count].
  Matrix<double> global_nonsil_stats_;
  
  Matrix<double> nonsil_nonsil_stats_; // stats weighted by p(nonsil)*p(nonsil)
  Matrix<double> sil_sil_stats_; // stats weighted by p(sil)*p(sil)
  Matrix<double> sil_nonsil_stats_; // stats weighted by p(sil)*p(nonsil).
  
  /// This is used as the return value of GetStats().  It will be
  /// a 2 x (dim + 1) matrix.
  Matrix<double> final_stats_;
};

// The following are utility functions used by code in balanced-cmvn.h.
// It's mostly for ease of testing that we put them in the header here;
// it's unlikely that user-level code will want to call them.

/// Some functions currently used just in class BalancedCmvn
/// They are pretty generic so we don't put them inside the class.

/// Initialize 2 x (Dim() + 1) matrix, 1st row ( 0, 0, 0, ... 1),
/// 2nd row ( 1, 1, 1, ... 0); this is the "target form" of the
/// stats, with unit count and mean and variance normalized.
void InitStandardCmvnStats(int32 dim,
                           Matrix<double> *stats);

/// Get the transform, i.e. the shift and scale such that y = shift + x * scale,
/// with the shift in 1st row of transform and scale in 2nd row, that would
/// transform "orig_stats" to "target_stats".  Here, "target_stats" must have
/// a count of 1.0.
void GetCmvnTransform(const MatrixBase<double> &orig_stats,
                      const MatrixBase<double> &target_stats,
                      Matrix<double> *transform);

/// Get the transform (expressed as shift and scale) that is necessary
/// to convert zero-mean, unit-variance stats into "stats".
void ConvertCmvnStatsToTransform(const MatrixBase<double> &stats,
                                 Matrix<double> *transform);

/// Get the transform that would convert "stats" into zero-mean
/// (and if transform_var == true, also unit-variance).
void ConvertCmvnStatsToInvTransform(const MatrixBase<double> &stats,
                                    Matrix<double> *inv_transform);


/// Converts a transform to stats, with unit count.  This gives us the stats
/// that you would have to give to ApplyCmvn to get this transform.
/// Forms a null-op cycle with ConvertStatsToTransfrom and InvertTransform.
void ConvertInvCmvnTransformToStats(const MatrixBase<double> &transform,
                                    Matrix<double> *stats);

/// Get the transform (expressed as shift and scale) equivalent
/// to applying first transform1 and then transform2.
void ComposeCmvnTransforms(const MatrixBase<double> &transform1,
                           const MatrixBase<double> &transform2,
                           Matrix<double> *transform);

/// Apply a transform (shift and scale) to CMVN stats and get the
/// transformed stats.
void TransformCmvnStats(const MatrixBase<double> &stats,
                        const MatrixBase<double> &transform,
                         Matrix<double> *transformed_stats);
  
void InvertCmvnTransform(const MatrixBase<double> &transform,
                         Matrix<double> *inverse);

}  // namespace kaldi

#endif  // KALDI_TRANSFORM_BALANCED_CMVN_H_

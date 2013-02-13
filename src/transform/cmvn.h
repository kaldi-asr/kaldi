// transform/cmvn.h

// Copyright 2009-2011 Microsoft Corporation

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


#ifndef KALDI_TRANSFORM_CMVN_H_
#define KALDI_TRANSFORM_CMVN_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "hmm/transition-model.h"

namespace kaldi {

/// This function initializes the matrix to dimension 2 by (dim+1);
/// 1st "dim" elements of 1st row are mean stats, 1st "dim" elements
/// of 2nd row are var stats, last element of 1st row is count,
/// last element of 2nd row is zero.
void InitCmvnStats(int32 dim, Matrix<double> *stats);

/// Accumulation from a single frame (weighted).
void AccCmvnStats(const VectorBase<BaseFloat> &feats, BaseFloat weight, MatrixBase<double> *stats);

/// Accumulation from a feature file (possibly weighted-- useful in excluding silence).
void AccCmvnStats(const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> *weights,  // or NULL
                  MatrixBase<double> *stats);

/// Apply cepstral mean and variance normalization to a matrix of features.
void ApplyCmvn(const MatrixBase<double> &stats,
               bool norm_vars,
               MatrixBase<BaseFloat> *feats);

/// Configuration class for class BalancedCmvn.
struct BalancedCmvnConfig {
  BaseFloat nonsilence_frames_cutoff;

  BalancedCmvnConfig(): nonsilence_frames_cutoff(500.0) { }
  
  void Register(ParseOptions *po) {
    po->Register("nonsilence-frames-cutoff", &nonsilence_frames_cutoff,
                 "A threshold, expressed as a number of frames, such that when "
                 "the number of non-silence frames falls below this value, we "
                 "start using silence frames also.");
  }
};
    
namespace cmvn_utils {
/// Some functions currently used just in class BalancedCmvn
/// They are pretty generic so we don't put them inside the class.

/// Initialize 2 x (Dim() + 1) matrix, 1st row ( 0, 0, 0, ... 1),
/// 2nd row ( 1, 1, 1, ... 0); this is the "target form" of the
/// stats, with unit count and mean and variance normalized.
void InitStandardStats(int32 dim,
                       Matrix<double> *stats);

/// Add a frame to standard CMVN stats as a 2 x (Dim() + 1) matrix,
/// where the count is indexed (0, Dim()).
void AddFrameToCmvnStats(BaseFloat weight,
                         VectorBase<BaseFloat> &frame,
                         MatrixBase<double> *stats);

/// Get the transform, i.e. the shift and scale such that y = shift + x * scale,
/// with the shift in 1st row of transform and scale in 2nd row, that would
/// transform "orig_stats" to "target_stats".  Here, "target_stats" must have
/// a count of 1.0.
void GetTransform(const MatrixBase<double> &orig_stats,
                  const MatrixBase<double> &target_stats,
                  Matrix<double> *transform);

/// Get the transform (expressed as shift and scale) that is necessary
/// to convert zero-mean, unit-variance stats into "stats".
void ConvertStatsToTransform(const MatrixBase<double> &stats,
                             Matrix<double> *transform);

/// Get the transform that would convert "stats" into zero-mean
/// (and if transform_var == true, also unit-variance).
void ConvertStatsToInvTransform(const MatrixBase<double> &stats,
                                Matrix<double> *inv_transform);


/// Converts a transform to stats, with unit count.  This gives us the stats
/// that you would have to give to ApplyCmvn to get this transform.
/// Forms a null-op cycle with ConvertStatsToTransfrom and InvertTransform.
void ConvertInvTransformToStats(const MatrixBase<double> &transform,
                                Matrix<double> *stats);

/// Get the transform (expressed as shift and scale) equivalent
/// to applying first transform1 and then transform2.
void ComposeTransforms(const MatrixBase<double> &transform1,
                       const MatrixBase<double> &transform2,
                       Matrix<double> *transform);

/// Apply a transform (shift and scale) to CMVN stats and get the
/// transformed stats.
void TransformStats(const MatrixBase<double> &stats,
                    const MatrixBase<double> &transform,
                    Matrix<double> *transformed_stats);
  
void InvertTransform(const MatrixBase<double> &transform,
                     Matrix<double> *inverse);
  

}

/// This class computes \"fake\" CMVN stats that if used by
/// ApplyCmvn, will give a CMVN transform that takes into account
/// silence and non-silence probabilities for frames.
class BalancedCmvn {
  friend class TestCmvn;
 public:
  BalancedCmvn(const BalancedCmvnConfig &config,
               int32 feat_dim);

  void AccStats(const MatrixBase<BaseFloat> &feats,
                const VectorBase<BaseFloat> &nonsilence_weight);

  double TotCount();
  
  const Matrix<double> &GetStats(const MatrixBase<double> &global_sil_stats,
                                 const MatrixBase<double> &global_nonsil_stats);

  // A simpler method, rescales to global proportions.
  const Matrix<double> &GetStats2(const MatrixBase<double> &global_sil_stats,
                                  const MatrixBase<double> &global_nonsil_stats);

  
 protected:
  int32 Dim() const { return sil_sil_stats_.NumCols() - 1; }

  /// Returns the scaling factor we'll apply to silence frames.
  BaseFloat SilenceWeight() const;


  const BalancedCmvnConfig &config_;
  
  /// This matrix stores the stats in standard CMVN format,
  /// weighted by p(sil) * p(sil).
  Matrix<double> sil_sil_stats_;
  /// This matrix stores the stats in standard CMVN format,
  /// weighted by p(non-sil) * p(non-sil).
  Matrix<double> nonsil_nonsil_stats_;
  /// This matrix stores the stats in standard CMVN format,
  /// weighted by p(sil) * p(non-sil).
  Matrix<double> sil_nonsil_stats_;

  /// This is used as the return value of GetStats().
  Matrix<double> final_stats_;
};


}  // namespace kaldi

#endif  // KALDI_TRANSFORM_CMVN_H_

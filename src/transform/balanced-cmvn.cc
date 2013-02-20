// transform/balanced-cmvn.cc

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

#include "transform/balanced-cmvn.h"
#include "transform/cmvn.h"

namespace kaldi {

BalancedCmvn::BalancedCmvn(const BalancedCmvnConfig &config,
                         const MatrixBase<double> &global_sil_stats,
                         const MatrixBase<double> &global_nonsil_stats):
    config_(config) {
  KALDI_ASSERT(global_sil_stats.NumRows() == 2 &&
               global_nonsil_stats.NumRows() == 2 &&
               global_sil_stats.NumCols() == global_nonsil_stats.NumCols());
  
  int32 dim = global_sil_stats.NumCols() - 1;

  nonsil_nonsil_stats_.Resize(2, dim + 1);
  sil_sil_stats_.Resize(2, dim + 1);
  sil_nonsil_stats_.Resize(2, dim + 1);

  double global_sil_count = global_sil_stats(0, dim),
      global_nonsil_count = global_nonsil_stats(0, dim);
  KALDI_ASSERT(global_sil_count > 0 && global_nonsil_count > 0);
  global_sil_stats_ = global_sil_stats;
  global_sil_stats_.Scale(1.0 / global_sil_count);
  global_nonsil_stats_ = global_nonsil_stats;
  global_nonsil_stats_.Scale(1.0 / global_nonsil_count);
}


void BalancedCmvn::AccStats(const MatrixBase<BaseFloat> &feats,
                           const VectorBase<BaseFloat> &nonsilence_weight) {
  KALDI_ASSERT(feats.NumRows() == nonsilence_weight.Dim() &&
               nonsilence_weight.Min() >= 0.0 &&
               nonsilence_weight.Max() <= 1.0);
  KALDI_ASSERT(feats.NumCols() == Dim());

  for (int32 t = 0; t < feats.NumRows(); t++) {
    SubVector<BaseFloat> x(feats, t);
    BaseFloat nonsil_prob = nonsilence_weight(t),
        sil_prob = 1.0 - nonsil_prob;
    AccCmvnStats(x, nonsil_prob * nonsil_prob, &nonsil_nonsil_stats_);
    AccCmvnStats(x, sil_prob * sil_prob, &sil_sil_stats_);
    AccCmvnStats(x, sil_prob * nonsil_prob, &sil_nonsil_stats_);
  }
}


double BalancedCmvn::TotCount() {
  double nonsil_nonsil_count = nonsil_nonsil_stats_(0, Dim()),
      sil_sil_count= sil_sil_stats_(0, Dim()),
      sil_nonsil_count = sil_nonsil_stats_(0, Dim());
  return  nonsil_nonsil_count + sil_sil_count + 2.0*sil_nonsil_count;
}

void BalancedCmvn::GetWeights(double *sil_scale, double *nonsil_scale) {
  double nonsil_nonsil_count = nonsil_nonsil_stats_(0, Dim()),
      sil_sil_count= sil_sil_stats_(0, Dim()),
      sil_nonsil_count = sil_nonsil_stats_(0, Dim()),
      nonsil_count = nonsil_nonsil_count + sil_nonsil_count,
      sil_count = sil_sil_count + sil_nonsil_count,
      tot_count = nonsil_count + sil_count;
  // nonsil_count is count we would get if we were to have all the weight on
  // nonsilence frames and zero on silence.  sil_count is the reverse.
  
  KALDI_ASSERT(tot_count > 0.0);
  if (tot_count <= config_.count_cutoff || sil_count == 0 || nonsil_count == 0) {
    *sil_scale = 1.0;
    *nonsil_scale = 1.0;
  } else {
    // Proportion of silence if we were to weight the nonsilence frames as 1,
    // and silence frames as 0:
    double nonsil_frames_sil_proportion = sil_nonsil_count / nonsil_count;
    // Proportion of silence if we were to weight the nonsilence frames as 0,
    // and silence frames as 1:
    double sil_frames_sil_proportion = sil_sil_count / sil_count;

    KALDI_ASSERT(sil_frames_sil_proportion >= nonsil_frames_sil_proportion);
    
    double target_sil_proportion = config_.silence_proportion;  // the target proportion.
    
    if (target_sil_proportion > sil_frames_sil_proportion) {
      // want all silence frames.
      *sil_scale = 1.0;
      *nonsil_scale = 0.0;
    } else if (target_sil_proportion < nonsil_frames_sil_proportion) {
      // want all nonsilence frames.
      *sil_scale = 0.0;
      *nonsil_scale = 1.0;
    } else {
      // sil_fraction below is the fraction of the stats that should come from
      // silence frames.
      // the +1.0e-10 is to prevent division by zero in a particular pathological
      // case that should never be encountered, wheren the two sil_proportions
      // are identical.
      double sil_fraction = (target_sil_proportion - nonsil_frames_sil_proportion)
          / (sil_frames_sil_proportion - nonsil_frames_sil_proportion + 1.0e-10);
      KALDI_ASSERT(sil_fraction >= 0.0 && sil_fraction < 1.0);

      // The following scales would give us "target_sil_proportion" as the
      // fraction of silence, but are not properly normalized.
      *sil_scale = sil_fraction / sil_count;
      *nonsil_scale = (1.0 - sil_fraction) / nonsil_count;
      double max_scale = std::max(*sil_scale, *nonsil_scale);
      *sil_scale /= max_scale;
      *nonsil_scale /= max_scale;
    }

    double count_cutoff = config_.count_cutoff;
    // Now, if applying these scales would take us below the count cutoff,
    // increase whichever scale is < 1.0.
    double count = *sil_scale * sil_count + *nonsil_scale + nonsil_count;
    if (count < count_cutoff) { // We scaled the stats down too far,
      // so we got below the count cutoff.  Re-do the scaling.
      double count_deficit = count_cutoff - count;
      // We only need to modify whichever scale is < 1.0.
      if (*sil_scale < 0.999 && sil_count > 0.0) {
        double new_scale = std::min(1.0, *sil_scale + count_deficit / sil_count);
        KALDI_VLOG(2) << "Count after scaling " << count << " was below cutoff "
                      << count_cutoff << ", increasing sil_scale from "
                      << *sil_scale << " to " << new_scale;
        *sil_scale = new_scale;
      }
      if (*nonsil_scale < 0.999 && nonsil_count > 0.0) {
        double new_scale = std::min(1.0,
                                    *nonsil_scale + count_deficit/nonsil_count);
        KALDI_VLOG(2) << "Count after scaling " << count << " was below cutoff "
                      << count_cutoff << ", increasing nonsil_scale from "
                      << *nonsil_scale << " to " << new_scale;
        *nonsil_scale = new_scale;
      }      
    }
  }
}

void BalancedCmvn::ComputeTransform(BaseFloat actual_sil_proportion,
                                   Matrix<double> *transform) {
  KALDI_ASSERT(actual_sil_proportion >= 0.0 && actual_sil_proportion <= 1.0);
  Matrix<double> global_stats_normal_weight(2, Dim() + 1),
      global_stats_our_weight(2, Dim() + 1);
  global_stats_normal_weight.AddMat(config_.silence_proportion,
                                    global_sil_stats_);
  global_stats_normal_weight.AddMat(1.0 - config_.silence_proportion,
                                    global_nonsil_stats_);
  global_stats_our_weight.AddMat(actual_sil_proportion,
                                 global_sil_stats_);
  global_stats_our_weight.AddMat(1.0 - actual_sil_proportion,
                                 global_nonsil_stats_);
  // the following is not a member-function call.
  GetCmvnTransform(global_stats_our_weight,
                   global_stats_normal_weight,
                   transform);
}



const Matrix<double> &BalancedCmvn::GetStats() {
  KALDI_ASSERT(TotCount() > 0.0);
  int32 dim = Dim();
  double sil_scale, nonsil_scale;
  GetWeights(&sil_scale, &nonsil_scale);

  /// the scale is applied to frames, not stats; if s is the probability
  /// a frame is silence, we weight it by s*sil_scale + (1-s)*nonsil_scale.
  double nonsil_nonsil_count = nonsil_nonsil_stats_(0, Dim()),
      sil_sil_count= sil_sil_stats_(0, Dim()),
      sil_nonsil_count = sil_nonsil_stats_(0, Dim()),
      sil_count = sil_sil_count * sil_scale + sil_nonsil_count * nonsil_scale,
      nonsil_count = nonsil_nonsil_count * nonsil_scale + sil_nonsil_count * sil_scale,
      tot_count = sil_count + nonsil_count;

  KALDI_VLOG(2) << "After weighting silence frames by " << sil_scale
                << " and non-silence frames by " << nonsil_scale
                << ", counts of silence and non-silence are "
                << sil_count << " and " << nonsil_count
                << ", total is " << tot_count
                << "; proportion of silence is " << (sil_count/tot_count)
                << " vs. target " << config_.silence_proportion;
  
  Matrix<double> weighted_stats(2, dim + 1);
  weighted_stats.AddMat(nonsil_scale, nonsil_nonsil_stats_);
  weighted_stats.AddMat(sil_scale, sil_sil_stats_);
  weighted_stats.AddMat(nonsil_scale + sil_scale, sil_nonsil_stats_);

  Matrix<double> transform;
  this->ComputeTransform(sil_count / tot_count, &transform);

  TransformCmvnStats(weighted_stats, transform, &final_stats_);

  return final_stats_;
}


void InitStandardCmvnStats(int32 dim,
                           Matrix<double> *target) {
  target->Resize(2, dim + 1);
  for (int32 i = 0; i < dim; i++) {
    (*target)(0, i) = 0.0;
    (*target)(1, i) = 1.0;
  }
  (*target)(0, dim) = 1.0; // the count.
  (*target)(1, dim) = 0.0; // This element is unused.
}


void GetCmvnTransform(const MatrixBase<double> &orig_stats,
                      const MatrixBase<double> &target_stats,
                      Matrix<double> *transform) {
  
  Matrix<double> unit_to_orig, orig_to_unit, unit_to_target;
  ConvertCmvnStatsToTransform(orig_stats,
                              &unit_to_orig);
  ConvertCmvnStatsToTransform(target_stats,
                              &unit_to_target);
  InvertCmvnTransform(unit_to_orig, &orig_to_unit);
  ComposeCmvnTransforms(orig_to_unit, unit_to_target, transform);
}


void TransformCmvnStats(const MatrixBase<double> &stats,
                        const MatrixBase<double> &transform,
                        Matrix<double> *transformed_stats) {
  int32 dim = stats.NumCols() - 1;
  KALDI_ASSERT(dim > 0 && stats.NumRows() == 2 && transform.NumRows() == 2
               && transform.NumCols() == dim);
  *transformed_stats = stats;
  double count = stats(0, dim);
  if (count == 0) return; // Nothing to do.
  transformed_stats->Scale(1.0 / count); // so we can forget about the count for now.
  
  // 1st row of transform is shift; 2nd row of transform is scale.
  // scale on x is: y = shift + x * scale.
  //              y^2 = (shift + x * scale)^2 = shift^2 + 2*shift*scale*x + scale*scale* x^2
  for (int32 i = 0; i < dim; i++) {
    double x = (*transformed_stats)(0, i), x2 = (*transformed_stats)(1, i),
        shift = transform(0, i), scale = transform(1, i),
        y = shift + x * scale,
        y2 = shift * shift + 2.0 * shift * scale * x + scale * scale * x2;
    (*transformed_stats)(0, i) = y;
    (*transformed_stats)(1, i) = y2;
  }
  transformed_stats->Scale(count);
}

/// Get the transform that would convert zero-mean, unit-variance
/// stats into the given stats.
void ConvertCmvnStatsToTransform(const MatrixBase<double> &stats,
                                 Matrix<double> *transform) {
  int32 dim = stats.NumCols() - 1;
  transform->Resize(2, dim);
  double count = stats(0, dim);
  KALDI_ASSERT(count > 0.0);
  
  Matrix<double> norm_stats(stats);
  norm_stats.Scale(1.0 / count); // So it's as if the count is 1.0.

  for (int32 d = 0; d < dim; d++) {
    double // x = 0.0, x2 = 1.0,
        y = norm_stats(0, d), y2 = norm_stats(1, d);
    // Now solve for "shift" and "scale" in:
    // y = shift + x * scale,
    // y2 = shift * shift + 2.0 * shift * scale * x + scale * scale * x2;
    double shift = y,
        scale2 = y2 - shift*shift; // scale2 is square of scale.
    if (scale2 <= 0.0) {
      KALDI_WARN << "Invalid scale: invalid CMVN stats?  Setting to one.";
      scale2 = 1.0;
    }
    double scale = std::sqrt(scale2);
    (*transform)(0, d) = shift;
    (*transform)(1, d) = scale;
  }
}

/// Convert stats to the transform that would take the stats
/// to zero-mean, unit-variance.
void ConverCmvntStatsToInvTransform(const MatrixBase<double> &stats,
                                Matrix<double> *inv_transform) {
  Matrix<double> transform;
  ConvertCmvnStatsToTransform(stats, &transform);
  InvertCmvnTransform(transform, inv_transform);
}

void InvertCmvnTransform(const MatrixBase<double> &transform,
                         Matrix<double> *inverse) {
  KALDI_ASSERT(transform.NumRows() == 2);
  int32 dim = transform.NumCols();
  inverse->Resize(2, dim);
  for (int32 i = 0; i < dim; i++) {
    double shift = transform(0, i), scale = transform(1, i);
    double inv_scale = 1.0 / scale, inv_shift = -shift * inv_scale;
    (*inverse)(0, i) = inv_shift;
    (*inverse)(1, i) = inv_scale;
  }
}

void ComposeCmvnTransforms(const MatrixBase<double> &transform1,
                           const MatrixBase<double> &transform2,
                           Matrix<double> *transform) {
  int32 dim = transform1.NumCols();
  KALDI_ASSERT(transform2.NumCols() == dim && transform1.NumRows() == 2
               && transform2.NumRows() == 2);
  transform->Resize(2, dim);
  // suppose we have a feature x, we apply transform1 and then transform2.  Let
  // the (shift,scale) of transform1 and transform2 be (a1,b1) and (a2,b2) respectively.
  // We we transform x->y->z, we have
  // y = a1 + b1*x
  // z = a2 + b2*y  = a2 + b2*a1 + b2*b1*x
  // So we have shift = shift2 + scale2 * shift1, and
  //            scale = scale1 * scale2.
  for (int32 i = 0; i < dim; i++) {
    double shift1 = transform1(0, i), shift2 = transform2(0, i),
        scale1 = transform1(1, i), scale2 = transform2(1, i),
        shift = shift2 + scale2 * shift1,
        scale = scale1 * scale2;
    (*transform)(0, i) = shift;
    (*transform)(1, i) = scale;
  }
}

void ConvertInvCmvnTransformToStats(const MatrixBase<double> &transform,
                                    Matrix<double> *stats) {
  int32 dim = transform.NumCols();
  KALDI_ASSERT(transform.NumRows() == 2);
  Matrix<double> standard_stats, inv_transform;
  InitStandardCmvnStats(dim, &standard_stats);
  InvertCmvnTransform(transform, &inv_transform);
  TransformCmvnStats(standard_stats, inv_transform, stats);
}



}  // namespace kaldi


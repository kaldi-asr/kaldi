// transform/cmvn.cc

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

#include "transform/cmvn.h"

namespace kaldi {

void InitCmvnStats(int32 dim, Matrix<double> *stats) {
  KALDI_ASSERT(dim > 0);
  stats->Resize(2, dim+1);
}

void AccCmvnStats(const VectorBase<BaseFloat> &feats, BaseFloat weight, MatrixBase<double> *stats) {
  int32 dim = feats.Dim();
  KALDI_ASSERT(stats != NULL);
  KALDI_ASSERT(stats->NumRows() == 2 && stats->NumCols() == dim+1);
  // Remove these __restrict__ modifiers if they cause compilation problems.
  // It's just an optimization.
   double *__restrict__ mean_ptr = stats->RowData(0),
       *__restrict__ var_ptr = stats->RowData(1),
       *__restrict__ count_ptr = mean_ptr + dim;
   const BaseFloat * __restrict__ feats_ptr = feats.Data();
  *count_ptr += weight;
  // Careful-- if we change the format of the matrix, the "mean_ptr < count_ptr"
  // statement below might become wrong.
  for (; mean_ptr < count_ptr; mean_ptr++, var_ptr++, feats_ptr++) {
    *mean_ptr += *feats_ptr * weight;
    *var_ptr +=  *feats_ptr * *feats_ptr * weight;
  }
}

void AccCmvnStats(const MatrixBase<BaseFloat> &feats,
                  const VectorBase<BaseFloat> *weights,
                  MatrixBase<double> *stats) {
  int32 num_frames = feats.NumRows();
  if (weights != NULL) {
    KALDI_ASSERT(weights->Dim() == num_frames);
  }
  for (int32 i = 0; i < num_frames; i++) {
    SubVector<BaseFloat> this_frame = feats.Row(i);
    BaseFloat weight = (weights == NULL ? 1.0 : (*weights)(i));
    if (weight != 0.0)
      AccCmvnStats(this_frame, weight, stats);
  }
}

void ApplyCmvn(const MatrixBase<double> &stats,
               bool var_norm,
               MatrixBase<BaseFloat> *feats) {
  KALDI_ASSERT(feats != NULL);
  int32 dim = stats.NumCols() - 1;
  if (stats.NumRows() != 2 || feats->NumCols() != dim) {
    KALDI_ERR << "Dim mismatch in ApplyCmvn: cmvn "
              << stats.NumRows() << 'x' << stats.NumCols()
              << ", feats " << feats->NumRows() << 'x' << feats->NumCols();
  }
  double count = stats(0, dim);
  if (count < 1.0)
    KALDI_ERR << "Insufficient stats for cepstral mean and variance normalization: "
              << "count = " << count;

  Matrix<BaseFloat> norm(2, dim);  // norm(0, d) = mean offset
  // norm(1, d) = scale, e.g. x(d) <-- x(d)*norm(1, d) + norm(0, d).
  for (int32 d = 0; d < dim; d++) {
    double mean, offset, scale;
    mean = stats(0, d)/count;
    if (!var_norm) {
      scale = 1.0;
      offset = -mean;
    } else {
      double var = (stats(1, d)/count) - mean*mean,
          floor = 1.0e-20;
      if (var < floor) {
        KALDI_WARN << "Flooring cepstral variance from " << var << " to "
                   << floor;
        var = floor;
      }
      scale = 1.0 / sqrt(var);
      if (scale != scale || 1/scale == 0.0)
        KALDI_ERR << "NaN or infinity in cepstral mean/variance computation\n";
      offset = -(mean*scale);
    }
    norm(0, d) = offset;
    norm(1, d) = scale;
  }
  int32 num_frames = feats->NumRows();

  // Apply the normalization.
  for (int32 i = 0; i < num_frames; i++) {
    for (int32 d = 0; d < dim; d++) {
      BaseFloat &f = (*feats)(i, d);
      f = norm(0, d) + f*norm(1, d);
    }
  }
}


BalancedCmvn::BalancedCmvn(const BalancedCmvnConfig &config,
                           int32 feat_dim):
    config_(config),
    sil_sil_stats_(2, feat_dim + 1), nonsil_nonsil_stats_(2, feat_dim + 1),
    sil_nonsil_stats_(2, feat_dim + 1) {
  KALDI_ASSERT(feat_dim > 0);
}

double BalancedCmvn::TotCount() {
  return sil_sil_stats_(0, Dim()) + nonsil_nonsil_stats_(0, Dim())
      + 2.0 * sil_nonsil_stats_(0, Dim());
}

void BalancedCmvn::AccStats(const MatrixBase<BaseFloat> &feats,
                            const VectorBase<BaseFloat> &nonsilence_weight) {
  using namespace cmvn_utils;
  // The following is a precondition.
  KALDI_ASSERT(feats.NumRows() == nonsilence_weight.Dim() &&
               nonsilence_weight.Min() >= 0.0 &&
               nonsilence_weight.Max() <= 1.0);
  KALDI_ASSERT(feats.NumCols() == Dim());

  for (int32 t = 0; t < feats.NumRows(); t++) {
    SubVector<BaseFloat> x(feats, t);
    BaseFloat nonsil_prob = nonsilence_weight(t),
        sil_prob = 1.0 - nonsil_prob;
    AddFrameToCmvnStats(sil_prob * sil_prob, x, &sil_sil_stats_);
    AddFrameToCmvnStats(nonsil_prob * nonsil_prob, x, &nonsil_nonsil_stats_);
    AddFrameToCmvnStats(sil_prob * nonsil_prob, x, &sil_nonsil_stats_);
  }
}

BaseFloat BalancedCmvn::SilenceWeight() const {
  int32 dim = Dim();
  BaseFloat sil_sil_count = sil_sil_stats_(0, dim),
      nonsil_nonsil_count = nonsil_nonsil_stats_(0, dim),
      sil_nonsil_count = sil_nonsil_stats_(0, dim);
  BaseFloat nonsil_count = sil_nonsil_count + nonsil_nonsil_count,
      sil_count = sil_sil_count + sil_nonsil_count;
  
  KALDI_ASSERT(nonsil_count + sil_count > 0.0);
  KALDI_ASSERT(config_.nonsilence_frames_cutoff >= 0.0);
  // nonsil_count is sum over all frames of the probability that frame
  // was non-silence.

  if (nonsil_count > config_.nonsilence_frames_cutoff) {
    // We have enough non-silence frames to robustly estimate the CMVN,
    // so we have no need for any silence frames.
    return 0.0;
  }
  if (sil_count == 0.0) {
    KALDI_WARN << "Zero silence count"; // Possible due to rounding or quantization
    return 1.0;                         // when computing weights, but unlikely.
  }
  BaseFloat required_count = config_.nonsilence_frames_cutoff - nonsil_count;
  BaseFloat scale = required_count / sil_count; // This is the scale we would have
  // to put on the silence in order to get up to the required count.
  if (scale > 1.0) { // We never scale up the silence frames-- this does not
                     // make sense, because speech is always at least as
                     // important as silence.
    return 1.0;
  } else {
    return scale;
  }
}

const Matrix<double> &BalancedCmvn::GetStats2(const MatrixBase<double> &global_sil_stats,
                                              const MatrixBase<double> &global_nonsil_stats) {
  int32 dim = Dim();
  BaseFloat global_sil_count = global_sil_stats(0, dim),
      global_nonsil_count = global_nonsil_stats(0, dim),
      global_tot_count = global_sil_count + global_nonsil_count;
  BaseFloat global_sil_proportion = global_sil_count / global_tot_count;
  
  BaseFloat sil_sil_count = sil_sil_stats_(0, dim),
      nonsil_nonsil_count = nonsil_nonsil_stats_(0, dim),
      sil_nonsil_count = sil_nonsil_stats_(0, dim);
  BaseFloat nonsil_count = sil_nonsil_count + nonsil_nonsil_count,
      sil_count = sil_sil_count + sil_nonsil_count;

  if (nonsil_count == 0 || sil_count == 0) {
    KALDI_WARN << "Non-silence count is " << nonsil_count << ", silence count is "
               << sil_count;
    // Return global sum of stats.  Not much we can do here.
    final_stats_ = sil_sil_stats_;
    final_stats_.AddMat(1.0, nonsil_nonsil_stats_);
    final_stats_.AddMat(2.0, sil_nonsil_stats_);
    return final_stats_;
  }
  
  // We want to scale at the frame level not the stats level.
  BaseFloat nonsil_sil_proportion = sil_nonsil_count / nonsil_count,
      sil_sil_proportion = sil_sil_count / sil_count;

  BaseFloat nonsil_scale, sil_scale;
  KALDI_ASSERT(sil_sil_proportion >= nonsil_sil_proportion);
  if (sil_sil_proportion < global_sil_proportion) { // Even if
    // we give 100% weight to the silence frame and zero to the nonsilence,
    // we still get too little silence -> give all weight to the silence frames.
    KALDI_WARN << "Even choosing only silence frames, get too small silence "
               << "proportion, " << sil_sil_proportion << " <  "
               << global_sil_proportion;
    nonsil_scale = 0.0;
    sil_scale = 1.0;
  } else if (nonsil_sil_proportion > global_sil_proportion) {  
    KALDI_WARN << "Even choosing only non-silence frames, get too much silence "
               << "proportion, " << nonsil_sil_proportion << " >  "
               << global_sil_proportion;
    nonsil_scale = 1.0;
    sil_scale = 0.0;
  } else {
    // Go this far along the line from all-silence to all-nonsilence.
    sil_scale = (global_sil_proportion - nonsil_sil_proportion) /
        (sil_sil_proportion - nonsil_sil_proportion);
    KALDI_ASSERT(sil_scale >= 0 && sil_scale <= 1);
    nonsil_scale = 1.0 - sil_scale;
    // Normalize so the largest one is 1.0, while keeping the same ratio.
    if (sil_scale > 0.5) {
      nonsil_scale /= sil_scale;
      sil_scale = 1.0;
    } else {
      sil_scale /= nonsil_scale;
      nonsil_scale = 1.0;
    }
  }

  final_stats_.Resize(2, Dim() + 1);
  final_stats_.AddMat(sil_scale, sil_sil_stats_);
  final_stats_.AddMat(sil_scale, sil_nonsil_stats_);
  final_stats_.AddMat(nonsil_scale, nonsil_nonsil_stats_);
  final_stats_.AddMat(nonsil_scale, sil_nonsil_stats_);
  return final_stats_;
}


const Matrix<double> &BalancedCmvn::GetStats(const MatrixBase<double> &global_sil_stats,
                                             const MatrixBase<double> &global_nonsil_stats) {
  using namespace cmvn_utils;
  int32 dim = Dim();
  BaseFloat silence_weight = SilenceWeight();
  Matrix<double> weighted_sil_stats(2, dim + 1),
      weighted_nonsil_stats(2, dim + 1);
  
  // First add nonsilence frames with weight one.  We're usng the fact that
  // p(nonsil)+ p(sil) == 1, so adding sil_nonsil_stats_ plus nonsil_nonsil_stats_,
  // is the same as adding frames proportional to p(nonsil) on those frames.
  weighted_nonsil_stats.AddMat(1.0, nonsil_nonsil_stats_);
  weighted_sil_stats.AddMat(1.0, sil_nonsil_stats_);

  // Next add silence frames with weight silence_weight, using similar logic.
  weighted_nonsil_stats.AddMat(silence_weight, sil_nonsil_stats_);
  weighted_sil_stats.AddMat(silence_weight, sil_sil_stats_);
  
  double tot_nonsil_count = weighted_nonsil_stats(0, dim),
      tot_sil_count = weighted_sil_stats(0, dim);

  KALDI_VLOG(2) << "Silence weighting factor is " << silence_weight
                << ", tot_nonsil_count = " << tot_nonsil_count
                << ", tot_sil_count = " << tot_sil_count;
  
  // weighted_stats is all the silence and non-silence stats, after weighting
  // frames.
  Matrix<double> weighted_speaker_stats(weighted_sil_stats);
  weighted_speaker_stats.AddMat(1.0, weighted_nonsil_stats);

  Matrix<double> global_tot_stats(global_sil_stats);
  global_tot_stats.AddMat(1.0, global_nonsil_stats);
  

  // global_inv_transform is the transform that would make "global_tot_stats"
  // zero-mean, unit-variance.
  Matrix<double> global_inv_transform;
  ConvertStatsToInvTransform(global_tot_stats, &global_inv_transform);

  // This is approximately what the (silence, nonsilence) global stats
  // will look like after normalization. 
  Matrix<double> global_norm_sil_stats, global_norm_nonsil_stats;
  TransformStats(global_sil_stats, global_inv_transform,
                 &global_norm_sil_stats);
  TransformStats(global_nonsil_stats, global_inv_transform,
                 &global_norm_nonsil_stats);

  // speaker_target_stats is what the speaker stats would look like if
  // the speech and silence for that speaker were the same as the global
  // averages.  We can treat this as a target.
  Matrix<double> speaker_target_stats(2, dim + 1);
  speaker_target_stats.AddMat(tot_sil_count / global_sil_stats(0, dim),
                              global_sil_stats); // _norm
  speaker_target_stats.AddMat(tot_nonsil_count / global_nonsil_stats(0, dim),
                              global_nonsil_stats); // _norm
  
  Matrix<double> speaker_transform;
  GetTransform(weighted_speaker_stats, speaker_target_stats,
               &speaker_transform);

  ConvertInvTransformToStats(speaker_transform,
                             &final_stats_);
  // give "final_stats" the same count as "weighted_speaker_stats"...
  // we can view this as cosmetic; the count doesn't really matter.
  final_stats_.Scale(weighted_speaker_stats(0, dim));
  return final_stats_; // Return reference to class member.
}


namespace cmvn_utils {
void InitStandardStats(int32 dim,
                       Matrix<double> *target) {
  target->Resize(2, dim + 1);
  for (int32 i = 0; i < dim; i++) {
    (*target)(0, i) = 0.0;
    (*target)(1, i) = 1.0;
  }
  (*target)(0, dim) = 1.0; // the count.
  (*target)(1, dim) = 0.0; // This element is unused.
}

void AddFrameToCmvnStats(BaseFloat weight,
                         VectorBase<BaseFloat> &frame,
                         MatrixBase<double> *stats) {
  KALDI_ASSERT(stats->NumRows() == 2 &&
               stats->NumCols() == frame.Dim() + 1);
  KALDI_ASSERT(weight >= 0.0);
  int32 dim = frame.Dim();
  for (int32 i = 0; i < dim; i++) {
    (*stats)(0, i) += weight * frame(i);
    (*stats)(1, i) += weight * frame(i) * frame(i);
  }
  (*stats)(0, dim) += weight; // The total count.
}

void GetTransform(const MatrixBase<double> &orig_stats,
                  const MatrixBase<double> &target_stats,
                  Matrix<double> *transform) {

  Matrix<double> unit_to_orig, orig_to_unit, unit_to_target;
  ConvertStatsToTransform(orig_stats,
                          &unit_to_orig);
  ConvertStatsToTransform(target_stats,
                          &unit_to_target);
  InvertTransform(unit_to_orig, &orig_to_unit);
  ComposeTransforms(orig_to_unit, unit_to_target, transform);
}


void TransformStats(const MatrixBase<double> &stats,
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
void ConvertStatsToTransform(const MatrixBase<double> &stats,
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
void ConvertStatsToInvTransform(const MatrixBase<double> &stats,
                                Matrix<double> *inv_transform) {
  Matrix<double> transform;
  ConvertStatsToTransform(stats, &transform);
  InvertTransform(transform, inv_transform);
}

void InvertTransform(const MatrixBase<double> &transform,
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

void ComposeTransforms(const MatrixBase<double> &transform1,
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

void ConvertInvTransformToStats(const MatrixBase<double> &transform,
                                Matrix<double> *stats) {
  int32 dim = transform.NumCols();
  KALDI_ASSERT(transform.NumRows() == 2);
  Matrix<double> standard_stats, inv_transform;
  InitStandardStats(dim, &standard_stats);
  InvertTransform(transform, &inv_transform);
  TransformStats(standard_stats, inv_transform, stats);
}




} // namespace cmvn_utils



}  // namespace kaldi


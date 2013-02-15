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

#include "transform/balanced-cmn.h"

namespace kaldi {

BalancedCmn::BalancedCmn(const BalancedCmnConfig &config,
                         const MatrixBase<double> &global_sil_stats,
                         const MatrixBase<double> &global_nonsil_stats):
    config_(config) {
  KALDI_ASSERT(global_sil_stats.NumRows() == 2 &&
               global_nonsil_stats.NumRows() == 2 &&
               global_sil_stats.NumCols() == global_nonsil_stats.NumCols());
  
  int32 dim = global_sil_stats.NumCols() - 1;
  
  speaker_stats_.Resize(3, dim);
  speaker_counts_.Resize(3);

  ComputeTargets(global_sil_stats,
                 global_nonsil_stats);
}

void BalancedCmn::ComputeTargets(const MatrixBase<double> &global_sil_stats,
                                 const MatrixBase<double> &global_nonsil_stats) {
  BaseFloat sil_proportion = config_.silence_proportion,
      nonsil_proportion = 1.0 - sil_proportion;
  // We expect that silence proportion will be quite small, e.g. less than half
  KALDI_ASSERT(sil_proportion >= 0.0 && sil_proportion <= 1.0);
  
  int32 dim = global_sil_stats.NumCols() - 1;
  global_target_sil_.Resize(dim);
  global_target_nonsil_.Resize(dim);
      
  double global_sil_count = global_sil_stats(0, dim),
      global_nonsil_count = global_nonsil_stats(0, dim);
  KALDI_ASSERT(global_sil_count > 0 && global_nonsil_count > 0);

  for (int32 d = 0; d < dim; d++) {
    double sil_avg = global_sil_stats(0, d) / global_sil_count,
        nonsil_avg = global_nonsil_stats(0, d) / global_nonsil_count,
        global_avg = sil_proportion * sil_avg + nonsil_proportion * nonsil_avg;
    // global_avg is the global speech+silence average, pre-normalization, mixed
    // according to the proportion "silence_proportion".  Under certain
    // independence assumptions, this is the offset we would subtract from the
    // global speech and silence statistics by normalizing each speaker to make
    // the mean zero after mixing to get "sil_proportion" as the silence
    // proportion.  So subtracting this gives us the approximate global
    // post-normalization averages of speech and silence.  We'll need this
    // information in case we have insufficient counts for this speaker, to
    // mix the speech and silence according to "silence_proportion".
    // The point at which this happens is controlled by "count_cutoff".
    
    global_target_sil_(d) = sil_avg - global_avg;
    global_target_nonsil_(d) = nonsil_avg - global_avg;

    double check = sil_proportion * global_target_sil_(d) +
        nonsil_proportion * global_target_nonsil_(d);
    // "check" should be zero.
    KALDI_ASSERT(check < 0.001 * fabs(sil_avg));
  }
}


void BalancedCmn::AccStats(const MatrixBase<BaseFloat> &feats,
                           const VectorBase<BaseFloat> &nonsilence_weight) {
  KALDI_ASSERT(feats.NumRows() == nonsilence_weight.Dim() &&
               nonsilence_weight.Min() >= 0.0 &&
               nonsilence_weight.Max() <= 1.0);
  KALDI_ASSERT(feats.NumCols() == Dim());

  for (int32 t = 0; t < feats.NumRows(); t++) {
    SubVector<BaseFloat> x(feats, t);
    BaseFloat nonsil_prob = nonsilence_weight(t),
        sil_prob = 1.0 - nonsil_prob;
    speaker_counts_(0) += nonsil_prob * nonsil_prob;
    speaker_stats_.Row(0).AddVec(nonsil_prob * nonsil_prob, x);
    speaker_counts_(1) += sil_prob * sil_prob;
    speaker_stats_.Row(1).AddVec(sil_prob * sil_prob, x);
    speaker_counts_(2) += sil_prob * nonsil_prob;
    speaker_stats_.Row(2).AddVec(sil_prob * nonsil_prob, x);
  }
}


double BalancedCmn::TotCount() {
  double nonsil_nonsil_count = speaker_counts_(0),
      sil_sil_count= speaker_counts_(1),
      sil_nonsil_count = speaker_counts_(2);
  return  nonsil_nonsil_count + sil_sil_count + 2.0*sil_nonsil_count;
}

void BalancedCmn::GetWeights(double *sil_scale, double *nonsil_scale) {
  double nonsil_nonsil_count = speaker_counts_(0),
      sil_sil_count= speaker_counts_(1),
      sil_nonsil_count = speaker_counts_(2),
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


const Matrix<double> &BalancedCmn::GetStats() {
  KALDI_ASSERT(TotCount() > 0.0);
  int32 dim = Dim();
  double sil_scale, nonsil_scale;
  GetWeights(&sil_scale, &nonsil_scale);

  /// the scale is applied to frames, not stats; if s is the probability
  /// a frame is silence, we weight it by s*sil_scale + (1-s)*nonsil_scale.
  Vector<double> weighted_stats(dim);
  SubVector<double> nonsil_nonsil_stats(speaker_stats_, 0),
      sil_sil_stats(speaker_stats_, 1),
      sil_nonsil_stats(speaker_stats_, 2);
  double nonsil_nonsil_count = speaker_counts_(0),
      sil_sil_count = speaker_counts_(1),
      sil_nonsil_count = speaker_counts_(2);
  
  double sil_count = sil_sil_count * sil_scale + sil_nonsil_count * nonsil_scale,
      nonsil_count = nonsil_nonsil_count * nonsil_scale + sil_nonsil_count * sil_scale,
      tot_count = sil_count + nonsil_count;

  KALDI_VLOG(2) << "After weighting silence frames by " << sil_scale
                << " and non-silence frames by " << nonsil_scale
                << ", counts of silence and non-silence are "
                << sil_count << " and " << nonsil_count
                << ", total is " << tot_count
                << "; proportion of silence is " << (sil_count/tot_count)
                << " vs. target " << config_.silence_proportion;
  
  weighted_stats.AddVec(nonsil_scale, nonsil_nonsil_stats);
  weighted_stats.AddVec(sil_scale, sil_sil_stats);
  weighted_stats.AddVec(nonsil_scale + sil_scale, sil_nonsil_stats);
  
  // target_stats is what we are trying to normalize these stats to.
  // in the normal case, if we have sufficient counts, it will be zero.
  Vector<double> target_stats(dim);
  target_stats.AddVec(sil_count, global_target_sil_);
  target_stats.AddVec(nonsil_count, global_target_nonsil_);

  // scale both types of stats so they're like data averages.
  weighted_stats.Scale(1.0 / tot_count);
  target_stats.Scale(1.0 / tot_count);
  
  if (sil_scale > 0.0 && nonsil_scale > 0.0 &&
      sil_count + nonsil_count > 1.1 * config_.count_cutoff &&
      sil_count > 0.0 && nonsil_count > 0.0) {
    // we're not in any kind of edge case, so the target should be zero.
    KALDI_ASSERT(target_stats.Norm(2.0) < 0.1);
  }

  // After the following statement, weighted_stats will be the
  // negative of the offset we want to apply to the data to normalize it.
  weighted_stats.AddVec(-1.0, target_stats);

  final_stats_.Resize(1, dim + 1);
  final_stats_.Row(0).Range(0, dim).CopyFromVec(weighted_stats);
  final_stats_(0, dim) = 1.0; // a "fake count" of 1, which will
  // cause the code in apply-cmvn to interpret final_stats as a
  // negative offset.
  return final_stats_;
}



}  // namespace kaldi


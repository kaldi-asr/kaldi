// gmm/ebw-diag-gmm.cc

// Copyright 2009-2011  Arnab Ghoshal, Petr Motlicek

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

#include <algorithm>  // for std::max
#include <string>
#include <vector>

#include "gmm/diag-gmm.h"
#include "gmm/ebw-diag-gmm.h"

namespace kaldi {

// This function is used inside the EBW update routines.
// returns true if all variances were positive.
static bool EBWUpdateGaussian(
    BaseFloat D,
    GmmFlagsType flags,
    const VectorBase<double> &orig_mean,
    const VectorBase<double> &orig_var,
    const VectorBase<double> &x_stats,
    const VectorBase<double> &x2_stats,
    double occ,
    VectorBase<double> *mean,
    VectorBase<double> *var,
    double *auxf_impr) {
  if (! (flags&(kGmmMeans|kGmmVariances))) { // nothing to do.
    if (auxf_impr) *auxf_impr = 0.0;
    mean->CopyFromVec(orig_mean);
    var->CopyFromVec(orig_var);
    return true; 
  }   
  KALDI_ASSERT(!( (flags&kGmmVariances) && !(flags&kGmmMeans))
               && "We didn't make the update cover this case sensibly (update vars not means)");
  
  mean->SetZero();
  var->SetZero();
  mean->AddVec(D, orig_mean);
  var->AddVec2(D, orig_mean);
  var->AddVec(D, orig_var);
  mean->AddVec(1.0, x_stats);
  var->AddVec(1.0, x2_stats);
  BaseFloat scale = 1.0 / (occ + D);
  mean->Scale(scale);
  var->Scale(scale);
  var->AddVec2(-1.0, *mean);
  
  if (!(flags&kGmmVariances)) var->CopyFromVec(orig_var);
  if (!(flags&kGmmMeans)) mean->CopyFromVec(orig_mean);

  // Return false if any NaN's.
  for (int32 i = 0; i < mean->Dim(); i++) {
    double m =  ((*mean)(i)), v = ((*var)(i));
    if (m!=m || v!=v || m-m != 0 || v-v != 0) {
      return false;
    }
  }
  
  if (var->Min() > 0.0) {
    if (auxf_impr != NULL) {
      // work out auxf improvement.  
      BaseFloat old_auxf = 0.0, new_auxf = 0.0;
      int32 dim = orig_mean.Dim();
      for (int32 i = 0; i < dim; i++) {
        BaseFloat mean_diff = (*mean)(i) - orig_mean(i);
        old_auxf += (occ+D) * -0.5 * (log(orig_var(i)) +
                                      ((*var)(i) + mean_diff*mean_diff)
                                      / orig_var(i));
        new_auxf += (occ+D) * -0.5 * (log((*var)(i)) + 1.0);
        
      }
      *auxf_impr = new_auxf - old_auxf;
    }
    return true;
  } else return false;
}

// Update Gaussian parameters only (no weights)
void UpdateEbwDiagGmm(const AccumDiagGmm &num_stats, // with I-smoothing, if used.
                      const AccumDiagGmm &den_stats,
                      GmmFlagsType flags,
                      const EbwOptions &opts,
                      DiagGmm *gmm,
                      BaseFloat *auxf_change_out,
                      BaseFloat *count_out,
                      int32 *num_floored_out) {
  GmmFlagsType acc_flags = num_stats.Flags();
  if (flags & ~acc_flags)
    KALDI_ERR << "Incompatible flags: you requested to update flags \""
              << GmmFlagsToString(flags) << "\" but accumulators have only \""
              << GmmFlagsToString(acc_flags) << '"';
  
  // It could be that the num stats actually contain the difference between
  // num and den (for mean and var stats), and den stats only have the weights.
  bool den_has_stats;
  if (den_stats.Flags() != acc_flags) {
    den_has_stats = false;
    if (den_stats.Flags() != kGmmWeights) 
      KALDI_ERR << "Incompatible flags: num stats have flags \""
                << GmmFlagsToString(acc_flags) << "\" vs. den stats \""
                << GmmFlagsToString(den_stats.Flags()) << '"';
  } else {
    den_has_stats = true;
  }
  int32 num_comp = num_stats.NumGauss();
  int32 dim = num_stats.Dim();
  KALDI_ASSERT(num_stats.NumGauss() == den_stats.NumGauss());
  KALDI_ASSERT(num_stats.Dim() == gmm->Dim());
  KALDI_ASSERT(gmm->NumGauss() == num_comp);
  
  if ( !(flags & (kGmmMeans | kGmmVariances)) ) {
    return; // Nothing to update.
  }
  
  // copy DiagGMM model and transform this to the normal case
  DiagGmmNormal diaggmmnormal;
  gmm->ComputeGconsts();
  diaggmmnormal.CopyFromDiagGmm(*gmm);

  // go over all components
  Vector<double> mean(dim), var(dim), mean_stats(dim), var_stats(dim);

  for (int32 g = 0; g < num_comp; g++) {
    BaseFloat num_count = num_stats.occupancy()(g),
        den_count = den_stats.occupancy()(g);
    if (num_count == 0.0 && den_count == 0.0) {
      KALDI_VLOG(2) << "Not updating Gaussian " << g << " since counts are zero";
      continue;
    }
    mean_stats.CopyFromVec(num_stats.mean_accumulator().Row(g));
    if (den_has_stats)
      mean_stats.AddVec(-1.0, den_stats.mean_accumulator().Row(g));
    if (flags & kGmmVariances) {
      var_stats.CopyFromVec(num_stats.variance_accumulator().Row(g));
      if (den_has_stats)
        var_stats.AddVec(-1.0, den_stats.variance_accumulator().Row(g));
    }
    double D = (opts.tau + opts.E * den_count) / 2;
    if (D+num_count-den_count <= 0.0) {
      // ensure +ve-- can be problem if num count == 0 and E=2.
      D = -1.0001*(num_count-den_count) + 1.0e-10;
      KALDI_ASSERT(D+num_count-den_count > 0.0);
    }
    // We initialize to half the value of D that would be dictated by E (and
    // tau); this is part of the strategy used to ensure that the value of D we
    // use is at least twice the value that would ensure positive variances.

    int32 iter, max_iter = 100;
    for (iter = 0; iter < max_iter; iter++) { // will normally break from the loop
      // the first time.
      if (EBWUpdateGaussian(D, flags,
                            diaggmmnormal.means_.Row(g),
                            diaggmmnormal.vars_.Row(g),
                            mean_stats, var_stats, num_count-den_count,
                            &mean, &var, NULL)) {
        // Succeeded in getting all +ve vars at this value of D.
        // So double D and commit changes.
        D *= 2.0;
        double auxf_impr = 0.0;
        bool ans = EBWUpdateGaussian(D, flags,
                                     diaggmmnormal.means_.Row(g),
                                     diaggmmnormal.vars_.Row(g),
                                     mean_stats, var_stats, num_count-den_count,
                                     &mean, &var, &auxf_impr);
        if (!ans) {
          KALDI_WARN << "Something went wrong in the EBW update. Check that your"
              "previous update phase looks reasonable, probably your model is "
              "already ruined.  Reverting to the old values";
        } else {
          if (auxf_change_out) *auxf_change_out += auxf_impr;
          if (count_out) *count_out += den_count; // The idea is that for MMI, this will
          // reflect the actual #frames trained on (the numerator one would be I-smoothed).
          // In general (e.g. for MPE), we won't know the #frames.
          diaggmmnormal.means_.CopyRowFromVec(mean, g);
          diaggmmnormal.vars_.CopyRowFromVec(var, g);
        }
        break;
      } else {
        // small step
        D *= 1.1; 
      }
    }
    if (iter > 0 && num_floored_out != NULL) (*num_floored_out)++;
    if (iter == max_iter) KALDI_WARN << "Dropped off end of loop, recomputing D. (unexpected.)";
  }
  // copy to natural representation according to flags.
  diaggmmnormal.CopyToDiagGmm(gmm, flags);
  gmm->ComputeGconsts();
}


void UpdateEbwWeightsDiagGmm(const AccumDiagGmm &num_stats, // should have no I-smoothing
                             const AccumDiagGmm &den_stats,
                             const EbwWeightOptions &opts,
                             DiagGmm *gmm,
                             BaseFloat *auxf_change_out,
                             BaseFloat *count_out) {

  DiagGmmNormal diaggmmnormal;
  gmm->ComputeGconsts();
  diaggmmnormal.CopyFromDiagGmm(*gmm);

  Vector<double> weights(diaggmmnormal.weights_),
      num_occs(num_stats.occupancy()),
      den_occs(den_stats.occupancy());
  if (opts.tau == 0.0 &&
      num_occs.Sum() + den_occs.Sum() < opts.min_num_count_weight_update) {
    KALDI_LOG << "Not updating weights for this state because total count is "
              << num_occs.Sum() + den_occs.Sum() << " < "
              << opts.min_num_count_weight_update;
    if (count_out)
      *count_out += num_occs.Sum();
    return;
  }
  num_occs.AddVec(opts.tau, weights);
  KALDI_ASSERT(weights.Dim() == num_occs.Dim() && num_occs.Dim() == den_occs.Dim());
  if (weights.Dim() == 1) return; // Nothing to do: only one mixture.
  double weight_auxf_at_start = 0.0, weight_auxf_at_end = 0.0;
  
  int32 num_comp = weights.Dim();
  for (int32 g = 0; g < num_comp; g++) {   // c.f. eq. 4.32 in Dan Povey's thesis.
    weight_auxf_at_start +=
        num_occs(g) * log (weights(g))
        - den_occs(g) * weights(g) / diaggmmnormal.weights_(g);
  }
  for (int32 iter = 0; iter < 50; iter++) {
    Vector<double> k_jm(num_comp); // c.f. eq. 4.35
    double max_m = 0.0;
    for (int32 g = 0; g < num_comp; g++)
      max_m = std::max(max_m, den_occs(g)/diaggmmnormal.weights_(g));
    for (int32 g = 0; g < num_comp; g++)
      k_jm(g) = max_m - den_occs(g)/diaggmmnormal.weights_(g);
    for (int32 g = 0; g < num_comp; g++) // c.f. eq. 4.34
      weights(g) = num_occs(g) + k_jm(g)*weights(g);
    weights.Scale(1.0 / weights.Sum()); // c.f. eq. 4.34 (denominator)
  }
  for (int32 g = 0; g < num_comp; g++) {   // weight flooring.
    if (weights(g) < opts.min_gaussian_weight)
      weights(g) = opts.min_gaussian_weight;
  }
  weights.Scale(1.0 / weights.Sum()); // renormalize after flooring..
  // floor won't be exact now but doesn't really matter.

  for (int32 g = 0; g < num_comp; g++) {   // c.f. eq. 4.32 in Dan Povey's thesis.
    weight_auxf_at_end +=
        num_occs(g) * log (weights(g))
        - den_occs(g) * weights(g) / diaggmmnormal.weights_(g);
  }

  if (auxf_change_out)
    *auxf_change_out += weight_auxf_at_end - weight_auxf_at_start;
  if (count_out)
    *count_out += num_occs.Sum(); // only really valid for MMI [not MPE, or MMI
  // with canceled stats]
  
  diaggmmnormal.weights_.CopyFromVec(weights);

  // copy to natural representation
  diaggmmnormal.CopyToDiagGmm(gmm, kGmmAll);
  gmm->ComputeGconsts();
}

void UpdateEbwAmDiagGmm(const AccumAmDiagGmm &num_stats, // with I-smoothing, if used.
                        const AccumAmDiagGmm &den_stats,
                        GmmFlagsType flags,
                        const EbwOptions &opts,
                        AmDiagGmm *am_gmm,
                        BaseFloat *auxf_change_out,
                        BaseFloat *count_out,
                        int32 *num_floored_out) {
  KALDI_ASSERT(num_stats.NumAccs() == den_stats.NumAccs()
               && num_stats.NumAccs() == am_gmm->NumPdfs());

  if (auxf_change_out) *auxf_change_out = 0.0;
  if (count_out) *count_out = 0.0;
  if (num_floored_out) *num_floored_out = 0.0;

  for (int32 pdf = 0; pdf < num_stats.NumAccs(); pdf++)
    UpdateEbwDiagGmm(num_stats.GetAcc(pdf), den_stats.GetAcc(pdf), flags,
                     opts, &(am_gmm->GetPdf(pdf)), auxf_change_out,
                     count_out, num_floored_out);
}                     


void UpdateEbwWeightsAmDiagGmm(const AccumAmDiagGmm &num_stats, // with I-smoothing, if used.
                               const AccumAmDiagGmm &den_stats,
                               const EbwWeightOptions &opts,
                               AmDiagGmm *am_gmm,
                               BaseFloat *auxf_change_out,
                               BaseFloat *count_out) {
  KALDI_ASSERT(num_stats.NumAccs() == den_stats.NumAccs()
               && num_stats.NumAccs() == am_gmm->NumPdfs());

  if (auxf_change_out) *auxf_change_out = 0.0;
  if (count_out) *count_out = 0.0;
  
  for (int32 pdf = 0; pdf < num_stats.NumAccs(); pdf++)
    UpdateEbwWeightsDiagGmm(num_stats.GetAcc(pdf), den_stats.GetAcc(pdf),
                            opts, &(am_gmm->GetPdf(pdf)), auxf_change_out,
                            count_out);
}                     

void IsmoothStatsDiagGmm(const AccumDiagGmm &src_stats,
                         double tau,
                         AccumDiagGmm *dst_stats) {
  KALDI_ASSERT(src_stats.NumGauss() == dst_stats->NumGauss());
  int32 dim = src_stats.Dim(), num_gauss = src_stats.NumGauss();
  for (int32 g = 0; g < num_gauss; g++) {
    double occ = src_stats.occupancy()(g);
    if (occ != 0.0) { // can only do this for nonzero occupancies...
      Vector<double> x_stats(dim), x2_stats(dim);
      if (dst_stats->Flags() & kGmmMeans)
        x_stats.CopyFromVec(src_stats.mean_accumulator().Row(g));
      if (dst_stats->Flags() & kGmmVariances)
        x2_stats.CopyFromVec(src_stats.variance_accumulator().Row(g));
      x_stats.Scale(tau / occ);
      x2_stats.Scale(tau / occ);
      dst_stats->AddStatsForComponent(g, tau, x_stats, x2_stats);
    }
  }
}

/// Creates stats from the GMM.  Resizes them as needed.
void DiagGmmToStats(const DiagGmm &gmm,
                    GmmFlagsType flags,
                    double state_occ,
                    AccumDiagGmm *dst_stats) {
  dst_stats->Resize(gmm, AugmentGmmFlags(flags));
  int32 num_gauss = gmm.NumGauss(), dim = gmm.Dim();
  DiagGmmNormal gmmnormal(gmm);
  Vector<double> x_stats(dim), x2_stats(dim);
  for (int32 g = 0; g < num_gauss; g++) {
    double occ = state_occ * gmmnormal.weights_(g);
    x_stats.SetZero();
    x_stats.AddVec(occ, gmmnormal.means_.Row(g));
    x2_stats.SetZero();
    x2_stats.AddVec2(occ, gmmnormal.means_.Row(g));
    x2_stats.AddVec(occ, gmmnormal.vars_.Row(g));
    dst_stats->AddStatsForComponent(g, occ, x_stats, x2_stats);
  }
}

void IsmoothStatsAmDiagGmm(const AccumAmDiagGmm &src_stats,
                           double tau,
                           AccumAmDiagGmm *dst_stats) {
  int num_pdfs = src_stats.NumAccs();
  KALDI_ASSERT(num_pdfs == dst_stats->NumAccs());
  for (int32 pdf = 0; pdf < num_pdfs; pdf++)
    IsmoothStatsDiagGmm(src_stats.GetAcc(pdf), tau, &(dst_stats->GetAcc(pdf)));
}

void IsmoothStatsAmDiagGmmFromModel(const AmDiagGmm &src_model,
                                    double tau,
                                    AccumAmDiagGmm *dst_stats) {
  int num_pdfs = src_model.NumPdfs();
  KALDI_ASSERT(num_pdfs == dst_stats->NumAccs());
  for (int32 pdf = 0; pdf < num_pdfs; pdf++) {
    AccumDiagGmm tmp_stats;
    double occ = 1.0; // its value doesn't matter.
    DiagGmmToStats(src_model.GetPdf(pdf), kGmmAll, occ, &tmp_stats);
    IsmoothStatsDiagGmm(tmp_stats, tau, &(dst_stats->GetAcc(pdf)));
  }
}



}  // End of namespace kaldi

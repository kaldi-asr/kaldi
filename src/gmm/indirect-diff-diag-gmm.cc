// gmm/indirect-diff-diag-gmm.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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

#include "gmm/indirect-diff-diag-gmm.h"

namespace kaldi {


void GetSingleStatsDerivative(
    double ml_count, double ml_x_stats, double ml_x2_stats,
    double disc_count, double disc_x_stats, double disc_x2_stats,
    double model_mean, double model_var, BaseFloat min_variance,
    double *ml_x_stats_deriv, double *ml_x2_stats_deriv) {
  
  double model_inv_var = 1.0/model_var,
      model_inv_var_sq = model_inv_var*model_inv_var,
      model_mean_sq = model_mean*model_mean;

  // First get derivative of discriminative objective function w.r.t. the
  // model mean and variance.
  // Below: eqs. 11 and 13 in 2005 ICASSP paper on fMPE.  Note: the factor of
  // kappa (in the fMPE case) is assumed to have been accounted for by
  // scaling the num and den accs at the command-line level.  We substituted
  // eq. 12 into 13 and rearranged to get the second expression.
  double diff_wrt_model_mean = (1.0/model_var) * (disc_x_stats - model_mean*disc_count),
      diff_wrt_model_var =
      0.5 * ((disc_x2_stats - 2*model_mean*disc_x_stats + disc_count*model_mean_sq)
             * model_inv_var_sq
             - disc_count*model_inv_var);

  double stats_mean = ml_x_stats / ml_count,
      stats_var = ml_x2_stats / ml_count - (ml_x_stats / ml_count)*(ml_x_stats / ml_count);

  // We assume the "rescaling" update will be as follows.  Apologies if this is
  // a bit confusing.  The idea is that if the mean and var from (stats versus
  // model) differ we assume that the model will be updated with
  // DoRescalingUpdate(), which takes two sets of ML accs (old and new).  The old ML
  // accs given to the update will be the current ml accumulators we have here in
  // this function, and the new ML accs will be affected by change in fMPE transform.
  // The update in DoRescalingUpdate() will preserve any current difference between
  // the ml stats and the model [represented as a shift in mean and factor in variance].
  // Concretely: the update in DoRescalingUpdate() will do:
  //
  // new_model_mean := old_model_mean + new_stats_mean - old_stats_mean   (eq. 1)
  // new_model_var := max(min_variance, old_model_var * new_stats_var / old_stats_var).  (eq. 2)
  //
  // We're differentiating back through this process to new_stats_mean.
  // If the model and the stats were actually the same (e.g. we had been doing ML updates),
  // then all this is equivalent to what was in the original fMPE paper.  It's just
  // extended to make sense outside of that scenario where you're doing ML.
  
  double diff_wrt_stats_mean = diff_wrt_model_mean; // This comes from eq. 1 above.
  double diff_wrt_stats_var;
  if (model_var <= min_variance*1.01) {
    diff_wrt_stats_var = 0.0; // model would be "pinned" at minimum variance.
    KALDI_VLOG(2) << "Variance derivative is zero (min variance)";
  } else {
    diff_wrt_stats_var = diff_wrt_model_var * model_var / stats_var; // note:
    // the factor "model_var / stats_var" comes from "old_model_var / old_stats_var" in eq. 2.
    // Also note: the {old_,new_} versions of variables are numerically the same here, at the
    // point where we're differentiating.
  }

  // The next equations don't appear in the paper but represent the backpropagation
  // of the derivative through the equations:
  // stats_mean := ml_x_stats / ml_count
  // stats_var := ml_x2_stats / ml_count - (ml_x_stats/ml_count)^2
  // [we use stats_mean = ml_x_stats/ml_count, here].
  *ml_x_stats_deriv = diff_wrt_stats_mean / ml_count - 2 * diff_wrt_stats_var * stats_mean / ml_count;
  *ml_x2_stats_deriv = diff_wrt_stats_var / ml_count;
}




// The function for just one GMM.  We don't export it as it's not currently
// needed outside this file.
void GetStatsDerivative(const DiagGmm &gmm,
                        const AccumDiagGmm &num_acc,
                        const AccumDiagGmm &den_acc,
                        const AccumDiagGmm &ml_acc,
                        BaseFloat min_variance,
                        BaseFloat min_gaussian_occupancy,
                        AccumDiagGmm *out_accs) {
  out_accs->Resize(gmm, kGmmAll);
  int32 num_gauss = gmm.NumGauss(), dim = gmm.Dim();
  KALDI_ASSERT(num_gauss == num_acc.NumGauss() && dim == num_acc.Dim());
  KALDI_ASSERT(num_gauss == den_acc.NumGauss()); // don't check den dim--
  // in the "compressed" form of stats (where num acc stores diff),
  // it could be zero.
  KALDI_ASSERT(num_gauss == ml_acc.NumGauss() && dim == ml_acc.Dim());

  KALDI_ASSERT((ml_acc.Flags() & (kGmmMeans|kGmmVariances)) ==
               (kGmmMeans|kGmmVariances));
  KALDI_ASSERT((num_acc.Flags() & (kGmmMeans|kGmmVariances)) ==
               (kGmmMeans|kGmmVariances));
  DiagGmmNormal gmm_normal(gmm);

  // if have_den_stats == false, we assume the num and den have been
  // "compressed" by putting the difference in mean and var stats in num.
  bool have_den_stats = ((den_acc.Flags() & (kGmmMeans|kGmmVariances)) != 0);

  for (int32 gauss = 0; gauss < num_gauss; gauss++) {
    Vector<double> x_stats_deriv(dim), x2_stats_deriv(dim);
    double num_count = num_acc.occupancy()(gauss),
        den_count = den_acc.occupancy()(gauss),
        ml_count = ml_acc.occupancy()(gauss);
    
    if (ml_count <= min_gaussian_occupancy) {
      // This Gaussian won't be updated since has small count
      KALDI_WARN << "Skipping Gaussian because very small ML count: (num,den,ml) = "
                 << num_count << ", " << den_count << ", " << ml_count;
    } else {
      double disc_count = num_count - den_count;
      for (int32 d = 0; d < dim; d++) {
        double disc_x_acc = num_acc.mean_accumulator()(gauss, d)
            - (have_den_stats ? den_acc.mean_accumulator()(gauss, d) : 0.0),
            disc_x2_acc = num_acc.variance_accumulator()(gauss, d)
            - (have_den_stats ? den_acc.variance_accumulator()(gauss, d) : 0.0),
            ml_x_acc = ml_acc.mean_accumulator()(gauss, d),
            ml_x2_acc = ml_acc.variance_accumulator()(gauss, d),
            model_mean = gmm_normal.means_(gauss, d),
            model_var = gmm_normal.vars_(gauss, d);

        double x_acc_deriv = 0.0, x2_acc_deriv = 0.0;
        GetSingleStatsDerivative(ml_count, ml_x_acc, ml_x2_acc,
                                 disc_count, disc_x_acc, disc_x2_acc,
                                 model_mean, model_var, min_variance,
                                 &x_acc_deriv, &x2_acc_deriv);

        x_stats_deriv(d) = x_acc_deriv;
        x2_stats_deriv(d) = x2_acc_deriv;
      }
      // set the stats to these quantities (we're adding, but the stats
      // are currently zero).
      out_accs->AddStatsForComponent(gauss, 0.0, x_stats_deriv, x2_stats_deriv);
    }
  }
}

void GetStatsDerivative(const AmDiagGmm &gmm,
                        const AccumAmDiagGmm &num_accs, // for MMI, would equal ml accs.
                        const AccumAmDiagGmm &den_accs,
                        const AccumAmDiagGmm &ml_accs,
                        BaseFloat min_variance,
                        BaseFloat min_gaussian_occupancy,
                        AccumAmDiagGmm *out_accs) {
  out_accs->Init(gmm, kGmmAll);
  int32 num_pdfs = gmm.NumPdfs();
  KALDI_ASSERT(num_accs.NumAccs() == num_pdfs);
  KALDI_ASSERT(den_accs.NumAccs() == num_pdfs);
  KALDI_ASSERT(ml_accs.NumAccs() == num_pdfs);
  for (int32 pdf = 0; pdf < num_pdfs; pdf++)
    GetStatsDerivative(gmm.GetPdf(pdf), num_accs.GetAcc(pdf), den_accs.GetAcc(pdf),
                       ml_accs.GetAcc(pdf), min_variance, min_gaussian_occupancy,
                       &(out_accs->GetAcc(pdf)));
  
}


void DoRescalingUpdate(const AccumDiagGmm &old_ml_acc,
                       const AccumDiagGmm &new_ml_acc,
                       BaseFloat min_variance,
                       BaseFloat min_gaussian_occupancy,
                       DiagGmm *gmm,
                       double *tot_count,
                       double *tot_divergence) {
  int32 num_gauss = gmm->NumGauss(), dim = gmm->Dim();
  KALDI_ASSERT(old_ml_acc.NumGauss() == num_gauss &&
               old_ml_acc.Dim() == dim);
  KALDI_ASSERT(new_ml_acc.NumGauss() == num_gauss &&
               new_ml_acc.Dim() == dim);
  KALDI_ASSERT((old_ml_acc.Flags() & (kGmmMeans|kGmmVariances)) ==
               (kGmmMeans|kGmmVariances));
  KALDI_ASSERT((new_ml_acc.Flags() & (kGmmMeans|kGmmVariances)) ==
               (kGmmMeans|kGmmVariances));

  DiagGmmNormal gmm_normal(*gmm);
  for (int32 gauss = 0; gauss < num_gauss; gauss++) {
    double old_ml_count = old_ml_acc.occupancy()(gauss),
        new_ml_count = new_ml_acc.occupancy()(gauss);
    if (old_ml_count <= min_gaussian_occupancy ||
        new_ml_count <= min_gaussian_occupancy) {
      KALDI_WARN << "Gaussian being skipped because it has small count: (old,new) = "
                 << old_ml_count << ", " << new_ml_count;
      continue;
    }
    *tot_count += new_ml_count;
    for (int32 d = 0; d < dim; d++) {
      double old_model_mean = gmm_normal.means_(gauss, d),
          old_model_var = gmm_normal.vars_(gauss, d),
          old_ml_mean = old_ml_acc.mean_accumulator()(gauss, d) / old_ml_count,
          old_ml_var = old_ml_acc.variance_accumulator()(gauss, d) / old_ml_count
          - old_ml_mean*old_ml_mean,
          new_ml_mean = new_ml_acc.mean_accumulator()(gauss, d) / new_ml_count,
          new_ml_var = new_ml_acc.variance_accumulator()(gauss, d) / new_ml_count
          - new_ml_mean*new_ml_mean,
          new_model_mean = old_model_mean + new_ml_mean - old_ml_mean,
          new_model_var = std::max(static_cast<double>(min_variance),
                                   old_model_var * new_ml_var / old_ml_var);
      double divergence = 
          0.5 *(((new_model_mean-old_model_mean)*(new_model_mean-old_model_mean) +
                 new_model_var - old_model_var)/old_model_var +
                log(old_model_var / new_model_var));
      if (divergence < 0.0)
        KALDI_WARN << "Negative divergence " << divergence;
      *tot_divergence += divergence * new_ml_count;
      gmm_normal.means_(gauss, d) = new_model_mean;
      gmm_normal.vars_(gauss, d) = new_model_var;
    }
  }
  gmm_normal.CopyToDiagGmm(gmm);
}


void DoRescalingUpdate(const AccumAmDiagGmm &old_ml_accs,
                       const AccumAmDiagGmm &new_ml_accs,
                       BaseFloat min_variance,
                       BaseFloat min_gaussian_occupancy,
                       AmDiagGmm *am_gmm) {
  int32 num_pdfs = am_gmm->NumPdfs();
  KALDI_ASSERT(old_ml_accs.NumAccs() == num_pdfs);
  KALDI_ASSERT(new_ml_accs.NumAccs() == num_pdfs);
  double tot_count = 0.0, tot_divergence = 0.0;
  for (int32 pdf = 0; pdf < num_pdfs; pdf++)
    DoRescalingUpdate(old_ml_accs.GetAcc(pdf), new_ml_accs.GetAcc(pdf),
                      min_variance, min_gaussian_occupancy, &am_gmm->GetPdf(pdf),
                      &tot_count, &tot_divergence);
  KALDI_LOG << "K-L divergence from old to new model is "
            << (tot_divergence/tot_count) << " over "
            << tot_count << " frames.";
  am_gmm->ComputeGconsts();
}



}  // End of namespace kaldi

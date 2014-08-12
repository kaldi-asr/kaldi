// transform/fmpe-test.cc

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

#include "util/common-utils.h"
#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/model-test-common.h"
#include "transform/fmpe.h"

namespace kaldi {


// Compute derivative of GMM log-likelihood w.r.t. features.
// Note: this code copied from gmm-get-feat-deriv.cc; had
// to simplify a bit.
void GetFeatDeriv(const DiagGmm &gmm,
                  const Matrix<BaseFloat> &feats,
                  Matrix<BaseFloat> *deriv) {
  
  deriv->Resize(feats.NumRows(), feats.NumCols());

  Vector<BaseFloat> gauss_posteriors;
  Vector<BaseFloat> temp_vec(feats.NumCols());
  for (int32 i = 0; i < feats.NumRows(); i++) {
    SubVector<BaseFloat> this_feat(feats, i);
    SubVector<BaseFloat> this_deriv(*deriv, i);
    gmm.ComponentPosteriors(this_feat, &gauss_posteriors);
    BaseFloat weight = 1.0;
    gauss_posteriors.Scale(weight);
    // The next line does: to i'th row of deriv, add
    // means_invvars^T * gauss_posteriors,
    // where each row of means_invvars is the mean times
    // diagonal inverse covariance... after transposing,
    // this becomes a weighted of these rows, weighted by
    // the posteriors.  This comes from the term
    //  feat^T * inv_var * mean
    // in the objective function.
    this_deriv.AddMatVec(1.0, gmm.means_invvars(), kTrans,
                         gauss_posteriors, 1.0);

    // next line does temp_vec == inv_vars^T * gauss_posteriors,
    // which sets temp_vec to a weighted sum of the inv_vars,
    // weighed by Gaussian posterior.
    temp_vec.AddMatVec(1.0, gmm.inv_vars(), kTrans,
                       gauss_posteriors, 0.0);
    // Add to the derivative, -(this_feat .* temp_vec),
    // which is the term that comes from the -0.5 * inv_var^T feat_sq,
    // in the objective function (where inv_var is a vector, and feat_sq
    // is a vector of squares of the feature values).
    this_deriv.AddVecVec(-1.0, this_feat, temp_vec, 1.0);
  }
}

// Gets total log-likelihood, summed over all frames.
BaseFloat GetGmmLike(const DiagGmm &gmm,
                     const Matrix<BaseFloat> &feats) {
  BaseFloat ans = 0.0;
  for (int32 i = 0; i < feats.NumRows(); i++)
    ans += gmm.LogLikelihood(feats.Row(i));
  return ans;
}

void TestFmpe() {
  int32 dim = 10 + (Rand() % 10);
  int32 num_comp = 10 + (Rand() % 10);
  DiagGmm gmm;
  unittest::InitRandDiagGmm(dim, num_comp, &gmm);
  
  int32 num_frames = 20;
  Matrix<BaseFloat> feats(num_frames, dim);

  for (int32 i = 0; i < num_frames; i++)
    for (int32 j = 0; j < dim; j++)
      feats(i, j) = RandGauss();

  FmpeOptions opts; // Default.
  {
    Fmpe fmpe(gmm, opts);
    {
      bool binary = (Rand() % 2 == 1);
      Output ko("tmpf", binary);
      fmpe.Write(ko.Stream(), binary);
    }
  }
  Fmpe fmpe(gmm, opts);
  {
    bool binary_in;
    Input ki("tmpf", &binary_in);
    fmpe.Read(ki.Stream(), binary_in);
  }

  // We'll first be testing that the feature derivative is
  // accurate, by measuring a small random offset in feature space.
  {
    Matrix<BaseFloat> deriv;
    Matrix<BaseFloat> random_offset(feats.NumRows(), feats.NumCols());
    for (int32 i = 0; i < feats.NumRows(); i++)
      for (int32 j = 0; j < feats.NumCols(); j++)
        random_offset(i, j) = 1.0e-03 * RandGauss();
    BaseFloat like_before = GetGmmLike(gmm, feats);
    feats.AddMat(1.0, random_offset);
    BaseFloat like_after = GetGmmLike(gmm, feats);
    feats.AddMat(-1.0, random_offset); // undo the change.
    GetFeatDeriv(gmm, feats, &deriv);
    BaseFloat change1 = like_after - like_before,
        change2 = TraceMatMat(random_offset, deriv, kTrans);
    KALDI_LOG << "Random offset led to like change "
              << change1 << " (manually), and " << change2
              << " (derivative)";
    // note: not making this threshold smaller, as don't want
    // spurious failures.  Seems to be OK though.
    KALDI_ASSERT( fabs(change1-change2) < 0.15*fabs(change1+change2));
  }

  std::vector<std::vector<int32> > gselect(feats.NumRows()); // make it have all Gaussians...
  for (int32 i = 0; i < feats.NumRows(); i++)
    for (int32 j = 0; j < gmm.NumGauss(); j++)
      gselect[i].push_back(j);

  Matrix<BaseFloat> fmpe_offset;
  // Check that the fMPE feature offset is zero.
  fmpe.ComputeFeatures(feats, gselect, &fmpe_offset);
  KALDI_ASSERT(fmpe_offset.IsZero());
  
  // Note: we're just using the ML objective function here.
  // This is just to make sure the derivatives are all computed
  // correctly.
  BaseFloat like_before_update = GetGmmLike(gmm, feats);
  // Now get stats for update.
  FmpeStats stats(fmpe);
  Matrix<BaseFloat> deriv;
  GetFeatDeriv(gmm, feats, &deriv);
  fmpe.AccStats(feats, gselect, deriv, NULL, &stats);
  FmpeUpdateOptions update_opts;
  update_opts.learning_rate = 0.001; // so linear assumption is more valid.
  BaseFloat delta = fmpe.Update(update_opts, stats);

  fmpe.ComputeFeatures(feats, gselect, &fmpe_offset);
  feats.AddMat(1.0, fmpe_offset);

  BaseFloat like_after_update = GetGmmLike(gmm, feats);

  BaseFloat delta2 = like_after_update - like_before_update;
  KALDI_LOG << "Change predicted by fMPE Update function is "
            << delta << ", change computed directly is "
            << delta2;
  KALDI_ASSERT(fabs(delta-delta2) < 0.15 * fabs(delta+delta2));
}

}


int main() {
  kaldi::g_kaldi_verbose_level = 5;
  for (int i = 0; i <= 10; i++)
    kaldi::TestFmpe();
  std::cout << "Test OK.\n";
}


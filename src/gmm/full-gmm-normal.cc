// gmm/full-gmm-normal.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Yanmin Qian
//                      Univ. Erlangen-Nuremberg, Korbinian Riedhammer
//                2013  Johns Hopkins University (author: Daniel Povey)

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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "gmm/full-gmm-normal.h"
#include "gmm/full-gmm.h"

namespace kaldi {

void FullGmmNormal::Resize(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);

  if (weights_.Dim() != nmix)
    weights_.Resize(nmix);

  if (means_.NumRows() != nmix ||
      means_.NumCols() != dim)
    means_.Resize(nmix, dim);

  if (vars_.size() != nmix)
    vars_.resize(nmix);
  for (int32 i = 0; i < nmix; i++) {
    if (vars_[i].NumRows() != nmix ||
        vars_[i].NumCols() != dim) {
      vars_[i].Resize(dim);
    }
  }
}

void FullGmmNormal::CopyFromFullGmm(const FullGmm &fullgmm) {
  /// resize the variables to fit the gmm
  size_t dim = fullgmm.Dim();
  size_t num_gauss = fullgmm.NumGauss();
  Resize(num_gauss, dim);

  /// copy weights
  weights_.CopyFromVec(fullgmm.weights_);

  /// we need to split the natural components for each gaussian
  Vector<double> mean_times_invcovar(dim);

  for (size_t i = 0; i < num_gauss; i++) {
    // copy and invert (inverse) covariance matrix
    vars_[i].CopyFromSp(fullgmm.inv_covars_[i]);
    vars_[i].InvertDouble();

    // multiply the (mean x icov) by (cov) to get the means back
    mean_times_invcovar.CopyFromVec(fullgmm.means_invcovars_.Row(i));
    (means_.Row(i)).AddSpVec(1.0, vars_[i], mean_times_invcovar, 0.0);
  }
}

void FullGmmNormal::CopyToFullGmm(FullGmm *fullgmm, GmmFlagsType flags) {
  KALDI_ASSERT(weights_.Dim() == fullgmm->weights_.Dim()
               && means_.NumCols() == fullgmm->Dim());

  FullGmmNormal oldg(*fullgmm);

  if (flags & kGmmWeights)
    fullgmm->weights_.CopyFromVec(weights_);

  size_t num_comp = fullgmm->NumGauss(), dim = fullgmm->Dim();
  for (size_t i = 0; i < num_comp; i++) {
    if (flags & kGmmVariances) {
      fullgmm->inv_covars_[i].CopyFromSp(vars_[i]);
      fullgmm->inv_covars_[i].InvertDouble();

      // update the mean-related natural part with old mean, if necessary
      if (!(flags & kGmmMeans)) {
        Vector<BaseFloat> mean_times_inv(dim);
        Vector<BaseFloat> mhelp(oldg.means_.Row(i));
        mean_times_inv.AddSpVec(1.0, fullgmm->inv_covars_[i], mhelp, 0.0f);
        fullgmm->means_invcovars_.Row(i).CopyFromVec(mean_times_inv);
      }
    }

    if (flags & kGmmMeans) {
      Vector<BaseFloat> mean_times_inv(dim), mean(means_.Row(i));
      mean_times_inv.AddSpVec(1.0, fullgmm->inv_covars_[i], mean, 0.0f);
      fullgmm->means_invcovars_.Row(i).CopyFromVec(mean_times_inv);
    }
  }

  fullgmm->valid_gconsts_ = false;
}

void FullGmmNormal::Rand(MatrixBase<BaseFloat> *feats) {
  int32 dim = means_.NumCols(), num_frames = feats->NumRows(),
      num_gauss = means_.NumRows();
  KALDI_ASSERT(feats->NumCols() == dim);
  std::vector<TpMatrix<BaseFloat> > sqrt_var(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    sqrt_var[i].Resize(dim);
    sqrt_var[i].Cholesky(SpMatrix<BaseFloat>(vars_[i]));
  }
  Vector<BaseFloat> rand(dim);
  for (int32 t = 0; t < num_frames; t++) {
    int32 i = weights_.RandCategorical(); // index with prob propto weights_[i].
    SubVector<BaseFloat> frame(*feats, t);
    frame.CopyFromVec(means_.Row(i));
    rand.SetRandn();
    frame.AddTpVec(1.0, sqrt_var[i], kNoTrans, rand, 1.0);
  }
}

}  // End namespace kaldi

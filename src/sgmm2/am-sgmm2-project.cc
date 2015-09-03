// sgmm2/am-sgmm2-project.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>
using std::vector;

#include "sgmm2/am-sgmm2-project.h"
#include "thread/kaldi-thread.h"
#include "gmm/full-gmm-normal.h"
#include "gmm/diag-gmm-normal.h"

namespace kaldi {

// The output pointer argument "projection" projects from the pre-LDA+MLLT space
// to the space we're going to model.  We retain "model_dim" dimensions, which
// means we're keeping all dimensions that have any variation at all.

void Sgmm2Project::ComputeProjection(const AmSgmm2 &sgmm,
                                     const Matrix<BaseFloat> &inv_lda_mllt,
                                     int32 start_dim,
                                     int32 end_dim, // last dim plus one
                                     Matrix<BaseFloat> *projection) {
  Matrix<double> inv_lda_mllt_dbl(inv_lda_mllt);
  KALDI_ASSERT(inv_lda_mllt.NumRows() == inv_lda_mllt.NumCols());
  
  // First, to compute the projection that we're going to use:
  
  SpMatrix<double> B; // between-class covar.
  SpMatrix<double> W; // within-class covar.

  int32 model_dim = sgmm.FeatureDim(),
      full_dim = inv_lda_mllt.NumRows();
  KALDI_ASSERT(full_dim > model_dim);
  KALDI_ASSERT(start_dim >= 0 && start_dim < end_dim && end_dim <= full_dim);

  ComputeLdaStats(sgmm.full_ubm(), &B, &W);
  // B and W are now of dim "model_dim".

  double diag_term = 0.001 / model_dim * B.Trace(); // This will ensure
  // that the between-class covariance is full rank within the original
  // feature space.
  for (int32 i = 0; i < B.NumRows(); i++)
    B(i, i) += diag_term;

  B.Resize(full_dim, kCopyData); // This extends the extra dims with
  // zeros, which is what we want, because we assume the means are zero in the
  // extra dimensions [this is valid because we have cmd'ed data].

  W.Resize(full_dim, kCopyData); // We want the within-class
  // covar to be unit in the extra dimensions, so we need to do something
  // about this... note, this is valid if we have an LDA-based feature
  // space, as we constructed the LDA matrix so that the covar in
  // the rejected dimensions is unit.  [note: we can gloss over differences
  // between within vs. total covar here, as it's almost exactly the same
  // for the rejected dimensions].
  for (int32 i = model_dim; i < full_dim; i++)
    W(i, i) = 1.0;
  
  // Next, we'll project these "extended" stats with the "inv_lda_mllt"
  // matrix, which takes us into the space where we were before LDA+MLLT.
  SpMatrix<double> B_orig(full_dim), W_orig(full_dim);
  B_orig.AddMat2Sp(1.0, inv_lda_mllt_dbl, kNoTrans, B, 0.0); // B_orig <-- inv_lda_mllt B inv_lda_mllt^T
  W_orig.AddMat2Sp(1.0, inv_lda_mllt_dbl, kNoTrans, W, 0.0); // W_orig <-- inv_lda_mllt W inv_lda_mllt^T

  // Now get versions of B_orig and W_orig that are limited to the
  // dimension range that we wanted.
  Matrix<double> B_orig_mat(B_orig), W_orig_mat(W_orig); // Get them as full matrices...
  SpMatrix<double> B_orig_limit(B_orig_mat.Range(start_dim, end_dim-start_dim,
                                                 start_dim, end_dim-start_dim)),
      W_orig_limit(W_orig_mat.Range(start_dim, end_dim-start_dim,
                                    start_dim, end_dim-start_dim));
  
  Matrix<double> proj;
  int32 retained_dim = model_dim;
  if (end_dim - start_dim < retained_dim) retained_dim = end_dim - start_dim;
  ComputeLdaTransform(B_orig_limit, W_orig_limit, retained_dim, &proj);
  
  // Now proj has the projection from the "limited-dimension" space.
  // We want a projection from the entire space.
  
  projection->Resize(retained_dim, full_dim); // This projection (which we output) will project from
  // full_dim to retained_dim; it goes from the pre-LDA+MLLT space to "retained_dim" which
  // is <= model_dim.
  
  // Copy the relevant dimensions of "projection" from the "proj" matrix that
  // we just computed.  The rest remain zero (corresponding to discarded dimensions).
  projection->Range(0, retained_dim, start_dim, end_dim-start_dim).CopyFromMat(proj);
}

void Sgmm2Project::ComputeLdaTransform(const SpMatrix<double> &B,
                                       const SpMatrix<double> &W,
                                       int32 dim_to_retain, 
                                       Matrix<double> *Projection) {
  int32 dim = B.NumRows();
  KALDI_ASSERT(dim_to_retain <= dim);

  // OK, now do LDA in this space...
  TpMatrix<double> T(dim);
  T.Cholesky(W); // do Cholesky factorization W_orig = T T^T.  Now,
  // T^{-1} is the projection that makes W unit.
  TpMatrix<double> Tinv(T); // get inverse of T.
  Tinv.Invert();
  
  // Now project B_orig with Tinv, to get between-class scatter in space where
  // W_orig is unit.
  SpMatrix<double> B_proj(dim);
  B_proj.AddTp2Sp(1.0, Tinv, kNoTrans, B, 0.0);
  
  // Now, in this space, do SVD.

  Matrix<double> P(dim, dim);
  Vector<double> s(dim);
  B_proj.SymPosSemiDefEig(&s, &P);
  // Now B_proj = P diag(s) P^T, with P orthogonal.  It's both SVD and eigenvalue
  // decomposition.
  // So P^{-1}, which equals P^T, is the transformation that
  // will make B_proj diagonal (with eigenvalues equal to s).

  P.Resize(dim, dim_to_retain, kCopyData); // keep only rows of P^T that we want.
  Projection->Resize(dim_to_retain, dim);
  // The next line sets "Projection" to the LDA matrix, which is (part of P^T) * T^{-1}
  Projection->AddMatTp(1.0, P, kTrans, Tinv, kNoTrans, 0.0);

  KALDI_LOG << "Eigenvalues of retained LDA dimensions: "
            << s.Range(0, dim_to_retain) << " (sum is:) "
            << s.Range(0, dim_to_retain).Sum();
  KALDI_LOG << "Eigenvalues of rejected LDA dimensions: "
            << s.Range(dim_to_retain, dim - dim_to_retain) << " (sum is:) "
            << s.Range(dim_to_retain, dim - dim_to_retain).Sum();

  { // Check that it's been done correctly by projecting the
    // matrices we got as input checking they become (diagonal, unit).
    SpMatrix<double> B_ldaproj(dim_to_retain), W_ldaproj(dim_to_retain);
    B_ldaproj.AddMat2Sp(1.0, *Projection, kNoTrans, B, 0.0);
    KALDI_ASSERT(B_ldaproj.IsDiagonal());
    W_ldaproj.AddMat2Sp(1.0, *Projection, kNoTrans, W, 0.0);
    KALDI_ASSERT(W_ldaproj.IsUnit());
  }
}


void Sgmm2Project::ComputeLdaStats(const FullGmm &full_ubm,
                                   SpMatrix<double> *between_covar,
                                   SpMatrix<double> *within_covar) {
  int32 dim = full_ubm.Dim(); // Feature dimension.
  between_covar->Resize(dim); // zeroes it.
  within_covar->Resize(dim); // zeroes it.
  FullGmmNormal full_gmm_normal(full_ubm);
  BaseFloat weight = 1.0 / full_ubm.NumGauss();
  Vector<double> avg_mean(dim);
  for (int32 i = 0; i < full_ubm.NumGauss(); i++) {
    between_covar->AddSp(weight, full_gmm_normal.vars_[i]);
    within_covar->AddVec2(weight, full_gmm_normal.means_.Row(i));
    avg_mean.AddVec(weight, full_gmm_normal.means_.Row(i));
  }
  between_covar->AddVec2(-1.0, avg_mean);
}

void Sgmm2Project::ApplyProjection(const Matrix<BaseFloat> &total_projection,
                                   AmSgmm2 *sgmm) {
  int32 dim = sgmm->FeatureDim();
  int32 retained_dim = total_projection.NumRows();
  KALDI_ASSERT(retained_dim <= dim);
  
  // Note: small_projection is as total_projection but ignoring the
  // higher dimensions of the input... this is valid as far as the means
  // are concerned, because we extend with zeros.
  SubMatrix<BaseFloat> small_projection(total_projection, 0, retained_dim, 0, dim);
  Matrix<double> small_projection_dbl(small_projection);
  Matrix<double> total_projection_dbl(total_projection);
  
  int32 I = sgmm->NumGauss();
  for (int32 i = 0; i < I; i++) {
    {
      // do M_i  <-- small_projection * M_i
      Matrix<BaseFloat> M(sgmm->M_[i]);
      sgmm->M_[i].Resize(retained_dim, M.NumCols());
      sgmm->M_[i].AddMatMat(1.0, small_projection, kNoTrans, M, kNoTrans, 0.0);
    }
    if (!sgmm->N_.empty()) {
      // do N_i  <-- small_projection * N_i
      Matrix<BaseFloat> N(sgmm->N_[i]);
      sgmm->N_[i].Resize(retained_dim, N.NumCols());
      sgmm->N_[i].AddMatMat(1.0, small_projection, kNoTrans, N, kNoTrans, 0.0);
    }
    ProjectVariance(total_projection_dbl, true, // inverted,
                    &(sgmm->SigmaInv_[i]));
  }    

  { // Project full_ubm.
    FullGmmNormal full_ubm_normal(sgmm->full_ubm_);
    for (int32 i = 0; i < I; i++) {
      ProjectVariance(total_projection_dbl, false, &(full_ubm_normal.vars_[i]));
    }
    Matrix<double> old_means(full_ubm_normal.means_);
    full_ubm_normal.means_.Resize(I, retained_dim);
    full_ubm_normal.means_.AddMatMat(1.0, old_means, kNoTrans,
                                     small_projection_dbl, kTrans, 0.0);
    sgmm->full_ubm_.Resize(I, retained_dim);
    full_ubm_normal.CopyToFullGmm(&sgmm->full_ubm_);
    sgmm->full_ubm_.ComputeGconsts();
  }
  sgmm->diag_ubm_.Resize(I, retained_dim);
  sgmm->diag_ubm_.CopyFromFullGmm(sgmm->full_ubm_);
  sgmm->diag_ubm_.ComputeGconsts();
  sgmm->n_.clear(); // The normalizers are invalid now, so clear them.
}

void Sgmm2Project::ProjectVariance(const Matrix<double> &total_projection,
                                   bool inverse,
                                   SpMatrix<double> *variance) {
  if (inverse) {
    SpMatrix<double> inv_var(*variance);
    inv_var.Invert();
    ProjectVariance(total_projection, false, &inv_var);
    inv_var.Invert();
    if (variance->NumRows() != inv_var.NumRows())
      variance->Resize(inv_var.NumRows());
    variance->CopyFromSp(inv_var);
  } else {
    SpMatrix<double> extended_var(*variance);
    KALDI_ASSERT(total_projection.NumCols() >= extended_var.NumRows());
    extended_var.Resize(total_projection.NumCols(), kCopyData);
    for (int32 i = variance->NumRows(); i < extended_var.NumRows(); i++)
      extended_var(i, i) = 1.0; // make new part of diagonal ones.
    int32 tgt_dim = total_projection.NumRows();
    KALDI_ASSERT(tgt_dim <= variance->NumRows());
    if (tgt_dim < variance->NumRows()) variance->Resize(tgt_dim);
    variance->AddMat2Sp(1.0, total_projection, kNoTrans, extended_var, 0.0);
  }
}

void Sgmm2Project::ProjectVariance (const Matrix<double> &total_projection,
                                    bool inverse,
                                    SpMatrix<float> *variance) {
  SpMatrix<double> variance_dbl(*variance);
  ProjectVariance(total_projection, inverse, &variance_dbl);
  if (variance->NumRows() != variance_dbl.NumRows())
    variance->Resize(variance_dbl.NumRows());
  variance->CopyFromSp(variance_dbl);
}


}  // namespace kaldi

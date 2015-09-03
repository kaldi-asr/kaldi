// transform/fmllr-diag-gmm.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Georg Stemmer
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

#include <utility>
#include <vector>
using std::vector;

#include "transform/fmllr-diag-gmm.h"

namespace kaldi {

void FmllrDiagGmmAccs:: AccumulateFromPosteriors(
    const DiagGmm &pdf,
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &posterior) {
  
  if (this->DataHasChanged(data)) {
    CommitSingleFrameStats();
    InitSingleFrameStats(data);
  }
  SingleFrameStats &stats = this->single_frame_stats_;
  stats.count += posterior.Sum();
  stats.a.AddMatVec(1.0, pdf.means_invvars(), kTrans, posterior, 1.0);
  stats.b.AddMatVec(1.0, pdf.inv_vars(), kTrans, posterior, 1.0);
}

void FmllrDiagGmmAccs:: AccumulateFromPosteriorsPreselect(
    const DiagGmm &pdf,
    const std::vector<int32> &gselect,
    const VectorBase<BaseFloat> &data,
    const VectorBase<BaseFloat> &posterior) {
  
  if (this->DataHasChanged(data)) {
    CommitSingleFrameStats();
    InitSingleFrameStats(data);
  }
  SingleFrameStats &stats = this->single_frame_stats_;
  stats.count += posterior.Sum();
  
  const Matrix<BaseFloat> &means_invvars = pdf.means_invvars(),
      &inv_vars = pdf.inv_vars();
  KALDI_ASSERT(static_cast<int32>(gselect.size()) == posterior.Dim());
  for (size_t i = 0; i < gselect.size(); i++) {
    stats.a.AddVec(posterior(i), means_invvars.Row(gselect[i]));
    stats.b.AddVec(posterior(i), inv_vars.Row(gselect[i]));
  }
}

FmllrDiagGmmAccs::FmllrDiagGmmAccs(const DiagGmm &gmm,
                                   const AccumFullGmm &fgmm_accs):
    single_frame_stats_(gmm.Dim()), opts_(FmllrOptions()) {
  KALDI_ASSERT(gmm.NumGauss() == fgmm_accs.NumGauss()
               && gmm.Dim() == fgmm_accs.Dim());
  Init(gmm.Dim());
  int32 dim = gmm.Dim(), num_gauss = gmm.NumGauss();
  for (int32 g = 0; g < num_gauss; g++) {
    double this_occ = fgmm_accs.occupancy()(g);
    if (this_occ == 0) continue;
    SubVector<BaseFloat> this_mean_invvar(gmm.means_invvars(), g);
    SubVector<BaseFloat> this_invvar(gmm.inv_vars(), g);
    SubVector<double> this_mean_acc(fgmm_accs.mean_accumulator(), g);
    Vector<double> this_mean_invvar_dbl(this_mean_invvar);
    Vector<double> this_extended_mean_acc(dim+1);
    this_extended_mean_acc.Range(0, dim).CopyFromVec(this_mean_acc);
    this_extended_mean_acc(dim) = this_occ; // acc of x^+
    Matrix<double> this_cov_acc(fgmm_accs.covariance_accumulator()[g]); // copy to
    // regular Matrix.
    Matrix<double> this_extended_cov_acc(dim+1, dim+1); // make as if accumulated
    // using x^+, not x.
    this_extended_cov_acc.Range(0, dim, 0, dim).CopyFromMat(this_cov_acc);
    this_extended_cov_acc.Row(dim).CopyFromVec(this_extended_mean_acc);
    this_extended_cov_acc.CopyColFromVec(this_extended_mean_acc, dim); // since
    // there is no Col() function, use a member-function of the matrix class.
    SpMatrix<double> this_extended_cov_acc_sp(this_extended_cov_acc);
    beta_ += this_occ;
    K_.AddVecVec(1.0, this_mean_invvar_dbl, this_extended_mean_acc);
    for (int32 d = 0; d < dim; d++)
      G_[d].AddSp(this_invvar(d), this_extended_cov_acc_sp);
  }
}


BaseFloat FmllrDiagGmmAccs::AccumulateForGmm(const DiagGmm &pdf,
                                             const VectorBase<BaseFloat> &data,
                                             BaseFloat weight) {
  int32 num_comp = pdf.NumGauss();
  Vector<BaseFloat> posterior(num_comp);
  BaseFloat loglike;
  
  loglike = pdf.ComponentPosteriors(data, &posterior);
  posterior.Scale(weight);
  AccumulateFromPosteriors(pdf, data, posterior);
  return loglike;
}

BaseFloat FmllrDiagGmmAccs::AccumulateForGmmPreselect(
    const DiagGmm &pdf,
    const std::vector<int32> &gselect,
    const VectorBase<BaseFloat> &data,
    BaseFloat weight) {
  KALDI_ASSERT(!gselect.empty() && "Empty gselect information");
  Vector<BaseFloat> loglikes;
  pdf.LogLikelihoodsPreselect(data, gselect, &loglikes);

  BaseFloat loglike = loglikes.ApplySoftMax(); // they are now posteriors.
  loglikes.Scale(weight);
  AccumulateFromPosteriorsPreselect(pdf, gselect, data, loglikes);
  return loglike;
}



void FmllrDiagGmmAccs::Update(const FmllrOptions &opts,
                              MatrixBase<BaseFloat> *fmllr_mat,
                              BaseFloat *objf_impr,
                              BaseFloat *count) {
  KALDI_ASSERT(fmllr_mat != NULL);
  CommitSingleFrameStats();
  if (fmllr_mat->IsZero())
    KALDI_ERR << "You must initialize the fMLLR matrix to a non-singular value "
        "(so we can report objective function changes); e.g. call SetUnit()";
  if (opts.update_type == "full" && this->opts_.update_type != "full") {
    KALDI_ERR << "You are requesting a full-fMLLR update but you accumulated "
              << "stats for more limited update type.";
  }
  if (beta_ > opts.min_count) {
    Matrix<BaseFloat> tmp_old(*fmllr_mat), tmp_new(*fmllr_mat);
    BaseFloat objf_change;
    if (opts.update_type == "full")
      objf_change = ComputeFmllrMatrixDiagGmmFull(tmp_old, *this, opts.num_iters, &tmp_new);
    else if (opts.update_type == "diag")
      objf_change = ComputeFmllrMatrixDiagGmmDiagonal(tmp_old, *this, &tmp_new);
    else if (opts.update_type == "offset")
      objf_change = ComputeFmllrMatrixDiagGmmOffset(tmp_old, *this, &tmp_new);
    else if (opts.update_type == "none")
      objf_change = 0.0;
    else
      KALDI_ERR << "Unknown fMLLR update type " << opts.update_type
                << ", fmllr-update-type must be one of \"full\"|\"diag\"|\"offset\"|\"none\"";
    fmllr_mat->CopyFromMat(tmp_new);
    if (objf_impr) *objf_impr = objf_change;
    if (count) *count = beta_;
  } else {  // Not changing matrix.
    KALDI_WARN << "Not updating fMLLR since below min-count: count is " << beta_;
    if (objf_impr) *objf_impr = 0.0;
    if (count) *count = beta_;
  }
}


BaseFloat ComputeFmllrMatrixDiagGmm(const MatrixBase<BaseFloat> &in_xform,
                                    const AffineXformStats &stats,
                                    std::string fmllr_type,  // "none", "offset", "diag", "full"
                                    int32 num_iters,
                                    MatrixBase<BaseFloat> *out_xform) {
  if (fmllr_type == "full") {
    return ComputeFmllrMatrixDiagGmmFull(in_xform, stats, num_iters, out_xform);
  } else if (fmllr_type == "diag") {
    return ComputeFmllrMatrixDiagGmmDiagonal(in_xform, stats, out_xform);
  } else if (fmllr_type == "offset") {
    return ComputeFmllrMatrixDiagGmmOffset(in_xform, stats, out_xform);
  } else if (fmllr_type == "none") {
    if (!in_xform.IsUnit())
      KALDI_WARN << "You set fMLLR type to \"none\" but your starting transform "
          "is not unit [this is strange, and diagnostics will be wrong].";
    out_xform->SetUnit();
    return 0.0;
  } else
    KALDI_ERR << "Unknown fMLLR update type " << fmllr_type
              << ", must be one of \"full\"|\"diag\"|\"offset\"|\"none\"";
  return 0.0;
}


void FmllrInnerUpdate(SpMatrix<double> &inv_G,
                      VectorBase<double> &k,
                      double beta,
                      int32 row,
                      MatrixBase<double> *transform) {
  int32 dim = transform->NumRows();
  KALDI_ASSERT(transform->NumCols() == dim + 1);
  KALDI_ASSERT(row >= 0 && row < dim);

  double logdet;
  // Calculating the matrix of cofactors (transpose of adjugate)
  Matrix<double> cofact_mat(dim, dim);
  cofact_mat.CopyFromMat(transform->Range(0, dim, 0, dim), kTrans);
  cofact_mat.Invert(&logdet);
  // Removed this step because it's not necessary and could lead to
  // under/overflow [Dan]
  // cofact_mat.Scale(exp(logdet));
  
  // The extended cofactor vector for the current row
  Vector<double> cofact_row(dim + 1);
  cofact_row.Range(0, dim).CopyRowFromMat(cofact_mat, row);
  cofact_row(dim) = 0;
  Vector<double> cofact_row_invg(dim + 1);
  cofact_row_invg.AddSpVec(1.0, inv_G, cofact_row, 0.0);

  // Solve the quadratic equation for step size
  double e1 = VecVec(cofact_row_invg, cofact_row);
  double e2 = VecVec(cofact_row_invg, k);
  double discr = std::sqrt(e2 * e2 + 4 * e1 * beta);
  double alpha1 = (-e2 + discr) / (2 * e1);
  double alpha2 = (-e2 - discr) / (2 * e1);
  double auxf1 = beta * Log(std::abs(alpha1 * e1 + e2)) -
      0.5 * alpha1 * alpha1 * e1;
  double auxf2 = beta * Log(std::abs(alpha2 * e1 + e2)) -
      0.5 * alpha2 * alpha2 * e1;
  double alpha = (auxf1 > auxf2) ? alpha1 : alpha2;

  // Update transform row: w_d = (\alpha cofact_d + k_d) G_d^{-1}
  cofact_row.Scale(alpha);
  cofact_row.AddVec(1.0, k);
  transform->Row(row).AddSpVec(1.0, inv_G, cofact_row, 0.0);
}

BaseFloat ComputeFmllrMatrixDiagGmmFull(const MatrixBase<BaseFloat> &in_xform,
                                        const AffineXformStats &stats,
                                        int32 num_iters,
                                        MatrixBase<BaseFloat> *out_xform) {
  int32 dim = static_cast<int32>(stats.G_.size());

  // Compute the inverse matrices of second-order statistics
  vector< SpMatrix<double> > inv_g(dim);
  for (int32 d = 0; d < dim; d++) {
    inv_g[d].Resize(dim + 1);
    inv_g[d].CopyFromSp(stats.G_[d]);
    inv_g[d].Invert();
  }

  Matrix<double> old_xform(in_xform), new_xform(in_xform);
  BaseFloat old_objf = FmllrAuxFuncDiagGmm(old_xform, stats);
  
  for (int32 iter = 0; iter < num_iters; ++iter) {
    for (int32 d = 0; d < dim; d++) {
      SubVector<double> k_d(stats.K_, d);
      FmllrInnerUpdate(inv_g[d], k_d, stats.beta_, d, &new_xform);
    }  // end of looping over rows
  }  // end of iterations

  BaseFloat new_objf = FmllrAuxFuncDiagGmm(new_xform, stats),
      objf_improvement = new_objf - old_objf;
  KALDI_LOG << "fMLLR objf improvement is "
            << (objf_improvement / (stats.beta_ + 1.0e-10))
            << " per frame over " << stats.beta_ << " frames.";
  if (objf_improvement < 0.0 && !ApproxEqual(new_objf, old_objf)) {
    KALDI_WARN << "No applying fMLLR transform change because objective "
               << "function did not increase.";
    return 0.0;
  } else {
    out_xform->CopyFromMat(new_xform, kNoTrans);
    return objf_improvement;
  }
}

BaseFloat ComputeFmllrMatrixDiagGmmDiagonal(const MatrixBase<BaseFloat> &in_xform,
                                            const AffineXformStats &stats,
                                            MatrixBase<BaseFloat> *out_xform) {
  // The "Diagonal" here means a diagonal fMLLR matrix, i.e. like W = [ A;  b] where
  // A is diagonal.
  // We re-derived the math (see exponential transform paper) to get a simpler
  // update rule.

  /*
  Write out_xform as D, which is a d x d+1 matrix (where d is the feature dimension).
  We are solving for s == d_{i,i}, and o == d_{i,d}  [assuming zero-based indexing];
      s is a scale, o is an offset.
  The stats are K (dimension d x d+1) and G_i for i=0..d-1 (dimension: d+1 x d+1),
    and the count beta.

 The auxf for the i'th row of the transform is (assuming zero-based indexing):

  s k_{i,i}  +  o k_{i,d}
  - \frac{1}{2} s^2 g_{i,i,i} - \frac{1}{2} o^2 g_{i,d,d} - s o g_{i,d,i}
   + \beta \log |s|

   Suppose we know s, we can solve for o:
      o = (k_{i,d} - s g_{i,d,i}) / g_{i,d,d}
   Substituting this expression for o into the auxf (and ignoring
   terms that don't vary with s), we have the auxf:

 \frac{1}{2} s^2 ( g_{i,d,i}^2 / g_{i,d,d}  -  g_{i,i,i} )
    +  s ( k_{i,i} - g_{i,d,i} k_{i,d} / g_{i,d,d} )
    + \beta \log |s|

  Differentiating w.r.t. s and assuming s is positive, we have
    a s + b + c/s = 0
 where
   a = (  g_{i,d,i}^2 / g_{i,d,d}  -  g_{i,i,i} ),
   b = ( k_{i,i} - g_{i,d,i} k_{i,d} / g_{i,d,d} )
   c = beta
 Multiplying by s, we have the equation
   a s^2 + b s + c = 0, where we assume s > 0.
 We solve it with:
  s = (-b - \sqrt{b^2 - 4ac}) / 2a
 [take the negative root because we know a is negative, and this gives
  the more positive solution for s; the other one would be negative].
 We then solve for o with the equation above, i.e.:
     o = (k_{i,d} - s g_{i,d,i}) / g_{i,d,d})
  */

  int32 dim = stats.G_.size();
  double beta = stats.beta_;
  out_xform->CopyFromMat(in_xform);
  if (beta == 0.0) {
    KALDI_WARN << "Computing diagonal fMLLR matrix: no stats [using original transform]";
    return 0.0;
  }
  BaseFloat old_obj = FmllrAuxFuncDiagGmm(*out_xform, stats);
  KALDI_ASSERT(out_xform->Range(0, dim, 0, dim).IsDiagonal()); // orig transform
  // must be diagonal.
  for(int32 i = 0; i < dim; i++) {
    double k_ii = stats.K_(i, i), k_id = stats.K_(i, dim),
        g_iii = stats.G_[i](i, i), g_idd = stats.G_[i](dim, dim),
        g_idi = stats.G_[i](dim, i);
    double a = g_idi*g_idi/g_idd - g_iii,
        b = k_ii - g_idi*k_id/g_idd,
        c = beta;
    double s = (-b - std::sqrt(b*b - 4*a*c)) / (2*a);
    KALDI_ASSERT(s > 0.0);
    double o = (k_id - s*g_idi) / g_idd;
    (*out_xform)(i, i) = s;
    (*out_xform)(i, dim) = o;
  }
  BaseFloat new_obj = FmllrAuxFuncDiagGmm(*out_xform, stats);
  KALDI_VLOG(2) << "fMLLR objective function improvement = "
                << (new_obj - old_obj);
  return new_obj - old_obj;
}

BaseFloat ComputeFmllrMatrixDiagGmmOffset(const MatrixBase<BaseFloat> &in_xform,
                                          const AffineXformStats &stats,
                                          MatrixBase<BaseFloat> *out_xform) {
  int32 dim = stats.G_.size();
  KALDI_ASSERT(in_xform.NumRows() == dim && in_xform.NumCols() == dim+1);
  SubMatrix<BaseFloat> square_part(in_xform, 0, dim, 0, dim);
  KALDI_ASSERT(square_part.IsUnit());
  BaseFloat objf_impr = 0.0;
  out_xform->CopyFromMat(in_xform);
  for (int32 i = 0; i < dim; i++) {
    // auxf in this offset b_i is:
    //  -0.5 b_i^2 G_i(dim, dim) - b_i G_i(i, dim)*1.0 + b_i K(i, dim)  (1)
    // answer is:
    // b_i = [K(i, dim) - G_i(i, dim)] / G_i(dim, dim)
    // objf change is given by (1)
    BaseFloat b_i = (*out_xform)(i, dim);
    BaseFloat objf_before = -0.5 * b_i * b_i * stats.G_[i](dim, dim)
        - b_i * stats.G_[i](i, dim) + b_i * stats.K_(i, dim);
    b_i = (stats.K_(i, dim) - stats.G_[i](i, dim)) / stats.G_[i](dim, dim);
    (*out_xform)(i, dim) = b_i;
    BaseFloat objf_after = -0.5 * b_i * b_i * stats.G_[i](dim, dim)
        - b_i * stats.G_[i](i, dim) + b_i * stats.K_(i, dim);
    if (objf_after < objf_before)
      KALDI_WARN << "Objf decrease in offset estimation:"
                 << objf_after << " < " << objf_before;
    objf_impr += objf_after - objf_before;
  }
  return objf_impr;
}


void ApplyFeatureTransformToStats(const MatrixBase<BaseFloat> &xform,
                                  AffineXformStats *stats) {
  KALDI_ASSERT(stats != NULL && stats->Dim() != 0);
  int32 dim = stats->Dim();
  // make sure the stats are of the standard diagonal kind.
  KALDI_ASSERT(stats->G_.size() == static_cast<size_t>(dim));
  KALDI_ASSERT( (xform.NumRows() == dim && xform.NumCols() == dim) // linear
                || (xform.NumRows() == dim && xform.NumCols() == dim+1) // affine
                || (xform.NumRows() == dim+1 && xform.NumCols() == dim+1));  // affine w/ extra row.
  if (xform.NumRows() == dim+1) {  // check last row of input
    // has correct value. 0 0 0 ..  0 1.
    for (int32 i = 0; i < dim; i++)
      KALDI_ASSERT(xform(dim, i) == 0.0);
    KALDI_ASSERT(xform(dim, dim) == 1.0);
  }

  // Get the transform into the (dim+1 x dim+1) format, with
  // 0 0 0 .. 0 1 as the last row.
  SubMatrix<BaseFloat> xform_square(xform, 0, dim, 0, dim);
  Matrix<double> xform_full(dim+1, dim+1);
  SubMatrix<double> xform_full_square(xform_full, 0, dim, 0, dim);
  xform_full_square.CopyFromMat(xform_square);
  if (xform.NumCols() == dim+1)  // copy offset.
    for (int32 i = 0; i < dim; i++)
      xform_full(i, dim) = xform(i, dim);

  xform_full(dim, dim) = 1.0;

  SpMatrix<double> Gtmp(dim+1);
  for (int32 i = 0; i < dim; i++) {
    // Gtmp <-- xform_full * stats->G_[i] * xform_full^T
    Gtmp.AddMat2Sp(1.0, xform_full, kNoTrans, stats->G_[i], 0.0);
    stats->G_[i].CopyFromSp(Gtmp);
  }
  Matrix<double> Ktmp(dim, dim+1);
  // Ktmp <-- stats->K_ * xform_full^T
  Ktmp.AddMatMat(1.0, stats->K_, kNoTrans, xform_full, kTrans, 0.0);
  stats->K_.CopyFromMat(Ktmp);
}

void ApplyModelTransformToStats(const MatrixBase<BaseFloat> &xform,
                                AffineXformStats *stats) {
  KALDI_ASSERT(stats != NULL && stats->Dim() != 0.0);
  int32 dim = stats->Dim();
  KALDI_ASSERT(xform.NumRows() == dim && xform.NumCols() == dim+1);
  {
    SubMatrix<BaseFloat> xform_square(xform, 0, dim, 0, dim);
    // Only works with diagonal transforms.
    KALDI_ASSERT(xform_square.IsDiagonal());
  }

  // Working out rules for transforming fMLLR statistics under diagonal
  // model-space transformations.
  //
  // We work out what the stats would be if we had accumulated
  // with offset/scaled means and vars. Let T be the transform
  // T = [ D; b ],
  // where D is diagonal, d_i is the i'th diagonal of D, and b_i
  // is the i'th offset element.  This is equivalent to the transform
  //   x_i -> y_i = d_i x_i + b_i,
  // so d_i is the diagonal and b_i is the offset term.  We work out the
  // reverse feature transform (from general to speaker-specific space),
  // which is
  //  y_i -> x_i = (y_i - b_i) / d_i
  // the corresponding mean transform to speaker-space is the same:
  //  mu_i -> (mu_i - b_i) / d_i
  // and the transfrom on the variances is:
  //  sigma_i^2 -> sigma_i^2 / d_i^2,
  // so on inverse variance this becomes:
  //  (1/sigma_i^2) -> (1/sigma_i^2) * d_i^2.
  //
  // Now, we work out the change in K and G_i from these effects on the
  // means and variances.
  //
  // Now, k_{ij} is \sum_{m, t} \gamma_m (1/\sigma^2_{m, i}) \mu_{m, i} x^+_j .
  //
  // If we are transforming to K', we want:
  //
  // k'_{ij} = \sum_{m, t} \gamma_m (d_i^2/\sigma^2_{m, i}) ((\mu_{m, i} - b_i)/d_i)  x^+_j .
  //         = d_i k_{i, j} - \sum_{m, t} \gamma_m (1/\sigma^2_{m, i}) d_i b_i x^+_j .
  //         = d_i k_{i, j} - d_i b_i g_{i, d, j},
  // where g_{i, d, j} is the {d, j}'th element of G_i. (in zero-based indexing).
  //
  //
  // G_i only depends on the variances and features, so the only change
  // in G_i is G_i -> d_i^2 G_i (this comes from the change in 1/sigma_i^2).
  // This is done after the change in K.

  for (int32 i = 0; i < dim; i++) {
    BaseFloat d_i = xform(i, i), b_i = xform(i, dim);
    for (int32 j = 0; j <= dim; j++) {
      stats->K_(i, j) = d_i * stats->K_(i, j) - d_i * b_i * stats->G_[i](dim, j);
    }
  }
  for (int32 i = 0; i < dim; i++) {
    BaseFloat d_i = xform(i, i);
    stats->G_[i].Scale(d_i * d_i);
  }
}

float FmllrAuxFuncDiagGmm(const MatrixBase<float> &xform,
                              const AffineXformStats &stats) {
  int32 dim = static_cast<int32>(stats.G_.size());
  Matrix<double> xform_d(xform);
  Vector<double> xform_row_g(dim + 1);
  SubMatrix<double> A(xform_d, 0, dim, 0, dim);
  double obj = stats.beta_ * A.LogDet() +
      TraceMatMat(xform_d, stats.K_, kTrans);
  for (int32 d = 0; d < dim; d++) {
    xform_row_g.AddSpVec(1.0, stats.G_[d], xform_d.Row(d), 0.0);
    obj -= 0.5 * VecVec(xform_row_g, xform_d.Row(d));
  }
  return obj;
}

double FmllrAuxFuncDiagGmm(const MatrixBase<double> &xform,
                           const AffineXformStats &stats) {
  int32 dim = static_cast<int32>(stats.G_.size());
  Vector<double> xform_row_g(dim + 1);
  SubMatrix<double> A(xform, 0, dim, 0, dim);
  double obj = stats.beta_ * A.LogDet() +
      TraceMatMat(xform, stats.K_, kTrans);
  for (int32 d = 0; d < dim; d++) {
    xform_row_g.AddSpVec(1.0, stats.G_[d], xform.Row(d), 0.0);
    obj -= 0.5 * VecVec(xform_row_g, xform.Row(d));
  }
  return obj;
}

BaseFloat FmllrAuxfGradient(const MatrixBase<BaseFloat> &xform,
                            // if this is changed back to Matrix<double>
                           // un-comment the Resize() below.
                            const AffineXformStats &stats,
                            MatrixBase<BaseFloat> *grad_out) {
  int32 dim = static_cast<int32>(stats.G_.size());
  Matrix<double> xform_d(xform);
  Vector<double> xform_row_g(dim + 1);
  SubMatrix<double> A(xform_d, 0, dim, 0, dim);
  double obj = stats.beta_ * A.LogDet() +
      TraceMatMat(xform_d, stats.K_, kTrans);
  Matrix<double> S(dim, dim + 1);
  for (int32 d = 0; d < dim; d++) {
    xform_row_g.AddSpVec(1.0, stats.G_[d], xform_d.Row(d), 0.0);
    obj -= 0.5 * VecVec(xform_row_g, xform_d.Row(d));
    S.CopyRowFromVec(xform_row_g, d);
  }

  // Compute the gradient: P = \beta [(A^{-1})^{T} , 0] + K - S
  // grad_out->Resize(dim, dim + 1);
  Matrix<double> tmp_grad(dim, dim + 1);
  tmp_grad.Range(0, dim, 0, dim).CopyFromMat(A);
  tmp_grad.Range(0, dim, 0, dim).Invert();
  tmp_grad.Range(0, dim, 0, dim).Transpose();
  tmp_grad.Scale(stats.beta_);
  tmp_grad.AddMat(-1.0, S, kNoTrans);
  tmp_grad.AddMat(1.0, stats.K_, kNoTrans);
  grad_out->CopyFromMat(tmp_grad, kNoTrans);

  return obj;
}

bool FmllrDiagGmmAccs::DataHasChanged(const VectorBase<BaseFloat> &data) const {
  KALDI_ASSERT(data.Dim() == this->Dim());
  return !data.ApproxEqual(single_frame_stats_.x, 0.0);
}

void FmllrDiagGmmAccs::SingleFrameStats::Init(int32 dim) {
  x.Resize(dim);
  a.Resize(dim);
  b.Resize(dim);
  count = 0.0;
}  

void FmllrDiagGmmAccs::InitSingleFrameStats(const VectorBase<BaseFloat> &data) {
  SingleFrameStats &stats = single_frame_stats_;
  stats.x.CopyFromVec(data);
  stats.count = 0.0;
  stats.a.SetZero();
  stats.b.SetZero();
}

void FmllrDiagGmmAccs::CommitSingleFrameStats() {
  // Commit the stats for this from (in SingleFrameStats).
  int32 dim = Dim();
  SingleFrameStats &stats = single_frame_stats_;
  if (stats.count == 0.0) return;

  Vector<double> xplus(dim+1);
  xplus.Range(0, dim).CopyFromVec(stats.x);
  xplus(dim) = 1.0;
  
  this->beta_ += stats.count;
  this->K_.AddVecVec(1.0, Vector<double>(stats.a), xplus);


  if (opts_.update_type == "full") {
    SpMatrix<double> scatter(dim+1);
    scatter.AddVec2(1.0, xplus);
    
    KALDI_ASSERT(static_cast<size_t>(dim) == this->G_.size());
    for (int32 i = 0; i < dim; i++)
      this->G_[i].AddSp(stats.b(i), scatter);
  } else {
    // We only need some elements of these stats, so just update those elements.
    for (int32 i = 0; i < dim; i++) {
      BaseFloat scale = stats.b(i), x_i = xplus(i);
      this->G_[i](i, i) += scale * x_i * x_i;
      this->G_[i](dim, i) += scale * 1.0 * x_i;
      this->G_[i](dim, dim) += scale * 1.0 * 1.0;
    }
  }

  stats.count = 0.0;
  stats.a.SetZero();
  stats.b.SetZero();
}
    



} // namespace kaldi

// adapt/differentiable-fmllr.cc

// Copyright     2018  Johns Hopkins University (author: Daniel Povey)

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

#include "adapt/differentiable-fmllr.h"
#include "matrix/matrix-functions.h"

namespace kaldi {
namespace differentiable_transform {


void FmllrEstimatorOptions::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<FmllrEstimatorOptions>");
  WriteToken(os, binary, "<SVFloor>");
  WriteBasicType(os, binary, singular_value_relative_floor);
  WriteToken(os, binary, "<VarFloor>");
  WriteBasicType(os, binary, variance_floor);
  WriteToken(os, binary, "<VarShareWeight>");
  WriteBasicType(os, binary, variance_sharing_weight);
  WriteToken(os, binary, "<SmoothingCount>");
  WriteBasicType(os, binary, smoothing_count);
  WriteToken(os, binary, "<SmoothingFactor>");
  WriteBasicType(os, binary, smoothing_between_class_factor);
  WriteToken(os, binary, "</FmllrEstimatorOptions>");
}

void FmllrEstimatorOptions::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<FmllrEstimatorOptions>");
  ExpectToken(is, binary, "<SVFloor>");
  ReadBasicType(is, binary, &singular_value_relative_floor);
  ExpectToken(is, binary, "<VarFloor>");
  ReadBasicType(is, binary, &variance_floor);
  ExpectToken(is, binary, "<VarShareWeight>");
  ReadBasicType(is, binary, &variance_sharing_weight);
  ExpectToken(is, binary, "<SmoothingCount>");
  ReadBasicType(is, binary, &smoothing_count);
  ExpectToken(is, binary, "<SmoothingFactor>");
  ReadBasicType(is, binary, &smoothing_between_class_factor);
  ExpectToken(is, binary, "</FmllrEstimatorOptions>");
}

void FmllrEstimatorOptions::ReadFromConfig(ConfigLine *config_line) {
  config_line->GetValue("singular-value-relative-floor",
                        &singular_value_relative_floor);
  config_line->GetValue("variance-floor", &variance_floor);
  config_line->GetValue("variance-sharing-weight", &variance_sharing_weight);
  config_line->GetValue("smoothing-count", &smoothing_count);
  config_line->GetValue("smoothing-between-class-factor",
                        &smoothing_between_class_factor);
}


CoreFmllrEstimator::CoreFmllrEstimator(
    const FmllrEstimatorOptions &opts,
    BaseFloat gamma,
    const MatrixBase<BaseFloat> &G,
    const MatrixBase<BaseFloat> &K,
    MatrixBase<BaseFloat> *A):
    opts_(opts),  gamma_(gamma),
    G_(G), K_(K), A_(A) {
  KALDI_ASSERT(opts.singular_value_relative_floor > 0.0 &&
               gamma > 0.0 && G.NumRows() == K.NumRows() &&
               K.NumRows() == K.NumCols() &&
               SameDim(K, *A));
}


BaseFloat CoreFmllrEstimator::Forward() {
  ComputeH();
  ComputeL();
  ComputeB();
  ComputeA();
  return ComputeObjfChange();
}

void CoreFmllrEstimator::ComputeH() {
  int32 dim = G_.NumRows();
  bool symmetric = true;
  G_rescaler_.Init(&G_, symmetric);
  BaseFloat *G_singular_values = G_rescaler_.InputSingularValues();

  {
    SubVector<BaseFloat> v(G_singular_values, dim);
    BaseFloat floor = v.Max() * opts_.singular_value_relative_floor;
    KALDI_ASSERT(floor > 0.0);
    MatrixIndexT num_floored = 0;
    v.ApplyFloor(floor, &num_floored);
    if (num_floored > 0.0)
      KALDI_WARN << num_floored << " out of " << dim
                 << " singular values floored in G matrix.";
  }
  BaseFloat *H_singular_values = G_rescaler_.OutputSingularValues(),
      *H_singular_value_derivs = G_rescaler_.OutputSingularValueDerivs();
  // We don't have to worry about elements of G_singular_values being zero,
  // since we floored them above.
  for (int32 i = 0; i < dim; i++) {
    H_singular_values[i] = 1.0 / std::sqrt(G_singular_values[i]);
    // The following expression is equivalent to
    // -0.5 * pow(G_singular_values[i], -1.5),
    // which is the derivative of lambda^{-0.5} w.r.t lambda.
    // (lambda, here, is G_singular_values[i]).
    H_singular_value_derivs[i] = -0.5 * (H_singular_values[i] /
                                         G_singular_values[i]);
  }
  H_.Resize(dim, dim, kUndefined);
  G_rescaler_.GetOutput(&H_);
}

void CoreFmllrEstimator::ComputeL() {
  int32 dim = G_.NumRows();
  L_.Resize(dim, dim);
  L_.AddMatMat(1.0, K_, kNoTrans, H_, kNoTrans, 0.0);
}

// Compute B = F(L), where F is the
// function that takes the singular values of L, puts them through the function
// f(lamba) = (lambda + sqrt(lambda^2 + 4 gamma)) / 2.
void CoreFmllrEstimator::ComputeB() {
  int32 dim = L_.NumRows();
  bool symmetric = false;
  L_rescaler_.Init(&L_, symmetric);
  BaseFloat *lambda = L_rescaler_.InputSingularValues();
  {  // This block deals with flooring lambda to avoid zero values.
    SubVector<BaseFloat> v(lambda, dim);
    BaseFloat floor = v.Max() * opts_.singular_value_relative_floor;
    KALDI_ASSERT(floor > 0.0);
    MatrixIndexT num_floored = 0;
    v.ApplyFloor(floor, &num_floored);
    static int num_warned = 100;
    if (num_floored > 0.0 && num_warned > 0)
      KALDI_WARN << num_floored << " out of " << dim
                 << " singular values floored in L matrix."
                 << (--num_warned == 0 ? "  Will not warn again." : "");
  }
  // f is where we put f(lambda).
  // f_prime is where we put f'(lambda) (the function-derivative of f w.r.t
  // lambda).
  BaseFloat *f = L_rescaler_.OutputSingularValues(),
      *f_prime = L_rescaler_.OutputSingularValueDerivs();

  BaseFloat gamma = gamma_;
  for (int32 i = 0; i < dim; i++) {
    BaseFloat lambda_i = lambda[i];
    f[i] = (lambda_i + std::sqrt(lambda_i * lambda_i + 4.0 * gamma)) / 2.0;
    f_prime[i] = (1.0 + lambda_i /
                  std::sqrt(lambda_i * lambda_i + 4.0 * gamma)) / 2.0;
  }
  B_.Resize(dim, dim, kUndefined);
  L_rescaler_.GetOutput(&B_);
}

void CoreFmllrEstimator::ComputeA() {
  A_->SetZero();  // Make sure there are no NaN's.
  A_->AddMatMat(1.0, B_, kNoTrans, H_, kNoTrans, 0.0);
}

BaseFloat CoreFmllrEstimator::ComputeObjfChange() {
  // we are computing the objective-function improvement from estimating
  // A (we'll later compute the improvement from estimating the offset b).
  // This is the equation which, from the writeup, is:
  // \gamma log |A| + tr(A^T K) - tr(K)
  //    + 1/2 tr(G) - 1/2 tr(B B^T).
  // and we note that log |A| = log |B| + log |G^{-0.5}| = log |B| -0.5 log |G|.
  // Here, |.| actually means the absolute value of the determinant.

  int32 dim = L_.NumRows();
  double logdet_g = 0.0, logdet_b = 0.0, tr_b_bt = 0.0, tr_g = 0.0;
  BaseFloat *G_singular_values = G_rescaler_.InputSingularValues(),
      *B_singular_values = L_rescaler_.OutputSingularValues();
  for (int32 i = 0; i < dim; i++) {
    // we have already ensured that G_singular_values[i] > 0.
    logdet_g += Log(G_singular_values[i]);
    tr_g += G_singular_values[i];
    logdet_b += Log(B_singular_values[i]);
    tr_b_bt += B_singular_values[i] * B_singular_values[i];
  }

  double logdet_A = logdet_b - 0.5 * logdet_g,
      tr_at_k = TraceMatMat(*A_, K_, kTrans),
      tr_k = K_.Trace();

  return BaseFloat(
      gamma_ * logdet_A + tr_at_k - tr_k + 0.5 * tr_g - 0.5 * tr_b_bt);
}

void CoreFmllrEstimator::BackpropA(const MatrixBase<BaseFloat> &A_deriv,
                                   MatrixBase<BaseFloat> *B_deriv,
                                   MatrixBase<BaseFloat> *H_deriv) {
  B_deriv->AddMatMat(1.0, A_deriv, kNoTrans, H_, kTrans, 0.0);
  H_deriv->AddMatMat(1.0, B_, kTrans, A_deriv, kNoTrans, 0.0);
}

void CoreFmllrEstimator::BackpropL(const MatrixBase<BaseFloat> &L_deriv,
                                   MatrixBase<BaseFloat> *K_deriv,
                                   MatrixBase<BaseFloat> *H_deriv) {
  K_deriv->AddMatMat(1.0, L_deriv, kNoTrans, H_, kTrans, 0.0);
  H_deriv->AddMatMat(1.0, K_, kTrans, L_deriv, kNoTrans, 1.0);
}


void CoreFmllrEstimator::Backward(const MatrixBase<BaseFloat> &A_deriv,
                                  Matrix<BaseFloat> *G_deriv,
                                  Matrix<BaseFloat> *K_deriv) {
  KALDI_ASSERT(SameDim(A_deriv, *A_) && SameDim(A_deriv, *G_deriv)
               && SameDim(*G_deriv, *K_deriv));
  int32 dim = A_->NumRows();
  Matrix<BaseFloat> B_deriv(dim, dim), H_deriv(dim, dim),
      L_deriv(dim, dim);
  BackpropA(A_deriv, &B_deriv, &H_deriv);
  // Backprop through the operation B = F(L).
  L_rescaler_.ComputeInputDeriv(B_deriv, &L_deriv);
  BackpropL(L_deriv, K_deriv, &H_deriv);
    // Backprop through the operation H = G^{-0.5}.
  G_rescaler_.ComputeInputDeriv(H_deriv, G_deriv);

  { // Make sure G_deriv is symmetric.  Use H_deriv as a temporary.
    H_deriv.CopyFromMat(*G_deriv);
    G_deriv->AddMat(1.0, H_deriv, kTrans);
    G_deriv->Scale(0.5);
  }
}


GaussianEstimator::GaussianEstimator(int32 num_classes, int32 feature_dim):
    gamma_(num_classes),
    m_(num_classes, feature_dim),
    v_(num_classes),
    variance_floor_(-1), variance_sharing_weight_(-1) {
  // the floor and weight are actually set later on, in Estimate().
  KALDI_ASSERT(num_classes > 0 && feature_dim > 0);
}

void GaussianEstimator::AccStats(const MatrixBase<BaseFloat> &feats,
                                 const SubPosterior &post) {
  KALDI_ASSERT(static_cast<int32>(post.size()) == feats.NumRows());
  int32 T = feats.NumRows(),
      num_classes = m_.NumRows();
  for (int32 t = 0; t < T; t++) {
    SubVector<BaseFloat> feat(feats, t);
    const std::vector<std::pair<int32, BaseFloat> > &this_post = post[t];
    auto iter2 = this_post.begin(),
        end2 = this_post.end();
    for (; iter2 != end2; ++iter2) {
      int32 i = iter2->first;
      KALDI_ASSERT(i >= 0 && i < num_classes &&
                   "Posteriors and adaptation model mismatch");
      BaseFloat p = iter2->second;
      gamma_(i) += p;
      SubVector<BaseFloat> this_m(m_, i);
      this_m.AddVec(p, feat);
      v_(i) += p * VecVec(feat, feat);
    }
  }
}

void GaussianEstimator::Estimate(const FmllrEstimatorOptions &opts) {
  variance_floor_ = opts.variance_floor;
  variance_sharing_weight_ = opts.variance_sharing_weight;
  KALDI_ASSERT(variance_floor_ > 0.0 &&
               variance_sharing_weight_ >= 0.0 &&
               variance_sharing_weight_ <= 1.0);
  KALDI_ASSERT(mu_.NumRows() == 0 &&
               "You cannot call Estimate() twice.");
  int32 num_classes = m_.NumRows(), dim = m_.NumCols();

  mu_ = m_;
  s_.Resize(num_classes, kUndefined);
  t_.Resize(num_classes, kUndefined);
  for (int32 i = 0; i < num_classes; i++) {
    BaseFloat gamma_i = gamma_(i);
    if (gamma_i == 0.0) {
      // the i'th row of mu will already be zero.
      s_(i) = variance_floor_;
    } else {
      SubVector<BaseFloat> mu_i(mu_, i);
      // We already copied m_ to mu_.
      mu_i.Scale(1.0 / gamma_i);
      s_(i) = std::max<BaseFloat>(variance_floor_,
                                  v_(i) / (gamma_i * dim) - VecVec(mu_i, mu_i) / dim);
    }
  }

  // apply variance_sharing_weight_.
  BaseFloat gamma = gamma_.Sum(),
      s = VecVec(gamma_, s_) / gamma,
      f = variance_sharing_weight_;
  KALDI_ASSERT(gamma != 0.0 &&
               "You cannot call Estimate() with no stats.");
  for (int32 i = 0; i < num_classes; i++) {
    t_(i) = (BaseFloat(1.0) - f) * s_(i) + f * s;
  }
  // Clear the stats, which won't be needed any longer.
  m_.Resize(0, 0);
  v_.Resize(0);
}

void GaussianEstimator::AddToOutputDerivs(
    const MatrixBase<BaseFloat> &mean_derivs,
    const VectorBase<BaseFloat> &var_derivs) {
  KALDI_ASSERT(SameDim(mean_derivs, mu_) &&
               var_derivs.Dim() == t_.Dim());
  int32 num_classes = mean_derivs.NumRows(),
      dim = mean_derivs.NumCols();
  BaseFloat f = variance_sharing_weight_,
      variance_floor = variance_floor_,
      gamma = gamma_.Sum();
  KALDI_ASSERT(gamma > 0.0);
  if (m_bar_.NumRows() == 0) {
    // This is the first time this function was called.
    m_bar_.Resize(num_classes, dim);
    v_bar_.Resize(num_classes);
  }

  const VectorBase<BaseFloat> &t_bar(var_derivs);
  const MatrixBase<BaseFloat> &mu_bar(mean_derivs);
  BaseFloat s_bar = f * t_bar.Sum();
  for (int32 i = 0; i < num_classes; i++) {
    SubVector<BaseFloat> m_bar_i(m_bar_, i);
    BaseFloat gamma_i = gamma_(i);
    if (gamma_i != 0.0) {
      if (s_(i) != variance_floor) {
        BaseFloat s_bar_i = (BaseFloat(1.0) - f) * t_bar(i) + s_bar * gamma_i / gamma;
        v_bar_(i) += s_bar_i / (gamma_i * dim);
        m_bar_i.AddVec(-2.0 * s_bar_i / (gamma_i * dim), mu_.Row(i));
      }
      m_bar_i.AddVec(1.0 / gamma_i, mu_bar.Row(i));
    }
  }
}

int32 GaussianEstimator::Dim() const {
  // One of these two will be nonempty.
  return std::max(m_.NumCols(), mu_.NumCols());
}

void GaussianEstimator::AccStatsBackward(
    const MatrixBase<BaseFloat> &feats,
    const SubPosterior &post,
    const MatrixBase<BaseFloat> *feats_deriv) {
  // The equation we're implementing is:
  // \bar{x}_t = \sum_i \gamma_{t,i} (\bar{m}_i + 2\bar{v}_i x_t)
  // See the comment in the header:
  // "Notes on implementation of GaussianEstimator".
  int32 T = feats.NumRows();
  KALDI_ASSERT(static_cast<BaseFloat>(post.size() == T) &&
               SameDim(feats, *feats_deriv));
  for (int32 t = 0; t < T; t++) {
    SubVector<BaseFloat> feat(feats, t),
        feat_deriv(*feats_deriv, t);
    const std::vector<std::pair<int32, BaseFloat> > &this_post = post[t];
    auto iter2 = this_post.begin(),
        end2 = this_post.end();
    for (; iter2 != end2; ++iter2) {
      int32 i = iter2->first;
      BaseFloat p = iter2->second;
      SubVector<BaseFloat> m_bar_i(m_bar_, i);
      feat_deriv.AddVec(p, m_bar_i);
      feat_deriv.AddVec(p * 2.0 * v_bar_(i), feat);
    }
  }
}

void GaussianEstimator::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<GaussianEstimator>");
  WriteToken(os, binary, "<Stats>");
  gamma_.Write(os, binary);
  m_.Write(os, binary);
  v_.Write(os, binary);
  WriteToken(os, binary, "<Config>");
  WriteBasicType(os, binary, variance_floor_);
  WriteBasicType(os, binary, variance_sharing_weight_);
  WriteToken(os, binary, "<Mu>");
  mu_.Write(os, binary);
  WriteToken(os, binary, "<t>");
  t_.Write(os, binary);
  WriteToken(os, binary, "</GaussianEstimator>");
}

void GaussianEstimator::Add(const GaussianEstimator &other) {
  gamma_.AddVec(1.0, other.gamma_);
  m_.AddMat(1.0, other.m_);
  v_.AddVec(1.0, other.v_);
}


void GaussianEstimator::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<GaussianEstimator>", "<Stats>");
  gamma_.Read(is, binary);
  m_.Read(is, binary);
  v_.Read(is, binary);
  ExpectToken(is, binary, "<Config>");
  ReadBasicType(is, binary, &variance_floor_);
  ReadBasicType(is, binary, &variance_sharing_weight_);
  ExpectToken(is, binary, "<Mu>");
  mu_.Read(is, binary);
  ExpectToken(is, binary, "<t>");
  t_.Read(is, binary);
  ExpectToken(is, binary, "</GaussianEstimator>");
}


FmllrEstimator::FmllrEstimator(const FmllrEstimatorOptions &opts,
                               const MatrixBase<BaseFloat> &mu,
                               const VectorBase<BaseFloat> &s):
    opts_(opts), mu_(mu), s_(s), estimator_(NULL) {
  int32 num_classes = mu_.NumRows(), dim = mu_.NumCols();
  opts_.Check();

  gamma_.Resize(num_classes);
  raw_G_.Resize(dim, dim);
  z_.Resize(num_classes, dim);
}

void FmllrEstimator::AccStats(const MatrixBase<BaseFloat> &feats,
                              const SubPosterior &post) {
  KALDI_ASSERT(static_cast<int32>(post.size() == feats.NumRows()));
  int32 num_classes = mu_.NumRows(),
      dim = mu_.NumCols(),
      T = feats.NumRows();

  // Use temporaries for the stats and later add them to the stats in the class;
  // this will reduce roundoff errors if this function is called more than once.
  Vector<BaseFloat> gamma_hat_t(T, kUndefined),
      gamma(num_classes);

  for (int32 t = 0; t < T; t++) {
    auto iter = post[t].begin(), end = post[t].end();
    SubVector<BaseFloat> x_t(feats, t);
    BaseFloat this_gamma_hat_t = 0.0;
    for (; iter != end; ++iter) {
      int32 i = iter->first;
      KALDI_ASSERT(i >= 0 && i < num_classes &&
                   "Posteriors and adaptation model mismatch");
      BaseFloat gamma_ti = iter->second,
          gamma_hat_ti = gamma_ti / s_(i);
      SubVector<BaseFloat> z_i(z_, i);
      z_i.AddVec(gamma_ti, x_t);
      gamma(i) += gamma_ti;
      this_gamma_hat_t += gamma_hat_ti;
    }
    gamma_hat_t(t) = this_gamma_hat_t;
  }
  gamma_.AddVec(1.0, gamma);

  SpMatrix<BaseFloat> G(dim);
  int32 rows_per_chunk = 100;
  for (int32 offset = 0; offset < T; offset += rows_per_chunk) {
    int32 n_frames = std::min<int32>(rows_per_chunk, feats.NumRows() - offset);
    SubMatrix<BaseFloat> feats_part(feats, offset, n_frames, 0, dim);
    SubVector<BaseFloat> gamma_hat_t_part(gamma_hat_t, offset, n_frames);
    // the 0.0 value for beta means we don't double-count stats.
    G.AddMat2Vec(1.0, feats_part, kTrans, gamma_hat_t_part, 0.0);
    raw_G_.AddSp(1.0, G);
  }
}


BaseFloat FmllrEstimator::Estimate() {
  int32 dim = mu_.NumCols();
  BaseFloat gamma_tot = gamma_.Sum();
  KALDI_ASSERT(gamma_tot > 0.0 &&
               "You cannot call Estimate() with zero stats.");

  Vector<BaseFloat> s_inv(s_);
  s_inv.InvertElements();

  // compute \hat{\gamma} = \sum_i \gamma_i / s_i
  gamma_hat_tot_ = VecVec(gamma_, s_inv);

  // compute n = (1/\hat{\gamma}) \sum_i (1/s_i) z_i
  n_.Resize(dim);
  n_.AddMatVec(1.0 / gamma_hat_tot_, z_, kTrans, s_inv, 0.0);

  {  // Set m = 1/\hat{\gamma} \sum_i (\gamma_i / s_i) \mu_i.
    Vector<BaseFloat> s_inv_gamma(s_inv);
    s_inv_gamma.MulElements(gamma_);
    m_.Resize(dim);
    m_.AddMatVec(1.0 / gamma_hat_tot_, mu_, kTrans, s_inv_gamma, 0.0);
  }


  {  // Set K := \sum_i (1/s_i) \mu_i z_i^T - \hat{\gamma} m n^T
    Matrix<BaseFloat> mu_s(mu_);
    mu_s.MulRowsVec(s_inv);
    K_.Resize(dim, dim);
    K_.AddMatMat(1.0, mu_s, kTrans, z_, kNoTrans, 0.0);
    K_.AddVecVec(-gamma_hat_tot_, m_, n_);
  }

  // In AccStats(), we did raw_G := \sum_t \hat{\gamma}_t x_t x_t^T.
  // Now we do: G  = raw_G - \hat{\gamma} n n^T
  G_ = raw_G_;
  G_.AddVecVec(-gamma_hat_tot_, n_, n_);
  KALDI_ASSERT(G_.IsSymmetric(0.0001));

  A_.Resize(dim, dim, kUndefined);

  BaseFloat gamma_tot_smoothed = gamma_tot;
  {
    /*
      Add smoothing counts to gamma_tot, K_ and G_.  This prevents the matrix
      from diverging too far from the identity, and ensures more reasonable
      transform values when counts are small or dimensions large.  We can ignore
      this smoothing for computing derivatives, because it happens that it
      doesn't affect anything; the quantities gamma_, K_ and G_ are never
      consumed in the backprop phase, and the expressions for the derivatives
      w.r.t. these quantities don't change from adding an extra term.
    */
    gamma_tot_smoothed = gamma_tot + opts_.smoothing_count;
    BaseFloat s = opts_.smoothing_between_class_factor;
    K_.AddToDiag(opts_.smoothing_count * s);
    G_.AddToDiag(opts_.smoothing_count * (1.0 + s));
  }
  // Compute A_.
  estimator_ = new CoreFmllrEstimator(opts_, gamma_tot_smoothed, G_, K_, &A_);
  // A_impr will be the objective-function improvement from estimating A
  // (vs. the unit matrix), divided by gamma_tot.  Note: the likelihood of the
  // 'fake data' we used for the smoothing could only have been made worse by
  // estimating this transform, so dividing the total objf-impr by gamma_tot
  // (rather than gamma_tot_smoothed, if different) will still be an
  // underestimate of the actual improvement.
  BaseFloat A_impr = (1.0  / gamma_tot) * estimator_->Forward();

  // Compute b = m - A n.
  b_ = m_;
  b_.AddMatVec(-1.0, A_, kNoTrans, n_, 1.0);

  // b_impr is the amount of objective-function improvement from estimating b
  // (vs. the default value), divided by the total-count gamma_tot.  See section
  // 'diagnostics' in the document.
  // Note: we aren't doing any smoothing for the offset term.
  BaseFloat b_impr = (0.5 * VecVec(b_, b_) * gamma_hat_tot_) / gamma_tot;
  return A_impr + b_impr;
}

bool FmllrEstimator::IsEstimated() const {
  return A_.NumRows() != 0;
}

void FmllrEstimator::AdaptFeatures(const MatrixBase<BaseFloat> &feats,
                                   MatrixBase<BaseFloat> *adapted_feats) const {
  KALDI_ASSERT(A_.NumRows() != 0 && "You cannot call AdaptFeatures before "
               "calling Estimate().");
  KALDI_ASSERT(SameDim(feats, *adapted_feats));
  adapted_feats->CopyRowsFromVec(b_);
  adapted_feats->AddMatMat(1.0, feats, kNoTrans, A_, kTrans, 1.0);
}


void FmllrEstimator::AdaptFeaturesBackward(
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> &adapted_feats_deriv,
    MatrixBase<BaseFloat> *feats_deriv) {
  KALDI_ASSERT(SameDim(feats, adapted_feats_deriv) &&
               SameDim(feats, *feats_deriv) &&
               G_bar_.NumRows() == 0);
  int32 rows_per_chunk = 100;
  if (feats.NumRows() > rows_per_chunk) {
    // Break it up into 100-frame chunks and recurse.  This will reduce roundoff
    // error due to the way we work with temporaries.
    for (int32 offset = 0; offset < feats.NumRows(); offset += rows_per_chunk) {
      int32 n = std::min<int32>(rows_per_chunk, feats.NumRows() - offset);
      SubMatrix<BaseFloat> feats_deriv_part = feats_deriv->RowRange(offset, n);
      AdaptFeaturesBackward(feats.RowRange(offset, n),
                            adapted_feats_deriv.RowRange(offset, n),
                            &feats_deriv_part);
    }
    return;
  }

  // in the writeup: \bar{x}_t <-- A^T \bar{y}_t.
  // In this implementation, x_t corresponds to a
  // row vector in feats and feats_deriv, so everything is
  // transposed to:
  //  \bar{x}_t^T <--- \bar{y}_t^T A.
  feats_deriv->AddMatMat(1.0, adapted_feats_deriv, kNoTrans,
                         A_, kNoTrans, 1.0);

  // We use temporaries below to possibly reduce roundoff error.
  // It's not clear whether this would make a difference-- it depends
  // how the BLAS we're using was implemented.
  int32 dim = mu_.NumCols();
  // \bar{b}  =  \sum_t \bar{y}_t
  Vector<BaseFloat> b_bar(dim);
  b_bar.AddRowSumMat(1.0, adapted_feats_deriv);
  if (b_bar_.Dim() == 0)
    b_bar_.Swap(&b_bar);
  else
    b_bar_.AddVec(1.0, b_bar);
  // \bar{A} <--  \sum_t \bar{y}_t x_t^T
  Matrix<BaseFloat> A_bar(dim, dim);
  A_bar.AddMatMat(1.0, adapted_feats_deriv, kTrans, feats, kNoTrans, 0.0);
  if (A_bar_.NumRows() == 0)
    A_bar_.Swap(&A_bar);
  else
    A_bar_.AddMat(1.0, A_bar);
}

void FmllrEstimator::EstimateBackward() {
  KALDI_ASSERT(G_bar_.NumRows() == 0 &&
               "You cannot call EstimateBackward() twice.");
  KALDI_ASSERT(A_bar_.NumRows() != 0 &&
               "You must call AdaptFeaturesBackward() before calling "
               "EstimateBackward().");

  Vector<BaseFloat> s_inv(s_);
  s_inv.InvertElements();
  Vector<BaseFloat> s_inv_gamma(s_inv);
  s_inv_gamma.MulElements(gamma_);

  // do \bar{A} -= \bar{b} n^T
  A_bar_.AddVecVec(-1.0, b_bar_, n_);

  int32 num_classes = mu_.NumRows(), dim = mu_.NumCols();
  G_bar_.Resize(dim, dim);
  K_bar_.Resize(dim, dim);
  estimator_->Backward(A_bar_, &G_bar_, &K_bar_);
  delete estimator_;
  estimator_ = NULL;
  KALDI_ASSERT(G_bar_.IsSymmetric());

  // \bar{n} = - (A^T \bar{b} + 2\bar{G} n + \bar{K}^T m)
  n_bar_.Resize(dim);
  n_bar_.AddMatVec(-1.0, A_, kTrans, b_bar_, 0.0);
  n_bar_.AddMatVec(-2.0 * gamma_hat_tot_, G_bar_, kNoTrans, n_, 1.0);
  n_bar_.AddMatVec(-1.0 * gamma_hat_tot_, K_bar_, kTrans, m_, 1.0);


  // \bar{m} = \bar{b} - \hat{\gamma} \bar{K} n
  m_bar_ = b_bar_;
  m_bar_.AddMatVec(-gamma_hat_tot_, K_bar_, kNoTrans, n_, 1.0);

  //  \bar{z}_i =  (1/s_i) \bar{K}^T \mu_i  +  1/(s_i \hat{\gamma}) \bar{n}
  z_bar_.Resize(num_classes, dim);
  // set \bar{z}_i := \bar{K}^T \mu_i.  It's transposed below.
  z_bar_.AddMatMat(1.0, mu_, kNoTrans, K_bar_, kNoTrans, 0.0);
  // \bar{z}_i += 1/\hat{\gamma} \bar{n}
  z_bar_.AddVecToRows(1.0 / gamma_hat_tot_, n_bar_);
  // \bar{z}_i /= s_i
  z_bar_.MulRowsVec(s_inv);

  // \bar{\hat{\gamma}} = - n^T \bar{G} n - m^t \bar{K} n
  //                      - \frac{1}{\hat{\gamma}} (n^T \bar{n} + m^T \bar{m})
  gamma_hat_tot_bar_ = -1.0 * VecMatVec(n_, G_bar_, n_)
      - VecMatVec(m_, K_bar_, n_)
      - (1.0 / gamma_hat_tot_) * (VecVec(n_, n_bar_) + VecVec(m_, m_bar_));

  // Set \bar{mu}_i = (1/s_i) \bar{K} z_i  +  (\gamma_i / (s_i \hat{\gamma})) \bar{m}
  mu_bar_.Resize(num_classes, dim);
  mu_bar_.AddMatMat(1.0, z_, kNoTrans, K_bar_, kTrans, 0.0);
  mu_bar_.MulRowsVec(s_inv);
  mu_bar_.AddVecVec(1.0 / gamma_hat_tot_, s_inv_gamma, m_bar_);

  // Add all terms in \bar{s}_i except the one involving \bar{\hat{\gamma}}_t.
  // The full equation (also present in the header) is:
  //    \bar{s}_i  =  -(1 / s_i^2) * (
  //          \mu_i^T \bar{K} z_i  +  (1 / \hat{\gamma}) \z_i^T \bar{n}
  //       + (\gamma_i / \hat{\gamma}) \mu_i^T \bar{m}  + \gamma_i \hat{\gamma}
  //       + \sum_t  \gamma_{t,i} \bar{\hat{\gamma}}_t )
  // Noticing that some expressions in it are common with \bar{\mu}_i, this can
  // be simplified to:
  //    \bar{s}_i = (-1/s_i) \mu_i^T \bar{\mu}_i
  //          - (1/s_i^2) * ((1 / \hat{\gamma}) \z_i^T \bar{n} + \gamma_i \hat{\gamma}
  //                          + \sum_t  \gamma_{t,i} \bar{\hat{\gamma}}_t )
  s_bar_.Resize(num_classes);
  // do s_bar_ -= (1 / \hat{\gamma}) \z_i^T \bar{n}.  We'll later multiply by 1/s_i^2.
  s_bar_.AddMatVec(-1.0 / gamma_hat_tot_, z_, kNoTrans, n_bar_, 0.0);
  // do s_bar_(i) -= \gamma_i \bar{\hat{\gamma}}
  s_bar_.AddVec(-1.0 * gamma_hat_tot_bar_, gamma_);
  // do s_bar_(i) *= 1/s_i
  s_bar_.MulElements(s_inv);
  // do s_bar_(i) -= \mu_i^T \bar{\mu}_i
  s_bar_.AddDiagMatMat(-1.0, mu_, kNoTrans, mu_bar_, kTrans, 1.0);
  // do s_bar_(i) *= 1/s_i
  s_bar_.MulElements(s_inv);
  // OK, s_bar_ is now set up with all but the last term.  It remains only to do:
  // \bar{s}_i += (-1/s_i^2) \sum_t  \gamma_{t,i} \bar{\hat{\gamma}}_t )
}

void FmllrEstimator::AccStatsBackward(
    const MatrixBase<BaseFloat> &feats,
    const SubPosterior &post,
    MatrixBase<BaseFloat> *feats_deriv) {
  KALDI_ASSERT(static_cast<int32>(post.size() == feats.NumRows()));
  int32 T = feats.NumRows(), num_classes = mu_.NumRows();

  // Use temporaries for s_bar_, to reduce roundoff error.
  Vector<BaseFloat> s_bar(num_classes);
  for (int32 t = 0; t < T; t++) {
    auto iter = post[t].begin(), end = post[t].end();
    SubVector<BaseFloat> x_t(feats, t),
        x_bar_t(*feats_deriv, t);
    BaseFloat gamma_hat_t = 0.0;
    for (; iter != end; ++iter) {
      int32 i = iter->first;
      BaseFloat gamma_ti = iter->second,
          gamma_hat_ti = gamma_ti / s_(i);
      gamma_hat_t += gamma_hat_ti;
      SubVector<BaseFloat> z_bar_i(z_bar_, i);
      // \bar{x}_t += \gamma_{t,i} \bar{z}_i
      x_bar_t.AddVec(gamma_ti, z_bar_i);
    }
    double gamma_hat_bar_t = VecMatVec(x_t, G_bar_, x_t);

    // \bar{x}_t += 2 \hat{\gamma}_t \bar{G} x_t
    x_bar_t.AddMatVec(2.0 * gamma_hat_t, G_bar_, kNoTrans, x_t, 1.0);

    for (iter = post[t].begin(); iter != end; ++iter) {
      int32 i = iter->first;
      BaseFloat gamma_ti = iter->second;
      SubVector<BaseFloat> mu_i(mu_, i);
      // \bar{s}_i -= \frac{1}{s_i^2} \gamma_{t,i} \bar{\hat{\gamma}}_t
      s_bar(i) -= 1.0 / (s_(i) * s_(i)) * gamma_ti * gamma_hat_bar_t;
    }
    if (t == T - 1 || (t > 0 && t % 200 == 0)) {
      s_bar_.AddVec(1.0, s_bar);
      if (t < T - 1)
        s_bar.SetZero();
    }
  }
}

BaseFloat FmllrEstimator::ForwardCombined(
    const MatrixBase<BaseFloat> &feats,
    const SubPosterior &post,
    MatrixBase<BaseFloat> *adapted_feats) {
  AccStats(feats, post);
  BaseFloat ans = Estimate();
  AdaptFeatures(feats, adapted_feats);
  return ans;
}

void FmllrEstimator::BackwardCombined(
    const MatrixBase<BaseFloat> &feats,
    const SubPosterior &post,
    const MatrixBase<BaseFloat> &adapted_feats_deriv,
    MatrixBase<BaseFloat> *feats_deriv) {
  AdaptFeaturesBackward(feats, adapted_feats_deriv, feats_deriv);
  EstimateBackward();
  AccStatsBackward(feats, post, feats_deriv);
}

FmllrEstimator::~FmllrEstimator() {
  delete estimator_;  // in case Estimate() was never called.
}


MeanOnlyTransformEstimator::MeanOnlyTransformEstimator(
    const MatrixBase<BaseFloat> &mu): mu_(mu) {
  int32 num_classes = mu_.NumRows(),
      dim = mu_.NumCols();
  gamma_.Resize(num_classes);
  input_sum_.Resize(dim);
}

void MeanOnlyTransformEstimator::AccStats(const MatrixBase<BaseFloat> &feats,
                                          const SubPosterior &post) {
  int32 T = feats.NumRows(),
      num_classes = mu_.NumRows();
  KALDI_ASSERT(static_cast<int32>(post.size()) == T);

  for (int32 t = 0; t < T; t++) {
    BaseFloat gamma_t = 0.0;  // Total weight for this frame.
    auto iter = post[t].begin(), end = post[t].end();
    for (; iter != end; ++iter) {
      int32 i = iter->first;
      KALDI_ASSERT(i >= 0 && i < num_classes &&
                   "Posteriors and adaptation model mismatch");
      BaseFloat gamma_ti = iter->second;
      gamma_t += gamma_ti;
      gamma_(i) += gamma_ti;
    }
    SubVector<BaseFloat> feat(feats, t);
    KALDI_ASSERT(gamma_t >= 0);
    input_sum_.AddVec(gamma_t, feat);
  }
}


void MeanOnlyTransformEstimator::Estimate() {
  double tot_gamma = gamma_.Sum();
  int32 dim = mu_.NumCols();
  if (tot_gamma <= 0.0)
    KALDI_ERR << "You cannot call Estimate() if total count is zero.";
  Vector<BaseFloat> gamma_float(gamma_);
  Vector<BaseFloat> expected_mean(dim);
  expected_mean.AddMatVec(1.0 / tot_gamma, mu_, kTrans, gamma_float, 0.0);
  // basically: offset_ = expected_mean - observed_mean,
  // where observed_mean = input_sum_ / tot_gamma.
  offset_ = expected_mean;
  offset_.AddVec(-1.0 / tot_gamma, input_sum_);
  output_deriv_sum_.Resize(dim);
}

bool MeanOnlyTransformEstimator::IsEstimated() const {
  return offset_.Dim() != 0;
}

void MeanOnlyTransformEstimator::AdaptFeatures(
    const MatrixBase<BaseFloat> &feats,
    MatrixBase<BaseFloat> *adapted_feats) const {
  adapted_feats->CopyRowsFromVec(offset_);
  adapted_feats->AddMat(1.0, feats);
}

void MeanOnlyTransformEstimator::AdaptFeaturesBackward(
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> &adapted_feats_deriv,
    MatrixBase<BaseFloat> *feats_deriv) {
  int32 dim = mu_.NumCols();
  Vector<BaseFloat> output_deriv_sum(dim);
  output_deriv_sum.AddRowSumMat(1.0, adapted_feats_deriv);
  output_deriv_sum_.AddVec(1.0, output_deriv_sum);
  feats_deriv->AddMat(1.0, adapted_feats_deriv);
}

void MeanOnlyTransformEstimator::EstimateBackward() {
  int32 num_classes = mu_.NumRows(), dim = mu_.NumCols();
  mu_bar_.Resize(num_classes, dim);
  Vector<BaseFloat> gamma(gamma_),
      output_deriv_sum(output_deriv_sum_);
  BaseFloat gamma_tot = gamma_.Sum();
  KALDI_ASSERT(gamma_tot > 0.0);
  mu_bar_.AddVecVec(1.0 / gamma_tot, gamma, output_deriv_sum);

  x_deriv_ = output_deriv_sum;
  x_deriv_.Scale(-1.0 / gamma_tot);
}


void MeanOnlyTransformEstimator::AccStatsBackward(
    const MatrixBase<BaseFloat> &feats,
    const SubPosterior &post,
    MatrixBase<BaseFloat> *feats_deriv) {

  int32 T = feats.NumRows();
  // tot_weight will be the total weight of the posteriors in 'post'
  // for each frame.
  Vector<BaseFloat> tot_weight(T, kUndefined);
  for (int32 t = 0; t < T; t++) {
    BaseFloat gamma_t = 0.0;  // Total weight for this frame.
    auto iter = post[t].begin(), end = post[t].end();
    for (; iter != end; ++iter)
      gamma_t += iter->second;
    tot_weight(t) = gamma_t;
  }
  feats_deriv->AddVecVec(1.0, tot_weight, x_deriv_);
}

void MeanOnlyTransformEstimator::ForwardCombined(
    const MatrixBase<BaseFloat> &feats,
    const SubPosterior &post,
    MatrixBase<BaseFloat> *adapted_feats) {
  AccStats(feats, post);
  Estimate();
  AdaptFeatures(feats, adapted_feats);
}

void MeanOnlyTransformEstimator::BackwardCombined(
    const MatrixBase<BaseFloat> &feats,
    const SubPosterior &post,
    const MatrixBase<BaseFloat> &adapted_feats_deriv,
    MatrixBase<BaseFloat> *feats_deriv) {
  AdaptFeaturesBackward(feats, adapted_feats_deriv, feats_deriv);
  EstimateBackward();
  AccStatsBackward(feats, post, feats_deriv);
}


}  // namespace differentiable_transform
}  // namespace kaldi

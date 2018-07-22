// ivector/ivector-extractor.cc

// Copyright 2013     Daniel Povey
//           2015     David Snyder

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

#include <vector>

#include "ivector/ivector-extractor.h"
#include "util/kaldi-thread.h"

namespace kaldi {

int32 IvectorExtractor::FeatDim() const {
  KALDI_ASSERT(!M_.empty());
  return M_[0].NumRows();
}

int32 IvectorExtractor::IvectorDim() const {
  if (M_.empty()) { return 0.0; }
  else { return M_[0].NumCols(); }
}

int32 IvectorExtractor::NumGauss() const {
  return static_cast<int32>(M_.size());
}


// This function basically inverts the input and puts it in the output, but it's
// smart numerically.  It uses the prior knowledge that the "inverse_floor" can
// have no eigenvalues less than one, so it applies that floor (in double
// precision) before inverting.  This avoids certain numerical problems that can
// otherwise occur.
// static
void IvectorExtractor::InvertWithFlooring(const SpMatrix<double> &inverse_var,
                                          SpMatrix<double> *var) {
  SpMatrix<double> dbl_var(inverse_var);
  int32 dim = inverse_var.NumRows();
  Vector<double> s(dim);
  Matrix<double> P(dim, dim);
  // Solve the symmetric eigenvalue problem, inverse_var = P diag(s) P^T.
  inverse_var.Eig(&s, &P);
  s.ApplyFloor(1.0);
  s.InvertElements();
  var->AddMat2Vec(1.0, P, kNoTrans, s, 0.0);
}


void IvectorExtractor::GetIvectorDistribution(
    const IvectorExtractorUtteranceStats &utt_stats,
    VectorBase<double> *mean,
    SpMatrix<double> *var) const {
  if (!IvectorDependentWeights()) {
    Vector<double> linear(IvectorDim());
    SpMatrix<double> quadratic(IvectorDim());
    GetIvectorDistMean(utt_stats, &linear, &quadratic);
    GetIvectorDistPrior(utt_stats, &linear, &quadratic);
    if (var != NULL) {
      var->CopyFromSp(quadratic);
      var->Invert(); // now it's a variance.

      // mean of distribution = quadratic^{-1} * linear...
      mean->AddSpVec(1.0, *var, linear, 0.0);
    } else {
      quadratic.Invert();
      mean->AddSpVec(1.0, quadratic, linear, 0.0);
    }
  } else {
    Vector<double> linear(IvectorDim());
    SpMatrix<double> quadratic(IvectorDim());
    GetIvectorDistMean(utt_stats, &linear, &quadratic);
    GetIvectorDistPrior(utt_stats, &linear, &quadratic);
    // At this point, "linear" and "quadratic" contain
    // the mean and prior-related terms, and we avoid
    // recomputing those.

    Vector<double> cur_mean(IvectorDim());

    SpMatrix<double> quadratic_inv(IvectorDim());
    InvertWithFlooring(quadratic, &quadratic_inv);
    cur_mean.AddSpVec(1.0, quadratic_inv, linear, 0.0);

    KALDI_VLOG(3) << "Trace of quadratic is " << quadratic.Trace()
                  << ", condition is " << quadratic.Cond();
    KALDI_VLOG(3) << "Trace of quadratic_inv is " << quadratic_inv.Trace()
                  << ", condition is " << quadratic_inv.Cond();

    // The loop is finding successively better approximation points
    // for the quadratic expansion of the weights.
    int32 num_iters = 4;
    double change_threshold = 0.1; // If the iVector changes by less than
    // this (in 2-norm), we abort early.
    for (int32 iter = 0; iter < num_iters; iter++) {
      if (GetVerboseLevel() >= 3) {
        KALDI_VLOG(3) << "Auxf on iter " << iter << " is "
                      << GetAuxf(utt_stats, cur_mean, &quadratic_inv);
        int32 show_dim = 5;
        if (show_dim > cur_mean.Dim()) show_dim = cur_mean.Dim();
        KALDI_VLOG(3) << "Current distribution mean is "
                      << cur_mean.Range(0, show_dim) << "... "
                      << ", var trace is " << quadratic_inv.Trace();
      }
      Vector<double> this_linear(linear);
      SpMatrix<double> this_quadratic(quadratic);
      GetIvectorDistWeight(utt_stats, cur_mean,
                           &this_linear, &this_quadratic);
      InvertWithFlooring(this_quadratic, &quadratic_inv);
      Vector<double> mean_diff(cur_mean);
      cur_mean.AddSpVec(1.0, quadratic_inv, this_linear, 0.0);
      mean_diff.AddVec(-1.0, cur_mean);
      double change = mean_diff.Norm(2.0);
      KALDI_VLOG(2) << "On iter " << iter << ", iVector changed by " << change;
      if (change < change_threshold)
        break;
    }
    mean->CopyFromVec(cur_mean);
    if (var != NULL)
      var->CopyFromSp(quadratic_inv);
  }
}


IvectorExtractor::IvectorExtractor(
    const IvectorExtractorOptions &opts,
    const FullGmm &fgmm) {
  KALDI_ASSERT(opts.ivector_dim > 0);
  Sigma_inv_.resize(fgmm.NumGauss());
  for (int32 i = 0; i < fgmm.NumGauss(); i++) {
    const SpMatrix<BaseFloat> &inv_var = fgmm.inv_covars()[i];
    Sigma_inv_[i].Resize(inv_var.NumRows());
    Sigma_inv_[i].CopyFromSp(inv_var);
  }
  Matrix<double> gmm_means;
  fgmm.GetMeans(&gmm_means);
  KALDI_ASSERT(!Sigma_inv_.empty());
  int32 feature_dim = Sigma_inv_[0].NumRows(),
      num_gauss = Sigma_inv_.size();

  prior_offset_ = 100.0; // hardwired for now.  Must be nonzero.
  gmm_means.Scale(1.0 / prior_offset_);

  M_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    M_[i].Resize(feature_dim, opts.ivector_dim);
    M_[i].SetRandn();
    M_[i].CopyColFromVec(gmm_means.Row(i), 0);
  }
  if (opts.use_weights) { // will regress the log-weights on the iVector.
    w_.Resize(num_gauss, opts.ivector_dim);
  } else {
    w_vec_.Resize(fgmm.NumGauss());
    w_vec_.CopyFromVec(fgmm.weights());
  }
  ComputeDerivedVars();
}

class IvectorExtractorComputeDerivedVarsClass {
 public:
  IvectorExtractorComputeDerivedVarsClass(IvectorExtractor *extractor,
                                          int32 i):
      extractor_(extractor), i_(i) { }
  void operator () () { extractor_->ComputeDerivedVars(i_); }
 private:
  IvectorExtractor *extractor_;
  int32 i_;
};

void IvectorExtractor::ComputeDerivedVars() {
  KALDI_LOG << "Computing derived variables for iVector extractor";
  gconsts_.Resize(NumGauss());
  for (int32 i = 0; i < NumGauss(); i++) {
    double var_logdet = -Sigma_inv_[i].LogPosDefDet();
    gconsts_(i) = -0.5 * (var_logdet + FeatDim() * M_LOG_2PI);
    // the gconsts don't contain any weight-related terms.
  }
  U_.Resize(NumGauss(), IvectorDim() * (IvectorDim() + 1) / 2);
  Sigma_inv_M_.resize(NumGauss());

  // Note, we could have used RunMultiThreaded for this and similar tasks we
  // have here, but we found that we don't get as complete CPU utilization as we
  // could because some tasks finish before others.
  {
    TaskSequencerConfig sequencer_opts;
    sequencer_opts.num_threads = g_num_threads;
    TaskSequencer<IvectorExtractorComputeDerivedVarsClass> sequencer(
        sequencer_opts);
    for (int32 i = 0; i < NumGauss(); i++)
      sequencer.Run(new IvectorExtractorComputeDerivedVarsClass(this, i));
  }
  KALDI_LOG << "Done.";
}


void IvectorExtractor::ComputeDerivedVars(int32 i) {
  SpMatrix<double> temp_U(IvectorDim());
  // temp_U = M_i^T Sigma_i^{-1} M_i
  temp_U.AddMat2Sp(1.0, M_[i], kTrans, Sigma_inv_[i], 0.0);
  SubVector<double> temp_U_vec(temp_U.Data(),
                               IvectorDim() * (IvectorDim() + 1) / 2);
  U_.Row(i).CopyFromVec(temp_U_vec);

  Sigma_inv_M_[i].Resize(FeatDim(), IvectorDim());
  Sigma_inv_M_[i].AddSpMat(1.0, Sigma_inv_[i], M_[i], kNoTrans, 0.0);
}


void IvectorExtractor::GetIvectorDistWeight(
    const IvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &mean,
    VectorBase<double> *linear,
    SpMatrix<double> *quadratic) const {
  // If there is no w_, then weights do not depend on the iVector
  // and the weights contribute nothing to the distribution.
  if (!IvectorDependentWeights())
    return;

  Vector<double> logw_unnorm(NumGauss());
  logw_unnorm.AddMatVec(1.0, w_, kNoTrans, mean, 0.0);

  Vector<double> w(logw_unnorm);
  w.ApplySoftMax(); // now w is the weights.

  // See eq.58 in SGMM paper
  // http://www.sciencedirect.com/science/article/pii/S088523081000063X
  // linear_coeff(i) = \gamma_{jmi} - \gamma_{jm} \hat{w}_{jmi} + \max(\gamma_{jmi}, \gamma_{jm} \hat{w}_{jmi} \hat{\w}_i \v_{jm}
  // here \v_{jm} corresponds to the iVector.  Ignore the j,m indices.
  Vector<double> linear_coeff(NumGauss());
  Vector<double> quadratic_coeff(NumGauss());
  double gamma = utt_stats.gamma_.Sum();
  for (int32 i = 0; i < NumGauss(); i++) {
    double gamma_i = utt_stats.gamma_(i);
    double max_term = std::max(gamma_i, gamma * w(i));
    linear_coeff(i) = gamma_i - gamma * w(i) + max_term * logw_unnorm(i);
    quadratic_coeff(i) = max_term;
  }
  linear->AddMatVec(1.0, w_, kTrans, linear_coeff, 1.0);

  // *quadratic += \sum_i quadratic_coeff(i) w_i w_i^T, where w_i is
  //    i'th row of w_.
  quadratic->AddMat2Vec(1.0, w_, kTrans, quadratic_coeff, 1.0);
}

void IvectorExtractor::GetIvectorDistMean(
    const IvectorExtractorUtteranceStats &utt_stats,
    VectorBase<double> *linear,
    SpMatrix<double> *quadratic) const {
  int32 I = NumGauss();
  for (int32 i = 0; i < I; i++) {
    double gamma = utt_stats.gamma_(i);
    if (gamma != 0.0) {
      SubVector<double> x(utt_stats.X_, i); // == \gamma(i) \m_i
      // next line: a += \gamma_i \M_i^T \Sigma_i^{-1} \m_i
      linear->AddMatVec(1.0, Sigma_inv_M_[i], kTrans, x, 1.0);
    }
  }
  SubVector<double> q_vec(quadratic->Data(), IvectorDim()*(IvectorDim()+1)/2);
  q_vec.AddMatVec(1.0, U_, kTrans, utt_stats.gamma_, 1.0);
}

void IvectorExtractor::GetIvectorDistPrior(
    const IvectorExtractorUtteranceStats &utt_stats,
    VectorBase<double> *linear,
    SpMatrix<double> *quadratic) const {

  (*linear)(0) += prior_offset_; // the zero'th dimension has an offset mean.
  /// The inverse-variance for the prior is the unit matrix.
  quadratic->AddToDiag(1.0);
}


double IvectorExtractor::GetAcousticAuxfWeight(
    const IvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &mean,
    const SpMatrix<double> *var) const {
  if (!IvectorDependentWeights()) { // Not using the weight-projection matrices.
    Vector<double> log_w_vec(w_vec_);
    log_w_vec.ApplyLog();
    return VecVec(log_w_vec, utt_stats.gamma_);
  } else {
    Vector<double> w(NumGauss());
    w.AddMatVec(1.0, w_, kNoTrans, mean, 0.0);  // now w is unnormalized
    // log-weights.

    double lse = w.LogSumExp();
    w.Add(-lse); // Normalize so log-weights sum to one.

    // "ans" below is the point-value of the weight auxf, without
    // considering the variance.  At the moment, "w" contains
    // the normalized log weights.
    double ans = VecVec(w, utt_stats.gamma_);

    w.ApplyExp(); // now w is the weights.

    if (var == NULL) {
      return ans;
    } else {
      // Below, "Jacobian" will be the derivative d(log_w) / d(ivector)
      // = (I - w w^T) W, where W (w_ in the code) is the projection matrix
      // from iVector space to unnormalized log-weights, and w is the normalized
      // weight values at the current point.
      Matrix<double> Jacobian(w_);
      Vector<double> WTw(IvectorDim()); // W^T w
      WTw.AddMatVec(1.0, w_, kTrans, w, 0.0);
      Jacobian.AddVecVec(1.0, w, WTw); // Jacobian += (w (W^T w)^T = w^T w W)

      // the matrix S is the negated 2nd derivative of the objf w.r.t. the iVector \x.
      SpMatrix<double> S(IvectorDim());
      S.AddMat2Vec(1.0, Jacobian, kTrans, Vector<double>(utt_stats.gamma_), 0.0);
      ans += -0.5 * TraceSpSp(S, *var);
      return ans;
    }
  }
}



double IvectorExtractor::GetAuxf(const IvectorExtractorUtteranceStats &utt_stats,
                                 const VectorBase<double> &mean,
                                 const SpMatrix<double> *var) const {

  double acoustic_auxf = GetAcousticAuxf(utt_stats, mean, var),
      prior_auxf = GetPriorAuxf(mean, var), num_frames = utt_stats.gamma_.Sum();
  KALDI_VLOG(3) << "Acoustic auxf is " << (acoustic_auxf/num_frames) << "/frame over "
                << num_frames << " frames, prior auxf is " << prior_auxf
                << " = " << (prior_auxf/num_frames) << " per frame.";
  return acoustic_auxf + prior_auxf;
}

// gets logdet of a matrix while suppressing exceptions; always returns finite
// value even if there was a problem.
static double GetLogDetNoFailure(const SpMatrix<double> &var) {
  try {
    return var.LogPosDefDet();
  } catch (...) {
    Vector<double> eigs(var.NumRows());
    var.Eig(&eigs);
    int32 floored;
    eigs.ApplyFloor(1.0e-20, &floored);
    if (floored > 0)
      KALDI_WARN << "Floored " << floored << " eigenvalues of variance.";
    eigs.ApplyLog();
    return eigs.Sum();
  }
}

/*
  Get the prior-related part of the auxiliary function.  Suppose
  the ivector is x, the prior distribution is p(x), the likelihood
  of the data given x (itself an auxiliary function) is q(x) and
  the prior is r(x).

  In the case where we just have a point ivector x and no p(x),
  the prior-related term we return will just be q(x).

  If we have a distribution over x, we define an auxiliary function
  t(x) = \int p(x) log(q(x)r(x) / p(x)) dx.
  We separate this into data-dependent and prior parts, where the prior
  part is
  \int p(x) log(q(x) / p(x)) dx
  (which this function returns), and the acoustic part is
  \int p(x) log(r(x)) dx.
  Note that the fact that we divide by p(x) in the prior part and not
  the acoustic part is a bit arbitrary, at least the way we have written
  it down here; it doesn't matter where we do it, but this way seems
  more natural.
*/
double IvectorExtractor::GetPriorAuxf(
    const VectorBase<double> &mean,
    const SpMatrix<double> *var) const {
  KALDI_ASSERT(mean.Dim() == IvectorDim());

  Vector<double> offset(mean);
  offset(0) -= prior_offset_; // The mean of the prior distribution
  // may only be nonzero in the first dimension.  Now, "offset" is the
  // offset of ivector from the prior's mean.


  if (var == NULL) {
    // The log-determinant of the variance of the prior distribution is one,
    // since it's the unit matrix.
    return -0.5 * (VecVec(offset, offset) + IvectorDim()*M_LOG_2PI);
  } else {
    // The mean-related part of the answer will be
    // -0.5 * (VecVec(offset, offset), just like above.
    // The variance-related part will be
    //  \int p(x) . -0.5 (x^T I x - x^T var^{-1} x  + logdet(I) - logdet(var))   dx
    // and using the fact that x is distributed with variance "var", this is:
    //= \int p(x) . -0.5 (x^T I x - x^T var^{-1} x  + logdet(I) - logdet(var))   dx
    // = -0.5 ( trace(var I) - trace(var^{-1} var) + 0.0 - logdet(var))
    // = -0.5 ( trace(var) - dim(var) - logdet(var))

    KALDI_ASSERT(var->NumRows() == IvectorDim());
    return -0.5 * (VecVec(offset, offset) + var->Trace() -
                   IvectorDim() - GetLogDetNoFailure(*var));
  }
}

/* Gets the acoustic-related part of the auxf.
   If the distribution over the ivector given by "mean" and
   "var" is p(x), and the acoustic auxiliary-function given
   x is r(x), this function returns
   \int p(x) log r(x) dx

*/
double IvectorExtractor::GetAcousticAuxf(
    const IvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &mean,
    const SpMatrix<double> *var) const {
  double weight_auxf = GetAcousticAuxfWeight(utt_stats, mean, var),
      gconst_auxf = GetAcousticAuxfGconst(utt_stats),
      mean_auxf = GetAcousticAuxfMean(utt_stats, mean, var),
      var_auxf = GetAcousticAuxfVariance(utt_stats),
      T = utt_stats.gamma_.Sum();
  KALDI_VLOG(3) << "Per frame, auxf is: weight " << (weight_auxf/T) << ", gconst "
                << (gconst_auxf/T) << ", mean " << (mean_auxf/T) << ", var "
                << (var_auxf/T) << ", over " << T << " frames.";
  return weight_auxf + gconst_auxf + mean_auxf + var_auxf;
}

/* This is the part of the auxf that involves the data
   means, and we also involve the gconsts here.
   Let \m_i be the observed data mean vector for the
   i'th Gaussian (which we get from the utterance-specific
   stats).  Let \mu_i(\x) be the mean of the i'th Gaussian,
   written as a function of the iVector \x.

   \int_\x p(\x) (\sum_i \gamma_i -0.5 (\mu(\x) - \m_i)^T \Sigma_i^{-1}  (\mu(\x) - \m_i)) d\x

   To compute this integral we'll first write out the summation as a function of \x.

   \sum_i   -0.5 \gamma_i \m_i^T \Sigma_i^{-1} \m_i
   + \gamma_i \mu(\x)^T \Sigma_i^{-1} \m_i
   -0.5 \gamma_i \mu(\x)^T \Sigma_i^{-1} \m_i
   =   \sum_i   -0.5 \gamma_i \m_i^T \Sigma_i^{-1} \m_i
   + \gamma_i \x^T \M_i^T \Sigma_i^{-1} \m_i
   -0.5 \gamma_i \x^T \M_i^T \Sigma_i^{-1} \x
   =         K  + \x^T \a  - 0.5  \x^T \B \x,
   where  K =  \sum_i -0.5 \gamma_i \m_i^T \Sigma_i^{-1} \m_i
   \a = \sum_i \gamma_i  \M_i^T \Sigma_i^{-1} \m_i
   \B = \sum_i \gamma_i \U_i  (where \U_i = \M_i^T \Sigma_i^{-1} \M_i
   has been stored previously).
   Note: the matrix M in the stats actually contains \gamma_i \m_i,
   i.e. the mean times the count, so we need to modify the definitions
   above accordingly.
*/
double IvectorExtractor::GetAcousticAuxfMean(
    const IvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &mean,
    const SpMatrix<double> *var) const {
  double K = 0.0;
  Vector<double> a(IvectorDim()), temp(FeatDim());

  int32 I = NumGauss();
  for (int32 i = 0; i < I; i++) {
    double gamma = utt_stats.gamma_(i);
    if (gamma != 0.0) {
      Vector<double> x(utt_stats.X_.Row(i)); // == \gamma(i) \m_i
      temp.AddSpVec(1.0 / gamma, Sigma_inv_[i], x, 0.0);
      // now temp = Sigma_i^{-1} \m_i.
      // next line: K += -0.5 \gamma_i \m_i^T \Sigma_i^{-1} \m_i
      K += -0.5 * VecVec(x, temp);
      // next line: a += \gamma_i \M_i^T \Sigma_i^{-1} \m_i
      a.AddMatVec(gamma, M_[i], kTrans, temp, 1.0);
    }
  }
  SpMatrix<double> B(IvectorDim());
  SubVector<double> B_vec(B.Data(), IvectorDim()*(IvectorDim()+1)/2);
  B_vec.AddMatVec(1.0, U_, kTrans, Vector<double>(utt_stats.gamma_), 0.0);

  double ans = K + VecVec(mean, a) - 0.5 * VecSpVec(mean, B, mean);
  if (var != NULL)
    ans -= 0.5 * TraceSpSp(*var, B);
  return ans;
}

double IvectorExtractor::GetAcousticAuxfGconst(
    const IvectorExtractorUtteranceStats &utt_stats) const {
  return VecVec(Vector<double>(utt_stats.gamma_),
                gconsts_);
}


double IvectorExtractor::GetAcousticAuxfVariance(
    const IvectorExtractorUtteranceStats &utt_stats) const {
  if (utt_stats.S_.empty()) {
    // we did not store the variance, so assume it's as predicted
    // by the model itself.
    // for each Gaussian i, we have a term -0.5 * gamma(i) * trace(Sigma[i] * Sigma[i]^{-1})
    //   = -0.5 * gamma(i) * FeatDim().
    return -0.5 * utt_stats.gamma_.Sum() * FeatDim();
  } else {
    int32 I = NumGauss();
    double ans = 0.0;
    for (int32 i = 0; i < I; i++) {
      double gamma = utt_stats.gamma_(i);
      if (gamma != 0.0) {
        SpMatrix<double> var(utt_stats.S_[i]);
        var.Scale(1.0 / gamma);
        Vector<double> mean(utt_stats.X_.Row(i));
        mean.Scale(1.0 / gamma);
        var.AddVec2(-1.0, mean); // get centered covariance..
        ans += -0.5 * gamma * TraceSpSp(var, Sigma_inv_[i]);
      }
    }
    return ans;
  }
}

void IvectorExtractor::TransformIvectors(const MatrixBase<double> &T,
                                         double new_prior_offset) {
  Matrix<double> Tinv(T);
  Tinv.Invert();
  // w <-- w Tinv.  (construct temporary copy with Matrix<double>(w))
  if (IvectorDependentWeights())
    w_.AddMatMat(1.0, Matrix<double>(w_), kNoTrans, Tinv, kNoTrans, 0.0);
  // next: M_i <-- M_i Tinv.  (construct temporary copy with Matrix<double>(M_[i]))
  for (int32 i = 0; i < NumGauss(); i++)
    M_[i].AddMatMat(1.0, Matrix<double>(M_[i]), kNoTrans, Tinv, kNoTrans, 0.0);
  KALDI_LOG << "Setting iVector prior offset to " << new_prior_offset;
  prior_offset_ = new_prior_offset;
}

void OnlineIvectorEstimationStats::AccStats(
    const IvectorExtractor &extractor,
    const VectorBase<BaseFloat> &feature,
    const std::vector<std::pair<int32, BaseFloat> > &gauss_post) {
  KALDI_ASSERT(extractor.IvectorDim() == this->IvectorDim());
  KALDI_ASSERT(!extractor.IvectorDependentWeights());

  Vector<double> feature_dbl(feature);
  double tot_weight = 0.0;
  int32 ivector_dim = this->IvectorDim(),
      quadratic_term_dim = (ivector_dim * (ivector_dim + 1)) / 2;
  SubVector<double> quadratic_term_vec(quadratic_term_.Data(),
                                       quadratic_term_dim);

  for (size_t idx = 0; idx < gauss_post.size(); idx++) {
    int32 g = gauss_post[idx].first;
    double weight = gauss_post[idx].second;
    // allow negative weights; it's needed in the online iVector extraction
    // with speech-silence detection based on decoder traceback (we subtract
    // stuff we previously added if the traceback changes).
    if (weight == 0.0)
      continue;
    linear_term_.AddMatVec(weight, extractor.Sigma_inv_M_[g], kTrans,
                           feature_dbl, 1.0);
    SubVector<double> U_g(extractor.U_, g);
    quadratic_term_vec.AddVec(weight, U_g);
    tot_weight += weight;
  }
  if (max_count_ > 0.0) {
    // see comments in header RE max_count for explanation.  It relates to
    // prior scaling when the count exceeds max_count_
    double old_num_frames = num_frames_,
        new_num_frames = num_frames_ + tot_weight;
    double old_prior_scale = std::max(old_num_frames, max_count_) / max_count_,
        new_prior_scale = std::max(new_num_frames, max_count_) / max_count_;
    // The prior_scales are the inverses of the scales we would put on the stats
    // if we were implementing this by scaling the stats.  Instead we
    // scale the prior term.
    double prior_scale_change = new_prior_scale - old_prior_scale;
    if (prior_scale_change != 0.0) {
      linear_term_(0) += prior_offset_ * prior_scale_change;
      quadratic_term_.AddToDiag(prior_scale_change);
    }
  }

  num_frames_ += tot_weight;
}

void OnlineIvectorEstimationStats::Scale(double scale) {
  KALDI_ASSERT(scale >= 0.0 && scale <= 1.0);
  double old_num_frames = num_frames_;
  num_frames_ *= scale;
  quadratic_term_.Scale(scale);
  linear_term_.Scale(scale);

  // Scale back up the prior term, by adding in whatever we scaled down.
  if (max_count_ == 0.0) {
    linear_term_(0) += prior_offset_ * (1.0 - scale);
    quadratic_term_.AddToDiag(1.0 - scale);
  } else {
    double new_num_frames = num_frames_;
    double old_prior_scale =
        scale * std::max(old_num_frames, max_count_) / max_count_,
        new_prior_scale = std::max(new_num_frames, max_count_) / max_count_;
    // old_prior_scale is the scale the prior term currently has in the stats,
    // i.e. the previous scale times "scale" as we just scaled the stats.
    // new_prior_scale is the scale we want the prior term to have.
    linear_term_(0) += prior_offset_ * (new_prior_scale - old_prior_scale);
    quadratic_term_.AddToDiag(new_prior_scale - old_prior_scale);
  }
}

void OnlineIvectorEstimationStats::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<OnlineIvectorEstimationStats>");
  WriteToken(os, binary, "<PriorOffset>");
  WriteBasicType(os, binary, prior_offset_);
  WriteToken(os, binary, "<MaxCount>");
  WriteBasicType(os, binary, max_count_);
  WriteToken(os, binary, "<NumFrames>");
  WriteBasicType(os, binary, num_frames_);
  WriteToken(os, binary, "<QuadraticTerm>");
  quadratic_term_.Write(os, binary);
  WriteToken(os, binary, "<LinearTerm>");
  linear_term_.Write(os, binary);
  WriteToken(os, binary, "</OnlineIvectorEstimationStats>");
}

void OnlineIvectorEstimationStats::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<OnlineIvectorEstimationStats>");
  ExpectToken(is, binary, "<PriorOffset>");
  ReadBasicType(is, binary, &prior_offset_);
  std::string tok;
  ReadToken(is, binary, &tok);
  if (tok == "<MaxCount>") {
    ReadBasicType(is, binary, &max_count_);
    ExpectToken(is, binary, "<NumFrames>");
    ReadBasicType(is, binary, &num_frames_);
  } else {
    KALDI_ASSERT(tok == "<NumFrames>");
    max_count_ = 0.0;
    ReadBasicType(is, binary, &num_frames_);
  }
  ExpectToken(is, binary, "<QuadraticTerm>");
  quadratic_term_.Read(is, binary);
  ExpectToken(is, binary, "<LinearTerm>");
  linear_term_.Read(is, binary);
  ExpectToken(is, binary, "</OnlineIvectorEstimationStats>");
}

void OnlineIvectorEstimationStats::GetIvector(
    int32 num_cg_iters,
    VectorBase<double> *ivector) const {
  KALDI_ASSERT(ivector != NULL && ivector->Dim() ==
               this->IvectorDim());

  if (num_frames_ > 0.0) {
    // could be done exactly as follows:
    // SpMatrix<double> quadratic_inv(quadratic_term_);
    // quadratic_inv.Invert();
    // ivector->AddSpVec(1.0, quadratic_inv, linear_term_, 0.0);
    if ((*ivector)(0) == 0.0)
      (*ivector)(0) = prior_offset_;  // better initial guess.
    LinearCgdOptions opts;
    opts.max_iters = num_cg_iters;
    LinearCgd(opts, quadratic_term_, linear_term_, ivector);
  } else {
    // Use 'default' value.
    ivector->SetZero();
    (*ivector)(0) = prior_offset_;
  }
  KALDI_VLOG(4) << "Objective function improvement from estimating the "
                << "iVector (vs. default value) is "
                << ObjfChange(*ivector);
}

double OnlineIvectorEstimationStats::ObjfChange(
    const VectorBase<double> &ivector) const {
  double ans = Objf(ivector) - DefaultObjf();
  KALDI_ASSERT(!KALDI_ISNAN(ans));
  return ans;
}

double OnlineIvectorEstimationStats::Objf(
    const VectorBase<double> &ivector) const {
  if (num_frames_ == 0.0) {
    return 0.0;
  } else {
    return (1.0 / num_frames_) * (-0.5 * VecSpVec(ivector, quadratic_term_,
                                                  ivector)
                                  + VecVec(ivector, linear_term_));
  }
}

double OnlineIvectorEstimationStats::DefaultObjf() const {
  if (num_frames_ == 0.0) {
    return 0.0;
  } else {
    double x = prior_offset_;
    return (1.0 / num_frames_) * (-0.5 * quadratic_term_(0, 0) * x * x
                                  + x * linear_term_(0));
  }
}

OnlineIvectorEstimationStats::OnlineIvectorEstimationStats(int32 ivector_dim,
                                                           BaseFloat prior_offset,
                                                           BaseFloat max_count):
    prior_offset_(prior_offset), max_count_(max_count), num_frames_(0.0),
    quadratic_term_(ivector_dim), linear_term_(ivector_dim) {
  if (ivector_dim != 0) {
    linear_term_(0) += prior_offset;
    quadratic_term_.AddToDiag(1.0);
  }
}

OnlineIvectorEstimationStats::OnlineIvectorEstimationStats(
    const OnlineIvectorEstimationStats &other):
    prior_offset_(other.prior_offset_),
    max_count_(other.max_count_),
    num_frames_(other.num_frames_),
    quadratic_term_(other.quadratic_term_),
    linear_term_(other.linear_term_) { }



void IvectorExtractor::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<IvectorExtractor>");
  WriteToken(os, binary, "<w>");
  w_.Write(os, binary);
  WriteToken(os, binary, "<w_vec>");
  w_vec_.Write(os, binary);
  WriteToken(os, binary, "<M>");
  int32 size = M_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    M_[i].Write(os, binary);
  WriteToken(os, binary, "<SigmaInv>");
  KALDI_ASSERT(size == static_cast<int32>(Sigma_inv_.size()));
  for (int32 i = 0; i < size; i++)
    Sigma_inv_[i].Write(os, binary);
  WriteToken(os, binary, "<IvectorOffset>");
  WriteBasicType(os, binary, prior_offset_);
  WriteToken(os, binary, "</IvectorExtractor>");
}


void IvectorExtractor::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<IvectorExtractor>");
  ExpectToken(is, binary, "<w>");
  w_.Read(is, binary);
  ExpectToken(is, binary, "<w_vec>");
  w_vec_.Read(is, binary);
  ExpectToken(is, binary, "<M>");
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size > 0);
  M_.resize(size);
  for (int32 i = 0; i < size; i++)
    M_[i].Read(is, binary);
  ExpectToken(is, binary, "<SigmaInv>");
  Sigma_inv_.resize(size);
  for (int32 i = 0; i < size; i++)
    Sigma_inv_[i].Read(is, binary);
  ExpectToken(is, binary, "<IvectorOffset>");
  ReadBasicType(is, binary, &prior_offset_);
  ExpectToken(is, binary, "</IvectorExtractor>");
  ComputeDerivedVars();
}


void IvectorExtractorUtteranceStats::AccStats(
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post) {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;
  int32 num_frames = feats.NumRows(),
      num_gauss = X_.NumRows(),
      feat_dim = feats.NumCols();
  KALDI_ASSERT(X_.NumCols() == feat_dim);
  KALDI_ASSERT(feats.NumRows() == static_cast<int32>(post.size()));
  bool update_variance = (!S_.empty());
  SpMatrix<double> outer_prod(feat_dim);
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    const VecType &this_post(post[t]);
    if (update_variance) {
      outer_prod.SetZero();
      outer_prod.AddVec2(1.0, frame);
    }
    for (VecType::const_iterator iter = this_post.begin();
         iter != this_post.end(); ++iter) {
      int32 i = iter->first; // Gaussian index.
      KALDI_ASSERT(i >= 0 && i < num_gauss &&
                   "Out-of-range Gaussian (mismatched posteriors?)");
      double weight = iter->second;
      gamma_(i) += weight;
      X_.Row(i).AddVec(weight, frame);
      if (update_variance)
        S_[i].AddSp(weight, outer_prod);
    }
  }
}

void IvectorExtractorUtteranceStats::Scale(double scale) {
  gamma_.Scale(scale);
  X_.Scale(scale);
  for (size_t i = 0; i < S_.size(); i++)
    S_[i].Scale(scale);
}

IvectorExtractorStats::IvectorExtractorStats(
    const IvectorExtractor &extractor,
    const IvectorExtractorStatsOptions &stats_opts):
    config_(stats_opts) {
  int32 S = extractor.IvectorDim(), D = extractor.FeatDim(),
      I = extractor.NumGauss();

  KALDI_ASSERT(config_.num_samples_for_weights > 1);
  tot_auxf_ = 0.0;
  gamma_.Resize(I);
  Y_.resize(I);
  for (int32 i = 0; i < I; i++)
    Y_[i].Resize(D, S);
  R_.Resize(I, S * (S + 1) / 2);
  R_num_cached_ = 0;
  KALDI_ASSERT(stats_opts.cache_size > 0 && "--cache-size=0 not allowed");

  R_gamma_cache_.Resize(stats_opts.cache_size, I);
  R_ivec_scatter_cache_.Resize(stats_opts.cache_size, S*(S+1)/2);

  if (extractor.IvectorDependentWeights()) {
    Q_.Resize(I, S * (S + 1) / 2);
    G_.Resize(I, S);
  }
  if (stats_opts.update_variances) {
    S_.resize(I);
    for (int32 i = 0; i < I; i++)
      S_[i].Resize(D);
  }
  num_ivectors_ = 0;
  ivector_sum_.Resize(S);
  ivector_scatter_.Resize(S);
}


void IvectorExtractorStats::CommitStatsForM(
    const IvectorExtractor &extractor,
    const IvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &ivec_mean,
    const SpMatrix<double> &ivec_var) {

  gamma_Y_lock_.lock();

  // We do the occupation stats here also.
  gamma_.AddVec(1.0, utt_stats.gamma_);

  // Stats for the linear term in M:
  for  (int32 i = 0; i < extractor.NumGauss(); i++) {
    Y_[i].AddVecVec(1.0, utt_stats.X_.Row(i),
                    Vector<double>(ivec_mean));
  }
  gamma_Y_lock_.unlock();

  SpMatrix<double> ivec_scatter(ivec_var);
  ivec_scatter.AddVec2(1.0, ivec_mean);

  R_cache_lock_.lock();
  while (R_num_cached_ == R_gamma_cache_.NumRows()) {
    // Cache full.  The "while" statement is in case of certain race conditions.
    R_cache_lock_.unlock();
    FlushCache();
    R_cache_lock_.lock();
  }
  R_gamma_cache_.Row(R_num_cached_).CopyFromVec(utt_stats.gamma_);
  int32 ivector_dim = ivec_mean.Dim();
  SubVector<double> ivec_scatter_vec(ivec_scatter.Data(),
                                     ivector_dim * (ivector_dim + 1) / 2);
  R_ivec_scatter_cache_.Row(R_num_cached_).CopyFromVec(ivec_scatter_vec);
  R_num_cached_++;
  R_cache_lock_.unlock();
}

void IvectorExtractorStats::FlushCache() {
  R_cache_lock_.lock();
  if (R_num_cached_ > 0) {
    KALDI_VLOG(1) << "Flushing cache for IvectorExtractorStats";
    // Store these quantities as copies in memory so other threads can use the
    // cache while we update R_ from the cache.
    Matrix<double> R_gamma_cache(
        R_gamma_cache_.Range(0, R_num_cached_,
                             0, R_gamma_cache_.NumCols()));
    Matrix<double> R_ivec_scatter_cache(
        R_ivec_scatter_cache_.Range(0, R_num_cached_,
                                    0, R_ivec_scatter_cache_.NumCols()));
    R_num_cached_ = 0; // As far as other threads are concerned, the cache is
                       // cleared and they may write to it.
    R_cache_lock_.unlock();
    R_lock_.lock();
    R_.AddMatMat(1.0, R_gamma_cache, kTrans,
                 R_ivec_scatter_cache, kNoTrans, 1.0);
    R_lock_.unlock();
  } else {
    R_cache_lock_.unlock();
  }
}


void IvectorExtractorStats::CommitStatsForSigma(
    const IvectorExtractor &extractor,
    const IvectorExtractorUtteranceStats &utt_stats) {
  variance_stats_lock_.lock();
  // Storing the raw scatter statistics per Gaussian.  In the update phase we'll
  // take into account some other terms relating to the model means and their
  // correlation with the data.
  for (int32 i = 0; i < extractor.NumGauss(); i++)
    S_[i].AddSp(1.0, utt_stats.S_[i]);
  variance_stats_lock_.unlock();
}


// This function commits stats for a single sample of the ivector,
// to update the weight projection w_.
void IvectorExtractorStats::CommitStatsForWPoint(
    const IvectorExtractor &extractor,
    const IvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &ivector,
    double weight) {
  int32 num_gauss = extractor.NumGauss();
  // Compare this function with GetIvectorDistWeight(), from which it
  // was derived.
  Vector<double> logw_unnorm(num_gauss);
  logw_unnorm.AddMatVec(1.0, extractor.w_, kNoTrans, ivector, 0.0);

  Vector<double> w(logw_unnorm);
  w.ApplySoftMax(); // now w is the weights.

  Vector<double> linear_coeff(num_gauss);
  Vector<double> quadratic_coeff(num_gauss);
  double gamma = utt_stats.gamma_.Sum();
  for (int32 i = 0; i < num_gauss; i++) {
    double gamma_i = utt_stats.gamma_(i);
    double max_term = std::max(gamma_i, gamma * w(i));
    linear_coeff(i) = gamma_i - gamma * w(i) + max_term * logw_unnorm(i);
    quadratic_coeff(i) = max_term;
  }
  weight_stats_lock_.lock();
  G_.AddVecVec(weight, linear_coeff, Vector<double>(ivector));

  int32 ivector_dim = extractor.IvectorDim();
  SpMatrix<double> outer_prod(ivector_dim);
  outer_prod.AddVec2(1.0, ivector);
  SubVector<double> outer_prod_vec(outer_prod.Data(),
                                   ivector_dim * (ivector_dim + 1) / 2);
  Q_.AddVecVec(weight, quadratic_coeff, outer_prod_vec);
  weight_stats_lock_.unlock();
}

void IvectorExtractorStats::CommitStatsForW(
    const IvectorExtractor &extractor,
    const IvectorExtractorUtteranceStats &utt_stats,
    const VectorBase<double> &ivec_mean,
    const SpMatrix<double> &ivec_var) {
  KALDI_ASSERT(config_.num_samples_for_weights > 1);

  Matrix<double> rand(config_.num_samples_for_weights, extractor.IvectorDim());
  rand.SetRandn();
  TpMatrix<double> ivec_stddev(extractor.IvectorDim());
  ivec_stddev.Cholesky(ivec_var);
  Matrix<double> ivecs(config_.num_samples_for_weights, extractor.IvectorDim());
  ivecs.AddMatTp(1.0, rand, kNoTrans, ivec_stddev, kTrans, 0.0);
  // Now make the ivecs zero-mean
  Vector<double> avg_ivec(extractor.IvectorDim());
  avg_ivec.AddRowSumMat(1.0 / config_.num_samples_for_weights, ivecs);
  ivecs.AddVecToRows(-1.0, avg_ivec);
  // Correct the variance for what we just did, so the expected
  // variance still has the correct value.
  ivecs.Scale(sqrt(config_.num_samples_for_weights / (config_.num_samples_for_weights - 1.0)));
  // Add the mean of the distribution to "ivecs".
  ivecs.AddVecToRows(1.0, ivec_mean);
  // "ivecs" is now a sample from the iVector distribution.
  for (int32 samp = 0; samp < config_.num_samples_for_weights; samp++)
    CommitStatsForWPoint(extractor, utt_stats,
                         ivecs.Row(samp),
                         1.0 / config_.num_samples_for_weights);
}

void IvectorExtractorStats::CommitStatsForPrior(
    const VectorBase<double> &ivec_mean,
    const SpMatrix<double> &ivec_var) {
  SpMatrix<double> ivec_scatter(ivec_var);
  ivec_scatter.AddVec2(1.0, ivec_mean);
  prior_stats_lock_.lock();
  num_ivectors_ += 1.0;
  ivector_sum_.AddVec(1.0, ivec_mean);
  ivector_scatter_.AddSp(1.0, ivec_scatter);
  prior_stats_lock_.unlock();
}


void IvectorExtractorStats::CommitStatsForUtterance(
    const IvectorExtractor &extractor,
    const IvectorExtractorUtteranceStats &utt_stats) {

  int32 ivector_dim = extractor.IvectorDim();
  Vector<double> ivec_mean(ivector_dim);
  SpMatrix<double> ivec_var(ivector_dim);

  extractor.GetIvectorDistribution(utt_stats,
                                   &ivec_mean,
                                   &ivec_var);

  if (config_.compute_auxf)
    tot_auxf_ += extractor.GetAuxf(utt_stats, ivec_mean, &ivec_var);

  CommitStatsForM(extractor, utt_stats, ivec_mean, ivec_var);
  if (extractor.IvectorDependentWeights())
    CommitStatsForW(extractor, utt_stats, ivec_mean, ivec_var);
  CommitStatsForPrior(ivec_mean, ivec_var);
  if (!S_.empty())
    CommitStatsForSigma(extractor, utt_stats);
}


void IvectorExtractorStats::CheckDims(const IvectorExtractor &extractor) const {
  int32 S = extractor.IvectorDim(), D = extractor.FeatDim(),
      I = extractor.NumGauss();
  KALDI_ASSERT(config_.num_samples_for_weights > 0);
  KALDI_ASSERT(gamma_.Dim() == I);
  KALDI_ASSERT(static_cast<int32>(Y_.size()) == I);
  for (int32 i = 0; i < I; i++)
    KALDI_ASSERT(Y_[i].NumRows() == D && Y_[i].NumCols() == S);
  KALDI_ASSERT(R_.NumRows() == I && R_.NumCols() == S*(S+1)/2);
  if (extractor.IvectorDependentWeights()) {
    KALDI_ASSERT(Q_.NumRows() == I && Q_.NumCols() == S*(S+1)/2);
    KALDI_ASSERT(G_.NumRows() == I && G_.NumCols() == S);
  } else {
    KALDI_ASSERT(Q_.NumRows() == 0);
    KALDI_ASSERT(G_.NumRows() == 0);
  }
  // S_ may be empty or not, depending on whether update_variances == true in
  // the options.
  if (!S_.empty()) {
    KALDI_ASSERT(static_cast<int32>(S_.size() == I));
    for (int32 i = 0; i < I; i++)
      KALDI_ASSERT(S_[i].NumRows() == D);
  }
  KALDI_ASSERT(num_ivectors_ >= 0);
  KALDI_ASSERT(ivector_sum_.Dim() == S);
  KALDI_ASSERT(ivector_scatter_.NumRows() == S);
}

void IvectorExtractorStats::AccStatsForUtterance(
    const IvectorExtractor &extractor,
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post) {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;

  CheckDims(extractor);

  int32 num_gauss = extractor.NumGauss(), feat_dim = extractor.FeatDim();

  if (feat_dim != feats.NumCols()) {
    KALDI_ERR << "Feature dimension mismatch, expected " << feat_dim
              << ", got " << feats.NumCols();
  }
  KALDI_ASSERT(static_cast<int32>(post.size()) == feats.NumRows());

  bool update_variance = (!S_.empty());

  // The zeroth and 1st-order stats are in "utt_stats".
  IvectorExtractorUtteranceStats utt_stats(num_gauss, feat_dim,
                                           update_variance);

  utt_stats.AccStats(feats, post);

  CommitStatsForUtterance(extractor, utt_stats);
}

double IvectorExtractorStats::AccStatsForUtterance(
    const IvectorExtractor &extractor,
    const MatrixBase<BaseFloat> &feats,
    const FullGmm &fgmm) {
  int32 num_frames = feats.NumRows();
  Posterior post(num_frames);

  double tot_log_like = 0.0;
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    Vector<BaseFloat> posterior(fgmm.NumGauss(), kUndefined);
    tot_log_like += fgmm.ComponentPosteriors(frame, &posterior);
    for (int32 i = 0; i < posterior.Dim(); i++)
      post[t].push_back(std::make_pair(i, posterior(i)));
  }
  AccStatsForUtterance(extractor, feats, post);
  return tot_log_like;
}

void IvectorExtractorStats::Add(const IvectorExtractorStats &other) {
  KALDI_ASSERT(config_.num_samples_for_weights ==
               other.config_.num_samples_for_weights);
  double weight = 1.0; // will later make this configurable if needed.
  tot_auxf_ += weight * other.tot_auxf_;
  gamma_.AddVec(weight, other.gamma_);
  KALDI_ASSERT(Y_.size() == other.Y_.size());
  for (size_t i = 0; i < Y_.size(); i++)
    Y_[i].AddMat(weight, other.Y_[i]);
  R_.AddMat(weight, other.R_);
  Q_.AddMat(weight, other.Q_);
  G_.AddMat(weight, other.G_);
  KALDI_ASSERT(S_.size() == other.S_.size());
  for (size_t i = 0; i < S_.size(); i++)
    S_[i].AddSp(weight, other.S_[i]);
  num_ivectors_ += weight * other.num_ivectors_;
  ivector_sum_.AddVec(weight, other.ivector_sum_);
  ivector_scatter_.AddSp(weight, other.ivector_scatter_);
}


void IvectorExtractorStats::Write(std::ostream &os, bool binary) {
  FlushCache(); // for R stats.
  ((const IvectorExtractorStats&)(*this)).Write(os, binary); // call const version.
}


void IvectorExtractorStats::Write(std::ostream &os, bool binary) const {
  KALDI_ASSERT(R_num_cached_ == 0 && "Please use the non-const Write().");
  WriteToken(os, binary, "<IvectorExtractorStats>");
  WriteToken(os, binary, "<TotAuxf>");
  WriteBasicType(os, binary, tot_auxf_);
  WriteToken(os, binary, "<gamma>");
  gamma_.Write(os, binary);
  WriteToken(os, binary, "<Y>");
  int32 size = Y_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Write(os, binary);
  WriteToken(os, binary, "<R>");
  Matrix<BaseFloat> R_float(R_);
  R_float.Write(os, binary);
  WriteToken(os, binary, "<Q>");
  Matrix<BaseFloat> Q_float(Q_);
  Q_float.Write(os, binary);
  WriteToken(os, binary, "<G>");
  G_.Write(os, binary);
  WriteToken(os, binary, "<S>");
  size = S_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    S_[i].Write(os, binary);
  WriteToken(os, binary, "<NumIvectors>");
  WriteBasicType(os, binary, num_ivectors_);
  WriteToken(os, binary, "<IvectorSum>");
  ivector_sum_.Write(os, binary);
  WriteToken(os, binary, "<IvectorScatter>");
  ivector_scatter_.Write(os, binary);
  WriteToken(os, binary, "</IvectorExtractorStats>");
}


void IvectorExtractorStats::Read(std::istream &is, bool binary, bool add) {
  ExpectToken(is, binary, "<IvectorExtractorStats>");
  ExpectToken(is, binary, "<TotAuxf>");
  ReadBasicType(is, binary, &tot_auxf_, add);
  ExpectToken(is, binary, "<gamma>");
  gamma_.Read(is, binary, add);
  ExpectToken(is, binary, "<Y>");
  int32 size;
  ReadBasicType(is, binary, &size);
  Y_.resize(size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<R>");
  R_.Read(is, binary, add);
  ExpectToken(is, binary, "<Q>");
  Q_.Read(is, binary, add);
  ExpectToken(is, binary, "<G>");
  G_.Read(is, binary, add);
  ExpectToken(is, binary, "<S>");
  ReadBasicType(is, binary, &size);
  S_.resize(size);
  for (int32 i = 0; i < size; i++)
    S_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<NumIvectors>");
  ReadBasicType(is, binary, &num_ivectors_, add);
  ExpectToken(is, binary, "<IvectorSum>");
  ivector_sum_.Read(is, binary, add);
  ExpectToken(is, binary, "<IvectorScatter>");
  ivector_scatter_.Read(is, binary, add);
  ExpectToken(is, binary, "</IvectorExtractorStats>");
}

double IvectorExtractorStats::Update(
    const IvectorExtractorEstimationOptions &opts,
    IvectorExtractor *extractor) const {
  CheckDims(*extractor);
  if (tot_auxf_ != 0.0) {
    KALDI_LOG << "Overall auxf/frame on training data was "
              << (tot_auxf_/gamma_.Sum()) << " per frame over "
              << gamma_.Sum() << " frames.";
  }

  double ans = 0.0;
  ans += UpdateProjections(opts, extractor);
  if (extractor->IvectorDependentWeights())
    ans += UpdateWeights(opts, extractor);
  if (!S_.empty())
    ans += UpdateVariances(opts, extractor);
  ans += UpdatePrior(opts, extractor); // This will also transform the ivector
                                       // space.  Note: this must be done as the
                                       // last stage, because it will make the
                                       // stats invalid for that model.
  KALDI_LOG << "Overall objective-function improvement per frame was " << ans;
  extractor->ComputeDerivedVars();
  return ans;
}

void IvectorExtractorStats::IvectorVarianceDiagnostic(
  const IvectorExtractor &extractor) {

  // W is an estimate of the total residual variance explained by the
  // speaker-adapated model.  B is an estimate of the total variance
  // explained by the Ivector-subspace.
  SpMatrix<double> W(extractor.Sigma_inv_[0].NumRows()),
                      B(extractor.M_[0].NumRows());
  Vector<double> w(gamma_);
  w.Scale(1.0 / gamma_.Sum());
  for (int32 i = 0; i < extractor.NumGauss(); i++) {
    SpMatrix<double> Sigma_i(extractor.FeatDim());
    extractor.InvertWithFlooring(extractor.Sigma_inv_[i], &Sigma_i);
    W.AddSp(w(i), Sigma_i);
    B.AddMat2(w(i), extractor.M_[i], kNoTrans, 1.0);
  }
  double trace_W = W.Trace(),
         trace_B = B.Trace();
  KALDI_LOG << "The proportion of within-Gaussian variance explained by "
            << "the iVectors is " << trace_B / (trace_B + trace_W) << ".";
}

double IvectorExtractorStats::UpdateProjection(
    const IvectorExtractorEstimationOptions &opts,
    int32 i,
    IvectorExtractor *extractor) const {
  int32 I = extractor->NumGauss(), S = extractor->IvectorDim();
  KALDI_ASSERT(i >= 0 && i < I);
  /*
    For Gaussian index i, maximize the auxiliary function
       Q_i(x) = tr(M_i^T Sigma_i^{-1} Y_i)  - 0.5 tr(Sigma_i^{-1} M_i R_i M_i^T)
   */
  if (gamma_(i) < opts.gaussian_min_count) {
    KALDI_WARN << "Skipping Gaussian index " << i << " because count "
               << gamma_(i) << " is below min-count.";
    return 0.0;
  }
  SpMatrix<double> R(S, kUndefined), SigmaInv(extractor->Sigma_inv_[i]);
  SubVector<double> R_vec(R_, i); // i'th row of R; vectorized form of SpMatrix.
  SubVector<double> R_sp(R.Data(), S * (S+1) / 2);
  R_sp.CopyFromVec(R_vec); // copy to SpMatrix's memory.

  Matrix<double> M(extractor->M_[i]);
  SolverOptions solver_opts;
  solver_opts.name = "M";
  solver_opts.diagonal_precondition = true;
  double impr = SolveQuadraticMatrixProblem(R, Y_[i], SigmaInv, solver_opts, &M),
      gamma = gamma_(i);
  if (i < 4) {
    KALDI_VLOG(1) << "Objf impr for M for Gaussian index " << i << " is "
                  << (impr / gamma) << " per frame over " << gamma << " frames.";
  }
  extractor->M_[i].CopyFromMat(M);
  return impr;
}

void IvectorExtractorStats::GetOrthogonalIvectorTransform(
                              const SubMatrix<double> &T,
                              IvectorExtractor *extractor,
                              Matrix<double> *A) const {
  extractor->ComputeDerivedVars(); // Update the extractor->U_ matrix.
  int32 ivector_dim = extractor->IvectorDim(),
        num_gauss = extractor->NumGauss();
  int32 quad_dim = ivector_dim*(ivector_dim + 1)/2;

  // Each row of extractor->U_ is an SpMatrix. We can compute the weighted
  // avg of these rows in a SubVector that updates the data of the SpMatrix
  // Uavg.
  SpMatrix<double> Uavg(ivector_dim), Vavg(ivector_dim - 1);
  SubVector<double> uavg_vec(Uavg.Data(), quad_dim);
  if (extractor->IvectorDependentWeights()) {
    Vector<double> w_uniform(num_gauss);
    for (int32 i = 0; i < num_gauss; i++) w_uniform(i) = 1.0;
    uavg_vec.AddMatVec(1.0/num_gauss, extractor->U_, kTrans, w_uniform, 0.0);
  } else {
    uavg_vec.AddMatVec(1.0, extractor->U_, kTrans, extractor->w_vec_, 0.0);
  }

  Matrix<double> Tinv(T);
  Tinv.Invert();
  Matrix<double> Vavg_temp(Vavg), Uavg_temp(Uavg);

  Vavg_temp.AddMatMatMat(1.0, Tinv, kTrans, SubMatrix<double>(Uavg_temp,
                           1, ivector_dim-1, 1, ivector_dim-1),
                         kNoTrans, Tinv, kNoTrans, 0.0);
  Vavg.CopyFromMat(Vavg_temp);

  Vector<double> s(ivector_dim-1);
  Matrix<double> P(ivector_dim-1, ivector_dim-1);
  Vavg.Eig(&s, &P);
  SortSvd(&s, &P);
  A->Resize(P.NumCols(), P.NumRows());
  A->SetZero();
  A->AddMat(1.0, P, kTrans);
  KALDI_LOG << "Eigenvalues of Vavg: " << s;
}

class IvectorExtractorUpdateProjectionClass {
 public:
  IvectorExtractorUpdateProjectionClass(const IvectorExtractorStats &stats,
                        const IvectorExtractorEstimationOptions &opts,
                        int32 i,
                        IvectorExtractor *extractor,
                        double *tot_impr):
      stats_(stats), opts_(opts), i_(i), extractor_(extractor),
      tot_impr_(tot_impr), impr_(0.0) { }
  void operator () () {
    impr_ = stats_.UpdateProjection(opts_, i_, extractor_);
  }
  ~IvectorExtractorUpdateProjectionClass() { *tot_impr_ += impr_; }
 private:
  const IvectorExtractorStats &stats_;
  const IvectorExtractorEstimationOptions &opts_;
  int32 i_;
  IvectorExtractor *extractor_;
  double *tot_impr_;
  double impr_;
};

double IvectorExtractorStats::UpdateProjections(
    const IvectorExtractorEstimationOptions &opts,
    IvectorExtractor *extractor) const {
  int32 I = extractor->NumGauss();
  double tot_impr = 0.0;
  {
    TaskSequencerConfig sequencer_opts;
    sequencer_opts.num_threads = g_num_threads;
    TaskSequencer<IvectorExtractorUpdateProjectionClass> sequencer(
        sequencer_opts);
    for (int32 i = 0; i < I; i++)
      sequencer.Run(new IvectorExtractorUpdateProjectionClass(
          *this, opts, i, extractor, &tot_impr));
  }
  double count = gamma_.Sum();
  KALDI_LOG << "Overall objective function improvement for M (mean projections) "
            << "was " << (tot_impr / count) << " per frame over "
            << count << " frames.";
  return tot_impr / count;
}

double IvectorExtractorStats::UpdateVariances(
    const IvectorExtractorEstimationOptions &opts,
    IvectorExtractor *extractor) const {
  int32 num_gauss = extractor->NumGauss(),
      feat_dim = extractor->FeatDim(),
      ivector_dim = extractor->IvectorDim();
  KALDI_ASSERT(!S_.empty());
  double tot_objf_impr = 0.0;

  // "raw_variances" will be the variances directly from
  // the stats, without any flooring.
  std::vector<SpMatrix<double> > raw_variances(num_gauss);
  SpMatrix<double> var_floor(feat_dim);
  double var_floor_count = 0.0;

  for (int32 i = 0; i < num_gauss; i++) {
    if (gamma_(i) < opts.gaussian_min_count) continue; // warned in UpdateProjections
    SpMatrix<double> &S(raw_variances[i]);
    S = S_[i]; // Set it to the raw scatter statistics.

    // The equations for estimating the variance are similar to
    // those used in SGMMs.  We need to convert it to a centered
    // covariance, and for this we can use a combination of other
    // stats and the model parameters.

    Matrix<double> M(extractor->M_[i]);
    // Y * M^T.
    Matrix<double> YM(feat_dim, feat_dim);
    YM.AddMatMat(1.0, Y_[i], kNoTrans, M, kTrans, 0.0);
    Matrix<double> YMMY(YM, kTrans);
    YMMY.AddMat(1.0, YM);
    // Now, YMMY = Y * M^T + M * Y^T.  This is a kind of cross-term
    // between the mean and the data, which we subtract.
    SpMatrix<double> YMMY_sp(YMMY, kTakeMeanAndCheck);
    S.AddSp(-1.0, YMMY_sp);

    // Add in a mean-squared term.
    SpMatrix<double> R(ivector_dim); // will be scatter of iVectors, weighted
                                     // by count for this Gaussian.
    SubVector<double> R_vec(R.Data(),
                            ivector_dim * (ivector_dim + 1) / 2);
    R_vec.CopyFromVec(R_.Row(i)); //

    S.AddMat2Sp(1.0, M, kNoTrans, R, 1.0);

    var_floor.AddSp(1.0, S);
    var_floor_count += gamma_(i);
    S.Scale(1.0 / gamma_(i));
  }
  KALDI_ASSERT(var_floor_count > 0.0);
  KALDI_ASSERT(opts.variance_floor_factor > 0.0 &&
               opts.variance_floor_factor <= 1.0);

  var_floor.Scale(opts.variance_floor_factor / var_floor_count);

  // var_floor should not be singular in any normal case, but previously
  // we've had situations where cholesky on it failed (perhaps due to
  // people using linearly dependent features).  So we floor its
  // singular values.
  int eig_floored = var_floor.ApplyFloor(var_floor.MaxAbsEig() * 1.0e-04);
  if (eig_floored > 0) {
    KALDI_WARN << "Floored " << eig_floored << " eigenvalues of the "
               << "variance floor matrix.  This is not expected.  Maybe your "
               << "feature data is linearly dependent.";
  }

  int32 tot_num_floored = 0;
  for (int32 i = 0; i < num_gauss; i++) {
    SpMatrix<double> &S(raw_variances[i]); // un-floored variance.
    if (S.NumRows() == 0) continue; // due to low count.
    SpMatrix<double> floored_var(S);
    SpMatrix<double> old_inv_var(extractor->Sigma_inv_[i]);

    int32 num_floored = floored_var.ApplyFloor(var_floor);
    tot_num_floored += num_floored;
    if (num_floored > 0)
      KALDI_LOG << "For Gaussian index " << i << ", floored "
                << num_floored << " eigenvalues of variance.";
    // this objf is per frame;
    double old_objf = -0.5 * (TraceSpSp(S, old_inv_var) -
                              old_inv_var.LogPosDefDet());

    SpMatrix<double> new_inv_var(floored_var);
    new_inv_var.Invert();

    double new_objf = -0.5 * (TraceSpSp(S, new_inv_var) -
                                 new_inv_var.LogPosDefDet());
    if (i < 4) {
      KALDI_VLOG(1) << "Objf impr/frame for variance for Gaussian index "
                    << i << " was " << (new_objf - old_objf);
    }
    tot_objf_impr += gamma_(i) * (new_objf - old_objf);
    extractor->Sigma_inv_[i].CopyFromSp(new_inv_var);
  }
  double floored_percent = tot_num_floored * 100.0 / (num_gauss * feat_dim);
  KALDI_LOG << "Floored " << floored_percent << "% of all Gaussian eigenvalues";

  KALDI_LOG << "Overall objf impr/frame for variances was "
            << (tot_objf_impr / gamma_.Sum()) << " over "
            << gamma_.Sum() << " frames.";
  return tot_objf_impr / gamma_.Sum();
}

double IvectorExtractorStats::UpdateWeight(
    const IvectorExtractorEstimationOptions &opts,
    int32 i,
    IvectorExtractor *extractor) const {

  int32 num_gauss = extractor->NumGauss(),
      ivector_dim = extractor->IvectorDim();
  KALDI_ASSERT(i >= 0 && i < num_gauss);

  SolverOptions solver_opts;
  solver_opts.diagonal_precondition = true;
  solver_opts.name = "w";

  SubVector<double> w_i(extractor->w_, i);
  SubVector<double> g_i(G_, i);
  SpMatrix<double> Q(ivector_dim);
  SubVector<double> Q_vec(Q.Data(), ivector_dim * (ivector_dim + 1) / 2);
  Q_vec.CopyFromVec(Q_.Row(i));
  double objf_impr = SolveQuadraticProblem(Q, g_i, solver_opts, &w_i);
  if (i < 4 && gamma_(i) != 0.0) {
    KALDI_VLOG(1) << "Auxf impr/frame for Gaussian index " << i
                  << " for weights is " << (objf_impr / gamma_(i))
                  << " over " << gamma_(i) << " frames.";
  }
  return objf_impr;
}

class IvectorExtractorUpdateWeightClass {
 public:
  IvectorExtractorUpdateWeightClass(const IvectorExtractorStats &stats,
                                    const IvectorExtractorEstimationOptions &opts,
                                    int32 i,
                                    IvectorExtractor *extractor,
                                    double *tot_impr):
      stats_(stats), opts_(opts), i_(i), extractor_(extractor),
      tot_impr_(tot_impr), impr_(0.0) { }
  void operator () () {
    impr_ = stats_.UpdateWeight(opts_, i_, extractor_);
  }
  ~IvectorExtractorUpdateWeightClass() { *tot_impr_ += impr_; }
 private:
  const IvectorExtractorStats &stats_;
  const IvectorExtractorEstimationOptions &opts_;
  int32 i_;
  IvectorExtractor *extractor_;
  double *tot_impr_;
  double impr_;
};

double IvectorExtractorStats::UpdateWeights(
    const IvectorExtractorEstimationOptions &opts,
    IvectorExtractor *extractor) const {

  int32 I = extractor->NumGauss();
  double tot_impr = 0.0;
  {
    TaskSequencerConfig sequencer_opts;
    sequencer_opts.num_threads = g_num_threads;
    TaskSequencer<IvectorExtractorUpdateWeightClass> sequencer(
        sequencer_opts);
    for (int32 i = 0; i < I; i++)
      sequencer.Run(new IvectorExtractorUpdateWeightClass(
          *this, opts, i, extractor, &tot_impr));
  }

  double num_frames = gamma_.Sum();
  KALDI_LOG << "Overall auxf impr/frame from weight update is "
            << (tot_impr / num_frames) << " over "
            << num_frames << " frames.";
  return tot_impr / num_frames;
}


double IvectorExtractorStats::PriorDiagnostics(double old_prior_offset) const {
  // The iVectors had a centered covariance "covar"; we want to figure out
  // the objective-function change from rescaling.  It's as if we were
  // formerly modeling "covar" with the unit matrix, and we're now modeling
  // it with "covar" itself.  This is ignoring flooring issues.  Of course,
  // we implement it through rescaling the space, but it has the same effect.
  // We also need to take into account that before the rescaling etc., the
  // old mean might have been wrong.

  int32 ivector_dim = ivector_sum_.Dim();
  Vector<double> sum(ivector_sum_);
  sum.Scale(1.0 / num_ivectors_);
  SpMatrix<double> covar(ivector_scatter_);
  covar.Scale(1.0 / num_ivectors_);
  covar.AddVec2(-1.0, sum); // Get the centered covariance.

  // Now work out the offset from the old prior's mean.
  Vector<double> mean_offset(sum);
  mean_offset(0) -= old_prior_offset;

  SpMatrix<double> old_covar(covar); // the covariance around the old mean.
  old_covar.AddVec2(1.0, mean_offset);
  // old likelihood = -0.5 * (Trace(I old_covar) + logdet(I) + [ignored])
  double old_like = -0.5 * old_covar.Trace();
  // new likelihood is if we updated the variance to equal "covar"... this isn't
  // how we did it (we use rescaling of the ivectors) but it has the same
  // effect.  -0.5 * (Trace(covar^{-1} covar)  + logdet(covar))
  double new_like = -0.5 * (ivector_dim + covar.LogPosDefDet()),
      like_change = new_like - old_like,
      like_change_per_frame = like_change * num_ivectors_ / gamma_.Sum();

  KALDI_LOG << "Overall auxf improvement from prior is " << like_change_per_frame
            << " per frame, or " << like_change << " per iVector.";
  return like_change_per_frame; // we'll be adding this to other per-frame
                                // quantities.
}


double IvectorExtractorStats::UpdatePrior(
    const IvectorExtractorEstimationOptions &opts,
    IvectorExtractor *extractor) const {

  KALDI_ASSERT(num_ivectors_ > 0.0);
  Vector<double> sum(ivector_sum_);
  sum.Scale(1.0 / num_ivectors_);
  SpMatrix<double> covar(ivector_scatter_);
  covar.Scale(1.0 / num_ivectors_);
  covar.AddVec2(-1.0, sum); // Get the centered covariance.

  int32 ivector_dim = extractor->IvectorDim();
  Vector<double> s(ivector_dim);
  Matrix<double> P(ivector_dim, ivector_dim);
  // decompose covar = P diag(s) P^T:
  covar.Eig(&s, &P);
  KALDI_LOG << "Eigenvalues of iVector covariance range from "
            << s.Min() << " to " << s.Max();
  int32 num_floored;
  s.ApplyFloor(1.0e-07, &num_floored);
  if (num_floored > 0)
    KALDI_WARN << "Floored " << num_floored << " eigenvalues of covar "
               << "of iVectors.";

  Matrix<double> T(P, kTrans);
  { // set T to a transformation that makes covar unit
    // (modulo floored eigenvalues).
    Vector<double> scales(s);
    scales.ApplyPow(-0.5);
    T.MulRowsVec(scales);
    if (num_floored == 0) { // a check..
      SpMatrix<double> Tproj(ivector_dim);
      Tproj.AddMat2Sp(1.0, T, kNoTrans, covar, 0.0);
      KALDI_ASSERT(Tproj.IsUnit(1.0e-06));
    }
  }

  Vector<double> sum_proj(ivector_dim);
  sum_proj.AddMatVec(1.0, T, kNoTrans, sum, 0.0);

  KALDI_ASSERT(sum_proj.Norm(2.0) != 0.0);

  // We need a projection that (like T) makes "covar" unit,
  // but also that sends "sum" to a multiple of the vector e0 = [ 1 0 0 0 .. ].
  // We'll do this by a transform that follows T, of the form
  // (I - 2 a a^T), where a is unit.  [i.e. a Householder reflection].
  // Firstly, let x equal sum_proj normalized to unit length.
  // We'll let a = alpha x + beta e0, for suitable coefficients alpha and beta,
  // To project sum_proj (or equivalenty, x) to a multiple of e0, we'll need that
  // the x term in
  //  (I - 2(alpha x + beta e0)(alpha x + beta e0)  x
  // equals zero., i.e. 1 - 2 alpha (alpha x^T x + beta e0^T x) == 0,
  //    (1 - 2 alpha^2 - 2 alpha beta x0) = 0
  // To ensure that a is unit, we require that
  // (alpha x + beta e0).(alpha x + beta e0) = 1, i.e.
  //    alpha^2 + beta^2 + 2 alpha beta x0 = 1
  // at wolframalpha.com,
  // Solve[ {a^2 + b^2 + 2 a b x = 1}, {1 - 2 a^2 - 2 a b x = 0}, {a, b} ]
  // gives different solutions, but the one that keeps the offset positive
  // after projection seems to be:
  //    alpha = 1/(sqrt(2)sqrt(1 - x0)), beta = -alpha

  Matrix<double> U(ivector_dim, ivector_dim);
  U.SetUnit();
  Vector<double> x(sum_proj);
  x.Scale(1.0 / x.Norm(2.0));
  double x0 = x(0), alpha, beta;
  alpha = 1.0 / (M_SQRT2 * sqrt(1.0 - x0));
  beta = -alpha;
  Vector<double> a(x);
  a.Scale(alpha);
  a(0) += beta;
  U.AddVecVec(-2.0, a, a);

  Matrix<double> V(ivector_dim, ivector_dim);
  V.AddMatMat(1.0, U, kNoTrans, T, kNoTrans, 0.0);

  // Optionally replace transform V with V' such that V' makes the
  // covariance unit and additionally diagonalizes the quadratic
  // term.
  if (opts.diagonalize) {

    SubMatrix<double> Vsub(V, 1, V.NumRows()-1, 0, V.NumCols());
    Matrix<double> Vtemp(SubMatrix<double>(V, 1, V.NumRows()-1,
                         0, V.NumCols())),
                   A;
    GetOrthogonalIvectorTransform(SubMatrix<double>(Vtemp, 0,
                                  Vtemp.NumRows(), 1, Vtemp.NumCols()-1),
                                  extractor, &A);

    // It is necessary to exclude the first row of V in this transformation
    // so that the sum_vproj has the form [ x 0 0 0 .. ], where x > 0.
    Vsub.AddMatMat(1.0, A, kNoTrans, Vtemp, kNoTrans, 0.0);
  }

  if (num_floored == 0) { // a check..
    SpMatrix<double> Vproj(ivector_dim);
    Vproj.AddMat2Sp(1.0, V, kNoTrans, covar, 0.0);
    KALDI_ASSERT(Vproj.IsUnit(1.0e-04));
  }


  Vector<double> sum_vproj(ivector_dim);
  sum_vproj.AddMatVec(1.0, V, kNoTrans, sum, 0.0);
  // Make sure sum_vproj is of the form [ x 0 0 0 .. ] with x > 0.
  // (the x > 0 part isn't really necessary, it's just nice to know.)
  KALDI_ASSERT(ApproxEqual(sum_vproj(0), sum_vproj.Norm(2.0)));

  double ans = PriorDiagnostics(extractor->prior_offset_);

  extractor->TransformIvectors(V, sum_vproj(0));

  return ans;
}

IvectorExtractorStats::IvectorExtractorStats (
    const IvectorExtractorStats &other):
    config_(other.config_), tot_auxf_(other.tot_auxf_), gamma_(other.gamma_),
    Y_(other.Y_), R_(other.R_), R_num_cached_(other.R_num_cached_),
    R_gamma_cache_(other.R_gamma_cache_),
    R_ivec_scatter_cache_(other.R_ivec_scatter_cache_),
    Q_(other.Q_), G_(other.G_), S_(other.S_), num_ivectors_(other.num_ivectors_),
    ivector_sum_(other.ivector_sum_), ivector_scatter_(other.ivector_scatter_) {
}



double EstimateIvectorsOnline(
    const Matrix<BaseFloat> &feats,
    const Posterior &post,
    const IvectorExtractor &extractor,
    int32 ivector_period,
    int32 num_cg_iters,
    BaseFloat max_count,
    Matrix<BaseFloat> *ivectors) {

  KALDI_ASSERT(ivector_period > 0);
  KALDI_ASSERT(static_cast<int32>(post.size()) == feats.NumRows());
  int32 num_frames = feats.NumRows(),
      num_ivectors = (num_frames + ivector_period - 1) / ivector_period;

  ivectors->Resize(num_ivectors, extractor.IvectorDim());

  OnlineIvectorEstimationStats online_stats(extractor.IvectorDim(),
                                            extractor.PriorOffset(),
                                            max_count);

  double ans = 0.0;

  Vector<double> cur_ivector(extractor.IvectorDim());
  cur_ivector(0) = extractor.PriorOffset();
  for (int32 frame = 0; frame < num_frames; frame++) {
    online_stats.AccStats(extractor,
                          feats.Row(frame),
                          post[frame]);
    if (frame % ivector_period == 0) {
      online_stats.GetIvector(num_cg_iters, &cur_ivector);
      int32 ivector_index = frame / ivector_period;
      ivectors->Row(ivector_index).CopyFromVec(cur_ivector);
      if (ivector_index == num_ivectors - 1)  // last iVector
        ans = online_stats.ObjfChange(cur_ivector);
    }
  }
  return ans;
}




} // namespace kaldi

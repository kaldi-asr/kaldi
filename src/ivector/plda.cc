// ivector/plda.cc

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
#include "ivector/plda.h"

namespace kaldi {

void Plda::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Plda>");
  mean_.Write(os, binary);
  transform_.Write(os, binary);
  psi_.Write(os, binary);
  WriteToken(os, binary, "</Plda>");
}

void Plda::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Plda>");
  mean_.Read(is, binary);
  transform_.Read(is, binary);
  psi_.Read(is, binary);
  ExpectToken(is, binary, "</Plda>");
  ComputeDerivedVars();
}


void Plda::ComputeDerivedVars() {
  KALDI_ASSERT(Dim() > 0);
  offset_.Resize(Dim());
  offset_.AddMatVec(-1.0, transform_, kNoTrans, mean_, 0.0);
}


/**
   This comment explains the thinking behind the function LogLikelihoodRatio.
   The reference is "Probabilistic Linear Discriminant Analysis" by
   Sergey Ioffe, ECCV 2006.

   I'm looking at the un-numbered equation between eqs. (4) and (5),
   that says
     P(u^p | u^g_{1...n}) =  N (u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n\Psi + I})

   Here, the superscript ^p refers to the "probe" example (e.g. the example
   to be classified), and u^g_1 is the first "gallery" example, i.e. the first
   training example of that class.  \psi is the between-class covariance
   matrix, assumed to be diagonalized, and I can be interpreted as the within-class
   covariance matrix which we have made unit.

   We want the likelihood ratio P(u^p | u^g_{1..n}) / P(u^p), where the
   numerator is the probability of u^p given that it's in that class, and the
   denominator is the probability of u^p with no class assumption at all
   (e.g. in its own class).

   The expression above even works for n = 0 (e.g. the denominator of the likelihood
   ratio), where it gives us
     P(u^p) = N(u^p | 0, I + \Psi)
   i.e. it's distributed with zero mean and covarance (within + between).
   The likelihood ratio we want is:
      N(u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n \Psi + I}) /
      N(u^p | 0, I + \Psi)
   where \bar{u}^g is the mean of the "gallery examples"; and we can expand the
   log likelihood ratio as
     - 0.5 [ (u^p - m) (I + \Psi/(n \Psi + I))^{-1} (u^p - m)  +  logdet(I + \Psi/(n \Psi + I)) ]
     + 0.5 [u^p (I + \Psi) u^p  +  logdet(I + \Psi) ]
   where m = (n \Psi)/(n \Psi + I) \bar{u}^g.

 */

double Plda::GetNormalizationFactor(
    const VectorBase<double> &transformed_ivector,
    int32 num_examples) const {
  KALDI_ASSERT(num_examples > 0);
  // Work out the normalization factor.  The covariance for an average over
  // "num_examples" training iVectors equals \Psi + I/num_examples.
  Vector<double> transformed_ivector_sq(transformed_ivector);
  transformed_ivector_sq.ApplyPow(2.0);
  // inv_covar will equal 1.0 / (\Psi + I/num_examples).
  Vector<double> inv_covar(psi_);
  inv_covar.Add(1.0 / num_examples);
  inv_covar.InvertElements();
  // "transformed_ivector" should have covariance (\Psi + I/num_examples), i.e.
  // within-class/num_examples plus between-class covariance.  So
  // transformed_ivector_sq . (I/num_examples + \Psi)^{-1} should be equal to
  //  the dimension.
  double dot_prod = VecVec(inv_covar, transformed_ivector_sq);
  return sqrt(Dim() / dot_prod);
}


double Plda::TransformIvector(const PldaConfig &config,
                              const VectorBase<double> &ivector,
                              int32 num_examples,
                              VectorBase<double> *transformed_ivector) const {
  KALDI_ASSERT(ivector.Dim() == Dim() && transformed_ivector->Dim() == Dim());
  double normalization_factor;
  transformed_ivector->CopyFromVec(offset_);
  transformed_ivector->AddMatVec(1.0, transform_, kNoTrans, ivector, 1.0);
  if (config.simple_length_norm)
    normalization_factor = sqrt(transformed_ivector->Dim())
      / transformed_ivector->Norm(2.0);
  else
    normalization_factor = GetNormalizationFactor(*transformed_ivector,
                                                  num_examples);
  if (config.normalize_length)
    transformed_ivector->Scale(normalization_factor);
  return normalization_factor;
}

// "float" version of TransformIvector.
float Plda::TransformIvector(const PldaConfig &config,
                             const VectorBase<float> &ivector,
                             int32 num_examples,
                             VectorBase<float> *transformed_ivector) const {
  Vector<double> tmp(ivector), tmp_out(ivector.Dim());
  float ans = TransformIvector(config, tmp, num_examples, &tmp_out);
  transformed_ivector->CopyFromVec(tmp_out);
  return ans;
}


// There is an extended comment within this file, referencing a paper by
// Ioffe, that may clarify what this function is doing.
double Plda::LogLikelihoodRatio(
    const VectorBase<double> &transformed_train_ivector,
    int32 n, // number of training utterances.
    const VectorBase<double> &transformed_test_ivector) const {
  int32 dim = Dim();
  double loglike_given_class, loglike_without_class;
  { // work out loglike_given_class.
    // "mean" will be the mean of the distribution if it comes from the
    // training example.  The mean is \frac{n \Psi}{n \Psi + I} \bar{u}^g
    // "variance" will be the variance of that distribution, equal to
    // I + \frac{\Psi}{n\Psi + I}.
    Vector<double> mean(dim, kUndefined);
    Vector<double> variance(dim, kUndefined);
    for (int32 i = 0; i < dim; i++) {
      mean(i) = n * psi_(i) / (n * psi_(i) + 1.0)
        * transformed_train_ivector(i);
      variance(i) = 1.0 + psi_(i) / (n * psi_(i) + 1.0);
    }
    double logdet = variance.SumLog();
    Vector<double> sqdiff(transformed_test_ivector);
    sqdiff.AddVec(-1.0, mean);
    sqdiff.ApplyPow(2.0);
    variance.InvertElements();
    loglike_given_class = -0.5 * (logdet + M_LOG_2PI * dim +
                                  VecVec(sqdiff, variance));
  }
  { // work out loglike_without_class.  Here the mean is zero and the variance
    // is I + \Psi.
    Vector<double> sqdiff(transformed_test_ivector); // there is no offset.
    sqdiff.ApplyPow(2.0);
    Vector<double> variance(psi_);
    variance.Add(1.0); // I + \Psi.
    double logdet = variance.SumLog();
    variance.InvertElements();
    loglike_without_class = -0.5 * (logdet + M_LOG_2PI * dim +
                                    VecVec(sqdiff, variance));
  }
  double loglike_ratio = loglike_given_class - loglike_without_class;
  return loglike_ratio;
}


void Plda::SmoothWithinClassCovariance(double smoothing_factor) {
  KALDI_ASSERT(smoothing_factor >= 0.0 && smoothing_factor <= 1.0);
  // smoothing_factor > 1.0 is possible but wouldn't really make sense.

  KALDI_LOG << "Smoothing within-class covariance by " << smoothing_factor
            << ", Psi is initially: " << psi_;
  Vector<double> within_class_covar(Dim());
  within_class_covar.Set(1.0); // It's now the current within-class covariance
                               // (a diagonal matrix) in the space transformed
                               // by transform_.
  within_class_covar.AddVec(smoothing_factor, psi_);
  /// We now revise our estimate of the within-class covariance to this
  /// larger value.  This means that the transform has to change to as
  /// to make this new, larger covariance unit.  And our between-class
  /// covariance in this space is now less.

  psi_.DivElements(within_class_covar);
  KALDI_LOG << "New value of Psi is " << psi_;

  within_class_covar.ApplyPow(-0.5);
  transform_.MulRowsVec(within_class_covar);

  ComputeDerivedVars();
}

void PldaStats::AddSamples(double weight,
                           const Matrix<double> &group) {
  if (dim_ == 0) {
    Init(group.NumCols());
  } else {
    KALDI_ASSERT(dim_ == group.NumCols());
  }
  int32 n = group.NumRows(); // number of examples for this class
  Vector<double> *mean = new Vector<double>(dim_);
  mean->AddRowSumMat(1.0 / n, group);

  offset_scatter_.AddMat2(weight, group, kTrans, 1.0);
  // the following statement has the same effect as if we
  // had first subtracted the mean from each element of
  // the group before the statement above.
  offset_scatter_.AddVec2(-n * weight, *mean);

  class_info_.push_back(ClassInfo(weight, mean, n));

  num_classes_ ++;
  num_examples_ += n;
  class_weight_ += weight;
  example_weight_ += weight * n;

  sum_.AddVec(weight, *mean);
}

PldaStats::~PldaStats() {
  for (size_t i = 0; i < class_info_.size(); i++)
    delete class_info_[i].mean;
}

bool PldaStats::IsSorted() const {
  for (size_t i = 0; i + 1 < class_info_.size(); i++)
    if (class_info_[i+1] < class_info_[i])
      return false;
  return true;
}

void PldaStats::Init(int32 dim) {
  KALDI_ASSERT(dim_ == 0);
  dim_ = dim;
  num_classes_ = 0;
  num_examples_ = 0;
  class_weight_ = 0.0;
  example_weight_ = 0.0;
  sum_.Resize(dim);
  offset_scatter_.Resize(dim);
  KALDI_ASSERT(class_info_.empty());
}


PldaEstimator::PldaEstimator(const PldaStats &stats):
    stats_(stats) {
  KALDI_ASSERT(stats.IsSorted());
  InitParameters();
}


double PldaEstimator::ComputeObjfPart1() const {
  // Returns the part of the objf relating to offsets from the class means.
  // within_class_count equals the sum over the classes, of the weight of that
  // class (normally 1) times (1 - #examples) of that class, which equals the
  // rank of the covariance we're modeling.  We imagine that we're modeling (1 -
  // #examples) separate samples, each with the within-class covariance.. the
  // argument is a little complicated and involves an orthogonal complement of a
  // matrix whose first row computes the mean.

  double within_class_count = stats_.example_weight_ - stats_.class_weight_,
      within_logdet, det_sign;
  SpMatrix<double> inv_within_var(within_var_);
  inv_within_var.Invert(&within_logdet, &det_sign);
  KALDI_ASSERT(det_sign == 1 && "Within-class covariance is singular");

  double objf = -0.5 * (within_class_count * (within_logdet + M_LOG_2PI * Dim())
                        + TraceSpSp(inv_within_var, stats_.offset_scatter_));
  return objf;
}

double PldaEstimator::ComputeObjfPart2() const {
  double tot_objf = 0.0;

  int32 n = -1; // the number of examples for the current class
  SpMatrix<double> combined_inv_var(Dim());
  // combined_inv_var = (between_var_ + within_var_ / n)^{-1}
  double combined_var_logdet;

  for (size_t i = 0; i < stats_.class_info_.size(); i++) {
    const ClassInfo &info = stats_.class_info_[i];
    if (info.num_examples != n) {
      n = info.num_examples;
      // variance of mean of n examples is between-class + 1/n * within-class
      combined_inv_var.CopyFromSp(between_var_);
      combined_inv_var.AddSp(1.0 / n, within_var_);
      combined_inv_var.Invert(&combined_var_logdet);
    }
    Vector<double> mean (*(info.mean));
    mean.AddVec(-1.0 / stats_.class_weight_, stats_.sum_);
    tot_objf += info.weight * -0.5 * (combined_var_logdet + M_LOG_2PI * Dim()
                                      + VecSpVec(mean, combined_inv_var, mean));
  }
  return tot_objf;
}

double PldaEstimator::ComputeObjf() const {
  double ans1 = ComputeObjfPart1(),
      ans2 = ComputeObjfPart2(),
      ans = ans1 + ans2,
      example_weights = stats_.example_weight_,
      normalized_ans = ans / example_weights;
  KALDI_LOG << "Within-class objf per sample is " << (ans1 / example_weights)
            << ", between-class is " << (ans2 / example_weights)
            << ", total is " << normalized_ans;
  return normalized_ans;
}

void PldaEstimator::InitParameters() {
  within_var_.Resize(Dim());
  within_var_.SetUnit();
  between_var_.Resize(Dim());
  between_var_.SetUnit();
}

void PldaEstimator::ResetPerIterStats() {
  within_var_stats_.Resize(Dim());
  within_var_count_ = 0.0;
  between_var_stats_.Resize(Dim());
  between_var_count_ = 0.0;
}

void PldaEstimator::GetStatsFromIntraClass() {
  within_var_stats_.AddSp(1.0, stats_.offset_scatter_);
  // Note: in the normal case, the expression below will be equal to the sum
  // over the classes, of (1-n), where n is the #examples for that class.  That
  // is the rank of the scatter matrix that "offset_scatter_" has for that
  // class. [if weights other than 1.0 are used, it will be different.]
  within_var_count_ += (stats_.example_weight_ - stats_.class_weight_);
}


/**
   GetStatsFromClassMeans() is the more complicated part of PLDA estimation.
   Let's suppose the mean of a particular class is m, and suppose that
   that class had n examples.  We suppose that
     m ~ N(0, between_var_ + 1/n within_var_)
   i.e. m is Gaussian-distributed with zero mean and variance equal to the
   between-class variance plus 1/n times the within-class variance.  Now, m
   is observed (as stats_.class_info_[something].mean).  We're doing an E-M
   procedure where we treat m as the sum of two variables:
     m = x + y
   where
     x ~ N(0, between_var_)
     y ~ N(0, 1/n * within_var_)
   The distribution of x will contribute to the stats of between_var_, and
   y to within_var_.  Now, y = m - x, so we can focus on working out the
   distribution of x and then we can very simply get the distribution of y.
   The following expression also includes the likelihood of y as a function of
   x.  Note: the C is different from line to line.

   log p(x) = C - 0.5 ( x^T between_var^{-1} x  + (m-x)^T (1/n within_var)^{-1) (m-x) )
            = C - 0.5 x^T (between_var^{-1} + n within_var^{-1}) x + x^T z

            where z = n within_var^{-1} m, and we can write this as:

   log p(x) = C - 0.5 (x-w)^T (between_var^{-1} + n within_var^{-1}) (x-w)

    where x^T (between_var^{-1} + n within_var^{-1}) w = x^T z, i.e.
       (between_var^{-1} + n within_var^{-1}) w = z = n within_var^{-1} m, so

       w = (between_var^{-1} + n within_var^{-1})^{-1} * n within_var^{-1} m

    We can see that the distribution over x is Gaussian, with mean w and variance
     (between_var^{-1} + n within_var^{-1})^{-1}.
    The distribution over y is Gaussian with the same variance, and mean m - w.
    So the update to the between-var stats will be:
       between-var-stats += w w^T + (between_var^{-1} + n within_var^{-1})^{-1}.
    and the update to the within-var stats will be:
       within-var-stats += n ( (m-w) (m-w)^T (between_var^{-1} + n within_var^{-1})^{-1} ).

    The drawback of this formulation is that each time we encounter a different
    value of n (number of examples) we will have to do a different matrix
    inversion.  We'll try to improve on this later using a suitable transform.
 */

void PldaEstimator::GetStatsFromClassMeans() {
  SpMatrix<double> between_var_inv(between_var_);
  between_var_inv.Invert();
  SpMatrix<double> within_var_inv(within_var_);
  within_var_inv.Invert();
  // mixed_var will equal (between_var^{-1} + n within_var^{-1})^{-1}.
  SpMatrix<double> mixed_var(Dim());
  int32 n = -1; // the current number of examples for the class.

  for (size_t i = 0; i < stats_.class_info_.size(); i++) {
    const ClassInfo &info = stats_.class_info_[i];
    double weight = info.weight;
    if (info.num_examples != n) {
      n = info.num_examples;
      mixed_var.CopyFromSp(between_var_inv);
      mixed_var.AddSp(n, within_var_inv);
      mixed_var.Invert();
    }
    Vector<double> m = *(info.mean); // the mean for this class.
    m.AddVec(-1.0 / stats_.class_weight_, stats_.sum_); // remove global mean
    Vector<double> temp(Dim()); // n within_var^{-1} m
    temp.AddSpVec(n, within_var_inv, m, 0.0);
    Vector<double> w(Dim()); // w, as defined in the comment.
    w.AddSpVec(1.0, mixed_var, temp, 0.0);
    Vector<double> m_w(m); // m - w
    m_w.AddVec(-1.0, w);
    between_var_stats_.AddSp(weight, mixed_var);
    between_var_stats_.AddVec2(weight, w);
    between_var_count_ += weight;
    within_var_stats_.AddSp(weight * n, mixed_var);
    within_var_stats_.AddVec2(weight * n, m_w);
    within_var_count_ += weight;
  }
}

void PldaEstimator::EstimateFromStats() {
  within_var_.CopyFromSp(within_var_stats_);
  within_var_.Scale(1.0 / within_var_count_);
  between_var_.CopyFromSp(between_var_stats_);
  between_var_.Scale(1.0 / between_var_count_);

  KALDI_LOG << "Trace of within-class variance is " << within_var_.Trace();
  KALDI_LOG << "Trace of between-class variance is " << between_var_.Trace();
}


void PldaEstimator::EstimateOneIter() {
  ResetPerIterStats();
  GetStatsFromIntraClass();
  GetStatsFromClassMeans();
  EstimateFromStats();
  KALDI_VLOG(2) << "Objective function is " << ComputeObjf();
}


void PldaEstimator::Estimate(const PldaEstimationConfig &config,
                             Plda *plda) {
  KALDI_ASSERT(stats_.example_weight_ > 0 && "Cannot estimate with no stats");
  for (int32 i = 0; i < config.num_em_iters; i++) {
    KALDI_LOG << "Plda estimation iteration " << i
              << " of " << config.num_em_iters;
    EstimateOneIter();
  }
  GetOutput(plda);
}

template<class Real>
static void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                        MatrixBase<Real> *proj) {
  int32 dim = covar.NumRows();
  TpMatrix<Real> C(dim);  // Cholesky of covar, covar = C C^T
  C.Cholesky(covar);
  C.Invert();  // The matrix that makes covar unit is C^{-1}, because
               // C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.
  proj->CopyFromTp(C, kNoTrans);  // set "proj" to C^{-1}.
}


void PldaEstimator::GetOutput(Plda *plda) {
  plda->mean_ = stats_.sum_;
  plda->mean_.Scale(1.0 / stats_.class_weight_);
  KALDI_LOG << "Norm of mean of iVector distribution is "
            << plda->mean_.Norm(2.0);

  Matrix<double> transform1(Dim(), Dim());
  ComputeNormalizingTransform(within_var_, &transform1);
  // now transform is a matrix that if we project with it,
  // within_var_ becomes unit.

  // between_var_proj is between_var after projecting with transform1.
  SpMatrix<double> between_var_proj(Dim());
  between_var_proj.AddMat2Sp(1.0, transform1, kNoTrans, between_var_, 0.0);

  Matrix<double> U(Dim(), Dim());
  Vector<double> s(Dim());
  // Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
  // where U is orthogonal.
  between_var_proj.Eig(&s, &U);

  KALDI_ASSERT(s.Min() >= 0.0);
  int32 n = s.ApplyFloor(0.0);
  if (n > 0) {
    KALDI_WARN << "Floored " << n << " eigenvalues of between-class "
               << "variance to zero.";
  }
  // Sort from greatest to smallest eigenvalue.
  SortSvd(&s, &U);

  // The transform U^T will make between_var_proj diagonal with value s
  // (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
  // makes within_var_ unit and between_var_ diagonal is U^T transform1,
  // i.e. first transform1 and then U^T.

  plda->transform_.Resize(Dim(), Dim());
  plda->transform_.AddMatMat(1.0, U, kTrans, transform1, kNoTrans, 0.0);
  plda->psi_ = s;

  KALDI_LOG << "Diagonal of between-class variance in normalized space is " << s;

  if (GetVerboseLevel() >= 2) { // at higher verbose levels, do a self-test
                                // (just tests that this function does what it
                                // should).
    SpMatrix<double> tmp_within(Dim());
    tmp_within.AddMat2Sp(1.0, plda->transform_, kNoTrans, within_var_, 0.0);
    KALDI_ASSERT(tmp_within.IsUnit(0.0001));
    SpMatrix<double> tmp_between(Dim());
    tmp_between.AddMat2Sp(1.0, plda->transform_, kNoTrans, between_var_, 0.0);
    KALDI_ASSERT(tmp_between.IsDiagonal(0.0001));
    Vector<double> psi(Dim());
    psi.CopyDiagFromSp(tmp_between);
    AssertEqual(psi, plda->psi_);
  }
  plda->ComputeDerivedVars();
}

void PldaUnsupervisedAdaptor::AddStats(double weight,
                                       const Vector<double> &ivector) {
  if (mean_stats_.Dim() == 0) {
    mean_stats_.Resize(ivector.Dim());
    variance_stats_.Resize(ivector.Dim());
  }
  KALDI_ASSERT(weight >= 0.0);
  tot_weight_ += weight;
  mean_stats_.AddVec(weight, ivector);
  variance_stats_.AddVec2(weight, ivector);
}

void PldaUnsupervisedAdaptor::AddStats(double weight,
                                       const Vector<float> &ivector) {
  Vector<double> ivector_dbl(ivector);
  this->AddStats(weight, ivector_dbl);
}

void PldaUnsupervisedAdaptor::UpdatePlda(const PldaUnsupervisedAdaptorConfig &config,
                                         Plda *plda) const {
  KALDI_ASSERT(tot_weight_ > 0.0);
  int32 dim = mean_stats_.Dim();
  KALDI_ASSERT(dim == plda->Dim());
  Vector<double> mean(mean_stats_);
  mean.Scale(1.0 / tot_weight_);
  SpMatrix<double> variance(variance_stats_);
  variance.Scale(1.0 / tot_weight_);
  variance.AddVec2(-1.0, mean);  // Make it the uncentered variance.

  // mean_diff of the adaptation data from the training data.  We optionally add
  // this to our total covariance matrix
  Vector<double> mean_diff(mean);
  mean_diff.AddVec(-1.0, plda->mean_);
  KALDI_ASSERT(config.mean_diff_scale >= 0.0);
  variance.AddVec2(config.mean_diff_scale, mean_diff);

  // update the plda's mean data-member with our adaptation-data mean.
  plda->mean_.CopyFromVec(mean);


  // transform_model_ is a row-scaled version of plda->transform_ that
  // transforms into the space where the total covariance is 1.0.  Because
  // plda->transform_ transforms into a space where the within-class covar is
  // 1.0 and the the between-class covar is diag(plda->psi_), we need to scale
  // each dimension i by 1.0 / sqrt(1.0 + plda->psi_(i))

  Matrix<double> transform_mod(plda->transform_);
  for (int32 i = 0; i < dim; i++)
    transform_mod.Row(i).Scale(1.0 / sqrt(1.0 + plda->psi_(i)));

  // project the variance of the adaptation set into this space where
  // the total covariance is unit.
  SpMatrix<double> variance_proj(dim);
  variance_proj.AddMat2Sp(1.0, transform_mod, kNoTrans,
                          variance, 0.0);

  // Do eigenvalue decomposition of variance_proj; this will tell us the
  // directions in which the adaptation-data covariance is more than
  // the training-data covariance.
  Matrix<double> P(dim, dim);
  Vector<double> s(dim);
  variance_proj.Eig(&s, &P);
  SortSvd(&s, &P);
  KALDI_LOG << "Eigenvalues of adaptation-data total-covariance in space where "
            << "training-data total-covariance is unit, is: " << s;

  // W, B are the (within,between)-class covars in the space transformed by
  // transform_mod.
  SpMatrix<double> W(dim), B(dim);
  for (int32 i = 0; i < dim; i++) {
    W(i, i) =           1.0 / (1.0 + plda->psi_(i)),
    B(i, i) = plda->psi_(i) / (1.0 + plda->psi_(i));
  }

  // OK, so variance_proj (projected by transform_mod) is P diag(s) P^T.
  // Suppose that after transform_mod we project by P^T.  Then the adaptation-data's
  // variance would be P^T P diag(s) P^T P = diag(s), and the PLDA model's
  // within class variance would be P^T W P and its between-class variance would be
  // P^T B P.  We'd still have that W+B = I in this space.
  // First let's compute these projected variances... we call the "proj2" because
  // it's after the data has been projected twice (actually, transformed, as there is no
  // dimension loss), by transform_mod and then P^T.

  SpMatrix<double> Wproj2(dim), Bproj2(dim);
  Wproj2.AddMat2Sp(1.0, P, kTrans, W, 0.0);
  Bproj2.AddMat2Sp(1.0, P, kTrans, B, 0.0);

  Matrix<double> Ptrans(P, kTrans);

  SpMatrix<double> Wproj2mod(Wproj2), Bproj2mod(Bproj2);

  for (int32 i = 0; i < dim; i++) {
    // For this eigenvalue, compute the within-class covar projected with this direction,
    // and the same for between.
    BaseFloat within = Wproj2(i, i),
        between = Bproj2(i, i);
    KALDI_LOG << "For " << i << "'th eigenvalue, value is " << s(i)
              << ", within-class covar in this direction is " << within
              << ", between-class is " << between;
    if (s(i) > 1.0) {
      double excess_eig = s(i) - 1.0;
      double excess_within_covar = excess_eig * config.within_covar_scale,
          excess_between_covar = excess_eig * config.between_covar_scale;
      Wproj2mod(i, i) += excess_within_covar;
      Bproj2mod(i, i) += excess_between_covar;
    } /*
        Below I was considering a method like below, to try to scale up
        the dimensions that had less variance than expected in our sample..
        this didn't help, and actually when I set that power to +0.2 instead
        of -0.5 it gave me an improvement on sre08.  But I'm not sure
        about this.. it just doesn't seem right.
      else {
      BaseFloat scale = pow(std::max(1.0e-10, s(i)), -0.5);
      BaseFloat max_scale = 10.0;  // I'll make this configurable later.
      scale = std::min(scale, max_scale);
      Ptrans.Row(i).Scale(scale);
      } */
  }

  // combined transform "transform_mod" and then P^T that takes us to the space
  // where {W,B}proj2{,mod} are.
  Matrix<double> combined_trans(dim, dim);
  combined_trans.AddMatMat(1.0, Ptrans, kNoTrans,
                           transform_mod, kNoTrans, 0.0);
  Matrix<double> combined_trans_inv(combined_trans);  // ... and its inverse.
  combined_trans_inv.Invert();

  // Wmod and Bmod are as Wproj2 and Bproj2 but taken back into the original
  // iVector space.
  SpMatrix<double> Wmod(dim), Bmod(dim);
  Wmod.AddMat2Sp(1.0, combined_trans_inv, kNoTrans, Wproj2mod, 0.0);
  Bmod.AddMat2Sp(1.0, combined_trans_inv, kNoTrans, Bproj2mod, 0.0);

  TpMatrix<double> C(dim);
  // Do Cholesky Wmod = C C^T.  Now if we use C^{-1} as a transform, we have
  // C^{-1} W C^{-T} = I, so it makes the within-class covar unit.
  C.Cholesky(Wmod);
  TpMatrix<double> Cinv(C);
  Cinv.Invert();

  // Bmod_proj is Bmod projected by Cinv.
  SpMatrix<double> Bmod_proj(dim);
  Bmod_proj.AddTp2Sp(1.0, Cinv, kNoTrans, Bmod, 0.0);
  Vector<double> psi_new(dim);
  Matrix<double> Q(dim, dim);
  // Do symmetric eigenvalue decomposition of Bmod_proj, so
  // Bmod_proj = Q diag(psi_new) Q^T
  Bmod_proj.Eig(&psi_new, &Q);
  SortSvd(&psi_new, &Q);
  // This means that if we use Q^T as a transform, then Q^T Bmod_proj Q =
  // diag(psi_new), hence Q^T diagonalizes Bmod_proj (while leaving the
  // within-covar unit).
  // The final transform we want, that projects from our original
  // space to our newly normalized space, is:
  // first Cinv, then Q^T, i.e. the
  // matrix Q^T Cinv.
  Matrix<double> final_transform(dim, dim);
  final_transform.AddMatTp(1.0, Q, kTrans, Cinv, kNoTrans, 0.0);

  KALDI_LOG << "Old diagonal of between-class covar was: "
            << plda->psi_ << ", new diagonal is "
            << psi_new;
  plda->transform_.CopyFromMat(final_transform);
  plda->psi_.CopyFromVec(psi_new);
}

} // namespace kaldi

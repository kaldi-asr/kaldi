// gmm/full-gmm.cc

// Copyright 2009-2011  Jan Silovsky;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Microsoft Corporation
// Copyright      2012       Arnab Ghoshal
// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey);

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
#include <limits>
#include <string>
#include <queue>
#include <utility>
using std::pair;
#include <vector>
using std::vector;

#include "gmm/full-gmm.h"
#include "gmm/full-gmm-normal.h"
#include "gmm/diag-gmm.h"
#include "util/stl-utils.h"

namespace kaldi {

void FullGmm::Resize(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);
  if (gconsts_.Dim() != nmix) gconsts_.Resize(nmix);
  if (weights_.Dim() != nmix) weights_.Resize(nmix);
  if (means_invcovars_.NumRows() != nmix
      || means_invcovars_.NumCols() != dim)
    means_invcovars_.Resize(nmix, dim);
  ResizeInvCovars(nmix, dim);
}

void FullGmm::ResizeInvCovars(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);
  if (inv_covars_.size() != static_cast<size_t>(nmix))
    inv_covars_.resize(nmix);
  for (int32 i = 0; i < nmix; i++) {
    if (inv_covars_[i].NumRows() != dim) {
      inv_covars_[i].Resize(dim);
      inv_covars_[i].SetUnit();
      // must be initialized to unit for case of calling SetMeans while having
      // covars/invcovars that are not set yet (i.e. zero)
    }
  }
}

void FullGmm::CopyFromFullGmm(const FullGmm &fullgmm) {
  Resize(fullgmm.NumGauss(), fullgmm.Dim());
  gconsts_.CopyFromVec(fullgmm.gconsts_);
  weights_.CopyFromVec(fullgmm.weights_);
  means_invcovars_.CopyFromMat(fullgmm.means_invcovars_);
  int32 ncomp = NumGauss();
  for (int32 mix = 0; mix < ncomp; mix++) {
    inv_covars_[mix].CopyFromSp(fullgmm.inv_covars_[mix]);
  }
  valid_gconsts_ = fullgmm.valid_gconsts_;
}

void FullGmm::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  Resize(diaggmm.NumGauss(), diaggmm.Dim());
  gconsts_.CopyFromVec(diaggmm.gconsts());
  weights_.CopyFromVec(diaggmm.weights());
  means_invcovars_.CopyFromMat(diaggmm.means_invvars());
  int32 ncomp = NumGauss(), dim = Dim();
  for (int32 mix = 0; mix < ncomp; mix++) {
    inv_covars_[mix].SetZero();
    for (int32 d = 0; d < dim; d++) {
      inv_covars_[mix](d, d) = diaggmm.inv_vars()(mix, d);
    }
  }
  ComputeGconsts();
}

int32 FullGmm::ComputeGconsts() {
  int32 num_mix = NumGauss(),
         dim = Dim();
  KALDI_ASSERT(num_mix > 0 && dim > 0);
  BaseFloat offset = -0.5 * M_LOG_2PI * dim;  // constant term in gconst.
  int32 num_bad = 0;

  // Resize if Gaussians have been removed during Update()
  if (num_mix != gconsts_.Dim()) gconsts_.Resize(num_mix);

  for (int32 mix = 0; mix < num_mix; mix++) {
    KALDI_ASSERT(weights_(mix) >= 0);  // Cannot have negative weights.
    BaseFloat gc = Log(weights_(mix)) + offset;  // May be -inf if weights == 0
    SpMatrix<BaseFloat> covar(inv_covars_[mix]);
    covar.InvertDouble();
    BaseFloat logdet = covar.LogPosDefDet();
    gc -= 0.5 * (logdet + VecSpVec(means_invcovars_.Row(mix),
                                   covar, means_invcovars_.Row(mix)));
    // Note that mean_invcovars(mix)' * covar(mix) * mean_invcovars(mix, d) is
    // really mean' * inv(covar) * mean, since mean_invcovars(mix, d) contains
    // the inverse covariance times mean.
    // So gc is the likelihood at zero feature value.

    if (KALDI_ISNAN(gc)) {  // negative infinity is OK but NaN is not acceptable
      KALDI_ERR << "At component " << mix
                << ", not a number in gconst computation";
    }
    if (KALDI_ISINF(gc)) {
      num_bad++;
      // If positive infinity, make it negative infinity.
      // Want to make sure the answer becomes -inf in the end, not NaN.
      if (gc > 0) gc = -gc;
    }
    gconsts_(mix) = gc;
  }

  valid_gconsts_ = true;
  return num_bad;
}

void FullGmm::Split(int32 target_components, float perturb_factor,
                    vector<int32> *history) {
  if (target_components <= NumGauss() || NumGauss() == 0) {
    KALDI_WARN << "Cannot split from " << NumGauss() <<  " to "
               << target_components << " components";
    return;
  }
  int32 current_components = NumGauss(), dim = Dim();
  FullGmm *tmp = new FullGmm();
  tmp->CopyFromFullGmm(*this);  // so we have copies of matrices...
  // First do the resize:
  weights_.Resize(target_components);
  weights_.Range(0, current_components).CopyFromVec(tmp->weights_);
  means_invcovars_.Resize(target_components, dim);
  means_invcovars_.Range(0, current_components, 0,
      dim).CopyFromMat(tmp->means_invcovars_);
  ResizeInvCovars(target_components, dim);
  for (int32 mix = 0; mix < current_components; mix++) {
    inv_covars_[mix].CopyFromSp(tmp->inv_covars_[mix]);
  }
  for (int32 mix = current_components; mix < target_components; mix++) {
    inv_covars_[mix].SetZero();
  }
  gconsts_.Resize(target_components);

  delete tmp;

  // future work(arnab): Use a priority queue instead?
  while (current_components < target_components) {
    BaseFloat max_weight = weights_(0);
    int32 max_idx = 0;
    for (int32 i = 1; i < current_components; i++) {
      if (weights_(i) > max_weight) {
        max_weight = weights_(i);
        max_idx = i;
      }
    }

    // remember history
    if (history != NULL)
      history->push_back(max_idx);

    weights_(max_idx) /= 2;
    weights_(current_components) = weights_(max_idx);
    Vector<BaseFloat> rand_vec(dim);
    rand_vec.SetRandn();
    TpMatrix<BaseFloat> invcovar_l(dim);
    invcovar_l.Cholesky(inv_covars_[max_idx]);
    rand_vec.MulTp(invcovar_l, kTrans);
    inv_covars_[current_components].CopyFromSp(inv_covars_[max_idx]);
    means_invcovars_.Row(current_components).CopyFromVec(means_invcovars_.Row(
        max_idx));
    means_invcovars_.Row(current_components).AddVec(perturb_factor, rand_vec);
    means_invcovars_.Row(max_idx).AddVec(-perturb_factor, rand_vec);
    current_components++;
  }
  ComputeGconsts();
}

void FullGmm::Perturb(float perturb_factor) {
  int32 num_comps = NumGauss(),
      dim = Dim();
  Vector<BaseFloat> rand_vec(dim);
  for (int32 i = 0; i < num_comps; i++) {
    rand_vec.SetRandn();
    TpMatrix<BaseFloat> invcovar_l(dim);
    invcovar_l.Cholesky(inv_covars_[i]);
    rand_vec.MulTp(invcovar_l, kTrans);
    means_invcovars_.Row(i).AddVec(perturb_factor, rand_vec);
  }
  ComputeGconsts();
}


void FullGmm::Merge(int32 target_components, vector<int32> *history) {
  if (target_components <= 0 || NumGauss() < target_components) {
    KALDI_ERR << "Invalid argument for target number of Gaussians (="
        << target_components << ")";
  }
  if (NumGauss() == target_components) {
    KALDI_WARN << "No components merged, as target = total.";
    return;
  }

  int32 num_comp = NumGauss(), dim = Dim();

  if (target_components == 1) {  // global mean and variance
    Vector<BaseFloat> weights(weights_);
    // Undo variance inversion and multiplication of mean by this
    vector<SpMatrix<BaseFloat> > covars(num_comp);
    Matrix<BaseFloat> means(num_comp, dim);
    for (int32 i = 0; i < num_comp; i++) {
      covars[i].Resize(dim);
      covars[i].CopyFromSp(inv_covars_[i]);
      covars[i].InvertDouble();
      means.Row(i).AddSpVec(1.0, covars[i], means_invcovars_.Row(i), 0.0);
      covars[i].AddVec2(1.0, means.Row(i));
    }

    // Slightly more efficient than calling this->Resize(1, dim)
    gconsts_.Resize(1);
    weights_.Resize(1);
    means_invcovars_.Resize(1, dim);
    inv_covars_.resize(1);
    inv_covars_[0].Resize(dim);
    Vector<BaseFloat> tmp_mean(dim);

    for (int32 i = 0; i < num_comp; i++) {
      weights_(0) += weights(i);
      tmp_mean.AddVec(weights(i), means.Row(i));
      inv_covars_[0].AddSp(weights(i), covars[i]);
    }
    if (!ApproxEqual(weights_(0), 1.0, 1e-6)) {
      KALDI_WARN << "Weights sum to " << weights_(0) << ": rescaling.";
      tmp_mean.Scale(weights_(0));
      inv_covars_[0].Scale(weights_(0));
      weights_(0) = 1.0;
    }
    inv_covars_[0].AddVec2(-1.0, tmp_mean);
    inv_covars_[0].InvertDouble();
    means_invcovars_.Row(0).AddSpVec(1.0, inv_covars_[0], tmp_mean, 0.0);
    ComputeGconsts();
    return;
  }

  // If more than 1 merged component is required, do greedy bottom-up
  // clustering, always picking the pair of components that lead to the smallest
  // decrease in likelihood.
  vector<bool> discarded_component(num_comp);
  Vector<BaseFloat> logdet(num_comp);   // logdet for each component
  logdet.SetZero();
  for (int32 i = 0; i < num_comp; i++) {
    discarded_component[i] = false;
    logdet(i) += 0.5 * inv_covars_[i].LogPosDefDet();
    // +0.5 because var is inverted
  }

  // Undo variance inversion and multiplication of mean by this
  // Makes copy of means and vars for all components.
  vector<SpMatrix<BaseFloat> > vars(num_comp);
  Matrix<BaseFloat> means(num_comp, dim);
  for (int32 i = 0; i < num_comp; i++) {
    vars[i].Resize(dim);
    vars[i].CopyFromSp(inv_covars_[i]);
    vars[i].InvertDouble();
    means.Row(i).AddSpVec(1.0, vars[i], means_invcovars_.Row(i), 0.0);

    // add means square to variances; get second-order stats
    // (normalized by zero-order stats)
    vars[i].AddVec2(1.0, means.Row(i));
  }

  // compute change of likelihood for all combinations of components
  SpMatrix<BaseFloat> delta_like(num_comp);
  for (int32 i = 0; i < num_comp; i++) {
    for (int32 j = 0; j < i; j++) {
      BaseFloat w1 = weights_(i), w2 = weights_(j), w_sum = w1 + w2;
      BaseFloat merged_logdet = MergedComponentsLogdet(w1, w2,
        means.Row(i), means.Row(j), vars[i], vars[j]);
      delta_like(i, j) = w_sum * merged_logdet
        - w1 * logdet(i) - w2 * logdet(j);
    }
  }

  // Merge components with smallest impact on the loglike
  for (int32 removed = 0; removed < num_comp - target_components; removed++) {
    // Search for the least significant change in likelihood
    // (maximum of negative delta_likes)
    BaseFloat max_delta_like = -std::numeric_limits<BaseFloat>::max();
    int32 max_i = 0, max_j = 0;
    for (int32 i = 0; i < NumGauss(); i++) {
      if (discarded_component[i]) continue;
      for (int32 j = 0; j < i; j++) {
        if (discarded_component[j]) continue;
        if (delta_like(i, j) > max_delta_like) {
          max_delta_like = delta_like(i, j);
          max_i = i;
          max_j = j;
        }
      }
    }

    // make sure that different components will be merged
    KALDI_ASSERT(max_i != max_j);

    // remember history
    if (history != NULL) {
      history->push_back(max_i);
      history->push_back(max_j);
    }

    // Merge components
    BaseFloat w1 = weights_(max_i), w2 = weights_(max_j);
    BaseFloat w_sum = w1 + w2;
    // merge means
    means.Row(max_i).AddVec(w2/w1, means.Row(max_j));
    means.Row(max_i).Scale(w1/w_sum);
    // merge vars
    vars[max_i].AddSp(w2/w1, vars[max_j]);
    vars[max_i].Scale(w1/w_sum);
    // merge weights
    weights_(max_i) = w_sum;

    // Update gmm for merged component
    // copy second-order stats (normalized by zero-order stats)
    inv_covars_[max_i].CopyFromSp(vars[max_i]);
    // centralize
    inv_covars_[max_i].AddVec2(-1.0, means.Row(max_i));
    // invert
    inv_covars_[max_i].InvertDouble();
    // copy first-order stats (normalized by zero-order stats)
    // and multiply by inv_vars
    means_invcovars_.Row(max_i).AddSpVec(1.0, inv_covars_[max_i],
      means.Row(max_i), 0.0);

    // Update logdet for merged component
    logdet(max_i) += 0.5 * inv_covars_[max_i].LogPosDefDet();
    // +0.5 because var is inverted

    // Label the removed component as discarded
    discarded_component[max_j] = true;

    // Update delta_like for merged component
    for (int32 j = 0; j < num_comp; j++) {
      if ((j == max_i) || (discarded_component[j])) continue;
      BaseFloat w1 = weights_(max_i), w2 = weights_(j), w_sum = w1 + w2;
      BaseFloat merged_logdet = MergedComponentsLogdet(w1, w2,
        means.Row(max_i), means.Row(j), vars[max_i], vars[j]);
      delta_like(max_i, j) = w_sum * merged_logdet
        - w1 * logdet(max_i) - w2 * logdet(j);
      // doesn't respect lower triangular indeces,
      // relies on implicitly performed swap of coordinates if necessary
    }
  }

  // Remove the consumed components
  int32 m = 0;
  for (int32 i = 0; i < num_comp; i++) {
    if (discarded_component[i]) {
      weights_.RemoveElement(m);
      means_invcovars_.RemoveRow(m);
      inv_covars_.erase(inv_covars_.begin() + m);
    } else {
      ++m;
    }
  }

  ComputeGconsts();
}

BaseFloat FullGmm::MergePreselect(int32 target_components,
                                  const vector<pair<int32, int32> > &preselect) {
  KALDI_ASSERT(!preselect.empty());
  double ans = 0.0;
  if (target_components <= 0 || NumGauss() < target_components) {
    KALDI_WARN << "Invalid argument for target number of Gaussians (="
               << target_components << "), currently "
               << NumGauss() << ", not mixing down";
    return 0.0;
  }
  if (NumGauss() == target_components) {
    KALDI_WARN << "No components merged, as target = total.";
    return 0.0;
  }
  // likelihood change (a negative or zero value), and then the pair of indices.
  typedef pair<BaseFloat, pair<int32, int32> > QueueElem;
  std::priority_queue<QueueElem> queue;

  int32 num_comp = NumGauss(), dim = Dim();

  // Do greedy bottom-up clustering, always picking the pair of components that
  // lead to the smallest decrease in likelihood.
  vector<bool> discarded_component(num_comp);
  Vector<BaseFloat> logdet(num_comp);   // logdet for each component
  logdet.SetZero();
  for (int32 i = 0; i < num_comp; i++) {
    discarded_component[i] = false;
    logdet(i) = 0.5 * inv_covars_[i].LogPosDefDet();
    // +0.5 because var is inverted
  }

  // Undo variance inversion and multiplication of mean by
  // inverse variance.
  // Makes copy of means and vars for all components.
  vector<SpMatrix<BaseFloat> > vars(num_comp);
  Matrix<BaseFloat> means(num_comp, dim);
  for (int32 i = 0; i < num_comp; i++) {
    vars[i].Resize(dim);
    vars[i].CopyFromSp(inv_covars_[i]);
    vars[i].InvertDouble();
    means.Row(i).AddSpVec(1.0, vars[i], means_invcovars_.Row(i), 0.0);

    // add means square to variances; get second-order stats
    // (normalized by zero-order stats)
    vars[i].AddVec2(1.0, means.Row(i));
  }

  // compute change of likelihood for all combinations of components
  for (int32 i = 0; i < preselect.size(); i++) {
    int32 idx1 = preselect[i].first, idx2 = preselect[i].second;
    KALDI_ASSERT(static_cast<size_t>(idx1) < static_cast<size_t>(num_comp));
    KALDI_ASSERT(static_cast<size_t>(idx2) < static_cast<size_t>(num_comp));
    BaseFloat w1 = weights_(idx1), w2 = weights_(idx2), w_sum = w1 + w2;
    BaseFloat merged_logdet = MergedComponentsLogdet(w1, w2,
                                                     means.Row(idx1), means.Row(idx2),
                                                     vars[idx1], vars[idx2]),
        delta_like = w_sum * merged_logdet - w1 * logdet(idx1) - w2 * logdet(idx2);
    queue.push(std::make_pair(delta_like, preselect[i]));
  }

  vector<int32> mapping(num_comp);  // map of old index to where it
  // got merged to.
  for (int32 i = 0; i < num_comp; i++) mapping[i] = i;

  // Merge components with smallest impact on the loglike
  int32 removed;
  for (removed = 0;
       removed < num_comp - target_components && !queue.empty(); ) {
    QueueElem qelem = queue.top();
    queue.pop();
    BaseFloat delta_log_like_old = qelem.first;
    int32 idx1 = qelem.second.first, idx2 = qelem.second.second;
    // the next 3 lines are to handle when components got merged
    // and moved to different indices, but we still want to consider
    // merging their descendants. [descendant = current index where
    // their data is.]
    while (discarded_component[idx1]) idx1 = mapping[idx1];
    while (discarded_component[idx2]) idx2 = mapping[idx2];
    if (idx1 == idx2) continue;  // can't merge something with itself.

    BaseFloat delta_log_like;
    { // work out delta_log_like.
      BaseFloat w1 = weights_(idx1), w2 = weights_(idx2), w_sum = w1 + w2;
      BaseFloat merged_logdet = MergedComponentsLogdet(w1, w2,
                                                       means.Row(idx1), means.Row(idx2),
                                                       vars[idx1], vars[idx2]);
      delta_log_like = w_sum * merged_logdet - w1 * logdet(idx1) - w2 * logdet(idx2);
    }
    if (ApproxEqual(delta_log_like, delta_log_like_old) ||
        delta_log_like > delta_log_like_old) {
      // if the log-like change did not change, or if it actually got smaller
      // (closer to zero, more positive), then merge the components; otherwise
      // put it back on the queue.  Note: there is no test for "freshness" --
      // we assume nothing is fresh.
      BaseFloat w1 = weights_(idx1), w2 = weights_(idx2);
      BaseFloat w_sum = w1 + w2;
      // merge means
      means.Row(idx1).AddVec(w2/w1, means.Row(idx2));
      means.Row(idx1).Scale(w1/w_sum);
      // merge vars
      vars[idx1].AddSp(w2/w1, vars[idx2]);
      vars[idx1].Scale(w1/w_sum);
      // merge weights
      weights_(idx1) = w_sum;

      // Update gmm for merged component
      // copy second-order stats (normalized by zero-order stats)
      inv_covars_[idx1].CopyFromSp(vars[idx1]);
      // centralize
      inv_covars_[idx1].AddVec2(-1.0, means.Row(idx1));
      // invert
      inv_covars_[idx1].InvertDouble();
      // copy first-order stats (normalized by zero-order stats)
      // and multiply by inv_vars
      means_invcovars_.Row(idx1).AddSpVec(1.0, inv_covars_[idx1],
                                           means.Row(idx1), 0.0);

      // Update logdet for merged component
      logdet(idx1) = 0.5 * inv_covars_[idx1].LogPosDefDet();
      // +0.5 because var is inverted

      // Label the removed component as discarded
      discarded_component[idx2] = true;
      KALDI_VLOG(2) << "Delta-log-like is " << delta_log_like <<  " (merging "
                    << idx1 << " and " << idx2 << ")";
      ans += delta_log_like;
      mapping[idx2] = idx1;
      removed++;
    } else {
      QueueElem new_elem(delta_log_like, std::make_pair(idx1, idx2));
      queue.push(new_elem);  // push back more accurate elem.
    }
  }

  // Renumber the components.
  int32 cur_idx = 0;
  for (int32 i = 0; i < num_comp; i++) {
    if (mapping[i] == i) {  // This component is kept, not merged into another.
      weights_(cur_idx) = weights_(i);
      means_invcovars_.Row(cur_idx).CopyFromVec(means_invcovars_.Row(i));
      inv_covars_[cur_idx].CopyFromSp(inv_covars_[i]);
      cur_idx++;
    }
  }
  KALDI_ASSERT(cur_idx + removed == num_comp);
  gconsts_.Resize(cur_idx);
  valid_gconsts_ = false;
  weights_.Resize(cur_idx, kCopyData);
  means_invcovars_.Resize(cur_idx, Dim(), kCopyData);
  inv_covars_.resize(cur_idx);
  ComputeGconsts();
  return ans;
}


BaseFloat FullGmm::MergedComponentsLogdet(BaseFloat w1, BaseFloat w2,
                                          const VectorBase<BaseFloat> &f1,
                                          const VectorBase<BaseFloat> &f2,
                                          const SpMatrix<BaseFloat> &s1,
                                          const SpMatrix<BaseFloat> &s2)
    const {
  int32 dim = f1.Dim();
  Vector<BaseFloat> tmp_mean(dim);
  SpMatrix<BaseFloat> tmp_var(dim);
  BaseFloat merged_logdet = 0.0;

  BaseFloat w_sum = w1 + w2;
  tmp_mean.CopyFromVec(f1);
  tmp_mean.AddVec(w2/w1, f2);
  tmp_mean.Scale(w1/w_sum);
  tmp_var.CopyFromSp(s1);
  tmp_var.AddSp(w2/w1, s2);
  tmp_var.Scale(w1/w_sum);
  tmp_var.AddVec2(-1.0, tmp_mean);
  merged_logdet -= 0.5 * tmp_var.LogPosDefDet();
  // -0.5 because var is not inverted
  return merged_logdet;
}

// returns the component of the log-likelihood due to this mixture
BaseFloat FullGmm::ComponentLogLikelihood(const VectorBase<BaseFloat> &data,
                                          int32 comp_id) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  if (data.Dim() != Dim()) {
    KALDI_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
        << "mismatch " << (data.Dim()) << "vs. "<< (Dim());
  }
  BaseFloat loglike;

  // loglike =  means * inv(vars) * data.
  loglike = VecVec(means_invcovars_.Row(comp_id), data);
  // loglike += -0.5 * tr(data*data'*inv(covar))
  loglike -= 0.5 * VecSpVec(data, inv_covars_[comp_id], data);
  return loglike + gconsts_(comp_id);
}



// Gets likelihood of data given this.
BaseFloat FullGmm::LogLikelihood(const VectorBase<BaseFloat> &data) const {
  Vector<BaseFloat> loglikes;
  LogLikelihoods(data, &loglikes);
  BaseFloat log_sum = loglikes.LogSumExp();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  return log_sum;
}

void FullGmm::LogLikelihoods(const VectorBase<BaseFloat> &data,
                             Vector<BaseFloat> *loglikes) const {
  loglikes->Resize(gconsts_.Dim(), kUndefined);
  loglikes->CopyFromVec(gconsts_);
  int32 dim = Dim();
  KALDI_ASSERT(dim == data.Dim());
  SpMatrix<BaseFloat> data_sq(dim);  // Initialize and make zero
  data_sq.AddVec2(1.0, data);
  // The following enables an optimization below: TraceSpSpLower, which is
  // just like a dot product internally.
  data_sq.ScaleDiag(0.5);

  // loglikes += mean' * inv(covar) * data.
  loglikes->AddMatVec(1.0, means_invcovars_, kNoTrans, data, 1.0);
  // loglikes -= 0.5 * data'*inv(covar)*data = 0.5 * tr(data*data'*inv(covar))
  int32 num_comp = NumGauss();
  for (int32 mix = 0; mix < num_comp; mix++) {
    // was: (*loglikes)(mix) -= 0.5 * TraceSpSp(data_sq, inv_covars_[mix]);
    (*loglikes)(mix) -= TraceSpSpLower(data_sq, inv_covars_[mix]);
  }
}

void FullGmm::LogLikelihoodsPreselect(const VectorBase<BaseFloat> &data,
                                      const vector<int32> &indices,
                                      Vector<BaseFloat> *loglikes) const {
  int32 dim = Dim();
  KALDI_ASSERT(dim == data.Dim());
  int32 num_indices = static_cast<int32>(indices.size());
  loglikes->Resize(num_indices, kUndefined);

  SpMatrix<BaseFloat> data_sq(dim);  // Initialize and make zero
  data_sq.AddVec2(1.0, data);
  // The following enables an optimization below: TraceSpSpLower, which is
  // just like a dot product internally.
  data_sq.ScaleDiag(0.5);

  for (int32 i = 0; i < num_indices; i++) {
    int32 idx = indices[i];
    (*loglikes)(i) = gconsts_(idx)
        + VecVec(means_invcovars_.Row(idx), data)
        - TraceSpSpLower(data_sq, inv_covars_[idx]);
  }
}


/// Get gaussian selection information for one frame.
BaseFloat FullGmm::GaussianSelection(const VectorBase<BaseFloat> &data,
                                     int32 num_gselect,
                                     std::vector<int32> *output) const {
  int32 num_gauss = NumGauss();
  Vector<BaseFloat> loglikes(num_gauss, kUndefined);
  output->clear();
  this->LogLikelihoods(data, &loglikes);

  BaseFloat thresh;
  if (num_gselect < num_gauss) {
    Vector<BaseFloat> loglikes_copy(loglikes);
    BaseFloat *ptr = loglikes_copy.Data();
    std::nth_element(ptr, ptr+num_gauss-num_gselect, ptr+num_gauss);
    thresh = ptr[num_gauss-num_gselect];
  } else {
    thresh = -std::numeric_limits<BaseFloat>::infinity();
  }
  BaseFloat tot_loglike = -std::numeric_limits<BaseFloat>::infinity();
  std::vector<std::pair<BaseFloat, int32> > pairs;
  for (int32 p = 0; p < num_gauss; p++) {
    if (loglikes(p) >= thresh) {
      pairs.push_back(std::make_pair(loglikes(p), p));
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            std::greater<std::pair<BaseFloat, int32> >());
  for (int32 j = 0;
       j < num_gselect && j < static_cast<int32>(pairs.size());
       j++) {
    output->push_back(pairs[j].second);
    tot_loglike = LogAdd(tot_loglike, pairs[j].first);
  }
  KALDI_ASSERT(!output->empty());
  return tot_loglike;
}


BaseFloat FullGmm::GaussianSelectionPreselect(
    const VectorBase<BaseFloat> &data,
    const std::vector<int32> &preselect,
    int32 num_gselect,
    std::vector<int32> *output) const {
  static bool warned_size = false;
  int32 preselect_sz = preselect.size();
  int32 this_num_gselect = std::min(num_gselect, preselect_sz);
  if (preselect_sz <= num_gselect && !warned_size) {
    warned_size = true;
    KALDI_WARN << "Preselect size is less or equal to than final size, "
               << "doing nothing: " << preselect_sz << " < " <<  num_gselect
               << " [won't warn again]";
  }
  Vector<BaseFloat> loglikes(preselect_sz);
  LogLikelihoodsPreselect(data, preselect, &loglikes);

  Vector<BaseFloat> loglikes_copy(loglikes);
  BaseFloat *ptr = loglikes_copy.Data();
  std::nth_element(ptr, ptr+preselect_sz-this_num_gselect,
                   ptr+preselect_sz);
  BaseFloat thresh = ptr[preselect_sz-this_num_gselect];

  BaseFloat tot_loglike = -std::numeric_limits<BaseFloat>::infinity();
  // we want the output sorted from best likelihood to worse
  // (so we can prune further without the model)...
  std::vector<std::pair<BaseFloat, int32> > pairs;
  for (int32 p = 0; p < preselect_sz; p++)
    if (loglikes(p) >= thresh)
      pairs.push_back(std::make_pair(loglikes(p), preselect[p]));
  std::sort(pairs.begin(), pairs.end(),
            std::greater<std::pair<BaseFloat, int32> >());
  output->clear();
  for (int32 j = 0;
       j < this_num_gselect && j < static_cast<int32>(pairs.size());
       j++) {
    output->push_back(pairs[j].second);
    tot_loglike = LogAdd(tot_loglike, pairs[j].first);
  }
  KALDI_ASSERT(!output->empty());
  return tot_loglike;
}


// Gets likelihood of data given this. Also provides per-Gaussian posteriors.
BaseFloat FullGmm::ComponentPosteriors(const VectorBase<BaseFloat> &data,
                                       VectorBase<BaseFloat> *posterior) const {
  if (posterior == NULL) KALDI_ERR << "NULL pointer passed as return argument.";
  Vector<BaseFloat> loglikes;
  LogLikelihoods(data, &loglikes);
  BaseFloat log_sum = loglikes.ApplySoftMax();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  posterior->CopyFromVec(loglikes);
  return log_sum;
}

void FullGmm::RemoveComponent(int32 gauss, bool renorm_weights) {
  KALDI_ASSERT(gauss < NumGauss());

  weights_.RemoveElement(gauss);
  gconsts_.RemoveElement(gauss);
  means_invcovars_.RemoveRow(gauss);
  inv_covars_.erase(inv_covars_.begin() + gauss);
  if (renorm_weights) {
    BaseFloat sum_weights = weights_.Sum();
    weights_.Scale(1.0/sum_weights);
    valid_gconsts_ = false;
  }
}

void FullGmm::RemoveComponents(const vector<int32> &gauss_in, bool renorm_weights) {
  vector<int32> gauss(gauss_in);
  std::sort(gauss.begin(), gauss.end());
  KALDI_ASSERT(IsSortedAndUniq(gauss));
  // If efficiency is later an issue, will code this specially (unlikely,
  // except for quite large GMMs).
  for (size_t i = 0; i < gauss.size(); i++) {
    RemoveComponent(gauss[i], renorm_weights);
    for (size_t j = i + 1; j < gauss.size(); j++)
      gauss[j]--;
  }
}

void FullGmm::Write(std::ostream &out_stream, bool binary) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before writing the model.";
  WriteToken(out_stream, binary, "<FullGMM>");
  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<GCONSTS>");
  gconsts_.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<WEIGHTS>");
  weights_.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MEANS_INVCOVARS>");
  means_invcovars_.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<INV_COVARS>");
  for (int32 i = 0; i < NumGauss(); i++) {
    inv_covars_[i].Write(out_stream, binary);
  }
  WriteToken(out_stream, binary, "</FullGMM>");
  if (!binary) out_stream << "\n";
}

std::ostream & operator <<(std::ostream & out_stream,
                           const kaldi::FullGmm &gmm) {
  gmm.Write(out_stream, false);
  return out_stream;
}

/// this = rho x source + (1-rho) x this
void FullGmm::Interpolate(BaseFloat rho, const FullGmm &source,
                          GmmFlagsType flags) {
  KALDI_ASSERT(NumGauss() == source.NumGauss());
  KALDI_ASSERT(Dim() == source.Dim());
  FullGmmNormal us(*this);
  FullGmmNormal them(source);

  if (flags & kGmmWeights) {
    us.weights_.Scale(1.0 - rho);
    us.weights_.AddVec(rho, them.weights_);
    us.weights_.Scale(1.0 / us.weights_.Sum());
  }

  if (flags & kGmmMeans) {
    us.means_.Scale(1.0 - rho);
    us.means_.AddMat(rho, them.means_);
  }

  if (flags & kGmmVariances) {
    for (int32 i = 0; i < NumGauss(); i++) {
      us.vars_[i].Scale(1.0 - rho);
      us.vars_[i].AddSp(rho, them.vars_[i]);
    }
  }

  us.CopyToFullGmm(this);
  ComputeGconsts();
}

void FullGmm::Read(std::istream &in_stream, bool binary) {
//  ExpectToken(in_stream, binary, "<FullGMMBegin>");
  std::string token;
  ReadToken(in_stream, binary, &token);
  // <FullGMMBegin> is for compatibility. Will be deleted later
  if (token != "<FullGMMBegin>" && token != "<FullGMM>")
    KALDI_ERR << "Expected <FullGMM>, got " << token;
//  ExpectToken(in_stream, binary, "<GCONSTS>");
  ReadToken(in_stream, binary, &token);
  if (token == "<GCONSTS>") {  // The gconsts are optional.
    gconsts_.Read(in_stream, binary);
    ExpectToken(in_stream, binary, "<WEIGHTS>");
  } else {
    if (token != "<WEIGHTS>")
      KALDI_ERR << "FullGmm::Read, expected <WEIGHTS> or <GCONSTS>, got "
                << token;
  }
  weights_.Read(in_stream, binary);
  ExpectToken(in_stream, binary, "<MEANS_INVCOVARS>");
  means_invcovars_.Read(in_stream, binary);
  ExpectToken(in_stream, binary, "<INV_COVARS>");
  int32 ncomp = weights_.Dim(), dim = means_invcovars_.NumCols();
  ResizeInvCovars(ncomp, dim);
  for (int32 i = 0; i < ncomp; i++) {
    inv_covars_[i].Read(in_stream, binary);
  }
//  ExpectToken(in_stream, binary, "<FullGMMEnd>");
  ReadToken(in_stream, binary, &token);
  // <FullGMMEnd> is for compatibility. Will be deleted later
  if (token != "<FullGMMEnd>" && token != "</FullGMM>")
    KALDI_ERR << "Expected </FullGMM>, got " << token;

  ComputeGconsts();  // safer option than trusting the read gconsts
}

std::istream & operator >>(std::istream & in_stream, kaldi::FullGmm &gmm) {
  gmm.Read(in_stream, false);  // false == non-binary.
  return in_stream;
}

}  // End namespace kaldi

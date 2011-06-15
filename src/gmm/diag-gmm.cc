// gmm/diag-gmm.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Georg Stemmer;  Jan Silovsky

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

#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"

namespace kaldi {

void DiagGmm::Resize(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);
  if (gconsts_.Dim() != nmix) gconsts_.Resize(nmix);
  if (weights_.Dim() != nmix) weights_.Resize(nmix);
  if (inv_vars_.NumRows() != nmix ||
      inv_vars_.NumCols() != dim) {
    inv_vars_.Resize(nmix, dim);
    inv_vars_.Set(1.0);
    // must be initialized to unit for case of calling SetMeans while having
    // covars/invcovars that are not set yet (i.e. zero)
  }
  if (means_invvars_.NumRows() != nmix ||
      means_invvars_.NumCols() != dim)
    means_invvars_.Resize(nmix, dim);
}

void DiagGmm::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  Resize(diaggmm.weights_.Dim(), diaggmm.means_invvars_.NumCols());
  gconsts_.CopyFromVec(diaggmm.gconsts_);
  weights_.CopyFromVec(diaggmm.weights_);
  inv_vars_.CopyFromMat(diaggmm.inv_vars_);
  means_invvars_.CopyFromMat(diaggmm.means_invvars_);
  valid_gconsts_ = diaggmm.valid_gconsts_;
}

void DiagGmm::CopyFromFullGmm(const FullGmm &fullgmm) {
  int32 num_comp = fullgmm.NumGauss(), dim = fullgmm.Dim();
  Resize(num_comp, dim);
  gconsts_.CopyFromVec(fullgmm.gconsts());
  weights_.CopyFromVec(fullgmm.weights());
  Matrix<BaseFloat> means(num_comp, dim);
  fullgmm.GetMeans(&means);
  int32 ncomp = NumGauss();
  for (int32 mix = 0; mix < ncomp; ++mix) {
    SpMatrix<double> covar(dim);
    covar.CopyFromSp(fullgmm.inv_covars()[mix]);
    covar.Invert();
    Vector<double> diag(dim);
    diag.CopyDiagFromPacked(covar);
    diag.InvertElements();
    inv_vars_.Row(mix).CopyFromVec(diag);
  }
  means_invvars_.CopyFromMat(means);
  means_invvars_.MulElements(inv_vars_);
  ComputeGconsts();
}

int32 DiagGmm::ComputeGconsts() {
  int32 num_mix = NumGauss();
  int32 dim = Dim();
  BaseFloat offset = -0.5 * M_LOG_2PI * dim;  // constant term in gconst.
  int32 num_bad = 0;

  // Resize if Gaussians have been removed during Update()
  if (num_mix != static_cast<int32>(gconsts_.Dim()))
    gconsts_.Resize(num_mix);

  for (int32 mix = 0; mix < num_mix; mix++) {
    KALDI_ASSERT(weights_(mix) >= 0);  // Cannot have negative weights.
    BaseFloat gc = log(weights_(mix)) + offset;  // May be -inf if weights == 0
    for (int32 d = 0; d < dim; d++) {
      gc += 0.5 * log(inv_vars_(mix, d)) - 0.5 * means_invvars_(mix, d)
        * means_invvars_(mix, d) / inv_vars_(mix, d);
    }
    // Change sign for logdet because var is inverted. Also, note that
    // mean_invvars(mix, d)*mean_invvars(mix, d)/inv_vars(mix, d) is the
    // mean-squared times inverse variance, since mean_invvars(mix, d) contains
    // the mean times inverse variance.
    // So gc is the likelihood at zero feature value.

    if (KALDI_ISNAN(gc)) {  // negative infinity is OK but NaN is not acceptable
      KALDI_ERR << "At component "  << mix
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

void DiagGmm::Split(int32 target_components, float perturb_factor) {
  if (target_components < NumGauss() || NumGauss() == 0) {
    KALDI_ERR << "Cannot split from "  << NumGauss() << " to "
              << target_components  << " components";
  }
  if (target_components == NumGauss()) {
    KALDI_WARN << "Already have the target # of Gaussians. Doing nothing.";
    return;
  }

  int32 current_components = NumGauss(), dim = Dim();
  DiagGmm *tmp = new DiagGmm;
  tmp->CopyFromDiagGmm(*this);  // so we have copies of matrices
  // First do the resize:
  weights_.Resize(target_components);
  weights_.Range(0, current_components).CopyFromVec(tmp->weights_);
  means_invvars_.Resize(target_components, dim);
  means_invvars_.Range(0, current_components, 0, dim).CopyFromMat(
      tmp->means_invvars_);
  inv_vars_.Resize(target_components, dim);
  inv_vars_.Range(0, current_components, 0, dim).CopyFromMat(tmp->inv_vars_);
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
    weights_(max_idx) /= 2;
    weights_(current_components) = weights_(max_idx);
    Vector<BaseFloat> rand_vec(dim);
    for (int32 i = 0; i < dim; i++) {
      rand_vec(i) = RandGauss() * std::sqrt(inv_vars_(max_idx, i));
      // note, this looks wrong but is really right because it's the
      // means_invvars we're multiplying and they have the dimension
      // of an inverse standard variance. [dan]
    }
    inv_vars_.Row(current_components).CopyFromVec(inv_vars_.Row(max_idx));
    means_invvars_.Row(current_components).CopyFromVec(means_invvars_.Row(
        max_idx));
    means_invvars_.Row(current_components).AddVec(perturb_factor, rand_vec);
    means_invvars_.Row(max_idx).AddVec(-perturb_factor, rand_vec);
    current_components++;
  }
  ComputeGconsts();
}

void DiagGmm::Merge(int32 target_components) {
  if (target_components <= 0 || NumGauss() < target_components) {
    KALDI_ERR << "Invalid argument for target number of Gaussians (="
        << target_components << ")";
  }
  if (NumGauss() == target_components) {
    KALDI_WARN << "No components merged, as target (" << target_components
               << ") = total.";
    return;
  }

  int32 num_comp = NumGauss(), dim = Dim();

  if (target_components == 1) {  // global mean and variance
    Vector<BaseFloat> weights(weights_);
    // Undo variance inversion and multiplication of mean by inv var.
    Matrix<BaseFloat> vars(inv_vars_);
    Matrix<BaseFloat> means(means_invvars_);
    vars.InvertElements();
    means.MulElements(vars);
    // add means square to variances; get second-order stats
    for (int32 i = 0; i < num_comp; ++i) {
      vars.Row(i).AddVec2(1.0, means.Row(i));
    }

    // Slightly more efficient than calling this->Resize(1, dim)
    gconsts_.Resize(1);
    weights_.Resize(1);
    means_invvars_.Resize(1, dim);
    inv_vars_.Resize(1, dim);

    for (int32 i = 0; i < num_comp; ++i) {
      weights_(0) += weights(i);
      means_invvars_.Row(0).AddVec(weights(i), means.Row(i));
      inv_vars_.Row(0).AddVec(weights(i), vars.Row(i));
    }
    if (!ApproxEqual(weights_(0), 1.0, 1e-6)) {
      KALDI_WARN << "Weights sum to " << weights_(0) << ": rescaling.";
      means_invvars_.Scale(weights_(0));
      inv_vars_.Scale(weights_(0));
      weights_(0) = 1.0;
    }
    inv_vars_.Row(0).AddVec2(-1.0, means_invvars_.Row(0));
    inv_vars_.InvertElements();
    means_invvars_.MulElements(inv_vars_);
    ComputeGconsts();
    return;
  }

  // If more than 1 merged component is required, use the hierarchical
  // clustering of components that lead to the smallest decrease in likelihood.
  std::vector<bool> discarded_component(num_comp);
  Vector<BaseFloat> logdet(num_comp);   // logdet for each component
  logdet.SetZero();
  for (int32 i = 0; i < num_comp; ++i) {
    discarded_component[i] = false;
    for (int32 d = 0; d < dim; ++d) {
      logdet(i) += 0.5 * log(inv_vars_(i, d));  // +0.5 because var is inverted
    }
  }

  // Undo variance inversion and multiplication of mean by this
  // Makes copy of means and vars for all components - memory inefficient?
  Matrix<BaseFloat> vars(inv_vars_);
  Matrix<BaseFloat> means(means_invvars_);
  vars.InvertElements();
  means.MulElements(vars);

  // add means square to variances; get second-order stats
  // (normalized by zero-order stats)
  for (int32 i = 0; i < num_comp; ++i) {
    vars.Row(i).AddVec2(1.0, means.Row(i));
  }

  // compute change of likelihood for all combinations of components
  SpMatrix<BaseFloat> delta_like(num_comp);
  for (int32 i = 0; i < num_comp; ++i) {
    for (int32 j = 0; j < i; ++j) {
      BaseFloat w1 = weights_(i), w2 = weights_(j), w_sum = w1 + w2;
      BaseFloat merged_logdet = merged_components_logdet(w1, w2,
        means.Row(i), means.Row(j), vars.Row(i), vars.Row(j));
      delta_like(i, j) = w_sum * merged_logdet
        - w1 * logdet(i) - w2 * logdet(j);
    }
  }

  // Merge components with smallest impact on the loglike
  for (int32 removed = 0; removed < num_comp - target_components; ++removed) {
    // Search for the least significant change in likelihood
    // (maximum of negative delta_likes)
    BaseFloat max_delta_like = -std::numeric_limits<BaseFloat>::max();
    int32 max_i = 0, max_j = 0;
    for (int32 i = 0; i < NumGauss(); ++i) {
      if (discarded_component[i]) continue;
      for (int32 j = 0; j < i; ++j) {
        if (discarded_component[j]) continue;
        if (delta_like(i, j) > max_delta_like) {
          max_delta_like = delta_like(i, j);
          max_i = i;
          max_j = j;
        }
      }
    }

    // make sure that different components will be merged
    assert(max_i != max_j);

    // Merge components
    BaseFloat w1 = weights_(max_i), w2 = weights_(max_j);
    BaseFloat w_sum = w1 + w2;
    // merge means
    means.Row(max_i).AddVec(w2/w1, means.Row(max_j));
    means.Row(max_i).Scale(w1/w_sum);
    // merge vars
    vars.Row(max_i).AddVec(w2/w1, vars.Row(max_j));
    vars.Row(max_i).Scale(w1/w_sum);
    // merge weights
    weights_(max_i) = w_sum;

    // Update gmm for merged component
    // copy second-order stats (normalized by zero-order stats)
    inv_vars_.Row(max_i).CopyFromVec(vars.Row(max_i));
    // centralize
    inv_vars_.Row(max_i).AddVec2(-1.0, means.Row(max_i));
    // invert
    inv_vars_.Row(max_i).InvertElements();
    // copy first-order stats (normalized by zero-order stats)
    means_invvars_.Row(max_i).CopyFromVec(means.Row(max_i));
    // multiply by inv_vars
    means_invvars_.Row(max_i).MulElements(inv_vars_.Row(max_i));

    // Update logdet for merged component
    logdet(max_i) = 0.0;
    for (int32 d = 0; d < dim; ++d) {
      logdet(max_i) += 0.5 * log(inv_vars_(max_i, d));
      // +0.5 because var is inverted
    }

    // Label the removed component as discarded
    discarded_component[max_j] = true;

    // Update delta_like for merged component
    for (int32 j = 0; j < num_comp; ++j) {
      if ((j == max_i) || (discarded_component[j])) continue;
      BaseFloat w1 = weights_(max_i),
                w2 = weights_(j),
                w_sum = w1 + w2;
      BaseFloat merged_logdet = merged_components_logdet(w1, w2,
          means.Row(max_i), means.Row(j), vars.Row(max_i), vars.Row(j));
      delta_like(max_i, j) = w_sum * merged_logdet - w1 * logdet(max_i)
          - w2 * logdet(j);
      // doesn't respect lower triangular indeces,
      // relies on implicitly performed swap of coordinates if necessary
    }
  }

  // Remove the consumed components
  int32 m = 0;
  for (int32 i = 0; i < num_comp; ++i) {
    if (discarded_component[i]) {
      weights_.RemoveElement(m);
      means_invvars_.RemoveRow(m);
      inv_vars_.RemoveRow(m);
    } else {
      ++m;
    }
  }

  ComputeGconsts();
}

BaseFloat DiagGmm::merged_components_logdet(BaseFloat w1, BaseFloat w2,
                                            const VectorBase<BaseFloat> &f1,
                                            const VectorBase<BaseFloat> &f2,
                                            const VectorBase<BaseFloat> &s1,
                                            const VectorBase<BaseFloat> &s2)
                                            const {
  int32 dim = f1.Dim();
  Vector<BaseFloat> tmp_mean(dim);
  Vector<BaseFloat> tmp_var(dim);
  BaseFloat merged_logdet = 0.0;

  BaseFloat w_sum = w1 + w2;
  tmp_mean.CopyFromVec(f1);
  tmp_mean.AddVec(w2/w1, f2);
  tmp_mean.Scale(w1/w_sum);
  tmp_var.CopyFromVec(s1);
  tmp_var.AddVec(w2/w1, s2);
  tmp_var.Scale(w1/w_sum);
  tmp_var.AddVec2(-1.0, tmp_mean);
  for (int32 d = 0; d < dim; ++d) {
    merged_logdet -= 0.5 * log(tmp_var(d));
    // -0.5 because var is not inverted
  }
  return merged_logdet;
}

BaseFloat DiagGmm::ComponentLogLikelihood(const VectorBase<BaseFloat> &data,
                                          int32 comp_id) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  if (static_cast<int32>(data.Dim()) != Dim()) {
    KALDI_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
        << "mismatch" << (data.Dim()) << "vs. "<< (Dim());
  }
  BaseFloat loglike;
  Vector<BaseFloat> data_sq(data);
  data_sq.ApplyPow(2.0);

  // loglike =  means * inv(vars) * data.
  loglike = VecVec(means_invvars_.Row(comp_id), data);
  // loglike += -0.5 * inv(vars) * data_sq.
  loglike -= 0.5 * VecVec(inv_vars_.Row(comp_id), data_sq);
  return loglike + gconsts_(comp_id);
}

// Gets likelihood of data given this.
BaseFloat DiagGmm::LogLikelihood(const VectorBase<BaseFloat> &data) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  Vector<BaseFloat> loglikes;
  LogLikelihoods(data, &loglikes);
  BaseFloat log_sum = loglikes.LogSumExp();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  return log_sum;
}

void DiagGmm::LogLikelihoods(const VectorBase<BaseFloat> &data,
                             Vector<BaseFloat> *loglikes) const {
  loglikes->Resize(gconsts_.Dim(), kUndefined);
  loglikes->CopyFromVec(gconsts_);
  if (static_cast<int32>(data.Dim()) != Dim()) {
    KALDI_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
        << "mismatch" << (data.Dim()) << "vs. "<< (Dim());
  }
  Vector<BaseFloat> data_sq(data);
  data_sq.ApplyPow(2.0);

  // loglikes +=  means * inv(vars) * data.
  loglikes->AddMatVec(1.0, means_invvars_, kNoTrans, data, 1.0);
  // loglikes += -0.5 * inv(vars) * data_sq.
  loglikes->AddMatVec(-0.5, inv_vars_, kNoTrans, data_sq, 1.0);
}

// Gets likelihood of data given this. Also provides per-Gaussian posteriors.
BaseFloat DiagGmm::ComponentPosteriors(const VectorBase<BaseFloat> &data,
                                       Vector<BaseFloat> *posterior) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
  if (posterior == NULL) KALDI_ERR << "NULL pointer passed as return argument.";
  Vector<BaseFloat> loglikes;
  LogLikelihoods(data, &loglikes);
  BaseFloat log_sum = loglikes.ApplySoftMax();
  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  if (posterior->Dim() != loglikes.Dim())
    posterior->Resize(loglikes.Dim());
  posterior->CopyFromVec(loglikes);
  return log_sum;
}

void DiagGmm::RemoveComponent(int32 gauss, bool renorm_weights) {
  KALDI_ASSERT(gauss < NumGauss());
  if (NumGauss() == 1)
    KALDI_ERR << "Attempting to remove the only remaining component.";
  weights_.RemoveElement(gauss);
  gconsts_.RemoveElement(gauss);
  means_invvars_.RemoveRow(gauss);
  inv_vars_.RemoveRow(gauss);
  BaseFloat sum_weights = weights_.Sum();
  if (renorm_weights) {
    weights_.Scale(1.0/sum_weights);
    valid_gconsts_ = false;
  }
}

void DiagGmm::RemoveComponents(const std::vector<int32> &gauss_in,
                               bool renorm_weights) {
  std::vector<int32> gauss(gauss_in);
  std::sort(gauss.begin(), gauss.end());
  KALDI_ASSERT(IsSortedAndUniq(gauss));
  // If efficiency is later an issue, will code this specially (unlikely).
  for (size_t i = 0; i < gauss.size(); i++) {
    RemoveComponent(gauss[i], renorm_weights);
    for (size_t j = i + 1; j < gauss.size(); j++)
      gauss[j]--;
  }
}


void DiagGmm::Write(std::ostream &out_stream, bool binary) const {
  if (!valid_gconsts_)
    KALDI_ERR << "Must call ComputeGconsts() before writing the model.";
  WriteMarker(out_stream, binary, "<DiagGMM>");
  if (!binary) out_stream << "\n";
  WriteMarker(out_stream, binary, "<GCONSTS>");
  gconsts_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<WEIGHTS>");
  weights_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<MEANS_INVVARS>");
  means_invvars_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<INV_VARS>");
  inv_vars_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</DiagGMM>");
  if (!binary) out_stream << "\n";
}

std::ostream & operator <<(std::ostream & out_stream,
                           const kaldi::DiagGmm &gmm) {
  gmm.Write(out_stream, false);
  return out_stream;
}

void DiagGmm::Read(std::istream &in_stream, bool binary) {
//  ExpectMarker(in_stream, binary, "<DiagGMMBegin>");
  std::string marker;
  ReadMarker(in_stream, binary, &marker);
  // <DiagGMMBegin> is for compatibility. Will be deleted later
  if (marker != "<DiagGMMBegin>" && marker != "<DiagGMM>")
    KALDI_ERR << "Expected <DiagGMM>, got " << marker;
  ReadMarker(in_stream, binary, &marker);
  if (marker == "<GCONSTS>") {  // The gconsts are optional.
    gconsts_.Read(in_stream, binary);
    ExpectMarker(in_stream, binary, "<WEIGHTS>");
  } else {
    if (marker != "<WEIGHTS>")
      KALDI_ERR << "DiagGmm::Read, expected <WEIGHTS> or <GCONSTS>, got "
                << marker;
  }
  weights_.Read(in_stream, binary);
  ExpectMarker(in_stream, binary, "<MEANS_INVVARS>");
  means_invvars_.Read(in_stream, binary);
  ExpectMarker(in_stream, binary, "<INV_VARS>");
  inv_vars_.Read(in_stream, binary);
//  ExpectMarker(in_stream, binary, "<DiagGMMEnd>");
  ReadMarker(in_stream, binary, &marker);
  // <DiagGMMEnd> is for compatibility. Will be deleted later
  if (marker != "<DiagGMMEnd>" && marker != "</DiagGMM>")
    KALDI_ERR << "Expected </DiagGMM>, got " << marker;

  ComputeGconsts();  // safer option than trusting the read gconsts
}

std::istream & operator >>(std::istream & rIn, kaldi::DiagGmm &gmm) {
  gmm.Read(rIn, false);  // false == non-binary.
  return rIn;
}

void DiagGmm::Generate(VectorBase<BaseFloat> *output) {
  KALDI_ASSERT(static_cast<int32>(output->Dim()) == Dim());
  BaseFloat tot = weights_.Sum();
  KALDI_ASSERT(tot > 0.0);
  double r = tot * RandUniform() * 0.99999;
  int32 i = 0;
  double sum = 0.0;
  while (sum + weights_(i) < r) {
    sum += weights_(i);
    i++;
    KALDI_ASSERT(i < static_cast<int32>(weights_.Dim()));
  }
  // now i is the index of the Gaussian we chose.
  SubVector<BaseFloat> inv_var(inv_vars_, i),
      mean_invvar(means_invvars_, i);
  for (int32 d = 0; d < inv_var.Dim(); d++) {
    BaseFloat stddev = 1.0 / sqrt(inv_var(d)),
        mean = mean_invvar(d) / inv_var(d);
    (*output)(d) = mean + RandGauss() * stddev;
  }
}

}  // End namespace kaldi

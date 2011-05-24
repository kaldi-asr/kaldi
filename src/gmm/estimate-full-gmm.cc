// gmm/estimate-full-gmm.cc

// Copyright 2009-2011  Jan Silovsky, Arnab Ghoshal (Saarland University), Microsoft Corporation, Georg Stemmer

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

#include <string>

#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/estimate-full-gmm.h"

namespace kaldi {


MlEstimateFullGmm::MlEstimateFullGmm(const MlEstimateFullGmm &other)
  : dim_(other.dim_), num_comp_(other.num_comp_),
    flags_(other.flags_), occupancy_(other.occupancy_),
    mean_accumulator_(other.mean_accumulator_),
    covariance_accumulator_(other.covariance_accumulator_) {}


GmmFlagsType MlEstimateFullGmm::AugmentFlags(GmmFlagsType f) {
  assert((f & ~kGmmAll) == 0);  // make sure only valid flags are present.
  if (f & kGmmVariances) f |= kGmmMeans;
  if (f & kGmmMeans) f |= kGmmWeights;
  assert(f & kGmmWeights);  // make sure zero-stats will be accumulated
  return f;
}


void MlEstimateFullGmm::ResizeAccumulators(int32 num_comp, int32 dim,
                                           GmmFlagsType flags) {
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentFlags(flags);
  occupancy_.Resize(num_comp);
  if (flags_ & kGmmMeans)
    mean_accumulator_.Resize(num_comp, dim);
  else
    mean_accumulator_.Resize(0, 0);
  if (flags_ & kGmmVariances)
    ResizeVarAccumulator(num_comp, dim);
  else
    covariance_accumulator_.clear();
}

void MlEstimateFullGmm::ZeroAccumulators(GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  if (flags & kGmmWeights) occupancy_.SetZero();
  if (flags & kGmmMeans) mean_accumulator_.SetZero();
  if (flags & kGmmVariances) {
    for (int32 i = 0, end = covariance_accumulator_.size(); i < end; ++i)
      covariance_accumulator_[i].SetZero();
  }
}

void MlEstimateFullGmm::ScaleAccumulators(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) occupancy_.Scale(d);
  if (flags & kGmmMeans) mean_accumulator_.Scale(d);
  if (flags & kGmmVariances) {
    for (int32 i = 0, end = covariance_accumulator_.size(); i < end; ++i)
      covariance_accumulator_[i].Scale(d);
  }
}

void MlEstimateFullGmm::ResizeVarAccumulator(int32 num_comp, int32 dim) {
  KALDI_ASSERT(num_comp > 0 && dim > 0);
  if (covariance_accumulator_.size() != static_cast<size_t>(num_comp))
    covariance_accumulator_.resize(num_comp);
  for (int32 i = 0; i < num_comp; ++i) {
    if (covariance_accumulator_[i].NumRows() != dim)
      covariance_accumulator_[i].Resize(dim);
  }
}

void MlEstimateFullGmm::AccumulateForComponent(
    const VectorBase<BaseFloat>& data, int32 comp_index, BaseFloat weight) {
  assert(data.Dim() == Dim());
  double wt = static_cast<double>(weight);

  // accumulate
  occupancy_(comp_index) += wt;
  if (flags_ & kGmmMeans) {
    Vector<double> data_d(data);  // Copy with type-conversion
    mean_accumulator_.Row(comp_index).AddVec(wt, data_d);
    if (flags_ & kGmmVariances) {
      covariance_accumulator_[comp_index].AddVec2(wt, data_d);
    }
  }
}

void MlEstimateFullGmm::AccumulateFromPosteriors(
    const VectorBase<BaseFloat>& data,
    const VectorBase<BaseFloat>& gauss_posteriors) {
  assert(gauss_posteriors.Dim() == NumGauss());
  assert(data.Dim() == Dim());
  Vector<double> data_d(data.Dim());
  data_d.CopyFromVec(data);
  Vector<double> post_d(gauss_posteriors.Dim());
  post_d.CopyFromVec(gauss_posteriors);

  occupancy_.AddVec(1.0, post_d);
  if (flags_ & (kGmmMeans|kGmmVariances)) {  // mean stats.
    if (static_cast<int32>(post_d.Norm(0.0)*2.0) > post_d.Dim()) {
      // If we're not very sparse... note: zero-norm is number of
      // nonzero elements.
      mean_accumulator_.AddVecVec(1.0, post_d, data_d);
    } else {
      for (int32 i = 0; i < post_d.Dim(); i++)
        if (post_d(i) != 0.0)
          mean_accumulator_.Row(i).AddVec(post_d(i), data_d);
    }
    if (flags_ & kGmmVariances) {
      SpMatrix<double> data_sq_d(data_d.Dim());
      data_sq_d.AddVec2(1.0, data_d);
      for (int32 mix = 0; mix < NumGauss(); ++mix)
        if (post_d(mix) !=  0.0)
          covariance_accumulator_[mix].AddSp(post_d(mix), data_sq_d);
    }
  }
}

BaseFloat MlEstimateFullGmm::AccumulateFromFull(const FullGmm &gmm,
    const VectorBase<BaseFloat>& data, BaseFloat frame_posterior) {
  assert(gmm.NumGauss() == NumGauss());
  assert(gmm.Dim() == Dim());

  Vector<BaseFloat> component_posterior(NumGauss());

  BaseFloat log_like = gmm.ComponentPosteriors(data, &component_posterior);
  component_posterior.Scale(frame_posterior);

  AccumulateFromPosteriors(data, component_posterior);
  return log_like;
}

BaseFloat MlEstimateFullGmm::AccumulateFromDiag(const DiagGmm &gmm,
    const VectorBase<BaseFloat>& data, BaseFloat frame_posterior) {
  assert(gmm.NumGauss() == NumGauss());
  assert(gmm.Dim() == Dim());

  Vector<BaseFloat> component_posterior(NumGauss());

  BaseFloat log_like = gmm.ComponentPosteriors(data, &component_posterior);
  component_posterior.Scale(frame_posterior);

  AccumulateFromPosteriors(data, component_posterior);
  return log_like;
}

int32 MlEstimateFullGmm::RemoveComponent(int32 comp) {
  KALDI_ASSERT(comp >= 0 && comp < occupancy_.Dim());
  occupancy_.RemoveElement(comp);
  if (flags_ & kGmmMeans) mean_accumulator_.RemoveRow(comp);
  if (flags_ & kGmmVariances)
    covariance_accumulator_.erase(covariance_accumulator_.begin() + comp);
  num_comp_--;
  return occupancy_.Dim();
}

BaseFloat MlEstimateFullGmm::MlObjective(const FullGmm& gmm) const {
  Vector<BaseFloat> occ_bf(occupancy_);
  Matrix<BaseFloat> mean_accs_bf(mean_accumulator_);
  SpMatrix<BaseFloat> covar_accs_bf(dim_);
  BaseFloat obj = VecVec(occ_bf, gmm.gconsts());
  if (flags_ & kGmmMeans)
    obj += TraceMatMat(mean_accs_bf, gmm.means_invcovars(), kTrans);
  if (flags_ & kGmmVariances) {
    for (int32 i = 0; i < num_comp_; ++i) {
      covar_accs_bf.CopyFromSp(covariance_accumulator_[i]);
      obj -= 0.5 * TraceSpSp(covar_accs_bf, gmm.inv_covars()[i]);
    }
  }
  return obj;
}

void MlEstimateFullGmm::Update(const MleFullGmmOptions &config,
                               GmmFlagsType flags,
                               FullGmm *gmm,
                               BaseFloat *obj_change_out,
                               BaseFloat *count_out) const {
  KALDI_ASSERT(gmm != NULL);
  KALDI_ASSERT(gmm->NumGauss() == NumGauss());
  if (flags & ~flags_)
    KALDI_ERR << "Flags in argument do not match the active accumulators";

  BaseFloat obj_old, obj_new, count = occupancy_.Sum();
  if (obj_change_out != NULL)
    obj_old = MlObjective(*gmm);

  if (count == 0.0) {
    KALDI_WARN << "MlEstimateFullGmm::Update, no data observed so not updating.";
    if (obj_change_out) *obj_change_out = 0.0;
    if (count_out) *count_out = count;
    return;
  }
  std::vector<int32> to_remove;
  int32 num_gauss = NumGauss(), dim = Dim();

  Vector<double> weights(num_gauss);
  Matrix<double> means(mean_accumulator_);
  if (!(flags_ & kGmmMeans)) gmm->GetMeans(&means);
  std::vector<SpMatrix<double> > inv_vars(num_gauss);

  for (int32 i = 0; i < num_gauss; i++) {
    double v = occupancy_(i) / count;
    weights(i) = v;

    if (occupancy_(i) > static_cast<double> (config.min_gaussian_occupancy)
        && v > static_cast<double> (config.min_gaussian_weight)) {
      if (flags_ & kGmmMeans)   // mean calculation
        means.Row(i).Scale(1 / occupancy_(i));
      SubVector<double> mean_dbl(means.Row(i));  // either updated, or old, mean.

      if (flags_ & kGmmVariances) {  // covariance calculation
        SpMatrix<double> var(covariance_accumulator_[i]);
        var.Scale(1 / occupancy_(i));
        if (flags_ & kGmmMeans)
          var.AddVec2(-1.0, mean_dbl);
        else {
          var.AddVec2(-1.0 / occupancy_(i), mean_accumulator_.Row(i));
          // now "var" is the variance of the data around its own mean.
          mean_dbl.AddVec(-1.0 / occupancy_(i), mean_accumulator_.Row(i));
          // now, "mean_dbl" is the difference between the observed and
          // model mean.
          var.AddVec2(1.0, mean_dbl);
        }
        // Now flooring etc. of variance's eigenvalues.
        BaseFloat floor = std::max(static_cast<double>(config.variance_floor),
                                   var.MaxAbsEig() / config.max_condition);
        // 2.0 in the next line implies full tolerance to non-+ve-definiteness..
        int32 num_floored = var.ApplyFloor(floor, 2.0);
        if (num_floored)
          KALDI_WARN << "Floored " << num_floored << " covariance eigenvalues for "
              "Gaussian " << i << ", count = " << occupancy_(i);

        var.Invert();
        inv_vars[i].Resize(dim);
        inv_vars[i].CopyFromSp(var);
      }
    } else {  // Below threshold to update -> just set to old mean/var.
      if (flags_ & kGmmVariances) {
        inv_vars[i].Resize(dim);
        inv_vars[i].CopyFromSp( (gmm->inv_covars())[i] );
      }
      if (flags_ & kGmmMeans) {
        SubVector<double> mean(means, i);
        gmm->GetComponentMean(i, &mean);
      }
      if (config.remove_low_count_gaussians) {
        to_remove.push_back(i);
        KALDI_WARN << "Removing Gaussian " << i << ", occupancy is "
                   << occupancy_(i);
      } else {
        KALDI_WARN << "Not updating mean and variance of Gaussian " << i << ", occupancy is "
                   << occupancy_(i);
      }
    }
  }
  if (flags & kGmmWeights) gmm->SetWeights(weights);
  if ((flags & kGmmMeans) && (flags & kGmmVariances))
    gmm->SetInvCovarsAndMeans(inv_vars, means);
  else if (flags & kGmmMeans)
    gmm->SetMeans(means);
  else if (flags & kGmmVariances)
    gmm->SetInvCovars(inv_vars);

  gmm->ComputeGconsts();

  if (obj_change_out != NULL) {
    if (gmm->NumGauss() == NumGauss()) {
      obj_new = MlObjective(*gmm);
      KALDI_VLOG(2) << "ML objective function: old = " << (obj_old/count)
                 << ", new = " << (obj_new/count) << ", change = "
                 << ((obj_new - obj_old)/count) << ", over "
                 << (count) << " frames.";
      *obj_change_out = obj_new - obj_old;
    } else { *obj_change_out = 0.0; }  // hard to compute, not doing it!
  }
  if (count_out != NULL) *count_out = count;

  if (!to_remove.empty()) {
    gmm->RemoveComponents(to_remove);
    gmm->ComputeGconsts();
  }
}


void MlEstimateFullGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectMarker(in_stream, binary, "<GMMACCS>");
  ExpectMarker(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dimension);
  ExpectMarker(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);
  KALDI_ASSERT(dimension > 0 && num_components > 0);
  ExpectMarker(in_stream, binary, "<FLAGS>");
  ReadBasicType(in_stream, binary, &flags);

  if (add) {
    if ((NumGauss() != 0 || Dim() != 0 || Flags() != 0)) {
      if (num_components != NumGauss() || dimension != Dim()
          || flags != Flags()) {
        KALDI_ERR << "MlEstimatediagGmm::Read, dimension or flags mismatch, "
            << (NumGauss()) << ", " << (Dim()) << ", "
            << (Flags()) +" vs. " + (num_components) << ", "
            << (dimension) << ", " << (flags);
      }
    } else {
      ResizeAccumulators(num_components, dimension, flags);
    }
  } else {
    ResizeAccumulators(num_components, dimension, flags);
  }

  // these are needed for demangling the variances.
  Vector<double> tmp_occs;
  Matrix<double> tmp_means;

  ReadMarker(in_stream, binary, &token);
  while (token != "</GMMACCS>") {
    if (token == "<OCCUPANCY>") {
      tmp_occs.Read(in_stream, binary, false);
      if (!add) occupancy_.SetZero();
      occupancy_.AddVec(1.0, tmp_occs);
    } else if (token == "<MEANACCS>") {
      tmp_means.Read(in_stream, binary, false);
      if (!add) mean_accumulator_.SetZero();
      mean_accumulator_.AddMat(1.0, tmp_means);
    } else if (token == "<FULLVARACCS>") {
      for (int32 i = 0; i < num_components; ++i) {
        SpMatrix<double> tmp_acc;
        tmp_acc.Read(in_stream, binary, add);
        if (tmp_occs(i) != 0) tmp_acc.AddVec2(1.0 / tmp_occs(i), tmp_means.Row(
            i));
        if (!add) covariance_accumulator_[i].SetZero();
        covariance_accumulator_[i].AddSp(1.0, tmp_acc);
      }
    } else {
      KALDI_ERR << "Unexpected token '" << token << "' in model file ";
    }
    ReadMarker(in_stream, binary, &token);
  }
}

void MlEstimateFullGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<GMMACCS>");
  WriteMarker(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteMarker(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteMarker(out_stream, binary, "<FLAGS>");
  WriteBasicType(out_stream, binary, flags_);

  Vector<BaseFloat> occupancy_bf(occupancy_);
  WriteMarker(out_stream, binary, "<OCCUPANCY>");
  occupancy_bf.Write(out_stream, binary);
  Matrix<BaseFloat> mean_accumulator_bf(mean_accumulator_);
  WriteMarker(out_stream, binary, "<MEANACCS>");
  mean_accumulator_bf.Write(out_stream, binary);

  if (num_comp_ != 0) assert(((flags_ & kGmmVariances) != 0 )
      == (covariance_accumulator_.size() != 0));  // sanity check.
  if (covariance_accumulator_.size() != 0) {
    WriteMarker(out_stream, binary, "<FULLVARACCS>");
    for (int32 i = 0; i < num_comp_; ++i) {
      SpMatrix<double> tmp_acc(covariance_accumulator_[i]);
      if (occupancy_(i) != 0) tmp_acc.AddVec2(-1.0 / occupancy_(i),
          mean_accumulator_.Row(i));
      SpMatrix<float> tmp_acc_bf(tmp_acc);
      tmp_acc_bf.Write(out_stream, binary);
    }
  }
  WriteMarker(out_stream, binary, "</GMMACCS>");
}

}  // End namespace kaldi
